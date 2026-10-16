"""Runs the Wan2.1-VACE transition pipeline by calling ComfyUI's node classes
directly, in-process -- no HTTP server, no subprocess, no graph JSON.

Bootstrap sequence mirrors what `main.py`'s `start_comfyui()` does before it
starts serving: minimal args, load extra_model_paths.yaml, instantiate
PromptServer (without binding a port -- some node code reports progress via
`server.PromptServer.instance` and expects it to exist), then
`nodes.init_extra_nodes()` to register built-in and custom (GGUF) nodes.
"""

from __future__ import annotations

import asyncio
import functools
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import typer
from tqdm import tqdm

from soto import config

_node_classes: dict = {}


def _bootstrap() -> None:
    if _node_classes:
        return

    sys.path.insert(0, str(config.COMFYUI_DIR))

    # ComfyUI's argparse actually runs at first `import comfy.cli_args`
    # (triggered transitively below), not inside enable_args_parsing() itself
    # -- so sys.argv must stay swapped out until that import has happened,
    # otherwise this CLI's own argv leaks into ComfyUI's parser.
    # --lowvram keeps the UNet's weights off the MPS device except while
    # actually in use.
    # The VAE runs where config.VAE_DEVICE says. It used to be forced onto
    # the CPU (--cpu-vae): encoding N frames at 832x480 on MPS took 2.87GB
    # at N=4, 24.4GB at N=9 and OOM'd at N=17, which looked like the causal
    # VAE's chunking not freeing memory between chunks. That was measured
    # before the torch.inference_mode() fix in run_vace_transition(), and
    # was the same autograd retention: since the fix it's 5.3GB at N=17 and
    # 9.6GB at N=81, and 2-3x faster than the CPU at 832x480 (see config.py).
    # --use-pytorch-cross-attention forces PyTorch's native SDPA kernel.
    # Without it, ComfyUI's default attention pick on MPS (no flash-attn/
    # xformers here) fell through to its own `sub_quadratic_attention`
    # fallback, which upcasts query/key to fp32 and -- even for a genuinely
    # tiny 192x112x9-frame VACE-conditioned sequence -- never took its
    # chunked code path, materializing a full fp32 attention matrix and
    # hitting the same ~42GB ceiling regardless of UNet size (14B or 1.3B).
    if config.VAE_DEVICE not in config.VAE_DEVICES:
        raise RuntimeError(f"SOTO_VAE_DEVICE must be one of {list(config.VAE_DEVICES)}, not {config.VAE_DEVICE!r}")
    if config.VAE_DTYPE not in config.VAE_DTYPES:
        raise RuntimeError(f"SOTO_VAE_DTYPE must be one of {list(config.VAE_DTYPES)}, not {config.VAE_DTYPE!r}")
    vae_args = ["--cpu-vae"] if config.VAE_DEVICE == "cpu" else [f"--{config.VAE_DTYPE}-vae"]
    old_argv, sys.argv = sys.argv, [
        "comfyui", "--lowvram", *vae_args, "--use-pytorch-cross-attention",
    ]
    try:
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy.cli_args  # noqa: F401 (triggers argparse.parse_args() now, while argv is swapped)
        import folder_paths  # noqa: F401 (must be imported before extra_config)
    finally:
        sys.argv = old_argv

    import utils.extra_config

    extra_paths_file = config.COMFYUI_DIR / "extra_model_paths.yaml"
    if extra_paths_file.is_file():
        utils.extra_config.load_extra_path_config(str(extra_paths_file))

    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.PromptServer(loop)  # populates PromptServer.instance; never started/bound

    import nodes

    loop.run_until_complete(
        nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)
    )

    for name in (
        "UNETLoader",
        "UnetLoaderGGUF",
        "CLIPLoaderGGUF",
        "CLIPTextEncode",
        "VAELoader",
        "ModelSamplingSD3",
        "WanVaceToVideo",
        "KSampler",
        "VAEDecode",
    ):
        if name not in nodes.NODE_CLASS_MAPPINGS:
            raise RuntimeError(
                f"ComfyUI node '{name}' not found after init_extra_nodes(); "
                "check that vendor/ComfyUI and ComfyUI-GGUF are set up correctly "
                "(run `soto setup`)."
            )
        _node_classes[name] = nodes.NODE_CLASS_MAPPINGS[name]

    # ComfyUI draws the sampler's progress bar itself; colour it to match the
    # VAE bars in run_vace_transition(). Only uni_pc's (config.SAMPLER_NAME).
    import comfy.extra_samplers.uni_pc as uni_pc

    uni_pc.trange = functools.partial(uni_pc.trange, colour="magenta", dynamic_ncols=True)


def _call(node_cls, **kwargs):
    """Calls a node regardless of whether it's a classic (INPUT_TYPES/FUNCTION)
    or newer io.ComfyNode (define_schema/execute classmethod) node, always
    returning a plain tuple of outputs."""
    if hasattr(node_cls, "FUNCTION"):
        instance = node_cls()
        return getattr(instance, node_cls.FUNCTION)(**kwargs)
    return node_cls.execute(**kwargs).args


def _wait_for(t: torch.Tensor) -> None:
    """MPS runs asynchronously: a module call returns as soon as its work is
    queued, so a progress hook that doesn't wait counts it done right away
    and the bar then sits at the end. The work waited on is sequential anyway
    (each VAE chunk needs the last one's cache, each block the last one's
    output), so this costs next to nothing."""
    if t.device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def _vae_progress(module: torch.nn.Module, total: int, colour: str, frames_per_call):
    """Shows a progress bar over `total` frames while the Wan VAE's encoder or
    decoder (`module`) runs. Neither reports progress itself, but WanVAE calls
    them once per temporal chunk, so a forward hook can count the chunks off;
    `frames_per_call(t, done)` says how many frames a chunk of t input frames
    is worth, given how many the bar has already counted."""
    bar = tqdm(total=total, unit="frame", colour=colour, dynamic_ncols=True)

    # Must return None: anything else replaces the module's output.
    def count(_module, args, _output) -> None:
        _wait_for(args[0])
        bar.update(frames_per_call(args[0].shape[2], bar.n))

    handle = module.register_forward_hook(count)
    try:
        yield
    finally:
        handle.remove()
        bar.close()


@contextmanager
def _block_progress(diffusion_model: torch.nn.Module):
    """Shows a bar over the transformer blocks of each model pass while
    sampling, beneath ComfyUI's own per-step bar. Each step is one pass, or
    two when CFG's cond and uncond halves aren't batched together; each pass
    gets its own bar, saying which half it's running.

    A pass's bar closes the moment its last block finishes: tqdm only clears
    a nested bar's line properly while the bar above it is still open, and
    ComfyUI closes the step bar as soon as the last step ends."""
    blocks = [*diffusion_model.blocks, *diffusion_model.vace_blocks]
    bar = None

    def close() -> None:
        nonlocal bar
        if bar is not None:
            bar.close()
            bar = None

    def start(_module, _args, kwargs) -> None:
        nonlocal bar
        close()
        halves = kwargs["transformer_options"]["cond_or_uncond"]
        bar = tqdm(total=len(blocks), unit="block", colour="magenta", leave=False, dynamic_ncols=True,
                   desc="+".join(("cond", "uncond")[h] for h in sorted(halves)))

    # Must return None: anything else replaces the block's output.
    def count(_module, args, _output) -> None:
        if bar is None:
            return
        _wait_for(args[0])
        bar.update()
        if bar.n == bar.total:
            close()

    handles = [diffusion_model.blocks[0].register_forward_pre_hook(start, with_kwargs=True)]
    handles += [block.register_forward_hook(count) for block in blocks]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        close()


@dataclass
class TransitionResult:
    frames: torch.Tensor  # (length, H, W, 3) float32 in [0, 1] -- full VAE-decoded sequence


def run_vace_transition(
    *,
    control_video: torch.Tensor,
    control_mask: torch.Tensor,
    width: int,
    height: int,
    length: int,
    model: str,
    prompt: str,
    negative_prompt: str,
    steps: int,
    seed: int,
) -> TransitionResult:
    # Stage lines are stamped with the time since the start, in front rather
    # than after, so ComfyUI's own warnings landing mid-stage can't split them.
    start = time.monotonic()

    def stage(label: str, colour: str | None = None) -> None:
        clock = typer.style(tqdm.format_interval(time.monotonic() - start), dim=True)
        typer.echo(f"  {clock} {typer.style(label, fg=colour, bold=True)}")

    if not _node_classes:
        stage("starting ComfyUI")
    _bootstrap()

    spec = config.MODELS[model]
    # What the VAE actually covers: it drops frames past the last whole stride.
    stride = config.VACE_LENGTH_STRIDE
    vae_frames = 1 + (length - 1) // stride * stride

    # ComfyUI's own graph executor (execution.py) always runs node execution
    # under torch.inference_mode(); calling node classes directly bypasses
    # that entirely. Without it, every VAE causal-conv chunk and every
    # KSampler denoising step builds and *retains* an autograd graph (nothing
    # ever calls .backward(), but the graph keeps every intermediate
    # activation alive), so memory grows across the run instead of getting
    # freed between iterations. This was the real cause of a whole day's
    # worth of what looked like MPS-specific OOMs at any model size/attention
    # backend -- not an MPS bug, a missing no-grad context in this file.
    with torch.inference_mode():
        stage(f"loading UNet ({model})")
        if spec.loader == "UNETLoader":
            (loaded_model,) = _call(
                _node_classes["UNETLoader"], unet_name=spec.filename, weight_dtype="default"
            )
        else:
            (loaded_model,) = _call(_node_classes["UnetLoaderGGUF"], unet_name=spec.filename)
        (loaded_model,) = _call(
            _node_classes["ModelSamplingSD3"], model=loaded_model, shift=config.MODEL_SHIFT
        )

        stage("loading text encoder")
        (clip,) = _call(
            _node_classes["CLIPLoaderGGUF"],
            clip_name=config.TEXT_ENCODER_FILENAME,
            type=config.CLIP_TYPE,
        )
        stage("loading VAE")
        (vae,) = _call(_node_classes["VAELoader"], vae_name=config.VAE_FILENAME)

        stage("encoding prompt")
        (positive,) = _call(_node_classes["CLIPTextEncode"], clip=clip, text=prompt)
        (negative,) = _call(_node_classes["CLIPTextEncode"], clip=clip, text=negative_prompt)

        # The text encoder (and anything else loaded so far) isn't needed for
        # the rest of the pipeline; free it explicitly rather than relying on
        # ComfyUI's automatic model-manager eviction, whose free-memory
        # estimation is unreliable on MPS.
        import comfy.model_management

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(force=True)

        # WanVaceToVideo encodes the control video twice: once with the masked
        # (to-generate) frames blanked, once with only those frames.
        stage(f"encoding {length} control frames", "cyan")
        with _vae_progress(vae.first_stage_model.encoder, 2 * vae_frames, "cyan", lambda t, _done: t):
            positive, negative, latent, _trim_latent = _call(
                _node_classes["WanVaceToVideo"],
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                strength=1.0,
                control_video=control_video,
                control_masks=control_mask,
            )

        stage(f"sampling ({steps} steps)", "magenta")
        with _block_progress(loaded_model.model.diffusion_model):
            (samples,) = _call(
                _node_classes["KSampler"],
                model=loaded_model,
                seed=seed,
                steps=steps,
                cfg=config.CFG,
                sampler_name=config.SAMPLER_NAME,
                scheduler=config.SCHEDULER,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=config.DENOISE,
            )

        # Same again before decoding: with the VAE on the GPU, the UNet would
        # otherwise still be there, and ComfyUI's estimate of how much room
        # the decode needs isn't to be trusted on MPS.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(force=True)

        # The decoder's input is latent frames: the first becomes 1 frame,
        # every later one `stride` frames.
        stage(f"decoding {vae_frames} frames", "green")
        with _vae_progress(vae.first_stage_model.decoder, vae_frames, "green",
                           lambda t, done: stride * t - (stride - 1 if done == 0 else 0)):
            (images,) = _call(_node_classes["VAEDecode"], vae=vae, samples=samples)

    stage("done", "green")
    return TransitionResult(frames=images)
