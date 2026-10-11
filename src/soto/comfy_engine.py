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
import sys
from dataclasses import dataclass

import torch

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
    # --cpu-vae works around a real, model-size-independent MPS memory bug in
    # Wan's causal VAE: measured directly, encoding N frames at 832x480 on
    # MPS took 2.87GB at N=4, 24.4GB at N=9, and OOM'd (>41GB) at N=17 --
    # wildly non-linear, not proportional to actual tensor size, so the
    # causal feat_cache chunking isn't releasing memory between chunks on
    # this backend. Forcing VAE encode/decode onto CPU sidesteps it; VAE is
    # cheap relative to the sampling loop, so this is a small wall-clock cost.
    # --use-pytorch-cross-attention forces PyTorch's native SDPA kernel.
    # Without it, ComfyUI's default attention pick on MPS (no flash-attn/
    # xformers here) fell through to its own `sub_quadratic_attention`
    # fallback, which upcasts query/key to fp32 and -- even for a genuinely
    # tiny 192x112x9-frame VACE-conditioned sequence -- never took its
    # chunked code path, materializing a full fp32 attention matrix and
    # hitting the same ~42GB ceiling regardless of UNet size (14B or 1.3B).
    old_argv, sys.argv = sys.argv, [
        "comfyui", "--lowvram", "--cpu-vae", "--use-pytorch-cross-attention",
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


def _call(node_cls, **kwargs):
    """Calls a node regardless of whether it's a classic (INPUT_TYPES/FUNCTION)
    or newer io.ComfyNode (define_schema/execute classmethod) node, always
    returning a plain tuple of outputs."""
    if hasattr(node_cls, "FUNCTION"):
        instance = node_cls()
        return getattr(instance, node_cls.FUNCTION)(**kwargs)
    return node_cls.execute(**kwargs).args


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
    _bootstrap()

    spec = config.MODELS[model]

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
        if spec.loader == "UNETLoader":
            (loaded_model,) = _call(
                _node_classes["UNETLoader"], unet_name=spec.filename, weight_dtype="default"
            )
        else:
            (loaded_model,) = _call(_node_classes["UnetLoaderGGUF"], unet_name=spec.filename)
        (loaded_model,) = _call(
            _node_classes["ModelSamplingSD3"], model=loaded_model, shift=config.MODEL_SHIFT
        )

        (clip,) = _call(
            _node_classes["CLIPLoaderGGUF"],
            clip_name=config.TEXT_ENCODER_FILENAME,
            type=config.CLIP_TYPE,
        )
        (vae,) = _call(_node_classes["VAELoader"], vae_name=config.VAE_FILENAME)

        (positive,) = _call(_node_classes["CLIPTextEncode"], clip=clip, text=prompt)
        (negative,) = _call(_node_classes["CLIPTextEncode"], clip=clip, text=negative_prompt)

        # The text encoder (and anything else loaded so far) isn't needed for
        # the rest of the pipeline; free it explicitly rather than relying on
        # ComfyUI's automatic model-manager eviction, whose free-memory
        # estimation is unreliable on MPS.
        import comfy.model_management

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(force=True)

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

        (images,) = _call(_node_classes["VAEDecode"], vae=vae, samples=samples)

    return TransitionResult(frames=images)
