from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import typer

from soto import config

app = typer.Typer(add_completion=False, no_args_is_help=True)

_CROSSFADE_HELP = (
    "Frames at each seam to crossfade from real footage into the model's own decoded version of it, "
    "instead of cutting straight to generated frames. Taken from the context, so can't exceed it."
)
_EXPORT_HELP = (
    "Also write every frame the model generated (including its regeneration of the real context "
    "frames) as a clip on the same timeline as the output, padded with the original footage "
    "wherever there are none -- for laying over the output and fading by hand in an editor."
)


def _fit_working_resolution(src_w: int, src_h: int) -> tuple[int, int]:
    target_area = config.DEFAULT_WIDTH * config.DEFAULT_HEIGHT
    aspect = src_w / src_h
    h = (target_area / aspect) ** 0.5
    w = h * aspect
    w = max(16, round(w / 16) * 16)
    h = max(16, round(h / 16) * 16)
    return int(w), int(h)


def _conform_clip(clip: Path, info, tmp_dir: Path, *, fps: float | None = None,
                  size: tuple[int, int] | None = None):
    """Returns (clip, info), re-encoded into `tmp_dir` only if the clip
    doesn't already match `fps` and `size`. `size` defaults to the clip's own,
    rounded down to even, since every segment is re-encoded as yuv420p
    and that needs even dimensions."""
    from soto import video

    fps = fps if fps is not None else info.fps
    width, height = size if size is not None else video.even_size(info.width, info.height)
    changes = []
    if abs(info.fps - fps) > 0.01:
        changes.append(f"fps {info.fps:.3f} -> {fps:.3f}")
    if (info.width, info.height) != (width, height):
        changes.append(f"size {info.width}x{info.height} -> {width}x{height}")
    if not changes:
        return clip, info
    print(f"  conforming {clip.name} ({', '.join(changes)})...")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    conformed = tmp_dir / f"{clip.stem}_conformed.mp4"
    video.conform(clip, conformed, fps=fps, width=width, height=height)
    return conformed, video.probe(conformed)


def _run_vace(context_a, context_b, gap_frames: int, width: int, height: int,
              model: str, prompt: str, steps: int, seed: int):
    """Generates `gap_frames` between two context windows; returns the full
    decoded sequence (context + gap + context)."""
    from soto import comfy_engine, video

    control_video, control_mask = video.build_control_batch(context_a, context_b, gap_frames, height, width)
    return comfy_engine.run_vace_transition(
        control_video=control_video,
        control_mask=control_mask,
        width=width,
        height=height,
        length=context_a.shape[0] + gap_frames + context_b.shape[0],
        model=model,
        prompt=prompt,
        negative_prompt=config.DEFAULT_NEGATIVE_PROMPT,
        steps=steps,
        seed=seed,
    )


def _plan_context(start: int, end: int, lo: int, hi: int, want: int) -> tuple[int, int] | None:
    """Picks how many real frames of context to take on each side of the blank
    span [start, end), for a VACE run of length n_a + (end - start) + n_b.

    - `lo` / `hi` bound the usable real footage: before-context must come from
      [lo, start), after-context from [end, hi) -- i.e. it can't reach past the
      clip edges or into a neighbouring blank interval.
    - `want` is the requested context per side (from --context), in frames.
    - The gap length is fixed (it's the blank we're replacing), so the
      VACE stride constraint (length - 1) % config.VACE_LENGTH_STRIDE == 0
      has to be satisfied by adjusting n_a / n_b instead.

    Returns (n_a, n_b), both >= 1, or None if no valid split exists."""
    gap = end - start
    max_a, max_b = start - lo, hi - end
    base_a, base_b = min(want, max_a), min(want, max_b)
    stride = config.VACE_LENGTH_STRIDE

    # The stride is tiny, so just search every split within one stride of the
    # clamped request. Prefer the smallest change, then adding context over
    # dropping it, then the most even split.
    candidates = [
        (a, b)
        for a in range(base_a - stride + 1, base_a + stride)
        for b in range(base_b - stride + 1, base_b + stride)
        if 1 <= a <= max_a and 1 <= b <= max_b and (a + gap + b - 1) % stride == 0
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda ab: (
            abs(ab[0] - base_a) + abs(ab[1] - base_b),
            (ab[0] + ab[1]) < (base_a + base_b),
            abs(ab[0] - ab[1]),
        ),
    )


def _plan_extend_context(gap: int, available: int, want: int) -> int | None:
    """Picks how many frames of context to take from the end of the clip for
    an extension of `gap` frames: the one-sided counterpart of _plan_context.
    `gap` is the user's exact --frames, so the stride constraint is met by
    nudging the context length instead. Returns None if no valid count exists."""
    base = min(want, available)
    stride = config.VACE_LENGTH_STRIDE
    candidates = [
        a for a in range(base - stride + 1, base + stride)
        if 1 <= a <= available and (a + gap - 1) % stride == 0
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda a: (abs(a - base), a < base))


@app.command()
def setup() -> None:
    """Vendor ComfyUI + ComfyUI-GGUF and download the model pieces
    (UNet, VAE, text encoder)."""
    from soto import setup as setup_mod

    setup_mod.run()


@app.command()
def doctor() -> None:
    """Check that the environment is ready to run `soto transition`."""
    ok = True

    ffmpeg_ok = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
    print(f"[{'ok' if ffmpeg_ok else 'FAIL'}] ffmpeg/ffprobe on PATH")
    ok &= ffmpeg_ok

    comfyui_ok = (config.COMFYUI_DIR / "main.py").is_file()
    print(f"[{'ok' if comfyui_ok else 'FAIL'}] vendor/ComfyUI present (run `soto setup`)")
    ok &= comfyui_ok

    gguf_node_ok = (config.COMFYUI_GGUF_DIR / "nodes.py").is_file()
    print(f"[{'ok' if gguf_node_ok else 'FAIL'}] ComfyUI-GGUF custom node present (run `soto setup`)")
    ok &= gguf_node_ok

    for name, spec in config.MODELS.items():
        present = (config.MODELS_DIR / spec.filename).is_file()
        note = "run `soto setup`" if spec.remote_path else "pre-downloaded"
        print(f"[{'ok' if present else 'FAIL'}] models/{spec.filename} (--model {name}; {note})")
        ok &= present

    vae_ok = (config.VAE_DIR / config.VAE_FILENAME).is_file()
    print(f"[{'ok' if vae_ok else 'FAIL'}] models/vae/{config.VAE_FILENAME} (run `soto setup`)")
    ok &= vae_ok

    te_ok = (config.TEXT_ENCODERS_DIR / config.TEXT_ENCODER_FILENAME).is_file()
    print(f"[{'ok' if te_ok else 'FAIL'}] models/text_encoders/{config.TEXT_ENCODER_FILENAME} (run `soto setup`)")
    ok &= te_ok

    try:
        import torch

        mps_ok = torch.backends.mps.is_available()
        print(f"[{'ok' if mps_ok else 'warn'}] torch MPS available (torch {torch.__version__})")
    except ImportError:
        print("[FAIL] torch not installed (run `uv sync`)")
        ok = False

    if not ok:
        raise typer.Exit(code=1)
    print("\nEverything looks good.")


@app.command()
def transition(
    clip_a: Path = typer.Argument(..., exists=True, help="First clip (the transition plays out of this one)."),
    clip_b: Path = typer.Argument(..., exists=True, help="Second clip (the transition plays into this one)."),
    output: Path = typer.Option(..., "-o", "--output", help="Output video path."),
    duration: float = typer.Option(..., "--duration", help="Seconds of *generated* transition."),
    context: float = typer.Option(
        config.DEFAULT_CONTEXT_S, "--context", help="Seconds of real context taken from each clip's edge."
    ),
    model: str = typer.Option(
        config.DEFAULT_MODEL, "--model",
        help=f"Which checkpoint to use: {list(config.MODELS)}.",
    ),
    prompt: str = typer.Option(config.DEFAULT_PROMPT, "--prompt", help="Positive prompt guiding the transition."),
    steps: int = typer.Option(config.DEFAULT_STEPS, "--steps", help="Sampling steps."),
    seed: int = typer.Option(None, "--seed", help="Random seed (default: random each run)."),
    crossfade: int = typer.Option(0, "--crossfade", min=0, help=_CROSSFADE_HELP),
    export_generated: Path = typer.Option(None, "--export-generated", dir_okay=False, help=_EXPORT_HELP),
) -> None:
    """Generate an AI-blended transition between two clips and splice it in."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")

    from soto import video

    print(f"Probing {clip_a} and {clip_b}...")
    info_a = video.probe(clip_a)
    info_b = video.probe(clip_b)

    # Clip B is conformed to clip A's frame rate and size, so the two can be
    # cut together into one stream.
    conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
    clip_a, info_a = _conform_clip(clip_a, info_a, conform_dir / "a")
    clip_b, info_b = _conform_clip(
        clip_b, info_b, conform_dir / "b", fps=info_a.fps, size=(info_a.width, info_a.height)
    )

    fps = info_a.fps
    context_frames = max(1, round(context * fps))
    gap_frames = max(1, round(duration * fps))
    length = 2 * context_frames + gap_frames
    # Wan's causal VAE has temporal stride 4 (+1 for the leading frame), so
    # length % 4 == 1 is the well-formed frame count -- round the gap up to it.
    remainder = (length - 1) % config.VACE_LENGTH_STRIDE
    if remainder != 0:
        gap_frames += config.VACE_LENGTH_STRIDE - remainder
        length = 2 * context_frames + gap_frames

    width, height = _fit_working_resolution(info_a.width, info_a.height)
    print(f"  working resolution {width}x{height}, {context_frames} context frames/side, {gap_frames} gap frames")

    print("Extracting context frames...")
    context_a = video.extract_tail_context(clip_a, info_a, context_frames, width, height)
    context_b = video.extract_head_context(clip_b, info_b, context_frames, width, height)
    max_crossfade = min(context_a.shape[0], context_b.shape[0])
    if crossfade > max_crossfade:
        raise typer.BadParameter(
            f"can't exceed the {max_crossfade} context frames per side", param_hint="--crossfade"
        )

    resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    print(f"Running Wan2.1-VACE ({model}) transition (seed={resolved_seed}, steps={steps})...")
    result = _run_vace(context_a, context_b, gap_frames, width, height, model, prompt, steps, resolved_seed)
    if export_generated is not None:
        print(f"Exporting generated frames to {export_generated}...")
        video.export_generated(
            [
                (clip_a, 0, info_a.n_frames - context_a.shape[0]),
                result.frames.cpu().numpy(),
                (clip_b, context_b.shape[0], info_b.n_frames),
            ],
            info_a,
            export_generated,
        )

    print("Splicing final output...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="soto-splice-"))
    output.parent.mkdir(parents=True, exist_ok=True)
    video.splice_output(
        clip_a=clip_a,
        clip_b=clip_b,
        info_a=info_a,
        info_b=info_b,
        generated_frames=result.frames,
        context_frames_a=context_a.shape[0],
        gap_frames=gap_frames,
        context_frames_b=context_b.shape[0],
        crossfade=crossfade,
        output_path=output,
        tmp_dir=tmp_dir,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(conform_dir, ignore_errors=True)
    print(f"Done: {output}")


@app.command()
def bridge(
    clip: Path = typer.Argument(..., exists=True, help="Clip containing blank (flat-colour) intervals."),
    output: Path = typer.Option(..., "-o", "--output", help="Output video path."),
    color: str = typer.Option("#000000", "--color", help="Hex colour of the blank frames, e.g. '#000000' or 'fff'."),
    tolerance: float = typer.Option(
        0.12, "--tolerance", help="Max per-channel distance from --color (0-1) for a pixel to count as blank."
    ),
    max_gap: float = typer.Option(5.0, "--max-gap", help="Blank intervals longer than this (seconds) are left as-is."),
    context: float = typer.Option(
        config.DEFAULT_CONTEXT_S, "--context", help="Seconds of real context taken from each side of a blank."
    ),
    model: str = typer.Option(
        config.DEFAULT_MODEL, "--model",
        help=f"Which checkpoint to use: {list(config.MODELS)}.",
    ),
    prompt: str = typer.Option(config.DEFAULT_PROMPT, "--prompt", help="Positive prompt guiding the generated frames."),
    steps: int = typer.Option(config.DEFAULT_STEPS, "--steps", help="Sampling steps."),
    seed: int = typer.Option(None, "--seed", help="Random seed (default: random each run)."),
    crossfade: int = typer.Option(0, "--crossfade", min=0, help=_CROSSFADE_HELP),
    export_generated: Path = typer.Option(None, "--export-generated", dir_okay=False, help=_EXPORT_HELP),
) -> None:
    """Find blank intervals inside a clip and replace each with a generated bridge."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    from soto import video

    try:
        rgb = video.parse_hex_color(color)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--color") from None

    print(f"Probing {clip}...")
    info = video.probe(clip)
    conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
    clip, info = _conform_clip(clip, info, conform_dir)
    fps = info.fps

    print(f"Scanning for blank frames matching {color} (tolerance {tolerance})...")
    intervals = video.detect_flat_intervals(clip, rgb, tolerance)
    max_gap_frames = round(max_gap * fps)
    context_frames = max(1, round(context * fps))
    width, height = _fit_working_resolution(info.width, info.height)

    plans: list[tuple[int, int, int, int]] = []
    for i, (start, end) in enumerate(intervals):
        span = f"frames {start}-{end - 1} ({(end - start) / fps:.2f}s)"
        if start == 0 or end == info.n_frames:
            print(f"  skip {span}: touches the clip edge")
            continue
        if end - start > max_gap_frames:
            print(f"  skip {span}: longer than --max-gap {max_gap}s")
            continue
        # Context can't reach into a neighbouring blank, bridged or not.
        lo = intervals[i - 1][1] if i > 0 else 0
        hi = intervals[i + 1][0] if i + 1 < len(intervals) else info.n_frames
        planned = _plan_context(start, end, lo, hi, context_frames)
        if planned is None:
            print(f"  skip {span}: not enough real footage around it for context")
            continue
        n_a, n_b = planned
        print(f"  bridge {span} with {n_a}+{n_b} context frames")
        plans.append((start, end, n_a, n_b))

    if not plans:
        shutil.rmtree(conform_dir, ignore_errors=True)
        print("Nothing to bridge; no output written.")
        return

    # Crossfades come out of each side's context, and two neighbouring
    # bridges can't both fade over the same real frames, so split any shared
    # stretch of real footage between them.
    fades = [[min(crossfade, n_a), min(crossfade, n_b)] for _, _, n_a, n_b in plans]
    for (_, prev_end, _, _), (start, _, _, _), prev, cur in zip(plans, plans[1:], fades, fades[1:]):
        real = start - prev_end
        if prev[1] + cur[0] > real:
            prev[1] = min(prev[1], real // 2)
            cur[0] = min(cur[0], real - prev[1])

    resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    print(f"Working resolution {width}x{height}, seed={resolved_seed}, steps={steps}")
    fills = []
    decoded_runs = []
    for k, ((start, end, n_a, n_b), (x_a, x_b)) in enumerate(zip(plans, fades), 1):
        fade_note = f" with {x_a}+{x_b} crossfade frames" if crossfade else ""
        print(f"[{k}/{len(plans)}] Running Wan2.1-VACE ({model}) for frames {start}-{end - 1}{fade_note}...")
        context_a = video.read_frames(clip, start - n_a, n_a, width, height)
        context_b = video.read_frames(clip, end, n_b, width, height)
        result = _run_vace(context_a, context_b, end - start, width, height, model, prompt, steps, resolved_seed)
        if export_generated is not None:
            decoded_runs.append((start - n_a, end + n_b, result.frames.cpu().numpy()))
        gap_end = n_a + (end - start)
        fills.append((
            start,
            end,
            result.frames[n_a:gap_end].cpu().numpy(),
            result.frames[n_a - x_a:n_a].cpu().numpy(),
            result.frames[gap_end:gap_end + x_b].cpu().numpy(),
        ))

    print("Splicing final output...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="soto-bridge-"))
    output.parent.mkdir(parents=True, exist_ok=True)
    video.splice_fills(clip=clip, info=info, fills=fills, output_path=output, tmp_dir=tmp_dir)
    if export_generated is not None:
        # Neighbouring runs' context can overlap; where it does, the earlier
        # run's trailing frames give way to the later run's leading ones.
        parts = []
        cursor = 0
        for i, (run_start, run_end, decoded) in enumerate(decoded_runs):
            parts.append((clip, cursor, run_start))
            next_start = decoded_runs[i + 1][0] if i + 1 < len(decoded_runs) else run_end
            parts.append(decoded[:min(run_end, next_start) - run_start])
            cursor = min(run_end, next_start)
        parts.append((clip, cursor, info.n_frames))
        print(f"Exporting generated frames to {export_generated}...")
        video.export_generated(parts, info, export_generated, audio_source=clip)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(conform_dir, ignore_errors=True)
    print(f"Done: {output}")


@app.command()
def extend(
    clip: Path = typer.Argument(..., exists=True, help="Clip to extend past its last frame."),
    output: Path = typer.Option(..., "-o", "--output", help="Output video path."),
    frames: int = typer.Option(..., "--frames", min=1, help="Frames to generate per iteration."),
    iterations: int = typer.Option(
        1, "--iterations", min=1,
        help="How many times to extend; each iteration continues from the previous one's result.",
    ),
    context: float = typer.Option(
        config.DEFAULT_CONTEXT_S, "--context", help="Seconds of context taken from the end of the clip."
    ),
    model: str = typer.Option(
        config.DEFAULT_MODEL, "--model",
        help=f"Which checkpoint to use: {list(config.MODELS)}.",
    ),
    prompt: str = typer.Option(config.DEFAULT_PROMPT, "--prompt", help="Positive prompt guiding the generated frames."),
    steps: int = typer.Option(config.DEFAULT_STEPS, "--steps", help="Sampling steps."),
    seed: int = typer.Option(None, "--seed", help="Random seed (default: random each run)."),
    crossfade: int = typer.Option(0, "--crossfade", min=0, help=_CROSSFADE_HELP),
    export_generated: Path = typer.Option(None, "--export-generated", dir_okay=False, help=_EXPORT_HELP),
) -> None:
    """Generate new frames continuing on from the end of a clip, optionally
    several times over, and append them."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")

    import numpy as np

    from soto import video

    print(f"Probing {clip}...")
    info = video.probe(clip)
    conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
    clip, info = _conform_clip(clip, info, conform_dir)
    fps = info.fps
    width, height = _fit_working_resolution(info.width, info.height)

    # Later iterations take context from a history that's only ever longer
    # than the original clip, so the count planned here works for all of them.
    n_ctx = _plan_extend_context(frames, info.n_frames, max(1, round(context * fps)))
    if n_ctx is None:
        raise typer.BadParameter(f"{clip} is too short to take context from", param_hint="clip")
    if crossfade > n_ctx:
        raise typer.BadParameter(f"can't exceed the {n_ctx} context frames", param_hint="--crossfade")
    total = frames * iterations
    print(f"  working resolution {width}x{height}, {n_ctx} context frames, "
          f"{iterations} x {frames} new frames ({total / fps:.2f}s)")

    # Context for every iteration after the first comes from generated frames,
    # kept at working resolution so nothing goes through a lossy re-encode
    # between iterations.
    history = video.extract_tail_context(clip, info, n_ctx, width, height)
    empty = np.zeros((0, height, width, 3), dtype=np.float32)
    resolved_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    print(f"seed={resolved_seed}, steps={steps}")

    generated = []
    for i in range(iterations):
        print(f"[{i + 1}/{iterations}] Running Wan2.1-VACE ({model}) for {frames} frames...")
        context_a = history[-n_ctx:]
        # Vary the seed per iteration so successive extensions don't reuse the
        # same noise and settle into repeating motion.
        iter_seed = (resolved_seed + i) % 2**32
        result = _run_vace(context_a, empty, frames, width, height, model, prompt, steps, iter_seed)
        chunk = result.frames[n_ctx:n_ctx + frames].cpu().numpy()
        if i == 0:
            # Later iterations only ever join generated frames to generated
            # frames, so the first seam is the only one with real footage.
            regenerated = result.frames[:n_ctx].cpu().numpy()
        generated.append(chunk)
        history = np.concatenate([context_a, chunk], axis=0)

    print("Splicing final output...")
    tmp_dir = Path(tempfile.mkdtemp(prefix="soto-extend-"))
    output.parent.mkdir(parents=True, exist_ok=True)
    video.append_frames(
        clip=clip, info=info, frames=np.concatenate(generated, axis=0),
        lead_in=regenerated[n_ctx - crossfade:], output_path=output, tmp_dir=tmp_dir,
    )
    if export_generated is not None:
        print(f"Exporting generated frames to {export_generated}...")
        video.export_generated(
            [(clip, 0, info.n_frames - n_ctx), regenerated, *generated], info, export_generated, audio_source=clip
        )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(conform_dir, ignore_errors=True)
    print(f"Done: {output}")


if __name__ == "__main__":
    app()
