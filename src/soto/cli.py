from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import typer

from soto import config
from soto.metadata import RunLog

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
_PREVIEW_HELP = (
    f"Quick low-quality preview: {list(config.PREVIEW_TIERS)}. 'fast' drops the working resolution and steps; "
    "'draft' only drops steps, so re-running with the same --seed and no --preview gives the same layout."
)
_STEPS_HELP = f"Sampling steps (default {config.DEFAULT_STEPS}, or fewer with --preview)."
_NO_METADATA_HELP = (
    "Don't write the <output>.json sidecar (settings, seeds, timings and the timeline of real, context "
    "and generated frames) or embed its chapter markers in the output."
)


def _fit_working_resolution(src_w: int, src_h: int, area: int | None = None) -> tuple[int, int]:
    target_area = area or config.DEFAULT_WIDTH * config.DEFAULT_HEIGHT
    aspect = src_w / src_h
    h = (target_area / aspect) ** 0.5
    w = h * aspect
    w = max(16, round(w / 16) * 16)
    h = max(16, round(h / 16) * 16)
    return int(w), int(h)


def _resolve_quality(preview: str | None, steps: int | None, log: RunLog) -> tuple[int | None, int]:
    """Returns (working area override or None, steps) for --preview; an
    explicit --steps always wins."""
    if preview is None:
        return None, steps if steps is not None else config.DEFAULT_STEPS
    if preview not in config.PREVIEW_TIERS:
        raise typer.BadParameter(f"must be one of {list(config.PREVIEW_TIERS)}", param_hint="--preview")
    area, preview_steps = config.PREVIEW_TIERS[preview]
    if steps is None:
        log.adjust("steps", config.DEFAULT_STEPS, preview_steps, f"--preview {preview}")
    if area is not None:
        log.adjust("working_area", config.DEFAULT_WIDTH * config.DEFAULT_HEIGHT, area, f"--preview {preview}")
    return area, steps if steps is not None else preview_steps


def _resolve_seed(seed: int | None, log: RunLog) -> int:
    resolved = seed if seed is not None else random.randint(0, 2**32 - 1)
    log.settings["seed"] = resolved
    log.settings["seed_source"] = "given" if seed is not None else "random"
    return resolved


def _log_vace_settings(log: RunLog, *, model: str, prompt: str, steps: int, preview: str | None,
                       width: int, height: int, context: float, crossfade: int,
                       export_generated: Path | None) -> None:
    log.settings.update({
        "model": model,
        "checkpoint": config.MODELS[model].filename,
        "prompt": prompt,
        "negative_prompt": config.DEFAULT_NEGATIVE_PROMPT,
        "steps": steps,
        "preview": preview,
        "working_resolution": [width, height],
        "cfg": config.CFG,
        "sampler": config.SAMPLER_NAME,
        "scheduler": config.SCHEDULER,
        "shift": config.MODEL_SHIFT,
        "denoise": config.DENOISE,
        "context_s": context,
        "crossfade_frames": crossfade,
        "export_generated": str(export_generated.resolve()) if export_generated else None,
    })


def _context_reason(want: int, available: int, used: int) -> str:
    reasons = []
    if available < want:
        reasons.append(f"only {available} real frames available")
    if used != min(want, available):
        reasons.append(f"nudged so the VACE length fits its stride of {config.VACE_LENGTH_STRIDE}")
    return "; ".join(reasons)


def _preview_note(preview: str | None) -> str:
    return f" (preview: {preview})" if preview else ""


def _print_done(output: Path, seed: int, preview: str | None, log: RunLog, fps: float) -> None:
    log.write(output, fps)
    print(f"Done: {output} (seed={seed})")
    if preview:
        print(f"Preview only -- re-run without --preview and with --seed {seed} for the full-quality version.")


def _conform_clip(clip: Path, info, tmp_dir: Path, log: RunLog, role: str, *, fps: float | None = None,
                  size: tuple[int, int] | None = None):
    """Returns (clip, info), re-encoded into `tmp_dir` only if the clip
    doesn't already match `fps` and `size`. `size` defaults to the clip's own,
    rounded down to even, since every segment is re-encoded as yuv420p
    and that needs even dimensions. Logs the clip as an input named `role`."""
    from soto import video

    fps = fps if fps is not None else info.fps
    width, height = size if size is not None else video.even_size(info.width, info.height)
    changes = []
    if abs(info.fps - fps) > 0.01:
        changes.append(f"fps {info.fps:.3f} -> {fps:.3f}")
    if (info.width, info.height) != (width, height):
        changes.append(f"size {info.width}x{info.height} -> {width}x{height}")
    log.input(role, clip, info, changes)
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

    # Only `soto dream` needs these, so their absence is a warning, not a failure.
    dreams = sorted(p.stem for p in config.PIX2PIXHD_DIR.glob("*.pth"))
    print(f"[{'ok' if dreams else 'warn'}] models/pix2pixhd/ checkpoints for `soto dream`: {', '.join(dreams) or 'none'}")

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
    steps: int = typer.Option(None, "--steps", help=_STEPS_HELP),
    preview: str = typer.Option(None, "--preview", help=_PREVIEW_HELP),
    seed: int = typer.Option(None, "--seed", help="Random seed (default: random each run)."),
    crossfade: int = typer.Option(0, "--crossfade", min=0, help=_CROSSFADE_HELP),
    export_generated: Path = typer.Option(None, "--export-generated", dir_okay=False, help=_EXPORT_HELP),
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_NO_METADATA_HELP),
) -> None:
    """Generate an AI-blended transition between two clips and splice it in."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    log = RunLog("transition", enabled=not no_metadata)
    with log.guard(output):
        area, steps = _resolve_quality(preview, steps, log)

        from soto import video

        print(f"Probing {clip_a} and {clip_b}...")
        info_a = video.probe(clip_a)
        info_b = video.probe(clip_b)

        # Clip B is conformed to clip A's frame rate and size, so the two can be
        # cut together into one stream.
        conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
        clip_a, info_a = _conform_clip(clip_a, info_a, conform_dir / "a", log, "a")
        clip_b, info_b = _conform_clip(
            clip_b, info_b, conform_dir / "b", log, "b", fps=info_a.fps, size=(info_a.width, info_a.height)
        )

        fps = info_a.fps
        context_frames = max(1, round(context * fps))
        gap_frames = requested_gap = max(1, round(duration * fps))
        length = 2 * context_frames + gap_frames
        # Wan's causal VAE has temporal stride 4 (+1 for the leading frame), so
        # length % 4 == 1 is the well-formed frame count -- round the gap up to it.
        remainder = (length - 1) % config.VACE_LENGTH_STRIDE
        if remainder != 0:
            gap_frames += config.VACE_LENGTH_STRIDE - remainder
            length = 2 * context_frames + gap_frames
            log.adjust("gap_frames", requested_gap, gap_frames,
                       f"rounded up so the VACE length fits its stride of {config.VACE_LENGTH_STRIDE}")

        width, height = _fit_working_resolution(info_a.width, info_a.height, area)
        _log_vace_settings(log, model=model, prompt=prompt, steps=steps, preview=preview, width=width, height=height,
                           context=context, crossfade=crossfade, export_generated=export_generated)
        log.settings.update({"duration_s": duration, "gap_frames": gap_frames, "context_frames": context_frames})
        print(f"  working resolution {width}x{height}{_preview_note(preview)}, "
              f"{context_frames} context frames/side, {gap_frames} gap frames")

        print("Extracting context frames...")
        context_a = video.extract_tail_context(clip_a, info_a, context_frames, width, height)
        context_b = video.extract_head_context(clip_b, info_b, context_frames, width, height)
        n_a, n_b = context_a.shape[0], context_b.shape[0]
        for side, n in (("a", n_a), ("b", n_b)):
            if n < context_frames:
                log.adjust(f"context_frames_{side}", context_frames, n, f"clip {side} only has {n} frames")
        max_crossfade = min(n_a, n_b)
        if crossfade > max_crossfade:
            raise typer.BadParameter(
                f"can't exceed the {max_crossfade} context frames per side", param_hint="--crossfade"
            )

        resolved_seed = _resolve_seed(seed, log)
        print(f"Running Wan2.1-VACE ({model}) transition (seed={resolved_seed}, steps={steps})...")
        with log.run(seed=resolved_seed, steps=steps, context_frames_before=n_a, gap_frames=gap_frames,
                     context_frames_after=n_b,
                     context_source_before={"clip": "a", "frames": [info_a.n_frames - n_a, info_a.n_frames]},
                     context_source_after={"clip": "b", "frames": [0, n_b]}):
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
        # Output timeline: all of clip A, the gap, then all of clip B.
        gap_start, gap_end = info_a.n_frames, info_a.n_frames + gap_frames
        log.segment("real", 0, gap_start - crossfade, "clip a", source="a", source_frames=[0, gap_start - crossfade])
        log.segment("crossfade", gap_start - crossfade, gap_start, "crossfade out of clip a", run=1)
        log.segment("generated", gap_start, gap_end, f"generated (seed {resolved_seed})", run=1)
        log.segment("crossfade", gap_end, gap_end + crossfade, "crossfade into clip b", run=1)
        log.segment("real", gap_end + crossfade, gap_end + info_b.n_frames, "clip b",
                    source="b", source_frames=[crossfade, info_b.n_frames])
        log.segment("context", gap_start - n_a, gap_start, "context from clip a", run=1)
        log.segment("context", gap_end, gap_end + n_b, "context from clip b", run=1)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, resolved_seed, preview, log, fps)


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
    steps: int = typer.Option(None, "--steps", help=_STEPS_HELP),
    preview: str = typer.Option(None, "--preview", help=_PREVIEW_HELP),
    seed: int = typer.Option(None, "--seed", help="Random seed (default: random each run)."),
    crossfade: int = typer.Option(0, "--crossfade", min=0, help=_CROSSFADE_HELP),
    export_generated: Path = typer.Option(None, "--export-generated", dir_okay=False, help=_EXPORT_HELP),
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_NO_METADATA_HELP),
) -> None:
    """Find blank intervals inside a clip and replace each with a generated bridge."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    log = RunLog("bridge", enabled=not no_metadata)
    with log.guard(output):
        area, steps = _resolve_quality(preview, steps, log)
        from soto import video

        try:
            rgb = video.parse_hex_color(color)
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--color") from None

        print(f"Probing {clip}...")
        info = video.probe(clip)
        conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
        clip, info = _conform_clip(clip, info, conform_dir, log, "clip")
        fps = info.fps

        print(f"Scanning for blank frames matching {color} (tolerance {tolerance})...")
        intervals = video.detect_flat_intervals(clip, rgb, tolerance)
        max_gap_frames = round(max_gap * fps)
        context_frames = max(1, round(context * fps))
        width, height = _fit_working_resolution(info.width, info.height, area)
        _log_vace_settings(log, model=model, prompt=prompt, steps=steps, preview=preview, width=width, height=height,
                           context=context, crossfade=crossfade, export_generated=export_generated)
        log.settings.update({"color": color, "tolerance": tolerance, "max_gap_s": max_gap,
                             "context_frames": context_frames})

        plans: list[tuple[int, int, int, int]] = []
        for i, (start, end) in enumerate(intervals):
            span = f"frames {start}-{end - 1} ({(end - start) / fps:.2f}s)"
            blank = {"start_frame": start, "end_frame": end, "seconds": round((end - start) / fps, 3)}
            if start == 0 or end == info.n_frames:
                print(f"  skip {span}: touches the clip edge")
                log.append("blank_intervals", {**blank, "outcome": "skipped", "reason": "touches the clip edge"})
                continue
            if end - start > max_gap_frames:
                print(f"  skip {span}: longer than --max-gap {max_gap}s")
                log.append("blank_intervals", {**blank, "outcome": "skipped",
                                               "reason": f"longer than --max-gap {max_gap}s"})
                continue
            # Context can't reach into a neighbouring blank, bridged or not.
            lo = intervals[i - 1][1] if i > 0 else 0
            hi = intervals[i + 1][0] if i + 1 < len(intervals) else info.n_frames
            planned = _plan_context(start, end, lo, hi, context_frames)
            if planned is None:
                print(f"  skip {span}: not enough real footage around it for context")
                log.append("blank_intervals", {**blank, "outcome": "skipped",
                                               "reason": "not enough real footage around it for context"})
                continue
            n_a, n_b = planned
            print(f"  bridge {span} with {n_a}+{n_b} context frames")
            log.append("blank_intervals", {**blank, "outcome": "bridged",
                                           "context_frames_before": n_a, "context_frames_after": n_b})
            for side, used, available in (("before", n_a, start - lo), ("after", n_b, hi - end)):
                if used != context_frames:
                    log.adjust(f"context_frames_{side} (frames {start}-{end - 1})", context_frames, used,
                               _context_reason(context_frames, available, used))
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
        if crossfade:
            for (start, end, n_a, n_b), (x_a, x_b) in zip(plans, fades):
                for side, used, n in (("before", x_a, n_a), ("after", x_b, n_b)):
                    if used < crossfade:
                        reason = (f"limited to the {n} context frames" if used == n
                                  else "split with the neighbouring bridge's fade over shared real footage")
                        log.adjust(f"crossfade_{side} (frames {start}-{end - 1})", crossfade, used, reason)

        resolved_seed = _resolve_seed(seed, log)
        print(f"Working resolution {width}x{height}{_preview_note(preview)}, seed={resolved_seed}, steps={steps}")
        fills = []
        decoded_runs = []
        for k, ((start, end, n_a, n_b), (x_a, x_b)) in enumerate(zip(plans, fades), 1):
            fade_note = f" with {x_a}+{x_b} crossfade frames" if crossfade else ""
            print(f"[{k}/{len(plans)}] Running Wan2.1-VACE ({model}) for frames {start}-{end - 1}{fade_note}...")
            context_a = video.read_frames(clip, start - n_a, n_a, width, height)
            context_b = video.read_frames(clip, end, n_b, width, height)
            with log.run(seed=resolved_seed, steps=steps, context_frames_before=n_a, gap_frames=end - start,
                         context_frames_after=n_b, gap=[start, end], crossfade=[x_a, x_b]):
                result = _run_vace(
                    context_a, context_b, end - start, width, height, model, prompt, steps, resolved_seed
                )
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
        # Bridging keeps every frame where it was, so output frames are input frames.
        cursor = 0
        for k, ((start, end, n_a, n_b), (x_a, x_b)) in enumerate(zip(plans, fades), 1):
            log.segment("real", cursor, start - x_a, "clip", source="clip", source_frames=[cursor, start - x_a])
            log.segment("crossfade", start - x_a, start, f"crossfade into bridge {k}", run=k)
            log.segment("generated", start, end, f"bridge {k} (seed {resolved_seed})", run=k)
            log.segment("crossfade", end, end + x_b, f"crossfade out of bridge {k}", run=k)
            log.segment("context", start - n_a, start, f"context before bridge {k}", run=k)
            log.segment("context", end, end + n_b, f"context after bridge {k}", run=k)
            cursor = end + x_b
        log.segment("real", cursor, info.n_frames, "clip", source="clip", source_frames=[cursor, info.n_frames])
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, resolved_seed, preview, log, fps)


_EXTEND_SIDES = ("start", "end", "bookend")


def _extend_side(clip: Path, info, n_ctx: int, frames: int, iterations: int, width: int, height: int,
                 model: str, prompt: str, steps: int, seed: int, log: RunLog, *, at_start: bool):
    """Runs `extend`'s iterations off one end of the clip. Returns (chunks,
    regenerated): the new frames in time order, and VACE's decoded version
    of the n_ctx real frames at that end, for the crossfade."""
    import numpy as np

    from soto import video

    # Context for every iteration after the first comes from generated frames,
    # kept at working resolution so nothing goes through a lossy re-encode
    # between iterations.
    if at_start:
        history = video.extract_head_context(clip, info, n_ctx, width, height)
    else:
        history = video.extract_tail_context(clip, info, n_ctx, width, height)
    empty = np.zeros((0, height, width, 3), dtype=np.float32)
    label = "start" if at_start else "end"

    chunks = []
    for i in range(iterations):
        print(f"[{i + 1}/{iterations} {label}] Running Wan2.1-VACE ({model}) for {frames} frames...")
        # Vary the seed per iteration so successive extensions don't reuse the
        # same noise and settle into repeating motion.
        iter_seed = (seed + i) % 2**32
        run = log.run(side=label, iteration=i + 1, seed=iter_seed, steps=steps, context_frames=n_ctx,
                      gap_frames=frames, context_source="clip" if i == 0 else "generated + clip")
        if at_start:
            context = history[:n_ctx]
            with run:
                result = _run_vace(empty, context, frames, width, height, model, prompt, steps, iter_seed)
            chunk = result.frames[:frames].cpu().numpy()
            if i == 0:
                # Later iterations only ever join generated frames to generated
                # frames, so the first seam is the only one with real footage.
                regenerated = result.frames[frames:].cpu().numpy()
            history = np.concatenate([chunk, context], axis=0)
        else:
            context = history[-n_ctx:]
            with run:
                result = _run_vace(context, empty, frames, width, height, model, prompt, steps, iter_seed)
            chunk = result.frames[n_ctx:n_ctx + frames].cpu().numpy()
            if i == 0:
                regenerated = result.frames[:n_ctx].cpu().numpy()
            history = np.concatenate([context, chunk], axis=0)
        chunks.append(chunk)
    # Extending backwards produces each chunk earlier in time than the last.
    return (chunks[::-1] if at_start else chunks), regenerated


@app.command()
def extend(
    clip: Path = typer.Argument(..., exists=True, help="Clip to extend before its first frame and/or past its last."),
    output: Path = typer.Option(..., "-o", "--output", help="Output video path."),
    frames: int = typer.Option(..., "--frames", min=1, help="Frames to generate per iteration, per side."),
    iterations: int = typer.Option(
        1, "--iterations", min=1,
        help="How many times to extend each side; each iteration continues from the previous one's result.",
    ),
    side: str = typer.Option(
        "end", "--side",
        help=f"Which end to extend: {list(_EXTEND_SIDES)}. 'bookend' does both.",
    ),
    context: float = typer.Option(
        config.DEFAULT_CONTEXT_S, "--context", help="Seconds of context taken from each extended end of the clip."
    ),
    model: str = typer.Option(
        config.DEFAULT_MODEL, "--model",
        help=f"Which checkpoint to use: {list(config.MODELS)}.",
    ),
    prompt: str = typer.Option(config.DEFAULT_PROMPT, "--prompt", help="Positive prompt guiding the generated frames."),
    steps: int = typer.Option(None, "--steps", help=_STEPS_HELP),
    preview: str = typer.Option(None, "--preview", help=_PREVIEW_HELP),
    seed: int = typer.Option(None, "--seed", help="Random seed (default: random each run)."),
    crossfade: int = typer.Option(0, "--crossfade", min=0, help=_CROSSFADE_HELP),
    export_generated: Path = typer.Option(None, "--export-generated", dir_okay=False, help=_EXPORT_HELP),
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_NO_METADATA_HELP),
) -> None:
    """Generate new frames leading into the start of a clip, continuing on
    from its end, or both, optionally several times over, and add them."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    if side not in _EXTEND_SIDES:
        raise typer.BadParameter(f"must be one of {list(_EXTEND_SIDES)}", param_hint="--side")
    log = RunLog("extend", enabled=not no_metadata)
    with log.guard(output):
        area, steps = _resolve_quality(preview, steps, log)
        do_start, do_end = side in ("start", "bookend"), side in ("end", "bookend")

        import numpy as np

        from soto import video

        print(f"Probing {clip}...")
        info = video.probe(clip)
        conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
        clip, info = _conform_clip(clip, info, conform_dir, log, "clip")
        fps = info.fps
        width, height = _fit_working_resolution(info.width, info.height, area)

        # Later iterations take context from a history that's only ever longer
        # than the original clip, so the count planned here works for all of them.
        want_ctx = max(1, round(context * fps))
        n_ctx = _plan_extend_context(frames, info.n_frames, want_ctx)
        if n_ctx is None:
            raise typer.BadParameter(f"{clip} is too short to take context from", param_hint="clip")
        if n_ctx != want_ctx:
            log.adjust("context_frames", want_ctx, n_ctx, _context_reason(want_ctx, info.n_frames, n_ctx))
        if crossfade > n_ctx:
            raise typer.BadParameter(f"can't exceed the {n_ctx} context frames", param_hint="--crossfade")
        if do_start and do_end and 2 * crossfade > info.n_frames:
            raise typer.BadParameter(
                f"the fades at both ends can't overlap: at most {info.n_frames // 2} for this clip",
                param_hint="--crossfade",
            )
        total = frames * iterations
        print(f"  working resolution {width}x{height}{_preview_note(preview)}, {n_ctx} context frames, "
              f"{iterations} x {frames} new frames ({total / fps:.2f}s) at the {side}")

        _log_vace_settings(log, model=model, prompt=prompt, steps=steps, preview=preview, width=width, height=height,
                           context=context, crossfade=crossfade, export_generated=export_generated)
        log.settings.update({"side": side, "frames": frames, "iterations": iterations, "context_frames": n_ctx})
        resolved_seed = _resolve_seed(seed, log)
        log.settings["seed_note"] = "iteration i (from 0) on each side uses seed + i"
        print(f"seed={resolved_seed}, steps={steps}")
        # Both sides use the same seeds, so `--side start` with a bookend run's
        # seed reproduces its start.
        args = (clip, info, n_ctx, frames, iterations, width, height, model, prompt, steps, resolved_seed, log)
        head, head_regen = _extend_side(*args, at_start=True) if do_start else ([], None)
        tail, tail_regen = _extend_side(*args, at_start=False) if do_end else ([], None)

        print("Splicing final output...")
        tmp_dir = Path(tempfile.mkdtemp(prefix="soto-extend-"))
        output.parent.mkdir(parents=True, exist_ok=True)
        empty = np.zeros((0, height, width, 3), dtype=np.float32)
        video.extend_frames(
            clip=clip,
            info=info,
            head=np.concatenate(head, axis=0) if head else None,
            head_fade=head_regen[:crossfade] if do_start else empty,
            tail=np.concatenate(tail, axis=0) if tail else None,
            tail_fade=tail_regen[n_ctx - crossfade:] if do_end else empty,
            output_path=output,
            tmp_dir=tmp_dir,
        )
        if export_generated is not None:
            # On a clip shorter than both contexts together, the regenerated
            # start and end overlap; the end's frames win there.
            head_ctx = min(n_ctx, info.n_frames - n_ctx) if do_start and do_end else n_ctx
            real_start = head_ctx if do_start else 0
            real_end = info.n_frames - n_ctx if do_end else info.n_frames
            print(f"Exporting generated frames to {export_generated}...")
            video.export_generated(
                [
                    *head,
                    *([head_regen[:head_ctx]] if do_start else []),
                    (clip, real_start, real_end),
                    *([tail_regen] if do_end else []),
                    *tail,
                ],
                info,
                export_generated,
                audio_source=clip,
                audio_offset_s=total / fps if do_start else 0.0,
            )
        # Output timeline: [new head] + clip + [new tail]. Each iteration's
        # context is the n_ctx frames next to its new frames, real or generated.
        clip_start = total if do_start else 0
        clip_end = clip_start + info.n_frames
        x_head, x_tail = (crossfade if do_start else 0), (crossfade if do_end else 0)
        for i in range(iterations if do_start else 0):
            g = clip_start - (i + 1) * frames
            log.segment("generated", g, g + frames, f"start, iteration {i + 1} (seed {(resolved_seed + i) % 2**32})",
                        side="start", iteration=i + 1)
            log.segment("context", g + frames, g + frames + n_ctx, f"context for start iteration {i + 1}",
                        side="start", iteration=i + 1)
        log.segment("crossfade", clip_start, clip_start + x_head, "crossfade into clip", side="start")
        log.segment("real", clip_start + x_head, clip_end - x_tail, "clip",
                    source="clip", source_frames=[x_head, info.n_frames - x_tail])
        log.segment("crossfade", clip_end - x_tail, clip_end, "crossfade out of clip", side="end")
        for i in range(iterations if do_end else 0):
            g = clip_end + i * frames
            log.segment("generated", g, g + frames, f"end, iteration {i + 1} (seed {(resolved_seed + i) % 2**32})",
                        side="end", iteration=i + 1)
            log.segment("context", g - n_ctx, g, f"context for end iteration {i + 1}", side="end", iteration=i + 1)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, resolved_seed, preview, log, fps)


def _resolve_checkpoint(checkpoint: str) -> Path:
    """A path to a .pth file, or the name of one in models/pix2pixhd/."""
    path = Path(checkpoint)
    if path.is_file():
        return path
    named = config.PIX2PIXHD_DIR / f"{checkpoint}.pth"
    if named.is_file():
        return named
    available = sorted(p.stem for p in config.PIX2PIXHD_DIR.glob("*.pth"))
    raise typer.BadParameter(
        f"no such file, and not one of the checkpoints in models/pix2pixhd/: {available}",
        param_hint="--checkpoint",
    )


def _with_progress(frames, total: int):
    for i, frame in enumerate(frames, 1):
        print(f"\r  frame {i}/{total}", end="", flush=True)
        yield frame
    print()


@app.command()
def dream(
    clip: Path = typer.Argument(..., exists=True, help="Clip to continue past its last frame."),
    output: Path = typer.Option(..., "-o", "--output", help="Output video path."),
    checkpoint: str = typer.Option(
        ..., "--checkpoint",
        help="pix2pixHD generator: the name of one in models/pix2pixhd/ (e.g. 'nasa') or a path to a .pth file.",
    ),
    frames: int = typer.Option(..., "--frames", min=1, help="Frames to dream."),
    zoom: float = typer.Option(
        0.0, "--zoom", min=0.0,
        help="Pixels cropped off each edge of every dreamed frame (and scaled back up) before it's fed back in: "
        "a steady push in.",
    ),
    heat_seeking: float = typer.Option(
        0.0, "--heat-seeking", min=0.0,
        help="Pixels per frame the camera drifts toward the brightest quadrant (smoothed over "
        "30 frames), plus as much zoom again to keep the edges covered.",
    ),
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_NO_METADATA_HELP),
) -> None:
    """Continue a clip past its last frame with a pix2pixHD next-frame
    generator fed its own output, frame by frame, and append the result.
    Seconds per second of footage rather than `extend`'s hours, but it only
    knows the one video its checkpoint was trained on."""
    import numpy as np
    import torch

    from soto import pix2pixhd, video

    log = RunLog("dream", enabled=not no_metadata)
    with log.guard(output):
        weights = _resolve_checkpoint(checkpoint)
        print(f"Probing {clip}...")
        info = video.probe(clip)
        conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
        clip, info = _conform_clip(clip, info, conform_dir, log, "clip")

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        net, stride, architecture = pix2pixhd.load_generator(weights, device)
        # The generator is fully convolutional, so it runs at the clip's own size
        # (as it was trained on its video's), just rounded to its stride.
        width, height = (max(stride, round(d / stride) * stride) for d in (info.width, info.height))
        if (width, height) != (info.width, info.height):
            log.adjust("working_resolution", [info.width, info.height], [width, height],
                       f"rounded to the generator's stride of {stride}")
        log.settings.update({
            "checkpoint": str(weights.resolve()),
            "architecture": architecture,
            "device": str(device),
            "frames": frames,
            "zoom": zoom,
            "heat_seeking": heat_seeking,
            "working_resolution": [width, height],
            "seed": None,
            "seed_note": "no randomness: the same clip, checkpoint and settings dream the same frames",
        })
        print(f"  working resolution {width}x{height}, {frames} frames ({frames / info.fps:.2f}s) on {device}")

        start = video.extract_tail_context(clip, info, 1, width, height)[0]
        print(f"Dreaming with {weights.name} and splicing...")
        tmp_dir = Path(tempfile.mkdtemp(prefix="soto-dream-"))
        output.parent.mkdir(parents=True, exist_ok=True)
        # Frames are dreamed lazily as they're encoded, so this times both.
        with log.run(frames=frames, context_frames=1,
                     context_source={"clip": "clip", "frames": [info.n_frames - 1, info.n_frames]}):
            video.extend_frames(
                clip=clip,
                info=info,
                head=None,
                head_fade=np.zeros((0, height, width, 3), dtype=np.float32),
                tail=_with_progress(pix2pixhd.dream(net, start, frames, zoom=zoom, heat_seeking=heat_seeking), frames),
                tail_fade=np.zeros((0, height, width, 3), dtype=np.float32),
                output_path=output,
                tmp_dir=tmp_dir,
            )
        log.segment("real", 0, info.n_frames, "clip", source="clip", source_frames=[0, info.n_frames])
        log.segment("context", info.n_frames - 1, info.n_frames, "starting frame", run=1)
        log.segment("generated", info.n_frames, info.n_frames + frames, f"dreamed ({weights.stem})", run=1)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(conform_dir, ignore_errors=True)
        log.write(output, info.fps)
        print(f"Done: {output}")


def _parse_size(value: str) -> tuple[int, int]:
    from soto import video

    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise typer.BadParameter(f"expected WIDTHxHEIGHT, e.g. 1920x1080, not {value!r}", param_hint="--size")
    if w < 16 or h < 16:
        raise typer.BadParameter("must be at least 16x16", param_hint="--size")
    return video.even_size(w, h)


def _expand_rect(src_w: int, src_h: int, out_w: int, out_h: int, fit: str) -> tuple[int, int, int, int]:
    """Where the source sits in the output, centred: as large as fits for
    'contain', at its own size for 'native'."""
    from soto import video

    if fit == "contain":
        scale = min(out_w / src_w, out_h / src_h)
        rw, rh = video.even_size(round(src_w * scale), round(src_h * scale))
    else:
        if src_w > out_w or src_h > out_h:
            raise typer.BadParameter(
                f"the clip ({src_w}x{src_h}) is larger than {out_w}x{out_h}; use --fit contain to scale it down",
                param_hint="--size",
            )
        rw, rh = src_w, src_h
    x0, y0 = (out_w - rw) // 2, (out_h - rh) // 2
    return x0, y0, x0 + rw, y0 + rh


def _scale_rect(rect: tuple[int, int, int, int], sx: float, sy: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return (max(0, round(x0 * sx)), max(0, round(y0 * sy)), min(w, round(x1 * sx)), min(h, round(y1 * sy)))


def _expand_windows(clip: Path, info, width: int, height: int, rect, window: int, overlap: int,
                    model: str, prompt: str, steps: int, seed: int, log: RunLog) -> list:
    """Outpaints the whole clip at working resolution, `window` frames per
    VACE run, each run after the first given the previous one's last
    `overlap` canvases. Returns one uint8 canvas per frame of the clip."""
    import numpy as np

    from soto import comfy_engine, video

    x0, y0, x1, y1 = rect
    rw, rh = x1 - x0, y1 - y0
    # The rect is usually a downscale of the source, but not for low-res clips.
    flags = "area" if rw <= info.width else "lanczos"
    n_windows = 1 + max(0, -(-(info.n_frames - window) // (window - overlap)))
    canvases: list = []
    start = 0
    for i in range(n_windows):
        given = (np.stack(canvases[-overlap:]).astype(np.float32) / 255.0 if canvases and overlap
                 else np.zeros((0, height, width, 3), dtype=np.float32))
        n_new = min(window - given.shape[0], info.n_frames - start)
        frames = video.read_frames(clip, start, n_new, rw, rh, flags=flags)
        # Round the run up to VACE's length stride with copies of the last
        # frame, and drop what they generate.
        length = given.shape[0] + n_new
        pad = (-(length - 1)) % config.VACE_LENGTH_STRIDE
        if pad:
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)
        control_video, control_mask = video.build_outpaint_batch(
            given, frames, width, height, rect, config.EXPAND_SEAM_PX
        )
        run_seed = (seed + i) % 2**32
        print(f"[{i + 1}/{n_windows}] Running Wan2.1-VACE ({model}) on frames {start}-{start + n_new - 1}...")
        with log.run(window=i + 1, seed=run_seed, steps=steps, frames=[start, start + n_new],
                     given_frames=given.shape[0], padded_frames=pad, length=length + pad):
            result = comfy_engine.run_vace_transition(
                control_video=control_video,
                control_mask=control_mask,
                width=width,
                height=height,
                length=length + pad,
                model=model,
                prompt=prompt,
                negative_prompt=config.DEFAULT_NEGATIVE_PROMPT,
                steps=steps,
                seed=run_seed,
            )
        new = result.frames[given.shape[0]:length].cpu().numpy()
        # Kept as uint8: a long clip's worth of float canvases adds up.
        canvases.extend(np.clip(new * 255.0 + 0.5, 0, 255).astype(np.uint8))
        start += n_new
    return canvases


@app.command()
def expand(
    clip: Path = typer.Argument(..., exists=True, help="Clip to expand."),
    output: Path = typer.Option(..., "-o", "--output", help="Output video path."),
    size: str = typer.Option(..., "--size", help="Output resolution, WIDTHxHEIGHT, e.g. 1920x1080."),
    fill: str = typer.Option(
        "vace", "--fill",
        help=f"How to fill the new edges: {list(config.EXPAND_FILLS)}. 'blur' is a blurred copy of the clip "
        "behind it: no model, seconds to render.",
    ),
    fit: str = typer.Option(
        "contain", "--fit",
        help=f"How the clip sits in the output, centred: {list(config.EXPAND_FITS)}. 'contain' scales it as "
        "large as fits; 'native' keeps its own size.",
    ),
    window: int = typer.Option(config.EXPAND_WINDOW, "--window", min=2, help="Frames per VACE run."),
    overlap: int = typer.Option(
        config.EXPAND_OVERLAP, "--overlap", min=0,
        help="Generated frames from the end of each run given to the next, so the edges carry on across runs.",
    ),
    model: str = typer.Option(
        config.DEFAULT_MODEL, "--model",
        help=f"Which checkpoint to use: {list(config.MODELS)}.",
    ),
    prompt: str = typer.Option(
        config.DEFAULT_EXPAND_PROMPT, "--prompt", help="Positive prompt guiding the generated edges."
    ),
    steps: int = typer.Option(None, "--steps", help=_STEPS_HELP),
    preview: str = typer.Option(None, "--preview", help=_PREVIEW_HELP),
    seed: int = typer.Option(None, "--seed", help="Random seed (default: random each run)."),
    export_generated: Path = typer.Option(
        None, "--export-generated", dir_okay=False,
        help="Also write the model's raw canvases, scaled to the output size, before the real footage is laid "
        "back over the middle -- for blending by hand in an editor.",
    ),
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_NO_METADATA_HELP),
) -> None:
    """Grow a clip's frame to a larger resolution, generating new picture
    around its edges from what's in the frame, with the original centred."""
    if fill not in config.EXPAND_FILLS:
        raise typer.BadParameter(f"must be one of {list(config.EXPAND_FILLS)}", param_hint="--fill")
    if fit not in config.EXPAND_FITS:
        raise typer.BadParameter(f"must be one of {list(config.EXPAND_FITS)}", param_hint="--fit")
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    if overlap >= window:
        raise typer.BadParameter("must be less than --window", param_hint="--overlap")
    out_w, out_h = _parse_size(size)

    import numpy as np

    from soto import video

    log = RunLog("expand", enabled=not no_metadata)
    with log.guard(output):
        print(f"Probing {clip}...")
        info = video.probe(clip)
        conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
        clip, info = _conform_clip(clip, info, conform_dir, log, "clip")
        fps = info.fps
        rect = _expand_rect(info.width, info.height, out_w, out_h, fit)
        if rect == (0, 0, out_w, out_h):
            raise typer.BadParameter(
                f"the clip already fills {out_w}x{out_h} with --fit {fit}: nothing to expand", param_hint="--size"
            )
        log.settings.update({
            "fill": fill,
            "fit": fit,
            "output_resolution": [out_w, out_h],
            "source_rect": list(rect),
        })
        print(f"  {info.width}x{info.height} -> {out_w}x{out_h}, clip at {rect[2] - rect[0]}x{rect[3] - rect[1]} "
              f"from ({rect[0]}, {rect[1]})")

        if fill == "blur":
            ignored = [flag for flag, value in (("--steps", steps), ("--preview", preview), ("--seed", seed),
                                                ("--export-generated", export_generated)) if value is not None]
            if ignored:
                print(f"  note: {', '.join(ignored)} only apply to --fill vace; ignored")
            log.settings["seed"] = None
            print("Rendering blur fill...")
            with log.run(fill="blur"):
                video.blur_fill(clip, output, out_w, out_h, rect)
            log.segment("real", 0, info.n_frames, "clip, expanded (blur)", source="clip",
                        source_frames=[0, info.n_frames])
            shutil.rmtree(conform_dir, ignore_errors=True)
            log.write(output, fps)
            print(f"Done: {output}")
            return

        area, steps = _resolve_quality(preview, steps, log)
        width, height = _fit_working_resolution(out_w, out_h, area)
        work_rect = _scale_rect(rect, width / out_w, height / out_h, width, height)
        _log_vace_settings(log, model=model, prompt=prompt, steps=steps, preview=preview, width=width,
                           height=height, context=0.0, crossfade=0, export_generated=export_generated)
        del log.settings["context_s"], log.settings["crossfade_frames"]
        log.settings.update({
            "source_rect_working": list(work_rect),
            "window": window,
            "overlap": overlap,
            "seam_px": config.EXPAND_SEAM_PX,
        })
        resolved_seed = _resolve_seed(seed, log)
        log.settings["seed_note"] = "window i (from 0) uses seed + i"
        print(f"  working resolution {width}x{height}{_preview_note(preview)}, "
              f"windows of {window} frames overlapping by {overlap}")
        print(f"seed={resolved_seed}, steps={steps}")

        canvases = _expand_windows(clip, info, width, height, work_rect, window, overlap, model, prompt, steps,
                                   resolved_seed, log)

        print("Compositing final output...")
        seam = config.EXPAND_SEAM_PX * out_w / width
        video.write_outpaint(video.composite_outpaint(canvases, clip, rect, out_w, out_h, seam),
                             fps, out_w, out_h, clip, output)
        if export_generated is not None:
            print(f"Exporting generated frames to {export_generated}...")
            video.write_outpaint((c.astype(np.float32) / 255.0 for c in canvases),
                                 fps, out_w, out_h, clip, export_generated)
        log.segment("real", 0, info.n_frames, "clip, expanded", source="clip", source_frames=[0, info.n_frames])
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, resolved_seed, preview, log, fps)


if __name__ == "__main__":
    app()
