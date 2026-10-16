from __future__ import annotations

import random
import shutil
import tempfile
from pathlib import Path

import typer

from soto import config
from soto.metadata import RunLog

app = typer.Typer(add_completion=False, no_args_is_help=True)

_OUTPUT_HELP = (
    "Output path, a .mkv: every frame the model generated, including its own version of the real "
    "context frames, losslessly at working resolution. Splice it into the footage with `soto crossfade`."
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
_GENERATED_NO_METADATA_HELP = _NO_METADATA_HELP + " Without it, `soto crossfade` can't splice the output."


def _check_lossless_output(output: Path) -> None:
    if output.suffix.lower() != ".mkv":
        raise typer.BadParameter(
            f"must be a .mkv, not {output.name!r}: the frames are kept losslessly as FFV1, which mp4 can't hold",
            param_hint="--output",
        )


class _Pieces:
    """Collects a generation command's output a piece at a time -- each one a
    run of generated frames with the model's own version of the context on
    either side -- written losslessly to a temp dir as it comes, so a long
    bridge never holds every run in memory, then joined into the output."""

    def __init__(self, fps: float):
        self.fps = fps
        self.dir = Path(tempfile.mkdtemp(prefix="soto-pieces-"))
        self.files: list[Path] = []
        self.frames = 0

    def add(self, frames) -> int:
        """Appends `frames` (an array or a lazy iterable); returns the frame
        of the output it starts at."""
        from soto import video

        start = self.frames
        file = self.dir / f"piece_{len(self.files):03d}.mkv"
        self.frames += video.write_lossless(frames, self.fps, file)
        self.files.append(file)
        return start

    def write(self, output: Path) -> None:
        from soto import video

        output.parent.mkdir(parents=True, exist_ok=True)
        video.concat_lossless(self.files, output)
        shutil.rmtree(self.dir, ignore_errors=True)


def _context_segment(log: RunLog, start: int, n: int, label: str, source: str, source_start: int, **details) -> None:
    """Frames [start, start + n) of the output are the model's version of
    real frames [source_start, source_start + n) of input `source`."""
    log.segment("context", start, start + n, label, source=source,
                source_frames=[source_start, source_start + n], **details)


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
                       width: int, height: int, context: float | None = None) -> None:
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
        "vae_device": config.VAE_DEVICE,
        "vae_dtype": config.VAE_DTYPE if config.VAE_DEVICE == "gpu" else "fp32",
    })
    if context is not None:
        log.settings["context_s"] = context


def _context_reason(want: int, available: int, used: int) -> str:
    reasons = []
    if available < want:
        reasons.append(f"only {available} real frames available")
    if used != min(want, available):
        reasons.append(f"nudged so the VACE length fits its stride of {config.VACE_LENGTH_STRIDE}")
    return "; ".join(reasons)


def _preview_note(preview: str | None) -> str:
    return f" (preview: {preview})" if preview else ""


def _print_done(output: Path, seed: int | None, preview: str | None, log: RunLog, fps: float, *,
                generated: bool = True) -> None:
    log.write(output, fps)
    print(f"Done: {output}" + (f" (seed={seed})" if seed is not None else ""))
    if preview:
        print(f"Preview only -- re-run without --preview and with --seed {seed} for the full-quality version.")
    if generated and log.enabled:
        print(f"Splice it into the footage with: soto crossfade {output} -o {output.with_suffix('.mp4')} --frames 8")


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
    log.input(role, clip, info, changes, conform_to={"fps": fps, "width": width, "height": height})
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
    output: Path = typer.Option(..., "-o", "--output", help=_OUTPUT_HELP),
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
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_GENERATED_NO_METADATA_HELP),
) -> None:
    """Generate an AI-blended transition between two clips, to splice in
    with `soto crossfade`."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    _check_lossless_output(output)
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
                           context=context)
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

        resolved_seed = _resolve_seed(seed, log)
        print(f"Running Wan2.1-VACE ({model}) transition (seed={resolved_seed}, steps={steps})...")
        with log.run(seed=resolved_seed, steps=steps, context_frames_before=n_a, gap_frames=gap_frames,
                     context_frames_after=n_b,
                     context_source_before={"clip": "a", "frames": [info_a.n_frames - n_a, info_a.n_frames]},
                     context_source_after={"clip": "b", "frames": [0, n_b]}):
            result = _run_vace(context_a, context_b, gap_frames, width, height, model, prompt, steps, resolved_seed)

        print(f"Writing {output}...")
        pieces = _Pieces(fps)
        pieces.add(result.frames.cpu().numpy())
        pieces.write(output)
        # The output is the whole run: context from a, the gap, context from b.
        gap_end = n_a + gap_frames
        _context_segment(log, 0, n_a, "context from clip a", "a", info_a.n_frames - n_a, run=1)
        log.segment("generated", n_a, gap_end, f"generated (seed {resolved_seed})", run=1)
        _context_segment(log, gap_end, n_b, "context from clip b", "b", 0, run=1)
        # Spliced: all of clip A, the gap, then all of clip B.
        log.append("timeline", {"kind": "real", "source": "a", "source_frames": [0, info_a.n_frames]})
        log.append("timeline", {"kind": "generated", "file_frames": [n_a, gap_end],
                                "label": f"generated (seed {resolved_seed})"})
        log.append("timeline", {"kind": "real", "source": "b", "source_frames": [0, info_b.n_frames]})
        log.data["audio"] = None
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, resolved_seed, preview, log, fps)


@app.command()
def bridge(
    clip: Path = typer.Argument(..., exists=True, help="Clip containing blank (flat-colour) intervals."),
    output: Path = typer.Option(..., "-o", "--output", help=_OUTPUT_HELP),
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
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_GENERATED_NO_METADATA_HELP),
) -> None:
    """Find blank intervals inside a clip and generate a bridge for each, to
    splice in with `soto crossfade`."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    _check_lossless_output(output)
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
                           context=context)
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

        resolved_seed = _resolve_seed(seed, log)
        print(f"Working resolution {width}x{height}{_preview_note(preview)}, seed={resolved_seed}, steps={steps}")
        pieces = _Pieces(fps)
        cursor = 0
        for k, (start, end, n_a, n_b) in enumerate(plans, 1):
            print(f"[{k}/{len(plans)}] Running Wan2.1-VACE ({model}) for frames {start}-{end - 1}...")
            context_a = video.read_frames(clip, start - n_a, n_a, width, height)
            context_b = video.read_frames(clip, end, n_b, width, height)
            with log.run(seed=resolved_seed, steps=steps, context_frames_before=n_a, gap_frames=end - start,
                         context_frames_after=n_b, gap=[start, end]):
                result = _run_vace(
                    context_a, context_b, end - start, width, height, model, prompt, steps, resolved_seed
                )
            # Each run goes into the output whole: its context, the bridge,
            # its context.
            offset = pieces.add(result.frames.cpu().numpy())
            gap_start, gap_end = offset + n_a, offset + n_a + (end - start)
            _context_segment(log, offset, n_a, f"context before bridge {k}", "clip", start - n_a, run=k)
            log.segment("generated", gap_start, gap_end, f"bridge {k} (seed {resolved_seed})", run=k)
            _context_segment(log, gap_end, n_b, f"context after bridge {k}", "clip", end, run=k)
            # Spliced, bridging keeps every frame where it was.
            log.append("timeline", {"kind": "real", "source": "clip", "source_frames": [cursor, start]})
            log.append("timeline", {"kind": "generated", "file_frames": [gap_start, gap_end],
                                    "label": f"bridge {k} (seed {resolved_seed})"})
            cursor = end
        log.append("timeline", {"kind": "real", "source": "clip", "source_frames": [cursor, info.n_frames]})
        log.data["audio"] = {"source": "clip", "offset_frames": 0}

        print(f"Writing {output}...")
        pieces.write(output)
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, resolved_seed, preview, log, fps)


_EXTEND_SIDES = ("start", "end", "bookend")


def _extend_side(clip: Path, info, n_ctx: int, frames: int, iterations: int, width: int, height: int,
                 model: str, prompt: str, steps: int, seed: int, log: RunLog, *, at_start: bool):
    """Runs `extend`'s iterations off one end of the clip. Returns (chunks,
    regenerated): the new frames in time order, and VACE's decoded version
    of the n_ctx real frames at that end, for `soto crossfade` to blend into."""
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
    output: Path = typer.Option(..., "-o", "--output", help=_OUTPUT_HELP),
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
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_GENERATED_NO_METADATA_HELP),
) -> None:
    """Generate new frames leading into the start of a clip, continuing on
    from its end, or both, optionally several times over, to add with
    `soto crossfade`."""
    if model not in config.MODELS:
        raise typer.BadParameter(f"--model must be one of {list(config.MODELS)}")
    if side not in _EXTEND_SIDES:
        raise typer.BadParameter(f"must be one of {list(_EXTEND_SIDES)}", param_hint="--side")
    _check_lossless_output(output)
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
        total = frames * iterations
        print(f"  working resolution {width}x{height}{_preview_note(preview)}, {n_ctx} context frames, "
              f"{iterations} x {frames} new frames ({total / fps:.2f}s) at the {side}")

        _log_vace_settings(log, model=model, prompt=prompt, steps=steps, preview=preview, width=width, height=height,
                           context=context)
        log.settings.update({"side": side, "frames": frames, "iterations": iterations, "context_frames": n_ctx})
        resolved_seed = _resolve_seed(seed, log)
        log.settings["seed_note"] = "iteration i (from 0) on each side uses seed + i"
        print(f"seed={resolved_seed}, steps={steps}")
        # Both sides use the same seeds, so `--side start` with a bookend run's
        # seed reproduces its start.
        args = (clip, info, n_ctx, frames, iterations, width, height, model, prompt, steps, resolved_seed, log)
        head, head_regen = _extend_side(*args, at_start=True) if do_start else ([], None)
        tail, tail_regen = _extend_side(*args, at_start=False) if do_end else ([], None)

        # The output is a piece per side, each the new frames with the model's
        # version of the real frames they join on to:
        # [start iterations, earliest first][context] [context][end iterations].
        print(f"Writing {output}...")
        pieces = _Pieces(fps)
        for i in range(iterations if do_start else 0):
            g = (iterations - 1 - i) * frames
            log.segment("generated", g, g + frames, f"start, iteration {i + 1} (seed {(resolved_seed + i) % 2**32})",
                        side="start", iteration=i + 1)
        if do_start:
            pieces.add(np.concatenate([*head, head_regen], axis=0))
            _context_segment(log, total, n_ctx, "context from the clip's start", "clip", 0, side="start")
        if do_end:
            offset = pieces.add(np.concatenate([tail_regen, *tail], axis=0))
            _context_segment(log, offset, n_ctx, "context from the clip's end", "clip", info.n_frames - n_ctx,
                             side="end")
            for i in range(iterations):
                g = offset + n_ctx + i * frames
                log.segment("generated", g, g + frames, f"end, iteration {i + 1} (seed {(resolved_seed + i) % 2**32})",
                            side="end", iteration=i + 1)
        pieces.write(output)

        # Spliced: [new head] + clip + [new tail], the audio delayed by the head.
        if do_start:
            log.append("timeline", {"kind": "generated", "file_frames": [0, total], "label": "extended start"})
        log.append("timeline", {"kind": "real", "source": "clip", "source_frames": [0, info.n_frames]})
        if do_end:
            log.append("timeline", {"kind": "generated", "file_frames": [offset + n_ctx, offset + n_ctx + total],
                                    "label": "extended end"})
        log.data["audio"] = {"source": "clip", "offset_frames": total if do_start else 0}
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
    output: Path = typer.Option(
        ..., "-o", "--output",
        help="Output path, a .mkv: the dreamed frames, preceded by the generator's own version of the "
        "context frames, losslessly. Splice it onto the clip with `soto crossfade`.",
    ),
    checkpoint: str = typer.Option(
        ..., "--checkpoint",
        help="pix2pixHD generator: the name of one in models/pix2pixhd/ (e.g. 'nasa') or a path to a .pth file.",
    ),
    frames: int = typer.Option(..., "--frames", min=1, help="Frames to dream."),
    context: float = typer.Option(
        config.DEFAULT_CONTEXT_S, "--context", min=0.0,
        help="Seconds at the end of the clip to also run the generator over, each frame predicted from "
        "the real one before it, for `soto crossfade` to blend into. Can't exceed the clip.",
    ),
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
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_GENERATED_NO_METADATA_HELP),
) -> None:
    """Continue a clip past its last frame with a pix2pixHD next-frame
    generator fed its own output, frame by frame, to add with `soto
    crossfade`. Seconds per second of footage rather than `extend`'s hours,
    but it only knows the one video its checkpoint was trained on."""
    import torch

    from soto import pix2pixhd, video

    _check_lossless_output(output)
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
            "context_s": context,
            "zoom": zoom,
            "heat_seeking": heat_seeking,
            "working_resolution": [width, height],
            "seed": None,
            "seed_note": "no randomness: the same clip, checkpoint and settings dream the same frames",
        })
        # Each context frame is predicted from the real one before it, so the
        # clip's first frame can't be one.
        want_ctx = round(context * info.fps)
        n_ctx = min(want_ctx, info.n_frames - 1)
        if n_ctx != want_ctx:
            log.adjust("context_frames", want_ctx, n_ctx,
                       f"the clip only has {info.n_frames} frames, and the first has none before it")
        log.settings["context_frames"] = n_ctx
        print(f"  working resolution {width}x{height}, {n_ctx} context frames, "
              f"{frames} frames ({frames / info.fps:.2f}s) on {device}")

        real = video.extract_tail_context(clip, info, n_ctx + 1, width, height)
        print(f"Dreaming with {weights.name} into {output}...")
        pieces = _Pieces(info.fps)
        # Frames are dreamed lazily as they're encoded, so this times both.
        with log.run(frames=frames, context_frames=n_ctx,
                     context_source={"clip": "clip", "frames": [info.n_frames - n_ctx - 1, info.n_frames]}):
            dreamed = pix2pixhd.dream(net, real, frames, zoom=zoom, heat_seeking=heat_seeking)
            pieces.add(_with_progress(dreamed, n_ctx + frames))
        pieces.write(output)
        _context_segment(log, 0, n_ctx, "context from the clip's end", "clip", info.n_frames - n_ctx, run=1)
        log.segment("generated", n_ctx, n_ctx + frames, f"dreamed ({weights.stem})", run=1)
        log.append("timeline", {"kind": "real", "source": "clip", "source_frames": [0, info.n_frames]})
        log.append("timeline", {"kind": "generated", "file_frames": [n_ctx, n_ctx + frames],
                                "label": f"dreamed ({weights.stem})"})
        log.data["audio"] = {"source": "clip", "offset_frames": 0}
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, None, None, log, info.fps)


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
                           height=height)
        log.settings.update({
            "export_generated": str(export_generated.resolve()) if export_generated else None,
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
        _print_done(output, resolved_seed, preview, log, fps, generated=False)


_CROSSFADE_HELP = (
    "Frames at each seam to crossfade from the real footage into the model's own version of it, instead of "
    "cutting straight to the generated frames. Each fade comes out of its seam's context frames, so it's "
    "limited to them, and two fades over the same stretch of real footage split it between them."
)


def _load_generated(generated: Path, output: Path) -> dict:
    """Reads and checks the sidecar of a generation command's output."""
    import json

    sidecar = generated.with_suffix(".json")
    if not sidecar.is_file():
        raise typer.BadParameter(f"no sidecar at {sidecar}: was it generated with --no-metadata?",
                                 param_hint="generated")
    data = json.loads(sidecar.read_text())
    if data.get("status") != "ok" or "timeline" not in data:
        raise typer.BadParameter(
            f"{sidecar} isn't from a finished transition, bridge, extend or dream run (or is from before "
            "they wrote a timeline)", param_hint="generated",
        )
    if output.resolve() == generated.resolve() or output.with_suffix(".json").resolve() == sidecar.resolve():
        raise typer.BadParameter(f"would overwrite {generated.name} or its sidecar: pick another name",
                                 param_hint="--output")
    return data


def _plan_fades(timeline: list[dict], segments: list[dict], want: int, log: RunLog) -> list[list[dict | None]]:
    """For each timeline entry, the fade at its start and at its end (None
    for none; generated entries never have one). Each fade is {frames,
    file_edge, label}: `frames` of the real footage at that edge blend into
    the model's version of them, the generated output's frames just before
    (or from) `file_edge`."""
    context = [seg for seg in segments if seg["kind"] == "context"]
    ending = {seg["end_frame"]: seg for seg in context}
    starting = {seg["start_frame"]: seg for seg in context}
    fades: list[list[dict | None]] = [[None, None] for _ in timeline]
    limits = {}
    for i, entry in enumerate(timeline):
        if entry["kind"] != "generated":
            continue
        g0, g1 = entry["file_frames"]
        # The context before a generated stretch is the end of the real
        # footage before it (side 1), the context after it the start of the
        # real footage after it (side 0).
        for j, ctx, side, file_edge in ((i - 1, ending.get(g0), 1, g0), (i + 1, starting.get(g1), 0, g1)):
            if ctx is None or not 0 <= j < len(timeline) or timeline[j]["kind"] != "real":
                continue
            real = timeline[j]
            if ctx["source"] != real["source"] or ctx["source_frames"][side] != real["source_frames"][side]:
                raise RuntimeError(f"the sidecar's {ctx['label']!r} doesn't line up with the footage next to it")
            n_ctx = ctx["end_frame"] - ctx["start_frame"]
            fades[j][side] = {"frames": min(want, n_ctx), "file_edge": file_edge, "label": entry["label"]}
            limits[id(fades[j][side])] = n_ctx

    # A stretch of real footage between two generated ones can't fade at
    # both ends over the same frames, so split it between them.
    for entry, (start, end) in zip(timeline, fades):
        if entry["kind"] != "real":
            continue
        length = entry["source_frames"][1] - entry["source_frames"][0]
        x_start, x_end = (start or {}).get("frames", 0), (end or {}).get("frames", 0)
        if x_start + x_end > length:
            x_start = min(x_start, length // 2)
            x_end = min(x_end, length - x_start)
        for fade, used, verb in ((start, x_start, "out of"), (end, x_end, "into")):
            if fade is None:
                continue
            fade["frames"] = used
            if used < want:
                n_ctx = limits[id(fade)]
                reason = (f"limited to the {n_ctx} context frames" if used == n_ctx
                          else "split with the fade at the other end of the same real footage")
                print(f"  crossfade {verb} {fade['label']}: {used} frames ({reason})")
                log.adjust(f"crossfade {verb} {fade['label']}", want, used, reason)
    return fades


@app.command()
def crossfade(
    generated: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="A .mkv written by transition, bridge, extend or dream."
    ),
    output: Path = typer.Option(
        ..., "-o", "--output",
        help="Output video path. A new file: not the input, and not one whose .json sidecar would replace the "
        "input's (so not the same name with another extension, in the same folder).",
    ),
    frames: int = typer.Option(..., "--frames", min=0, help=_CROSSFADE_HELP),
    no_metadata: bool = typer.Option(False, "--no-metadata", help=_NO_METADATA_HELP),
) -> None:
    """Splice a generation command's frames into the footage they came from,
    crossfading each seam from the real frames into the model's own version
    of them. Takes seconds, so try as many --frames as you like."""
    data = _load_generated(generated, output)
    log = RunLog("crossfade", enabled=not no_metadata)
    with log.guard(output):
        from soto import video

        made = video.probe(generated)
        if made.n_frames != data["output"]["frames"]:
            raise typer.BadParameter(
                f"has {made.n_frames} frames, but its sidecar says {data['output']['frames']}",
                param_hint="generated",
            )
        log.input("generated", generated, made, [])
        settings = data["settings"]
        log.settings.update({
            "frames": frames,
            "generated_from": {
                "path": str(generated.resolve()),
                "command": data["command"],
                "seed": settings.get("seed"),
                "model": settings.get("model"),
                "checkpoint": settings.get("checkpoint"),
            },
        })

        # The footage, conformed just as it was for generating.
        conform_dir = Path(tempfile.mkdtemp(prefix="soto-conform-"))
        sources = {}
        for recorded in data["inputs"]:
            role, path = recorded["role"], Path(recorded["path"])
            if not path.is_file():
                raise typer.BadParameter(f"{path} (input {role!r}) no longer exists", param_hint="generated")
            info = video.probe(path)
            if ((info.width, info.height, info.n_frames) != (recorded["width"], recorded["height"], recorded["frames"])
                    or abs(info.fps - recorded["fps"]) > 1e-6):
                raise typer.BadParameter(f"{path} (input {role!r}) has changed since {generated.name} was generated",
                                         param_hint="generated")
            target = recorded["conform_to"]
            sources[role] = _conform_clip(path, info, conform_dir / role, log, role, fps=target["fps"],
                                          size=(target["width"], target["height"]))
        _, info = next(iter(sources.values()))

        timeline = data["timeline"]
        fades = _plan_fades(timeline, data["segments"], frames, log)

        def fade_frames(fade: dict, before: bool):
            start = fade["file_edge"] - fade["frames"] if before else fade["file_edge"]
            return video.read_frames(generated, start, fade["frames"], made.width, made.height)

        parts = []
        cursor = 0

        def place(kind: str, n: int, label: str, **details) -> None:
            nonlocal cursor
            log.segment(kind, cursor, cursor + n, label, **details)
            cursor += n

        for entry, (start, end) in zip(timeline, fades):
            if entry["kind"] == "generated":
                g0, g1 = entry["file_frames"]
                parts.append(video.Generated(generated, g0, g1, made.width, made.height))
                place("generated", g1 - g0, entry["label"], file_frames=[g0, g1])
                continue
            role = entry["source"]
            path, _ = sources[role]
            a, b = entry["source_frames"]
            x_start, x_end = (start or {}).get("frames", 0), (end or {}).get("frames", 0)
            if x_start:
                parts.append(video.Fade(path, a, fade_frames(start, before=False), fade_in=False))
                place("crossfade", x_start, f"crossfade out of {start['label']}")
            parts.append(video.Real(path, a + x_start, b - x_end))
            place("real", b - x_end - (a + x_start), "clip" if role == "clip" else f"clip {role}",
                  source=role, source_frames=[a + x_start, b - x_end])
            if x_end:
                parts.append(video.Fade(path, b - x_end, fade_frames(end, before=True), fade_in=True))
                place("crossfade", x_end, f"crossfade into {end['label']}")

        audio = data.get("audio")
        print(f"Splicing {generated.name} into the footage with {frames}-frame crossfades...")
        video.assemble(parts, info, output,
                       audio=(sources[audio["source"]][0], audio["offset_frames"] / info.fps) if audio else None)
        shutil.rmtree(conform_dir, ignore_errors=True)
        _print_done(output, None, None, log, info.fps, generated=False)


if __name__ == "__main__":
    app()
