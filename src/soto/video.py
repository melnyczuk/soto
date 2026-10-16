"""ffmpeg orchestration: probing clips, extracting context frames, building
the VACE control-video/mask batch, and splicing the final output.

Deliberately ffmpeg-only (no opencv): opencv-python and `av` (a ComfyUI
dependency) each bundle their own private copy of ffmpeg's libavdevice, and
having both loaded in the same process triggers an Objective-C class
collision on macOS (AVFAudioReceiver/AVFFrameReceiver defined twice) that
prints "may cause spurious casting failures and mysterious crashes" at
import time. ffmpeg is already a hard requirement (see `soto doctor`), so
there's no need for a second video I/O library that risks that collision."""

from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ClipInfo:
    fps: float
    width: int
    height: int
    n_frames: int


def probe(path: Path) -> ClipInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return ClipInfo(
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
        n_frames=int(stream["nb_read_frames"]),
    )


def _read_frames(path: Path, start_frame: int, n_frames: int, width: int, height: int) -> np.ndarray:
    end_frame = start_frame + n_frames - 1
    select = f"between(n\\,{start_frame}\\,{end_frame})"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-vf", f"select='{select}',scale={width}:{height}:flags=area",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frame_size = width * height * 3
    n_got = len(raw) // frame_size
    if n_got < n_frames:
        raise RuntimeError(
            f"{path}: expected {n_frames} frames starting at frame {start_frame}, got {n_got}"
        )
    arr = np.frombuffer(raw, dtype=np.uint8, count=n_frames * frame_size)
    return arr.reshape(n_frames, height, width, 3).astype(np.float32) / 255.0


def extract_tail_context(path: Path, info: ClipInfo, context_frames: int, width: int, height: int) -> np.ndarray:
    context_frames = min(context_frames, info.n_frames)
    start = info.n_frames - context_frames
    return _read_frames(path, start, context_frames, width, height)


def extract_head_context(path: Path, info: ClipInfo, context_frames: int, width: int, height: int) -> np.ndarray:
    context_frames = min(context_frames, info.n_frames)
    return _read_frames(path, 0, context_frames, width, height)


def build_control_batch(
    context_a: np.ndarray, context_b: np.ndarray, gap_frames: int, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the (control_video, control_mask) pair WanVaceToVideo expects.

    Convention (reverse-engineered from Comfy-Org's own bundled Wan-VACE
    FLF2V template): mask 0 = real/given frame, mask 1 = frame to generate.
    The placeholder pixel content for the to-be-generated span doesn't feed
    the "inactive" stream (it's zeroed by the mask there), only the weaker
    "reactive" stream -- the template uses solid white, which we match.
    """
    n_a, n_b = context_a.shape[0], context_b.shape[0]
    gap_placeholder = np.ones((gap_frames, height, width, 3), dtype=np.float32)
    video = np.concatenate([context_a, gap_placeholder, context_b], axis=0)

    length = n_a + gap_frames + n_b
    mask = np.zeros((length, height, width), dtype=np.float32)
    mask[n_a:n_a + gap_frames] = 1.0

    return torch.from_numpy(video), torch.from_numpy(mask)


def even_size(width: int, height: int) -> tuple[int, int]:
    """libx264's yuv420p needs even dimensions; rounds odd ones down."""
    return width - width % 2, height - height % 2


def conform(input_path: Path, output_path: Path, *, fps: float, width: int, height: int) -> None:
    """Re-encodes to `fps` and `width`x`height`. A different aspect ratio is
    scaled to cover the frame and centre-cropped, never stretched or
    letterboxed, so the model never sees distorted footage or black bars."""
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path),
        "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _write_frames_to_video(
    frames: Iterable[np.ndarray], fps: float, out_width: int, out_height: int, output_path: Path
) -> None:
    """frames: (N, H, W, 3) float32 in [0, 1] at any resolution -- scaled to
    out_width/out_height. Any iterable of (H, W, 3) frames works too, and is
    consumed lazily, so the frames never all have to be in memory at once."""
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[0], first.shape[1]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-vf", f"scale={out_width}:{out_height}:flags=lanczos",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in itertools.chain([first], frames):
        frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        proc.stdin.write(frame_u8.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {output_path}")


def _scale_frames(frames: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rescales (N, H, W, 3) float32 frames with the same lanczos filter
    _write_frames_to_video uses, so frames scaled here match the ones it
    scales on the way out."""
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    if (w, h) == (width, height):
        return frames
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    frames_u8 = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    raw = subprocess.run(cmd, input=frames_u8.tobytes(), capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=np.uint8, count=n * width * height * 3)
    return arr.reshape(n, height, width, 3).astype(np.float32) / 255.0


def crossfade_frames(
    path: Path, info: ClipInfo, start_frame: int, generated: np.ndarray, fade_in: bool
) -> np.ndarray:
    """Linearly blends the real frames of `path` starting at `start_frame`
    with `generated` -- VACE's own decoded version of those same frames (any
    resolution), so the two sides line up and the blend doesn't ghost.
    fade_in=True ramps real -> generated, False ramps generated -> real.
    Neither end of the ramp is a pure copy of either side. Returns frames at
    the clip's native resolution."""
    n = generated.shape[0]
    real = _read_frames(path, start_frame, n, info.width, info.height)
    generated = _scale_frames(generated, info.width, info.height)
    alpha = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[:, None, None, None]
    if not fade_in:
        alpha = 1.0 - alpha
    return (1.0 - alpha) * real + alpha * generated


def export_generated(
    parts: list[tuple[Path, int, int] | np.ndarray],
    info: ClipInfo,
    output_path: Path,
    audio_source: Path | None = None,
    audio_offset_s: float = 0.0,
) -> None:
    """Writes `parts` in order as one clip at `info`'s native resolution and
    frame rate. Each part is either a (clip, start_frame, end_frame_exclusive)
    run of real footage or an array of the model's decoded frames (any
    resolution); empty parts are skipped. For --export-generated: every frame
    the model produced, padded out with real footage to the main output's
    timeline so it can be laid straight over it in an editor."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="soto-export-"))
    files: list[Path] = []
    for i, part in enumerate(parts):
        file = tmp_dir / f"part_{i:03d}.mp4"
        if isinstance(part, np.ndarray):
            if not part.shape[0]:
                continue
            _write_frames_to_video(part, info.fps, info.width, info.height, file)
        else:
            path, start, end = part
            if end <= start:
                continue
            _trim_frames(path, start, end, file)
        files.append(file)

    if audio_source is None:
        _concat(files, output_path)
    else:
        video_only = tmp_dir / "video_only.mp4"
        _concat(files, video_only)
        _mux_audio(video_only, audio_source, output_path, offset_s=audio_offset_s)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _trim_frames(input_path: Path, start_frame: int, end_frame: int | None, output_path: Path) -> None:
    """end_frame is exclusive; None means to the end of the clip."""
    if end_frame is not None:
        select = f"between(n\\,{start_frame}\\,{end_frame - 1})"
    else:
        select = f"gte(n\\,{start_frame})"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path),
        "-vf", f"select='{select}',setpts=N/FRAME_RATE/TB",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _concat(parts: list[Path], output_path: Path) -> None:
    list_file = output_path.with_suffix(".concat.txt")
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in parts))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink()


def splice_output(
    *,
    clip_a: Path,
    clip_b: Path,
    info_a: ClipInfo,
    info_b: ClipInfo,
    generated_frames: torch.Tensor,
    context_frames_a: int,
    gap_frames: int,
    context_frames_b: int,
    crossfade: int,
    output_path: Path,
    tmp_dir: Path,
) -> None:
    """Stitches clip_a + the generated gap + clip_b. The gap carries on from
    clip_a's last frame and leads into clip_b's first (those were its
    context), so both clips are kept whole -- except the `crossfade` frames
    either side of the gap, which ramp from real footage into VACE's decoded
    version of them and back out again (see crossfade_frames)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)

    generated = generated_frames.cpu().numpy()
    gap_start = context_frames_a
    gap_end = context_frames_a + gap_frames
    fps, width, height = info_a.fps, info_a.width, info_a.height
    parts: list[Path] = []

    lead_end = info_a.n_frames - crossfade
    if lead_end > 0:
        lead_video = tmp_dir / "lead.mp4"
        _trim_frames(clip_a, 0, lead_end, lead_video)
        parts.append(lead_video)

    if crossfade:
        fade_a = crossfade_frames(clip_a, info_a, lead_end, generated[gap_start - crossfade:gap_start], fade_in=True)
        fade_a_video = tmp_dir / "fade_a.mp4"
        _write_frames_to_video(fade_a, fps, width, height, fade_a_video)
        parts.append(fade_a_video)

    gap_video = tmp_dir / "gap.mp4"
    _write_frames_to_video(generated[gap_start:gap_end], fps, width, height, gap_video)
    parts.append(gap_video)

    if crossfade:
        fade_b = crossfade_frames(clip_b, info_b, 0, generated[gap_end:gap_end + crossfade], fade_in=False)
        fade_b_video = tmp_dir / "fade_b.mp4"
        _write_frames_to_video(fade_b, fps, width, height, fade_b_video)
        parts.append(fade_b_video)

    if crossfade < info_b.n_frames:
        tail_video = tmp_dir / "tail.mp4"
        _trim_frames(clip_b, crossfade, None, tail_video)
        parts.append(tail_video)

    _concat(parts, output_path)


def read_frames(path: Path, start_frame: int, n_frames: int, width: int, height: int) -> np.ndarray:
    return _read_frames(path, start_frame, n_frames, width, height)


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """'#1a2b3c', '1a2b3c', '#fff' or 'fff' -> (r, g, b) in 0-255."""
    s = value.strip().removeprefix("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"not a hex colour: {value!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


# Frames are downscaled to this width before colour matching -- a flat frame
# is flat at any resolution, and geq is evaluated per-pixel in software.
_DETECT_WIDTH = 96


def detect_flat_intervals(
    path: Path, color: tuple[int, int, int], tolerance: float, coverage: float = 0.98
) -> list[tuple[int, int]]:
    """Finds runs of frames that are a flat `color`, as (start, end_exclusive) frame indices.

    ffmpeg has no "match this colour" detector, but it does have one for black,
    so we turn colour matching into black detection: geq rewrites every pixel
    to its Chebyshev distance from `color` (max |channel difference|), which
    is ~0 exactly where the frame matches, then blackframe flags frames where
    `coverage` of pixels fall under `tolerance`. blackframe (rather than
    blackdetect) reports exact frame indices instead of timestamps, so there's
    no seconds -> frames rounding at the interval edges."""
    r, g, b = color
    dist = f"max(max(abs(r(X\\,Y)-{r})\\,abs(g(X\\,Y)-{g}))\\,abs(b(X\\,Y)-{b}))"
    threshold = max(1, round(tolerance * 255))
    vf = (
        f"scale={_DETECT_WIDTH}:-2,format=gbrp,geq=r='{dist}':g='{dist}':b='{dist}',"
        f"format=gray,blackframe=amount={round(coverage * 100)}:threshold={threshold}"
    )
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-an", "-vf", vf, "-f", "null", "-"]
    stderr = subprocess.run(cmd, capture_output=True, text=True, check=True).stderr
    flat = sorted({int(m) for m in re.findall(r"\bframe:(\d+) pblack:", stderr)})

    intervals: list[tuple[int, int]] = []
    for n in flat:
        if intervals and intervals[-1][1] == n:
            intervals[-1] = (intervals[-1][0], n + 1)
        else:
            intervals.append((n, n + 1))
    return intervals


def splice_fills(
    *,
    clip: Path,
    info: ClipInfo,
    fills: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    tmp_dir: Path,
) -> None:
    """Rebuilds `clip` with each (start, end_exclusive, frames, before, after)
    span replaced by `frames` (at any resolution; scaled back to the clip's
    own). `before` / `after` are VACE's decoded versions of the real frames
    just before `start` / from `end` (either may be empty), crossfaded over
    those frames to soften each seam. Every fill is exactly as long as the
    span it replaces, so the duration is unchanged and the original audio
    track (if any) is carried over as-is."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    cursor = 0
    for i, (start, end, frames, before, after) in enumerate(sorted(fills, key=lambda f: f[0])):
        fade_start = start - before.shape[0]
        if fade_start > cursor:
            real = tmp_dir / f"real_{i:03d}.mp4"
            _trim_frames(clip, cursor, fade_start, real)
            parts.append(real)
        if before.shape[0]:
            fade_in = tmp_dir / f"fade_in_{i:03d}.mp4"
            blended = crossfade_frames(clip, info, fade_start, before, fade_in=True)
            _write_frames_to_video(blended, info.fps, info.width, info.height, fade_in)
            parts.append(fade_in)
        gap = tmp_dir / f"gap_{i:03d}.mp4"
        _write_frames_to_video(frames, info.fps, info.width, info.height, gap)
        parts.append(gap)
        if after.shape[0]:
            fade_out = tmp_dir / f"fade_out_{i:03d}.mp4"
            blended = crossfade_frames(clip, info, end, after, fade_in=False)
            _write_frames_to_video(blended, info.fps, info.width, info.height, fade_out)
            parts.append(fade_out)
        cursor = end + after.shape[0]
    if cursor < info.n_frames:
        real = tmp_dir / "real_end.mp4"
        _trim_frames(clip, cursor, None, real)
        parts.append(real)

    video_only = tmp_dir / "video_only.mp4"
    _concat(parts, video_only)
    _mux_audio(video_only, clip, output_path)


def extend_frames(
    *,
    clip: Path,
    info: ClipInfo,
    head: np.ndarray | None,
    head_fade: np.ndarray,
    tail: Iterable[np.ndarray] | None,
    tail_fade: np.ndarray,
    output_path: Path,
    tmp_dir: Path,
) -> None:
    """Writes `head` + `clip` + `tail`, where `head` / `tail` (either may be
    None) are new frames at any resolution, scaled to the clip's own. `tail`
    can also be an iterable, consumed lazily as it's encoded. `head_fade` /
    `tail_fade` are VACE's decoded versions of the clip's first / last few
    frames (either may be empty), crossfaded over them to soften each seam.
    The original audio track (if any) is carried over as-is, delayed by the
    length of `head` so it stays in sync with the real footage."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    head_frames = 0
    if head is not None:
        head_video = tmp_dir / "head.mp4"
        head_frames = head.shape[0]
        _write_frames_to_video(head, info.fps, info.width, info.height, head_video)
        parts.append(head_video)
    real_start = head_fade.shape[0]
    real_end = info.n_frames - tail_fade.shape[0]
    if real_start:
        fade = tmp_dir / "fade_head.mp4"
        blended = crossfade_frames(clip, info, 0, head_fade, fade_in=False)
        _write_frames_to_video(blended, info.fps, info.width, info.height, fade)
        parts.append(fade)
    if real_end > real_start:
        real = tmp_dir / "real.mp4"
        _trim_frames(clip, real_start, real_end, real)
        parts.append(real)
    if tail_fade.shape[0]:
        fade = tmp_dir / "fade_tail.mp4"
        blended = crossfade_frames(clip, info, real_end, tail_fade, fade_in=True)
        _write_frames_to_video(blended, info.fps, info.width, info.height, fade)
        parts.append(fade)
    if tail is not None:
        tail_video = tmp_dir / "tail.mp4"
        _write_frames_to_video(tail, info.fps, info.width, info.height, tail_video)
        parts.append(tail_video)

    video_only = tmp_dir / "video_only.mp4"
    _concat(parts, video_only)
    _mux_audio(video_only, clip, output_path, offset_s=head_frames / info.fps)


def _mux_audio(video_only: Path, audio_source: Path, output_path: Path, offset_s: float = 0.0) -> None:
    """Copies `video_only`'s video and `audio_source`'s audio (if it has any)
    into `output_path`, the audio starting `offset_s` seconds in."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only),
        *(["-itsoffset", f"{offset_s:.6f}"] if offset_s else []), "-i", str(audio_source),
        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", str(output_path),
    ]
    subprocess.run(cmd, check=True)
