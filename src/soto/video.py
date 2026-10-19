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


def _read_frames(path: Path, start_frame: int, n_frames: int, width: int, height: int,
                 flags: str = "area") -> np.ndarray:
    end_frame = start_frame + n_frames - 1
    select = f"between(n\\,{start_frame}\\,{end_frame})"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-vf", f"select='{select}',scale={width}:{height}:flags={flags}",
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


def iter_frames(path: Path, width: int, height: int, flags: str = "area") -> Iterable[np.ndarray]:
    """Yields every frame of `path`, scaled to width x height, as (H, W, 3)
    uint8, decoding the whole clip in one pass (read_frames re-decodes from
    the start on every call, which is quadratic over a long clip)."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-vf", f"scale={width}:{height}:flags={flags}",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frame_size = width * height * 3
    try:
        while len(raw := proc.stdout.read(frame_size)) == frame_size:
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def write_preview(frames: Iterable[np.ndarray], fps: float, output_path: Path) -> int:
    """Writes (H, W, 3) float32 frames in [0, 1] as a small H.264 mp4, unscaled."""
    return _pipe_frames(frames, fps, output_path, ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"])


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


def _pipe_frames(frames: Iterable[np.ndarray], fps: float, output_path: Path, out_args: list[str]) -> int:
    """Encodes (H, W, 3) float32 frames in [0, 1] with ffmpeg, `out_args`
    choosing the filters and codec. The frames are consumed lazily, so they
    never all have to be in memory at once. Returns how many there were."""
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[0], first.shape[1]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", *out_args, str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    n = 0
    for frame in itertools.chain([first], frames):
        frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        proc.stdin.write(frame_u8.tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {output_path}")
    return n


def _write_frames_to_video(
    frames: Iterable[np.ndarray], fps: float, out_width: int, out_height: int, output_path: Path
) -> None:
    """frames: (N, H, W, 3) float32 in [0, 1] at any resolution -- scaled to
    out_width/out_height. Any iterable of (H, W, 3) frames works too."""
    _pipe_frames(frames, fps, output_path, [
        "-vf", f"scale={out_width}:{out_height}:flags=lanczos", "-c:v", "libx264", "-pix_fmt", "yuv420p",
    ])


def write_lossless(frames: Iterable[np.ndarray], fps: float, output_path: Path) -> int:
    """Writes frames as they are, unscaled, as FFV1 in Matroska. Reading them
    back with read_frames gives exactly the values _write_frames_to_video
    would have encoded from the originals: both quantize the same way, and
    k/255 * 255 comes back to k. Returns how many frames were written."""
    return _pipe_frames(frames, fps, output_path, ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0"])


def concat_lossless(parts: list[Path], output_path: Path) -> None:
    """Joins write_lossless files by stream copy: FFV1 is intra-only, so it
    can be cut and joined anywhere without re-encoding."""
    list_file = output_path.with_suffix(".concat.txt")
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in parts))
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        list_file.unlink(missing_ok=True)


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


@dataclass
class Real:
    """Frames [start, end) of a clip, as they are."""
    path: Path
    start: int
    end: int


@dataclass
class Fade:
    """The real frames of `path` from `start`, blended with `generated`
    (see crossfade_frames)."""
    path: Path
    start: int
    generated: np.ndarray
    fade_in: bool


@dataclass
class Generated:
    """Frames [start, end) of a write_lossless file at `width`x`height`."""
    path: Path
    start: int
    end: int
    width: int
    height: int


def _read_chunked(part: Generated, chunk: int = 32) -> Iterable[np.ndarray]:
    for start in range(part.start, part.end, chunk):
        yield from _read_frames(part.path, start, min(chunk, part.end - start), part.width, part.height)


def assemble(
    parts: list[Real | Fade | Generated],
    info: ClipInfo,
    output_path: Path,
    audio: tuple[Path, float] | None = None,
) -> None:
    """Writes `parts` in order as one clip at `info`'s resolution and frame
    rate. Real footage is cut out of its clip, and generated frames are
    scaled up to the clip's size. `audio` is (clip, offset in seconds): that
    clip's audio, if it has any, starting that far in; with None the output
    is silent."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="soto-assemble-"))
    files: list[Path] = []
    for i, part in enumerate(parts):
        file = tmp_dir / f"part_{i:03d}.mp4"
        if isinstance(part, Real):
            if part.end <= part.start:
                continue
            _trim_frames(part.path, part.start, part.end, file)
        elif isinstance(part, Fade):
            if not part.generated.shape[0]:
                continue
            blended = crossfade_frames(part.path, info, part.start, part.generated, part.fade_in)
            _write_frames_to_video(blended, info.fps, info.width, info.height, file)
        else:
            if part.end <= part.start:
                continue
            _write_frames_to_video(_read_chunked(part), info.fps, info.width, info.height, file)
        files.append(file)

    if audio is None:
        _concat(files, output_path)
    else:
        video_only = tmp_dir / "video_only.mp4"
        _concat(files, video_only)
        _mux_audio(video_only, audio[0], output_path, offset_s=audio[1])
    shutil.rmtree(tmp_dir, ignore_errors=True)


def read_frames(path: Path, start_frame: int, n_frames: int, width: int, height: int,
                flags: str = "area") -> np.ndarray:
    return _read_frames(path, start_frame, n_frames, width, height, flags)


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


def _mux_audio(video_only: Path, audio_source: Path, output_path: Path, offset_s: float = 0.0) -> None:
    """Copies `video_only`'s video and `audio_source`'s audio (if it has any)
    into `output_path`, the audio starting `offset_s` seconds in."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_only),
        *(["-itsoffset", f"{offset_s:.6f}"] if offset_s else []), "-i", str(audio_source),
        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", str(output_path),
    ]
    subprocess.run(cmd, check=True)


# A source rect inside a larger canvas, as (x0, y0, x1, y1) in pixels, x1/y1 exclusive.
Rect = tuple[int, int, int, int]


def _inner_rect(rect: Rect, width: int, height: int, seam: float) -> tuple[float, float, float, float]:
    """`rect` shrunk by `seam` on each side that has a border to fill, i.e.
    isn't flush with the canvas edge."""
    x0, y0, x1, y1 = rect
    return (
        x0 + seam if x0 > 0 else x0,
        y0 + seam if y0 > 0 else y0,
        x1 - seam if x1 < width else x1,
        y1 - seam if y1 < height else y1,
    )


def build_outpaint_batch(
    given: np.ndarray, frames: np.ndarray, width: int, height: int, rect: Rect, seam_px: int,
    surround: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the (control_video, control_mask) pair for outpainting:
    `frames` (already scaled to `rect`'s size) placed on a white canvas (or
    on `surround`, (n, height, width, 3), if given),
    masked (1 = generate) everywhere outside `rect` and `seam_px` into it,
    so VACE redraws a band across the seam for the composite to blend over.
    `given` are whole canvases from the previous window, kept as they are
    (mask 0) so the new borders carry on from the old ones."""
    x0, y0, x1, y1 = rect
    n_given = given.shape[0]
    placed = (np.ones((frames.shape[0], height, width, 3), dtype=np.float32) if surround is None
              else surround.astype(np.float32, copy=True))
    placed[:, y0:y1, x0:x1] = frames
    video = np.concatenate([given, placed], axis=0)

    ix0, iy0, ix1, iy1 = (int(v) for v in _inner_rect(rect, width, height, seam_px))
    mask = np.ones((video.shape[0], height, width), dtype=np.float32)
    mask[:n_given] = 0.0
    mask[n_given:, iy0:iy1, ix0:ix1] = 0.0
    return torch.from_numpy(video), torch.from_numpy(mask)


def build_dream_outpaint_batch(
    given: np.ndarray, canvases: np.ndarray, weight: np.ndarray, generate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (control_video, control_mask) pair for `--fill vace-dream`:
    `canvases` are already whole dream-filled frames, masked at `generate`
    times `weight` (n, h, w, 0-1, how far each pixel is blurred from the
    clip as it was), so VACE restyles the blur and leaves the subject be.
    `given` are kept as they are, as in build_outpaint_batch."""
    video = np.concatenate([given, canvases], axis=0)
    mask = np.concatenate([np.zeros(given.shape[:3], dtype=np.float32), generate * weight], axis=0)
    return torch.from_numpy(video), torch.from_numpy(mask.astype(np.float32))


def _seam_alpha(rect: Rect, width: int, height: int, seam: float) -> np.ndarray:
    """(h, w, 1) weight of the real footage over `rect`: 1 in the middle,
    ramping to 0 at each edge that borders generated picture over `seam` px."""
    x0, y0, x1, y1 = rect

    def ramp(n: int, lo_open: bool, hi_open: bool) -> np.ndarray:
        d = np.arange(n, dtype=np.float32) + 0.5
        a = np.ones(n, dtype=np.float32)
        if seam > 0:
            if lo_open:
                a = np.minimum(a, d / seam)
            if hi_open:
                a = np.minimum(a, d[::-1] / seam)
        return np.clip(a, 0.0, 1.0)

    ax = ramp(x1 - x0, x0 > 0, x1 < width)
    ay = ramp(y1 - y0, y0 > 0, y1 < height)
    return np.minimum(ay[:, None], ax[None, :])[..., None]


def composite_outpaint(
    canvases: list[np.ndarray], clip: Path, rect: Rect, width: int, height: int, seam: float,
    chunk: int = 32,
) -> Iterable[np.ndarray]:
    """Yields output frames at `width`x`height`: each generated canvas (uint8,
    any resolution) scaled up, with the clip's own frame, read straight from
    the source at `rect`'s size, laid over `rect` and feathered into the
    generated picture across `seam` px."""
    x0, y0, x1, y1 = rect
    rw, rh = x1 - x0, y1 - y0
    alpha = _seam_alpha(rect, width, height, seam)
    for start in range(0, len(canvases), chunk):
        batch = np.stack(canvases[start:start + chunk]).astype(np.float32) / 255.0
        up = _scale_frames(batch, width, height)
        real = _read_frames(clip, start, batch.shape[0], rw, rh, flags="lanczos")
        up[:, y0:y1, x0:x1] = alpha * real + (1.0 - alpha) * up[:, y0:y1, x0:x1]
        yield from up


def write_outpaint(frames: Iterable[np.ndarray], fps: float, width: int, height: int,
                   audio_source: Path, output_path: Path) -> None:
    """Encodes `frames` (already at `width`x`height`) with `audio_source`'s audio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="soto-expand-"))
    video_only = tmp_dir / "video_only.mp4"
    _write_frames_to_video(frames, fps, width, height, video_only)
    _mux_audio(video_only, audio_source, output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def blur_fill(input_path: Path, output_path: Path, width: int, height: int, rect: Rect) -> None:
    """The no-model fill: the clip scaled to `rect` over a heavily blurred,
    slightly darkened copy of itself scaled to cover `width`x`height`. Audio
    is copied as-is."""
    x0, y0, x1, y1 = rect
    sigma = max(4.0, 0.04 * width)
    graph = (
        "[0:v]split[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase:flags=area,crop={width}:{height},"
        f"gblur=sigma={sigma:.1f},eq=brightness=-0.05[bgb];"
        f"[fg]scale={x1 - x0}:{y1 - y0}:flags=lanczos[fgs];"
        f"[bgb][fgs]overlay={x0}:{y0},setsar=1[v]"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path),
        "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", str(output_path),
    ]
    subprocess.run(cmd, check=True)
