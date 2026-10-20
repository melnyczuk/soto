"""The anamorphic look for `soto expand`: the clip's subject stays sharp, and
everything else is blurred more the further it is from it, out through the
clip's own background and on into the new edges. The blur is already at the
fill's strength by the time it reaches the clip's edge, so there's no seam
where the real picture stops.

The subject is found with a depth model (Depth Anything V2 Small) and a
cheap guess at where it is: colour that stands out from the rest of the
frame, plus local motion and fine detail, with a pull toward the centre.
The guess picks out the subject, the depth map finds the rest of it, and
everything else is blurred by how far it is in depth from it, as through a
lens focused on it. Without depth, the guess alone is the subject, and the
blur grows with distance from it in the frame instead.

Outside the clip is FLUX.1 Fill's outpainting (see `flux_plates`), bent as by
an anamorphic lens (see `_deform`). The blur is then worked out per pixel as
a "level": 0 is untouched, and it rises to ANAMORPHIC_EDGE at the clip's edge and
on to ANAMORPHIC_FAR at the output's corners."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from soto import config, video

Rect = tuple[int, int, int, int]

# Long side of the grids the subject is found on and the blur level is worked
# out on. Both are smooth, so they're upsampled from there.
_FOCUS_SIZE = 192
# Long side of the frames the depth model sees; how far in depth from the
# subject still counts as it (as a fraction of the frame's depth range); and
# how far from the colour-and-motion guess it can reach, as a fraction of
# the frame's short side.
_DEPTH_SIZE = 518
_DEPTH_TOLERANCE = 0.08
_SUBJECT_REACH = 0.35
# Segmenting with SAM: every how many frames, prompted on how fine a grid of
# points (across, down), keeping objects between these fractions of the
# frame, and at most this many of them.
_SEGMENT_EVERY = 3
_SEGMENT_GRID = (8, 5)
_SEGMENT_AREA = (0.004, 0.55)
_SEGMENT_MAX = 3
# The least score a candidate needs to count as a focus at all: footage with
# no subject (a landscape) shouldn't have one picked out of it.
_SEGMENT_MIN = 0.2
_LEVEL_SIZE = 256
# The blur at a level of 1, as a fraction of the clip's short side (5% of
# the width of a 16:9 clip).
_EDGE_SIGMA = 0.089
# The most of a frame that's ever taken for the subject, before it's rounded out.
_MAX_SUBJECT = 0.2
# The smear that pads Flux's first keyframe: how many samples it's averaged
# over, reaching how far toward the clip's centre.
_SMEAR_SAMPLES = 8
_SMEAR_REACH = 0.85
# The bend, at --deform 1 and fully out of focus. It comes in over an
# ellipse round the clip's centre, at the clip's aspect (1 = the middle of
# its edges): nothing inside _DEFORM_INNER, easing up to full at the
# output's far corners, so the clip's own corners bend a little too. There:
# how much nearer the centre each point is drawn from, across and down
# (stretching it outward, mostly sideways), how far it's swirled round, in
# radians (about 3.5 degrees), and how far red and blue are drawn apart.
_DEFORM_INNER = 0.45
_DEFORM_STRETCH_X = 0.12
_DEFORM_STRETCH_Y = 0.04
_DEFORM_SWIRL = 0.06
_FRINGE = 0.008
# The slow drift of the bent edges at --wobble 1, as a fraction of the
# clip's short side: a few waves, each (period in seconds, waves across the
# frame, phase).
_WOBBLE = 0.01
_WOBBLE_WAVES = ((3.1, 1.3, 0.0), (4.7, 2.1, 1.9), (6.9, 0.7, 4.2))
# How much the bent edges drift is set by how much the camera moves: the
# whole frame's shift over _MOTION_SPAN seconds, as a fraction of its width
# per second, measured on frames _MOTION_SIZE wide. A shot that holds still
# doesn't drift at all; one moving _MOTION_FULL or more drifts fully.
_MOTION_SIZE = 192
_MOTION_SPAN = 0.2
_MOTION_FULL = 0.1
# Above this much bend, a pixel of the clip has moved, so it's left out of
# the corruption check.
_BENT = 0.01
# Checking for a corrupted batch: below this blur level a pixel's barely
# touched (well under a pixel of blur), so if those pixels are off from the
# frames by more than _CORRUPT on average, the batch is garbage.
_UNBLURRED = 0.02
_CORRUPT = 0.03
# The clip is left as it was but for this band along its open edges, as a
# fraction of its short side, where it eases into the new picture.
_KEEP_BAND = 0.06


@dataclass
class AnamorphicLook:
    strength: float  # blur at the clip's edge, x 5% of the output width
    feather: float  # how far in from the clip's edges the blur reaches it, fraction of the short side
    falloff: float  # how far from the subject the background takes to blur, fraction of the short side
    streak: float  # opacity of the horizontal light streaks
    depth_falloff: float = 0.3  # with depth: how far in depth from the subject the background takes to blur
    deform: float = 1.0  # how much the new edges bend, as through an anamorphic lens (1 = default)
    wobble: float = 1.0  # how much the bent edges drift (1 = default)
    fps: float = 24.0  # the clip's, to time the drift by


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gauss(x: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """Separable gaussian blur of (n, c, h, w), reflecting at the borders."""
    c = x.shape[1]
    for sigma, horizontal in ((sx, True), (sy, False)):
        r = min(int(3 * sigma), x.shape[-1 if horizontal else -2] - 1)
        if sigma < 0.3 or r < 1:
            continue
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2)
        k = k / k.sum()
        if horizontal:
            x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        else:
            x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x


def _blur(x: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """A wide gaussian blur, done at a reduced size so the kernel stays small."""
    h, w = x.shape[-2:]
    f = max(1.0, min(sx, sy) / 2.0)
    if f == 1.0:
        return _gauss(x, sx, sy)
    small = F.interpolate(x, size=(max(2, round(h / f)), max(2, round(w / f))), mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(_gauss(small, sx / f, sy / f), size=(h, w), mode="bilinear", align_corners=False)


def _grid(width: int, height: int, long_side: int) -> tuple[int, int]:
    s = long_side / max(width, height)
    return max(8, round(width * s)), max(8, round(height * s))


def _ema(x: np.ndarray, keep: float) -> np.ndarray:
    """Forward-backward exponential smoothing along the first axis, so it
    doesn't lag behind the footage."""
    out = x.astype(np.float32)
    for i in range(1, len(out)):
        out[i] = keep * out[i - 1] + (1.0 - keep) * out[i]
    for i in range(len(out) - 2, -1, -1):
        out[i] = keep * out[i + 1] + (1.0 - keep) * out[i]
    return out


def _salience(clip: Path, info: video.ClipInfo, chunk: int = 64) -> np.ndarray:
    """(n, h, w) float16 in [0, 1], on a small grid at the clip's aspect:
    how much each part of each frame stands out as the subject, by colour,
    motion and detail alone."""
    import kornia

    fw, fh = _grid(info.width, info.height, _FOCUS_SIZE)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, fh), torch.linspace(-1, 1, fw), indexing="ij")
    centre = 0.4 + 0.6 * torch.exp(-(xx**2 + yy**2) / (2 * 0.6**2))
    weights = torch.tensor([0.5, 1.0, 1.0]).view(1, 3, 1, 1)
    laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)

    def unit(x: torch.Tensor, floor: float) -> torch.Tensor:
        # Per frame, so a 99th-percentile response is 1, but never scaling up
        # a frame with nothing in it past `floor`.
        p = torch.quantile(x.flatten(1), 0.99, dim=1).clamp(min=floor)
        return (x / p.view(-1, 1, 1, 1)).clamp(0.0, 1.5)

    scores: list[np.ndarray] = []
    prev_l = None
    frames = video.iter_frames(clip, fw, fh)
    while batch := [f for _, f in zip(range(chunk), frames)]:
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float() / 255.0
        lab = kornia.color.rgb_to_lab(x)
        light = lab[:, :1]
        # Colour that stands out from the frame's typical colour.
        typical = lab.flatten(2).median(dim=2).values.view(-1, 3, 1, 1)
        rarity = unit(_gauss((((lab - typical) * weights) ** 2).sum(1, keepdim=True).sqrt(), 1.5, 1.5), 20.0)
        # Motion against the frame's overall change, so a camera move doesn't
        # count everywhere at once.
        before = torch.cat([light[:1] if prev_l is None else prev_l, light[:-1]])
        diff = _gauss((light - before).abs(), 2.0, 2.0)
        motion = unit((diff - diff.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)).clamp(min=0.0), 4.0)
        # Fine detail: what's in focus, if the footage has a shallow depth of field.
        detail = unit(_gauss(F.conv2d(F.pad(light, (1, 1, 1, 1), mode="replicate"), laplacian).abs(), 2.0, 2.0), 2.0)
        prev_l = light[-1:]
        score = (0.6 * rarity + 0.2 * motion + 0.2 * detail)[:, 0] * centre
        scores.append(score.numpy().astype(np.float16))

    score = torch.from_numpy(_ema(np.concatenate(scores), keep=0.8))
    # Soft-threshold against each frame's strongest response, so there's
    # always a subject, but never most of the frame: footage that's all
    # texture (smoke, water) has no subject to keep, so it blurs out too.
    rel = score / score.flatten(1).amax(dim=1).clamp(min=1e-3).view(-1, 1, 1)
    floor = torch.quantile(rel.flatten(1).float(), 1.0 - _MAX_SUBJECT, dim=1).clamp(min=0.3).view(-1, 1, 1)
    m = _smoothstep(0.0, 1.0, (rel - floor) / 0.35)[:, None]
    m = _gauss(F.max_pool2d(m, 9, stride=1, padding=4), 2.0, 2.0)[:, 0]
    return m.clamp(0.0, 1.0).numpy().astype(np.float16)


@dataclass
class Focus:
    """Per frame, on a small grid at the clip's aspect: `mask`, how much each
    part is the subject, and `defocus`, how far it is in depth from the
    subject (0 = as near as it, 1 = the whole frame's depth range), and
    `near`, how near it is (0 = farthest, 1 = nearest); both None when
    found without depth."""
    mask: np.ndarray
    defocus: np.ndarray | None = None
    near: np.ndarray | None = None
    depth: np.ndarray | None = None  # (n,) the subject's own `near`
    motion: np.ndarray | None = None  # (n,) how much the camera's moving, 0-1 (see _camera_motion)

    def __len__(self) -> int:
        return self.mask.shape[0]

    def _map(self, f) -> Focus:
        return Focus(f(self.mask), *(None if x is None else f(x)
                                     for x in (self.defocus, self.near, self.depth, self.motion)))

    def __getitem__(self, frames) -> Focus:
        return self._map(lambda x: x[frames])

    def padded(self, n: int) -> Focus:
        """With `n` copies of the last frame on the end."""
        return self._map(lambda x: np.concatenate([x, np.repeat(x[-1:], n, axis=0)], axis=0))


def _depth(clip: Path, info: video.ClipInfo, width: int, height: int, chunk: int = 8) -> np.ndarray:
    """(n, height, width) float32 nearness of each frame, with Depth Anything
    V2: 0 for the farthest, 1 for the nearest of each frame, smoothed over
    time. It's relative (the model only knows what's nearer than what), so
    it's spread over each frame's own range."""
    from tqdm import tqdm
    from transformers import AutoModelForDepthEstimation

    if not (config.DEPTH_MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"no depth model in {config.DEPTH_MODEL_DIR}: run `soto setup`, "
                                "or pass --focus heuristic")
    dev = _device()
    model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL_DIR).to(dev).eval()
    # The model works in multiples of its 14px patches.
    s = _DEPTH_SIZE / max(info.width, info.height)
    iw, ih = (max(14, round(v * s / 14) * 14) for v in (info.width, info.height))
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    frames = video.iter_frames(clip, iw, ih)
    with torch.inference_mode(), tqdm(total=info.n_frames, unit="frame", desc="depth") as bar:
        while batch := [f for _, f in zip(range(chunk), frames)]:
            x = torch.from_numpy(np.stack(batch)).to(dev).permute(0, 3, 1, 2).float() / 255.0
            d = model(pixel_values=(x - mean) / std).predicted_depth[:, None]
            d = F.interpolate(d, size=(height, width), mode="bilinear", antialias=True, align_corners=False)[:, 0]
            lo = torch.quantile(d.flatten(1), 0.02, dim=1).view(-1, 1, 1)
            hi = torch.quantile(d.flatten(1), 0.98, dim=1).view(-1, 1, 1)
            out.append(((d - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0).cpu().numpy())
            bar.update(len(batch))
    return _ema(np.concatenate(out), keep=0.5)


def _segment(clip: Path, info: video.ClipInfo, guess: np.ndarray, near: np.ndarray, at_depth: np.ndarray,
             chunk: int = 4) -> np.ndarray | None:
    """(n, h, w) float32: the objects that are the frame's focus, segmented
    with SlimSAM, or None without its weights. Every `_SEGMENT_EVERY`th frame
    is segmented and the frames between blended from those either side.

    SAM is prompted all over the frame, so every object is a candidate, and
    each is scored as a focus: how much it stands out in colour and motion
    (`guess`), whether it's in front of what's around it in depth (an object
    is; the ground it stands on isn't), how much of it is at the depth of
    what stands out (`at_depth`) and how much of that it covers, a pull toward
    the centre, whether it runs along the frame's edge (the ground does; a
    subject mostly doesn't), and whether it was the focus the last time. The
    best is kept if it scores well enough at all, and any others nearly as
    good: several subjects all stay sharp."""
    from tqdm import tqdm
    from transformers import SamModel, SamProcessor

    if not (config.SEGMENT_MODEL_DIR / "config.json").is_file():
        return None
    dev = _device()
    model = SamModel.from_pretrained(config.SEGMENT_MODEL_DIR).to(dev).eval()
    processor = SamProcessor.from_pretrained(config.SEGMENT_MODEL_DIR)
    n, fh, fw = guess.shape
    keys = list(range(0, n, _SEGMENT_EVERY))
    if keys[-1] != n - 1:
        keys.append(n - 1)
    # SAM resizes to 1024 on its long side anyway.
    s = 1024 / max(info.width, info.height)
    iw, ih = round(info.width * s), round(info.height * s)
    gy, gx = np.meshgrid((np.arange(_SEGMENT_GRID[1]) + 0.5) / _SEGMENT_GRID[1],
                         (np.arange(_SEGMENT_GRID[0]) + 0.5) / _SEGMENT_GRID[0], indexing="ij")
    grid_points = np.stack([gx.ravel() * iw, gy.ravel() * ih], axis=1)
    yy, xx = np.meshgrid((np.arange(fh) + 0.5) / fh, (np.arange(fw) + 0.5) / fw, indexing="ij")

    found: dict[int, np.ndarray] = {}
    last = None
    wanted = set(keys)
    frames = ((i, f) for i, f in enumerate(video.iter_frames(clip, iw, ih)) if i in wanted)
    with torch.inference_mode(), tqdm(total=len(keys), unit="frame", desc="segment") as bar:
        for i, frame in frames:
            g, nr, d = (torch.from_numpy(x[i].astype(np.float32)) for x in (guess, near, at_depth))
            # The grid, plus wherever the guess is strongest.
            py, px = np.unravel_index(np.argsort(guess[i].ravel())[-3:], (fh, fw))
            points = np.concatenate([grid_points, np.stack([(px + 0.5) / fw * iw, (py + 0.5) / fh * ih], axis=1)])
            inputs = processor(frame.copy(), input_points=[[[p] for p in points.tolist()]], return_tensors="pt")
            emb = model.get_image_embeddings(inputs["pixel_values"].float().to(dev))
            out = model(image_embeddings=emb, input_points=inputs["input_points"].float().to(dev),
                        multimask_output=True)
            masks = processor.image_processor.post_process_masks(
                out.pred_masks.float().cpu(), [[fh, fw]], inputs["reshaped_input_sizes"].tolist(), binarize=True,
            )[0].reshape(-1, fh, fw).float()
            quality = out.iou_scores.float().cpu().reshape(-1)

            area = masks.mean(dim=(1, 2))
            ok = (area > _SEGMENT_AREA[0]) & (area < _SEGMENT_AREA[1])
            if not ok.any():
                found[i] = np.zeros((fh, fw), dtype=np.float32)
                bar.update(1)
                continue
            masks, quality, area = masks[ok], quality[ok], area[ok]
            size = masks.sum(dim=(1, 2)).clamp(min=1.0)
            stands_out = (masks * g).sum(dim=(1, 2)) / size
            stands_out = stands_out / stands_out.max().clamp(min=1e-3)
            ring = F.max_pool2d(masks[:, None], 9, stride=1, padding=4)[:, 0] * (1 - masks)
            in_front = (masks * nr).sum(dim=(1, 2)) / size - (ring * nr).sum(dim=(1, 2)) / ring.sum(dim=(1, 2)).clamp(min=1.0)
            in_front = (in_front * 5).clamp(-1.0, 1.0)
            at = d > 0.5
            pure = (masks * at).sum(dim=(1, 2)) / size
            cover = (masks * at).sum(dim=(1, 2)) / at.sum().clamp(min=1)
            cy = (masks * torch.from_numpy(yy).float()).sum(dim=(1, 2)) / size
            cx = (masks * torch.from_numpy(xx).float()).sum(dim=(1, 2)) / size
            central = (1.0 - torch.hypot(cx - 0.5, cy - 0.5) / 0.7).clamp(0.0, 1.0)
            # The ground, a table top, a hillside run along the frame's edge; a
            # subject mostly sits inside it.
            border = torch.cat([masks[:, 0], masks[:, -1], masks[:, :, 0], masks[:, :, -1]], dim=1).mean(dim=1)
            inside = (1.0 - 2.5 * border).clamp(0.0, 1.0)
            score = (quality * (0.3 * stands_out + 0.3 * in_front + 0.2 * pure + 0.2 * cover) * (0.7 + 0.3 * central)
                     * inside)
            if last is not None:
                kept = torch.from_numpy(last) > 0.5
                union = (masks.bool() | kept).sum(dim=(1, 2)).clamp(min=1)
                score = score + 0.4 * (masks.bool() & kept).sum(dim=(1, 2)) / union
            order = torch.argsort(score, descending=True)
            chosen = torch.zeros((fh, fw), dtype=torch.bool)
            for j in order[:_SEGMENT_MAX].tolist():
                if score[j] < max(0.75 * score[order[0]], _SEGMENT_MIN):
                    break
                m = masks[j].bool()
                if chosen.any() and (m & chosen).sum() > 0.3 * m.sum():
                    continue
                chosen |= m
            found[i] = chosen.float().numpy()
            last = found[i]
            bar.update(1)

    out = np.zeros((n, fh, fw), dtype=np.float32)
    for a, b in itertools.pairwise(keys):
        for t in range(a, b + 1):
            w = (t - a) / max(b - a, 1)
            out[t] = (1 - w) * found[a] + w * found[b]
    return _ema(out, keep=0.5)


def _camera_motion(clip: Path, info: video.ClipInfo) -> np.ndarray:
    """(n,) float32, 0-1: how much the camera's moving at each frame. The
    whole frame's shift from `_MOTION_SPAN` seconds before, found by phase
    correlation (so a subject moving about in a still frame barely counts,
    and a pan, a tilt or a shaky hand does), as a fraction of its width per
    second, against `_MOTION_FULL`, and smoothed over time."""
    w, h = _grid(info.width, info.height, _MOTION_SIZE)
    span = max(1, round(info.fps * _MOTION_SPAN))
    window = torch.hann_window(h)[:, None] * torch.hann_window(w)[None]
    spectra: list[torch.Tensor] = []
    speed = np.zeros(info.n_frames, dtype=np.float32)
    for i, frame in enumerate(video.iter_frames(clip, w, h)):
        grey = torch.from_numpy(frame.astype(np.float32)).mean(-1) / 255.0
        spectra.append(torch.fft.rfft2((grey - grey.mean()) * window))
        if i >= span:
            r = spectra[-1] * spectra[-1 - span].conj()
            corr = torch.fft.irfft2(r / r.abs().clamp(min=1e-6), s=(h, w))
            dy, dx = divmod(int(corr.argmax()), w)
            dy, dx = (dy - h if dy > h // 2 else dy), (dx - w if dx > w // 2 else dx)
            speed[i] = np.hypot(dx, dy) / w / (span / info.fps)
            spectra.pop(0)
    speed[:span] = speed[span] if info.n_frames > span else 0.0
    moving = _smoothstep(0.0, _MOTION_FULL, torch.from_numpy(speed)).numpy()
    return _ema(moving, keep=0.9)


def focus_maps(clip: Path, info: video.ClipInfo, depth: bool = True, segment: bool = True) -> Focus:
    """Finds the subject of each frame. Without `depth`, it's `_salience`'s
    guess. With it, that guess says roughly where the subject is and how far
    away. With `segment` too, the subject is the objects SAM finds that are
    the frame's focus (see `_segment`); without it, or if that finds none,
    it's what's near the guess at about its depth."""
    fw, fh = _grid(info.width, info.height, _FOCUS_SIZE)
    salient = _salience(clip, info)
    motion = _camera_motion(clip, info)
    if not depth:
        return Focus(salient, motion=motion)
    from scipy.ndimage import distance_transform_edt

    near = _depth(clip, info, fw, fh)
    guess = salient.astype(np.float32)
    # The subject's depth: toward the near end of where the guess is, as the
    # guess spills onto what's behind the subject, steadied over time.
    subject_depth = np.array([np.percentile(n[g >= 0.5], 75) if (g >= 0.5).any() else np.percentile(n, 75)
                              for n, g in zip(near, guess, strict=True)], dtype=np.float32)
    subject_depth = _ema(subject_depth, keep=0.8)
    defocus = np.abs(near - subject_depth[:, None, None])
    at_depth = 1.0 - _smoothstep(_DEPTH_TOLERANCE, 2 * _DEPTH_TOLERANCE, torch.from_numpy(defocus)).numpy()
    # Near the guess: the same depth far across the frame is the ground or a
    # wall the subject stands on, not the subject.
    reach = _SUBJECT_REACH * min(fw, fh)
    dist = np.stack([distance_transform_edt(g < 0.5) if (g >= 0.5).any() else np.full(g.shape, reach * 4)
                     for g in guess]).astype(np.float32)

    def within(r: float) -> np.ndarray:
        return 1.0 - _smoothstep(0.5 * r, r, torch.from_numpy(dist)).numpy()

    def soften(x: np.ndarray) -> np.ndarray:
        return _gauss(torch.from_numpy(x)[:, None], 1.0, 1.0)[:, 0].clamp(0.0, 1.0).numpy().astype(np.float16)

    # The guess only places it: it bleeds into what's around the subject.
    subject = at_depth * within(reach)
    objects = _segment(clip, info, guess, near, at_depth) if segment else None
    if objects is not None:
        # Where it finds nothing, there's no subject: depth alone would only
        # pick out the ground.
        subject = objects
        # And the subject's depth is the segmented object's.
        inside = [n[o > 0.5] for n, o in zip(near, subject, strict=True)]
        segmented_depth = np.array([np.median(x) if x.size > 4 else d for x, d in zip(inside, subject_depth, strict=True)],
                                   dtype=np.float32)
        subject_depth = _ema(segmented_depth, keep=0.8)
        defocus = np.abs(near - subject_depth[:, None, None])
    mask = soften(subject)
    return Focus(mask, defocus.astype(np.float16), near.astype(np.float16), subject_depth.astype(np.float32), motion)


def blur_levels(focus: Focus, rect: Rect, width: int, height: int, look: AnamorphicLook) -> torch.Tensor:
    """(n, 1, gh, gw) blur level over the `width`x`height` canvas for each of
    `focus`'s frames, on a small grid (see the module docstring)."""
    from scipy.ndimage import distance_transform_edt

    gw, gh = _grid(width, height, _LEVEL_SIZE)
    s = gw / width
    x0, y0, x1, y1 = rect
    gx0, gy0, gx1, gy1 = round(x0 * s), round(y0 * s), max(round(x0 * s) + 1, round(x1 * s)), max(round(y0 * s) + 1, round(y1 * s))
    short = min(gw, gh)

    # Distance from the subject, averaged over a few thresholds so it moves
    # smoothly as the subject's outline does.
    n = len(focus)

    def place(x: np.ndarray, carry: bool = False) -> torch.Tensor:
        # `carry` carries the edge's values on outward, rather than 0.
        inner = F.interpolate(torch.from_numpy(x.astype(np.float32))[:, None], size=(gy1 - gy0, gx1 - gx0),
                              mode="bilinear", align_corners=False)
        return F.pad(inner, (gx0, gw - gx1, gy0, gh - gy1), mode="replicate" if carry else "constant")

    canvas = place(focus.mask)
    level = torch.zeros((n, 1, gh, gw))
    if focus.defocus is not None:
        # As a lens would: blurred by how far it is in depth from the subject.
        if look.depth_falloff > 0:
            # Carried on past the clip's edge: the new picture there is as far
            # off as the clip's own edge, so it's blurred at least as much.
            level = config.ANAMORPHIC_BACKGROUND * _smoothstep(0.0, look.depth_falloff, place(focus.defocus, carry=True))
    elif look.falloff > 0:
        dist = np.zeros((n, gh, gw), dtype=np.float32)
        m = canvas[:, 0].numpy()
        for i in range(n):
            for t in (0.25, 0.5, 0.75):
                outside = m[i] < t
                dist[i] += distance_transform_edt(outside) if not outside.all() else short
        dist /= 3 * short
        level = config.ANAMORPHIC_BACKGROUND * _smoothstep(0.0, look.falloff, torch.from_numpy(dist))[:, None]

    # Up to ANAMORPHIC_EDGE at the clip's edges that border new picture, and from
    # there to ANAMORPHIC_FAR beyond them, toward the far corners.
    xs = (torch.arange(gw) + 0.5) / s
    ys = (torch.arange(gh) + 0.5) / s
    feather = max(1.0, look.feather * min(width, height))

    def edge(coord: torch.Tensor, lo: int, hi: int, extent: int) -> torch.Tensor:
        d = torch.full_like(coord, float("inf"))
        if lo > 0:
            d = torch.minimum(d, coord - lo)
        if hi < extent:
            d = torch.minimum(d, hi - coord)
        return d

    inside = torch.minimum(edge(ys, y0, y1, height)[:, None], edge(xs, x0, x1, width)[None, :])
    out_x = torch.clamp(torch.maximum(x0 - xs, xs - x1), min=0.0)
    out_y = torch.clamp(torch.maximum(y0 - ys, ys - y1), min=0.0)
    beyond = (out_y[:, None] ** 2 + out_x[None, :] ** 2).sqrt()
    in_rect = beyond == 0
    rim = 1.0 - _smoothstep(0.0, feather, inside.clamp(min=0.0))
    # The subject is spared the rim, but for the last stretch before the
    # edge, so one that's cut off by the frame still softens into the fill.
    tight = 1.0 - _smoothstep(0.0, config.ANAMORPHIC_SUBJECT_EDGE * min(width, height), inside.clamp(min=0.0))
    rim = rim * (1.0 - canvas * (1.0 - tight))
    edge_level = config.ANAMORPHIC_EDGE
    # Eased, so it carries on from the rim without a kink and builds slowly.
    far = edge_level + (config.ANAMORPHIC_FAR - edge_level) * _smoothstep(0.0, 1.0, beyond / beyond.max().clamp(min=1.0))
    # A soft union, so there's no crease where the rim overtakes the
    # background; and the same outside, so the blur carries straight on
    # across the clip's edge (rim is 1 there, and `far` the edge's level).
    return torch.where(in_rect, 1.0 - (1.0 - level) * (1.0 - edge_level * rim), 1.0 - (1.0 - level) * (1.0 - far))


def _beyond(rect: Rect, width: int, height: int, device: torch.device) -> torch.Tensor:
    """(height, width): how far each pixel is outside `rect` (0 inside)."""
    x0, y0, x1, y1 = rect
    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    return torch.hypot(((x0 - xs).clamp(min=0) + (xs - x1).clamp(min=0))[None, :],
                       ((y0 - ys).clamp(min=0) + (ys - y1).clamp(min=0))[:, None])


def _smear(x: torch.Tensor, rect: Rect, sigma: float) -> torch.Tensor:
    """`x` (n, c, h, w, the clip at `rect` mirrored outward) smeared outward
    from the clip's centre, as by a zoom blur, then blurred `sigma` wide and
    a quarter as tall: the ground the pieces of the fill sit on, carrying
    the clip's colours out across the gaps between them. It's `x` as it is
    at the clip's edge, so the clip fades into it without a seam, and fully
    smeared from `sigma` / 2 beyond it."""
    n, _, h, w = x.shape
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ys = torch.arange(h, device=x.device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=x.device, dtype=torch.float32) + 0.5
    dy, dx = torch.meshgrid(ys - cy, xs - cx, indexing="ij")
    out = torch.zeros_like(x)
    for k in range(_SMEAR_SAMPLES):
        scale = 1.0 - (1.0 - _SMEAR_REACH) * k / max(1, _SMEAR_SAMPLES - 1)
        grid = torch.stack([(cx + scale * dx) / w * 2 - 1, (cy + scale * dy) / h * 2 - 1], dim=-1)
        out += F.grid_sample(x, grid.expand(n, -1, -1, -1), mode="bilinear", padding_mode="reflection",
                             align_corners=False)
    t = _smoothstep(0.0, sigma / 2, _beyond(rect, w, h, x.device))
    return (1 - t) * x + t * _blur(out / _SMEAR_SAMPLES, sigma, sigma / 4)


def _edge_alpha(rect: Rect, width: int, height: int, band: float) -> np.ndarray:
    """(rh, rw) weight of the clip over the fill: 1 inside, easing to 0 at
    each edge that borders new picture over `band` px."""
    x0, y0, x1, y1 = rect

    def ramp(n: int, lo_open: bool, hi_open: bool) -> np.ndarray:
        d = np.arange(n, dtype=np.float32) + 0.5
        a = np.ones(n, dtype=np.float32)
        if lo_open:
            a = np.minimum(a, d / band)
        if hi_open:
            a = np.minimum(a, d[::-1] / band)
        a = np.clip(a, 0.0, 1.0)
        return a * a * (3 - 2 * a)

    return np.minimum(ramp(y1 - y0, y0 > 0, y1 < height)[:, None], ramp(x1 - x0, x0 > 0, x1 < width)[None, :])


def _deform(x: torch.Tensor, rect: Rect, width: int, height: int, amount: torch.Tensor, still: torch.Tensor,
            times: torch.Tensor, wobble: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`x` (n, c, height, width, colour first) bent as the edges of an
    anamorphic lens bend: stretched outward from the clip's centre, mostly
    sideways, swirled round it, with red and blue drawn apart, and drifting
    slowly at `times` (n,) seconds, `wobble` (n,) as much in each frame.
    It comes in softly over an ellipse round the clip's centre (see
    _DEFORM_INNER), so the middle stays put and the clip's own corners bend
    a little, times `amount` (n, 1, height, width), and never where `still`
    (the subject) is. Returns it and how bent each
    pixel is (0-1)."""
    x0, y0, x1, y1 = rect
    dev = x.device
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ys = torch.arange(height, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=dev, dtype=torch.float32) + 0.5
    dx, dy = xs[None, :] - cx, ys[:, None] - cy
    r = torch.hypot(dx / ((x1 - x0) / 2), dy / ((y1 - y0) / 2))
    t = _smoothstep(_DEFORM_INNER, max(float(r.max()), _DEFORM_INNER + 1e-3), r)[None, None] * amount * (1 - still)
    # Sampling from nearer the centre stretches outward; turning the sample
    # point the other way swirls the picture round.
    angle = -_DEFORM_SWIRL * t
    cos, sin = torch.cos(angle), torch.sin(angle)
    rx, ry = cos * dx - sin * dy, sin * dx + cos * dy
    # The drift: a few slow waves, travelling across and down the frame.
    drift_x, drift_y = torch.zeros_like(t), torch.zeros_like(t)
    if bool((wobble > 0).any()):
        sec = times.to(dev).view(-1, 1, 1, 1)
        for period, waves, phase in _WOBBLE_WAVES:
            drift_x = drift_x + torch.sin(2 * torch.pi * (sec / period + waves * ys[:, None] / height) + phase)
            drift_y = drift_y + torch.sin(2 * torch.pi * (sec / (1.3 * period) + waves * xs[None, :] / width) + phase + 1.0)
        reach = wobble.to(dev).view(-1, 1, 1, 1) * _WOBBLE * min(x1 - x0, y1 - y0) * t / len(_WOBBLE_WAVES)
        drift_x, drift_y = drift_x * reach, drift_y * reach

    def sample(src: torch.Tensor, fringe: float) -> torch.Tensor:
        sx = cx + (1.0 - (_DEFORM_STRETCH_X + fringe) * t) * rx + drift_x
        sy = cy + (1.0 - (_DEFORM_STRETCH_Y + fringe) * t) * ry + drift_y
        grid = torch.stack([sx[:, 0] / width * 2 - 1, sy[:, 0] / height * 2 - 1], dim=-1)
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    rest = sample(x, 0.0)
    red, blue = sample(x[:, :1], _FRINGE), sample(x[:, 2:3], -_FRINGE)
    return torch.cat([red, rest[:, 1:2], blue, rest[:, 3:]], dim=1), t


def surround(frames: np.ndarray, rect: Rect, width: int, height: int) -> np.ndarray:
    """(n, height, width, 3): `frames` (n, rh, rw, 3, 0-1, at `rect`'s size)
    mirrored outward, smeared and blurred into soft colour, to pad VACE's
    frame with instead of white: no hard border for it to copy, just the
    clip's colours carrying on."""
    x0, y0, x1, y1 = rect
    x = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)).permute(0, 3, 1, 2)
    iy, ix = _mirror_index(height, y0, y1 - y0, x.device), _mirror_index(width, x0, x1 - x0, x.device)
    sigma = 0.25 * min(x1 - x0, y1 - y0)
    soft = _blur(_smear(x[:, :, iy][:, :, :, ix], rect, sigma), sigma / 2, sigma / 2)
    return soft.clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()


def _mirror_index(n_out: int, lo: int, size: int, device: torch.device) -> torch.Tensor:
    i = (torch.arange(n_out, device=device) - lo).remainder(2 * size)
    return torch.where(i < size, i, 2 * size - 1 - i)


def render(frames: np.ndarray, focus: Focus, levels: torch.Tensor, rect: Rect, width: int, height: int,
           look: AnamorphicLook, base: np.ndarray, seam: float, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`frames` (n, rh, rw, 3, 0-1, at `rect`'s size, frame `start` on) on a
    `width`x`height` canvas, laid over `base` (n, height, width, 3, 0-1: the
    new edges, scaled up) and feathered into it across `seam` px; and blurred
    to `levels` (from `blur_levels` for the same `focus`), with the anamorphic
    grade and streaks coming in with the blur. Everything but the unblurred
    picture is blurred from a plate with the subject filled in from around it,
    so the subject doesn't smear into the background.
    Returns the (n, height, width, 3) canvases and the (n, height, width)
    blur level relative to the clip's edge, clamped to [0, 1]: how far each
    pixel is from the clip as it was, for anything blending another picture
    in.

    Now and then MPS hands back a batch of garbage, with no error. Anywhere
    left all but unblurred must come out as it went in, so a batch where it
    doesn't is done again, and then on the CPU."""
    for dev in (_device(), _device(), torch.device("cpu")):
        out, g, ok = _render(frames, focus, levels, rect, width, height, look, base, seam, start, dev)
        if ok:
            break
        print(f"  warning: frames {start}-{start + len(frames) - 1} came back corrupted on {dev.type}; redoing them")
    return out, g


def _render(frames: np.ndarray, focus: Focus, levels: torch.Tensor, rect: Rect, width: int, height: int,
            look: AnamorphicLook, base: np.ndarray, seam: float, start: int,
            dev: torch.device) -> tuple[np.ndarray, np.ndarray, bool]:
    x0, y0, x1, y1 = rect
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(dev).permute(0, 3, 1, 2)
    m = torch.from_numpy(focus.mask.astype(np.float32)).to(dev)[:, None]
    m = F.interpolate(m, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
    if focus.near is not None:
        # Depth goes along as one more channel, so the new edges get the
        # depth at the clip's edge.
        near = F.interpolate(torch.from_numpy(focus.near.astype(np.float32)).to(dev)[:, None],
                             size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
        x = torch.cat([x, near], dim=1)
    # Scaled to the clip, not the output, so a wider output isn't blurred more.
    edge_sigma = max(2.0, _EDGE_SIGMA * min(x1 - x0, y1 - y0) * look.strength)
    # The new edges are the base's, the clip laid back over them and
    # feathered across the band the base redrew into it.
    pad = (x0, width - x1, y0, height - y1)
    generated = torch.from_numpy(np.ascontiguousarray(base)).to(dev).permute(0, 3, 1, 2)
    # The generated picture is held from a keyframe, and the footage's
    # light and colour drift from it: match it to each frame, broadly,
    # by how the two differ along the clip's edge, carried on out into
    # the new picture and fading away from the clip.
    band = torch.zeros((1, 1, height, width), device=dev)
    band[:, :, y0:y1, x0:x1] = 1.0 - torch.from_numpy(_edge_alpha(rect, width, height, 2 * seam)).to(dev)
    reach = 2 * max(seam, 8.0)
    differs = _blur((F.pad(x[:, :3], pad) - generated) * band, reach, reach)
    weight = _blur(band, reach, reach)
    fade = 1.0 - _smoothstep(0.0, 3 * reach, _beyond(rect, width, height, dev))
    generated = generated + differs / weight.clamp(min=1e-3) * fade * (weight > 1e-3)
    if focus.near is not None:
        # Only for how much each part bends: the depth at the clip's edge, carried on outward.
        generated = torch.cat([generated, F.pad(x[:, 3:], pad, mode="replicate")], dim=1)
    clip_alpha = torch.zeros((1, 1, height, width), device=dev)
    clip_alpha[:, :, y0:y1, x0:x1] = torch.from_numpy(_edge_alpha(rect, width, height, max(seam, 1.0))).to(dev)
    canvas = clip_alpha * F.pad(x, pad) + (1.0 - clip_alpha) * generated
    m = F.pad(m, pad)
    if focus.near is not None:
        canvas, near, x = canvas[:, :3], canvas[:, 3:], x[:, :3]
    bent = torch.zeros_like(canvas[:, :1])
    if look.deform > 0:
        # How out of focus each part is, by its depth: the further from the
        # subject's, the more it bends.
        if focus.near is not None and focus.depth is not None:
            subject = torch.from_numpy(focus.depth.astype(np.float32)).to(dev).view(-1, 1, 1, 1)
            out_of_focus = _smoothstep(0.0, max(look.depth_falloff, 1e-3), (near - subject).abs())
        else:
            out_of_focus = torch.ones_like(canvas[:, :1])
        times = (start + torch.arange(canvas.shape[0], device=dev, dtype=torch.float32)) / look.fps
        # The edges drift with the camera: a held shot not at all, a pan or a shaky hand some.
        moving = (torch.ones(canvas.shape[0]) if focus.motion is None
                  else torch.from_numpy(focus.motion.astype(np.float32)))
        wobble = look.wobble * moving
        warped, bent = _deform(torch.cat([canvas, m], dim=1), rect, width, height,
                               look.deform * (0.35 + 0.65 * out_of_focus), m, times, wobble)
        canvas, m = warped.split([3, 1], dim=1)
    level = F.interpolate(levels.to(dev), size=(height, width), mode="bilinear", align_corners=False)

    # The plate: the subject filled with the blurred picture around it. But
    # where it's meant to soften (cut off by the clip's edge), it blurs as
    # itself: filled in there, its texture would cut out in a hard line.
    g = (level / config.ANAMORPHIC_EDGE).clamp(0.0, 1.0)
    keep = 1.0 - m * (1.0 - g)
    around = _blur(canvas * keep, edge_sigma, edge_sigma) / _blur(keep, edge_sigma, edge_sigma).clamp(min=1e-3)
    plate = canvas * keep + around * (1.0 - keep)

    # Blend between a stack of blurs, so each pixel gets its own amount.
    sigma = edge_sigma * torch.where(level <= 1.0, level**2, level)
    # Half an octave apart: further apart, and fading between two very
    # different blurs shows as a band.
    stack = [0.0, *(edge_sigma * 2 ** (k / 2) for k in range(-8, 1 + max(0, round(2 * np.log2(config.ANAMORPHIC_FAR)))))]
    # Each pixel's position in the stack, then a weight on its two nearest.
    pos = sum(((sigma - lo) / (hi - lo)).clamp(0.0, 1.0) for lo, hi in itertools.pairwise(stack))
    out = torch.zeros_like(canvas)
    at_edge = None
    for k, s in enumerate(stack):
        w = (1.0 - (pos - k).abs()).clamp(min=0.0)
        if s == edge_sigma or w.any():
            blurred = canvas if s == 0 else _blur(plate, s, s / config.ANAMORPHIC_SQUEEZE)
            out += w * blurred
            if s == edge_sigma:
                at_edge = blurred

    gray = out.mean(dim=1, keepdim=True)
    out = gray + (out - gray) * (1.0 + 0.25 * g)
    out = (out - 0.5) * (1.0 - 0.08 * g) + 0.5 + 0.02 * g

    if look.streak > 0:
        # Highlights smeared sideways and cooled, as off an anamorphic lens,
        # worked out at quarter size.
        qh, qw = max(4, height // 4), max(4, width // 4)
        hl = ((F.interpolate(at_edge, size=(qh, qw), mode="bilinear", antialias=True, align_corners=False) - 0.55) / 0.45).clamp(0.0, 1.0)
        hl = _gauss(hl, 1.5 * edge_sigma, max(0.25, edge_sigma / 48))
        hl = F.interpolate(hl, size=(height, width), mode="bilinear", align_corners=False)
        hl = hl * torch.tensor([0.75, 0.95, 1.2], device=dev).view(1, 3, 1, 1)
        out = 1.0 - (1.0 - out.clamp(0.0, 1.0)) * (1.0 - (look.streak * g * hl).clamp(0.0, 1.0))

    out = out.clamp(0.0, 1.0)
    # Barely blurred, unbent clip comes through all but untouched (the grade
    # and streaks come in with the blur), so it must still match the frames.
    sharp = ((level[:, :, y0:y1, x0:x1] < _UNBLURRED) & (bent[:, :, y0:y1, x0:x1] <= _BENT)).expand(-1, 3, -1, -1)
    ok = bool(not sharp.any() or (out[:, :, y0:y1, x0:x1] - x)[sharp].abs().mean() < _CORRUPT)
    # Only the new edges are the look's: the clip comes through as it was
    # but for a thin band along the edges that border them, and VACE is left
    # out of it (g = 0 there).
    keep = torch.zeros((1, 1, height, width), device=dev)
    keep[:, :, y0:y1, x0:x1] = torch.from_numpy(
        _edge_alpha(rect, width, height, max(1.0, _KEEP_BAND * min(x1 - x0, y1 - y0)))).to(dev)
    out = keep * F.pad(x, pad) + (1.0 - keep) * out
    g = g * (1.0 - keep)
    return out.permute(0, 2, 3, 1).cpu().numpy(), g[:, 0].cpu().numpy(), ok


def scaled_plates(plates: Sequence[np.ndarray], start: int, n: int, width: int, height: int,
                  pad: int = 0) -> np.ndarray:
    """(n + pad, height, width, 3, 0-1): `plates`' frames [start, start + n)
    scaled to `width`x`height`, then `pad` copies of the last."""
    up = video._scale_frames(np.stack(plates[start:start + n]).astype(np.float32) / 255.0, width, height)
    return np.concatenate([up, np.repeat(up[-1:], pad, axis=0)], axis=0) if pad else up


def composite(canvases: list[np.ndarray], plates: Sequence[np.ndarray], clip: Path, focus: Focus, rect: Rect,
              width: int, height: int, look: AnamorphicLook, seam: float, chunk: int = 8,
              first: int = 0) -> Iterable[np.ndarray]:
    """The output's frames. The anamorphic look over the clip and Flux's `plates`
    (every frame of the clip, indexed from 0; `seam` is how far they reach into
    the clip, at the output's size) at full quality, fading out as the blur
    comes in, where VACE was asked to restyle it: there, `canvases` (VACE's,
    uint8, any resolution, of the clip's frames from `first` on) scaled up
    show instead."""
    from tqdm import tqdm

    x0, y0, x1, y1 = rect
    frames = itertools.islice(video.iter_frames(clip, x1 - x0, y1 - y0, flags="lanczos"), first, None)
    with tqdm(total=len(canvases), unit="frame") as bar:
        for offset in range(0, len(canvases), chunk):
            start = first + offset
            batch = np.stack(canvases[offset:offset + chunk]).astype(np.float32) / 255.0
            n = batch.shape[0]
            up = video._scale_frames(batch, width, height)
            real = np.stack([f for _, f in zip(range(n), frames)]).astype(np.float32) / 255.0
            window = focus[start:start + n]
            base, g = render(real, window, blur_levels(window, rect, width, height, look), rect, width, height,
                             look, scaled_plates(plates, start, n, width, height), seam, start)
            yield from (1.0 - g[..., None]) * base + g[..., None] * up
            bar.update(n)



@dataclass
class PlateTrack:
    """The new edges FLUX.1 Fill made for a few keyframes (`keys`, each
    `plates[k]`, uint8 at working size, the whole canvas), crossfaded
    between them for the frames in between. Indexes like a list of
    canvases."""
    plates: list[np.ndarray]
    keys: list[int]
    n_frames: int

    def __len__(self) -> int:
        return self.n_frames

    def _frame(self, i: int) -> np.ndarray:
        k = max(0, bisect.bisect_right(self.keys, i) - 1)
        if k + 1 >= len(self.keys):
            return self.plates[k]
        t = (i - self.keys[k]) / (self.keys[k + 1] - self.keys[k])
        mixed = (1 - t) * self.plates[k].astype(np.float32) + t * self.plates[k + 1].astype(np.float32)
        return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)

    def __getitem__(self, frames: slice) -> list[np.ndarray]:
        return [self._frame(i) for i in range(*frames.indices(self.n_frames))]

    def anchor_weights(self, start: int, n: int) -> np.ndarray:
        """(n,) how free VACE is over frames [start, start + n): 0 at a
        keyframe, where its plate is held as an anchor, easing up to 1 midway
        between two of them, where VACE has to invent the picture and carry it
        from one anchor to the next. Frames past the last keyframe are
        anchored to it."""
        i = np.arange(start, start + n, dtype=np.float32)
        keys = np.asarray(self.keys, dtype=np.float32)
        k = np.clip(np.searchsorted(keys, i, side="right") - 1, 0, len(keys) - 1)
        before, after = keys[k], keys[np.minimum(k + 1, len(keys) - 1)]
        # How far to the nearer anchor, against the half-gap it sits in.
        t = np.minimum(i - before, after - i).clip(min=0.0) * 2.0 / np.maximum(after - before, 1.0)
        t = t.clip(0.0, 1.0)
        return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def flux_plates(clip: Path, info: video.ClipInfo, rect: Rect, width: int, height: int, prompt: str, steps: int,
                seed: int, until: int | None = None) -> tuple[PlateTrack, float]:
    """Outpaints a keyframe every `config.FLUX_KEY_SECONDS` of the clip to
    the whole `width`x`height` canvas with FLUX.1 Fill, at a working size
    near `config.FLUX_AREA`. Each is made on its own, from the clip smeared
    outward (see `surround`) and on the same seed, so nothing compounds from
    one to the next: chaining them flattened the new edges a little further
    each time, and over a long clip they washed out to flat grey altogether.
    They're left to disagree, and VACE carries the picture between them (see
    `PlateTrack.anchor_weights`). With `until`, stops at the first keyframe at
    or past that frame. Returns the plates and the seam to lay the clip back
    over them with, at the output's size.

    Each plate is kept in the work dir as a PNG (see `workdir`), so a later
    run reuses them, and one edited by hand is used as edited without
    disturbing any other."""
    from soto import comfy_engine, workdir

    x0, y0, x1, y1 = rect
    aspect = width / height
    fh = max(16, round((config.FLUX_AREA / aspect) ** 0.5 / 16) * 16)
    fw = max(16, round(fh * aspect / 16) * 16)
    s = fw / width
    wr = (round(x0 * s), round(y0 * fh / height), round(x1 * s), round(y1 * fh / height))
    rw, rh = wr[2] - wr[0], wr[3] - wr[1]
    every = max(1, round(info.fps * config.FLUX_KEY_SECONDS))
    keys = sorted({*range(0, info.n_frames, every), info.n_frames - 1})
    if until is not None:
        keys = keys[:next((n for n, k in enumerate(keys) if k >= until), len(keys) - 1) + 1]
    # New everywhere outside the clip, and a little way into it, softly.
    mask = np.ones((fh, fw), dtype=np.float32)
    mask[wr[1]:wr[3], wr[0]:wr[2]] = 1.0 - _edge_alpha(wr, fw, fh, config.FLUX_SEAM_PX)

    print(f"  {len(keys)} keyframes at {fw}x{fh}, {steps} steps each")
    plate_seed = seed % 2**32
    flux = None
    plates: list[np.ndarray] = []
    wanted = set(keys)
    for n, (i, frame) in enumerate((i, f) for i, f in enumerate(video.iter_frames(clip, rw, rh, flags="area"))
                                   if i in wanted):
        real = frame.astype(np.float32) / 255.0
        pixels = np.ascontiguousarray(surround(real[None], wr, fw, fh)[0])
        pixels[wr[1]:wr[3], wr[0]:wr[2]] = real
        k = workdir.key(kind="flux", model=config.FLUX_FILL_FILENAME, guidance=config.FLUX_GUIDANCE,
                        sampler=config.FLUX_SAMPLER, scheduler=config.FLUX_SCHEDULER, prompt=prompt, steps=steps,
                        pixels=pixels, mask=mask, seed=plate_seed, denoise=1.0)
        label = f"key-{n:03d}-frame{i:06d}"

        def make():
            nonlocal flux
            if flux is None:
                flux = comfy_engine.FluxFill(prompt, steps)
            print(f"  keyframe {n + 1}/{len(keys)} (frame {i})")
            out = flux.fill(torch.from_numpy(pixels), torch.from_numpy(mask), plate_seed, 1.0,
                            on_step=workdir.step_preview(label))
            # Inside, its redrawn seam is kept: that's what the clip fades into.
            return out.numpy()[:fh, :fw]

        plate, kept = workdir.cached_image(k, label, make)
        if kept:
            print(f"  [cached] keyframe {n + 1}/{len(keys)} (frame {i})")
        workdir.show(label, [plate])
        plates.append(plate)
    return PlateTrack(plates, keys, info.n_frames), config.FLUX_SEAM_PX / s
