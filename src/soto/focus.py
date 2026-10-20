"""Finding the subject of each frame of a clip, for `soto expand`'s anamorphic
look: a depth model (Depth Anything V2 Small) and a cheap guess at where the
subject is (colour that stands out from the rest of the frame, plus local
motion and fine detail, with a pull toward the centre). The guess picks out
the subject, SAM or the depth map finds the rest of it, and how far
everything else is from it in depth is what the look blurs by. Without
depth, the guess alone is the subject.

These are slow, so they're kept in the work dir keyed on this module's own
source: changing the look (anamorphic.py) doesn't throw them away."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from soto import config, video

# Long side of the grid the subject is found on. It's smooth, so it's
# upsampled from there.
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
# The most of a frame that's ever taken for the subject, before it's rounded out.
_MAX_SUBJECT = 0.2
# How much the bent edges drift is set by how much the camera moves: the
# whole frame's shift over _MOTION_SPAN seconds, as a fraction of its width
# per second, measured on frames _MOTION_SIZE wide. A shot that holds still
# doesn't drift at all; one moving _MOTION_FULL or more drifts fully.
_MOTION_SIZE = 192
_MOTION_SPAN = 0.2
_MOTION_FULL = 0.1


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
