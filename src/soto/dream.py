"""The dream fills for `soto expand`: the clip's subject stays sharp, and
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

Outside the clip, the new picture is quilted from patches of the clip's own
background (see `plan_quilt`), so it keeps real detail, and the patches are
cut from each frame in turn, so it moves with the footage. The blur is then
worked out per pixel as a "level": 0 is untouched, and it rises to
DREAM_EDGE at the clip's edge and on to DREAM_FAR at the output's corners."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Iterable
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
# Long side of the grid the quilt is planned on; each patch's side as a
# fraction of the clip's short side there; and how many of the best-matching
# sources (flipped or not) each patch is drawn from.
_QUILT_SIZE = 384
_PATCH_FRACTION = 0.25
_PATCH_CHOICES = 24
# How much of each patch overlaps its neighbours, crossfading into them: a
# wide overlap hides the edges between patches that differ a lot (bright
# sky against dark cloud). And how far the clip fades into the fill, in
# patch overlaps.
_PATCH_OVERLAP = 0.5
_CLIP_BLEND = 1.5
# Each quilt covers this long a stretch of the clip, matched over this many
# of its frames; and when the subject reaches nearly everywhere, the least
# subject-like fraction of sources are still allowed.
_QUILT_SECONDS = 1.0
_QUILT_SAMPLES = 3
_MIN_SOURCES = 0.15
# How much a difference in depth (0-1) counts against a patch, next to one
# in colour (0-1 per channel).
_DEPTH_WEIGHT = 1.5
# The bend at the far corners of the new edges, at --deform 1 and fully out
# of focus: how much nearer the centre they're drawn from (stretching them
# outward), and how far they're swirled round it, in radians (about 2 degrees).
_DEFORM_STRETCH = 0.035
_DEFORM_SWIRL = 0.035
# Checking for a corrupted batch: below this blur level a pixel's barely
# touched (well under a pixel of blur), so if those pixels are off from the
# frames by more than _CORRUPT on average, the batch is garbage.
_UNBLURRED = 0.02
_CORRUPT = 0.03
# How much more a source costs for each patch already taken from near it.
_REUSE_PENALTY = 0.6
# And how much more one from right beside where it goes costs, up to this
# many patch widths away.
_NEIGHBOUR_PENALTY = 0.8
_NEIGHBOUR_REACH = 1.5


@dataclass
class DreamLook:
    strength: float  # blur at the clip's edge, x 5% of the output width
    feather: float  # how far in from the clip's edges the blur reaches it, fraction of the short side
    falloff: float  # how far from the subject the background takes to blur, fraction of the short side
    streak: float  # opacity of the horizontal light streaks
    depth_falloff: float = 0.3  # with depth: how far in depth from the subject the background takes to blur
    deform: float = 1.0  # how much the new edges bend, as through an anamorphic lens (1 = default)


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
    found without depth. `avoid` is a slightly wider `mask`, that patches of
    the fill are never cut from."""
    mask: np.ndarray
    defocus: np.ndarray | None = None
    near: np.ndarray | None = None
    avoid: np.ndarray | None = None
    depth: np.ndarray | None = None  # (n,) the subject's own `near`

    def __len__(self) -> int:
        return self.mask.shape[0]

    def _map(self, f) -> Focus:
        return Focus(f(self.mask), *(None if x is None else f(x)
                                     for x in (self.defocus, self.near, self.avoid, self.depth)))

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


def focus_maps(clip: Path, info: video.ClipInfo, depth: bool = True, segment: bool = True) -> Focus:
    """Finds the subject of each frame. Without `depth`, it's `_salience`'s
    guess. With it, that guess says roughly where the subject is and how far
    away. With `segment` too, the subject is the objects SAM finds that are
    the frame's focus (see `_segment`); without it, or if that finds none,
    it's what's near the guess at about its depth."""
    fw, fh = _grid(info.width, info.height, _FOCUS_SIZE)
    salient = _salience(clip, info)
    if not depth:
        return Focus(salient)
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
    # Patches stay off the subject and a little way round it.
    avoid = soften(F.max_pool2d(torch.from_numpy(subject)[:, None], 5, stride=1, padding=2)[:, 0].numpy())
    return Focus(mask, defocus.astype(np.float16), near.astype(np.float16), avoid, subject_depth.astype(np.float32))


def blur_levels(focus: Focus, rect: Rect, width: int, height: int, look: DreamLook) -> torch.Tensor:
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

    def place(x: np.ndarray) -> torch.Tensor:
        out = torch.zeros((n, 1, gh, gw))
        out[:, :, gy0:gy1, gx0:gx1] = F.interpolate(torch.from_numpy(x.astype(np.float32))[:, None],
                                                     size=(gy1 - gy0, gx1 - gx0), mode="bilinear",
                                                     align_corners=False)
        return out

    canvas = place(focus.mask)
    level = torch.zeros((n, 1, gh, gw))
    if focus.defocus is not None:
        # As a lens would: blurred by how far it is in depth from the subject.
        if look.depth_falloff > 0:
            level = config.DREAM_BACKGROUND * _smoothstep(0.0, look.depth_falloff, place(focus.defocus))
    elif look.falloff > 0:
        dist = np.zeros((n, gh, gw), dtype=np.float32)
        m = canvas[:, 0].numpy()
        for i in range(n):
            for t in (0.25, 0.5, 0.75):
                outside = m[i] < t
                dist[i] += distance_transform_edt(outside) if not outside.all() else short
        dist /= 3 * short
        level = config.DREAM_BACKGROUND * _smoothstep(0.0, look.falloff, torch.from_numpy(dist))[:, None]

    # Up to DREAM_EDGE at the clip's edges that border new picture, and from
    # there to DREAM_FAR beyond them, toward the far corners.
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
    tight = 1.0 - _smoothstep(0.0, config.DREAM_SUBJECT_EDGE * min(width, height), inside.clamp(min=0.0))
    rim = rim * (1.0 - canvas * (1.0 - tight))
    edge_level = config.DREAM_EDGE
    far = edge_level + (config.DREAM_FAR - edge_level) * beyond / beyond.max().clamp(min=1.0)
    # A soft union, so there's no crease where the rim overtakes the background.
    return torch.where(in_rect, 1.0 - (1.0 - level) * (1.0 - edge_level * rim), far)


@dataclass
class Quilt:
    """Where each patch of the new edges comes from, on a small grid of the
    canvas (`grid` = (gw, gh), the clip at `rect` in it): (ty, tx) on the
    canvas, (sy, sx) in the clip, both top-left, `size` square, and whether
    it's flipped left to right."""
    grid: tuple[int, int]
    rect: Rect
    size: int
    overlap: int
    patches: list[tuple[int, int, int, int, bool]]


def _axis_positions(lo: int, hi: int, n: int, size: int, overlap: int) -> list[int]:
    """Patch starts along one axis of the canvas: stepping outward from the
    clip's edges, so the first patch past each edge overlaps the clip by
    `overlap`, and across the clip for the bands beside it."""
    step = size - overlap
    pos = set()
    if lo > 0:
        y = lo + overlap - size
        while True:
            pos.add(max(y, 0))
            if y <= 0:
                break
            y -= step
    y = lo
    while y + size < hi:
        pos.add(y)
        y += step
    pos.add(max(lo, hi - size))
    if hi < n:
        y = hi - overlap
        while True:
            pos.add(min(y, n - size))
            if y + size >= n:
                break
            y += step
    return sorted(p for p in pos if 0 <= p <= n - size)


@dataclass
class QuiltTrack:
    """A quilt per stretch of the clip: `plans[k]` covers the frames from
    `starts[k]`, crossfading into the next over `fade` frames either side of
    each boundary."""
    plans: list[Quilt]
    starts: list[int]
    fade: int

    def at(self, i: int) -> list[tuple[Quilt, float]]:
        """The quilts frame `i` is made from, and their weights."""
        k = max(0, bisect.bisect_right(self.starts, i) - 1)
        if k + 1 < len(self.plans) and self.starts[k + 1] - i <= self.fade:
            t = 0.5 - (self.starts[k + 1] - i) / (2 * self.fade)
            return [(self.plans[k], 1.0 - t), (self.plans[k + 1], t)]
        if k > 0 and i - self.starts[k] < self.fade:
            t = 0.5 + (i - self.starts[k]) / (2 * self.fade)
            return [(self.plans[k - 1], 1.0 - t), (self.plans[k], t)]
        return [(self.plans[k], 1.0)]


def plan_quilt(clip: Path, info: video.ClipInfo, focus: Focus, rect: Rect, width: int, height: int,
               seed: int) -> QuiltTrack:
    """Tiles everything outside `rect` with patches of the clip's background,
    the way content-aware fill used to: outward from the clip's edges, each
    patch one whose overlap with what's already there (the clip, or the
    patches before it) matches well, and never from anywhere the subject is.

    The footage changes as it goes, so each `_QUILT_SECONDS` stretch gets its
    own quilt, matched over a few of its frames, and crossfades into the
    next. A patch keeps the previous stretch's source while that still
    matches well, so footage that holds steady keeps one quilt throughout.

    So the fill doesn't read as a repeating pattern: the patches sit on a
    jittered lattice, any of them can be flipped left to right, a source
    gets more expensive each time it's used, and each patch is drawn at
    random from the good matches, weighted toward the best."""
    gw, gh = _grid(width, height, _QUILT_SIZE)
    s = gw / width
    x0, y0, x1, y1 = rect
    gx0, gy0 = round(x0 * s), round(y0 * s)
    rgw, rgh = max(8, round(x1 * s) - gx0), max(8, round(y1 * s) - gy0)
    gx1, gy1 = min(gw, gx0 + rgw), min(gh, gy0 + rgh)
    rgw, rgh = gx1 - gx0, gy1 - gy0
    size = int(np.clip(round(min(rgw, rgh) * _PATCH_FRACTION), 8, min(rgw, rgh)))
    overlap = max(2, round(size * _PATCH_OVERLAP))
    rng = np.random.default_rng(seed)

    # One layout of patches for the whole clip, so stretches can share sources.
    known = torch.zeros((gh, gw))
    known[gy0:gy1, gx0:gx1] = 1.0
    ys = _axis_positions(gy0, gy1, gh, size, overlap)
    xs = _axis_positions(gx0, gx1, gw, size, overlap)
    jitter = max(1, overlap // 3)

    def nudge(p: int, lo: int, hi: int, n: int) -> int:
        # Patches that straddle the clip's edge stay put, so they keep their overlap with it.
        if p in (lo + overlap - size, hi - overlap) or lo <= p < hi:
            return p
        return int(np.clip(p + rng.integers(-jitter, jitter + 1), 0, n - size))

    targets = sorted({(nudge(ty, gy0, gy1, gh), nudge(tx, gx0, gx1, gw)) for ty in ys for tx in xs})
    targets = [t for t in targets if (known[t[0]:t[0] + size, t[1]:t[1] + size] < 1).any()]
    # The jitter can open gaps: fill any with the unjittered lattice.
    covered = known.clone()
    for ty, tx in targets:
        covered[ty:ty + size, tx:tx + size] = 1.0
    targets += [(ty, tx) for ty in ys for tx in xs
                if (covered[ty:ty + size, tx:tx + size] < 1).any() and (ty, tx) not in targets]

    def away(t: tuple[int, int]) -> float:
        cy, cx = t[0] + size / 2, t[1] + size / 2
        return float(np.hypot(max(gy0 - cy, 0, cy - gy1), max(gx0 - cx, 0, cx - gx1)))

    targets.sort(key=away)

    seg = max(1, round(info.fps * _QUILT_SECONDS))
    starts = list(range(0, info.n_frames, seg))
    if len(starts) > 1 and info.n_frames - starts[-1] < seg // 2:
        starts.pop()
    bounds = [*starts, info.n_frames]
    picks = [np.linspace(a, b - 1, _QUILT_SAMPLES).round().astype(int).tolist() for a, b in itertools.pairwise(bounds)]
    wanted = {i for p in picks for i in p}
    frames = {i: f for i, f in enumerate(video.iter_frames(clip, rgw, rgh)) if i in wanted}

    plans: list[Quilt] = []
    prev: dict[tuple[int, int], tuple[int, int, bool]] = {}
    for (a, b), pick in zip(itertools.pairwise(bounds), picks, strict=True):
        refs = sorted(set(pick))
        src = torch.from_numpy(np.stack([frames[i] for i in refs])).permute(0, 3, 1, 2).float() / 255.0
        if focus.near is not None:
            # Depth as one more channel to match on, so patches carry on from
            # what's at the same depth: near ground from near ground, sky from sky.
            near = torch.from_numpy(focus[refs].near.astype(np.float32))[:, None]
            near = F.interpolate(near, size=(rgh, rgw), mode="bilinear", align_corners=False)
            src = torch.cat([src, _DEPTH_WEIGHT * near], dim=1)
        stretch = focus[a:b]
        avoid = stretch.mask if stretch.avoid is None else stretch.avoid
        patches = _plan_segment(src, avoid, (gw, gh), (gx0, gy0, gx1, gy1), size, targets, rng, prev)
        prev = {(ty, tx): (sy, sx, flip) for ty, tx, sy, sx, flip in patches}
        plans.append(Quilt(grid=(gw, gh), rect=(gx0, gy0, gx1, gy1), size=size, overlap=overlap, patches=patches))
    return QuiltTrack(plans=plans, starts=starts, fade=max(1, seg // 3))


def _plan_segment(src: torch.Tensor, focus: np.ndarray, grid: tuple[int, int], rect: Rect, size: int,
                  targets: list[tuple[int, int]], rng: np.random.Generator,
                  prev: dict[tuple[int, int], tuple[int, int, bool]]) -> list[tuple[int, int, int, int, bool]]:
    """One stretch's quilt (see plan_quilt): `src` are a few of its frames at
    the clip's size on the grid (colour, and maybe depth), `focus` all of
    its subject masks."""
    gw, gh = grid
    gx0, gy0, gx1, gy1 = rect
    n_ref, channels, rgh, rgw = src.shape
    img = torch.zeros((n_ref, channels, gh, gw))
    img[:, :, gy0:gy1, gx0:gx1] = src
    known = torch.zeros((gh, gw))
    known[gy0:gy1, gx0:gx1] = 1.0

    # Sources the subject touches at all in this stretch are out, unless
    # that's nearly all of them: then only the least subject-like are in.
    reach = F.interpolate(torch.from_numpy(focus.astype(np.float32)).amax(dim=0)[None, None],
                          size=(rgh, rgw), mode="bilinear", align_corners=False)
    subject = F.max_pool2d(reach, size, stride=1)[0, 0] + F.avg_pool2d(reach, size, stride=1)[0, 0]
    banned = subject > 0.25
    if (~banned).float().mean() < _MIN_SOURCES:
        banned = subject > torch.quantile(subject.flatten(), _MIN_SOURCES)

    # The matching runs on the GPU: it's a few big convolutions per patch.
    dev = _device()
    flat = src.reshape(1, n_ref * channels, rgh, rgw).to(dev)
    flat_sq = (flat**2).sum(dim=1, keepdim=True)
    flat_flip, flat_sq_flip = flat.flip(-1), flat_sq.flip(-1)
    banned = banned.to(dev)
    used = torch.zeros_like(subject, device=dev)
    # How much of the subject is at each point of the canvas, so far: where a
    # patch carries on from the subject (one cut off by the clip's edge), it
    # may come from the subject too, so the subject carries on past the edge.
    subjectness = torch.zeros((gh, gw))
    subjectness[gy0:gy1, gx0:gx1] = reach[0, 0]
    # Where each source sits on the canvas, to keep patches from being cut
    # from right beside where they go: that just repeats what's at the edge.
    src_y = (torch.arange(subject.shape[0], device=dev) + gy0).view(-1, 1).float()
    src_x = (torch.arange(subject.shape[1], device=dev) + gx0).view(1, -1).float()

    patches = []
    for ty, tx in targets:
        m = known[ty:ty + size, tx:tx + size]
        t = img[:, :, ty:ty + size, tx:tx + size]
        if m.sum() < 1:
            costs = torch.from_numpy(rng.random((2, *subject.shape), dtype=np.float32)).to(dev)
        else:
            # Sum of squared differences over the known overlap, at every
            # source position at once: sum(M*C^2) - 2*sum(M*T*C) + sum(M*T^2).
            # The flipped costs are flipped back, so both are indexed by the
            # source's left edge in the clip.
            mk = m.view(1, 1, size, size).to(dev)
            mt = (m * t).reshape(1, n_ref * channels, size, size).to(dev)
            costs = torch.stack([
                (F.conv2d(flat_sq, mk) - 2 * F.conv2d(flat, mt))[0, 0],
                (F.conv2d(flat_sq_flip, mk) - 2 * F.conv2d(flat_flip, mt))[0, 0].flip(-1),
            ])
            costs = ((costs + float((m * t**2).sum())) / float(m.sum() * n_ref * channels)).clamp(min=0.0)
        beside = (1.0 - torch.hypot(src_y - ty, src_x - tx) / (_NEIGHBOUR_REACH * size)).clamp(min=0.0)
        costs = costs * (1.0 + _REUSE_PENALTY * used + _NEIGHBOUR_PENALTY * beside)
        continues_subject = m.sum() >= 1 and float((m * subjectness[ty:ty + size, tx:tx + size]).sum() / m.sum()) > 0.5
        if not continues_subject:
            costs = costs.masked_fill(banned, float("inf"))
        flat_costs = costs.flatten().cpu()
        order = torch.argsort(flat_costs)[:_PATCH_CHOICES]
        best = flat_costs[order[0]]
        last = prev.get((ty, tx))
        last_pick = None if last is None else int(last[2]) * costs.shape[1] * costs.shape[2] + last[0] * costs.shape[2] + last[1]
        if last_pick is not None and flat_costs[last_pick] <= best * 1.3 + 1e-3:
            # The last stretch's source still fits: keep it, so steady footage keeps its quilt.
            pick = last_pick
        else:
            # Drawn at random from the good matches, the closer the likelier.
            good = order[flat_costs[order] <= best * 1.5 + 1e-3]
            odds = torch.exp(-(flat_costs[good] - best) / (0.25 * best + 1e-3)).numpy().astype(np.float64)
            pick = int(good[rng.choice(len(good), p=odds / odds.sum())])
        flip, rest = divmod(pick, costs.shape[1] * costs.shape[2])
        sy, sx = divmod(rest, costs.shape[2])
        piece = src[:, :, sy:sy + size, sx:sx + size]
        if flip:
            piece = piece.flip(-1)
        img[:, :, ty:ty + size, tx:tx + size] = m * t + (1 - m) * piece
        taken = reach[0, 0, sy:sy + size, sx:sx + size]
        here = subjectness[ty:ty + size, tx:tx + size]
        subjectness[ty:ty + size, tx:tx + size] = m * here + (1 - m) * (taken.flip(-1) if flip else taken)
        known[ty:ty + size, tx:tx + size] = 1.0
        # Sources near this one cost more from now on.
        y_lo, x_lo = max(0, sy - size // 2), max(0, sx - size // 2)
        used[y_lo:sy + size // 2 + 1, x_lo:sx + size // 2 + 1] += 1.0
        patches.append((ty, tx, sy, sx, bool(flip)))
    return patches


def _apply_quilt(x: torch.Tensor, quilt: Quilt, rect: Rect, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, c, height, width) canvases tiled from `x` (n, c, rh, rw, the clip
    at `rect`'s size) by `quilt`, each patch crossfading into its
    neighbours over their overlap, and the (1, 1, height, width) weight of
    patches at each pixel (0 where none reached)."""
    x0, y0, x1, y1 = rect
    rh, rw = y1 - y0, x1 - x0
    gw, gh = quilt.grid
    s, sx_scale, sy_scale = width / gw, rw / (quilt.rect[2] - quilt.rect[0]), rh / (quilt.rect[3] - quilt.rect[1])
    size = int(np.ceil(quilt.size * s)) + 1
    ramp = torch.sin(torch.pi * (torch.arange(size, device=x.device) + 0.5) / size) ** 2
    window = (0.02 + ramp[:, None] * ramp[None, :])[None, None]
    acc = torch.zeros((x.shape[0], x.shape[1], height, width), device=x.device)
    weight = torch.zeros((1, 1, height, width), device=x.device)
    for ty, tx, sy, sx, flip in quilt.patches:
        Ty, Tx = round(ty * height / gh), round(tx * s)
        # The whole patch, so a flipped one maps the right columns even
        # where the canvas cuts it short.
        Sy, Sx = min(round(sy * sy_scale), max(0, rh - size)), min(round(sx * sx_scale), max(0, rw - size))
        piece = x[:, :, Sy:Sy + size, Sx:Sx + size]
        if flip:
            piece = piece.flip(-1)
        ph, pw = min(piece.shape[2], height - Ty), min(piece.shape[3], width - Tx)
        if ph <= 0 or pw <= 0:
            continue
        w = window[:, :, :ph, :pw]
        acc[:, :, Ty:Ty + ph, Tx:Tx + pw] += piece[:, :, :ph, :pw] * w
        weight[:, :, Ty:Ty + ph, Tx:Tx + pw] += w
    return acc / weight.clamp(min=1e-6), weight


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


def _deform(x: torch.Tensor, rect: Rect, width: int, height: int, amount: torch.Tensor) -> torch.Tensor:
    """`x` (n, c, height, width) with everything outside `rect` bent, as the
    edges of an anamorphic lens bend: stretched a little outward from the
    clip's centre and swirled a little around it. Nothing moves at the
    clip's edge, and it grows toward the output's corners, times `amount`
    (n, 1, height, width) there."""
    x0, y0, x1, y1 = rect
    dev = x.device
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ys = torch.arange(height, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=dev, dtype=torch.float32) + 0.5
    beyond = (torch.clamp(torch.maximum(y0 - ys, ys - y1), min=0.0)[:, None] ** 2
              + torch.clamp(torch.maximum(x0 - xs, xs - x1), min=0.0)[None, :] ** 2).sqrt()
    t = _smoothstep(0.0, 1.0, beyond / beyond.max().clamp(min=1.0))[None, None] * amount
    dx, dy = xs[None, :] - cx, ys[:, None] - cy
    # Sampling from nearer the centre stretches outward; turning the sample
    # point the other way swirls the picture round.
    scale = 1.0 - _DEFORM_STRETCH * t
    angle = -_DEFORM_SWIRL * t
    cos, sin = torch.cos(angle), torch.sin(angle)
    sx = cx + scale * (cos * dx - sin * dy)
    sy = cy + scale * (sin * dx + cos * dy)
    grid = torch.stack([sx[:, 0] / width * 2 - 1, sy[:, 0] / height * 2 - 1], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _mirror_index(n_out: int, lo: int, size: int, device: torch.device) -> torch.Tensor:
    i = (torch.arange(n_out, device=device) - lo).remainder(2 * size)
    return torch.where(i < size, i, 2 * size - 1 - i)


def render(frames: np.ndarray, focus: Focus, levels: torch.Tensor, rect: Rect, width: int, height: int,
           look: DreamLook, quilt: QuiltTrack | None = None, start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`frames` (n, rh, rw, 3, 0-1, at `rect`'s size, frame `start` on) on a
    `width`x`height` canvas, filled out with `quilt` (from `plan_quilt`, or
    mirrored outward without one), fading from the clip into it over the
    patches' overlap, and blurred to `levels` (from
    `blur_levels` for the same `focus`), with the dream grade and streaks
    coming in with the blur. Everything but the unblurred picture is blurred
    from a plate with the subject filled in from around it, so the subject
    doesn't smear into the background or turn up mirrored in the fill.
    Returns the (n, height, width, 3) canvases and the (n, height, width)
    blur level relative to the clip's edge, clamped to [0, 1]: how far each
    pixel is from the clip as it was, for anything blending another picture
    in.

    Now and then MPS hands back a batch of garbage, with no error. Anywhere
    left all but unblurred must come out as it went in, so a batch where it
    doesn't is done again, and then on the CPU."""
    for dev in (_device(), _device(), torch.device("cpu")):
        out, g, ok = _render(frames, focus, levels, rect, width, height, look, quilt, start, dev)
        if ok:
            break
        print(f"  warning: frames {start}-{start + len(frames) - 1} came back corrupted on {dev.type}; redoing them")
    return out, g


def _render(frames: np.ndarray, focus: Focus, levels: torch.Tensor, rect: Rect, width: int, height: int,
            look: DreamLook, quilt: QuiltTrack | None, start: int,
            dev: torch.device) -> tuple[np.ndarray, np.ndarray, bool]:
    x0, y0, x1, y1 = rect
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(dev).permute(0, 3, 1, 2)
    m = torch.from_numpy(focus.mask.astype(np.float32)).to(dev)[:, None]
    m = F.interpolate(m, size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
    iy, ix = _mirror_index(height, y0, y1 - y0, dev), _mirror_index(width, x0, x1 - x0, dev)
    canvas = x[:, :, iy][:, :, :, ix]
    if focus.near is not None:
        # Depth goes along as one more channel, so the new edges get the
        # depth of the patches they're made from.
        near = F.interpolate(torch.from_numpy(focus.near.astype(np.float32)).to(dev)[:, None],
                             size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
        x = torch.cat([x, near], dim=1)
        canvas = x[:, :, iy][:, :, :, ix]
    if quilt is None:
        m = m[:, :, iy][:, :, :, ix]
    else:
        # Patches never come from the subject, so there's none outside the clip.
        m = F.pad(m, (x0, width - x1, y0, height - y1))
        weights: dict[int, tuple[Quilt, torch.Tensor]] = {}
        for j in range(x.shape[0]):
            for q, w in quilt.at(start + j):
                weights.setdefault(id(q), (q, torch.zeros(x.shape[0], device=dev)))[1][j] += w
        tiles = torch.zeros_like(canvas)
        for q, w in weights.values():
            t, covered = _apply_quilt(x, q, rect, width, height)
            tiles += t * w.view(-1, 1, 1, 1)
        overlap = _CLIP_BLEND * quilt.plans[0].overlap * width / quilt.plans[0].grid[0]
        clip_alpha = torch.zeros((1, 1, height, width), device=dev)
        clip_alpha[:, :, y0:y1, x0:x1] = torch.from_numpy(
            _edge_alpha(rect, width, height, overlap)).to(dev)[None, None]
        clip_alpha = torch.where(covered > 0, clip_alpha, torch.ones_like(clip_alpha))
        filled = torch.where(covered > 0, tiles, canvas)
        canvas = clip_alpha * canvas + (1.0 - clip_alpha) * filled
    if focus.near is not None:
        canvas, near, x = canvas[:, :3], canvas[:, 3:], x[:, :3]
    if look.deform > 0:
        # How out of focus each part of the new edges is, by its depth: the
        # further from the subject's, the more it bends.
        if focus.near is not None and focus.depth is not None:
            subject = torch.from_numpy(focus.depth.astype(np.float32)).to(dev).view(-1, 1, 1, 1)
            out_of_focus = _smoothstep(0.0, max(look.depth_falloff, 1e-3), (near - subject).abs())
        else:
            out_of_focus = torch.ones_like(canvas[:, :1])
        canvas, m = _deform(torch.cat([canvas, m], dim=1), rect, width, height,
                            look.deform * (0.35 + 0.65 * out_of_focus)).split([3, 1], dim=1)
    level = F.interpolate(levels.to(dev), size=(height, width), mode="bilinear", align_corners=False)
    # Scaled to the clip, not the output, so a wider output isn't blurred more.
    edge_sigma = max(2.0, _EDGE_SIGMA * min(x1 - x0, y1 - y0) * look.strength)

    # The plate: the subject filled with the blurred picture around it.
    keep = 1.0 - m
    around = _blur(canvas * keep, edge_sigma, edge_sigma) / _blur(keep, edge_sigma, edge_sigma).clamp(min=1e-3)
    plate = canvas * keep + around * m

    # Blend between a stack of blurs, so each pixel gets its own amount.
    sigma = edge_sigma * torch.where(level <= 1.0, level**2, level)
    # Half an octave apart: further apart, and fading between two very
    # different blurs shows as a band.
    stack = [0.0, *(edge_sigma * 2 ** (k / 2) for k in range(-8, 1 + max(0, round(2 * np.log2(config.DREAM_FAR)))))]
    # Each pixel's position in the stack, then a weight on its two nearest.
    pos = sum(((sigma - lo) / (hi - lo)).clamp(0.0, 1.0) for lo, hi in itertools.pairwise(stack))
    out = torch.zeros_like(canvas)
    at_edge = None
    for k, s in enumerate(stack):
        w = (1.0 - (pos - k).abs()).clamp(min=0.0)
        if s == edge_sigma or w.any():
            blurred = canvas if s == 0 else _blur(plate, s, s / config.DREAM_ANAMORPHIC)
            out += w * blurred
            if s == edge_sigma:
                at_edge = blurred

    g = (level / config.DREAM_EDGE).clamp(0.0, 1.0)
    gray = out.mean(dim=1, keepdim=True)
    out = gray + (out - gray) * (1.0 + 0.25 * g)
    out = (out - 0.5) * (1.0 - 0.08 * g) + 0.5 + 0.02 * g

    if look.streak > 0:
        # Highlights smeared sideways and cooled, as off an anamorphic lens,
        # worked out at quarter size.
        qh, qw = max(4, height // 4), max(4, width // 4)
        hl = ((F.interpolate(at_edge, size=(qh, qw), mode="bilinear", antialias=True, align_corners=False) - 0.55) / 0.45).clamp(0.0, 1.0)
        hl = _gauss(hl, edge_sigma / 2, max(0.25, edge_sigma / 48))
        hl = F.interpolate(hl, size=(height, width), mode="bilinear", align_corners=False)
        hl = hl * torch.tensor([0.75, 0.95, 1.2], device=dev).view(1, 3, 1, 1)
        out = 1.0 - (1.0 - out.clamp(0.0, 1.0)) * (1.0 - (look.streak * g * hl).clamp(0.0, 1.0))

    out = out.clamp(0.0, 1.0)
    # Barely blurred clip comes through all but untouched (the grade and
    # streaks come in with the blur), so it must still match the frames.
    sharp = (level[:, :, y0:y1, x0:x1] < _UNBLURRED).expand(-1, 3, -1, -1)
    ok = bool(not sharp.any() or (out[:, :, y0:y1, x0:x1] - x)[sharp].abs().mean() < _CORRUPT)
    return out.permute(0, 2, 3, 1).cpu().numpy(), g[:, 0].cpu().numpy(), ok


def dream_frames(clip: Path, info: video.ClipInfo, focus: Focus, rect: Rect, width: int, height: int,
                 look: DreamLook, quilt: QuiltTrack | None, chunk: int = 8) -> Iterable[np.ndarray]:
    """The no-model dream fill: yields every output frame."""
    from tqdm import tqdm

    x0, y0, x1, y1 = rect
    frames = video.iter_frames(clip, x1 - x0, y1 - y0, flags="lanczos")
    start = 0
    with tqdm(total=info.n_frames, unit="frame") as bar:
        while batch := [f for _, f in zip(range(chunk), frames)]:
            real = np.stack(batch).astype(np.float32) / 255.0
            window = focus[start:start + len(batch)]
            yield from render(real, window, blur_levels(window, rect, width, height, look), rect, width, height,
                              look, quilt, start)[0]
            start += len(batch)
            bar.update(len(batch))


def composite(canvases: list[np.ndarray], clip: Path, focus: Focus, rect: Rect, width: int, height: int,
              look: DreamLook, quilt: QuiltTrack | None, chunk: int = 8) -> Iterable[np.ndarray]:
    """`--fill vace-dream`'s output frames: VACE's canvases (uint8, any
    resolution) scaled up, with the dream fill of the clip at full quality
    laid over them, fading out as the blur comes in, where VACE was asked to
    restyle it."""
    x0, y0, x1, y1 = rect
    frames = video.iter_frames(clip, x1 - x0, y1 - y0, flags="lanczos")
    for start in range(0, len(canvases), chunk):
        batch = np.stack(canvases[start:start + chunk]).astype(np.float32) / 255.0
        up = video._scale_frames(batch, width, height)
        real = np.stack([f for _, f in zip(range(batch.shape[0]), frames)]).astype(np.float32) / 255.0
        window = focus[start:start + batch.shape[0]]
        base, g = render(real, window, blur_levels(window, rect, width, height, look), rect, width, height, look,
                         quilt, start)
        yield from (1.0 - g[..., None]) * base + g[..., None] * up
