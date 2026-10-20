"""The anamorphic look for `soto expand`, as if one anamorphic lens had shot
the whole output frame: the clip's subject stays sharp, the rest of it is
defocused by how far it is in depth from the subject, and everything
softens, bends and fringes more the further it is off the lens's axis. The
clip sits in the middle of that lens, so the smaller a part of the output it
is, the less of the look reaches it.

How far off axis a pixel is, is its lens radius (see `lens_radius`): in
halves of the output's longer side, round the centre (see _LENS_VERTICAL),
so 1 is the middle of a landscape output's sides. The blur is worked
out per pixel as a "level" (see `blur_levels`): 0 is untouched. The field
softness builds steadily with the lens radius, from --feather to
ANAMORPHIC_FAR at the output's corners, and the depth blur
(up to ANAMORPHIC_BACKGROUND) comes from how far each part is from the
subject in depth (see `focus.py`). Over the new picture and the band where
the clip meets it, it's never under _SEAM_FLOOR, just enough to hide VACE's
picture being scaled up. Every part of it is smooth across the clip's
edges: a step there, split by the fringe, shows as a rainbow line.

The look goes over the whole canvas at the output's size, the clip and the
new picture alike (VACE's, scaled up, or FLUX.1 Fill's outpainting: see
`flux_plates` and `lay_over`)."""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from soto import config, video
from soto.focus import Focus, _device, _gauss, _grid, _smoothstep

Rect = tuple[int, int, int, int]

# Long side of the grid the blur level is worked out on. It's smooth, so
# it's upsampled from there.
_LEVEL_SIZE = 256
# The blur at a level of 1, as a fraction of a short side (5% of the width
# of a 16:9 frame): the clip's for its depth blur, as through the same lens
# whatever the output's size, and the output's for the lens's own softness.
_EDGE_SIGMA = 0.089
# The lens radius: how much the height counts against the width (1 = a
# circle; less squashes it, softening more toward the sides than up and down).
_LENS_VERTICAL = 1.0
# How the field softness builds, from --feather to ANAMORPHIC_FAR at the
# output's corners: as this power of the way there, so it comes in gently
# and builds steadily, with no shelf.
_FIELD_POWER = 1.5
# The bend comes in from this lens radius, easing up to full at the
# output's corners (so the cow's clip sides bend about 0.44 of full).
_LENS_INNER = 0.42
# The level is never under _SEAM_FLOOR over the new picture, just enough to
# hide VACE's picture being scaled up from working size, and along the
# clip's edges that border it, easing off over _SEAM_BAND of its short side
# inside it: as far in as blend.py fades the clip's own detail out.
_SEAM_FLOOR = 0.1
_SEAM_BAND = 0.12
# The smear that pads Flux's keyframes: how many samples it's averaged
# over, reaching how far toward the clip's centre.
_SMEAR_SAMPLES = 8
_SMEAR_REACH = 0.85
# The bend, at --deform 1 and fully out of focus, at the output's corners:
# how much nearer the centre each point is drawn from, across and down
# (stretching it outward, mostly sideways), how far it's swirled round, in
# radians (about 3.5 degrees), and how far red and blue are drawn apart.
_DEFORM_STRETCH_X = 0.12
_DEFORM_STRETCH_Y = 0.04
_DEFORM_SWIRL = 0.06
_FRINGE = 0.008
# The slow drift of the bent edges at --wobble 1, as a fraction of the
# clip's short side: a few waves, each (period in seconds, waves across the
# frame, phase). How much it drifts is set by how much the camera moves
# (see focus._camera_motion).
_WOBBLE = 0.01
_WOBBLE_WAVES = ((3.1, 1.3, 0.0), (4.7, 2.1, 1.9), (6.9, 0.7, 4.2))
# Above this much bend, a pixel of the clip has moved, so it's left out of
# the corruption check.
_BENT = 0.01
# Checking for a corrupted batch: below this blur level a pixel's barely
# touched (well under a pixel of blur), so if those pixels are off from the
# frames by more than _CORRUPT on average, the batch is garbage.
_UNBLURRED = 0.02
_CORRUPT = 0.03
# How much two neighbouring Flux keyframes disagree, place by place (mean
# difference, 0-1, blurred over this much of the short side), from where VACE
# is held to the anchor between them to where it's let free to bridge them:
# keyframes that agree crossfade cleanly, and VACE left free drifts from
# them (a grey sky turned blue).
_DISAGREE_BLUR = 0.02
_DISAGREE = (0.015, 0.06)


@dataclass
class AnamorphicLook:
    strength: float  # blur at a level of 1, x 5% of the width
    feather: float  # the lens radius the field softness starts at (1 = half the output's longer side)
    falloff: float  # how far from the subject the background takes to blur, fraction of the short side
    streak: float  # opacity of the horizontal light streaks
    depth_falloff: float = 0.3  # with depth: how far in depth from the subject the background takes to blur
    deform: float = 1.0  # how much the picture bends off axis, as through an anamorphic lens (1 = default)
    wobble: float = 1.0  # how much the bent edges drift (1 = default)
    fps: float = 24.0  # the clip's, to time the drift by


def _blur(x: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    """A wide gaussian blur, done at a reduced size so the kernel stays small."""
    h, w = x.shape[-2:]
    f = max(1.0, min(sx, sy) / 2.0)
    if f == 1.0:
        return _gauss(x, sx, sy)
    small = F.interpolate(x, size=(max(2, round(h / f)), max(2, round(w / f))), mode="bilinear", antialias=True, align_corners=False)
    return F.interpolate(_gauss(small, sx / f, sy / f), size=(h, w), mode="bilinear", align_corners=False)


def lens_radius(width: int, height: int, device: torch.device | None = None) -> torch.Tensor:
    """(height, width): how far off the lens's axis each pixel of a
    `width`x`height` frame is, in halves of its longer side, as the lens's
    image circle covers that: 1 at the middle of a landscape frame's sides,
    and a portrait frame's sides nearer in. Up and down count
    _LENS_VERTICAL as much. The same at any size, so it works on a small grid too."""
    half = max(width, height) / 2
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5 - height / 2) / half
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5 - width / 2) / half
    return torch.hypot(xs[None, :], _LENS_VERTICAL * ys[:, None])


def _sigma(level: torch.Tensor) -> torch.Tensor:
    """A blur level's width, in blurs at a level of 1."""
    return torch.where(level <= 1.0, level**2, level)


def _level(sigma: torch.Tensor) -> torch.Tensor:
    """The blur level of a width (the inverse of `_sigma`)."""
    return torch.where(sigma <= 1.0, sigma.clamp(min=0.0).sqrt(), sigma)


def _union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A soft union of two blur levels: as `1 - (1 - a)(1 - b)` while both
    are under 1, and never less than the larger."""
    hi, lo = torch.maximum(a, b), torch.minimum(a, b)
    return hi + lo * (1.0 - hi).clamp(min=0.0)


def _field(rho: torch.Tensor, feather: float) -> torch.Tensor:
    """The lens's own softness at lens radius `rho`, as a blur level in the
    output's units: 0 inside `feather`, building as _FIELD_POWER of the way
    from there to the output's corners (the largest `rho`), where it's
    ANAMORPHIC_FAR."""
    corner = max(float(rho.max()), feather + 1e-3)
    return config.ANAMORPHIC_FAR * ((rho - feather) / (corner - feather)).clamp(0.0, 1.0) ** _FIELD_POWER


def blur_levels(focus: Focus, rect: Rect, width: int, height: int, look: AnamorphicLook) -> torch.Tensor:
    """(n, 1, gh, gw) blur level over the `width`x`height` canvas for each of
    `focus`'s frames, on a small grid, in the clip's units (see the module
    docstring)."""
    from scipy.ndimage import distance_transform_edt

    gw, gh = _grid(width, height, _LEVEL_SIZE)
    s = gw / width
    x0, y0, x1, y1 = rect
    gx0, gy0, gx1, gy1 = round(x0 * s), round(y0 * s), max(round(x0 * s) + 1, round(x1 * s)), max(round(y0 * s) + 1, round(y1 * s))
    short = min(gw, gh)
    clip_short = min(x1 - x0, y1 - y0)

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

    xs = (torch.arange(gw) + 0.5) / s
    ys = (torch.arange(gh) + 0.5) / s

    def edge(coord: torch.Tensor, lo: int, hi: int, extent: int) -> torch.Tensor:
        d = torch.full_like(coord, float("inf"))
        if lo > 0:
            d = torch.minimum(d, coord - lo)
        if hi < extent:
            d = torch.minimum(d, hi - coord)
        return d

    # How far in from the clip's edges that border new picture.
    inside = torch.minimum(edge(ys, y0, y1, height)[:, None], edge(xs, x0, x1, width)[None, :]).clamp(min=0.0)
    in_rect = (xs[None, :] >= x0) & (xs[None, :] < x1) & (ys[:, None] >= y0) & (ys[:, None] < y1)
    # The lens's softness, in the output's units, put in the clip's.
    field = _level(_sigma(_field(lens_radius(gw, gh), look.feather)) * min(width, height) / clip_short)
    # Never under the floor over the new picture and the seam into it.
    floor = _SEAM_FLOOR * torch.where(
        in_rect, 1.0 - _smoothstep(0.0, _SEAM_BAND * clip_short, inside), torch.ones_like(inside))
    field = torch.maximum(field, floor)
    # The subject is spared it, but for the last stretch before an open edge,
    # so one that's cut off by the frame still softens into the new picture.
    tight = 1.0 - _smoothstep(0.0, config.ANAMORPHIC_SUBJECT_EDGE * clip_short, inside)
    return _union(level, field * (1.0 - canvas * (1.0 - tight)))


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
    """`x` (n, c, height, width, colour first) bent as an anamorphic lens
    bends off its axis: stretched outward from the centre, mostly sideways,
    swirled round it, with red and blue drawn apart, and drifting slowly at
    `times` (n,) seconds, `wobble` (n,) as much in each frame. It comes in
    softly with the lens radius (from _LENS_INNER to full at the output's
    corners), times `amount` (n, 1, height, width), and never where `still`
    (the subject) is. `rect` (the clip) sets how far it drifts. Returns it
    and how bent each pixel is (0-1)."""
    x0, y0, x1, y1 = rect
    dev = x.device
    cx, cy = width / 2, height / 2
    ys = torch.arange(height, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=dev, dtype=torch.float32) + 0.5
    dx, dy = xs[None, :] - cx, ys[:, None] - cy
    rho = lens_radius(width, height, dev)
    t = _smoothstep(_LENS_INNER, max(float(rho.max()), _LENS_INNER + 1e-3), rho)[None, None] * amount * (1 - still)
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


def lay_over(frames: np.ndarray, base: np.ndarray, rect: Rect, seam: float) -> np.ndarray:
    """`frames` (n, rh, rw, 3, 0-1, at `rect`'s size) laid over `base` (n,
    height, width, 3, 0-1: the new picture, held from a keyframe) and
    feathered into it across the `seam` px it redrew inside the clip; the
    base matched to each frame first, broadly, by how the two differ along
    the clip's edge, carried on out into the new picture and fading away
    from the clip, as the footage's light and colour drift from the
    keyframe's. Returns (n, height, width, 3) float32 0-1."""
    x0, y0, x1, y1 = rect
    height, width = base.shape[1:3]
    dev = _device()
    x = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32)).to(dev).permute(0, 3, 1, 2)
    generated = torch.from_numpy(np.ascontiguousarray(base, dtype=np.float32)).to(dev).permute(0, 3, 1, 2)
    pad = (x0, width - x1, y0, height - y1)
    band = torch.zeros((1, 1, height, width), device=dev)
    band[:, :, y0:y1, x0:x1] = 1.0 - torch.from_numpy(_edge_alpha(rect, width, height, 2 * seam)).to(dev)
    reach = 2 * max(seam, 8.0)
    differs = _blur((F.pad(x, pad) - generated) * band, reach, reach)
    weight = _blur(band, reach, reach)
    fade = 1.0 - _smoothstep(0.0, 3 * reach, _beyond(rect, width, height, dev))
    generated = generated + differs / weight.clamp(min=1e-3) * fade * (weight > 1e-3)
    clip_alpha = torch.zeros((1, 1, height, width), device=dev)
    clip_alpha[:, :, y0:y1, x0:x1] = torch.from_numpy(_edge_alpha(rect, width, height, max(seam, 1.0))).to(dev)
    canvas = clip_alpha * F.pad(x, pad) + (1.0 - clip_alpha) * generated
    return canvas.clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()


def new_picture(rect: Rect, width: int, height: int, seam: float) -> np.ndarray:
    """(height, width): how much of each pixel is new picture, 1 outside
    the clip, easing to 0 across the `seam` px inside its open edges, as
    `lay_over` feathers it."""
    x0, y0, x1, y1 = rect
    out = np.ones((height, width), dtype=np.float32)
    out[y0:y1, x0:x1] = 1.0 - _edge_alpha(rect, width, height, max(seam, 1.0))
    return out


def render(canvases: np.ndarray, focus: Focus, levels: torch.Tensor, rect: Rect, look: AnamorphicLook,
           start: int = 0) -> np.ndarray:
    """The look over `canvases` (n, height, width, 3, 0-1: whole output
    frames, the clip at `rect`, frame `start` on), blurred to `levels` (from
    `blur_levels` for the same `focus`) and bent, with the anamorphic grade
    and streaks coming in with the blur. Everything but the unblurred
    picture is blurred from a plate with the subject filled in from around
    it, so the subject doesn't smear into the background.

    Now and then MPS hands back a batch of garbage, with no error. Anywhere
    left all but unblurred must come out as it went in, so a batch where it
    doesn't is done again, and then on the CPU."""
    for dev in (_device(), _device(), torch.device("cpu")):
        out, ok = _render(canvases, focus, levels, rect, look, start, dev)
        if ok:
            break
        print(f"  warning: frames {start}-{start + len(canvases) - 1} came back corrupted on {dev.type}; redoing them")
    return out


def _render(canvases: np.ndarray, focus: Focus, levels: torch.Tensor, rect: Rect, look: AnamorphicLook,
            start: int, dev: torch.device) -> tuple[np.ndarray, bool]:
    x0, y0, x1, y1 = rect
    height, width = canvases.shape[1:3]
    canvas = torch.from_numpy(np.ascontiguousarray(canvases, dtype=np.float32)).to(dev).permute(0, 3, 1, 2)
    original = canvas
    pad = (x0, width - x1, y0, height - y1)

    def at_clip(a: np.ndarray) -> torch.Tensor:
        return F.interpolate(torch.from_numpy(a.astype(np.float32)).to(dev)[:, None], size=(y1 - y0, x1 - x0),
                             mode="bilinear", align_corners=False)

    m = F.pad(at_clip(focus.mask), pad)
    # Scaled to the clip, not the output: the same lens, whatever's around it.
    edge_sigma = max(2.0, _EDGE_SIGMA * min(x1 - x0, y1 - y0) * look.strength)
    bent = torch.zeros_like(canvas[:, :1])
    if look.deform > 0:
        # How out of focus each part is, by its depth (the clip's edge's
        # carried on outward): the further from the subject's, the more it bends.
        if focus.near is not None and focus.depth is not None:
            near = F.pad(at_clip(focus.near), pad, mode="replicate")
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
    sigma = edge_sigma * _sigma(level)
    # Half an octave apart: further apart, and fading between two very
    # different blurs shows as a band. Up to the widest asked for: the lens's
    # softness, in the clip's units, can be past a level of 1 when the
    # output is much bigger than the clip.
    top = max(0, int(np.ceil(2 * np.log2(max(float(sigma.max()) / edge_sigma, config.ANAMORPHIC_FAR)))))
    stack = [0.0, *(edge_sigma * 2 ** (k / 2) for k in range(-8, 1 + top))]
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
    ok = bool(not sharp.any() or (out - original)[:, :, y0:y1, x0:x1][sharp].abs().mean() < _CORRUPT)
    return out.permute(0, 2, 3, 1).cpu().numpy(), ok


def scaled_plates(plates: Sequence[np.ndarray], start: int, n: int, width: int, height: int,
                  pad: int = 0) -> np.ndarray:
    """(n + pad, height, width, 3, 0-1): `plates`' frames [start, start + n)
    scaled to `width`x`height`, then `pad` copies of the last."""
    up = video._scale_frames(np.stack(plates[start:start + n]).astype(np.float32) / 255.0, width, height)
    return np.concatenate([up, np.repeat(up[-1:], pad, axis=0)], axis=0) if pad else up


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

    def bridge_weights(self, start: int, n: int, width: int, height: int) -> np.ndarray:
        """(n, height, width) how free VACE is over frames [start, start +
        n), place by place: `anchor_weights`, but only as much as the two
        keyframes either side disagree there (see _DISAGREE), so where
        they agree it's held to their crossfade all the way."""
        weights = self.anchor_weights(start, n)
        keys = np.asarray(self.keys)
        out = np.zeros((n, height, width), dtype=np.float32)
        for j, i in enumerate(range(start, start + n)):
            k = int(np.clip(np.searchsorted(keys, i, side="right") - 1, 0, len(keys) - 1))
            if weights[j] > 0 and k + 1 < len(keys):
                out[j] = weights[j] * self._disagreement(k, width, height)
        return out

    def _disagreement(self, k: int, width: int, height: int) -> np.ndarray:
        """(height, width), 0-1: how much keyframes k and k + 1 disagree."""
        cache = self.__dict__.setdefault("_disagree", {})
        if (k, width, height) not in cache:
            a, b = (torch.from_numpy(self.plates[j].astype(np.float32) / 255.0).permute(2, 0, 1)[None]
                    for j in (k, k + 1))
            sigma = _DISAGREE_BLUR * min(a.shape[-2:])
            d = _blur((a - b).abs().mean(1, keepdim=True), sigma, sigma)
            d = F.interpolate(d, size=(height, width), mode="bilinear", align_corners=False)
            cache[(k, width, height)] = _smoothstep(*_DISAGREE, d)[0, 0].numpy()
        return cache[(k, width, height)]


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
