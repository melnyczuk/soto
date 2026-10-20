"""Puts `soto expand`'s output together from VACE's canvases and the clip.

VACE's canvas has the seamless picture: the clip it was given runs into
the new edges without a line. But it's at working size, so scaled up it's
soft where the clip is sharp. So the output is the canvas scaled up, with
the clip's fine detail (the clip less the clip as VACE saw it, at working
size) added back over the clip and faded out toward the edges that border
new picture, where it eases into the edges' softness instead of stopping at
a line.

Then the anamorphic look goes over the lot at the output's size (see
anamorphic.py), the clip and the new edges alike, as one lens over the
whole frame.

Last, the new edges are matched to the footage, place by place: their local
chroma and hue (grey cloud and haze, and the subject, left out of the
measure) are set to the footage's, carried out from the clip and steadied
over time, over the whole frame but the subject (the look reaches into the
clip too). It's measured on the canvases with the look at working size, so
the look's light comes through but it can't pull the colour away from the
footage. (The older
"ring" match, one transform per frame, is still here for comparison.)"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from soto import config, video
from soto.anamorphic import AnamorphicLook, Rect, _beyond, _blur, _edge_alpha, _smoothstep, blur_levels, render
from soto.focus import Focus

# Over how much of the clip, in from each edge that borders new picture, its
# fine detail fades out, as a fraction of its short side.
_DETAIL_BAND = 0.12
# How wide the rings matched to each other are, either side of the clip's
# edges, as a fraction of its short side.
_MATCH_RING = 0.25
# The match is steadied over about this many seconds either side.
_MATCH_SMOOTH = 0.5
# Limits on the match, so an odd frame can't throw it: how far each channel's
# spread may be scaled, and how far its mean may move (Lab units: L, a, b).
_MATCH_RATIO = (0.5, 2.0)
_MATCH_SHIFT = (20.0, 15.0, 15.0)
# The match eases off away from the clip: from full at its edges to
# _MATCH_FLOOR of full by _MATCH_FADE of its short side out.
_MATCH_FADE = 1.0
_MATCH_FLOOR = 1.0
# Which match: "ring" (one transform per frame, from rings either side of
# the clip's edges) or "field" (the colour of the new edges set, place by
# place, to the footage's, carried out from the clip).
_MATCH_MODE = "field"
# The field match: how widely the footage's colour is carried out from the
# clip, and how widely the new edges' own colour is measured, as fractions
# of the clip's short side.
_FIELD_REACH = 0.5
_FIELD_LOCAL = 0.25
# How far it may change the new edges: chroma scaled within these, hue
# turned by up to this many degrees, L moved by up to this. L is left alone:
# VACE's brightness is already close, and blue-tinted generated clouds count
# as sky, so moving it leaves dark blotches.
_FIELD_GAIN = (0.25, 2.0)
_FIELD_TURN = 30.0
_FIELD_LIFT = 0.0
# Pixels with less chroma than this (Lab) are near grey (cloud, haze): they
# count toward the colour measured only as they rise through it.
_FIELD_GREY = (4.0, 12.0)
# The fields are worked out at 1/this of working size.
_FIELD_GRID = 8
# Above this, part of the subject: left out of the rings.
_SUBJECT = 0.5
# A ring with fewer pixels than this left once the subject's out is taken whole.
_MIN_RING = 64


def _inset(rect: Rect, width: int, height: int) -> np.ndarray:
    """(rh, rw): how far each pixel of `rect` is from its nearest edge that
    borders new picture (inf if none does)."""
    x0, y0, x1, y1 = rect

    def dist(n: int, lo_open: bool, hi_open: bool) -> np.ndarray:
        c = np.arange(n, dtype=np.float32) + 0.5
        d = np.full(n, np.inf, dtype=np.float32)
        if lo_open:
            d = np.minimum(d, c)
        if hi_open:
            d = np.minimum(d, c[::-1])
        return d

    return np.minimum(dist(y1 - y0, y0 > 0, y1 < height)[:, None], dist(x1 - x0, x0 > 0, x1 < width)[None, :])


def _lab(x: np.ndarray) -> torch.Tensor:
    """(n, h, w, 3) uint8 or 0-1 -> (n, 3, h, w) Lab."""
    import kornia

    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
    return kornia.color.rgb_to_lab(t / 255.0 if x.dtype == np.uint8 else t)


def _stats(lab: torch.Tensor, ring: torch.Tensor, clear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and spread of each channel of `lab` over `ring` where it's
    `clear` of the subject (or the whole ring, if the subject fills it):
    (n, 3) each."""
    weight = ring * clear
    weight = torch.where((weight.sum((1, 2, 3)) < _MIN_RING).view(-1, 1, 1, 1), ring.expand_as(weight), weight)
    total = weight.sum((2, 3)).clamp(min=1e-3)
    mu = (lab * weight).sum((2, 3)) / total
    var = (((lab - mu[..., None, None]) ** 2) * weight).sum((2, 3)) / total
    return mu, var.clamp(min=1e-4).sqrt()


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """`x` (n, ...) blurred along its first axis, held at the ends."""
    if sigma <= 0 or len(x) < 2:
        return x
    r = int(np.ceil(2 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.concatenate([np.repeat(x[:1], r, axis=0), x, np.repeat(x[-1:], r, axis=0)])
    return sum(w * padded[i:i + len(x)] for i, w in enumerate(k))


def _subject(masks: np.ndarray, rect: Rect, width: int, height: int) -> torch.Tensor:
    """(n, 1, height, width): the subject, at `rect` and carried on out from
    its edges."""
    x0, y0, x1, y1 = rect
    m = F.interpolate(torch.from_numpy(masks.astype(np.float32))[:, None], size=(y1 - y0, x1 - x0),
                      mode="bilinear", align_corners=False)
    return F.pad(m, (x0, width - x1, y0, height - y1), mode="replicate")


def colour_stats(canvases: Sequence[np.ndarray], sources: Sequence[np.ndarray], focus: Focus, rect: Rect,
                 fps: float, chunk: int = 16) -> dict[str, np.ndarray]:
    """Per frame, the Lab mean and spread (each (n, 3)) of the footage in a
    ring just inside the clip's open edges (`sources`, uint8, at `rect`'s
    size) and of `canvases` (uint8, working size) in a ring just outside
    them, the subject left out of both, steadied over time."""
    height, width = canvases[0].shape[:2]
    x0, y0, x1, y1 = rect
    ring = _MATCH_RING * min(x1 - x0, y1 - y0)
    inner = torch.from_numpy((_inset(rect, width, height) < ring).astype(np.float32))[None, None]
    beyond = _beyond(rect, width, height, torch.device("cpu"))
    outer = ((beyond > 0) & (beyond < ring)).float()[None, None]
    out: dict[str, list] = {"mu_r": [], "sd_r": [], "mu_t": [], "sd_t": []}
    for s in range(0, len(canvases), chunk):
        e = min(s + chunk, len(canvases))
        clear = (_subject(focus.mask[s:e], rect, width, height) <= _SUBJECT).float()
        mu, sd = _stats(_lab(np.stack(sources[s:e])), inner, clear[:, :, y0:y1, x0:x1])
        out["mu_r"].append(mu), out["sd_r"].append(sd)
        mu, sd = _stats(_lab(np.stack(canvases[s:e])), outer, clear)
        out["mu_t"].append(mu), out["sd_t"].append(sd)
    sigma = _MATCH_SMOOTH * fps / 2
    return {k: _smooth(torch.cat(v).numpy(), sigma) for k, v in out.items()}


def match_colour(canvases: np.ndarray, stats: dict[str, np.ndarray], rect: Rect,
                 strength: float = config.EXPAND_COLOUR_MATCH) -> np.ndarray:
    """`canvases` ((n, h, w, 3) uint8, working size) with their new edges
    matched to the footage by `stats` (those frames' rows of `colour_stats`),
    fully outside the clip and fading out across `_DETAIL_BAND` inside it.
    Returns (n, h, w, 3) float32 0-1."""
    import kornia

    x = canvases.astype(np.float32) / 255.0
    if strength <= 0:
        return x
    height, width = x.shape[1:3]
    x0, y0, x1, y1 = rect
    short = min(x1 - x0, y1 - y0)
    far = _smoothstep(0.0, max(1.0, _MATCH_FADE * short), _beyond(rect, width, height, torch.device("cpu")))
    c = (1.0 - (1.0 - _MATCH_FLOOR) * far).numpy()
    c[y0:y1, x0:x1] = 1.0 - _edge_alpha(rect, width, height, _DETAIL_BAND * short)
    mu_r, sd_r, mu_t, sd_t = (torch.from_numpy(stats[k]).float()[..., None, None]
                              for k in ("mu_r", "sd_r", "mu_t", "sd_t"))
    ratio = (sd_r / sd_t).clamp(*_MATCH_RATIO)
    limit = torch.tensor(_MATCH_SHIFT).view(1, 3, 1, 1)
    shift = torch.maximum(torch.minimum(mu_r - mu_t, limit), -limit)
    lab = _lab(x)
    matched = kornia.color.lab_to_rgb((lab - mu_t) * ratio + mu_t + shift).clamp(0.0, 1.0)
    matched = matched.permute(0, 2, 3, 1).numpy()
    return x + (strength * c)[None, :, :, None] * (matched - x)


def _spread(v: torch.Tensor, weight: torch.Tensor, sigma: float) -> torch.Tensor:
    """`v` (n, c, h, w) averaged by `weight` (n, 1, h, w) over a gaussian
    `sigma` wide; where there's little weight nearby, eased to the average
    over the whole frame."""
    eps = 1e-3
    whole = (v * weight).sum((2, 3), keepdim=True) / weight.sum((2, 3), keepdim=True).clamp(min=1e-6)
    return (_blur(v * weight, sigma, sigma) + eps * whole) / (_blur(weight, sigma, sigma) + eps)


def _colour(lab: torch.Tensor, weight: torch.Tensor, sigma: float) -> torch.Tensor:
    """Local average L, chroma and direction of hue (a unit (a, b) vector,
    not renormalised) of `lab` (n, 3, h, w) by `weight`: (n, 4, h, w)."""
    c = torch.hypot(lab[:, 1:2], lab[:, 2:3])
    return _spread(torch.cat([lab[:, :1], c, lab[:, 1:] / c.clamp(min=1e-3)], 1), weight, sigma)


def colour_fields(canvases: Sequence[np.ndarray], sources: Sequence[np.ndarray], focus: Focus, rect: Rect,
                  fps: float, chunk: int = 16, whole: bool = False) -> np.ndarray:
    """Per frame, on a grid `_FIELD_GRID` times smaller than `canvases`
    (uint8, working size), how far to move L, scale chroma (as its log) and
    turn hue (radians) so that the colour of the new edges, place by place,
    is the footage's (`sources`, uint8, at `rect`'s size) carried out from
    the clip (`whole`: of the whole frame, the clip too, as after the look).
    Grey (clouds, haze) and the subject are left out of both measures.
    Steadied over time: (n, 3, gh, gw)."""
    height, width = canvases[0].shape[:2]
    gh, gw = max(2, round(height / _FIELD_GRID)), max(2, round(width / _FIELD_GRID))
    s = gw / width
    x0, y0, x1, y1 = rect
    g0, g1 = (round(x0 * s), round(y0 * s)), (max(round(x0 * s) + 1, round(x1 * s)), max(round(y0 * s) + 1, round(y1 * s)))
    grect = (g0[0], g0[1], g1[0], g1[1])
    short = min(g1[0] - g0[0], g1[1] - g0[1])
    inside = torch.zeros((1, 1, gh, gw))
    inside[..., g0[1]:g1[1], g0[0]:g1[0]] = 1.0
    out = []
    for i in range(0, len(canvases), chunk):
        e = min(i + chunk, len(canvases))

        def small(frames: Sequence[np.ndarray], h: int, w: int) -> torch.Tensor:
            t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
            return F.interpolate(t, size=(h, w), mode="area")

        import kornia

        v = kornia.color.rgb_to_lab(small(canvases[i:e], gh, gw))
        src = F.pad(kornia.color.rgb_to_lab(small(sources[i:e], g1[1] - g0[1], g1[0] - g0[0])),
                    (g0[0], gw - g1[0], g0[1], gh - g1[1]))
        clear = (_subject(focus.mask[i:e], grect, gw, gh) <= _SUBJECT).float()

        def coloured(lab: torch.Tensor) -> torch.Tensor:
            return _smoothstep(*_FIELD_GREY, torch.hypot(lab[:, 1:2], lab[:, 2:3])) * clear

        ref = _colour(src, coloured(src) * inside, _FIELD_REACH * short)
        here = _colour(v, coloured(v) * (1.0 if whole else 1.0 - inside), _FIELD_LOCAL * short)
        lift = ref[:, :1] - here[:, :1]
        gain = torch.log(ref[:, 1:2].clamp(min=1.0) / here[:, 1:2].clamp(min=1.0))
        ur, uh = ref[:, 2:], here[:, 2:]
        turn = torch.atan2(uh[:, :1] * ur[:, 1:] - uh[:, 1:] * ur[:, :1], (uh * ur).sum(1, keepdim=True))
        out.append(torch.cat([lift, gain, turn], 1))
    return _smooth(torch.cat(out).numpy(), _MATCH_SMOOTH * fps / 2)


def match_fields(canvases: np.ndarray, fields: np.ndarray, rect: Rect,
                 strength: float = config.EXPAND_COLOUR_MATCH, where: np.ndarray | None = None) -> np.ndarray:
    """`canvases` ((n, h, w, 3) uint8 or float32 0-1, any size, the clip at
    `rect`) with their new edges moved by `fields` (those frames' rows of
    `colour_fields`), fully outside the clip and fading out across
    `_DETAIL_BAND` inside it, or by `where` ((n, h, w), 0-1) instead.
    Returns (n, h, w, 3) float32 0-1."""
    import kornia

    x = canvases.astype(np.float32) / 255.0 if canvases.dtype == np.uint8 else canvases
    if strength <= 0:
        return x
    height, width = x.shape[1:3]
    x0, y0, x1, y1 = rect
    if where is None:
        c = np.ones((height, width), dtype=np.float32)
        c[y0:y1, x0:x1] = 1.0 - _edge_alpha(rect, width, height, _DETAIL_BAND * min(x1 - x0, y1 - y0))
        c = torch.from_numpy(strength * c)
    else:
        c = torch.from_numpy(strength * where.astype(np.float32))[:, None]
    f = F.interpolate(torch.from_numpy(fields).float(), size=(height, width), mode="bilinear", align_corners=False)
    lift = f[:, :1].clamp(-_FIELD_LIFT, _FIELD_LIFT) * c
    gain = torch.exp(f[:, 1:2].clamp(*np.log(_FIELD_GAIN)) * c)
    turn = f[:, 2:].clamp(-np.radians(_FIELD_TURN), np.radians(_FIELD_TURN)) * c
    lab = _lab(x)
    a, b = lab[:, 1:2], lab[:, 2:3]
    cos, sin = torch.cos(turn), torch.sin(turn)
    lab = torch.cat([lab[:, :1] + lift, gain * (a * cos - b * sin), gain * (a * sin + b * cos)], 1)
    return kornia.color.lab_to_rgb(lab).clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()


def composite(canvases: list[np.ndarray], clip: Path, info: video.ClipInfo, focus: Focus, rect: Rect,
              work_rect: Rect, width: int, height: int, first: int = 0, only: Sequence[int] | None = None,
              chunk: int = 8, strength: float = config.EXPAND_COLOUR_MATCH,
              look: AnamorphicLook | None = None) -> Iterable[np.ndarray]:
    """The output's frames (`width`x`height`, float32 0-1): `canvases`
    (VACE's, uint8, at working size with the clip at `work_rect`, of the
    clip's frames from `first` on) matched to the footage, scaled up, with
    the clip's fine detail laid back over them at `rect`. `only` picks which
    of them (indexes into `canvases`) to make; the colour's steadied over
    all of them either way. Then `look` goes over each whole frame (none
    without it), and the colour's matched after it."""
    from tqdm import tqdm

    x0, y0, x1, y1 = rect
    wx0, wy0, wx1, wy1 = work_rect
    # As VACE was given the clip (see cli._expand_windows).
    flags = "area" if wx1 - wx0 <= info.width else "lanczos"
    n = len(canvases)
    small = list(itertools.islice(video.iter_frames(clip, wx1 - wx0, wy1 - wy0, flags=flags), first, first + n))
    window = focus[first:first + n]
    # With the look, the colour's matched after it, measured on the canvases
    # with it at working size: its grade and cool streaks would otherwise
    # push a sky full of bright cloud bluer and more violet than the footage.
    after = look is not None and _MATCH_MODE == "field"
    if after:
        looked = []
        for s in range(0, n, chunk):
            part = window[s:s + chunk]
            v = np.stack(canvases[s:s + chunk]).astype(np.float32) / 255.0
            v = render(v, part, blur_levels(part, work_rect, v.shape[2], v.shape[1], look), work_rect, look, first + s)
            looked.extend(np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8))
        stats = {"fields": colour_fields(looked, small, window, work_rect, info.fps, whole=True)}
    elif _MATCH_MODE == "field":
        stats = {"fields": colour_fields(canvases, small, window, work_rect, info.fps)}
    else:
        stats = colour_stats(canvases, small, window, work_rect, info.fps)
    detail = _edge_alpha(rect, width, height, _DETAIL_BAND * min(x1 - x0, y1 - y0))[..., None]
    wanted = sorted(range(n) if only is None else only)
    frames = itertools.islice(video.iter_frames(clip, x1 - x0, y1 - y0, flags="lanczos"), first, None)
    keep = set(wanted)
    real = (f for i, f in enumerate(frames) if i in keep)
    with tqdm(total=len(wanted), unit="frame") as bar:
        for s in range(0, len(wanted), chunk):
            idx = wanted[s:s + chunk]
            pick = np.stack([canvases[i] for i in idx])
            if after:
                v = pick.astype(np.float32) / 255.0
            elif _MATCH_MODE == "field":
                v = match_fields(pick, stats["fields"][idx], work_rect, strength)
            else:
                v = match_colour(pick, {k: a[idx] for k, a in stats.items()}, work_rect, strength)
            out = video._scale_frames(v, width, height)
            sharp = np.stack([next(real) for _ in idx]).astype(np.float32) / 255.0
            low = video._scale_frames(np.stack([small[i] for i in idx]).astype(np.float32) / 255.0,
                                      x1 - x0, y1 - y0)
            out[:, y0:y1, x0:x1] += detail * (sharp - low)
            out = np.clip(out, 0.0, 1.0)
            if look is not None:
                part = window[idx]
                out = render(out, part, blur_levels(part, rect, width, height, look), rect, look, first + idx[0])
            if after:
                # The look reaches into the clip too, so the whole frame is
                # matched, all but the subject, which keeps its own colour.
                subject = F.pad(F.interpolate(torch.from_numpy(part.mask.astype(np.float32))[:, None],
                                              size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False),
                                (x0, width - x1, y0, height - y1))[:, 0].numpy()
                out = match_fields(out, stats["fields"][idx], rect, strength, where=1.0 - subject)
            yield from out
            bar.update(len(idx))
