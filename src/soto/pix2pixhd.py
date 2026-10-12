"""pix2pixHD next-frame "dreaming", ported from the older soto-pix2pixHD
project: a GlobalGenerator trained on a single video to predict frame t+1
from frame t, fed its own output in a loop to continue a clip for as long as
you like. Inference only -- training stayed behind in that project, and its
generator checkpoints (`checkpoints/<name>/latest_net_G.pth`) load here
as-is.

The camera moves (zoom, heat-seeking) are reimplemented as one on-device
affine resample per frame. The original round-tripped every frame through
uint8 + PIL for each move, re-quantizing it every time it went back into the
loop."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Heat-seeking pans are averaged over this many frames, so the camera eases
# toward its target instead of jerking between quadrants. The window starts
# full of zero pans, so it also eases in from a standstill.
HEAT_SEEKING_WINDOW = 30


class _ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # Layer order (and so the state-dict indices) matches pix2pixHD's
        # ResnetBlock with reflect padding, instance norm and no dropout.
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


def _global_generator(ngf: int, n_downsampling: int, n_blocks: int) -> nn.Sequential:
    """pix2pixHD's GlobalGenerator (RGB in, RGB out, instance norm), as the
    bare Sequential it wraps -- checkpoint keys are this with a "model." prefix."""
    layers: list[nn.Module] = [
        nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
    ]
    for i in range(n_downsampling):
        ch = ngf * 2**i
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
    layers += [_ResnetBlock(ngf * 2**n_downsampling) for _ in range(n_blocks)]
    for i in range(n_downsampling):
        ch = ngf * 2 ** (n_downsampling - i)
        layers += [
            nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ch // 2),
            nn.ReLU(True),
        ]
    layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, 7), nn.Tanh()]
    return nn.Sequential(*layers)


def load_generator(path: Path, device: torch.device) -> tuple[nn.Sequential, int]:
    """Loads a pix2pixHD GlobalGenerator checkpoint. Returns (net, stride):
    frames fed to `net` need both dimensions to be a multiple of `stride`.

    The architecture is read off the weights rather than hard-coded: instance
    norm has no parameters, so the only top-level weights are the convs -- one
    in, one per downsample, one per upsample, one out -- and every resnet
    block has its own conv_block.* keys."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("model."): v for k, v in state.items()}
    convs = sorted(int(k.split(".")[0]) for k in state if k.count(".") == 1 and k.endswith(".weight"))
    n_blocks = len({k.split(".")[0] for k in state if ".conv_block." in k})
    n_downsampling = (len(convs) - 2) // 2
    ngf = state[f"{convs[0]}.weight"].shape[0]

    net = _global_generator(ngf, n_downsampling, n_blocks)
    net.load_state_dict(state)
    return net.to(device).eval(), 2**n_downsampling


def _move_camera(x: torch.Tensor, zoom: float, heat_seeking: float, pans: deque) -> torch.Tensor:
    """Applies the original's zoom_in(zoom) followed by heat_seeking(heat_seeking)
    as one bilinear resample. x is (1, 3, H, W) in [-1, 1].

    Both are PIL EXTENT crops of z px off each edge, scaled back up -- in
    affine_grid's normalized coordinates, a scale of (1 - 2z/size) about the
    centre. Heat-seeking also pans by the moving average of the last
    HEAT_SEEKING_WINDOW frames' pans of `heat_seeking` px toward the brightest
    quadrant, with its zoom keeping the pan from exposing the edges."""
    _, _, h, w = x.shape
    zoom_x, zoom_y = 1 - 2 * zoom / w, 1 - 2 * zoom / h
    heat_x, heat_y = 1 - 2 * heat_seeking / w, 1 - 2 * heat_seeking / h
    pan_x = pan_y = 0.0
    if heat_seeking:
        top, bottom = x[..., :h // 2, :], x[..., h // 2:, :]
        quadrants = (top[..., :w // 2], top[..., w // 2:], bottom[..., :w // 2], bottom[..., w // 2:])
        brightest = int(torch.stack([q.mean() for q in quadrants]).argmax())
        pans.append((
            heat_seeking if brightest % 2 else -heat_seeking,
            heat_seeking if brightest >= 2 else -heat_seeking,
        ))
        pan_x, pan_y = np.mean(pans, axis=0)

    # Output samples the source at zoom * (heat * p + pan): the heat-seeking
    # crop and pan act on the already-zoomed frame, as in the original.
    theta = torch.tensor(
        [[zoom_x * heat_x, 0.0, zoom_x * 2 * pan_x / w], [0.0, zoom_y * heat_y, zoom_y * 2 * pan_y / h]],
        dtype=x.dtype, device=x.device,
    )[None]
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def dream(
    net: nn.Module, start: np.ndarray, n_frames: int, *, zoom: float = 0.0, heat_seeking: float = 0.0
) -> Iterator[np.ndarray]:
    """Yields `n_frames` frames continuing on from `start`, each the
    generator's prediction from the one before (after any camera move, which
    is part of what gets fed back). `start` and every yielded frame are
    (H, W, 3) float32 in [0, 1]; H and W must be multiples of the generator's
    stride. Frames are computed lazily, one at a time, so a long dream never
    has to fit in memory at once."""
    device = next(net.parameters()).device
    x = torch.from_numpy(start).to(device).permute(2, 0, 1)[None] * 2 - 1
    pans = deque([(0.0, 0.0)] * HEAT_SEEKING_WINDOW, maxlen=HEAT_SEEKING_WINDOW)
    for _ in range(n_frames):
        # Entered per frame rather than around the loop, so inference mode
        # doesn't stay switched on in the caller's code while this is
        # suspended at a yield.
        with torch.inference_mode():
            x = net(x)
            if zoom or heat_seeking:
                x = _move_camera(x, zoom, heat_seeking, pans)
            frame = ((x[0].permute(1, 2, 0) + 1) / 2).clamp(0, 1).cpu().numpy()
        yield frame
