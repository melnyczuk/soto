"""Training for `soto dream`'s pix2pixHD next-frame generators, written from
scratch after the older soto-pix2pixHD project's training code was lost.
The networks and losses are pix2pixHD's: the same GlobalGenerator as
pix2pixhd.py (its checkpoints load there as-is), a multiscale PatchGAN
discriminator, LSGAN, discriminator feature matching and VGG19 perceptual
loss.

Two changes target what made the old generators weak:

- Training on its own output. The old generators only ever saw real frames
  as input, so at dream time, fed their own output, every small error
  compounded until the video melted. Here the generator first runs on its
  own for a few frames from a real one (without gradients, so it's cheap --
  the "pushforward trick"), then is trained to predict the next real frame
  from where that left it. How far it may run grows over the first half of
  training, and each step picks a depth at random from 0 up to that, so it
  learns both to follow the footage and to steer its own output back to it.
- Full-resolution crops. Frames are kept at the clip's own resolution and
  trained on in random crops, rather than downscaled to fit, so the
  generator -- fully convolutional -- learns detail at the scale it dreams at.

Frames are decoded once into a uint8 memmap per clip, so memory stays flat
however long the clips are."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from soto import config

# --- discriminator -----------------------------------------------------------


class _NLayerDiscriminator(nn.Module):
    """pix2pixHD's PatchGAN discriminator with instance norm, returning every
    layer's output so the generator can be trained to match them."""

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [[nn.Conv2d(input_nc, ndf, 4, stride=2, padding=2), nn.LeakyReLU(0.2, True)]]
        nf = ndf
        for n in range(1, n_layers):
            prev, nf = nf, min(nf * 2, 512)
            layers.append([nn.Conv2d(prev, nf, 4, stride=2, padding=2), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)])
        prev, nf = nf, min(nf * 2, 512)
        layers.append([nn.Conv2d(prev, nf, 4, stride=1, padding=2), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)])
        layers.append([nn.Conv2d(nf, 1, 4, stride=1, padding=2)])
        self.layers = nn.ModuleList(nn.Sequential(*layer) for layer in layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class MultiscaleDiscriminator(nn.Module):
    """`num_d` PatchGANs, each seeing the input at half the previous one's
    resolution: the full-size one judges texture, the smaller ones structure."""

    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3, num_d: int = 2):
        super().__init__()
        self.scales = nn.ModuleList(_NLayerDiscriminator(input_nc, ndf, n_layers) for _ in range(num_d))

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        out = []
        for i, d in enumerate(self.scales):
            if i:
                x = F.avg_pool2d(x, 3, stride=2, padding=1, count_include_pad=False)
            out.append(d(x))
        return out


# --- losses ------------------------------------------------------------------


def _gan_loss(preds: list[list[torch.Tensor]], real: bool) -> torch.Tensor:
    """LSGAN on each scale's final output."""
    target = 1.0 if real else 0.0
    return sum(F.mse_loss(p[-1], torch.full_like(p[-1], target)) for p in preds) / len(preds)


def _feature_matching_loss(fake: list[list[torch.Tensor]], real: list[list[torch.Tensor]]) -> torch.Tensor:
    """L1 between the discriminator's intermediate features of real and fake,
    averaged over layers and scales (as pix2pixHD weights it)."""
    n_layers = len(fake[0]) - 1
    loss = 0.0
    for f_scale, r_scale in zip(fake, real):
        for f, r in zip(f_scale[:-1], r_scale[:-1]):
            loss = loss + F.l1_loss(f, r.detach())
    return loss / (len(fake) * n_layers)


class VGGLoss(nn.Module):
    """pix2pixHD's perceptual loss: L1 between VGG19 activations at relu1_1
    .. relu5_1, the deeper layers weighted more."""

    WEIGHTS = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0)

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.slices = nn.ModuleList(features[a:b] for a, b in ((0, 2), (2, 7), (7, 12), (12, 21), (21, 30)))
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        # Inputs are in [-1, 1]; VGG wants ImageNet-normalized [0, 1].
        x = ((fake + 1) / 2 - self.mean) / self.std
        y = ((real + 1) / 2 - self.mean) / self.std
        loss = 0.0
        for w, s in zip(self.WEIGHTS, self.slices):
            x, y = s(x), s(y)
            loss = loss + w * F.l1_loss(x, y.detach())
        return loss


# --- data --------------------------------------------------------------------


@dataclass
class ClipFrames:
    path: Path
    fps: float
    frames: np.ndarray  # (N, H, W, 3) uint8 memmap


def cache_frames(clip: Path, info, width: int, height: int, cache_dir: Path) -> ClipFrames:
    """Decodes `clip` at width x height into a uint8 memmap in `cache_dir`,
    reusing one from an earlier run if the clip (by path, size and mtime) and
    size haven't changed."""
    from soto import video

    stat = clip.stat()
    key = hashlib.sha1(f"{clip.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:12]
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frames-{clip.stem}-{key}-{width}x{height}"
    done = cache_dir / f"{stem}.u8"
    frame_bytes = width * height * 3
    if done.is_file():
        n = done.stat().st_size // frame_bytes
        frames = np.memmap(done, dtype=np.uint8, mode="r", shape=(n, height, width, 3))
        return ClipFrames(clip, info.fps, frames)

    print(f"  decoding {clip.name} at {width}x{height} into the frame cache "
          f"(~{info.n_frames * frame_bytes / 1e9:.1f} GB)...")
    partial = cache_dir / f"{stem}.partial"
    n = 0
    with open(partial, "wb") as f:
        for frame in video.iter_frames(clip, width, height):
            f.write(frame.tobytes())
            n += 1
            if n % 50 == 0:
                print(f"\r    frame {n}/{info.n_frames}", end="", flush=True)
    print(f"\r    frame {n}/{info.n_frames}")
    os.replace(partial, done)
    frames = np.memmap(done, dtype=np.uint8, mode="r", shape=(n, height, width, 3))
    return ClipFrames(clip, info.fps, frames)


class WindowSampler:
    """Random windows of consecutive frames, cropped, from any of the clips.
    A window never crosses from one clip into the next, and clips are picked
    in proportion to how many windows they hold, so every frame is equally
    likely whatever the clip lengths."""

    def __init__(self, clips: list[ClipFrames], crop: int, flip: bool, seed: int = 0):
        self.clips = clips
        self.crop = crop
        self.flip = flip
        self.rng = np.random.default_rng(seed)

    def crop_size(self, clip: ClipFrames) -> tuple[int, int]:
        _, h, w, _ = clip.frames.shape
        return min(self.crop, h), min(self.crop, w)

    def rollout(self, max_rollout: int) -> int:
        return int(self.rng.integers(max_rollout + 1))

    def sample(self, length: int, device: torch.device) -> torch.Tensor:
        """(length, 3, h, w) float in [-1, 1]."""
        counts = np.array([max(0, len(c.frames) - length + 1) for c in self.clips], dtype=np.float64)
        clip = self.clips[self.rng.choice(len(self.clips), p=counts / counts.sum())]
        t = int(self.rng.integers(len(clip.frames) - length + 1))
        _, h, w, _ = clip.frames.shape
        ch, cw = self.crop_size(clip)
        y, x = int(self.rng.integers(h - ch + 1)), int(self.rng.integers(w - cw + 1))
        window = np.ascontiguousarray(clip.frames[t:t + length, y:y + ch, x:x + cw])
        if self.flip and self.rng.random() < 0.5:
            window = np.ascontiguousarray(window[:, :, ::-1])
        return torch.from_numpy(window).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1


# --- training ----------------------------------------------------------------


def generator_state(net: nn.Module) -> dict[str, torch.Tensor]:
    """The generator's weights keyed as pix2pixHD saved them, so
    pix2pixhd.load_generator reads them like the old checkpoints."""
    return {f"model.{k}": v.detach().cpu() for k, v in net.state_dict().items()}


def save_atomic(obj, path: Path) -> None:
    """torch.save via a temp file and rename, so a `soto dream` reading the
    checkpoint mid-training never sees half a file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def max_rollout_at(step: int, total: int, unroll: int) -> int:
    """The most frames the generator may run on its own output before the one
    it's trained on, at `step`: 0 at the start, growing evenly to `unroll` by
    halfway through training."""
    half = max(1, total // 2)
    return int(unroll * min(1.0, step / half))


def lr_at(step: int, total: int) -> float:
    half = total // 2
    if step < half:
        return config.TRAIN_LR
    return config.TRAIN_LR * max(0.0, (total - step) / max(1, total - half))


class Trainer:
    def __init__(self, net_g: nn.Module, device: torch.device, *, noise: float):
        self.device = device
        self.noise = noise
        self.net_g = net_g.to(device).train()
        self.net_d = MultiscaleDiscriminator(input_nc=6).to(device).train()
        self.vgg = VGGLoss().to(device)
        self.opt_g = torch.optim.Adam(self.net_g.parameters(), lr=config.TRAIN_LR, betas=config.TRAIN_BETAS)
        self.opt_d = torch.optim.Adam(self.net_d.parameters(), lr=config.TRAIN_LR, betas=config.TRAIN_BETAS)

    def state(self, step: int) -> dict:
        return {
            "step": step,
            "net_g": self.net_g.state_dict(),
            "net_d": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }

    def load_state(self, state: dict) -> int:
        self.net_g.load_state_dict(state["net_g"])
        self.net_d.load_state_dict(state["net_d"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        return state["step"]

    def set_lr(self, lr: float) -> None:
        for opt in (self.opt_g, self.opt_d):
            for group in opt.param_groups:
                group["lr"] = lr

    def _perturb(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.noise * torch.randn_like(x)).clamp(-1, 1) if self.noise else x

    def step(self, window: torch.Tensor) -> dict[str, float]:
        """One optimizer step on a window of real frames (r + 2, 3, h, w):
        the generator runs r frames from the first on its own output, without
        gradients, then is trained to predict the window's last frame from its
        r-th prediction (or from the real first frame, if r is 0)."""
        x = self._perturb(window[:1])
        with torch.no_grad():
            for _ in range(window.shape[0] - 2):
                x = self._perturb(self.net_g(x))
        real = window[-1:]
        self.opt_g.zero_grad(set_to_none=True)
        self.opt_d.zero_grad(set_to_none=True)
        fake = self.net_g(x)

        pred_fake = self.net_d(torch.cat([x, fake.detach()], 1))
        pred_real = self.net_d(torch.cat([x, real], 1))
        loss_d = 0.5 * (_gan_loss(pred_fake, False) + _gan_loss(pred_real, True))
        loss_d.backward()

        # Frozen so the generator's losses don't also put gradients in D.
        self.net_d.requires_grad_(False)
        pred_fake = self.net_d(torch.cat([x, fake], 1))
        loss_gan = _gan_loss(pred_fake, True)
        loss_feat = _feature_matching_loss(pred_fake, pred_real)
        loss_vgg = self.vgg(fake, real)
        loss_g = loss_gan + config.TRAIN_LAMBDA_FEAT * loss_feat + config.TRAIN_LAMBDA_VGG * loss_vgg
        loss_g.backward()
        self.net_d.requires_grad_(True)

        self.opt_g.step()
        self.opt_d.step()
        return {"d": loss_d.item(), "gan": loss_gan.item(), "feat": loss_feat.item(), "vgg": loss_vgg.item()}
