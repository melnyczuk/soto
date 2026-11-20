"""Which torch device soto runs on: CUDA if there is one, else Apple's MPS,
else the CPU. SOTO_DEVICE (cuda, mps or cpu) overrides the pick."""

from __future__ import annotations

import os

import torch


def pick() -> torch.device:
    forced = os.environ.get("SOTO_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(dev: torch.device) -> None:
    """Waits for queued work on `dev`: CUDA and MPS both return from a call
    as soon as it's queued."""
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def empty_cache(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()


def describe() -> str:
    """One line on what's available, for `soto doctor`."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        return f"CUDA: {name}, {mem:.0f}GB (torch {torch.__version__}, CUDA {torch.version.cuda})"
    if torch.backends.mps.is_available():
        return f"MPS (torch {torch.__version__})"
    return f"no GPU, CPU only (torch {torch.__version__})"
