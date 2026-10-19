"""A long command's work dir, `<output-stem>.work/` next to its output: every
model call's result kept as it's made, and PNGs to watch the run by.

Each kept result is filed under a key hashed from everything that went into
the call (the control frames themselves, the seed, steps, prompt...), not
from where it falls in the run. So a run carries on wherever the last one
stopped, and a later run that asks for more of the same (more iterations,
more grow passes, more keyframes) reuses what's there. A result that feeds
the next call (extend's context, expand's overlap, a FLUX plate) is part of
that call's key, so when one changes -- or is edited by hand -- everything
after it is made again. Results are always uint8, fresh or kept, so a kept
one is bit-for-bit what it would have been.

One work dir is active at a time (see `use`); with none, nothing is kept or
written, and the helpers just run what they're given."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

_active: WorkDir | None = None


class WorkDir:
    def __init__(self, root: Path):
        self.root = root
        self.runs = root / "runs"
        self.plates = root / "plates"
        self.analysis = root / "analysis"
        self.progress = root / "progress"
        self.stills = root / "stills"
        for d in (self.runs, self.plates, self.analysis, self.progress, self.stills):
            d.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def find(self, folder: Path, key: str, suffix: str) -> Path | None:
        """The kept file for `key`, whatever it was labelled when it was made."""
        found = sorted(folder.glob(f"*-{key[:16]}{suffix}"))
        return found[0] if found else None

    def size(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())


def use(output: Path, work: Path | None = None, enabled: bool = True) -> WorkDir | None:
    """Makes `work` (default `<output-stem>.work/` beside `output`) the active
    work dir, or none if not `enabled`."""
    global _active
    _active = WorkDir(work or output.with_name(f"{output.stem}.work")) if enabled else None
    if _active is not None:
        print(f"  work dir: {_active.root} (progress PNGs in {_active.progress.name}/)")
    return _active


def active() -> WorkDir | None:
    return _active


def finish() -> None:
    """Says what the work dir holds and how to reclaim it."""
    if _active is None:
        return
    print(f"Kept {_active.misses} new and reused {_active.hits} model results in {_active.root} "
          f"({_active.size() / 1e6:,.0f}MB). Re-running reuses them; delete it once you're done.")


def interrupted_hint() -> None:
    if _active is not None and _active.misses + _active.hits:
        print(f"Finished model runs are kept in {_active.root}: re-run the same command to carry on.")


def key(**parts) -> str:
    """A hash of everything that goes into a call: scalars, and arrays or
    tensors by their dtype, shape and bytes."""
    h = hashlib.sha1()
    for name in sorted(parts):
        value = parts[name]
        h.update(name.encode())
        if hasattr(value, "detach"):  # a torch tensor
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            h.update(f"{value.dtype}{value.shape}".encode())
            h.update(np.ascontiguousarray(value).data)  # hashed in place, not copied
        else:
            h.update(json.dumps(value, sort_keys=True, default=str).encode())
    return h.hexdigest()


def to_uint8(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    if x.dtype == np.uint8:
        return x
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Writes via a temp name, so a run killed mid-write never leaves a
    half-written result to be taken for a whole one."""
    tmp = path.with_name(f".tmp-{path.name}")
    write(tmp)
    tmp.replace(path)


def cached_frames(k: str, label: str, make: Callable[[], object]) -> tuple[np.ndarray, bool]:
    """`make()`'s frames, (n, h, w, 3), as uint8, kept losslessly under `k`.
    Returns (frames, whether they were kept from before)."""
    from soto import video

    wd = _active
    if wd is not None and (path := wd.find(wd.runs, k, ".mkv")) is not None:
        w, h = video.frame_size(path)
        wd.hits += 1
        return np.stack(list(video.iter_frames(path, w, h))), True
    frames = to_uint8(make())
    if wd is not None:
        wd.misses += 1
        _atomic(wd.runs / f"{label}-{k[:16]}.mkv", lambda p: video.write_lossless(frames, 1.0, p))
    return frames, False


def cached_image(k: str, label: str, make: Callable[[], object]) -> tuple[np.ndarray, bool]:
    """`make()`'s image, (h, w, 3), as uint8, kept as a PNG under `k`. The
    PNG can be edited by hand: it's what's used from then on, and whatever
    was made from it is made again."""
    from PIL import Image

    wd = _active
    if wd is not None and (path := wd.find(wd.plates, k, ".png")) is not None:
        wd.hits += 1
        return np.asarray(Image.open(path).convert("RGB")), True
    image = to_uint8(make())
    if wd is not None:
        wd.misses += 1
        _atomic(wd.plates / f"{label}-{k[:16]}.png", lambda p: write_png(p, image))
    return image, False


def cached_object(k: str, label: str, make: Callable[[], object]) -> tuple[object, bool]:
    """`make()`'s result (an analysis: numpy arrays in dataclasses), pickled
    under `k`. Returns (it, whether it was kept from before)."""
    wd = _active
    if wd is not None and (path := wd.find(wd.analysis, k, ".pkl")) is not None:
        wd.hits += 1
        with path.open("rb") as fh:
            return pickle.load(fh), True
    value = make()
    if wd is not None:
        wd.misses += 1

        def write(p: Path) -> None:
            with p.open("wb") as fh:
                pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)

        _atomic(wd.analysis / f"{label}-{k[:16]}.pkl", write)
    return value, False


def source_hash(module) -> str:
    """A module's source, hashed, for keying analyses on the code that made
    them as well as their inputs."""
    return hashlib.sha1(Path(module.__file__).read_bytes()).hexdigest()


def file_key(path: Path) -> dict:
    """Stands in for a file's contents: where it is, how big, and when it changed."""
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def write_png(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(path, format="PNG")


def sheet(images: list[np.ndarray], columns: int | None = None, gap: int = 4) -> np.ndarray:
    """Tiles same-sized uint8 images left to right, top to bottom, on black."""
    images = [to_uint8(i) for i in images]
    h, w = images[0].shape[:2]
    columns = columns or len(images)
    rows = -(-len(images) // columns)
    out = np.zeros((rows * h + (rows - 1) * gap, columns * w + (columns - 1) * gap, 3), dtype=np.uint8)
    for n, image in enumerate(images):
        r, c = divmod(n, columns)
        out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = image
    return out


def show(label: str, frames: list[np.ndarray] | np.ndarray) -> None:
    """A contact sheet of a finished run's first, middle and last frames, as
    `progress/<label>.png` and `progress/latest.png`."""
    wd = _active
    if wd is None or len(frames) == 0:
        return
    picks = sorted({0, len(frames) // 2, len(frames) - 1})
    image = sheet([frames[i] for i in picks])
    write_png(wd.progress / f"{label}.png", image)
    shutil.copyfile(wd.progress / f"{label}.png", wd.progress / "latest.png")


def write_stills(stills: dict[int, np.ndarray], folder: Path, tile_width: int = 960) -> None:
    """Each still as `frame-NNNNNN.png` in `folder`, and all of them tiled,
    scaled down to at most `tile_width` across, as `sheet.png`."""
    from PIL import Image

    tiles = []
    for f, image in sorted(stills.items()):
        write_png(folder / f"frame-{f:06d}.png", image)
        im = Image.fromarray(to_uint8(image))
        if im.width > tile_width:
            im = im.resize((tile_width, round(im.height * tile_width / im.width)), Image.LANCZOS)
        tiles.append(np.asarray(im))
    write_png(folder / "sheet.png", sheet(tiles, columns=min(len(tiles), 3)))
    print(f"Stills: {folder} ({len(stills)} frames, tiled in sheet.png)")


def step_preview(label: str, latent_frames: list[int] | None = None):
    """An `on_step(step, total, rgb)` for comfy_engine: `rgb` is the model's
    current guess at the finished picture, (t, h, w, 3) uint8 at latent size,
    of which `latent_frames` (all of them if None, at most a few) are tiled,
    scaled up 8x, and written to `progress/step.png` and
    `progress/<label>-step-NN.png`. None with no work dir."""
    wd = _active
    if wd is None:
        return None
    from soto import config

    def on_step(step: int, total: int, rgb: np.ndarray) -> None:
        picks = latent_frames if latent_frames is not None else range(rgb.shape[0])
        picks = [i for i in picks if 0 <= i < rgb.shape[0]] or [0]
        if len(picks) > config.STEP_PREVIEW_FRAMES:
            picks = [picks[round(j * (len(picks) - 1) / (config.STEP_PREVIEW_FRAMES - 1))]
                     for j in range(config.STEP_PREVIEW_FRAMES)]
        from PIL import Image

        h, w = rgb.shape[1:3]
        tiles = [np.asarray(Image.fromarray(rgb[i]).resize((8 * w, 8 * h), Image.BILINEAR)) for i in picks]
        image = sheet(tiles)
        write_png(wd.progress / f"{label}-step-{step:02d}.png", image)
        shutil.copyfile(wd.progress / f"{label}-step-{step:02d}.png", wd.progress / "step.png")

    return on_step


def latent_index(frame: int) -> int:
    """The latent frame Wan's VAE puts pixel frame `frame` in: the first on
    its own, then 4 to a latent frame."""
    from soto import config

    return 0 if frame <= 0 else (frame - 1) // config.VACE_LENGTH_STRIDE + 1
