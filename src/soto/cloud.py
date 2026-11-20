"""`soto-cloud`: runs a soto command on a Modal GPU instead of this machine.

    uv run soto-cloud setup                       # once: fetch weights into the cloud
    uv run soto-cloud transition a.mp4 b.mp4 -o gen.mkv --duration 1.5
    uv run soto crossfade gen.mkv -o out.mp4 --frames 8   # locally, as usual

Everything after `soto-cloud` is an ordinary soto command. Local files named
in it are uploaded (once: they're stored by content, so a re-run of the same
clip finds it there and the work dir's cache still holds), the command runs
in a container with the GPU, and what it wrote next to its output comes back
next to the local output. Its log streams here as it runs.

Two Modal volumes hold the state between runs: `soto-models` is soto's
models/ directory (weights, pix2pixHD generators from `soto train`), and
`soto-jobs` holds the uploaded inputs and each output's work dir, so an
interrupted or repeated run picks up where it was, as it does locally.

Flags for soto-cloud itself go before the command:
    --gpu L4|A10|L40S|A100-80GB|H100   (default L4, 24GB; SOTO_CLOUD_GPU)
    --fetch-work                       also download the work dir (progress PNGs, kept results)

Needs `uv sync --extra cloud` and `uv run modal setup` once on the machine
you run it from (or MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import modal
import modal.exception

from soto import config

APP_NAME = "soto"
GPUS = ("T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB", "H100")
DEFAULT_GPU = os.environ.get("SOTO_CLOUD_GPU", "L4")
# Modal's ceiling for one call. A long `expand` at full size can take hours.
TIMEOUT_S = 24 * 60 * 60

REMOTE_ROOT = PurePosixPath("/soto")
JOBS_MOUNT = PurePosixPath("/jobs")

# Options whose value is a path the command writes: they're pointed into the
# job's out/ folder and brought back from there.
_OUTPUT_OPTIONS = ("-o", "--output", "--work", "--export-generated")

models_volume = modal.Volume.from_name("soto-models", create_if_missing=True)
jobs_volume = modal.Volume.from_name("soto-jobs", create_if_missing=True)

_here = Path(__file__).resolve()
_repo = _here.parents[2]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    # The locked dependencies, into the system Python so Modal's runtime
    # sees them. Linux torch wheels from PyPI carry CUDA.
    .add_local_file(_repo / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True)
    .add_local_file(_repo / "uv.lock", f"{REMOTE_ROOT}/uv.lock", copy=True)
    .add_local_file(_repo / "README.md", f"{REMOTE_ROOT}/README.md", copy=True)
    .run_commands(f"cd {REMOTE_ROOT} && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --no-install-project")
    # ComfyUI and ComfyUI-GGUF at soto's pinned commits.
    .run_commands(
        f"git clone {config.COMFYUI_REPO_URL} {REMOTE_ROOT}/vendor/ComfyUI",
        f"cd {REMOTE_ROOT}/vendor/ComfyUI && git checkout {config.COMFYUI_COMMIT}",
        f"git clone {config.COMFYUI_GGUF_REPO_URL} {REMOTE_ROOT}/vendor/ComfyUI/custom_nodes/ComfyUI-GGUF",
        f"cd {REMOTE_ROOT}/vendor/ComfyUI/custom_nodes/ComfyUI-GGUF && git checkout {config.COMFYUI_GGUF_COMMIT}",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT}/src", "PYTHONUNBUFFERED": "1"})
    # Last, and not copied into a layer, so editing soto's code doesn't
    # rebuild the image. At /soto/src, not on Modal's default path, because
    # config.py finds vendor/ and models/ relative to its own file.
    .add_local_dir(_repo / "src", f"{REMOTE_ROOT}/src", ignore=["**/__pycache__"])
)

app = modal.App(APP_NAME, image=image)


def _soto(args: list[str]) -> int:
    from soto import setup as setup_mod

    setup_mod.write_extra_model_paths()
    cmd = [sys.executable, "-c", "import sys; sys.argv[0] = 'soto'; from soto.cli import app; app()", *args]
    print(f"$ soto {' '.join(args)}", flush=True)
    return subprocess.run(cmd, cwd=str(JOBS_MOUNT)).returncode


@app.function(gpu=DEFAULT_GPU, timeout=TIMEOUT_S,
              volumes={str(REMOTE_ROOT / "models"): models_volume, str(JOBS_MOUNT): jobs_volume})
def run(args: list[str]) -> int:
    """Runs `soto <args>` in the container, with paths already pointing into /jobs."""
    try:
        return _soto(args)
    finally:
        # A `train` writes its generator into models/; everything else only into /jobs.
        models_volume.commit()
        jobs_volume.commit()


@app.function(timeout=2 * 60 * 60, volumes={str(REMOTE_ROOT / "models"): models_volume})
def setup(flux: bool = False) -> None:
    """Fetches the weights into the soto-models volume (ComfyUI is already in the image)."""
    from soto import setup as setup_mod

    setup_mod.write_extra_model_paths()
    setup_mod.download_small_weights()
    setup_mod.download_subject_models()
    if flux:
        setup_mod.download_flux_models()
    models_volume.commit()


# --- the local side --------------------------------------------------------


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _exists(volume: modal.Volume, path: str) -> bool:
    parent, name = str(PurePosixPath(path).parent), PurePosixPath(path).name
    try:
        return any(PurePosixPath(e.path).name == name for e in volume.listdir(parent))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False


class _Job:
    """Rewrites a soto command's local paths to the container's, uploading inputs."""

    def __init__(self, args: list[str]):
        self.args = args
        self.uploads: list[tuple[Path, str]] = []
        self.remote_to_local: dict[str, str] = {}
        self.placed: dict[str, Path] = {}  # out/'s top-level names -> where they go here
        self.work: tuple[str, Path] | None = None  # (volume path, local path) of a --work dir
        self.out_dir: str | None = None
        self.local_out_dir = Path.cwd()
        self.remote_args = self._rewrite(args)

    def _input(self, value: str) -> str:
        local = Path(value).expanduser()
        key = _sha(local)
        rel = f"in/{key}/{local.name}"
        self.uploads.append((local, rel))
        sidecar = local.with_suffix(".json")
        if sidecar.is_file() and sidecar != local:
            self.uploads.append((sidecar, f"in/{key}/{sidecar.name}"))
        remote = str(JOBS_MOUNT / rel)
        self.remote_to_local[remote] = str(local.resolve())
        return remote

    def _output(self, option: str, value: str) -> str:
        local = Path(value).expanduser().resolve()
        if option == "--work":
            # A named work dir lives under work/, by its folder name, apart
            # from any one output: several outputs (`soto take`'s takes of
            # one scene) share it, and reuse each other's kept results.
            rel = f"work/{local.name}"
            self.work = (rel, local)
        else:
            rel = f"{self.out_dir}/{local.name}"
            self.placed[local.name] = local
        remote = str(JOBS_MOUNT / rel)
        self.remote_to_local[remote] = str(local)
        return remote

    def _rewrite(self, args: list[str]) -> list[str]:
        # One job folder per output name, so a re-run keeps its work dir.
        output = self._find(args, ("-o", "--output"))
        self.local_out_dir = Path(output).expanduser().resolve().parent if output else Path.cwd()
        self.out_dir = f"out/{Path(output).stem}" if output else None
        out: list[str] = []
        i = 0
        while i < len(args):
            arg = args[i]
            name, eq, inline = arg.partition("=")
            if arg.startswith("-") and name in _OUTPUT_OPTIONS and self.out_dir:
                if eq:
                    out.append(f"{name}={self._output(name, inline)}")
                elif i + 1 < len(args):
                    out += [arg, self._output(name, args[i + 1])]
                    i += 1
                else:
                    out.append(arg)
            elif not arg.startswith("-") and Path(arg).expanduser().is_file():
                out.append(self._input(arg))
            else:
                out.append(arg)
            i += 1
        if self.out_dir:
            # The default work dir and sidecars sit beside the output.
            self.remote_to_local[str(JOBS_MOUNT / self.out_dir)] = str(self.local_out_dir)
        return out

    @staticmethod
    def _find(args: list[str], options: tuple[str, ...]) -> str | None:
        for i, arg in enumerate(args):
            name, eq, inline = arg.partition("=")
            if name in options:
                return inline if eq else (args[i + 1] if i + 1 < len(args) else None)
        return None

    def upload(self) -> None:
        todo = [(local, rel) for local, rel in self.uploads if not _exists(jobs_volume, rel)]
        if not todo:
            return
        for local, _ in todo:
            print(f"uploading {local} ({local.stat().st_size / 2**20:.0f}MB)...")
        with jobs_volume.batch_upload() as batch:
            for local, rel in todo:
                batch.put_file(str(local), rel)

    def download(self, fetch_work: bool) -> None:
        if self.out_dir is None:
            return
        for entry in jobs_volume.listdir(self.out_dir, recursive=True):
            if entry.type != modal.volume.FileEntryType.FILE:
                continue
            rel = PurePosixPath(entry.path).relative_to(self.out_dir)
            if len(rel.parts) > 1 and not fetch_work:
                continue  # inside the default work dir
            top = self.placed.get(rel.parts[0], self.local_out_dir / rel.parts[0])
            self._fetch(entry.path, top.joinpath(*rel.parts[1:]))
        if fetch_work and self.work and _exists(jobs_volume, self.work[0]):
            rel_root, local_root = self.work
            for entry in jobs_volume.listdir(rel_root, recursive=True):
                if entry.type == modal.volume.FileEntryType.FILE:
                    rel = PurePosixPath(entry.path).relative_to(rel_root)
                    self._fetch(entry.path, local_root.joinpath(*rel.parts), quiet=True)
            print(f"  -> {local_root}/")

    def _fetch(self, remote: str, dest: Path, quiet: bool = False) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            jobs_volume.read_file_into_fileobj(remote, f)
        if dest.suffix == ".json":
            self._localize(dest)
        if not quiet:
            print(f"  -> {dest}")

    def _localize(self, sidecar: Path) -> None:
        """Points a sidecar's paths back at the local files, so `soto
        crossfade` finds the footage here."""
        text = sidecar.read_text()
        for remote, local in sorted(self.remote_to_local.items(), key=lambda kv: -len(kv[0])):
            text = text.replace(remote, local)
        sidecar.write_text(text)


def main() -> None:
    argv = sys.argv[1:]
    gpu, fetch_work = DEFAULT_GPU, False
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--gpu" and argv:
            gpu = argv.pop(0)
        elif flag.startswith("--gpu="):
            gpu = flag.split("=", 1)[1]
        elif flag == "--fetch-work":
            fetch_work = True
        else:
            sys.exit(f"soto-cloud: unknown flag {flag} (soto-cloud flags go before the command)")
    if not argv:
        sys.exit(__doc__)
    if gpu not in GPUS:
        sys.exit(f"soto-cloud: --gpu must be one of {list(GPUS)}")

    if argv[0] == "setup":
        with modal.enable_output(), app.run():
            setup.remote(flux="--flux" in argv)
        return
    if argv[0] == "crossfade":
        print("note: crossfade takes seconds and needs no GPU; `soto crossfade` runs it here.", file=sys.stderr)

    sys.exit(run_remote(argv, gpu=gpu, fetch_work=fetch_work))


def run_remote(argv: list[str], gpu: str = DEFAULT_GPU, fetch_work: bool = False) -> int:
    """Runs `soto <argv>` on a Modal GPU, local paths and all; returns its exit code."""
    if gpu not in GPUS:
        raise ValueError(f"gpu must be one of {list(GPUS)}, not {gpu!r}")
    job = _Job(argv)
    with modal.enable_output(), app.run():
        job.upload()
        code = run.with_options(gpu=gpu).remote(job.remote_args)
        job.download(fetch_work)
    return code


if __name__ == "__main__":
    main()
