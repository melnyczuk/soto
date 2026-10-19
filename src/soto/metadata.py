"""Per-run metadata: an `<output>.json` sidecar recording what a command was
asked to do, what it actually did (and every setting it changed along the
way), and what each stretch of the output is -- plus the same segments
embedded in the output as chapter markers. A generation command's sidecar
also holds the `timeline` and `audio` that `soto crossfade` splices its
frames into the original footage by."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import typer

from soto import config

def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_state() -> dict | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=config.REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": bool(dirty)}


def _version() -> str | None:
    try:
        return importlib_metadata.version("soto")
    except importlib_metadata.PackageNotFoundError:
        return None


def timecode(frame: int, fps: float) -> str:
    """HH:MM:SS:FF, counting frames at the nearest whole frame rate (non-drop)."""
    base = max(1, round(fps))
    seconds, ff = divmod(frame, base)
    minutes, ss = divmod(seconds, 60)
    hh, mm = divmod(minutes, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _escape(value: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        value = value.replace(ch, "\\" + ch)
    return value


class RunLog:
    """Filled in by a command as it runs, then written by `write`. With
    `enabled=False` it still collects everything (it's cheap) but writes nothing."""

    def __init__(self, command: str, *, enabled: bool = True):
        self.enabled = enabled
        self._t0 = time.perf_counter()
        self.data: dict = {
            "command": command,
            "argv": sys.argv[1:],
            "started_at": _now(),
            "soto_version": _version(),
            "git": _git_state() if enabled else None,
            "inputs": [],
            "settings": {},
            "adjustments": [],
            "runs": [],
            "segments": [],
        }

    @property
    def settings(self) -> dict:
        return self.data["settings"]

    def input(self, role: str, path: Path, info, conformed: list[str], conform_to: dict | None = None) -> None:
        """`conform_to` is the fps and size the clip was conformed to, so
        `soto crossfade` can conform it the same way again."""
        self.data["inputs"].append({
            "role": role,
            "path": str(Path(path).resolve()),
            "fps": info.fps,
            "width": info.width,
            "height": info.height,
            "frames": info.n_frames,
            "seconds": round(info.n_frames / info.fps, 3),
            "conformed": conformed,
            **({"conform_to": conform_to} if conform_to else {}),
        })

    def adjust(self, setting: str, requested, used, reason: str) -> None:
        """A value soto changed from what was asked for (or picked itself)."""
        self.data["adjustments"].append(
            {"setting": setting, "requested": requested, "used": used, "reason": reason}
        )

    def append(self, key: str, item) -> None:
        self.data.setdefault(key, []).append(item)

    @contextmanager
    def run(self, **details):
        """Times one model call; `details` are its seed, steps, frame counts, etc."""
        entry = {"index": len(self.data["runs"]) + 1, "started_at": _now(), **details}
        t0 = time.perf_counter()
        try:
            yield entry
        except BaseException:
            entry["failed"] = True
            raise
        finally:
            entry["elapsed_s"] = round(time.perf_counter() - t0, 2)
            self.data["runs"].append(entry)

    @contextmanager
    def guard(self, output: Path):
        """If the body raises, writes what was logged so far to
        `<output>.failed.json` before re-raising. Usage errors (a bad option,
        caught before any work) and deliberate exits aren't recorded."""
        try:
            yield
        except (typer.BadParameter, typer.Exit, typer.Abort, SystemExit):
            raise
        except BaseException as e:
            self.write_failure(output, e)
            from soto import workdir

            workdir.interrupted_hint()
            raise

    def segment(self, kind: str, start: int, end: int, label: str | None = None, **details) -> None:
        """One span [start, end) of the output, in output frame indices.
        Segments tile the output end to end, and each becomes a chapter."""
        if end <= start:
            return
        self.data["segments"].append({
            "kind": kind, "label": label or kind, "start_frame": start, "end_frame": end, **details,
        })

    def _finish(self, fps: float | None) -> None:
        self.data["finished_at"] = _now()
        self.data["elapsed_s"] = round(time.perf_counter() - self._t0, 2)
        if fps is None:
            return
        for seg in self.data["segments"]:
            seg["start_s"] = round(seg["start_frame"] / fps, 3)
            seg["end_s"] = round(seg["end_frame"] / fps, 3)
            seg["start_tc"] = timecode(seg["start_frame"], fps)
            seg["end_tc"] = timecode(seg["end_frame"], fps)
        self.data["segments"].sort(key=lambda s: s["start_frame"])

    def write(self, output: Path, fps: float) -> Path | None:
        if not self.enabled:
            return None
        self.data["status"] = "ok"
        self._finish(fps)
        chapters = self.data["segments"]
        total = max((s["end_frame"] for s in chapters), default=0)
        self.data["output"] = {
            "path": str(output.resolve()),
            "fps": fps,
            "frames": total,
            "seconds": round(total / fps, 3),
            "timecode_end": timecode(total, fps),
        }

        sidecar = output.with_suffix(".json")
        sidecar.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        _failure_path(output).unlink(missing_ok=True)  # from an earlier crashed run
        print(f"Metadata: {sidecar}")
        if chapters:
            try:
                _embed_chapters(output, chapters, fps)
            except (OSError, subprocess.CalledProcessError) as e:
                print(f"  warning: couldn't embed chapter markers in {output}: {e}")
        return sidecar

    def write_failure(self, output: Path, exc: BaseException) -> Path | None:
        """Writes everything logged so far, plus the error, to a separate
        `<output>.failed.json`, so a crash never overwrites the sidecar of an
        earlier successful run's output. Never raises: it mustn't mask `exc`."""
        if not self.enabled:
            return None
        self.data["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        self.data["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc)),
        }
        inputs = self.data["inputs"]
        self._finish(inputs[0]["fps"] if inputs else None)
        path = _failure_path(output)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        except OSError as e:
            print(f"  warning: couldn't write failure record {path}: {e}")
            return None
        print(f"Failure record: {path}")
        return path


def _failure_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failed.json")


def _embed_chapters(output: Path, chapters: list[dict], fps: float) -> None:
    """Adds `chapters` to `output` in place (stream copy, no re-encode)."""
    lines = [";FFMETADATA1"]
    for seg in chapters:
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={round(seg['start_frame'] / fps * 1000)}",
            f"END={round(seg['end_frame'] / fps * 1000)}",
            f"title={_escape(seg['label'])}",
        ]
    meta = output.with_name(f"{output.stem}.chapters.txt")
    tmp = output.with_name(f"{output.stem}.chapters{output.suffix}")
    meta.write_text("\n".join(lines) + "\n")
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(output), "-i", str(meta),
            "-map", "0", "-map_metadata", "0", "-map_chapters", "1", "-c", "copy", str(tmp),
        ]
        subprocess.run(cmd, check=True)
        tmp.replace(output)
    finally:
        meta.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
