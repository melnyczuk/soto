"""`soto take`: every soto run as a numbered take of a scene, kept in order.

    soto take run A4S03 --brief "carry the sock on past the cut" --cloud -- \\
        extend rushes/windsock.mp4 --frames 32 --preview fast
    soto take run A4S03 --brief "same, full quality" --cloud -- \\
        extend rushes/windsock.mp4 --frames 32 --seed 1234
    soto take run A4S03 --brief "and again from that" -- extend @t002 --frames 32
    soto take mark A4S03_t002 select
    soto take list A4S03

Takes live under the takes root (--root, or SOTO_TAKES), one folder per
scene:

    <root>/A4S03/
        takes.csv           the ledger: one line per take, oldest first
        A4S03_t001.mov      ProRes 422 HQ master, for Resolve (+ its .json)
        A4S03_t002.mov
        raw/                what the model made (.mkv + .json), for crossfade

A scene's takes share one work dir, so a refinement reuses what earlier
takes already made. It's kept out of the takes root, which may well be a
synced folder: with --cloud it stays in Modal's soto-jobs volume, and
locally it's under $SOTO_WORK (default ~/.cache/soto/work).

A take's file is never written twice: a new idea is a new take. `@t002` (or
`@A4S03_t002` for another scene's) in the command stands for that take's
master, so a take can be made from another. The generation commands
(transition, bridge, extend, dream) are spliced into their footage with
`crossfade` (--fade frames) straight into the master; `expand` writes it
directly. With --cloud the generation runs on Modal (see cloud.py) and the
splice here.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import typer

from soto import config

app = typer.Typer(help="Numbered takes of a scene, with a ledger, as ProRes masters for Resolve.",
                  no_args_is_help=True)

STATUSES = ("draft", "review", "select", "rejected", "failed")
FIELDS = ("take", "status", "created", "brief", "parent", "command", "seed", "where", "elapsed_s", "soto",
          "file", "note")
# Commands whose output is the model's frames alone, spliced in by crossfade.
_SPLICED = ("transition", "bridge", "extend", "dream")
# Commands that keep a work dir (--work).
_WORKED = ("transition", "bridge", "extend", "expand")
_SCENE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_TAKE = re.compile(r"^(?P<scene>.+)_t(?P<n>\d{3,})$")
_REF = re.compile(r"^@(?:(?P<scene>.+)_)?t(?P<n>\d{3,})$")

_ROOT_HELP = "Where takes are kept (default: $SOTO_TAKES)."


def _root(root: Path | None) -> Path:
    if root is None:
        env = os.environ.get("SOTO_TAKES")
        if not env:
            raise typer.BadParameter("set SOTO_TAKES (e.g. to the film's renders/soto folder), or pass --root",
                                     param_hint="--root")
        root = Path(env)
    return root.expanduser().resolve()


def take_id(scene: str, n: int) -> str:
    return f"{scene}_t{n:03d}"


class Scene:
    def __init__(self, root: Path, name: str):
        if not _SCENE.match(name):
            raise typer.BadParameter(f"{name!r}: use letters, digits, '-', '_' or '.'", param_hint="scene")
        self.name = name
        self.dir = root / name
        self.ledger = self.dir / "takes.csv"
        self.raw = self.dir / "raw"
        work_root = Path(os.environ.get("SOTO_WORK", "~/.cache/soto/work")).expanduser()
        self.work = work_root / f"{name}.work"

    def rows(self) -> list[dict]:
        if not self.ledger.is_file():
            return []
        with self.ledger.open(newline="") as f:
            return list(csv.DictReader(f))

    def write(self, rows: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(self.ledger)

    def next_number(self) -> int:
        used = [int(m["n"]) for r in self.rows() if (m := _TAKE.match(r["take"]))]
        used += [int(m["n"]) for p in self.dir.glob(f"{self.name}_t*.*") if (m := _TAKE.match(p.stem))]
        used += [int(m["n"]) for p in self.raw.glob(f"{self.name}_t*.*") if (m := _TAKE.match(p.stem))]
        return max(used, default=0) + 1

    def master(self, take: str) -> Path:
        return self.dir / f"{take}.mov"


def _resolve_ref(root: Path, scene: Scene, arg: str) -> tuple[str, str] | None:
    """(take id, its master's path) for an `@t002` / `@A4S03_t002` argument."""
    m = _REF.match(arg)
    if not m:
        return None
    other = Scene(root, m["scene"]) if m["scene"] else scene
    take = take_id(other.name, int(m["n"]))
    path = other.master(take)
    if not path.is_file():
        raise typer.BadParameter(f"{arg}: no master at {path}")
    return take, str(path)


def _git() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=config.REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return commit + ("-dirty" if dirty else "")


def _soto_local(args: list[str]) -> int:
    cmd = [sys.executable, "-c", "import sys; sys.argv[0] = 'soto'; from soto.cli import app; app()", *args]
    return subprocess.run(cmd).returncode


def _has(args: list[str], *options: str) -> bool:
    return any(a.partition("=")[0] in options for a in args)


@app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    scene: str = typer.Argument(..., help="Scene id from the shot list, e.g. A4S03."),
    brief: str = typer.Option("", "--brief", help="What this take is for, in your words. Goes in the ledger."),
    cloud: bool = typer.Option(False, "--cloud", help="Generate on a Modal GPU (soto-cloud); splice here."),
    gpu: str = typer.Option(None, "--gpu", help="Modal GPU with --cloud (default L4, or $SOTO_CLOUD_GPU)."),
    fade: int = typer.Option(8, "--fade", min=0, help="Crossfade frames when splicing into the footage."),
    root: Path = typer.Option(None, "--root", file_okay=False, help=_ROOT_HELP),
) -> None:
    """Make the scene's next take: `soto take run SCENE [options] -- COMMAND ARGS...`.
    COMMAND ARGS is an ordinary soto command without -o; `@t002` in it stands
    for an earlier take's master."""
    args = list(ctx.args)
    if not args:
        raise typer.BadParameter("give the soto command after --, e.g. -- extend clip.mp4 --frames 32")
    command = args[0]
    if command not in (*_SPLICED, "expand"):
        raise typer.BadParameter(f"{command!r}: takes are made with {', '.join((*_SPLICED, 'expand'))}")
    if _has(args, "-o", "--output", "--work", "--no-work", "--no-metadata"):
        raise typer.BadParameter("leave out -o, --work, --no-work and --no-metadata: the take sets them")

    root_dir = _root(root)
    sc = Scene(root_dir, scene)
    take = take_id(sc.name, sc.next_number())

    parent = ""
    resolved = []
    for a in args:
        ref = _resolve_ref(root_dir, sc, a)
        if ref:
            parent = parent or ref[0]
            resolved.append(ref[1])
        else:
            if not parent and not a.startswith("-") and Path(a).expanduser().is_file():
                parent = Path(a).name
            resolved.append(a)

    master = sc.master(take)
    sc.dir.mkdir(parents=True, exist_ok=True)
    gen_out = sc.raw / f"{take}.mkv" if command in _SPLICED else master
    full = [*resolved, "-o", str(gen_out)]
    if command in _WORKED:
        full += ["--work", str(sc.work)]
    sc.raw.mkdir(parents=True, exist_ok=True)

    row = {
        "take": take, "status": "failed", "created": datetime.now().isoformat(timespec="seconds"),
        "brief": brief, "parent": parent, "command": " ".join(args), "seed": "",
        "where": f"modal {gpu or os.environ.get('SOTO_CLOUD_GPU', 'L4')}" if cloud else "local",
        "elapsed_s": "", "soto": _git(), "file": "", "note": "",
    }
    print(f"{take}: soto {' '.join(full)}" + (" (on Modal)" if cloud else ""))
    t0 = datetime.now()
    try:
        if cloud:
            from soto import cloud as cloud_mod

            code = cloud_mod.run_remote(full, gpu=gpu or cloud_mod.DEFAULT_GPU)
        else:
            code = _soto_local(full)
        if code == 0 and command in _SPLICED:
            code = _soto_local(["crossfade", str(gen_out), "-o", str(master), "--frames", str(fade)])
    except BaseException:
        sc.write([*sc.rows(), row])
        raise
    row["elapsed_s"] = str(round((datetime.now() - t0).total_seconds()))

    seed_from = gen_out.with_suffix(".json")
    if seed_from.is_file():
        seed = json.loads(seed_from.read_text()).get("settings", {}).get("seed")
        row["seed"] = "" if seed is None else str(seed)
    if code == 0 and master.is_file():
        row["status"] = "draft" if _has(args, "--preview") else "review"
        row["file"] = str(master.relative_to(root_dir))
    sc.write([*sc.rows(), row])
    if row["status"] == "failed":
        print(f"{take}: failed (logged in {sc.ledger})")
        raise typer.Exit(code or 1)
    print(f"{take}: {row['status']} -> {master}")


@app.command("mark")
def mark(
    take: str = typer.Argument(..., help="A take, e.g. A4S03_t002."),
    status: str = typer.Argument(..., help=f"One of {', '.join(STATUSES[:4])}."),
    note: str = typer.Option(None, "--note", help="A note to keep with it (replaces the last)."),
    root: Path = typer.Option(None, "--root", file_okay=False, help=_ROOT_HELP),
) -> None:
    """Set a take's status. Selecting one moves the scene's previous select back to review."""
    m = _TAKE.match(take)
    if not m:
        raise typer.BadParameter(f"{take!r} isn't a take id like A4S03_t002", param_hint="take")
    if status not in STATUSES[:4]:
        raise typer.BadParameter(f"must be one of {list(STATUSES[:4])}", param_hint="status")
    sc = Scene(_root(root), m["scene"])
    rows = sc.rows()
    if not any(r["take"] == take for r in rows):
        raise typer.BadParameter(f"no {take} in {sc.ledger}", param_hint="take")
    for r in rows:
        if r["take"] == take:
            r["status"] = status
            if note is not None:
                r["note"] = note
        elif status == "select" and r["status"] == "select":
            r["status"] = "review"
            print(f"{r['take']}: select -> review")
    sc.write(rows)
    print(f"{take}: {status}")


@app.command("list")
def list_(
    scene: str = typer.Argument(None, help="A scene; all of them if left out."),
    root: Path = typer.Option(None, "--root", file_okay=False, help=_ROOT_HELP),
) -> None:
    """Show the ledger: every take, its status and brief."""
    root_dir = _root(root)
    names = [scene] if scene else sorted(p.parent.name for p in root_dir.glob("*/takes.csv"))
    for name in names:
        rows = Scene(root_dir, name).rows()
        if not rows:
            continue
        print(name)
        for r in rows:
            mark_ = "*" if r["status"] == "select" else " "
            print(f" {mark_} {r['take']:<14} {r['status']:<8} {r['parent'] or '-':<22} {r['brief']}")


@app.command("path")
def path(
    take: str = typer.Argument(..., help="A take, e.g. A4S03_t002."),
    root: Path = typer.Option(None, "--root", file_okay=False, help=_ROOT_HELP),
) -> None:
    """Print a take's master path."""
    m = _TAKE.match(take)
    if not m:
        raise typer.BadParameter(f"{take!r} isn't a take id like A4S03_t002", param_hint="take")
    print(Scene(_root(root), m["scene"]).master(take))
