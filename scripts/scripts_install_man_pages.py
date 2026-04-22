#!/usr/bin/env python3

"""Generate and optionally install pvx man pages."""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "man" / "man1"
USER_MAN_DIR = Path.home() / ".local" / "share" / "man" / "man1"


PAGES: tuple[tuple[str, list[str], str], ...] = (
    ("pvx", [sys.executable, "-m", "pvx.cli.pvx", "--help"], "Unified command-line interface for pvx"),
    ("pvxvoc", [sys.executable, "-m", "pvx.core.voc", "--help"], "Phase-vocoder time/pitch processor"),
    ("pvxfreeze", [sys.executable, "-m", "pvx.cli.pvxfreeze", "--help"], "Spectral freeze renderer"),
    ("pvxfollow", [sys.executable, "-m", "pvx.cli.pvx", "follow", "--help"], "Single-command sidechain helper"),
    ("pvxchain", [sys.executable, "-m", "pvx.cli.pvx", "chain", "--help"], "Managed serial chain runner"),
)


def _run_help(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    text = (proc.stdout or "") + ((("\n" + proc.stderr) if proc.stderr else ""))
    if proc.returncode != 0 and not text.strip():
        raise RuntimeError(f"Failed to collect help text: {' '.join(cmd)}")
    return text.strip() or "(no help text emitted)"


def _roff_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("-", r"\-")


def _build_man_page(name: str, summary: str, help_text: str) -> str:
    date_label = _dt.datetime.now().strftime("%Y-%m-%d")
    header = [
        f'.TH "{name.upper()}" "1" "{date_label}" "pvx" "User Commands"',
        ".SH NAME",
        f"{name} \\- {summary}",
        ".SH SYNOPSIS",
        f".B {name}",
        ".SH DESCRIPTION",
        "Auto-generated from built-in --help output.",
        ".SH OPTIONS",
        ".nf",
    ]
    body = [_roff_escape(line.rstrip()) for line in help_text.splitlines()]
    footer = [
        ".fi",
        ".SH ATTRIBUTION",

        "",
    ]
    return "\n".join(header + body + footer)


def _write_pages(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, cmd, summary in PAGES:
        help_text = _run_help(cmd)
        content = _build_man_page(name=name, summary=summary, help_text=help_text)
        path = out_dir / f"{name}.1"
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def _install_pages(paths: list[Path], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for src in paths:
        dst = target_dir / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        installed.append(dst)
    return installed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate pvx man pages and optionally install them to user manpath.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for generated man pages (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--install-user",
        action="store_true",
        help=f"Also install generated pages into {USER_MAN_DIR}",
    )
    args = parser.parse_args(argv)

    written = _write_pages(Path(args.out_dir).expanduser().resolve())
    print(f"[man] generated {len(written)} page(s) in {Path(args.out_dir).expanduser().resolve()}")
    for path in written:
        print(f"[man]   {path}")

    if args.install_user:
        installed = _install_pages(written, USER_MAN_DIR)
        print(f"[man] installed {len(installed)} page(s) in {USER_MAN_DIR}")
        print(f"[man] hint: export MANPATH=\"{USER_MAN_DIR.parent}:$MANPATH\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
