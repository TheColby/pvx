#!/usr/bin/env python3

"""Copy the repo-local Homebrew formula into a tap checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FORMULA = ROOT / "Formula" / "pvx.rb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync Formula/pvx.rb into a local Homebrew tap checkout.",
    )
    parser.add_argument(
        "tap_checkout",
        type=Path,
        help="Path to a checked-out homebrew-* tap repository.",
    )
    parser.add_argument(
        "--formula",
        type=Path,
        default=DEFAULT_FORMULA,
        help="Formula source path (default: Formula/pvx.rb in this repo).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    formula_path = Path(args.formula).resolve()
    tap_root = Path(args.tap_checkout).resolve()

    if not formula_path.exists():
        parser.error(f"Formula not found: {formula_path}")
    if not tap_root.exists():
        parser.error(f"Tap checkout not found: {tap_root}")
    if not tap_root.name.startswith("homebrew-"):
        parser.error(f"Tap checkout should be a homebrew-* repository clone, got: {tap_root.name}")

    target_dir = tap_root / "Formula"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "pvx.rb"
    shutil.copy2(formula_path, target_path)
    print(f"Synced {formula_path} -> {target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
