#!/usr/bin/env python3
"""Top-level project helper CLI.

This file exists as a lightweight entrypoint that points users to the
specialized `pvx*` command-line tools and the generated algorithm library.
Run `python3 main.py --help` to view quick navigation commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="pvx toolkit navigator (tool list, algorithm package, and docs index)",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the primary pvx CLI tools",
    )
    parser.add_argument(
        "--list-algorithm-package",
        action="store_true",
        help="Print location of pvxalgorithms package and registry files",
    )
    parser.add_argument(
        "--show-docs",
        action="store_true",
        help="Print key generated documentation paths",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    canonical_algorithm_dir = repo_root / "src" / "pvx" / "algorithms"
    legacy_shim_dir = repo_root / "pvxalgorithms"

    print("pvx toolkit: use pvxvoc/pvxfreeze/pvxharmonize/.../pvxlayer")
    if args.list_tools:
        print("Tools: pvxvoc, pvxfreeze, pvxharmonize, pvxconform, pvxmorph, pvxwarp, pvxformant, pvxtransient, pvxunison, pvxdenoise, pvxdeverb, pvxretune, pvxlayer")
    if args.list_algorithm_package:
        print(f"Algorithm package (canonical): {canonical_algorithm_dir}/")
        print(f"Algorithm registry (canonical): {canonical_algorithm_dir / 'registry.py'}")
        print(f"Compatibility shim package: {legacy_shim_dir}/")
    if args.show_docs:
        print(f"Python file help: {repo_root / 'docs' / 'PYTHON_FILE_HELP.md'}")
        print(f"Algorithm params: {repo_root / 'docs' / 'pvx_ALGORITHM_PARAMS.md'}")
    if not (args.list_tools or args.list_algorithm_package or args.show_docs):
        print("Use --help for options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
