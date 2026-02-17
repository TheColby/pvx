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

    print("pvx toolkit: use pvxvoc/pvxfreeze/pvxharmonize/.../pvxlayer")
    if args.list_tools:
        print("Tools: pvxvoc, pvxfreeze, pvxharmonize, pvxconform, pvxmorph, pvxwarp, pvxformant, pvxtransient, pvxunison, pvxdenoise, pvxdeverb, pvxretune, pvxlayer")
    if args.list_algorithm_package:
        print("Algorithm package: /Users/cleider/dev/pvx/pvxalgorithms/")
        print("Algorithm registry: /Users/cleider/dev/pvx/pvxalgorithms/registry.py")
    if args.show_docs:
        print("Python file help: /Users/cleider/dev/pvx/docs/PYTHON_FILE_HELP.md")
        print("Algorithm params: /Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md")
    if not (args.list_tools or args.list_algorithm_package or args.show_docs):
        print("Use --help for options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
