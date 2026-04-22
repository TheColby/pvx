#!/usr/bin/env python3

"""Chordmapper wrapper."""

from __future__ import annotations

from pvx.cli.pvxharmmap import run_harmmap_cli


def main(argv: list[str] | None = None) -> int:
    return run_harmmap_cli(argv, default_operator="chordmapper", prog="pvx chordmapper")


if __name__ == "__main__":
    raise SystemExit(main())
