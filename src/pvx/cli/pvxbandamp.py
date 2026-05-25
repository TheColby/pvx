#!/usr/bin/env python3

"""Response-peak band emphasis wrapper."""

from __future__ import annotations

from pvx.cli.pvxfilter import run_filter_cli


def main(argv: list[str] | None = None) -> int:
    return run_filter_cli(argv, default_operator="bandamp", prog="pvx bandamp")


if __name__ == "__main__":
    raise SystemExit(main())
