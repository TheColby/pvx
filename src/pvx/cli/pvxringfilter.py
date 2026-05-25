#!/usr/bin/env python3

"""Ring + resonator filter wrapper."""

from __future__ import annotations

from pvx.cli.pvxring import run_ring_cli


def main(argv: list[str] | None = None) -> int:
    return run_ring_cli(argv, default_operator="ringfilter", prog="pvx ringfilter")


if __name__ == "__main__":
    raise SystemExit(main())
