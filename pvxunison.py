#!/usr/bin/env python3
"""Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxunison` after the
src-layout migration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pvx.cli.pvxunison import *  # noqa: F401,F403
from pvx.cli.pvxunison import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
