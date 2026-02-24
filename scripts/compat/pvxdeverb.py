#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxdeverb` after the
src-layout migration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pvx.cli.pvxdeverb import *  # noqa: F401,F403
from pvx.cli.pvxdeverb import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
