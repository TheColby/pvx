#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Compatibility wrapper for the HPS pitch tracker CLI."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pvx.cli.hps_pitch_track import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
