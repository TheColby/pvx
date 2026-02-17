#!/usr/bin/env python3
"""Compatibility wrapper.

This root module forwards imports/execution to `pvx.core.common` after the
src-layout migration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pvx.core.common import *  # noqa: F401,F403


if __name__ == "__main__":
    print("pvxcommon is a library module. Use pvxvoc/pvxfreeze/... for CLI workflows.")
