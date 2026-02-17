"""Compatibility shim for `pvxalgorithms.registry`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pvx.algorithms.registry import *  # noqa: F401,F403
