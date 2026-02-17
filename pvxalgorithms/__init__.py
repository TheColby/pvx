"""Compatibility shim for `pvxalgorithms` namespace.

Use `pvx.algorithms` as the canonical import path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_LEGACY_SUBMODULE_ROOT = _SRC / "pvx" / "algorithms"

sys.path.insert(0, str(_SRC))
if _LEGACY_SUBMODULE_ROOT.exists():
    __path__.append(str(_LEGACY_SUBMODULE_ROOT))  # type: ignore[name-defined]

from pvx.algorithms import *  # noqa: F401,F403
from pvx.algorithms import ALGORITHM_COUNT, ALGORITHM_REGISTRY

__all__ = ["ALGORITHM_COUNT", "ALGORITHM_REGISTRY"]
