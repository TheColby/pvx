"""Compatibility shim for `pvxalgorithms` namespace.

Use `pvx.algorithms` as the canonical import path.
"""

from __future__ import annotations

from pathlib import Path

_LEGACY_SUBMODULE_ROOT = Path(__file__).resolve().parents[1] / "pvx" / "algorithms"
if _LEGACY_SUBMODULE_ROOT.exists():
    __path__.append(str(_LEGACY_SUBMODULE_ROOT))  # type: ignore[name-defined]

from pvx.algorithms import *  # noqa: F401,F403
from pvx.algorithms import ALGORITHM_COUNT, ALGORITHM_REGISTRY

__all__ = ["ALGORITHM_COUNT", "ALGORITHM_REGISTRY"]
