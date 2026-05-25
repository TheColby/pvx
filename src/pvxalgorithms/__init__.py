"""Compatibility shim for `pvxalgorithms` namespace.

Use `pvx.algorithms` as the canonical import path.
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.warn(
    "`pvxalgorithms` is deprecated; import from `pvx.algorithms` instead. "
    "The compatibility alias is planned for removal in v0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

_LEGACY_SUBMODULE_ROOT = Path(__file__).resolve().parents[1] / "pvx" / "algorithms"
if _LEGACY_SUBMODULE_ROOT.exists():
    __path__.append(str(_LEGACY_SUBMODULE_ROOT))  # type: ignore[name-defined]

from pvx.algorithms import *  # noqa: F403
from pvx.algorithms import ALGORITHM_COUNT, ALGORITHM_REGISTRY

__all__ = ["ALGORITHM_COUNT", "ALGORITHM_REGISTRY"]
