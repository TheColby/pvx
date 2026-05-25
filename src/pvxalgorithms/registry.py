"""Compatibility shim for `pvxalgorithms.registry`."""

from __future__ import annotations

import warnings

warnings.warn(
    "`pvxalgorithms.registry` is deprecated; import from `pvx.algorithms.registry` instead. "
    "The compatibility alias is planned for removal in v0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

from pvx.algorithms.registry import *  # noqa: F403
