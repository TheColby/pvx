"""Compatibility shim for `pvxalgorithms.registry`."""

from __future__ import annotations

import warnings

warnings.warn(
    "`pvxalgorithms.registry` is deprecated; import from `pvx.algorithms.registry` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pvx.algorithms.registry import *  # noqa: F403
