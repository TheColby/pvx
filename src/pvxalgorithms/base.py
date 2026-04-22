"""Compatibility shim for `pvxalgorithms.base`."""

from __future__ import annotations

import warnings

warnings.warn(
    "`pvxalgorithms.base` is deprecated; import from `pvx.algorithms.base` instead.",
    DeprecationWarning,
    stacklevel=2,
)

from pvx.algorithms.base import *  # noqa: F403
