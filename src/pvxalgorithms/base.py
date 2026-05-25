"""Compatibility shim for `pvxalgorithms.base`."""

from __future__ import annotations

import warnings

warnings.warn(
    "`pvxalgorithms.base` is deprecated; import from `pvx.algorithms.base` instead. "
    "The compatibility alias is planned for removal in v0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

from pvx.algorithms.base import *  # noqa: F403
