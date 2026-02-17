#!/usr/bin/env python3
"""Sinusoidal+residual+transient decomposition DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'separation_and_decomposition.sinusoidal_residual_transient_decomposition'
ALGORITHM_NAME = 'Sinusoidal+residual+transient decomposition'
THEME = 'Separation and Decomposition'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Sinusoidal+residual+transient decomposition on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
