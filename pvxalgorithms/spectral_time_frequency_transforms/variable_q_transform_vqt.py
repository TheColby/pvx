#!/usr/bin/env python3
"""Variable-Q Transform (VQT) DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'spectral_time_frequency_transforms.variable_q_transform_vqt'
ALGORITHM_NAME = 'Variable-Q Transform (VQT)'
THEME = 'Spectral and Time-Frequency Transforms'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Variable-Q Transform (VQT) on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
