#!/usr/bin/env python3
"""Minimum-statistics noise tracking DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'denoise_and_restoration.minimum_statistics_noise_tracking'
ALGORITHM_NAME = 'Minimum-statistics noise tracking'
THEME = 'Denoise and Restoration'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Minimum-statistics noise tracking on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
