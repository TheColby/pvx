#!/usr/bin/env python3
"""Chirplet transform analysis DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'spectral_time_frequency_transforms.chirplet_transform_analysis'
ALGORITHM_NAME = 'Chirplet transform analysis'
THEME = 'Spectral and Time-Frequency Transforms'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Chirplet transform analysis on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
