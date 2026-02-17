#!/usr/bin/env python3
"""Spectral contrast exaggeration DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'creative_spectral_effects.spectral_contrast_exaggeration'
ALGORITHM_NAME = 'Spectral contrast exaggeration'
THEME = 'Creative Spectral Effects'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Spectral contrast exaggeration on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
