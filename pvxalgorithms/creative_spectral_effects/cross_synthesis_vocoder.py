#!/usr/bin/env python3
"""Cross-synthesis vocoder DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'creative_spectral_effects.cross_synthesis_vocoder'
ALGORITHM_NAME = 'Cross-synthesis vocoder'
THEME = 'Creative Spectral Effects'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Cross-synthesis vocoder on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
