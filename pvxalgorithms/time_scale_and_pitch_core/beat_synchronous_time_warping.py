#!/usr/bin/env python3
"""Beat-synchronous time warping DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'time_scale_and_pitch_core.beat_synchronous_time_warping'
ALGORITHM_NAME = 'Beat-synchronous time warping'
THEME = 'Time-Scale and Pitch Core'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Beat-synchronous time warping on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
