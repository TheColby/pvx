#!/usr/bin/env python3
"""Viterbi-smoothed pitch contour tracking DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'pitch_detection_and_tracking.viterbi_smoothed_pitch_contour_tracking'
ALGORITHM_NAME = 'Viterbi-smoothed pitch contour tracking'
THEME = 'Pitch Detection and Tracking'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Viterbi-smoothed pitch contour tracking on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
