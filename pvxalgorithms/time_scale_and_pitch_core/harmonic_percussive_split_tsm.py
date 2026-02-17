#!/usr/bin/env python3
"""Harmonic/percussive split TSM DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'time_scale_and_pitch_core.harmonic_percussive_split_tsm'
ALGORITHM_NAME = 'Harmonic/percussive split TSM'
THEME = 'Time-Scale and Pitch Core'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Harmonic/percussive split TSM on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
