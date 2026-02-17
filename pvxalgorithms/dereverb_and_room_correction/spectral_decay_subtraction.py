#!/usr/bin/env python3
"""Spectral decay subtraction DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'dereverb_and_room_correction.spectral_decay_subtraction'
ALGORITHM_NAME = 'Spectral decay subtraction'
THEME = 'Dereverb and Room Correction'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Spectral decay subtraction on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
