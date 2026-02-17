#!/usr/bin/env python3
"""Ambisonics encode/decode DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'spatial_and_multichannel.ambisonics_encode_decode'
ALGORITHM_NAME = 'Ambisonics encode/decode'
THEME = 'Spatial and Multichannel'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Ambisonics encode/decode on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
