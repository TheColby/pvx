#!/usr/bin/env python3
"""Time-varying cents maps DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'retune_and_intonation.time_varying_cents_maps'
ALGORITHM_NAME = 'Time-varying cents maps'
THEME = 'Retune and Intonation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Time-varying cents maps on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
