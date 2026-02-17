#!/usr/bin/env python3
"""AM/FM/ring modulation blocks DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'granular_and_modulation.am_fm_ring_modulation_blocks'
ALGORITHM_NAME = 'AM/FM/ring modulation blocks'
THEME = 'Granular and Modulation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run AM/FM/ring modulation blocks on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
