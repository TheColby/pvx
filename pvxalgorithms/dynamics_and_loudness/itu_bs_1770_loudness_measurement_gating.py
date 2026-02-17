#!/usr/bin/env python3
"""ITU BS.1770 loudness measurement/gating DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'dynamics_and_loudness.itu_bs_1770_loudness_measurement_gating'
ALGORITHM_NAME = 'ITU BS.1770 loudness measurement/gating'
THEME = 'Dynamics and Loudness'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run ITU BS.1770 loudness measurement/gating on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
