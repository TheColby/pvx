#!/usr/bin/env python3
"""Spectral dynamics (bin-wise compressor/expander) DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'dynamics_and_loudness.spectral_dynamics_bin_wise_compressor_expander'
ALGORITHM_NAME = 'Spectral dynamics (bin-wise compressor/expander)'
THEME = 'Dynamics and Loudness'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Spectral dynamics (bin-wise compressor/expander) on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
