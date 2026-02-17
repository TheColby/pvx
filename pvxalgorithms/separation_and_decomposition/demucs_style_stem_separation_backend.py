#!/usr/bin/env python3
"""Demucs-style stem separation backend DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'separation_and_decomposition.demucs_style_stem_separation_backend'
ALGORITHM_NAME = 'Demucs-style stem separation backend'
THEME = 'Separation and Decomposition'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Demucs-style stem separation backend on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
