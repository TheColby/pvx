#!/usr/bin/env python3
"""Adaptive intonation (context-sensitive intervals) DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'retune_and_intonation.adaptive_intonation_context_sensitive_intervals'
ALGORITHM_NAME = 'Adaptive intonation (context-sensitive intervals)'
THEME = 'Retune and Intonation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Adaptive intonation (context-sensitive intervals) on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
