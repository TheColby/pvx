#!/usr/bin/env python3
"""Key-aware retuning with confidence weighting DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'retune_and_intonation.key_aware_retuning_with_confidence_weighting'
ALGORITHM_NAME = 'Key-aware retuning with confidence weighting'
THEME = 'Retune and Intonation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Key-aware retuning with confidence weighting on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
