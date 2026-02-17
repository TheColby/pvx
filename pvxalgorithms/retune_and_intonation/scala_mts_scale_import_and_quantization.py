#!/usr/bin/env python3
"""Scala/MTS scale import and quantization DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'retune_and_intonation.scala_mts_scale_import_and_quantization'
ALGORITHM_NAME = 'Scala/MTS scale import and quantization'
THEME = 'Retune and Intonation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Scala/MTS scale import and quantization on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
