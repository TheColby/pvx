#!/usr/bin/env python3
"""Grain-cloud pitch textures DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'granular_and_modulation.grain_cloud_pitch_textures'
ALGORITHM_NAME = 'Grain-cloud pitch textures'
THEME = 'Granular and Modulation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Grain-cloud pitch textures on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
