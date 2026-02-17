#!/usr/bin/env python3
"""Probabilistic latent component separation DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'separation_and_decomposition.probabilistic_latent_component_separation'
ALGORITHM_NAME = 'Probabilistic latent component separation'
THEME = 'Separation and Decomposition'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Probabilistic latent component separation on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
