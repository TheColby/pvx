#!/usr/bin/env python3
"""Diffusion-based speech/audio denoise DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'denoise_and_restoration.diffusion_based_speech_audio_denoise'
ALGORITHM_NAME = 'Diffusion-based speech/audio denoise'
THEME = 'Denoise and Restoration'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Diffusion-based speech/audio denoise on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
