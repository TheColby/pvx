#!/usr/bin/env python3
"""Silence/speech/music classifiers DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'analysis_qa_and_automation.silence_speech_music_classifiers'
ALGORITHM_NAME = 'Silence/speech/music classifiers'
THEME = 'Analysis, QA, and Automation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Silence/speech/music classifiers on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
