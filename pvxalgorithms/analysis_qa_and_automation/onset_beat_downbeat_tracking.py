#!/usr/bin/env python3
"""Onset/beat/downbeat tracking DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'analysis_qa_and_automation.onset_beat_downbeat_tracking'
ALGORITHM_NAME = 'Onset/beat/downbeat tracking'
THEME = 'Analysis, QA, and Automation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Onset/beat/downbeat tracking on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
