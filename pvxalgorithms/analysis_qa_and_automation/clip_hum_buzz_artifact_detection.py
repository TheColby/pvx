#!/usr/bin/env python3
"""Clip/hum/buzz artifact detection DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'analysis_qa_and_automation.clip_hum_buzz_artifact_detection'
ALGORITHM_NAME = 'Clip/hum/buzz artifact detection'
THEME = 'Analysis, QA, and Automation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run Clip/hum/buzz artifact detection on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
