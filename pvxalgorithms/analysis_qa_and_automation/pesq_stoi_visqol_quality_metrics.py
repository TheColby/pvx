#!/usr/bin/env python3
"""PESQ/STOI/VISQOL quality metrics DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics'
ALGORITHM_NAME = 'PESQ/STOI/VISQOL quality metrics'
THEME = 'Analysis, QA, and Automation'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run PESQ/STOI/VISQOL quality metrics on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
