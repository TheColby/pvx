#!/usr/bin/env python3
"""WPE dereverberation DSP implementation module."""

from __future__ import annotations

from typing import Any

import numpy as np

from pvxalgorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'dereverb_and_room_correction.wpe_dereverberation'
ALGORITHM_NAME = 'WPE dereverberation'
THEME = 'Dereverb and Room Correction'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run WPE dereverberation on the provided audio buffer."""
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )
