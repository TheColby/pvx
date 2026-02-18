"""Stereo/multichannel helper utilities."""

from __future__ import annotations

import math

import numpy as np


def validate_ref_channel(ref_channel: int, channels: int) -> int:
    ch = int(ref_channel)
    if channels <= 0:
        raise ValueError("channels must be > 0")
    if ch < 0 or ch >= channels:
        raise ValueError(f"ref-channel must be in [0, {channels - 1}]")
    return ch


def lr_to_ms(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("mid/side conversion requires exactly 2 channels")
    scale = 1.0 / math.sqrt(2.0)
    mid = (arr[:, 0] + arr[:, 1]) * scale
    side = (arr[:, 0] - arr[:, 1]) * scale
    return np.stack([mid, side], axis=1)


def ms_to_lr(audio_ms: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio_ms, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("M/S decoding requires exactly 2 channels")
    scale = 1.0 / math.sqrt(2.0)
    left = (arr[:, 0] + arr[:, 1]) * scale
    right = (arr[:, 0] - arr[:, 1]) * scale
    return np.stack([left, right], axis=1)

