
"""Trajectory-aware multichannel convolution reverb helpers."""

from __future__ import annotations

import math

import numpy as np

try:
    from scipy.signal import fftconvolve  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback when scipy is unavailable
    fftconvolve = None  # type: ignore[assignment]


def _parse_float_triplet(text: str) -> tuple[float, float, float]:
    parts = [chunk.strip() for chunk in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got: {text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_coordinate(text: str, coord_system: str) -> np.ndarray:
    x, y, z = _parse_float_triplet(text)
    if coord_system == "cartesian":
        return np.array([x, y, z], dtype=np.float64)
    if coord_system == "spherical":
        az = math.radians(x)
        el = math.radians(y)
        r = float(z)
        cos_el = math.cos(el)
        return np.array(
            [
                r * cos_el * math.cos(az),
                r * cos_el * math.sin(az),
                r * math.sin(el),
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported coord_system: {coord_system}")


def parse_speaker_angles(text: str, channels: int) -> np.ndarray:
    entries = [chunk.strip() for chunk in str(text).split(";") if chunk.strip()]
    if len(entries) != channels:
        raise ValueError(
            f"--speaker-angles defined {len(entries)} positions but impulse response has {channels} channels"
        )
    rows: list[tuple[float, float]] = []
    for entry in entries:
        parts = [p.strip() for p in entry.split(",")]
        if len(parts) < 2:
            raise ValueError(f"Speaker angle entry must be az,el (degrees); got {entry!r}")
        rows.append((float(parts[0]), float(parts[1])))
    return np.asarray(rows, dtype=np.float64)


def default_speaker_angles(channels: int) -> np.ndarray:
    if channels <= 0:
        raise ValueError("channels must be > 0")
    if channels == 1:
        return np.asarray([(0.0, 0.0)], dtype=np.float64)
    if channels == 2:
        return np.asarray([(-30.0, 0.0), (30.0, 0.0)], dtype=np.float64)
    if channels == 4:
        return np.asarray(
            [(-45.0, 0.0), (45.0, 0.0), (135.0, 0.0), (-135.0, 0.0)], dtype=np.float64
        )
    az = np.linspace(0.0, 360.0, num=channels, endpoint=False, dtype=np.float64)
    el = np.zeros_like(az)
    return np.stack([az, el], axis=1)


def _angles_to_unit_vectors(angles_deg: np.ndarray) -> np.ndarray:
    if angles_deg.ndim != 2 or angles_deg.shape[1] != 2:
        raise ValueError("angles_deg must have shape (channels, 2) with azimuth/elevation degrees")
    az = np.deg2rad(angles_deg[:, 0])
    el = np.deg2rad(angles_deg[:, 1])
    cos_el = np.cos(el)
    vec = np.stack(
        [
            cos_el * np.cos(az),
            cos_el * np.sin(az),
            np.sin(el),
        ],
        axis=1,
    )
    denom = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(denom, 1e-12)


def _shape_curve(t: np.ndarray, shape: str) -> np.ndarray:
    if shape == "linear":
        return t
    if shape == "ease-in":
        return t * t
    if shape == "ease-out":
        inv = 1.0 - t
        return 1.0 - (inv * inv)
    if shape == "ease-in-out":
        return (3.0 * t * t) - (2.0 * t * t * t)
    raise ValueError(f"Unsupported trajectory shape: {shape}")


def compute_trajectory_gains(
    samples: int,
    channels: int,
    *,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    speaker_angles_deg: np.ndarray | None = None,
    shape: str = "linear",
    distance_law: str = "inverse",
    normalize_per_sample: bool = True,
) -> np.ndarray:
    if samples <= 0:
        return np.zeros((0, channels), dtype=np.float64)
    if channels <= 0:
        raise ValueError("channels must be > 0")

    if speaker_angles_deg is None:
        speaker_angles_deg = default_speaker_angles(channels)
    if speaker_angles_deg.shape[0] != channels:
        raise ValueError("speaker layout channel count must match output channels")

    speaker_dirs = _angles_to_unit_vectors(np.asarray(speaker_angles_deg, dtype=np.float64))
    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    end = np.asarray(end_xyz, dtype=np.float64).reshape(3)

    t = np.linspace(0.0, 1.0, num=samples, endpoint=True, dtype=np.float64)
    curve = _shape_curve(t, shape)
    traj = start[None, :] + (curve[:, None] * (end - start)[None, :])
    dist = np.linalg.norm(traj, axis=1)
    src_dirs = traj / np.maximum(dist[:, None], 1e-12)

    # Hemispherical cosine-like panning with non-negative support.
    dots = np.matmul(src_dirs, speaker_dirs.T)
    weights = np.maximum(0.0, 0.5 * (1.0 + dots))
    if normalize_per_sample:
        weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)

    if distance_law == "none":
        distance_gain = np.ones_like(dist)
    elif distance_law == "inverse":
        distance_gain = 1.0 / np.maximum(dist, 0.25)
    elif distance_law == "inverse-square":
        distance_gain = 1.0 / np.maximum(dist, 0.25) ** 2.0
    else:
        raise ValueError(f"Unsupported distance law: {distance_law}")

    return weights * distance_gain[:, None]


def _fft_convolve_or_fallback(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    if fftconvolve is not None:
        return np.asarray(fftconvolve(x, h, mode="full"), dtype=np.float64)
    return np.asarray(np.convolve(x, h, mode="full"), dtype=np.float64)


def apply_multichannel_trajectory_reverb(
    source_mono: np.ndarray,
    impulse_response: np.ndarray,
    gains: np.ndarray,
    *,
    wet: float = 1.0,
    dry: float = 0.0,
) -> np.ndarray:
    x = np.asarray(source_mono, dtype=np.float64).reshape(-1)
    h = np.asarray(impulse_response, dtype=np.float64)
    if h.ndim == 1:
        h = h[:, None]
    if gains.shape[0] != x.size:
        raise ValueError("gains length must equal source length")
    if gains.shape[1] != h.shape[1]:
        raise ValueError("gains channels must equal impulse-response channels")
    if not (0.0 <= wet <= 2.0):
        raise ValueError("wet must be in [0, 2]")
    if not (0.0 <= dry <= 2.0):
        raise ValueError("dry must be in [0, 2]")

    n = x.size
    ir_len = h.shape[0]
    channels = h.shape[1]
    out_len = n + ir_len - 1
    out = np.zeros((out_len, channels), dtype=np.float64)
    if dry > 0.0:
        out[:n, :] += dry * (x[:, None] * gains)
    for ch in range(channels):
        weighted = x * gains[:, ch]
        out[:, ch] += wet * _fft_convolve_or_fallback(weighted, h[:, ch])
    return out


def resample_audio_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError("sample rates must be positive")
    if src_sr == dst_sr or arr.size == 0:
        return arr
    out_len = max(1, int(round(arr.shape[0] * float(dst_sr) / float(src_sr))))
    x_old = np.linspace(0.0, 1.0, arr.shape[0], endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, out_len, endpoint=False, dtype=np.float64)
    out = np.zeros((out_len, arr.shape[1]), dtype=np.float64)
    for ch in range(arr.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, arr[:, ch])
    return out
