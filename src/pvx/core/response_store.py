#!/usr/bin/env python3

"""Persistent frequency-response artifact storage.

PVXRF schema:
- container: NumPy NPZ (compressed)
- required members:
  - meta_json: UTF-8 JSON metadata payload
  - frequencies_hz: float64 array, shape (bins,)
  - magnitude: float64 array, shape (channels, bins)
  - phase: float64 array, shape (channels, bins)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pvx.core.analysis_store import AnalysisArtifact, analysis_digest

PVXRF_SCHEMA = "PVXRF"
PVXRF_VERSION = 1

ResponseMethod = Literal["median", "mean", "rms", "max"]
PhaseMode = Literal["mean", "zero", "first"]
NormalizeMode = Literal["none", "peak", "rms"]


@dataclass(frozen=True)
class ResponseArtifact:
    """In-memory representation of a PVXRF artifact."""

    sample_rate: int
    bins: int
    channels: int
    frequencies_hz: np.ndarray  # shape (bins,)
    magnitude: np.ndarray  # shape (channels, bins)
    phase: np.ndarray  # shape (channels, bins)
    method: ResponseMethod
    phase_mode: PhaseMode
    normalize: NormalizeMode
    smoothing_bins: int
    source_analysis_sha256: str | None = None
    source_path: str | None = None
    schema: str = PVXRF_SCHEMA
    version: int = PVXRF_VERSION


def _canonical_meta(artifact: ResponseArtifact) -> dict[str, Any]:
    return {
        "schema": str(artifact.schema),
        "version": int(artifact.version),
        "sample_rate": int(artifact.sample_rate),
        "bins": int(artifact.bins),
        "channels": int(artifact.channels),
        "method": str(artifact.method),
        "phase_mode": str(artifact.phase_mode),
        "normalize": str(artifact.normalize),
        "smoothing_bins": int(artifact.smoothing_bins),
        "source_analysis_sha256": None
        if artifact.source_analysis_sha256 is None
        else str(artifact.source_analysis_sha256),
        "source_path": None if artifact.source_path is None else str(artifact.source_path),
    }


def _moving_average_1d(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(int(width), dtype=np.float64) / float(width)
    pad = int(width) // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: x.size]


def _aggregate_magnitude(mag: np.ndarray, method: ResponseMethod) -> np.ndarray:
    if method == "median":
        return np.median(mag, axis=1)
    if method == "mean":
        return np.mean(mag, axis=1)
    if method == "rms":
        return np.sqrt(np.mean(np.square(mag), axis=1))
    if method == "max":
        return np.max(mag, axis=1)
    raise ValueError(f"Unsupported response method: {method}")


def _aggregate_phase(phase: np.ndarray, mode: PhaseMode) -> np.ndarray:
    if mode == "zero":
        return np.zeros((phase.shape[0], phase.shape[2]), dtype=np.float64)
    if phase.shape[1] == 0:
        return np.zeros((phase.shape[0], phase.shape[2]), dtype=np.float64)
    if mode == "first":
        return np.asarray(phase[:, 0, :], dtype=np.float64)
    if mode == "mean":
        unit = np.exp(1j * phase)
        return np.angle(np.mean(unit, axis=1))
    raise ValueError(f"Unsupported phase mode: {mode}")


def _normalize_magnitude(mag: np.ndarray, normalize: NormalizeMode) -> np.ndarray:
    out = np.asarray(mag, dtype=np.float64)
    if normalize == "none":
        return out
    if normalize == "peak":
        peak = float(np.max(out))
        if peak > 1e-12:
            return out / peak
        return out
    if normalize == "rms":
        rms = float(np.sqrt(np.mean(np.square(out))))
        if rms > 1e-12:
            return out / rms
        return out
    raise ValueError(f"Unsupported normalize mode: {normalize}")


def response_digest(artifact: ResponseArtifact) -> str:
    """Return deterministic SHA256 digest for metadata + response payload."""
    digest = hashlib.sha256()
    meta_json = json.dumps(_canonical_meta(artifact), sort_keys=True, separators=(",", ":"))
    digest.update(meta_json.encode("utf-8"))
    digest.update(np.ascontiguousarray(artifact.frequencies_hz, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(artifact.magnitude, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(artifact.phase, dtype=np.float64).tobytes())
    return digest.hexdigest()


def summarize_response_artifact(artifact: ResponseArtifact) -> dict[str, Any]:
    magnitude = np.asarray(artifact.magnitude, dtype=np.float64)
    return {
        "schema": artifact.schema,
        "version": artifact.version,
        "sample_rate": artifact.sample_rate,
        "bins": artifact.bins,
        "channels": artifact.channels,
        "method": artifact.method,
        "phase_mode": artifact.phase_mode,
        "normalize": artifact.normalize,
        "smoothing_bins": artifact.smoothing_bins,
        "source_path": artifact.source_path,
        "source_analysis_sha256": artifact.source_analysis_sha256,
        "magnitude_min": float(np.min(magnitude)) if magnitude.size else 0.0,
        "magnitude_max": float(np.max(magnitude)) if magnitude.size else 0.0,
        "sha256": response_digest(artifact),
    }


def response_from_analysis(
    analysis: AnalysisArtifact,
    *,
    method: ResponseMethod = "median",
    phase_mode: PhaseMode = "mean",
    normalize: NormalizeMode = "peak",
    smoothing_bins: int = 1,
) -> ResponseArtifact:
    """Derive frequency-response artifact from a PVXAN analysis artifact."""
    if int(smoothing_bins) <= 0:
        raise ValueError("smoothing_bins must be >= 1")
    spec = np.asarray(analysis.spectrum, dtype=np.complex128)  # (channels, frames, bins)
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    mag_agg = _aggregate_magnitude(magnitude, method=method)
    if smoothing_bins > 1:
        smooth = np.zeros_like(mag_agg)
        for channel in range(smooth.shape[0]):
            smooth[channel, :] = _moving_average_1d(mag_agg[channel, :], int(smoothing_bins))
        mag_agg = smooth
    mag_agg = _normalize_magnitude(mag_agg, normalize=normalize)
    phase_agg = _aggregate_phase(phase, mode=phase_mode)
    freqs = np.linspace(
        0.0, float(analysis.sample_rate) * 0.5, int(analysis.bins), dtype=np.float64
    )
    return ResponseArtifact(
        sample_rate=int(analysis.sample_rate),
        bins=int(analysis.bins),
        channels=int(analysis.channels),
        frequencies_hz=freqs,
        magnitude=np.asarray(mag_agg, dtype=np.float64),
        phase=np.asarray(phase_agg, dtype=np.float64),
        method=method,
        phase_mode=phase_mode,
        normalize=normalize,
        smoothing_bins=int(smoothing_bins),
        source_analysis_sha256=analysis_digest(analysis),
        source_path=analysis.source_path,
    )


def save_response_artifact(path: str | Path, artifact: ResponseArtifact) -> Path:
    """Save ResponseArtifact to compressed NPZ file."""
    out_path = Path(path).expanduser().resolve()
    if out_path.suffix.lower() != ".npz":
        out_path = Path(f"{out_path}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(_canonical_meta(artifact), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        out_path,
        meta_json=np.asarray(meta_json),
        frequencies_hz=np.asarray(artifact.frequencies_hz, dtype=np.float64),
        magnitude=np.asarray(artifact.magnitude, dtype=np.float64),
        phase=np.asarray(artifact.phase, dtype=np.float64),
    )
    return out_path


def load_response_artifact(path: str | Path) -> ResponseArtifact:
    """Load ResponseArtifact from compressed NPZ file."""
    in_path = Path(path).expanduser().resolve()
    if not in_path.exists() and in_path.suffix.lower() != ".npz":
        alt = Path(f"{in_path}.npz")
        if alt.exists():
            in_path = alt
    with np.load(in_path, allow_pickle=False) as payload:
        for key in ("meta_json", "frequencies_hz", "magnitude", "phase"):
            if key not in payload:
                raise ValueError(f"PVXRF payload missing required member: {key}")

        meta_raw = payload["meta_json"]
        if isinstance(meta_raw, np.ndarray):
            meta_text = str(meta_raw.item()) if meta_raw.shape == () else str(meta_raw.tolist())
        else:
            meta_text = str(meta_raw)
        meta = json.loads(meta_text)
        schema = str(meta.get("schema", ""))
        if schema != PVXRF_SCHEMA:
            raise ValueError(f"Unsupported response schema '{schema}'. Expected '{PVXRF_SCHEMA}'.")
        version = int(meta.get("version", -1))
        if version != PVXRF_VERSION:
            raise ValueError(
                f"Unsupported response schema version '{version}'. Expected '{PVXRF_VERSION}'."
            )

        freqs = np.asarray(payload["frequencies_hz"], dtype=np.float64)
        mag = np.asarray(payload["magnitude"], dtype=np.float64)
        phase = np.asarray(payload["phase"], dtype=np.float64)
        if freqs.ndim != 1:
            raise ValueError("PVXRF frequencies_hz must be a 1D vector")
        if mag.ndim != 2 or phase.ndim != 2:
            raise ValueError("PVXRF magnitude/phase arrays must be 2D (channels, bins)")
        if mag.shape != phase.shape:
            raise ValueError("PVXRF magnitude and phase shapes differ")
        if mag.shape[1] != freqs.size:
            raise ValueError("PVXRF bins mismatch between frequency and magnitude arrays")

        return ResponseArtifact(
            sample_rate=int(meta["sample_rate"]),
            bins=int(meta.get("bins", freqs.size)),
            channels=int(meta.get("channels", mag.shape[0])),
            frequencies_hz=freqs,
            magnitude=mag,
            phase=phase,
            method=str(meta.get("method", "median")),  # type: ignore[arg-type]
            phase_mode=str(meta.get("phase_mode", "mean")),  # type: ignore[arg-type]
            normalize=str(meta.get("normalize", "none")),  # type: ignore[arg-type]
            smoothing_bins=int(meta.get("smoothing_bins", 1)),
            source_analysis_sha256=None
            if meta.get("source_analysis_sha256") is None
            else str(meta.get("source_analysis_sha256")),
            source_path=None if meta.get("source_path") is None else str(meta.get("source_path")),
            schema=schema,
            version=version,
        )
