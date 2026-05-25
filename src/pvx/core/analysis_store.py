#!/usr/bin/env python3

"""Persistent phase-vocoder analysis artifact storage.

PVXAN schema:
- container: NumPy NPZ (compressed)
- required members:
  - meta_json: UTF-8 JSON metadata payload
  - spectrum_real: float64 array, shape (channels, frames, bins)
  - spectrum_imag: float64 array, shape (channels, frames, bins)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pvx.core.voc import VocoderConfig, stft

PVXAN_SCHEMA = "PVXAN"
PVXAN_VERSION = 1


@dataclass(frozen=True)
class AnalysisArtifact:
    """In-memory representation of a PVXAN artifact."""

    sample_rate: int
    n_fft: int
    win_length: int
    hop_size: int
    window: str
    center: bool
    transform: str
    kaiser_beta: float
    channels: int
    frames: int
    bins: int
    spectrum: np.ndarray  # complex128, shape (channels, frames, bins)
    source_path: str | None = None
    created_utc: str | None = None
    schema: str = PVXAN_SCHEMA
    version: int = PVXAN_VERSION


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_complex_spectrum(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    if real.shape != imag.shape:
        raise ValueError("PVXAN payload mismatch: spectrum_real and spectrum_imag shapes differ")
    if real.ndim != 3:
        raise ValueError(
            "PVXAN payload mismatch: expected 3D spectrum arrays (channels, frames, bins)"
        )
    return np.asarray(real, dtype=np.float64) + 1j * np.asarray(imag, dtype=np.float64)


def _canonical_meta(artifact: AnalysisArtifact) -> dict[str, Any]:
    return {
        "schema": str(artifact.schema),
        "version": int(artifact.version),
        "created_utc": str(artifact.created_utc or _utc_now_iso()),
        "sample_rate": int(artifact.sample_rate),
        "n_fft": int(artifact.n_fft),
        "win_length": int(artifact.win_length),
        "hop_size": int(artifact.hop_size),
        "window": str(artifact.window),
        "center": bool(artifact.center),
        "transform": str(artifact.transform),
        "kaiser_beta": float(artifact.kaiser_beta),
        "channels": int(artifact.channels),
        "frames": int(artifact.frames),
        "bins": int(artifact.bins),
        "source_path": None if artifact.source_path is None else str(artifact.source_path),
    }


def analysis_digest(artifact: AnalysisArtifact) -> str:
    """Return deterministic SHA256 digest for metadata + spectrum payload."""
    digest = hashlib.sha256()
    meta_json = json.dumps(_canonical_meta(artifact), sort_keys=True, separators=(",", ":"))
    digest.update(meta_json.encode("utf-8"))
    spec = np.asarray(artifact.spectrum, dtype=np.complex128)
    digest.update(np.ascontiguousarray(spec.real, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(spec.imag, dtype=np.float64).tobytes())
    return digest.hexdigest()


def summarize_analysis_artifact(artifact: AnalysisArtifact) -> dict[str, Any]:
    duration_sec = 0.0
    if artifact.frames > 0 and artifact.sample_rate > 0:
        duration_sec = float(
            (max(0, artifact.frames - 1) * artifact.hop_size + artifact.n_fft)
            / artifact.sample_rate
        )
    return {
        "schema": artifact.schema,
        "version": artifact.version,
        "created_utc": artifact.created_utc,
        "source_path": artifact.source_path,
        "sample_rate": artifact.sample_rate,
        "channels": artifact.channels,
        "frames": artifact.frames,
        "bins": artifact.bins,
        "duration_sec_approx": duration_sec,
        "n_fft": artifact.n_fft,
        "win_length": artifact.win_length,
        "hop_size": artifact.hop_size,
        "window": artifact.window,
        "kaiser_beta": artifact.kaiser_beta,
        "center": artifact.center,
        "transform": artifact.transform,
        "sha256": analysis_digest(artifact),
    }


def analyze_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: VocoderConfig,
    *,
    source_path: str | None = None,
) -> AnalysisArtifact:
    """Compute STFT analysis payload and package as AnalysisArtifact."""
    work = np.asarray(audio, dtype=np.float64)
    if work.ndim == 1:
        work = work[:, None]
    if work.ndim != 2:
        raise ValueError("audio must be 1D mono or 2D (samples, channels)")

    channels = int(work.shape[1])
    stacks: list[np.ndarray] = []
    for channel in range(channels):
        spec = stft(work[:, channel], config)  # shape (bins, frames)
        stacks.append(np.asarray(spec, dtype=np.complex128).T)  # (frames, bins)
    spectrum = np.stack(stacks, axis=0)  # (channels, frames, bins)
    frames = int(spectrum.shape[1])
    bins = int(spectrum.shape[2])
    return AnalysisArtifact(
        sample_rate=int(sample_rate),
        n_fft=int(config.n_fft),
        win_length=int(config.win_length),
        hop_size=int(config.hop_size),
        window=str(config.window),
        center=bool(config.center),
        transform=str(config.transform),
        kaiser_beta=float(config.kaiser_beta),
        channels=channels,
        frames=frames,
        bins=bins,
        spectrum=spectrum,
        source_path=source_path,
        created_utc=_utc_now_iso(),
    )


def save_analysis_artifact(path: str | Path, artifact: AnalysisArtifact) -> Path:
    """Save AnalysisArtifact to compressed NPZ file."""
    out_path = Path(path).expanduser().resolve()
    if out_path.suffix.lower() != ".npz":
        out_path = Path(f"{out_path}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(_canonical_meta(artifact), sort_keys=True, separators=(",", ":"))
    spec = np.asarray(artifact.spectrum, dtype=np.complex128)
    np.savez_compressed(
        out_path,
        meta_json=np.asarray(meta_json),
        spectrum_real=np.asarray(spec.real, dtype=np.float64),
        spectrum_imag=np.asarray(spec.imag, dtype=np.float64),
    )
    return out_path


def load_analysis_artifact(path: str | Path) -> AnalysisArtifact:
    """Load AnalysisArtifact from compressed NPZ file."""
    in_path = Path(path).expanduser().resolve()
    if not in_path.exists() and in_path.suffix.lower() != ".npz":
        alt = Path(f"{in_path}.npz")
        if alt.exists():
            in_path = alt
    with np.load(in_path, allow_pickle=False) as payload:
        if "meta_json" not in payload:
            raise ValueError("PVXAN payload missing required member: meta_json")
        if "spectrum_real" not in payload or "spectrum_imag" not in payload:
            raise ValueError("PVXAN payload missing required spectrum members")

        meta_raw = payload["meta_json"]
        if isinstance(meta_raw, np.ndarray):
            if meta_raw.shape == ():
                meta_text = str(meta_raw.item())
            else:
                meta_text = str(meta_raw.tolist())
        else:
            meta_text = str(meta_raw)
        meta = json.loads(meta_text)
        schema = str(meta.get("schema", ""))
        if schema != PVXAN_SCHEMA:
            raise ValueError(f"Unsupported analysis schema '{schema}'. Expected '{PVXAN_SCHEMA}'.")
        version = int(meta.get("version", -1))
        if version != PVXAN_VERSION:
            raise ValueError(
                f"Unsupported analysis schema version '{version}'. Expected '{PVXAN_VERSION}'."
            )

        spectrum = _as_complex_spectrum(payload["spectrum_real"], payload["spectrum_imag"])
        channels, frames, bins = spectrum.shape

        return AnalysisArtifact(
            sample_rate=int(meta["sample_rate"]),
            n_fft=int(meta["n_fft"]),
            win_length=int(meta["win_length"]),
            hop_size=int(meta["hop_size"]),
            window=str(meta["window"]),
            center=bool(meta["center"]),
            transform=str(meta["transform"]),
            kaiser_beta=float(meta.get("kaiser_beta", 14.0)),
            channels=int(meta.get("channels", channels)),
            frames=int(meta.get("frames", frames)),
            bins=int(meta.get("bins", bins)),
            spectrum=spectrum,
            source_path=None if meta.get("source_path") is None else str(meta.get("source_path")),
            created_utc=None if meta.get("created_utc") is None else str(meta.get("created_utc")),
            schema=schema,
            version=version,
        )
