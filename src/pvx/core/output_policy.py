"""Shared output policy helpers for bit depth, dither, true-peak, and metadata sidecars."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


BIT_DEPTH_CHOICES: tuple[str, ...] = ("inherit", "16", "24", "32f")
DITHER_CHOICES: tuple[str, ...] = ("none", "tpdf")
METADATA_POLICY_CHOICES: tuple[str, ...] = ("none", "sidecar", "copy")

_BIT_DEPTH_TO_SUBTYPE: dict[str, str] = {
    "16": "PCM_16",
    "24": "PCM_24",
    "32f": "FLOAT",
}

_SUBTYPE_TO_BITS: dict[str, int] = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
}


def db_to_amplitude(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _resample_linear_1d(signal: np.ndarray, output_samples: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = x.size
    m = max(1, int(output_samples))
    if n == 0:
        return np.zeros(m, dtype=np.float64)
    if n == m:
        return x.copy()
    x_old = np.linspace(0.0, 1.0, num=n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=m, endpoint=False)
    return np.interp(x_new, x_old, x).astype(np.float64)


def true_peak_dbtp(audio: np.ndarray, sample_rate: int, *, oversample: int = 4) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return float("-120.0")

    factor = max(1, int(oversample))
    max_peak = 0.0
    for ch in range(arr.shape[1]):
        data = arr[:, ch]
        if data.size == 0:
            continue
        up_n = max(1, int(data.size * factor))
        up = _resample_linear_1d(data, up_n)
        max_peak = max(max_peak, float(np.max(np.abs(up))))
    return float(20.0 * np.log10(max(max_peak, 1e-12)))


def enforce_true_peak_limit(audio: np.ndarray, sample_rate: int, max_dbtp: float) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return arr
    current = true_peak_dbtp(arr, int(sample_rate))
    target_amp = db_to_amplitude(max_dbtp)
    current_amp = db_to_amplitude(current)
    if current_amp <= target_amp + 1e-15:
        return arr
    scale = target_amp / max(current_amp, 1e-15)
    return arr * float(scale)


def resolve_output_subtype(
    args: argparse.Namespace,
    *,
    explicit_subtype: str | None = None,
) -> str | None:
    if getattr(args, "subtype", None):
        return str(getattr(args, "subtype"))
    bit_depth = str(getattr(args, "bit_depth", "inherit") or "inherit").lower()
    if bit_depth in _BIT_DEPTH_TO_SUBTYPE:
        return _BIT_DEPTH_TO_SUBTYPE[bit_depth]
    return explicit_subtype


def subtype_bit_depth(subtype: str | None) -> int | None:
    if subtype is None:
        return None
    return _SUBTYPE_TO_BITS.get(str(subtype).upper())


def apply_dither_if_needed(audio: np.ndarray, args: argparse.Namespace, *, subtype: str | None) -> np.ndarray:
    mode = str(getattr(args, "dither", "none") or "none").lower()
    if mode == "none":
        return np.asarray(audio, dtype=np.float64)
    if mode != "tpdf":
        return np.asarray(audio, dtype=np.float64)

    bits = subtype_bit_depth(subtype)
    if bits is None:
        # Skip dithering for floating-point/unknown output formats.
        return np.asarray(audio, dtype=np.float64)

    arr = np.asarray(audio, dtype=np.float64).copy()
    levels = float(2 ** (bits - 1))
    lsb = 1.0 / levels
    seed = getattr(args, "dither_seed", None)
    rng = np.random.default_rng(None if seed is None else int(seed))
    noise = (rng.random(arr.shape) - rng.random(arr.shape)) * lsb
    arr = arr + noise
    arr = np.clip(arr, -1.0, 1.0)
    return arr


def prepare_output_audio(
    audio: np.ndarray,
    sample_rate: int,
    args: argparse.Namespace,
    *,
    explicit_subtype: str | None = None,
) -> tuple[np.ndarray, str | None]:
    """Apply true-peak limiting + dithering and return processed audio + resolved subtype."""
    subtype = resolve_output_subtype(args, explicit_subtype=explicit_subtype)
    out = np.asarray(audio, dtype=np.float64)

    tp_limit = getattr(args, "true_peak_max_dbtp", None)
    if tp_limit is not None:
        out = enforce_true_peak_limit(out, int(sample_rate), float(tp_limit))

    out = apply_dither_if_needed(out, args, subtype=subtype)
    return out, subtype


def source_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if str(path) == "-":
        return {"path": "-", "note": "stdin source"}
    try:
        info = sf.info(str(path))
        return {
            "path": str(path),
            "samplerate": int(getattr(info, "samplerate", 0) or 0),
            "channels": int(getattr(info, "channels", 0) or 0),
            "frames": int(getattr(info, "frames", 0) or 0),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
            "format": str(getattr(info, "format", "")),
            "subtype": str(getattr(info, "subtype", "")),
        }
    except Exception as exc:  # pragma: no cover - defensive metadata path
        return {"path": str(path), "error": str(exc)}


def write_metadata_sidecar(
    *,
    output_path: Path,
    input_path: Path | None,
    audio: np.ndarray,
    sample_rate: int,
    subtype: str | None,
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    policy = str(getattr(args, "metadata_policy", "none") or "none").lower()
    if policy == "none":
        return None
    if str(output_path) == "-":
        return None

    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    payload: dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata_policy": policy,
        "input": source_metadata(input_path),
        "output": {
            "path": str(output_path),
            "sample_rate": int(sample_rate),
            "channels": int(arr.shape[1]) if arr.ndim == 2 else 1,
            "samples": int(arr.shape[0]) if arr.ndim == 2 else int(arr.size),
            "duration_s": float((arr.shape[0] if arr.ndim == 2 else arr.size) / max(1, int(sample_rate))),
            "subtype": subtype,
            "true_peak_dbtp": float(true_peak_dbtp(arr, int(sample_rate))),
        },
        "output_policy": {
            "bit_depth": str(getattr(args, "bit_depth", "inherit")),
            "dither": str(getattr(args, "dither", "none")),
            "dither_seed": getattr(args, "dither_seed", None),
            "true_peak_max_dbtp": getattr(args, "true_peak_max_dbtp", None),
        },
        "note": (
            "Embedded metadata copy is not guaranteed by libsndfile; pvx writes a deterministic JSON sidecar."
            if policy == "copy"
            else "Sidecar metadata written by pvx."
        ),
    }
    if extra:
        payload["extra"] = extra

    sidecar = output_path.with_suffix(output_path.suffix + ".metadata.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def validate_output_policy_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    bit_depth = str(getattr(args, "bit_depth", "inherit") or "inherit").lower()
    if bit_depth not in BIT_DEPTH_CHOICES:
        parser.error(f"--bit-depth must be one of: {', '.join(BIT_DEPTH_CHOICES)}")
    dither = str(getattr(args, "dither", "none") or "none").lower()
    if dither not in DITHER_CHOICES:
        parser.error(f"--dither must be one of: {', '.join(DITHER_CHOICES)}")
    if dither != "none" and bit_depth == "32f":
        parser.error("--dither cannot be used with floating-point --bit-depth 32f")
    if getattr(args, "dither_seed", None) is not None and int(getattr(args, "dither_seed")) < 0:
        parser.error("--dither-seed must be >= 0")
    tp = getattr(args, "true_peak_max_dbtp", None)
    if tp is not None and not np.isfinite(float(tp)):
        parser.error("--true-peak-max-dbtp must be finite")
    policy = str(getattr(args, "metadata_policy", "none") or "none").lower()
    if policy not in METADATA_POLICY_CHOICES:
        parser.error(f"--metadata-policy must be one of: {', '.join(METADATA_POLICY_CHOICES)}")
