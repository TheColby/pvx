#!/usr/bin/env python3

"""Compare TimeStretch/PitchShift backends on tiny deterministic signals."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pvx.augment import PitchShift, TimeStretch


def _tone(freq: float, sr: int, duration_s: float) -> np.ndarray:
    t = np.arange(int(round(sr * duration_s)), dtype=np.float32) / float(sr)
    audio = 0.42 * np.sin(2.0 * np.pi * freq * t)
    audio += 0.08 * np.sin(2.0 * np.pi * freq * 2.0 * t)
    if audio.size > sr // 10:
        audio[sr // 10] += 0.35
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _dominant_freq(audio: np.ndarray, sr: int) -> float:
    if audio.size == 0:
        return float("nan")
    window = np.hanning(audio.shape[-1])
    spectrum = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(audio.shape[-1], d=1.0 / sr)
    return float(freqs[int(np.argmax(spectrum))])


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_case(
    *,
    engine: str,
    operation: str,
    audio: np.ndarray,
    sr: int,
    base_freq: float,
    stretch: float,
    semitones: float,
    wavelet: str,
    wavelet_levels: int,
    seed: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        if operation == "stretch":
            transform = TimeStretch(
                rate=(stretch, stretch),
                engine=engine,
                wavelet=wavelet,
                wavelet_levels=wavelet_levels,
                p=1.0,
            )
            expected_len = int(round(audio.shape[-1] * stretch))
            expected_f0 = base_freq
        else:
            transform = PitchShift(
                semitones=(semitones, semitones),
                engine=engine,
                wavelet=wavelet,
                wavelet_levels=wavelet_levels,
                p=1.0,
            )
            expected_len = int(audio.shape[-1])
            expected_f0 = base_freq * (2.0 ** (semitones / 12.0))
        out, _ = transform(audio, sr, seed=seed)
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        observed_f0 = _dominant_freq(out, sr)
        return {
            "engine": engine,
            "operation": operation,
            "status": "ok",
            "runtime_ms": elapsed_ms,
            "samples": int(out.shape[-1]),
            "expected_samples": expected_len,
            "sample_error": int(out.shape[-1] - expected_len),
            "dominant_freq_hz": observed_f0,
            "expected_freq_hz": expected_f0,
            "freq_error_hz": float(observed_f0 - expected_f0),
            "rms": _rms(out),
            "peak": float(np.max(np.abs(out))) if out.size else 0.0,
        }
    except Exception as exc:
        return {
            "engine": engine,
            "operation": operation,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_ms": 1000.0 * (time.perf_counter() - start),
        }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# pvx Backend Comparison",
        "",
        f"- sample_rate: `{report['sample_rate']}`",
        f"- input_samples: `{report['input_samples']}`",
        f"- base_freq_hz: `{report['base_freq_hz']}`",
        "",
        "| Engine | Operation | Status | Runtime ms | Samples | Freq Hz | Freq Error Hz | Note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        note = row.get("error", "")
        lines.append(
            "| {engine} | {operation} | {status} | {runtime_ms:.2f} | {samples} | "
            "{dominant_freq_hz:.2f} | {freq_error_hz:.2f} | {note} |".format(
                engine=row["engine"],
                operation=row["operation"],
                status=row["status"],
                runtime_ms=float(row.get("runtime_ms", 0.0)),
                samples=int(row.get("samples", 0)),
                dominant_freq_hz=float(row.get("dominant_freq_hz", float("nan"))),
                freq_error_hz=float(row.get("freq_error_hz", float("nan"))),
                note=note.replace("|", "\\|"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    sr = int(args.sample_rate)
    base_freq = float(args.base_freq)
    audio = _tone(base_freq, sr, float(args.duration))
    engines = [item.strip() for item in str(args.engines).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    for engine in engines:
        for operation in ("stretch", "pitch"):
            rows.append(
                _run_case(
                    engine=engine,
                    operation=operation,
                    audio=audio,
                    sr=sr,
                    base_freq=base_freq,
                    stretch=float(args.stretch),
                    semitones=float(args.semitones),
                    wavelet=str(args.wavelet),
                    wavelet_levels=int(args.wavelet_levels),
                    seed=int(args.seed),
                )
            )
    return {
        "benchmark": "pvx_backend_compare",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sample_rate": sr,
        "input_samples": int(audio.shape[-1]),
        "base_freq_hz": base_freq,
        "stretch": float(args.stretch),
        "semitones": float(args.semitones),
        "wavelet": str(args.wavelet),
        "wavelet_levels": int(args.wavelet_levels),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", default="pytorch,pvx-cli,wavelet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "out_backend_compare")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--base-freq", type=float, default=440.0)
    parser.add_argument("--stretch", type=float, default=1.5)
    parser.add_argument("--semitones", type=float, default=7.0)
    parser.add_argument("--wavelet", default="auto")
    parser.add_argument("--wavelet-levels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--strict", action="store_true", help="Return non-zero if any backend fails")
    args = parser.parse_args(argv)

    report = build_report(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "backend_compare.json"
    md_path = args.out_dir / "backend_compare.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"[backend-compare] report json -> {json_path}")
    print(f"[backend-compare] report md   -> {md_path}")
    if args.strict and any(row.get("status") != "ok" for row in report["rows"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
