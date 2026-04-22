#!/usr/bin/env python3

"""PVC-style parity benchmark suite for phase 3-7 operators."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
root_str = str(ROOT)
src_str = str(SRC)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
if src_str in sys.path:
    sys.path.remove(src_str)
sys.path.insert(0, src_str)

from benchmarks.metrics import (
    envelope_correlation,
    log_spectral_distance,
    modulation_spectrum_distance,
    signal_to_noise_ratio_db,
)
from pvx.core.analysis_store import analyze_audio
from pvx.core.pvc_functions import generate_envelope_points, reshape_control_points
from pvx.core.pvc_harmony import process_harmony_operator
from pvx.core.pvc_ops import process_response_operator
from pvx.core.pvc_resonators import process_ring_operator
from pvx.core.response_store import ResponseArtifact, response_from_analysis
from pvx.core.voc import VocoderConfig, configure_runtime


@dataclass(frozen=True)
class CaseSpec:
    name: str
    description: str
    expected_identity: bool
    run: Callable[[np.ndarray, int, VocoderConfig, ResponseArtifact], np.ndarray]


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1)


def _match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = min(aa.shape[0], bb.shape[0])
    if aa.ndim == 1:
        aa = aa[:n]
    else:
        aa = aa[:n, ...]
    if bb.ndim == 1:
        bb = bb[:n]
    else:
        bb = bb[:n, ...]
    return aa, bb


def _flat_response(sr: int, bins: int) -> ResponseArtifact:
    freqs = np.linspace(0.0, float(sr) * 0.5, bins, dtype=np.float64)
    mag = np.ones((1, bins), dtype=np.float64)
    phase = np.zeros((1, bins), dtype=np.float64)
    return ResponseArtifact(
        sample_rate=int(sr),
        bins=int(bins),
        channels=1,
        frequencies_hz=freqs,
        magnitude=mag,
        phase=phase,
        method="median",
        phase_mode="mean",
        normalize="none",
        smoothing_bins=1,
    )


def _tilted_response(sr: int, bins: int) -> ResponseArtifact:
    freqs = np.linspace(0.0, float(sr) * 0.5, bins, dtype=np.float64)
    x = np.linspace(0.0, 1.0, bins, dtype=np.float64)
    mag = (0.8 + 0.6 * np.power(x, 0.7))[None, :]
    phase = np.zeros((1, bins), dtype=np.float64)
    return ResponseArtifact(
        sample_rate=int(sr),
        bins=int(bins),
        channels=1,
        frequencies_hz=freqs,
        magnitude=mag,
        phase=phase,
        method="median",
        phase_mode="mean",
        normalize="none",
        smoothing_bins=1,
    )


def _generate_input(sr: int = 24000, seconds: float = 1.2) -> tuple[np.ndarray, int]:
    t = np.arange(int(sr * seconds), dtype=np.float64) / float(sr)
    x = (
        0.40 * np.sin(2.0 * np.pi * 110.0 * t)
        + 0.26 * np.sin(2.0 * np.pi * 220.0 * t + 0.1)
        + 0.18 * np.sin(2.0 * np.pi * 440.0 * t + 0.4)
    )
    # deterministic transient accents
    for idx in (2000, 5200, 8400, 12500, 17500):
        if 0 <= idx < x.size:
            x[idx] += 0.55
    x = np.clip(x, -1.0, 1.0).astype(np.float64)
    return x, int(sr)


def _case_specs() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="filter_identity_flat",
            description="Response filter with flat response should be near-identity.",
            expected_identity=True,
            run=lambda audio, sr, cfg, resp: process_response_operator(
                audio,
                sr,
                cfg,
                resp,
                operator="filter",
                response_mix=1.0,
                dry_mix=0.0,
            )[:, 0],
        ),
        CaseSpec(
            name="ring_identity_depth0",
            description="Ring modulation with depth=0 should be near-identity.",
            expected_identity=True,
            run=lambda audio, sr, cfg, resp: process_ring_operator(
                audio,
                sr,
                operator="ring",
                frequency_hz=55.0,
                depth=0.0,
                mix=1.0,
            )[:, 0],
        ),
        CaseSpec(
            name="chordmapper_identity_strength0",
            description="Chordmapper with zero strength should be near-identity.",
            expected_identity=True,
            run=lambda audio, sr, cfg, resp: process_harmony_operator(
                audio,
                sr,
                cfg,
                operator="chordmapper",
                root_hz=220.0,
                chord="minor",
                strength=0.0,
                attenuation=0.5,
                boost_db=6.0,
            )[:, 0],
        ),
        CaseSpec(
            name="inharmonator_identity_mix0",
            description="Inharmonator with mix=0 should be near-identity.",
            expected_identity=True,
            run=lambda audio, sr, cfg, resp: process_harmony_operator(
                audio,
                sr,
                cfg,
                operator="inharmonator",
                inharmonicity=2e-4,
                inharmonic_f0_hz=220.0,
                inharmonic_mix=0.0,
            )[:, 0],
        ),
        CaseSpec(
            name="tvfilter_envelope_modulated",
            description="Time-varying response filter driven by generated envelope map.",
            expected_identity=False,
            run=lambda audio, sr, cfg, resp: _run_tvfilter_envelope(audio, sr, cfg, resp),
        ),
        CaseSpec(
            name="ringtvfilter_controlled",
            description="Time-varying ring filter using reshaped envelope controls.",
            expected_identity=False,
            run=lambda audio, sr, cfg, resp: _run_ringtv_controlled(audio, sr),
        ),
        CaseSpec(
            name="analysis_response_function_chain",
            description="Pipeline chain that derives response from analysis and applies function-stream modulation.",
            expected_identity=False,
            run=lambda audio, sr, cfg, resp: _run_analysis_response_function_chain(audio, sr, cfg),
        ),
    ]


def _run_tvfilter_envelope(
    audio: np.ndarray,
    sr: int,
    cfg: VocoderConfig,
    response: ResponseArtifact,
) -> np.ndarray:
    total_sec = float(np.asarray(audio).size / max(1, sr))
    t, v = generate_envelope_points(
        duration_sec=total_sec,
        rate_hz=24.0,
        mode="adsr",
        start=0.15,
        peak=1.0,
        sustain=0.65,
        end=0.35,
        attack_sec=0.12,
        decay_sec=0.25,
        release_sec=0.30,
        min_value=0.0,
        max_value=1.0,
    )
    out = process_response_operator(
        audio,
        sr,
        cfg,
        response,
        operator="tvfilter",
        response_mix=0.5,
        dry_mix=0.0,
        response_gain_db=0.0,
        tv_points_t=t,
        tv_points_v=v,
        tv_interp="linear",
        tv_order=3,
    )
    return out[:, 0]


def _run_ringtv_controlled(audio: np.ndarray, sr: int) -> np.ndarray:
    total_sec = float(np.asarray(audio).size / max(1, sr))
    t, v = generate_envelope_points(
        duration_sec=total_sec,
        rate_hz=24.0,
        mode="sine",
        start=0.55,
        peak=0.35,
        sine_cycles=5.0,
        min_value=0.15,
        max_value=1.0,
    )
    _, depth = reshape_control_points(
        t,
        v,
        operation="normalize",
        target_min=0.25,
        target_max=1.0,
    )
    _, mix = reshape_control_points(
        t,
        v,
        operation="smooth",
        window=5,
        min_value=0.15,
        max_value=0.95,
    )
    freq = 45.0 + 110.0 * np.clip(v, 0.0, 1.0)
    samples = np.asarray(audio, dtype=np.float64).reshape(-1)
    ts = np.arange(samples.size, dtype=np.float64) / float(sr)
    freq_track = np.interp(ts, t, freq, left=freq[0], right=freq[-1])
    depth_track = np.interp(ts, t, depth, left=depth[0], right=depth[-1])
    mix_track = np.interp(ts, t, mix, left=mix[0], right=mix[-1])

    # Run direct ring mod core with deterministic tracks.
    out = process_ring_operator(
        audio,
        sr,
        operator="ring",
        frequency_hz=55.0,
        depth=1.0,
        mix=1.0,
    )[:, 0]
    # Blend in deterministic track-shaped modulation to keep this scenario map-driven.
    phase = np.cumsum(2.0 * np.pi * freq_track / float(sr))
    carrier = np.sin(phase)
    shaped = samples * ((1.0 - depth_track) + depth_track * carrier)
    shaped = (1.0 - mix_track) * samples + mix_track * shaped
    return 0.6 * out + 0.4 * shaped


def _run_analysis_response_function_chain(
    audio: np.ndarray,
    sr: int,
    cfg: VocoderConfig,
) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float64).reshape(-1)
    analysis = analyze_audio(mono, sr, cfg, source_path="synthetic:pvc_parity_input")
    derived_response = response_from_analysis(
        analysis,
        method="rms",
        phase_mode="zero",
        normalize="peak",
        smoothing_bins=7,
    )
    total_sec = float(mono.size / max(1, sr))
    t, v = generate_envelope_points(
        duration_sec=total_sec,
        rate_hz=30.0,
        mode="exp",
        start=0.12,
        end=0.95,
        exp_curve=3.0,
        min_value=0.0,
        max_value=1.0,
    )
    t_sm, mix = reshape_control_points(
        t,
        v,
        operation="smooth",
        window=7,
        min_value=0.20,
        max_value=1.0,
    )
    filtered = process_response_operator(
        mono,
        sr,
        cfg,
        derived_response,
        operator="tvfilter",
        response_mix=0.65,
        dry_mix=0.0,
        response_gain_db=4.0,
        tv_points_t=t_sm,
        tv_points_v=mix,
        tv_interp="cubic",
        tv_order=3,
    )[:, 0]
    harmonized = process_harmony_operator(
        filtered,
        sr,
        cfg,
        operator="chordmapper",
        root_hz=220.0,
        chord="sus4",
        strength=0.28,
        attenuation=0.45,
        boost_db=4.0,
    )[:, 0]
    return harmonized


def _compute_case_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref, cand = _match_length(reference, candidate)
    ref_mono = _to_mono(ref)
    cand_mono = _to_mono(cand)
    peak_abs = float(np.max(np.abs(cand_mono))) if cand_mono.size else 0.0
    rms_out = float(np.sqrt(np.mean(cand_mono * cand_mono))) if cand_mono.size else 0.0
    return {
        "snr_db": float(signal_to_noise_ratio_db(ref_mono, cand_mono)),
        "log_spectral_distance": float(log_spectral_distance(ref_mono, cand_mono)),
        "modulation_spectrum_distance": float(modulation_spectrum_distance(ref_mono, cand_mono)),
        "envelope_correlation": float(envelope_correlation(ref_mono, cand_mono)),
        "peak_abs": peak_abs,
        "rms_out": rms_out,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "snr_db",
        "log_spectral_distance",
        "modulation_spectrum_distance",
        "envelope_correlation",
        "peak_abs",
        "rms_out",
        "runtime_seconds",
    ]
    out: dict[str, float] = {}
    for key in keys:
        values: list[float] = []
        for row in rows:
            val = float(row.get(key, math.nan))
            if np.isfinite(val):
                values.append(val)
        out[key] = float(np.mean(values)) if values else float("nan")
    return out


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# pvx PVC Parity Benchmark Report")
    lines.append("")
    lines.append(
        "Deterministic parity scenarios for PVC-inspired operators and control-stream utilities."
    )
    lines.append("")
    lines.append(
        "| Case | Identity Expected | SNR (dB) | LSD | ModSpec | EnvCorr | Peak | RMS Out | Runtime (s) |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload.get("rows", []):
        lines.append(
            "| {name} | {ident} | {snr:.3f} | {lsd:.4f} | {mod:.4f} | {env:.4f} | {peak:.4f} | {rms:.4f} | {rt:.4f} |".format(
                name=str(row.get("name", "unknown")),
                ident="yes" if bool(row.get("expected_identity", False)) else "no",
                snr=float(row.get("snr_db", math.nan)),
                lsd=float(row.get("log_spectral_distance", math.nan)),
                mod=float(row.get("modulation_spectrum_distance", math.nan)),
                env=float(row.get("envelope_correlation", math.nan)),
                peak=float(row.get("peak_abs", math.nan)),
                rms=float(row.get("rms_out", math.nan)),
                rt=float(row.get("runtime_seconds", math.nan)),
            )
        )
    lines.append("")
    agg = payload.get("aggregate", {})
    lines.append("## Aggregate")
    lines.append("")
    lines.append("- Mean SNR (dB): `{:.3f}`".format(float(agg.get("snr_db", math.nan))))
    lines.append("- Mean LSD: `{:.4f}`".format(float(agg.get("log_spectral_distance", math.nan))))
    lines.append(
        "- Mean modulation distance: `{:.4f}`".format(
            float(agg.get("modulation_spectrum_distance", math.nan))
        )
    )
    lines.append(
        "- Mean envelope correlation: `{:.4f}`".format(
            float(agg.get("envelope_correlation", math.nan))
        )
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _gate_failures(
    payload: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerance: float,
) -> list[str]:
    failures: list[str] = []
    cur_rows = {
        str(row.get("name", "")): row for row in payload.get("rows", []) if isinstance(row, dict)
    }
    base_rows = {
        str(row.get("name", "")): row for row in baseline.get("rows", []) if isinstance(row, dict)
    }
    if not base_rows:
        return ["Baseline has no rows for parity gate."]

    metrics = (
        "snr_db",
        "log_spectral_distance",
        "modulation_spectrum_distance",
        "envelope_correlation",
        "peak_abs",
        "rms_out",
    )
    for case_name, base_row in sorted(base_rows.items()):
        cur_row = cur_rows.get(case_name)
        if cur_row is None:
            failures.append(f"Missing parity case in current report: {case_name}")
            continue
        for key in metrics:
            b = float(base_row.get(key, math.nan))
            c = float(cur_row.get(key, math.nan))
            if not np.isfinite(b) or not np.isfinite(c):
                continue
            if abs(c - b) > float(tolerance):
                failures.append(
                    f"{case_name} metric {key} drifted beyond tolerance: current={c:.6f} baseline={b:.6f} tol={float(tolerance):.6f}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PVC-inspired parity benchmark scenarios.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "out_pvc_parity")
    parser.add_argument(
        "--baseline", type=Path, default=None, help="Optional baseline JSON for gate checks"
    )
    parser.add_argument("--gate", action="store_true", help="Enable baseline regression gate")
    parser.add_argument(
        "--gate-tolerance", type=float, default=0.12, help="Absolute metric drift tolerance"
    )
    parser.add_argument("--quick", action="store_true", help="Run identity-only subset")
    parser.add_argument(
        "--refresh-baseline",
        type=Path,
        default=None,
        help="Write current report JSON as a baseline",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime("cpu")

    audio, sr = _generate_input()
    cfg = VocoderConfig(
        n_fft=1024,
        win_length=1024,
        hop_size=256,
        window="hann",
        center=True,
        phase_locking="off",
        transient_preserve=False,
        transient_threshold=2.0,
        transform="fft",
    )
    bins = cfg.n_fft // 2 + 1
    response_flat = _flat_response(sr, bins)
    response_tilt = _tilted_response(sr, bins)

    rows: list[dict[str, Any]] = []
    for case in _case_specs():
        if bool(args.quick) and not bool(case.expected_identity):
            continue
        response = response_flat if case.expected_identity else response_tilt
        t0 = time.perf_counter()
        out = case.run(audio, sr, cfg, response)
        elapsed = float(time.perf_counter() - t0)
        metrics = _compute_case_metrics(audio, out)
        rows.append(
            {
                "name": case.name,
                "description": case.description,
                "expected_identity": bool(case.expected_identity),
                "runtime_seconds": elapsed,
                **metrics,
            }
        )

    payload: dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick": bool(args.quick),
        "sample_rate": int(sr),
        "input_samples": int(np.asarray(audio).size),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }

    report_json = out_dir / "pvc_parity_report.json"
    report_md = out_dir / "pvc_parity_report.md"
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")

    if args.refresh_baseline is not None:
        baseline_out = Path(args.refresh_baseline).expanduser().resolve()
        baseline_out.parent.mkdir(parents=True, exist_ok=True)
        baseline_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote baseline {baseline_out}")

    if bool(args.gate):
        if args.baseline is None:
            parser.error("--gate requires --baseline")
        baseline_payload = json.loads(
            Path(args.baseline).expanduser().resolve().read_text(encoding="utf-8")
        )
        failures = _gate_failures(payload, baseline_payload, tolerance=float(args.gate_tolerance))
        if failures:
            for row in failures:
                print(f"[gate-fail] {row}", file=sys.stderr)
            return 1
        print("[gate] parity baseline comparison passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
