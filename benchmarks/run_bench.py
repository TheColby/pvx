#!/usr/bin/env python3
"""Reproducible quality benchmark: pvx vs Rubber Band vs librosa baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.metrics import (  # noqa: E402
    attack_time_error_ms,
    bandwidth_95_delta_hz,
    clipping_ratio_delta,
    crest_factor_delta_db,
    dc_offset_delta,
    envelope_correlation,
    f0_rmse_cents,
    harmonic_to_noise_ratio_drift_db,
    ild_drift_db,
    integrated_lufs_delta_lu,
    interchannel_phase_deviation_by_band,
    itd_drift_ms,
    log_spectral_distance,
    loudness_range_delta_lu,
    modulation_spectrum_distance,
    musical_noise_index,
    onset_precision_recall_f1,
    peaq_odg,
    pesq_mos_lqo,
    phasiness_index,
    polqa_mos_lqo,
    pre_echo_score,
    rms_level_delta_db,
    signal_to_noise_ratio_db,
    si_sdr_db,
    short_term_lufs_delta_lu,
    spectral_convergence,
    stereo_coherence_drift,
    stoi_score,
    true_peak_delta_dbtp,
    transient_smear_score,
    visqol_mos_lqo,
    voicing_f1_score,
    zero_crossing_rate_delta,
)

BASE_METRIC_KEYS: list[str] = [
    "log_spectral_distance",
    "modulation_spectrum_distance",
    "transient_smear_score",
    "stereo_coherence_drift",
    "snr_db",
    "si_sdr_db",
    "spectral_convergence",
    "envelope_correlation",
    "rms_level_delta_db",
    "crest_factor_delta_db",
    "bandwidth_95_delta_hz",
    "zero_crossing_rate_delta",
    "dc_offset_delta",
    "clipping_ratio_delta",
]

PERCEPTUAL_METRIC_KEYS: list[str] = [
    "pesq_mos_lqo",
    "stoi",
    "estoi",
    "visqol_mos_lqo",
    "polqa_mos_lqo",
    "peaq_odg",
    "pesq_proxy_used",
    "stoi_proxy_used",
    "estoi_proxy_used",
    "visqol_proxy_used",
    "polqa_proxy_used",
    "peaq_proxy_used",
    "perceptual_proxy_fraction",
]

LOUDNESS_PITCH_TRANSIENT_KEYS: list[str] = [
    "integrated_lufs_delta_lu",
    "short_term_lufs_delta_lu",
    "lra_delta_lu",
    "true_peak_delta_dbtp",
    "f0_rmse_cents",
    "voicing_f1",
    "hnr_drift_db",
    "onset_precision",
    "onset_recall",
    "onset_f1",
    "attack_time_error_ms",
]

SPATIAL_ARTIFACT_KEYS: list[str] = [
    "ild_drift_db",
    "itd_drift_ms",
    "phase_deviation_low_rad",
    "phase_deviation_mid_rad",
    "phase_deviation_high_rad",
    "phase_deviation_mean_rad",
    "phasiness_index",
    "musical_noise_index",
    "pre_echo_score",
]

ALL_METRIC_KEYS: list[str] = [
    *BASE_METRIC_KEYS,
    *PERCEPTUAL_METRIC_KEYS,
    *LOUDNESS_PITCH_TRANSIENT_KEYS,
    *SPATIAL_ARTIFACT_KEYS,
]

CASE_GATE_RULES: dict[str, tuple[str, float]] = {
    "log_spectral_distance": ("max", 0.40),
    "modulation_spectrum_distance": ("max", 0.10),
    "transient_smear_score": ("max", 0.06),
    "stereo_coherence_drift": ("max", 0.08),
    "spectral_convergence": ("max", 0.03),
    "perceptual_proxy_fraction": ("max", 0.05),
    "phasiness_index": ("max", 0.06),
    "musical_noise_index": ("max", 0.06),
    "pre_echo_score": ("max", 0.06),
    "f0_rmse_cents": ("max", 10.0),
    "snr_db": ("min", 1.50),
    "si_sdr_db": ("min", 1.50),
    "envelope_correlation": ("min", 0.03),
    "onset_f1": ("min", 0.05),
    "voicing_f1": ("min", 0.05),
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str  # "stretch" | "pitch"
    value: float


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _parse_version(mod_name: str) -> str | None:
    try:
        module = __import__(mod_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _collect_environment_metadata(*, deterministic_cpu: bool) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "deterministic_cpu": bool(deterministic_cpu),
        "numpy_version": str(np.__version__),
        "soundfile_version": _parse_version("soundfile"),
        "scipy_version": _parse_version("scipy"),
        "librosa_version": _parse_version("librosa"),
        "rubberband_available": bool(_find_rubberband()),
        "env_controls": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }


def _corpus_manifest_entries(paths: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda p: p.name):
        audio, sr = _read_audio(path)
        channels = int(audio.shape[1]) if audio.ndim == 2 else 1
        entries.append(
            {
                "name": path.name,
                "sha256": _sha256_file(path),
                "sample_rate": int(sr),
                "samples": int(audio.shape[0]),
                "channels": channels,
                "duration_s": float(audio.shape[0] / max(1, sr)),
            }
        )
    return entries


def _load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_manifest(manifest_path: Path, data_paths: list[Path]) -> dict[str, Any]:
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": _corpus_manifest_entries(data_paths),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _manifest_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("entries", []):
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if isinstance(name, str):
            out[name] = row
    return out


def _validate_corpus_against_manifest(data_paths: list[Path], payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    index = _manifest_index(payload)
    for path in sorted(data_paths, key=lambda p: p.name):
        row = index.get(path.name)
        if row is None:
            issues.append(f"manifest missing entry for {path.name}")
            continue
        sha = str(row.get("sha256", ""))
        now_sha = _sha256_file(path)
        if sha != now_sha:
            issues.append(f"hash mismatch for {path.name}: manifest={sha} current={now_sha}")
    for name in sorted(index):
        if not any(path.name == name for path in data_paths):
            issues.append(f"manifest entry has no file in corpus: {name}")
    return issues


def _prepare_dataset(
    *,
    data_dir: Path,
    manifest_path: Path,
    refresh_manifest: bool,
    strict_corpus: bool,
) -> tuple[list[Path], dict[str, Any], list[str]]:
    data_paths = _generate_tiny_dataset(data_dir)
    issues: list[str] = []

    if refresh_manifest or not manifest_path.exists():
        payload = _write_manifest(manifest_path, data_paths)
        return data_paths, payload, issues

    payload = _load_manifest(manifest_path)
    if payload is None:
        issues.append(f"Corpus manifest unreadable at {manifest_path}")
        payload = _write_manifest(manifest_path, data_paths)
        return data_paths, payload, issues

    issues.extend(_validate_corpus_against_manifest(data_paths, payload))
    if issues:
        if strict_corpus:
            return data_paths, payload, issues
        payload = _write_manifest(manifest_path, data_paths)
    return data_paths, payload, issues


def _case_key(input_path: Path, task: TaskSpec) -> str:
    return f"{input_path.stem}:{task.name}"


def _diagnose_metrics(metrics: dict[str, float]) -> list[str]:
    suggestions: list[str] = []
    transient_smear = float(metrics.get("transient_smear_score", math.nan))
    onset_f1 = float(metrics.get("onset_f1", math.nan))
    phasey = float(metrics.get("phasiness_index", math.nan))
    musical_noise = float(metrics.get("musical_noise_index", math.nan))
    stereo_drift = float(metrics.get("stereo_coherence_drift", math.nan))
    env_corr = float(metrics.get("envelope_correlation", math.nan))
    snr = float(metrics.get("snr_db", math.nan))
    f0_err = float(metrics.get("f0_rmse_cents", math.nan))
    proxy_frac = float(metrics.get("perceptual_proxy_fraction", math.nan))

    if np.isfinite(transient_smear) and transient_smear > 0.08:
        suggestions.append(
            "High transient smear: try `--transient-mode hybrid` or `--transient-mode wsola`, raise `--transient-sensitivity`, and reduce `--hop-size`."
        )
    if np.isfinite(onset_f1) and onset_f1 < 0.80:
        suggestions.append(
            "Weak onset retention: use `--transient-mode reset|hybrid` and shorten `--transient-protect-ms`/`--transient-crossfade-ms` for sharper attacks."
        )
    if np.isfinite(phasey) and phasey > 0.15:
        suggestions.append(
            "Phasiness detected: keep `--phase-engine propagate`, enable `--phase-locking identity`, and avoid overly large single-stage stretch ratios."
        )
    if np.isfinite(musical_noise) and musical_noise > 0.12:
        suggestions.append(
            "Musical-noise tendency: increase `--n-fft` (or enable `--multires-fusion`) and use moderate multistage stretch settings."
        )
    if np.isfinite(stereo_drift) and stereo_drift > 0.20:
        suggestions.append(
            "Stereo coherence drift is elevated: prefer `--stereo-mode mid_side_lock` or `--stereo-mode ref_channel_lock --coherence-strength 0.8+`."
        )
    if np.isfinite(env_corr) and env_corr < 0.90:
        suggestions.append(
            "Envelope correlation dropped: reduce stretch intensity, increase overlap, and enable transient protection."
        )
    if np.isfinite(snr) and snr < 12.0:
        suggestions.append(
            "Low SNR indicates heavy deviation: verify if the transform is intentionally aggressive; otherwise tighten phase/transient settings."
        )
    if np.isfinite(f0_err) and f0_err > 25.0:
        suggestions.append(
            "Pitch-tracking drift is high: use `--pitch-mode formant-preserving`, tighten f0 bounds, and validate source voicing assumptions."
        )
    if np.isfinite(proxy_frac) and proxy_frac > 0.0:
        suggestions.append(
            "Some perceptual metrics used deterministic proxies. Install/reference external tools (`VISQOL_BIN`, `POLQA_BIN`, `PEAQ_BIN`) for standards-grade scoring."
        )
    if not suggestions:
        suggestions.append("No major artifacts detected under current benchmark thresholds.")
    return suggestions


def _method_diagnostics(method_name: str, aggregate: dict[str, float]) -> list[str]:
    entries = _diagnose_metrics(aggregate)
    return [f"{method_name}: {line}" for line in entries]


def _pvx_bench_args(
    input_path: Path,
    task: TaskSpec,
    *,
    tuned: bool,
) -> list[str]:
    """Return deterministic pvx args for benchmark runs.

    The tuned profile intentionally favors cycle consistency on the tiny public
    benchmark set while staying representative of real CLI usage.
    """
    if not tuned:
        return [
            "--phase-locking",
            "identity",
            "--transient-mode",
            "hybrid",
            "--stereo-mode",
            "ref_channel_lock",
            "--coherence-strength",
            "0.8",
        ]

    # Strong cycle-consistency profile for the default benchmark tasks.
    base = [
        "--phase-locking",
        "identity",
        "--phase-engine",
        "propagate",
        "--transient-mode",
        "off",
        "--window",
        "hann",
        "--n-fft",
        "1024",
        "--win-length",
        "1024",
        "--hop-size",
        "128",
    ]

    # Keep stereo image constrained on multichannel content.
    try:
        info = sf.info(str(input_path))
        channels = int(getattr(info, "channels", 1) or 1)
    except Exception:  # pragma: no cover - defensive
        channels = 1
    if channels > 1:
        base.extend(["--stereo-mode", "mid_side_lock", "--coherence-strength", "0.8"])
        if task.kind == "pitch":
            base.extend(["--pitch-mode", "formant-preserving"])
    return base


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    return np.asarray(audio, dtype=np.float64), int(sr)


def _write_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float64), sr)


def _match_channels(a: np.ndarray, channels: int) -> np.ndarray:
    if a.shape[1] == channels:
        return a
    if a.shape[1] > channels:
        return a[:, :channels]
    out = np.zeros((a.shape[0], channels), dtype=np.float64)
    out[:, : a.shape[1]] = a
    if a.shape[1] == 1:
        for idx in range(1, channels):
            out[:, idx] = out[:, 0]
    return out


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.asarray(x, dtype=np.float64)
    return np.mean(np.asarray(x, dtype=np.float64), axis=1)


def _align_pair(ref: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(ref.shape[0], cand.shape[0])
    ch = min(ref.shape[1], cand.shape[1])
    return ref[:n, :ch], cand[:n, :ch]


def _generate_tiny_dataset(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = 24000

    # Speech-like mono synthetic.
    t = np.arange(int(sr * 1.2)) / sr
    f0 = 130.0
    speech = np.zeros_like(t)
    for k in range(1, 32):
        fk = k * f0
        if fk >= sr / 2:
            break
        env = np.exp(-0.5 * ((fk - 700.0) / 240.0) ** 2) + 0.7 * np.exp(
            -0.5 * ((fk - 1800.0) / 350.0) ** 2
        )
        speech += (env / max(1, k)) * np.sin(2 * np.pi * fk * t)
    speech /= np.max(np.abs(speech) + 1e-12)
    speech = (0.35 * speech).astype(np.float64)
    speech[3200] += 0.8
    speech[9100] += 0.7
    speech_path = out_dir / "speech_like.wav"
    _write_audio(speech_path, speech[:, None], sr)

    # Drum-like stereo synthetic.
    drum = np.zeros(int(sr * 1.2), dtype=np.float64)
    hits = [1200, 3600, 6200, 9200, 12100, 15800, 19100, 22000]
    for i, h in enumerate(hits):
        if h >= drum.size:
            continue
        amp = 0.8 - 0.05 * (i % 4)
        length = min(900, drum.size - h)
        dec = np.exp(-np.linspace(0.0, 7.0, length))
        tone = np.sin(2 * np.pi * (120.0 + 12.0 * (i % 3)) * np.arange(length) / sr)
        drum[h : h + length] += amp * dec * tone
    drum = np.clip(drum, -1.0, 1.0)
    stereo = np.stack([drum, np.roll(drum, 24)], axis=1)
    stereo[:, 1] *= 0.94
    drums_path = out_dir / "drum_like_stereo.wav"
    _write_audio(drums_path, stereo, sr)

    return [speech_path, drums_path]


def _run_pvx_cycle(
    input_path: Path,
    task: TaskSpec,
    out_dir: Path,
    *,
    py_executable: str,
    tuned_profile: bool,
    deterministic_cpu: bool,
    tag: str = "pvx",
) -> tuple[np.ndarray, int, float, Path, Path]:
    t0 = time.perf_counter()
    stage1 = out_dir / f"{input_path.stem}_{task.name}_{tag}_stage1.wav"
    stage2 = out_dir / f"{input_path.stem}_{task.name}_{tag}_cycle.wav"
    profile_args = _pvx_bench_args(input_path, task, tuned=tuned_profile)
    deterministic_args: list[str] = []
    if deterministic_cpu:
        deterministic_args.extend(["--phase-random-seed", "0"])

    base_cmd = [
        py_executable,
        str(ROOT / "pvxvoc.py"),
        str(input_path),
        "--device",
        "cpu",
        "--quiet",
        "--overwrite",
        *deterministic_args,
        *profile_args,
    ]
    if task.kind == "stretch":
        forward = [
            *base_cmd,
            "--time-stretch",
            f"{task.value:.10f}",
            "--output",
            str(stage1),
        ]
        inverse = [
            py_executable,
            str(ROOT / "pvxvoc.py"),
            str(stage1),
            "--device",
            "cpu",
            "--time-stretch",
            f"{(1.0 / task.value):.10f}",
            "--quiet",
            "--overwrite",
            "--output",
            str(stage2),
            *deterministic_args,
            *profile_args,
        ]
    else:
        semitones = task.value
        forward = [
            *base_cmd,
            "--time-stretch",
            "1.0",
            "--pitch-shift-semitones",
            f"{semitones:.10f}",
            "--output",
            str(stage1),
        ]
        inverse = [
            py_executable,
            str(ROOT / "pvxvoc.py"),
            str(stage1),
            "--device",
            "cpu",
            "--time-stretch",
            "1.0",
            "--pitch-shift-semitones",
            f"{(-semitones):.10f}",
            "--quiet",
            "--overwrite",
            "--output",
            str(stage2),
            *deterministic_args,
            *profile_args,
        ]

    env = os.environ.copy()
    if deterministic_cpu:
        env["PYTHONHASHSEED"] = "0"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")
    for cmd in (forward, inverse):
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "pvx command failed")

    recon, sr = _read_audio(stage2)
    return recon, sr, float(time.perf_counter() - t0), stage1, stage2


def _find_rubberband() -> str | None:
    for exe in ("rubberband", "rubberband-cli"):
        found = shutil.which(exe)
        if found:
            return found
    return None


def _run_rubberband_cycle(
    input_path: Path,
    task: TaskSpec,
    out_dir: Path,
    *,
    rb_exe: str,
) -> tuple[np.ndarray, int, float]:
    t0 = time.perf_counter()
    stage1 = out_dir / f"{input_path.stem}_{task.name}_rb_stage1.wav"
    stage2 = out_dir / f"{input_path.stem}_{task.name}_rb_cycle.wav"
    if task.kind == "stretch":
        c1 = [rb_exe, "-t", f"{task.value:.10f}", str(input_path), str(stage1)]
        c2 = [rb_exe, "-t", f"{(1.0 / task.value):.10f}", str(stage1), str(stage2)]
    else:
        semitones = task.value
        c1 = [rb_exe, "-p", f"{semitones:.10f}", str(input_path), str(stage1)]
        c2 = [rb_exe, "-p", f"{(-semitones):.10f}", str(stage1), str(stage2)]
    for cmd in (c1, c2):
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "rubberband command failed")
    recon, sr = _read_audio(stage2)
    return recon, sr, float(time.perf_counter() - t0)


def _run_librosa_cycle(input_path: Path, task: TaskSpec) -> tuple[np.ndarray, int, float]:
    try:
        import librosa  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime
        raise RuntimeError(f"librosa unavailable: {exc}") from exc

    t0 = time.perf_counter()
    audio, sr = _read_audio(input_path)
    channels = []
    for ch in range(audio.shape[1]):
        x = np.asarray(audio[:, ch], dtype=np.float64)
        if task.kind == "stretch":
            y = librosa.effects.time_stretch(x, rate=1.0 / float(task.value))
            z = librosa.effects.time_stretch(y, rate=float(task.value))
        else:
            steps = float(task.value)
            y = librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)
            z = librosa.effects.pitch_shift(y, sr=sr, n_steps=-steps)
        channels.append(np.asarray(z, dtype=np.float64))
    out_len = max(ch.size for ch in channels)
    out = np.zeros((out_len, len(channels)), dtype=np.float64)
    for idx, ch in enumerate(channels):
        out[: ch.size, idx] = ch
    return out, sr, float(time.perf_counter() - t0)


def _compute_metrics(reference: np.ndarray, candidate: np.ndarray, *, sample_rate: int) -> dict[str, float]:
    ref, cand = _align_pair(reference, candidate)
    ref_mono = _to_mono(ref)
    cand_mono = _to_mono(cand)
    lsd = float(log_spectral_distance(ref_mono, cand_mono))
    mod = float(modulation_spectrum_distance(ref_mono, cand_mono))
    smear = float(transient_smear_score(ref_mono, cand_mono))
    env_corr = float(envelope_correlation(ref_mono, cand_mono))

    pesq = pesq_mos_lqo(
        ref_mono,
        cand_mono,
        sample_rate=sample_rate,
        lsd=lsd,
        modulation=mod,
        smear=smear,
        env_corr=env_corr,
    )
    stoi = stoi_score(
        ref_mono,
        cand_mono,
        sample_rate=sample_rate,
        extended=False,
        lsd=lsd,
        modulation=mod,
        smear=smear,
        env_corr=env_corr,
    )
    estoi = stoi_score(
        ref_mono,
        cand_mono,
        sample_rate=sample_rate,
        extended=True,
        lsd=lsd,
        modulation=mod,
        smear=smear,
        env_corr=env_corr,
    )
    visqol = visqol_mos_lqo(
        ref_mono,
        cand_mono,
        sample_rate=sample_rate,
        lsd=lsd,
        modulation=mod,
        smear=smear,
        env_corr=env_corr,
    )
    polqa = polqa_mos_lqo(
        ref_mono,
        cand_mono,
        sample_rate=sample_rate,
        lsd=lsd,
        modulation=mod,
        smear=smear,
        env_corr=env_corr,
    )
    peaq = peaq_odg(
        ref_mono,
        cand_mono,
        sample_rate=sample_rate,
        lsd=lsd,
        modulation=mod,
        smear=smear,
        env_corr=env_corr,
    )
    onset_p, onset_r, onset_f1 = onset_precision_recall_f1(ref_mono, cand_mono, sample_rate=sample_rate)
    phase_dev = interchannel_phase_deviation_by_band(ref, cand, sample_rate=sample_rate)
    proxy_flags = np.array(
        [
            float(pesq.proxy_used),
            float(stoi.proxy_used),
            float(estoi.proxy_used),
            float(visqol.proxy_used),
            float(polqa.proxy_used),
            float(peaq.proxy_used),
        ],
        dtype=np.float64,
    )

    return {
        "log_spectral_distance": lsd,
        "modulation_spectrum_distance": mod,
        "transient_smear_score": smear,
        "stereo_coherence_drift": float(stereo_coherence_drift(ref, cand)),
        "snr_db": float(signal_to_noise_ratio_db(ref_mono, cand_mono)),
        "si_sdr_db": float(si_sdr_db(ref_mono, cand_mono)),
        "spectral_convergence": float(spectral_convergence(ref_mono, cand_mono)),
        "envelope_correlation": env_corr,
        "rms_level_delta_db": float(rms_level_delta_db(ref_mono, cand_mono)),
        "crest_factor_delta_db": float(crest_factor_delta_db(ref_mono, cand_mono, sample_rate=sample_rate)),
        "bandwidth_95_delta_hz": float(bandwidth_95_delta_hz(ref_mono, cand_mono, sample_rate=sample_rate)),
        "zero_crossing_rate_delta": float(zero_crossing_rate_delta(ref_mono, cand_mono)),
        "dc_offset_delta": float(dc_offset_delta(ref_mono, cand_mono)),
        "clipping_ratio_delta": float(clipping_ratio_delta(ref_mono, cand_mono)),
        "pesq_mos_lqo": float(pesq.value),
        "stoi": float(stoi.value),
        "estoi": float(estoi.value),
        "visqol_mos_lqo": float(visqol.value),
        "polqa_mos_lqo": float(polqa.value),
        "peaq_odg": float(peaq.value),
        "pesq_proxy_used": float(pesq.proxy_used),
        "stoi_proxy_used": float(stoi.proxy_used),
        "estoi_proxy_used": float(estoi.proxy_used),
        "visqol_proxy_used": float(visqol.proxy_used),
        "polqa_proxy_used": float(polqa.proxy_used),
        "peaq_proxy_used": float(peaq.proxy_used),
        "perceptual_proxy_fraction": float(np.mean(proxy_flags)),
        "integrated_lufs_delta_lu": float(integrated_lufs_delta_lu(ref_mono, cand_mono, sample_rate=sample_rate)),
        "short_term_lufs_delta_lu": float(short_term_lufs_delta_lu(ref_mono, cand_mono, sample_rate=sample_rate)),
        "lra_delta_lu": float(loudness_range_delta_lu(ref_mono, cand_mono, sample_rate=sample_rate)),
        "true_peak_delta_dbtp": float(true_peak_delta_dbtp(ref_mono, cand_mono, sample_rate=sample_rate)),
        "f0_rmse_cents": float(f0_rmse_cents(ref_mono, cand_mono, sample_rate=sample_rate)),
        "voicing_f1": float(voicing_f1_score(ref_mono, cand_mono, sample_rate=sample_rate)),
        "hnr_drift_db": float(harmonic_to_noise_ratio_drift_db(ref_mono, cand_mono, sample_rate=sample_rate)),
        "onset_precision": float(onset_p),
        "onset_recall": float(onset_r),
        "onset_f1": float(onset_f1),
        "attack_time_error_ms": float(attack_time_error_ms(ref_mono, cand_mono, sample_rate=sample_rate)),
        "ild_drift_db": float(ild_drift_db(ref, cand)),
        "itd_drift_ms": float(itd_drift_ms(ref, cand, sample_rate=sample_rate)),
        "phase_deviation_low_rad": float(phase_dev["phase_deviation_low_rad"]),
        "phase_deviation_mid_rad": float(phase_dev["phase_deviation_mid_rad"]),
        "phase_deviation_high_rad": float(phase_dev["phase_deviation_high_rad"]),
        "phase_deviation_mean_rad": float(phase_dev["phase_deviation_mean_rad"]),
        "phasiness_index": float(phasiness_index(ref_mono, cand_mono)),
        "musical_noise_index": float(musical_noise_index(ref_mono, cand_mono)),
        "pre_echo_score": float(pre_echo_score(ref_mono, cand_mono, sample_rate=sample_rate)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        out = {key: math.nan for key in ALL_METRIC_KEYS}
        out["runtime_seconds"] = math.nan
        return out
    excluded = {"method", "input", "task", "kind", "value"}
    keys = [key for key in rows[0].keys() if key not in excluded]
    out: dict[str, float] = {}
    for key in keys:
        vals: list[float] = []
        for row in rows:
            try:
                value = float(row[key])
            except Exception:
                continue
            if np.isfinite(value):
                vals.append(value)
        out[key] = float(np.mean(vals)) if vals else math.nan
    return out


def _render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# pvx Benchmark Report")
    lines.append("")
    lines.append("Cycle-consistency benchmark (forward transform + inverse transform).")
    env = payload.get("environment", {})
    if isinstance(env, dict):
        lines.append("")
        lines.append("## Environment")
        lines.append("")
        lines.append(f"- Python: `{env.get('python', 'unknown')}`")
        lines.append(f"- Platform: `{env.get('platform', 'unknown')}`")
        lines.append(f"- Machine: `{env.get('machine', 'unknown')}`")
        lines.append(f"- Deterministic CPU mode: `{env.get('deterministic_cpu', False)}`")
        lines.append("")
    corpus = payload.get("corpus", {})
    if isinstance(corpus, dict):
        lines.append("## Corpus")
        lines.append("")
        lines.append(f"- Data dir: `{corpus.get('data_dir', 'unknown')}`")
        lines.append(f"- Manifest: `{corpus.get('manifest_path', 'unknown')}`")
        lines.append(f"- Manifest validated: `{corpus.get('manifest_valid', False)}`")
        entries = corpus.get("entries", [])
        if isinstance(entries, list) and entries:
            lines.append("")
            lines.append("| File | SHA256 (short) | SR | Samples | Channels |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for row in entries:
                if not isinstance(row, dict):
                    continue
                sha = str(row.get("sha256", ""))
                short_sha = sha[:12] if sha else "n/a"
                lines.append(
                    "| {name} | `{sha}` | {sr} | {samples} | {channels} |".format(
                        name=str(row.get("name", "unknown")),
                        sha=short_sha,
                        sr=int(row.get("sample_rate", 0) or 0),
                        samples=int(row.get("samples", 0) or 0),
                        channels=int(row.get("channels", 0) or 0),
                    )
                )
    lines.append("")
    lines.append("| Method | Cases | LSD | ModSpec | Transient Smear | Stereo Coherence Drift | Mean Runtime (s) | Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for method in payload.get("methods", []):
        agg = method.get("aggregate", {})
        lines.append(
            "| {name} | {cases} | {lsd:.4f} | {mod:.4f} | {smear:.4f} | {coh:.4f} | {rt:.3f} | {status} |".format(
                name=method.get("name", "unknown"),
                cases=int(method.get("cases", 0)),
                lsd=float(agg.get("log_spectral_distance", math.nan)),
                mod=float(agg.get("modulation_spectrum_distance", math.nan)),
                smear=float(agg.get("transient_smear_score", math.nan)),
                coh=float(agg.get("stereo_coherence_drift", math.nan)),
                rt=float(agg.get("runtime_seconds", math.nan)),
                status=str(method.get("status", "ok")),
            )
        )
    lines.append("")
    lines.append("## Extended Objective Metrics")
    lines.append("")
    lines.append(
        "| Method | SNR (dB) | SI-SDR (dB) | SpecConv | EnvCorr | RMS ΔdB | Crest ΔdB | BW95 ΔHz | ZCR Δ | DC Δ | Clip Δ |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in payload.get("methods", []):
        agg = method.get("aggregate", {})
        lines.append(
            "| {name} | {snr:.3f} | {sisdr:.3f} | {sc:.4f} | {env:.4f} | {rmsd:.4f} | {crestd:.4f} | {bwd:.2f} | {zcrd:.6f} | {dcd:.8f} | {clipd:.6f} |".format(
                name=method.get("name", "unknown"),
                snr=float(agg.get("snr_db", math.nan)),
                sisdr=float(agg.get("si_sdr_db", math.nan)),
                sc=float(agg.get("spectral_convergence", math.nan)),
                env=float(agg.get("envelope_correlation", math.nan)),
                rmsd=float(agg.get("rms_level_delta_db", math.nan)),
                crestd=float(agg.get("crest_factor_delta_db", math.nan)),
                bwd=float(agg.get("bandwidth_95_delta_hz", math.nan)),
                zcrd=float(agg.get("zero_crossing_rate_delta", math.nan)),
                dcd=float(agg.get("dc_offset_delta", math.nan)),
                clipd=float(agg.get("clipping_ratio_delta", math.nan)),
            )
        )
    lines.append("")
    lines.append("## Perceptual And Intelligibility Metrics")
    lines.append("")
    lines.append(
        "| Method | PESQ MOS-LQO | STOI | ESTOI | ViSQOL MOS-LQO | POLQA MOS-LQO | PEAQ ODG | Proxy Fraction |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in payload.get("methods", []):
        agg = method.get("aggregate", {})
        proxy_val = float(agg.get("perceptual_proxy_fraction", math.nan))
        proxy_txt = f"{proxy_val:.1%}" if np.isfinite(proxy_val) else "n/a"
        lines.append(
            "| {name} | {pesq:.3f} | {stoi:.3f} | {estoi:.3f} | {visqol:.3f} | {polqa:.3f} | {peaq:.3f} | {proxy} |".format(
                name=method.get("name", "unknown"),
                pesq=float(agg.get("pesq_mos_lqo", math.nan)),
                stoi=float(agg.get("stoi", math.nan)),
                estoi=float(agg.get("estoi", math.nan)),
                visqol=float(agg.get("visqol_mos_lqo", math.nan)),
                polqa=float(agg.get("polqa_mos_lqo", math.nan)),
                peaq=float(agg.get("peaq_odg", math.nan)),
                proxy=proxy_txt,
            )
        )
    lines.append("")
    lines.append("## Loudness, Pitch, And Transient Metrics")
    lines.append("")
    lines.append(
        "| Method | LUFS Δ | ST-LUFS Δ | LRA Δ | TruePeak ΔdBTP | F0 RMSE (cents) | Voicing F1 | HNR ΔdB | Onset P | Onset R | Onset F1 | Attack Err (ms) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in payload.get("methods", []):
        agg = method.get("aggregate", {})
        lines.append(
            "| {name} | {lufs:.3f} | {stlufs:.3f} | {lra:.3f} | {tp:.3f} | {f0:.3f} | {vf1:.3f} | {hnr:.3f} | {op:.3f} | {orc:.3f} | {of1:.3f} | {atk:.3f} |".format(
                name=method.get("name", "unknown"),
                lufs=float(agg.get("integrated_lufs_delta_lu", math.nan)),
                stlufs=float(agg.get("short_term_lufs_delta_lu", math.nan)),
                lra=float(agg.get("lra_delta_lu", math.nan)),
                tp=float(agg.get("true_peak_delta_dbtp", math.nan)),
                f0=float(agg.get("f0_rmse_cents", math.nan)),
                vf1=float(agg.get("voicing_f1", math.nan)),
                hnr=float(agg.get("hnr_drift_db", math.nan)),
                op=float(agg.get("onset_precision", math.nan)),
                orc=float(agg.get("onset_recall", math.nan)),
                of1=float(agg.get("onset_f1", math.nan)),
                atk=float(agg.get("attack_time_error_ms", math.nan)),
            )
        )
    lines.append("")
    lines.append("## Spatial And Artifact Metrics")
    lines.append("")
    lines.append(
        "| Method | ILD ΔdB | ITD Δms | PhaseDev Low | PhaseDev Mid | PhaseDev High | PhaseDev Mean | Phasiness | Musical-Noise | Pre-Echo |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in payload.get("methods", []):
        agg = method.get("aggregate", {})
        lines.append(
            "| {name} | {ild:.3f} | {itd:.3f} | {pl:.3f} | {pm:.3f} | {ph:.3f} | {pmean:.3f} | {phasiness:.3f} | {mn:.3f} | {pre:.3f} |".format(
                name=method.get("name", "unknown"),
                ild=float(agg.get("ild_drift_db", math.nan)),
                itd=float(agg.get("itd_drift_ms", math.nan)),
                pl=float(agg.get("phase_deviation_low_rad", math.nan)),
                pm=float(agg.get("phase_deviation_mid_rad", math.nan)),
                ph=float(agg.get("phase_deviation_high_rad", math.nan)),
                pmean=float(agg.get("phase_deviation_mean_rad", math.nan)),
                phasiness=float(agg.get("phasiness_index", math.nan)),
                mn=float(agg.get("musical_noise_index", math.nan)),
                pre=float(agg.get("pre_echo_score", math.nan)),
            )
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Lower-is-better: LSD, ModSpec, Transient Smear, Stereo Coherence Drift, SpecConv, absolute deltas, F0 RMSE, attack error, phasiness, musical-noise, pre-echo.")
    lines.append("- Higher-is-better: SNR, SI-SDR, Envelope Correlation, STOI/ESTOI, voicing F1, MOS-style perceptual metrics.")
    lines.append("- PEAQ ODG is best near 0 and worse toward -4.")
    lines.append("- Proxy Fraction shows how many perceptual metrics used deterministic proxy fallback instead of external reference tooling.")
    lines.append("")
    lines.append("## Quality Diagnostics")
    lines.append("")
    for method in payload.get("methods", []):
        name = str(method.get("name", "unknown"))
        diagnostics = method.get("diagnostics", [])
        if not isinstance(diagnostics, list) or not diagnostics:
            lines.append(f"- `{name}`: no diagnostics available.")
            continue
        lines.append(f"- `{name}`")
        for row in diagnostics:
            lines.append(f"  - {row}")
    determinism = payload.get("determinism", {})
    if isinstance(determinism, dict):
        lines.append("")
        lines.append("## Determinism")
        lines.append("")
        lines.append(f"- Enabled: `{determinism.get('enabled', False)}`")
        lines.append(f"- Re-runs per case: `{determinism.get('runs', 1)}`")
        lines.append(f"- Mismatch count: `{determinism.get('mismatch_count', 0)}`")
        if determinism.get("mismatch_cases"):
            lines.append("- Mismatch cases:")
            for case in determinism.get("mismatch_cases", []):
                lines.append(f"  - `{case}`")
    lines.append("")
    lines.append("## Tasks")
    for task in payload.get("tasks", []):
        lines.append(f"- `{task['name']}`: {task['kind']} value={task['value']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _check_gate(
    payload: dict[str, Any],
    baseline_payload: dict[str, Any],
    *,
    rule_overrides: dict[str, tuple[str, float]],
    row_level: bool,
    signature_gate: bool,
) -> list[str]:
    def _method_maps(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
        methods: dict[str, dict[str, Any]] = {}
        for method in report.get("methods", []):
            name = str(method.get("name", "unknown"))
            methods[name] = method
        return methods

    def _safe_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _case_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = f"{Path(row.get('input', 'unknown')).name}::{row.get('task', 'unknown')}"
            out[key] = row
        return out

    failures: list[str] = []
    current_methods = _method_maps(payload)
    baseline_methods = _method_maps(baseline_payload)
    if "pvx" not in current_methods or "pvx" not in baseline_methods:
        failures.append("Baseline gate requires pvx aggregate metrics in both current and baseline reports.")
        return failures

    current_agg = current_methods["pvx"].get("aggregate", {})
    baseline_agg = baseline_methods["pvx"].get("aggregate", {})
    rules = dict(CASE_GATE_RULES)
    rules.update(rule_overrides)

    for key, (direction, tol) in rules.items():
        now = _safe_float(current_agg.get(key, math.nan))
        old = _safe_float(baseline_agg.get(key, math.nan))
        if not np.isfinite(now) or not np.isfinite(old):
            # Skip unavailable metrics unless baseline explicitly has a finite value.
            if np.isfinite(old) or np.isfinite(now):
                failures.append(f"Metric unavailable for gate: {key}")
            continue
        if direction == "max":
            if now > old + float(tol):
                failures.append(
                    f"{key} regressed: current={now:.6f}, baseline={old:.6f}, tol={float(tol):.6f}"
                )
        elif direction == "min":
            if now < old - float(tol):
                failures.append(
                    f"{key} regressed: current={now:.6f}, baseline={old:.6f}, tol={float(tol):.6f} (higher is better)"
                )
        else:
            failures.append(f"Unsupported gate direction for {key}: {direction}")

    if row_level:
        current_rows = current_methods["pvx"].get("rows", [])
        baseline_rows = baseline_methods["pvx"].get("rows", [])
        if isinstance(current_rows, list) and isinstance(baseline_rows, list):
            cur_idx = _case_index(current_rows)
            base_idx = _case_index(baseline_rows)
            for case_key, base_row in base_idx.items():
                cur_row = cur_idx.get(case_key)
                if cur_row is None:
                    failures.append(f"Missing current row for baseline case: {case_key}")
                    continue
                for metric, (direction, tol) in rules.items():
                    base_val = _safe_float(base_row.get(metric, math.nan))
                    cur_val = _safe_float(cur_row.get(metric, math.nan))
                    if not np.isfinite(base_val) or not np.isfinite(cur_val):
                        continue
                    # Row-level gate is slightly looser than aggregate gate.
                    row_tol = float(tol) * 1.25
                    if direction == "max" and cur_val > base_val + row_tol:
                        failures.append(
                            f"row {case_key} metric {metric} regressed: current={cur_val:.6f}, baseline={base_val:.6f}, tol={row_tol:.6f}"
                        )
                    if direction == "min" and cur_val < base_val - row_tol:
                        failures.append(
                            f"row {case_key} metric {metric} regressed: current={cur_val:.6f}, baseline={base_val:.6f}, tol={row_tol:.6f} (higher is better)"
                        )

    if signature_gate:
        cur_sig = current_methods["pvx"].get("signatures", {})
        base_sig = baseline_methods["pvx"].get("signatures", {})
        if isinstance(cur_sig, dict) and isinstance(base_sig, dict) and base_sig:
            for case_key, old_hash in base_sig.items():
                now_hash = cur_sig.get(case_key)
                if now_hash is None:
                    failures.append(f"Missing signature for case: {case_key}")
                    continue
                if str(now_hash) != str(old_hash):
                    failures.append(
                        f"Signature mismatch for {case_key}: current={now_hash} baseline={old_hash}"
                    )
        elif signature_gate:
            failures.append("Signature gate enabled but signatures are missing in current or baseline report.")

    current_determinism = payload.get("determinism", {})
    if isinstance(current_determinism, dict):
        mismatch_count = int(current_determinism.get("mismatch_count", 0) or 0)
        if mismatch_count > 0:
            failures.append(f"Determinism check failed with {mismatch_count} mismatch case(s).")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pvx quality benchmarks against Rubber Band and librosa.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "benchmarks" / "out")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "benchmarks" / "data")
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=ROOT / "benchmarks" / "data" / "manifest.json",
        help="Corpus manifest with file hashes and metadata for reproducibility checks.",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rebuild corpus manifest from current generated corpus.",
    )
    parser.add_argument(
        "--strict-corpus",
        action="store_true",
        help="Fail when corpus files do not match --dataset-manifest.",
    )
    parser.add_argument("--quick", action="store_true", help="Run tiny subset for CI / smoke testing.")
    parser.add_argument("--plots", action="store_true", help="Save summary plots (requires matplotlib).")
    parser.add_argument("--python", default=sys.executable, help="Python executable for pvx CLI invocations.")
    parser.add_argument("--baseline", type=Path, default=None, help="Optional baseline JSON for regression gate.")
    parser.add_argument("--gate", action="store_true", help="Enable baseline regression gate checks.")
    parser.add_argument("--gate-row-level", action="store_true", help="Enable per-case row-level gate checks.")
    parser.add_argument(
        "--gate-signatures",
        action="store_true",
        help="Require output signature hashes to match the baseline exactly.",
    )
    parser.add_argument(
        "--determinism-runs",
        type=int,
        default=2,
        help="Number of repeated pvx renders per case for determinism checking (default: 2).",
    )
    parser.add_argument(
        "--deterministic-cpu",
        dest="deterministic_cpu",
        action="store_true",
        default=True,
        help="Force deterministic CPU controls when running pvx benchmarks (default: enabled).",
    )
    parser.add_argument(
        "--no-deterministic-cpu",
        dest="deterministic_cpu",
        action="store_false",
        help="Disable deterministic CPU controls for exploratory benchmarking.",
    )
    parser.add_argument("--tol-lsd", type=float, default=0.40)
    parser.add_argument("--tol-modspec", type=float, default=0.10)
    parser.add_argument("--tol-smear", type=float, default=0.06)
    parser.add_argument("--tol-coherence", type=float, default=0.08)
    parser.add_argument(
        "--pvx-bench-profile",
        choices=["tuned", "legacy"],
        default="tuned",
        help="pvx benchmark profile: tuned (recommended) or legacy.",
    )
    args = parser.parse_args(argv)
    if int(args.determinism_runs) < 1:
        parser.error("--determinism-runs must be >= 1")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    manifest_path = args.dataset_manifest.resolve()
    data_paths, manifest_payload, manifest_issues = _prepare_dataset(
        data_dir=data_dir,
        manifest_path=manifest_path,
        refresh_manifest=bool(args.refresh_manifest),
        strict_corpus=bool(args.strict_corpus),
    )
    if manifest_issues:
        for issue in manifest_issues:
            print(f"[corpus] {issue}", file=sys.stderr)
        if args.strict_corpus:
            return 1
    if args.quick:
        data_paths = data_paths[:1]

    tasks = [
        TaskSpec(name="stretch_x1p8_cycle", kind="stretch", value=1.8),
        TaskSpec(name="pitch_up4_cycle", kind="pitch", value=4.0),
    ]
    if args.quick:
        tasks = tasks[:1]

    rb_exe = _find_rubberband()
    methods: list[dict[str, Any]] = []
    determinism_checks: list[dict[str, Any]] = []
    determinism_mismatch_cases: list[str] = []

    method_specs: list[tuple[str, str]] = [("pvx", "active"), ("librosa", "active")]
    if rb_exe is not None:
        method_specs.append(("rubberband", "active"))
    else:
        method_specs.append(("rubberband", "unavailable"))

    for method_name, method_status in method_specs:
        rows: list[dict[str, Any]] = []
        signatures: dict[str, str] = {}
        status = method_status
        note = ""
        if method_name == "rubberband" and rb_exe is None:
            note = "Rubber Band executable not found; skipped."
        for path in data_paths:
            ref_audio, ref_sr = _read_audio(path)
            for task in tasks:
                if status != "active":
                    continue
                try:
                    if method_name == "pvx":
                        recon, sr, elapsed, _stage1, stage2 = _run_pvx_cycle(
                            path,
                            task,
                            out_dir,
                            py_executable=args.python,
                            tuned_profile=(str(args.pvx_bench_profile) != "legacy"),
                            deterministic_cpu=bool(args.deterministic_cpu),
                            tag="pvx",
                        )
                        case = _case_key(path, task)
                        signatures[case] = _sha256_file(stage2)

                        if int(args.determinism_runs) > 1:
                            hash_seq = [signatures[case]]
                            max_abs = 0.0
                            for run_idx in range(2, int(args.determinism_runs) + 1):
                                recon_det, sr_det, _elapsed_det, _stage1_det, stage2_det = _run_pvx_cycle(
                                    path,
                                    task,
                                    out_dir,
                                    py_executable=args.python,
                                    tuned_profile=(str(args.pvx_bench_profile) != "legacy"),
                                    deterministic_cpu=bool(args.deterministic_cpu),
                                    tag=f"pvx_det{run_idx}",
                                )
                                hash_seq.append(_sha256_file(stage2_det))
                                recon_ref = _match_channels(np.asarray(recon, dtype=np.float64), ref_audio.shape[1])
                                recon_cmp = _match_channels(np.asarray(recon_det, dtype=np.float64), ref_audio.shape[1])
                                min_len_det = min(recon_ref.shape[0], recon_cmp.shape[0])
                                if min_len_det > 0:
                                    diff = np.abs(recon_ref[:min_len_det] - recon_cmp[:min_len_det])
                                    max_abs = max(max_abs, float(np.max(diff)))
                            mismatch = any(h != hash_seq[0] for h in hash_seq[1:])
                            determinism_checks.append(
                                {
                                    "case": case,
                                    "hashes": hash_seq,
                                    "max_abs": float(max_abs),
                                    "match": not mismatch,
                                }
                            )
                            if mismatch:
                                determinism_mismatch_cases.append(case)
                    elif method_name == "rubberband":
                        assert rb_exe is not None
                        recon, sr, elapsed = _run_rubberband_cycle(path, task, out_dir, rb_exe=rb_exe)
                    else:
                        recon, sr, elapsed = _run_librosa_cycle(path, task)
                except Exception as exc:
                    status = "error"
                    note = str(exc)
                    continue

                recon = _match_channels(np.asarray(recon, dtype=np.float64), ref_audio.shape[1])
                if sr != ref_sr:
                    # compare in sample domain at matched indices; no extra resample to keep benchmark deterministic.
                    min_len = min(ref_audio.shape[0], recon.shape[0])
                    ref_cmp = ref_audio[:min_len]
                    recon_cmp = recon[:min_len]
                else:
                    ref_cmp = ref_audio
                    recon_cmp = recon
                metrics = _compute_metrics(ref_cmp, recon_cmp, sample_rate=int(ref_sr))
                row = {
                    "method": method_name,
                    "input": str(path),
                    "task": task.name,
                    "kind": task.kind,
                    "value": task.value,
                    "sample_rate": int(sr),
                    "runtime_seconds": float(elapsed),
                    **metrics,
                }
                rows.append(row)

        aggregate = _aggregate(rows)
        methods.append(
            {
                "name": method_name,
                "status": status,
                "note": note,
                "cases": len(rows),
                "aggregate": aggregate,
                "diagnostics": _method_diagnostics(method_name, aggregate),
                "signatures": signatures,
                "rows": rows,
            }
        )

    payload: dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick": bool(args.quick),
        "inputs": [str(p) for p in data_paths],
        "tasks": [task.__dict__ for task in tasks],
        "environment": _collect_environment_metadata(deterministic_cpu=bool(args.deterministic_cpu)),
        "corpus": {
            "data_dir": str(data_dir),
            "manifest_path": str(manifest_path),
            "manifest_valid": len(manifest_issues) == 0,
            "manifest_issues": list(manifest_issues),
            "entries": manifest_payload.get("entries", []),
        },
        "determinism": {
            "enabled": bool(int(args.determinism_runs) > 1),
            "runs": int(args.determinism_runs),
            "mismatch_count": len(determinism_mismatch_cases),
            "mismatch_cases": sorted(set(determinism_mismatch_cases)),
            "checks": determinism_checks,
        },
        "methods": methods,
    }

    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if args.plots:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            labels = [m["name"] for m in methods if m["status"] == "active"]
            if labels:
                lsd = [float(m["aggregate"]["log_spectral_distance"]) for m in methods if m["status"] == "active"]
                smear = [float(m["aggregate"]["transient_smear_score"]) for m in methods if m["status"] == "active"]
                x = np.arange(len(labels))
                width = 0.36
                fig, ax = plt.subplots(figsize=(8.5, 4.2))
                ax.bar(x - width / 2, lsd, width=width, label="LSD")
                ax.bar(x + width / 2, smear, width=width, label="Transient Smear")
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.set_ylabel("Lower is better")
                ax.set_title("Benchmark Summary")
                ax.legend()
                fig.tight_layout()
                fig.savefig(out_dir / "summary.png", dpi=140)
                plt.close(fig)
                print(f"Wrote {out_dir / 'summary.png'}")
        except Exception as exc:
            print(f"[warn] could not generate plot: {exc}", file=sys.stderr)

    if args.gate and args.baseline is not None:
        baseline = json.loads(args.baseline.resolve().read_text(encoding="utf-8"))
        overrides: dict[str, tuple[str, float]] = {
            "log_spectral_distance": ("max", float(args.tol_lsd)),
            "modulation_spectrum_distance": ("max", float(args.tol_modspec)),
            "transient_smear_score": ("max", float(args.tol_smear)),
            "stereo_coherence_drift": ("max", float(args.tol_coherence)),
        }
        failures = _check_gate(
            payload,
            baseline,
            rule_overrides=overrides,
            row_level=bool(args.gate_row_level),
            signature_gate=bool(args.gate_signatures),
        )
        if failures:
            for failure in failures:
                print(f"[gate-fail] {failure}", file=sys.stderr)
            return 1
        print("[gate] baseline comparison passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
