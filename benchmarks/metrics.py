"""Objective metrics for pvx benchmark comparisons."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from pvx.core.audio_metrics import summarize_audio_metrics
from pvx.core.voc import estimate_f0_autocorrelation
from pvx.metrics.coherence import stereo_coherence_drift_score

EPS = 1e-12


@dataclass(frozen=True)
class OptionalMetricValue:
    value: float
    proxy_used: bool


def _match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _principal_angle(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _to_mono(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.mean(arr, axis=1)


def _stft_complex(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    signal = np.asarray(x, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex128)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size), mode="constant")
    rem = (signal.size - n_fft) % hop
    if rem:
        signal = np.pad(signal, (0, hop - rem), mode="constant")
    frames = 1 + (signal.size - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float64)
    out = np.empty((n_fft // 2 + 1, frames), dtype=np.complex128)
    for idx in range(frames):
        start = idx * hop
        out[:, idx] = np.fft.rfft(signal[start : start + n_fft] * win, n=n_fft)
    return out


def _stft_mag_db(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    spec = _stft_complex(x, n_fft=n_fft, hop=hop)
    return 20.0 * np.log10(np.abs(spec) + 1e-9)


def _resample_signal(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if src_sr <= 0 or dst_sr <= 0 or arr.size == 0 or src_sr == dst_sr:
        return arr
    try:
        import scipy.signal as sps  # type: ignore

        g = int(np.gcd(src_sr, dst_sr))
        up = dst_sr // g
        down = src_sr // g
        return np.asarray(sps.resample_poly(arr, up, down), dtype=np.float64)
    except Exception:
        n_out = max(1, int(round(arr.size * float(dst_sr) / float(src_sr))))
        x_old = np.linspace(0.0, 1.0, arr.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, arr).astype(np.float64)


def _onset_envelope(x: np.ndarray, *, n_fft: int = 1024, hop_size: int = 256) -> np.ndarray:
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    spec_db = _stft_mag_db(x, n_fft=n_fft, hop=hop_size)
    mag = np.power(10.0, spec_db / 20.0)
    if mag.shape[1] <= 1:
        return np.zeros(mag.shape[1], dtype=np.float64)
    delta = np.maximum(0.0, mag[:, 1:] - mag[:, :-1])
    flux = np.sqrt(np.sum(delta * delta, axis=0))
    flux = np.concatenate([np.zeros(1, dtype=np.float64), flux], axis=0)
    if np.max(flux) > EPS:
        flux = flux / np.max(flux)
    return flux


def _detect_onsets(x: np.ndarray, sample_rate: int, *, hop_size: int = 256) -> np.ndarray:
    env = _onset_envelope(np.asarray(x, dtype=np.float64).reshape(-1), n_fft=1024, hop_size=hop_size)
    if env.size < 3:
        return np.zeros(0, dtype=np.int64)
    smooth = np.convolve(env, np.array([0.2, 0.6, 0.2], dtype=np.float64), mode="same")
    threshold = float(np.mean(smooth) + 0.55 * np.std(smooth))
    peaks: list[int] = []
    min_sep = max(1, int(round(0.03 * sample_rate / hop_size)))
    last = -10_000_000
    for idx in range(1, smooth.size - 1):
        if smooth[idx] <= threshold:
            continue
        if not (smooth[idx] >= smooth[idx - 1] and smooth[idx] > smooth[idx + 1]):
            continue
        if idx - last < min_sep:
            if peaks and smooth[idx] > smooth[last]:
                peaks[-1] = idx
                last = idx
            continue
        peaks.append(idx)
        last = idx
    return np.asarray(peaks, dtype=np.int64) * int(hop_size)


def _match_events(reference: np.ndarray, candidate: np.ndarray, tolerance: int) -> list[tuple[int, int]]:
    if reference.size == 0 or candidate.size == 0:
        return []
    ref = np.asarray(reference, dtype=np.int64)
    cand = np.asarray(candidate, dtype=np.int64)
    used = np.zeros(ref.size, dtype=bool)
    matches: list[tuple[int, int]] = []
    for ci, c in enumerate(cand):
        diffs = np.abs(ref - c)
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tolerance and not used[idx]:
            used[idx] = True
            matches.append((idx, ci))
    return matches


def _attack_time_ms(x: np.ndarray, onset_sample: int, sample_rate: int) -> float:
    mono = np.asarray(x, dtype=np.float64).reshape(-1)
    if mono.size == 0:
        return float("nan")
    pre = int(round(0.005 * sample_rate))
    win = int(round(0.08 * sample_rate))
    start = max(0, int(onset_sample) - pre)
    stop = min(mono.size, start + win)
    seg = np.abs(mono[start:stop])
    if seg.size < 8:
        return float("nan")
    env = np.convolve(seg, np.ones(max(1, int(round(0.001 * sample_rate)))) / max(1, int(round(0.001 * sample_rate))), mode="same")
    peak = float(np.max(env))
    if peak <= EPS:
        return float("nan")
    lo = 0.1 * peak
    hi = 0.9 * peak
    idx_lo = int(np.argmax(env >= lo))
    idx_hi = int(np.argmax(env >= hi))
    if idx_hi <= idx_lo:
        return float("nan")
    return float(1000.0 * (idx_hi - idx_lo) / max(1, sample_rate))


def _f0_track_and_voicing(
    x: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = 2048,
    hop_size: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 1200.0,
) -> tuple[np.ndarray, np.ndarray]:
    mono = np.asarray(x, dtype=np.float64).reshape(-1)
    if mono.size < frame_size:
        mono = np.pad(mono, (0, frame_size - mono.size), mode="constant")
    rem = (mono.size - frame_size) % hop_size
    if rem:
        mono = np.pad(mono, (0, hop_size - rem), mode="constant")
    frame_count = 1 + (mono.size - frame_size) // hop_size
    f0 = np.zeros(frame_count, dtype=np.float64)
    rms = np.zeros(frame_count, dtype=np.float64)
    for idx in range(frame_count):
        start = idx * hop_size
        frame = mono[start : start + frame_size]
        rms[idx] = float(np.sqrt(np.mean(frame * frame)))
        if rms[idx] <= EPS:
            continue
        try:
            f0[idx] = float(estimate_f0_autocorrelation(frame, sample_rate, f0_min, f0_max))
        except Exception:
            f0[idx] = 0.0
    nz = rms[rms > EPS]
    threshold = max(1e-5, float(np.median(nz)) * 0.4) if nz.size else 1e-5
    voiced = (rms >= threshold) & np.isfinite(f0) & (f0 > 0.0)
    f0[~voiced] = 0.0
    return f0, voiced


def _hnr_track(
    x: np.ndarray,
    sample_rate: int,
    *,
    frame_size: int = 1024,
    hop_size: int = 256,
    f0_min: float = 50.0,
    f0_max: float = 500.0,
) -> np.ndarray:
    mono = np.asarray(x, dtype=np.float64).reshape(-1)
    if mono.size < frame_size:
        mono = np.pad(mono, (0, frame_size - mono.size), mode="constant")
    rem = (mono.size - frame_size) % hop_size
    if rem:
        mono = np.pad(mono, (0, hop_size - rem), mode="constant")
    frame_count = 1 + (mono.size - frame_size) // hop_size
    out = np.zeros(frame_count, dtype=np.float64)
    lag_min = max(1, int(sample_rate / max(f0_max, 1.0)))
    lag_max = max(lag_min + 1, int(sample_rate / max(f0_min, 1.0)))
    win = np.hanning(frame_size).astype(np.float64)
    for idx in range(frame_count):
        start = idx * hop_size
        frame = mono[start : start + frame_size] * win
        if not np.any(np.abs(frame) > EPS):
            continue
        ac = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        ac0 = float(ac[0]) + EPS
        ac = ac / ac0
        hi = min(ac.size, lag_max)
        if hi <= lag_min + 1:
            continue
        r = float(np.max(ac[lag_min:hi]))
        r = float(np.clip(r, 1e-5, 1.0 - 1e-5))
        out[idx] = float(10.0 * np.log10(r / (1.0 - r)))
    return out


def _cross_correlation_lag_samples(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x, y = _match_length(x, y)
    if x.size == 0:
        return 0
    n = int(2 ** np.ceil(np.log2(max(1, x.size + y.size))))
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    cc = np.fft.irfft(X * np.conj(Y), n=n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    lag = int(np.argmax(np.abs(cc)) - max_lag)
    return lag


def _run_external_quality_tool(
    executable: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    metric_patterns: list[str],
    extra_args: list[str] | None = None,
) -> float | None:
    args = list(extra_args or [])
    with tempfile.TemporaryDirectory(prefix="pvx_metric_") as tmp:
        ref_path = os.path.join(tmp, "reference.wav")
        cand_path = os.path.join(tmp, "candidate.wav")
        sf.write(ref_path, np.asarray(reference, dtype=np.float64), int(sample_rate))
        sf.write(cand_path, np.asarray(candidate, dtype=np.float64), int(sample_rate))
        cmd = [executable, "--reference_file", ref_path, "--degraded_file", cand_path, *args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
        except Exception:
            return None
        text = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode != 0:
            return None
        for pattern in metric_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
    return None


def _proxy_quality_scalar(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    lsd: float | None = None,
    modulation: float | None = None,
    smear: float | None = None,
    env_corr: float | None = None,
) -> float:
    lsd_val = float(lsd if lsd is not None else log_spectral_distance(reference, candidate))
    mod_val = float(modulation if modulation is not None else modulation_spectrum_distance(reference, candidate))
    smear_val = float(smear if smear is not None else transient_smear_score(reference, candidate))
    env_val = float(env_corr if env_corr is not None else envelope_correlation(reference, candidate))
    env_norm = float(np.clip((env_val + 1.0) * 0.5, 0.0, 1.0))
    spec = float(np.exp(-lsd_val / 18.0))
    mod = float(np.exp(-mod_val * 1.5))
    trans = float(np.exp(-smear_val * 12.0))
    return float(np.clip(0.40 * env_norm + 0.30 * spec + 0.15 * mod + 0.15 * trans, 0.0, 1.0))


def log_spectral_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_size: int = 512,
) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    a = _stft_mag_db(ref, n_fft=n_fft, hop=hop_size)
    b = _stft_mag_db(cand, n_fft=n_fft, hop=hop_size)
    bins = min(a.shape[0], b.shape[0])
    frames = min(a.shape[1], b.shape[1])
    if bins == 0 or frames == 0:
        return 0.0
    diff = a[:bins, :frames] - b[:bins, :frames]
    return float(np.sqrt(np.mean(diff * diff)))


def modulation_spectrum_distance(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    n_fft: int = 1024,
    hop_size: int = 256,
) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    ref_db = _stft_mag_db(ref, n_fft=n_fft, hop=hop_size)
    cand_db = _stft_mag_db(cand, n_fft=n_fft, hop=hop_size)
    bins = min(ref_db.shape[0], cand_db.shape[0])
    frames = min(ref_db.shape[1], cand_db.shape[1])
    if bins == 0 or frames <= 2:
        return 0.0
    ref_mod = np.abs(np.fft.rfft(ref_db[:bins, :frames], axis=1))
    cand_mod = np.abs(np.fft.rfft(cand_db[:bins, :frames], axis=1))
    mod_bins = min(ref_mod.shape[1], cand_mod.shape[1])
    diff = ref_mod[:, :mod_bins] - cand_mod[:, :mod_bins]
    scale = np.mean(np.abs(ref_mod[:, :mod_bins])) + EPS
    return float(np.sqrt(np.mean(diff * diff)) / scale)


def transient_smear_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    env_ref = _onset_envelope(ref)
    env_cand = _onset_envelope(cand)
    n = min(env_ref.size, env_cand.size)
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(env_ref[:n] - env_cand[:n])))


def signal_to_noise_ratio_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    err = ref - cand
    p_sig = float(np.mean(ref * ref)) + EPS
    p_err = float(np.mean(err * err)) + EPS
    return float(10.0 * np.log10(p_sig / p_err))


def si_sdr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    denom = float(np.dot(ref, ref)) + EPS
    alpha = float(np.dot(cand, ref) / denom)
    target = alpha * ref
    noise = cand - target
    p_target = float(np.dot(target, target)) + EPS
    p_noise = float(np.dot(noise, noise)) + EPS
    return float(10.0 * np.log10(p_target / p_noise))


def spectral_convergence(reference: np.ndarray, candidate: np.ndarray, *, n_fft: int = 2048, hop_size: int = 512) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    a = np.abs(_stft_complex(ref, n_fft=n_fft, hop=hop_size))
    b = np.abs(_stft_complex(cand, n_fft=n_fft, hop=hop_size))
    bins = min(a.shape[0], b.shape[0])
    frames = min(a.shape[1], b.shape[1])
    if bins == 0 or frames == 0:
        return 0.0
    num = float(np.linalg.norm(a[:bins, :frames] - b[:bins, :frames]))
    den = float(np.linalg.norm(a[:bins, :frames])) + EPS
    return num / den


def envelope_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size < 8:
        return 1.0
    env_ref = _onset_envelope(ref, n_fft=1024, hop_size=256)
    env_cand = _onset_envelope(cand, n_fft=1024, hop_size=256)
    n = min(env_ref.size, env_cand.size)
    if n < 3:
        return 1.0
    a = env_ref[:n] - float(np.mean(env_ref[:n]))
    b = env_cand[:n] - float(np.mean(env_cand[:n]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(np.dot(a, b) / denom)


def rms_level_delta_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    rms_ref = float(np.sqrt(np.mean(ref * ref))) + EPS
    rms_cand = float(np.sqrt(np.mean(cand * cand))) + EPS
    return float(20.0 * np.log10(rms_cand / rms_ref))


def crest_factor_delta_db(reference: np.ndarray, candidate: np.ndarray, *, sample_rate: int = 48000) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    s_ref = summarize_audio_metrics(ref[:, None], sample_rate)
    s_cand = summarize_audio_metrics(cand[:, None], sample_rate)
    return float(s_cand.crest_db - s_ref.crest_db)


def bandwidth_95_delta_hz(reference: np.ndarray, candidate: np.ndarray, *, sample_rate: int = 48000) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    s_ref = summarize_audio_metrics(ref[:, None], sample_rate)
    s_cand = summarize_audio_metrics(cand[:, None], sample_rate)
    return float(s_cand.bandwidth_95_hz - s_ref.bandwidth_95_hz)


def zero_crossing_rate_delta(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size < 2:
        return 0.0
    zr = float(np.mean(np.signbit(ref[:-1]) != np.signbit(ref[1:])))
    zc = float(np.mean(np.signbit(cand[:-1]) != np.signbit(cand[1:])))
    return zc - zr


def dc_offset_delta(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    return float(np.mean(cand) - np.mean(ref))


def clipping_ratio_delta(reference: np.ndarray, candidate: np.ndarray, *, threshold: float = 0.999) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    cr = float(np.mean(np.abs(ref) >= threshold))
    cc = float(np.mean(np.abs(cand) >= threshold))
    return cc - cr


def integrated_lufs_delta_lu(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    def _integrated_lufs(x: np.ndarray) -> float:
        signal = np.asarray(x, dtype=np.float64).reshape(-1)
        try:
            import pyloudnorm as pyln  # type: ignore

            meter = pyln.Meter(sample_rate)
            return float(meter.integrated_loudness(signal))
        except Exception:
            rms = float(np.sqrt(np.mean(signal * signal)) + EPS)
            return float(-0.691 + 20.0 * np.log10(rms + EPS))

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    return float(_integrated_lufs(cand) - _integrated_lufs(ref))


def short_term_lufs_delta_lu(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    def _integrated_lufs_frame(signal: np.ndarray) -> float:
        try:
            import pyloudnorm as pyln  # type: ignore

            meter = pyln.Meter(sample_rate)
            return float(meter.integrated_loudness(signal))
        except Exception:
            rms = float(np.sqrt(np.mean(signal * signal)) + EPS)
            return float(-0.691 + 20.0 * np.log10(rms + EPS))

    def _short_term_lufs(x: np.ndarray) -> float:
        signal = np.asarray(x, dtype=np.float64).reshape(-1)
        win = max(1, int(round(3.0 * sample_rate)))
        hop = max(1, int(round(1.0 * sample_rate)))
        if signal.size <= win:
            return _integrated_lufs_frame(signal)
        vals: list[float] = []
        for start in range(0, signal.size - win + 1, hop):
            seg = signal[start : start + win]
            vals.append(_integrated_lufs_frame(seg))
        if not vals:
            return 0.0
        return float(np.mean(vals))

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    return float(_short_term_lufs(cand) - _short_term_lufs(ref))


def loudness_range_delta_lu(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    def _lra(x: np.ndarray) -> float:
        signal = np.asarray(x, dtype=np.float64).reshape(-1)
        try:
            import pyloudnorm as pyln  # type: ignore

            meter = pyln.Meter(sample_rate)
            return float(meter.loudness_range(signal))
        except Exception:
            win = max(1, int(round(3.0 * sample_rate)))
            hop = max(1, int(round(1.0 * sample_rate)))
            values: list[float] = []
            for start in range(0, max(1, signal.size - win + 1), hop):
                seg = signal[start : start + win]
                if seg.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(seg * seg)) + EPS)
                values.append(float(20.0 * np.log10(rms + EPS)))
            if len(values) < 2:
                return 0.0
            return float(np.percentile(values, 95) - np.percentile(values, 10))

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    return float(_lra(cand) - _lra(ref))


def true_peak_delta_dbtp(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    def _true_peak_dbtp(x: np.ndarray) -> float:
        signal = np.asarray(x, dtype=np.float64).reshape(-1)
        up = _resample_signal(signal, sample_rate, sample_rate * 4)
        peak = float(np.max(np.abs(up))) if up.size else 0.0
        return float(20.0 * np.log10(max(peak, EPS)))

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    return float(_true_peak_dbtp(cand) - _true_peak_dbtp(ref))


def pesq_mos_lqo(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    lsd: float | None = None,
    modulation: float | None = None,
    smear: float | None = None,
    env_corr: float | None = None,
) -> OptionalMetricValue:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return OptionalMetricValue(value=1.0, proxy_used=True)
    target_sr = 16000 if sample_rate >= 16000 else 8000
    mode = "wb" if target_sr == 16000 else "nb"
    ref_rs = _resample_signal(ref, sample_rate, target_sr)
    cand_rs = _resample_signal(cand, sample_rate, target_sr)
    try:
        from pesq import pesq as pesq_api  # type: ignore

        score = float(pesq_api(target_sr, ref_rs, cand_rs, mode))
        return OptionalMetricValue(value=score, proxy_used=False)
    except Exception:
        q = _proxy_quality_scalar(ref_rs, cand_rs, lsd=lsd, modulation=modulation, smear=smear, env_corr=env_corr)
        return OptionalMetricValue(value=float(1.0 + 3.5 * q), proxy_used=True)


def stoi_score(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    extended: bool = False,
    lsd: float | None = None,
    modulation: float | None = None,
    smear: float | None = None,
    env_corr: float | None = None,
) -> OptionalMetricValue:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return OptionalMetricValue(value=1.0, proxy_used=True)
    try:
        from pystoi import stoi as stoi_api  # type: ignore

        score = float(stoi_api(ref, cand, sample_rate, extended=bool(extended)))
        return OptionalMetricValue(value=score, proxy_used=False)
    except Exception:
        q = _proxy_quality_scalar(ref, cand, lsd=lsd, modulation=modulation, smear=smear, env_corr=env_corr)
        if extended:
            q = float(np.clip(0.9 * q + 0.1 * np.exp(-float(modulation or 0.0)), 0.0, 1.0))
        return OptionalMetricValue(value=q, proxy_used=True)


def visqol_mos_lqo(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    lsd: float | None = None,
    modulation: float | None = None,
    smear: float | None = None,
    env_corr: float | None = None,
) -> OptionalMetricValue:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return OptionalMetricValue(value=1.0, proxy_used=True)
    visqol_exe = os.getenv("VISQOL_BIN") or shutil.which("visqol")
    if visqol_exe:
        score = _run_external_quality_tool(
            visqol_exe,
            ref,
            cand,
            sample_rate,
            metric_patterns=[
                r"mos[-_ ]?lqo[^0-9\-]*([0-9]+(?:\.[0-9]+)?)",
                r"overall[^0-9\-]*([0-9]+(?:\.[0-9]+)?)",
            ],
            extra_args=[],
        )
        if score is not None and np.isfinite(score):
            return OptionalMetricValue(value=float(score), proxy_used=False)
    q = _proxy_quality_scalar(ref, cand, lsd=lsd, modulation=modulation, smear=smear, env_corr=env_corr)
    return OptionalMetricValue(value=float(1.0 + 4.0 * q), proxy_used=True)


def polqa_mos_lqo(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    lsd: float | None = None,
    modulation: float | None = None,
    smear: float | None = None,
    env_corr: float | None = None,
) -> OptionalMetricValue:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return OptionalMetricValue(value=1.0, proxy_used=True)
    polqa_exe = os.getenv("POLQA_BIN")
    if polqa_exe:
        score = _run_external_quality_tool(
            polqa_exe,
            ref,
            cand,
            sample_rate,
            metric_patterns=[
                r"mos[-_ ]?lqo[^0-9\-]*([0-9]+(?:\.[0-9]+)?)",
                r"polqa[^0-9\-]*([0-9]+(?:\.[0-9]+)?)",
            ],
            extra_args=[],
        )
        if score is not None and np.isfinite(score):
            return OptionalMetricValue(value=float(score), proxy_used=False)
    q = _proxy_quality_scalar(ref, cand, lsd=lsd, modulation=modulation, smear=smear, env_corr=env_corr)
    return OptionalMetricValue(value=float(1.0 + 4.0 * q), proxy_used=True)


def peaq_odg(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    lsd: float | None = None,
    modulation: float | None = None,
    smear: float | None = None,
    env_corr: float | None = None,
) -> OptionalMetricValue:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return OptionalMetricValue(value=0.0, proxy_used=True)
    peaq_exe = os.getenv("PEAQ_BIN")
    if peaq_exe:
        score = _run_external_quality_tool(
            peaq_exe,
            ref,
            cand,
            sample_rate,
            metric_patterns=[
                r"\bodg\b[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)",
                r"objective[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)",
            ],
            extra_args=[],
        )
        if score is not None and np.isfinite(score):
            return OptionalMetricValue(value=float(score), proxy_used=False)
    q = _proxy_quality_scalar(ref, cand, lsd=lsd, modulation=modulation, smear=smear, env_corr=env_corr)
    odg = float(-4.0 * (1.0 - q))
    return OptionalMetricValue(value=odg, proxy_used=True)


def f0_rmse_cents(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    ref_f0, ref_voiced = _f0_track_and_voicing(reference, sample_rate)
    cand_f0, cand_voiced = _f0_track_and_voicing(candidate, sample_rate)
    n = min(ref_f0.size, cand_f0.size)
    if n == 0:
        return 0.0
    ref_f0 = ref_f0[:n]
    cand_f0 = cand_f0[:n]
    mask = ref_voiced[:n] & cand_voiced[:n] & (ref_f0 > 0.0) & (cand_f0 > 0.0)
    if not np.any(mask):
        return 0.0
    cents = 1200.0 * np.log2((cand_f0[mask] + EPS) / (ref_f0[mask] + EPS))
    return float(np.sqrt(np.mean(cents * cents)))


def voicing_f1_score(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    _, ref_voiced = _f0_track_and_voicing(reference, sample_rate)
    _, cand_voiced = _f0_track_and_voicing(candidate, sample_rate)
    n = min(ref_voiced.size, cand_voiced.size)
    if n == 0:
        return 1.0
    rv = ref_voiced[:n]
    cv = cand_voiced[:n]
    tp = float(np.sum(rv & cv))
    fp = float(np.sum(~rv & cv))
    fn = float(np.sum(rv & ~cv))
    precision = tp / max(EPS, tp + fp)
    recall = tp / max(EPS, tp + fn)
    return float(2.0 * precision * recall / max(EPS, precision + recall))


def harmonic_to_noise_ratio_drift_db(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    h_ref = _hnr_track(reference, sample_rate)
    h_cand = _hnr_track(candidate, sample_rate)
    n = min(h_ref.size, h_cand.size)
    if n == 0:
        return 0.0
    return float(np.mean(h_cand[:n]) - np.mean(h_ref[:n]))


def onset_precision_recall_f1(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    tolerance_ms: float = 30.0,
) -> tuple[float, float, float]:
    ref_events = _detect_onsets(reference, sample_rate)
    cand_events = _detect_onsets(candidate, sample_rate)
    tol = max(1, int(round(float(tolerance_ms) * sample_rate / 1000.0)))
    matches = _match_events(ref_events, cand_events, tol)
    tp = float(len(matches))
    precision = tp / max(EPS, float(cand_events.size))
    recall = tp / max(EPS, float(ref_events.size))
    f1 = 2.0 * precision * recall / max(EPS, precision + recall)
    return float(precision), float(recall), float(f1)


def attack_time_error_ms(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    ref_events = _detect_onsets(reference, sample_rate)
    cand_events = _detect_onsets(candidate, sample_rate)
    if ref_events.size == 0 or cand_events.size == 0:
        return 0.0
    tol = max(1, int(round(0.03 * sample_rate)))
    matches = _match_events(ref_events, cand_events, tol)
    if not matches:
        return float(np.mean(np.abs(ref_events[: min(ref_events.size, cand_events.size)] - cand_events[: min(ref_events.size, cand_events.size)])) * 1000.0 / max(1, sample_rate))
    errors: list[float] = []
    for ri, ci in matches:
        rt = _attack_time_ms(reference, int(ref_events[ri]), sample_rate)
        ct = _attack_time_ms(candidate, int(cand_events[ci]), sample_rate)
        if np.isfinite(rt) and np.isfinite(ct):
            errors.append(abs(ct - rt))
    if not errors:
        return 0.0
    return float(np.mean(errors))


def ild_drift_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape[1] < 2 or cand.shape[1] < 2:
        return float("nan")
    n = min(ref.shape[0], cand.shape[0])
    if n == 0:
        return float("nan")
    ref = ref[:n, :2]
    cand = cand[:n, :2]

    def _ild(x: np.ndarray) -> float:
        l_rms = float(np.sqrt(np.mean(x[:, 0] * x[:, 0])) + EPS)
        r_rms = float(np.sqrt(np.mean(x[:, 1] * x[:, 1])) + EPS)
        return float(20.0 * np.log10(l_rms / r_rms))

    return float(_ild(cand) - _ild(ref))


def itd_drift_ms(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape[1] < 2 or cand.shape[1] < 2:
        return float("nan")
    max_lag = max(1, int(round(0.002 * sample_rate)))
    ref_lag = _cross_correlation_lag_samples(ref[:, 0], ref[:, 1], max_lag=max_lag)
    cand_lag = _cross_correlation_lag_samples(cand[:, 0], cand[:, 1], max_lag=max_lag)
    return float(1000.0 * (cand_lag - ref_lag) / max(1, sample_rate))


def interchannel_phase_deviation_by_band(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 1024,
    hop_size: int = 256,
) -> dict[str, float]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.ndim != 2 or cand.ndim != 2 or ref.shape[1] < 2 or cand.shape[1] < 2:
        return {
            "phase_deviation_low_rad": float("nan"),
            "phase_deviation_mid_rad": float("nan"),
            "phase_deviation_high_rad": float("nan"),
            "phase_deviation_mean_rad": float("nan"),
        }
    n = min(ref.shape[0], cand.shape[0])
    ref = ref[:n, :2]
    cand = cand[:n, :2]
    ref0 = _stft_complex(ref[:, 0], n_fft=n_fft, hop=hop_size)
    ref1 = _stft_complex(ref[:, 1], n_fft=n_fft, hop=hop_size)
    cand0 = _stft_complex(cand[:, 0], n_fft=n_fft, hop=hop_size)
    cand1 = _stft_complex(cand[:, 1], n_fft=n_fft, hop=hop_size)
    bins = min(ref0.shape[0], ref1.shape[0], cand0.shape[0], cand1.shape[0])
    frames = min(ref0.shape[1], ref1.shape[1], cand0.shape[1], cand1.shape[1])
    if bins == 0 or frames == 0:
        return {
            "phase_deviation_low_rad": float("nan"),
            "phase_deviation_mid_rad": float("nan"),
            "phase_deviation_high_rad": float("nan"),
            "phase_deviation_mean_rad": float("nan"),
        }
    ref_ipd = _principal_angle(np.angle(ref0[:bins, :frames]) - np.angle(ref1[:bins, :frames]))
    cand_ipd = _principal_angle(np.angle(cand0[:bins, :frames]) - np.angle(cand1[:bins, :frames]))
    diff = np.abs(_principal_angle(cand_ipd - ref_ipd))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))[:bins]

    def _band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(diff[mask, :]))

    low = _band(20.0, 500.0)
    mid = _band(500.0, 2000.0)
    high = _band(2000.0, min(8000.0, float(sample_rate) * 0.5))
    mean_val = float(np.nanmean(np.array([low, mid, high], dtype=np.float64)))
    return {
        "phase_deviation_low_rad": low,
        "phase_deviation_mid_rad": mid,
        "phase_deviation_high_rad": high,
        "phase_deviation_mean_rad": mean_val,
    }


def phasiness_index(reference: np.ndarray, candidate: np.ndarray, *, n_fft: int = 1024, hop_size: int = 256) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    ref_spec = _stft_complex(ref, n_fft=n_fft, hop=hop_size)
    cand_spec = _stft_complex(cand, n_fft=n_fft, hop=hop_size)
    bins = min(ref_spec.shape[0], cand_spec.shape[0])
    frames = min(ref_spec.shape[1], cand_spec.shape[1])
    if bins == 0 or frames == 0:
        return 0.0
    phase_diff = _principal_angle(np.angle(cand_spec[:bins, :frames]) - np.angle(ref_spec[:bins, :frames]))
    coherence = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
    return float(np.mean(1.0 - coherence))


def musical_noise_index(reference: np.ndarray, candidate: np.ndarray, *, n_fft: int = 1024, hop_size: int = 256) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    residual = np.abs(_stft_complex(cand - ref, n_fft=n_fft, hop=hop_size))
    if residual.size == 0:
        return 0.0
    mean_mag = np.mean(residual, axis=0) + EPS
    geom_mag = np.exp(np.mean(np.log(residual + EPS), axis=0))
    flatness = geom_mag / mean_mag
    sparsity = 1.0 - np.clip(flatness, 0.0, 1.0)
    frame_var = np.std(residual, axis=0) / mean_mag
    return float(np.mean(sparsity * np.clip(frame_var, 0.0, 10.0)))


def pre_echo_score(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref, cand = _match_length(ref, cand)
    if ref.size == 0:
        return 0.0
    onsets = _detect_onsets(ref, sample_rate)
    if onsets.size == 0:
        return 0.0
    pre = max(1, int(round(0.02 * sample_rate)))
    post = max(1, int(round(0.02 * sample_rate)))
    vals: list[float] = []
    for onset in onsets:
        s0 = int(max(0, onset - pre))
        s1 = int(onset)
        s2 = int(min(ref.size, onset + post))
        if s1 <= s0 or s2 <= s1:
            continue
        ref_pre = float(np.mean(ref[s0:s1] * ref[s0:s1]))
        ref_post = float(np.mean(ref[s1:s2] * ref[s1:s2])) + EPS
        cand_pre = float(np.mean(cand[s0:s1] * cand[s0:s1]))
        cand_post = float(np.mean(cand[s1:s2] * cand[s1:s2])) + EPS
        ratio_ref = ref_pre / ref_post
        ratio_cand = cand_pre / cand_post
        vals.append(max(0.0, ratio_cand - ratio_ref))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def stereo_coherence_drift(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    n_fft: int = 1024,
    hop_size: int = 256,
) -> float:
    return float(
        stereo_coherence_drift_score(
            np.asarray(reference, dtype=np.float64),
            np.asarray(candidate, dtype=np.float64),
            n_fft=n_fft,
            hop_size=hop_size,
        )
    )
