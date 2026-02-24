#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Frame-level feature tracking for control-rate audio modulation maps."""

from __future__ import annotations

import math

import librosa
import numpy as np

EPS = 1e-12


def _safe_div(
    numer: np.ndarray | float, denom: np.ndarray | float
) -> np.ndarray | float:
    return np.asarray(numer) / np.maximum(np.asarray(denom), EPS)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    n_bins = (n_fft // 2) + 1
    hz_bins = np.linspace(0.0, float(sr) * 0.5, num=n_bins, dtype=np.float64)
    mel_min = _hz_to_mel(np.asarray([fmin], dtype=np.float64))[0]
    mel_max = _hz_to_mel(np.asarray([fmax], dtype=np.float64))[0]
    mel_points = np.linspace(mel_min, mel_max, num=n_mels + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]
        left_slope = _safe_div((hz_bins - left), (center - left))
        right_slope = _safe_div((right - hz_bins), (right - center))
        tri = np.maximum(0.0, np.minimum(left_slope, right_slope))
        fb[i, :] = tri
    norm = np.sum(fb, axis=1, keepdims=True)
    fb = np.asarray(_safe_div(fb, norm), dtype=np.float64)
    return fb


def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """Vectorized Levinson-Durbin recursion.

    Args:
        r: (order+1, n_frames) autocorrelation coefficients.
        order: LPC order.

    Returns:
        a: (order+1, n_frames) prediction error filter coefficients.
    """
    n_frames = r.shape[1]
    a = np.zeros((1, n_frames), dtype=r.dtype)
    a[0] = 1.0
    e = r[0].copy()

    for k in range(order):
        # We need r[k+1] ... r[1] reversed.
        r_slice = r[1 : k + 2][::-1]
        delta = np.sum(a * r_slice, axis=0)
        gamma = -delta / np.maximum(e, EPS)

        a_pad = np.vstack([a, np.zeros((1, n_frames), dtype=r.dtype)])
        a_rev_pad = np.vstack([np.zeros((1, n_frames), dtype=r.dtype), a[::-1]])

        a = a_pad + gamma * a_rev_pad
        e = e * (1.0 - gamma * gamma)

    return a


def _acf_peak_ratio(
    frame: np.ndarray, sr: int, fmin: float, fmax: float
) -> tuple[float, float]:
    work = np.asarray(frame, dtype=np.float64)
    work = work - float(np.mean(work))
    n = int(work.size)
    if n < 8:
        return 0.0, 0.0
    energy = float(np.dot(work, work))
    if energy <= EPS:
        return 0.0, 0.0
    corr = np.correlate(work, work, mode="full")[n - 1 :]
    if corr.size < 3:
        return 0.0, 0.0
    min_lag = max(1, int(sr / max(fmax, 1e-9)))
    max_lag = min(corr.size - 1, int(sr / max(fmin, 1e-9)))
    if max_lag <= min_lag:
        return 0.0, 0.0
    window = corr[min_lag : max_lag + 1]
    lag = int(np.argmax(window)) + min_lag
    peak = float(corr[lag])
    conf = float(np.clip(peak / max(corr[0], EPS), 0.0, 1.0))
    hz = float(sr / lag) if lag > 0 else 0.0
    return hz, conf


def _estimate_formants_lpc(
    frame: np.ndarray, sr: int, count: int = 3
) -> tuple[float, float, float]:
    work = np.asarray(frame, dtype=np.float64)
    if work.size < 32:
        return float("nan"), float("nan"), float("nan")
    pre = np.copy(work)
    pre[1:] = pre[1:] - 0.97 * pre[:-1]
    pre *= np.hamming(pre.size)
    order = int(min(20, max(8, sr // 1000 + 2)))
    corr = np.correlate(pre, pre, mode="full")[pre.size - 1 : pre.size + order + 1]
    if corr.size < order + 1 or corr[0] <= EPS:
        return float("nan"), float("nan"), float("nan")

    r = np.asarray(corr[: order + 1], dtype=np.float64)
    toeplitz = np.empty((order, order), dtype=np.float64)
    for i in range(order):
        for j in range(order):
            toeplitz[i, j] = r[abs(i - j)]
    try:
        a = np.linalg.solve(toeplitz + (1e-9 * np.eye(order)), r[1 : order + 1])
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), float("nan")
    lpc = np.concatenate(([1.0], -a))
    roots = np.roots(lpc)
    roots = roots[np.imag(roots) > 0.01]
    if roots.size == 0:
        return float("nan"), float("nan"), float("nan")
    ang = np.angle(roots)
    freqs = ang * (float(sr) / (2.0 * np.pi))
    bw = -0.5 * (float(sr) / np.pi) * np.log(np.maximum(np.abs(roots), EPS))
    valid = (freqs > 90.0) & (freqs < 5000.0) & (bw < 700.0)
    formants = np.sort(freqs[valid])
    if formants.size == 0:
        return float("nan"), float("nan"), float("nan")
    padded = np.full((count,), np.nan, dtype=np.float64)
    padded[: min(count, formants.size)] = formants[:count]
    return float(padded[0]), float(padded[1]), float(padded[2])


def _estimate_tempo_bpm(onset_env: np.ndarray, hop_size: int, sr: int) -> float:
    env = np.asarray(onset_env, dtype=np.float64)
    if env.size < 8 or float(np.max(env)) <= EPS:
        return 120.0
    centered = env - float(np.mean(env))
    ac = np.correlate(centered, centered, mode="full")[env.size - 1 :]
    lag_min = max(1, int((60.0 / 240.0) * float(sr) / float(hop_size)))
    lag_max = min(ac.size - 1, int((60.0 / 40.0) * float(sr) / float(hop_size)))
    if lag_max <= lag_min:
        return 120.0
    lag = int(np.argmax(ac[lag_min : lag_max + 1])) + lag_min
    bpm = 60.0 * float(sr) / (float(lag) * float(hop_size))
    return float(np.clip(bpm, 40.0, 240.0))


def _estimate_inharmonicity(
    mag: np.ndarray,
    freqs: np.ndarray,
    f0_hz: float,
    max_partials: int = 8,
) -> float:
    if f0_hz <= 0.0 or mag.size < 8:
        return 0.0
    nyq = float(freqs[-1])

    targets = np.arange(1, max_partials + 1, dtype=np.float64) * f0_hz
    targets = targets[targets < nyq]

    if targets.size == 0:
        return 0.0

    m = np.asarray(mag, dtype=np.float64)
    if m.size < 3:
        return 0.0

    is_peak = (m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:])
    peak_indices = np.flatnonzero(is_peak) + 1

    if peak_indices.size == 0:
        return 0.0

    pf = freqs[peak_indices]
    pm = m[peak_indices]

    # Vectorized matching of partials to nearest peaks
    # targets (T,), pf (P,).
    diffs = np.abs(pf[None, :] - targets[:, None])  # (T, P)
    best_peak_indices = np.argmin(diffs, axis=1)  # (T,)

    closest_pf = pf[best_peak_indices]
    closest_pm = pm[best_peak_indices]

    devs = np.abs(closest_pf - targets) / np.maximum(targets, EPS)
    weights = closest_pm

    w_sum = np.sum(weights)
    if w_sum <= EPS:
        return float(np.mean(devs))

    return float(np.sum(devs * weights) / w_sum)


def extract_feature_tracks(
    audio: np.ndarray,
    sr: int,
    *,
    frame_length: int,
    hop_size: int,
    f0_hz: np.ndarray,
    confidence: np.ndarray,
    mfcc_count: int = 13,
    fmin: float = 50.0,
    fmax: float = 1200.0,
) -> dict[str, np.ndarray]:
    """Extract a broad control-rate feature set from audio frames."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    mono = np.mean(x, axis=1)
    n_frames = int(f0_hz.size)
    if n_frames <= 0:
        return {}

    n_fft = int(frame_length)
    window = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr)).astype(np.float64)
    mel_fb = _mel_filterbank(
        sr=int(sr),
        n_fft=n_fft,
        n_mels=40,
        fmin=20.0,
        fmax=min(float(sr) * 0.5, 20000.0),
    )
    mfcc_count = max(0, min(int(mfcc_count), 40))

    left = np.asarray(x[:, 0], dtype=np.float64)
    right = np.asarray(x[:, 1], dtype=np.float64) if x.shape[1] >= 2 else left

    # Prepare padded signals for vectorized framing.
    required_samples = int((n_frames - 1) * hop_size + n_fft)
    if mono.size < required_samples:
        pad_width = required_samples - mono.size
        mono = np.pad(mono, (0, pad_width))
        left = np.pad(left, (0, pad_width))
        right = np.pad(right, (0, pad_width))

    # Create frame views: (n_fft, n_frames)
    # librosa.util.frame returns (frame_length, n_frames)
    # We slice to exactly n_frames to handle cases where the input audio is longer than needed.
    frames_mono = librosa.util.frame(mono, frame_length=n_fft, hop_length=hop_size)[
        :, :n_frames
    ]
    frames_left = librosa.util.frame(left, frame_length=n_fft, hop_length=hop_size)[
        :, :n_frames
    ]
    frames_right = librosa.util.frame(right, frame_length=n_fft, hop_length=hop_size)[
        :, :n_frames
    ]

    # Vectorized time-domain features.
    frames_mono_sq = frames_mono * frames_mono
    rms = np.sqrt(np.mean(frames_mono_sq, axis=0))
    frames_mono_abs = np.abs(frames_mono)
    peak_val = np.max(frames_mono_abs, axis=0)

    rms_db = 20.0 * np.log10(np.maximum(rms, EPS))
    clip_ratio = np.mean(frames_mono_abs >= 0.999, axis=0)
    crest_db = 20.0 * np.log10(np.maximum(peak_val, EPS) / np.maximum(rms, EPS))

    zc = np.abs(np.diff(np.signbit(frames_mono), axis=0)).astype(np.float64)
    zcr = (
        np.mean(zc, axis=0) if zc.size > 0 else np.zeros((n_frames,), dtype=np.float64)
    )
    # Vectorized spectral features.
    window_broad = window[:, None]
    # FFT on axis 0, then transpose to (n_frames, n_bins) for better locality in loop.
    spec = np.fft.rfft(frames_mono * window_broad, axis=0).T
    mag = np.abs(spec)
    pwr = mag * mag
    total_mag = np.sum(mag, axis=1)
    total_pwr = np.sum(pwr, axis=1)

    freqs_broad = freqs[None, :]
    centroid = np.sum(freqs_broad * mag, axis=1) / np.maximum(total_mag, EPS)

    centered = freqs_broad - centroid[:, None]
    spread = np.sqrt(
        np.sum((centered * centered) * mag, axis=1) / np.maximum(total_mag, EPS)
    )

    flatness = np.exp(np.mean(np.log(pwr + EPS), axis=1)) / np.maximum(
        np.mean(pwr, axis=1), EPS
    )

    mag_diff = np.diff(
        mag, axis=0, prepend=np.zeros((1, mag.shape[1]), dtype=np.float64)
    )
    flux = np.sum(np.maximum(mag_diff, 0.0), axis=1)
    onset_strength = flux

    csum = np.cumsum(pwr, axis=1)
    thresholds = 0.95 * csum[:, -1]
    rolloff_indices = np.argmax(csum >= thresholds[:, None], axis=1)
    rolloff = freqs[rolloff_indices]
    harmonic_ratio = np.zeros((n_frames,), dtype=np.float64)
    formant_f1 = np.full((n_frames,), np.nan, dtype=np.float64)
    formant_f2 = np.full((n_frames,), np.nan, dtype=np.float64)
    formant_f3 = np.full((n_frames,), np.nan, dtype=np.float64)
    inharmonicity = np.zeros((n_frames,), dtype=np.float64)
    itd_ms = np.zeros((n_frames,), dtype=np.float64)
    itd_max_lag = int(round((1.0e-3) * float(sr)))  # ~1 ms

    spectral_crest = np.max(mag, axis=1) / np.maximum(np.mean(mag, axis=1), EPS)

    ref_mag = mag[:, 1:]
    ref_first = ref_mag[:, 0]
    n_vec = np.arange(2, mag.shape[1] + 1, dtype=np.float64)[None, :]
    spectral_decrease = np.sum(
        (ref_mag - ref_first[:, None]) / n_vec, axis=1
    ) / np.maximum(np.sum(ref_mag, axis=1), EPS)
    # Handle single bin case (unlikely but safe)
    if mag.shape[1] <= 1:
        spectral_decrease[:] = 0.0

    hi_band = freqs >= 6000.0
    hiss_ratio = np.sum(pwr[:, hi_band], axis=1) / np.maximum(total_pwr, EPS)

    def _vectorized_hum_ratio(base_hz, power_spec, freqs, total_pwr_arr):
        accum = np.zeros(power_spec.shape[0], dtype=np.float64)
        for h in range(1, 6):
            target = h * base_hz
            if target >= freqs[-1]:
                break
            b = int(np.argmin(np.abs(freqs - target)))
            lo_b = max(0, b - 1)
            hi_b = min(power_spec.shape[1], b + 2)
            accum += np.sum(power_spec[:, lo_b:hi_b], axis=1)
        return accum / np.maximum(total_pwr_arr, EPS)

    hum_50 = _vectorized_hum_ratio(50.0, pwr, freqs, total_pwr)
    hum_60 = _vectorized_hum_ratio(60.0, pwr, freqs, total_pwr)

    # MPEG-7 ASE
    edges = np.linspace(0, pwr.shape[1], num=11, dtype=np.int64)
    ase = np.zeros((n_frames, 10), dtype=np.float64)
    for b in range(10):
        sl = pwr[:, edges[b] : edges[b + 1]]
        sub_sum = np.sum(sl, axis=1)
        ase[:, b] = np.log10(np.maximum(sub_sum, EPS))
    ase_mean = np.mean(ase, axis=1, keepdims=True)
    ase -= ase_mean

    if mfcc_count > 0:
        mel_ener = pwr @ mel_fb.T
        mel_log = np.log(mel_ener + EPS)
        n_mels = mel_fb.shape[0]
        k = np.arange(mfcc_count, dtype=np.float64)[:, None]
        i = np.arange(n_mels, dtype=np.float64)[None, :]
        basis = np.cos(np.pi * (i + 0.5) * k / float(n_mels))
        scale = np.sqrt(2.0 / float(n_mels))
        dct_basis = scale * basis
        mfcc = mel_log @ dct_basis.T
        if mfcc_count > 0:
            mfcc[:, 0] *= 1.0 / math.sqrt(2.0)
    else:
        mfcc = np.zeros((n_frames, mfcc_count), dtype=np.float64)

    # Stereo features (RMS, ILD)
    lrms = np.sqrt(np.mean(frames_left * frames_left, axis=0))
    rrms = np.sqrt(np.mean(frames_right * frames_right, axis=0))
    ild_db = 20.0 * np.log10(np.maximum(lrms, EPS) / np.maximum(rrms, EPS))

    # Vectorized Harmonic Ratio (ACF)
    frames_centered = frames_mono - np.mean(frames_mono, axis=0)
    n_fft_pad = 2 * n_fft
    frames_padded = np.pad(frames_centered, ((0, n_fft), (0, 0)))
    spec_raw = np.fft.rfft(frames_padded, axis=0)
    pwr_raw = spec_raw * np.conj(spec_raw)
    acf = np.fft.irfft(pwr_raw, n=n_fft_pad, axis=0).real

    min_lag = max(1, int(sr / max(fmax, 1e-9)))
    max_lag_limit = min(acf.shape[0] - 1, int(sr / max(fmin, 1e-9)))

    if max_lag_limit > min_lag:
        acf_window = acf[min_lag : max_lag_limit + 1, :]
        lag_indices = np.argmax(acf_window, axis=0)
        best_lags = lag_indices + min_lag
        peaks = acf[best_lags, np.arange(n_frames)]
        energy = acf[0, :]
        harmonic_ratio = np.clip(peaks / np.maximum(energy, EPS), 0.0, 1.0)
        # Note: hz can also be extracted here if needed: sr / best_lags
    else:
        harmonic_ratio[:] = 0.0

    # Vectorized ITD (Cross-correlation)
    fr_l_padded = np.pad(frames_left, ((0, n_fft), (0, 0)))
    fr_r_padded = np.pad(frames_right, ((0, n_fft), (0, 0)))
    spec_l = np.fft.rfft(fr_l_padded, axis=0)
    spec_r = np.fft.rfft(fr_r_padded, axis=0)
    x_spec = spec_l * np.conj(spec_r)
    x_corr = np.fft.irfft(x_spec, n=n_fft_pad, axis=0).real

    lags = np.arange(-itd_max_lag, itd_max_lag + 1)
    indices = np.mod(lags, n_fft_pad)
    x_corr_window = x_corr[indices, :]
    lag_idx = np.argmax(x_corr_window, axis=0)
    best_lags = lags[lag_idx]
    itd_ms = 1000.0 * best_lags / float(sr)

    # Vectorized LPC (Formants)
    # Pre-emphasis
    frames_pre = frames_mono.copy()
    frames_pre[1:, :] -= 0.97 * frames_pre[:-1, :]
    frames_pre *= np.hamming(n_fft)[:, None]

    lpc_order = int(min(20, max(8, sr // 1000 + 2)))

    # Calculate autocorrelation using FFT
    frames_pre_padded = np.pad(frames_pre, ((0, n_fft), (0, 0)))
    spec_pre = np.fft.rfft(frames_pre_padded, axis=0)
    pwr_pre = spec_pre * np.conj(spec_pre)
    r = np.fft.irfft(pwr_pre, n=n_fft_pad, axis=0).real
    r = r[: lpc_order + 1, :]

    lpc_coeffs = _levinson_durbin(r, lpc_order)

    # Loop for remaining parts (Roots for Formants, Inharmonicity)
    for idx in range(n_frames):
        mag_i = mag[idx, :]
        inharmonicity[idx] = _estimate_inharmonicity(mag_i, freqs, float(f0_hz[idx]))

        # Formants from LPC coeffs
        a = lpc_coeffs[:, idx]
        # Roots
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0.01]
        if roots.size == 0:
            continue

        ang = np.angle(roots)
        freqs_formant = ang * (float(sr) / (2.0 * np.pi))
        bw = -0.5 * (float(sr) / np.pi) * np.log(np.maximum(np.abs(roots), EPS))
        valid = (freqs_formant > 90.0) & (freqs_formant < 5000.0) & (bw < 700.0)
        formants = np.sort(freqs_formant[valid])

        count = 3
        if formants.size > 0:
            formant_f1[idx] = formants[0]
        if formants.size > 1:
            formant_f2[idx] = formants[1]
        if formants.size > 2:
            formant_f3[idx] = formants[2]

    flux_norm = np.asarray(
        _safe_div(flux, np.percentile(flux, 95.0) + EPS), dtype=np.float64
    )
    flux_norm = np.clip(flux_norm, 0.0, 1.0)
    onset_norm = np.asarray(
        _safe_div(onset_strength, np.percentile(onset_strength, 95.0) + EPS),
        dtype=np.float64,
    )
    onset_norm = np.clip(onset_norm, 0.0, 1.0)
    rms_norm = np.asarray(_safe_div(rms, np.max(rms) + EPS), dtype=np.float64)
    centroid_norm = np.asarray(_safe_div(centroid, (0.5 * float(sr))), dtype=np.float64)
    centroid_norm = np.clip(centroid_norm, 0.0, 1.0)
    rolloff_norm = np.asarray(_safe_div(rolloff, (0.5 * float(sr))), dtype=np.float64)
    rolloff_norm = np.clip(rolloff_norm, 0.0, 1.0)

    voicing_prob = np.clip(
        np.maximum(np.asarray(confidence, dtype=np.float64), harmonic_ratio), 0.0, 1.0
    )
    diff = np.diff(np.asarray(f0_hz, dtype=np.float64), prepend=float(f0_hz[0]))
    cents_jump = 1200.0 * np.log2(
        np.maximum(np.asarray(f0_hz, dtype=np.float64), EPS)
        / np.maximum(np.asarray(f0_hz, dtype=np.float64) - diff, EPS)
    )
    cents_jump = np.nan_to_num(cents_jump, nan=0.0, posinf=0.0, neginf=0.0)
    pitch_stability = np.exp(-np.abs(cents_jump) / 50.0)
    note_boundary = ((np.abs(cents_jump) >= 80.0) | (onset_norm > 0.6)).astype(
        np.float64
    )
    transient_mask = (onset_norm > 0.55).astype(np.float64)
    transientness = np.clip(0.5 * onset_norm + 0.5 * flux_norm, 0.0, 1.0)

    # Very lightweight content classifier probabilities.
    silence_prob = np.clip((-rms_db - 45.0) / 30.0, 0.0, 1.0)
    speech_score = np.clip(
        voicing_prob * (1.0 - flatness) * (1.0 - hiss_ratio), 0.0, 1.0
    )
    music_score = np.clip(
        (1.0 - silence_prob)
        * (0.45 * (1.0 - flatness) + 0.35 * centroid_norm + 0.2 * (1.0 - zcr)),
        0.0,
        1.0,
    )
    noise_score = np.clip(
        (0.5 * flatness + 0.5 * hiss_ratio) * (1.0 - silence_prob), 0.0, 1.0
    )
    score_stack = np.stack(
        [silence_prob, speech_score, music_score, noise_score], axis=1
    )
    score_sum = np.sum(score_stack, axis=1, keepdims=True)
    probs = np.asarray(_safe_div(score_stack, score_sum + EPS), dtype=np.float64)
    silence_prob = probs[:, 0]
    speech_prob = probs[:, 1]
    music_prob = probs[:, 2]
    noise_prob = probs[:, 3]

    tempo_bpm = _estimate_tempo_bpm(onset_strength, hop_size=hop_size, sr=sr)
    times_sec = (np.arange(n_frames, dtype=np.float64) * float(hop_size)) / float(sr)
    beats = (times_sec * tempo_bpm) / 60.0
    beat_phase = beats - np.floor(beats)
    downbeat_phase = (beats / 4.0) - np.floor(beats / 4.0)

    temporal_centroid = float(np.sum(times_sec * (rms + EPS)) / np.sum(rms + EPS))
    attack_idx = int(np.argmax(onset_strength))
    attack_time_s = float(times_sec[attack_idx]) if attack_idx < times_sec.size else 0.0
    log_attack_time_s = float(math.log10(max(attack_time_s, 1e-5)))

    features: dict[str, np.ndarray] = {
        "rms": rms,
        "rms_db": rms_db,
        "zcr": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "spectral_flatness": flatness,
        "spectral_flux": flux,
        "onset_strength": onset_strength,
        "rolloff_hz": rolloff,
        "voicing_prob": voicing_prob,
        "pitch_stability": pitch_stability,
        "harmonic_ratio": harmonic_ratio,
        "inharmonicity": inharmonicity,
        "formant_f1_hz": formant_f1,
        "formant_f2_hz": formant_f2,
        "formant_f3_hz": formant_f3,
        "clip_ratio": clip_ratio,
        "crest_factor_db": crest_db,
        "transient_mask": transient_mask,
        "note_boundary": note_boundary,
        "transientness": transientness,
        "short_lufs_db": rms_db,  # lightweight proxy in frame domain
        "rms_norm": rms_norm,
        "centroid_norm": centroid_norm,
        "flux_norm": flux_norm,
        "onset_norm": onset_norm,
        "rolloff_norm": rolloff_norm,
        "tempo_bpm": np.full((n_frames,), tempo_bpm, dtype=np.float64),
        "beat_phase": beat_phase,
        "downbeat_phase": downbeat_phase,
        "silence_prob": silence_prob,
        "speech_prob": speech_prob,
        "music_prob": music_prob,
        "noise_prob": noise_prob,
        "ild_db": ild_db,
        "itd_ms": itd_ms,
        "hiss_ratio": hiss_ratio,
        "hum_50_ratio": hum_50,
        "hum_60_ratio": hum_60,
        # MPEG-7-style descriptors.
        "mpeg7_audio_power_db": rms_db,
        "mpeg7_spectral_centroid_hz": centroid,
        "mpeg7_spectral_spread_hz": spread,
        "mpeg7_spectral_flatness": flatness,
        "mpeg7_spectral_flux": flux,
        "mpeg7_spectral_rolloff_hz": rolloff,
        "mpeg7_zero_crossing_rate": zcr,
        "mpeg7_spectral_crest": spectral_crest,
        "mpeg7_spectral_decrease": spectral_decrease,
        "mpeg7_temporal_centroid_s": np.full(
            (n_frames,), temporal_centroid, dtype=np.float64
        ),
        "mpeg7_log_attack_time_s": np.full(
            (n_frames,), log_attack_time_s, dtype=np.float64
        ),
    }

    for b in range(ase.shape[1]):
        features[f"mpeg7_audio_spectrum_envelope_{b + 1:02d}"] = ase[:, b]
    for m in range(mfcc.shape[1]):
        features[f"mfcc_{m + 1:02d}"] = mfcc[:, m]
    return features


def feature_subset(
    features: dict[str, np.ndarray],
    *,
    subset: str,
) -> dict[str, np.ndarray]:
    """Select feature view: none/basic/advanced/all."""
    mode = str(subset).strip().lower()
    if mode == "none":
        return {}
    if mode == "all":
        return dict(features)

    basic_keys = (
        "rms",
        "rms_db",
        "zcr",
        "spectral_centroid_hz",
        "spectral_flatness",
        "spectral_flux",
        "onset_strength",
        "rolloff_hz",
        "voicing_prob",
        "pitch_stability",
        "harmonic_ratio",
        "clip_ratio",
    )
    if mode == "basic":
        return {k: features[k] for k in basic_keys if k in features}

    if mode == "advanced":
        out = dict(features)
        # Keep advanced computational descriptors and control-friendly normalized tracks.
        return out

    return dict(features)


def as_serializable_columns(
    features: dict[str, np.ndarray],
    *,
    n_rows: int,
) -> dict[str, np.ndarray]:
    """Normalize feature columns for CSV emission."""
    out: dict[str, np.ndarray] = {}
    for key, value in features.items():
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            arr = np.zeros((n_rows,), dtype=np.float64)
        if arr.size < n_rows:
            pad = np.full(
                (n_rows - arr.size,), arr[-1] if arr.size else 0.0, dtype=np.float64
            )
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.size > n_rows:
            arr = arr[:n_rows]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = arr
    return out
