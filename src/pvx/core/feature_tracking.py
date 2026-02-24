#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Frame-level feature tracking for control-rate audio modulation maps."""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def _safe_div(numer: np.ndarray | float, denom: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(numer) / np.maximum(np.asarray(denom), EPS)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
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


def _dct_type2(x: np.ndarray, n_coeffs: int) -> np.ndarray:
    n = int(x.size)
    if n == 0:
        return np.zeros((n_coeffs,), dtype=np.float64)
    k = np.arange(n_coeffs, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    basis = np.cos(np.pi * (i + 0.5) * k / float(n))
    scale = np.sqrt(2.0 / float(n))
    out = scale * (basis @ x)
    if n_coeffs > 0:
        out[0] *= 1.0 / math.sqrt(2.0)
    return np.asarray(out, dtype=np.float64)


def _frame(audio: np.ndarray, start: int, length: int) -> np.ndarray:
    end = start + length
    if end <= audio.size:
        return np.asarray(audio[start:end], dtype=np.float64)
    out = np.zeros((length,), dtype=np.float64)
    if start < audio.size:
        avail = audio.size - start
        out[:avail] = np.asarray(audio[start:], dtype=np.float64)
    return out


def _acf_peak_ratio(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> tuple[float, float]:
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


def _estimate_formants_lpc(frame: np.ndarray, sr: int, count: int = 3) -> tuple[float, float, float]:
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
    partials = []
    for k in range(1, max_partials + 1):
        target = k * f0_hz
        if target >= nyq:
            break
        partials.append(target)
    if not partials:
        return 0.0

    m = np.asarray(mag, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    if m.size < 3:
        return 0.0

    # Vectorized peak picking
    # Find local maxima: m[i] > m[i-1] AND m[i] >= m[i+1]
    is_peak = (m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:])
    indices = np.where(is_peak)[0] + 1

    if indices.size == 0:
        return 0.0

    pf = np.asarray(freqs[indices], dtype=np.float64)
    pm = np.asarray(m[indices], dtype=np.float64)

    deviations = []
    weights = []
    for target in partials:
        idx = int(np.argmin(np.abs(pf - target)))
        dev = abs(pf[idx] - target) / max(target, EPS)
        deviations.append(dev)
        weights.append(pm[idx])
    w = np.asarray(weights, dtype=np.float64)
    if float(np.sum(w)) <= EPS:
        return float(np.mean(deviations))
    return float(np.sum(np.asarray(deviations, dtype=np.float64) * w) / np.sum(w))


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

    rms = np.zeros((n_frames,), dtype=np.float64)
    rms_db = np.zeros((n_frames,), dtype=np.float64)
    zcr = np.zeros((n_frames,), dtype=np.float64)
    clip_ratio = np.zeros((n_frames,), dtype=np.float64)
    crest_db = np.zeros((n_frames,), dtype=np.float64)
    centroid = np.zeros((n_frames,), dtype=np.float64)
    spread = np.zeros((n_frames,), dtype=np.float64)
    flatness = np.zeros((n_frames,), dtype=np.float64)
    flux = np.zeros((n_frames,), dtype=np.float64)
    rolloff = np.zeros((n_frames,), dtype=np.float64)
    onset_strength = np.zeros((n_frames,), dtype=np.float64)
    harmonic_ratio = np.zeros((n_frames,), dtype=np.float64)
    formant_f1 = np.full((n_frames,), np.nan, dtype=np.float64)
    formant_f2 = np.full((n_frames,), np.nan, dtype=np.float64)
    formant_f3 = np.full((n_frames,), np.nan, dtype=np.float64)
    inharmonicity = np.zeros((n_frames,), dtype=np.float64)
    ild_db = np.zeros((n_frames,), dtype=np.float64)
    itd_ms = np.zeros((n_frames,), dtype=np.float64)
    hiss_ratio = np.zeros((n_frames,), dtype=np.float64)
    hum_50 = np.zeros((n_frames,), dtype=np.float64)
    hum_60 = np.zeros((n_frames,), dtype=np.float64)
    spectral_crest = np.zeros((n_frames,), dtype=np.float64)
    spectral_decrease = np.zeros((n_frames,), dtype=np.float64)
    ase = np.zeros((n_frames, 10), dtype=np.float64)
    mfcc = np.zeros((n_frames, mfcc_count), dtype=np.float64)

    prev_mag = np.zeros(((n_fft // 2) + 1,), dtype=np.float64)
    itd_max_lag = int(round((1.0e-3) * float(sr)))  # ~1 ms

    for idx in range(n_frames):
        start = idx * int(hop_size)
        fr = _frame(mono, start, n_fft)
        fr_l = _frame(left, start, n_fft)
        fr_r = _frame(right, start, n_fft)

        rms_i = float(np.sqrt(np.mean(fr * fr)))
        peak_i = float(np.max(np.abs(fr)))
        rms[idx] = rms_i
        rms_db[idx] = 20.0 * math.log10(max(rms_i, EPS))
        clip_ratio[idx] = float(np.mean(np.abs(fr) >= 0.999))
        crest_db[idx] = 20.0 * math.log10(max(peak_i, EPS) / max(rms_i, EPS))
        zc = np.abs(np.diff(np.signbit(fr))).astype(np.float64)
        zcr[idx] = float(np.mean(zc)) if zc.size > 0 else 0.0

        spec = np.fft.rfft(fr * window)
        mag = np.abs(spec).astype(np.float64)
        pwr = mag * mag
        total_mag = float(np.sum(mag))
        total_pwr = float(np.sum(pwr))
        centroid[idx] = float(np.sum(freqs * mag) / max(total_mag, EPS))
        centered = freqs - centroid[idx]
        spread[idx] = float(np.sqrt(np.sum((centered * centered) * mag) / max(total_mag, EPS)))
        flatness[idx] = float(np.exp(np.mean(np.log(pwr + EPS))) / max(np.mean(pwr), EPS))
        flux[idx] = float(np.sum(np.maximum(mag - prev_mag, 0.0)))
        onset_strength[idx] = flux[idx]
        prev_mag = mag

        csum = np.cumsum(pwr)
        threshold = 0.95 * csum[-1] if csum.size else 0.0
        ridx = int(np.searchsorted(csum, threshold)) if csum.size else 0
        ridx = int(np.clip(ridx, 0, freqs.size - 1))
        rolloff[idx] = float(freqs[ridx])

        _, acf_conf = _acf_peak_ratio(fr, sr, fmin=fmin, fmax=fmax)
        harmonic_ratio[idx] = acf_conf

        f1, f2, f3 = _estimate_formants_lpc(fr, sr, count=3)
        formant_f1[idx], formant_f2[idx], formant_f3[idx] = f1, f2, f3

        inharmonicity[idx] = _estimate_inharmonicity(mag, freqs, float(f0_hz[idx]))

        # Stereo cues.
        lrms = float(np.sqrt(np.mean(fr_l * fr_l)))
        rrms = float(np.sqrt(np.mean(fr_r * fr_r)))
        ild_db[idx] = 20.0 * math.log10(max(lrms, EPS) / max(rrms, EPS))
        corr = np.correlate(fr_l, fr_r, mode="full")
        mid = fr_l.size - 1
        lo = max(0, mid - itd_max_lag)
        hi = min(corr.size, mid + itd_max_lag + 1)
        lag = int(np.argmax(corr[lo:hi])) + lo - mid
        itd_ms[idx] = 1000.0 * float(lag) / float(sr)

        # Noise markers.
        hi_band = freqs >= 6000.0
        hiss_ratio[idx] = float(np.sum(pwr[hi_band]) / max(total_pwr, EPS))
        def _hum_ratio(base_hz: float) -> float:
            accum = 0.0
            for h in range(1, 6):
                target = h * base_hz
                if target >= freqs[-1]:
                    break
                b = int(np.argmin(np.abs(freqs - target)))
                lo_b = max(0, b - 1)
                hi_b = min(pwr.size, b + 2)
                accum += float(np.sum(pwr[lo_b:hi_b]))
            return float(accum / max(total_pwr, EPS))
        hum_50[idx] = _hum_ratio(50.0)
        hum_60[idx] = _hum_ratio(60.0)

        spectral_crest[idx] = float(np.max(mag) / max(np.mean(mag), EPS))
        if mag.size > 1:
            ref = mag[1:]
            n = np.arange(2, mag.size + 1, dtype=np.float64)
            spectral_decrease[idx] = float(np.sum((ref - ref[0]) / n) / max(np.sum(ref), EPS))
        else:
            spectral_decrease[idx] = 0.0

        # MPEG-7-like Audio Spectrum Envelope (10 coarse subbands, normalized log-power).
        edges = np.linspace(0, pwr.size, num=11, dtype=np.int64)
        sub = []
        for b in range(10):
            sl = pwr[edges[b] : edges[b + 1]]
            sub.append(math.log10(max(float(np.sum(sl)), EPS)))
        sub_arr = np.asarray(sub, dtype=np.float64)
        sub_arr -= float(np.mean(sub_arr))
        ase[idx, :] = sub_arr

        if mfcc_count > 0:
            mel_ener = mel_fb @ pwr
            mel_log = np.log(mel_ener + EPS)
            mfcc[idx, :] = _dct_type2(mel_log, mfcc_count)

    flux_norm = np.asarray(_safe_div(flux, np.percentile(flux, 95.0) + EPS), dtype=np.float64)
    flux_norm = np.clip(flux_norm, 0.0, 1.0)
    onset_norm = np.asarray(_safe_div(onset_strength, np.percentile(onset_strength, 95.0) + EPS), dtype=np.float64)
    onset_norm = np.clip(onset_norm, 0.0, 1.0)
    rms_norm = np.asarray(_safe_div(rms, np.max(rms) + EPS), dtype=np.float64)
    centroid_norm = np.asarray(_safe_div(centroid, (0.5 * float(sr))), dtype=np.float64)
    centroid_norm = np.clip(centroid_norm, 0.0, 1.0)
    rolloff_norm = np.asarray(_safe_div(rolloff, (0.5 * float(sr))), dtype=np.float64)
    rolloff_norm = np.clip(rolloff_norm, 0.0, 1.0)

    voicing_prob = np.clip(np.maximum(np.asarray(confidence, dtype=np.float64), harmonic_ratio), 0.0, 1.0)
    diff = np.diff(np.asarray(f0_hz, dtype=np.float64), prepend=float(f0_hz[0]))
    cents_jump = 1200.0 * np.log2(np.maximum(np.asarray(f0_hz, dtype=np.float64), EPS) / np.maximum(np.asarray(f0_hz, dtype=np.float64) - diff, EPS))
    cents_jump = np.nan_to_num(cents_jump, nan=0.0, posinf=0.0, neginf=0.0)
    pitch_stability = np.exp(-np.abs(cents_jump) / 50.0)
    note_boundary = ((np.abs(cents_jump) >= 80.0) | (onset_norm > 0.6)).astype(np.float64)
    transient_mask = (onset_norm > 0.55).astype(np.float64)
    transientness = np.clip(0.5 * onset_norm + 0.5 * flux_norm, 0.0, 1.0)

    # Very lightweight content classifier probabilities.
    silence_prob = np.clip((-rms_db - 45.0) / 30.0, 0.0, 1.0)
    speech_score = np.clip(voicing_prob * (1.0 - flatness) * (1.0 - hiss_ratio), 0.0, 1.0)
    music_score = np.clip((1.0 - silence_prob) * (0.45 * (1.0 - flatness) + 0.35 * centroid_norm + 0.2 * (1.0 - zcr)), 0.0, 1.0)
    noise_score = np.clip((0.5 * flatness + 0.5 * hiss_ratio) * (1.0 - silence_prob), 0.0, 1.0)
    score_stack = np.stack([silence_prob, speech_score, music_score, noise_score], axis=1)
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
        "mpeg7_temporal_centroid_s": np.full((n_frames,), temporal_centroid, dtype=np.float64),
        "mpeg7_log_attack_time_s": np.full((n_frames,), log_attack_time_s, dtype=np.float64),
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
            pad = np.full((n_rows - arr.size,), arr[-1] if arr.size else 0.0, dtype=np.float64)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.size > n_rows:
            arr = arr[:n_rows]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = arr
    return out
