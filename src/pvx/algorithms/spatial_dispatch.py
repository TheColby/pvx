#!/usr/bin/env python3

"""Spatial algorithm dispatch helpers for pvx algorithm wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, signal

from pvx.algorithms.base import (
    ensure_length,
    granular_time_stretch,
    istft_multi,
    normalize_peak,
    pitch_shift,
    stft_multi,
)


def _spatial_to_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    channels = max(1, int(channels))
    if audio.shape[1] == channels:
        return audio.copy()
    mono = np.mean(audio, axis=1, keepdims=True)
    if channels == 1:
        return mono
    return np.repeat(mono, channels, axis=1)


def _spatial_fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    idx = np.arange(x.size, dtype=np.float64) - float(delay_samples)
    lo = np.floor(idx).astype(int)
    hi = lo + 1
    frac = idx - lo
    lo = np.clip(lo, 0, x.size - 1)
    hi = np.clip(hi, 0, x.size - 1)
    return (1.0 - frac) * x[lo] + frac * x[hi]


def _spatial_apply_delays(audio: np.ndarray, delays: list[float]) -> np.ndarray:
    out = np.zeros_like(audio)
    for ch in range(audio.shape[1]):
        delay = float(delays[ch]) if ch < len(delays) else 0.0
        out[:, ch] = _spatial_fractional_delay(audio[:, ch], delay)
    return out


def _spatial_circular_gains(
    num_channels: int, azimuth_deg: float, width: float = 1.0, rolloff: float = 2.0
) -> np.ndarray:
    num_channels = max(1, int(num_channels))
    angles = np.linspace(-180.0, 180.0, num=num_channels, endpoint=False)
    delta = np.abs(((angles - azimuth_deg + 180.0) % 360.0) - 180.0)
    spread = max(4.0, 45.0 * max(0.2, width))
    gains = 1.0 / (1.0 + np.power(delta / spread, max(1.0, rolloff)))
    gains /= np.sqrt(np.sum(gains * gains) + 1e-12)
    return gains


def _spatial_delay_by_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    n = 1
    target = max(2, x.size + y.size)
    while n < target:
        n <<= 1
    x_fft = np.fft.rfft(x, n=n)
    y_fft = np.fft.rfft(y, n=n)
    response = x_fft * np.conj(y_fft)
    response /= np.abs(response) + 1e-12
    cc = np.fft.irfft(response, n=n)
    max_lag = int(max(1, max_lag))
    cc = np.concatenate((cc[-max_lag:], cc[: max_lag + 1]))
    lag = int(np.argmax(np.abs(cc)) - max_lag)
    return float(lag)


def _spatial_estimate_channel_delays(audio: np.ndarray, max_lag: int = 128) -> list[float]:
    delays = [0.0]
    if audio.shape[1] < 2:
        return delays
    ref = audio[:, 0]
    for ch in range(1, audio.shape[1]):
        delays.append(_spatial_delay_by_xcorr(ref, audio[:, ch], max_lag=max_lag))
    return delays


def _spatial_synthetic_rir(
    length: int, decay_s: float, sr: int, seed: int, channel_index: int
) -> np.ndarray:
    length = max(16, int(length))
    rng = np.random.default_rng(int(seed) + int(channel_index) * 37)
    t = np.arange(length, dtype=np.float64) / float(sr)
    decay = np.exp(-t / max(1e-3, float(decay_s)))
    rir = 0.12 * decay * rng.standard_normal(length)
    rir[0] += 0.7
    early = int(round((0.003 + 0.0015 * channel_index) * sr))
    if 0 <= early < length:
        rir[early] += 1.0
    return rir


def dispatch_spatial(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    work = audio

    if slug == "vbap_adaptive_panning":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        azimuth_deg = float(params.get("azimuth_deg", 0.0))
        width = float(params.get("width", 1.0))
        mono = np.mean(work, axis=1)
        gains = _spatial_circular_gains(output_channels, azimuth_deg, width=width)
        out = mono[:, None] * gains[None, :]
        notes.append("Rendered source with VBAP-style adaptive gain distribution.")
        extras["speaker_gains"] = gains.tolist()

    elif slug == "dbap_distance_based_amplitude_panning":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        source_x = float(params.get("source_x", 0.25))
        source_y = float(params.get("source_y", 0.0))
        rolloff = float(params.get("rolloff", 1.8))
        mono = np.mean(work, axis=1)
        angles = np.linspace(0.0, 2.0 * np.pi, num=output_channels, endpoint=False)
        speakers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        src = np.array([source_x, source_y], dtype=np.float64)
        dist = np.linalg.norm(speakers - src[None, :], axis=1) + 1e-3
        gains = 1.0 / np.power(dist, max(0.5, rolloff))
        gains /= np.sqrt(np.sum(gains * gains) + 1e-12)
        out = mono[:, None] * gains[None, :]
        notes.append("Applied DBAP weighting from virtual source position.")
        extras["speaker_gains"] = gains.tolist()

    elif slug == "binaural_itd_ild_synthesis":
        azimuth_deg = float(params.get("azimuth_deg", 30.0))
        itd_max_ms = float(params.get("itd_max_ms", 0.7))
        ild_db = float(params.get("ild_db", 8.0))
        mono = np.mean(work, axis=1)
        az = np.deg2rad(azimuth_deg)
        itd_samples = np.sin(az) * itd_max_ms * 1e-3 * sr
        ild = ild_db * np.sin(az)
        g_l = 10.0 ** ((-0.5 * ild) / 20.0)
        g_r = 10.0 ** ((0.5 * ild) / 20.0)
        left_delay = max(0.0, itd_samples)
        right_delay = max(0.0, -itd_samples)
        left = _spatial_fractional_delay(mono, left_delay) * g_l
        right = _spatial_fractional_delay(mono, right_delay) * g_r
        out = np.stack([left, right], axis=1)
        notes.append("Synthesized binaural cues from ITD/ILD model.")

    elif slug == "transaural_crosstalk_cancellation":
        cancellation = float(params.get("cancellation", 0.6))
        delay_ms = float(params.get("delay_ms", 0.22))
        stereo = _spatial_to_channels(work, 2)
        delay_samples = delay_ms * 1e-3 * sr
        left = stereo[:, 0] - cancellation * _spatial_fractional_delay(stereo[:, 1], delay_samples)
        right = stereo[:, 1] - cancellation * _spatial_fractional_delay(stereo[:, 0], delay_samples)
        out = np.stack([left, right], axis=1)
        notes.append("Applied transaural crosstalk cancellation matrix with delayed crossfeed.")

    elif slug == "stereo_width_frequency_dependent_control":
        width_low = float(params.get("width_low", 0.8))
        width_high = float(params.get("width_high", 1.35))
        crossover_hz = float(params.get("crossover_hz", 1400.0))
        stereo = _spatial_to_channels(work, 2)
        mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
        side = 0.5 * (stereo[:, 0] - stereo[:, 1])
        b, a = signal.butter(2, min(0.98, max(0.001, crossover_hz / (0.5 * sr))), btype="low")
        side_low = signal.lfilter(b, a, side)
        side_high = side - side_low
        side2 = width_low * side_low + width_high * side_high
        out = np.stack([mid + side2, mid - side2], axis=1)
        notes.append("Applied frequency-dependent stereo width control in mid/side domain.")

    elif slug == "phase_aligned_mid_side_field_rotation":
        rotation_deg = float(params.get("rotation_deg", 20.0))
        stereo = _spatial_to_channels(work, 2)
        mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
        side = 0.5 * (stereo[:, 0] - stereo[:, 1])
        theta = np.deg2rad(rotation_deg)
        m2 = np.cos(theta) * mid - np.sin(theta) * side
        s2 = np.sin(theta) * mid + np.cos(theta) * side
        out = np.stack([m2 + s2, m2 - s2], axis=1)
        notes.append("Rotated sound field in phase-aligned mid/side space.")

    elif slug == "pvx_interchannel_phase_locking":
        lock_strength = float(params.get("lock_strength", 0.7))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        ref = pha[:, :, 0]
        for ch in range(1, pha.shape[2]):
            pha[:, :, ch] = (1.0 - lock_strength) * pha[:, :, ch] + lock_strength * ref
        out = istft_multi(mag * np.exp(1j * pha), n_fft=2048, hop=512, length=work.shape[0])
        notes.append(
            "Locked interchannel phase to a reference channel in the phase-vocoder domain."
        )

    elif slug == "pvx_spatial_transient_preservation":
        transient_threshold = float(params.get("transient_threshold", 1.6))
        phase_smooth = float(params.get("phase_smooth", 0.85))
        preserve_amount = float(params.get("preserve_amount", 0.8))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        energy = np.mean(mag, axis=(0, 2))
        diff = np.diff(energy, prepend=energy[0])
        threshold = transient_threshold * (np.mean(np.abs(diff)) + 1e-12)
        transient = diff > threshold
        pha2 = pha.copy()
        for ch in range(pha.shape[2]):
            for t in range(1, pha.shape[1]):
                if transient[t]:
                    pha2[:, t, ch] = (
                        preserve_amount * pha[:, t, ch]
                        + (1.0 - preserve_amount) * pha2[:, t - 1, ch]
                    )
                else:
                    pha2[:, t, ch] = (
                        phase_smooth * pha2[:, t - 1, ch] + (1.0 - phase_smooth) * pha[:, t, ch]
                    )
        out = istft_multi(mag * np.exp(1j * pha2), n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Preserved transients while smoothing inter-frame spatial phase trajectories.")

    elif slug == "pvx_interaural_coherence_shaping":
        coherence_target = float(params.get("coherence_target", 0.75))
        stereo = _spatial_to_channels(work, 2)
        spec, _, _ = stft_multi(stereo, n_fft=2048, hop=512)
        left = spec[:, :, 0]
        right = spec[:, :, 1]
        mid = (left + right) / np.sqrt(2.0)
        side = (left - right) / np.sqrt(2.0)
        rng = np.random.default_rng(1307)
        rand_phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=side.shape))
        side2 = coherence_target * side + (1.0 - coherence_target) * np.abs(side) * rand_phase
        left2 = (mid + side2) / np.sqrt(2.0)
        right2 = (mid - side2) / np.sqrt(2.0)
        out = istft_multi(
            np.stack([left2, right2], axis=2), n_fft=2048, hop=512, length=work.shape[0]
        )
        notes.append("Shaped interaural coherence by controlled side-channel decorrelation.")

    elif slug == "pvx_directional_spectral_warp":
        warp_amount = float(params.get("warp_amount", 0.16))
        azimuth_deg = float(params.get("azimuth_deg", 30.0))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        bins = np.arange(mag.shape[0], dtype=np.float64)
        pos = np.linspace(-1.0, 1.0, num=mag.shape[2])
        azimuth = np.deg2rad(azimuth_deg)
        mag2 = np.zeros_like(mag)
        for ch in range(mag.shape[2]):
            shift = warp_amount * pos[ch] * (bins / max(1.0, bins[-1])) * mag.shape[0] * 0.35
            src = np.clip(bins - shift, 0.0, bins[-1])
            for t in range(mag.shape[1]):
                mag2[:, t, ch] = np.interp(src, bins, mag[:, t, ch])
            pha[:, :, ch] = pha[:, :, ch] + azimuth * pos[ch] * (bins[:, None] / max(1.0, bins[-1]))
        out = istft_multi(mag2 * np.exp(1j * pha), n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied directional spectral warp with channel-dependent phase skew.")

    elif slug == "pvx_multichannel_time_alignment":
        max_lag = int(params.get("max_lag", max(8, int(0.002 * sr))))
        if work.shape[1] < 2:
            out = work.copy()
            delays = [0.0]
        else:
            delays = _spatial_estimate_channel_delays(work, max_lag=max_lag)
            out = _spatial_apply_delays(work, delays)
        extras["estimated_delays_samples"] = delays
        notes.append(
            "Aligned channels by phase-weighted cross-correlation delay estimation and fractional delay compensation."
        )

    elif slug == "pvx_spatial_freeze_and_trajectory":
        frame_ratio = float(params.get("frame_ratio", 0.35))
        orbit_hz = float(params.get("orbit_hz", 0.12))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        idx = int(np.clip(round(frame_ratio * (spec.shape[1] - 1)), 0, max(0, spec.shape[1] - 1)))
        frozen = spec[:, idx, :]
        mag = np.abs(frozen)
        base_phase = np.angle(frozen)
        out_spec = np.zeros_like(spec)
        frame_rate = sr / 512.0
        for t in range(spec.shape[1]):
            theta = 2.0 * np.pi * orbit_hz * (t / max(1.0, frame_rate))
            for ch in range(spec.shape[2]):
                phase = base_phase[:, ch] + theta * (1.0 + 0.15 * ch)
                out_spec[:, t, ch] = mag[:, ch] * np.exp(1j * phase)
        out = istft_multi(out_spec, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Froze spatial spectrum and animated channel trajectories with phase orbits.")

    elif slug == "multichannel_wiener_postfilter":
        noise_floor = float(params.get("noise_floor", 0.15))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        noise = np.percentile(mag, 20, axis=1, keepdims=True)
        gain = mag * mag / (mag * mag + np.power(noise * (1.0 + noise_floor), 2.0) + 1e-12)
        out = istft_multi(spec * gain, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied multichannel Wiener postfilter using percentile noise estimate.")

    elif slug == "coherence_based_dereverb_multichannel":
        coherence_threshold = float(params.get("coherence_threshold", 0.5))
        decay = float(params.get("decay", 0.92))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        ref = spec[:, :, 0]
        coh = np.ones((spec.shape[0], spec.shape[1]), dtype=np.float64)
        for ch in range(1, spec.shape[2]):
            coh += np.abs(ref * np.conj(spec[:, :, ch])) / (
                np.abs(ref) * np.abs(spec[:, :, ch]) + 1e-12
            )
        coh /= float(spec.shape[2])
        tail = np.zeros((mag.shape[0], mag.shape[2]), dtype=np.float64)
        mask = np.zeros_like(mag)
        for t in range(mag.shape[1]):
            tail = np.maximum(decay * tail, mag[:, t, :])
            base = np.clip(
                (coh[:, t] - coherence_threshold) / (1.0 - coherence_threshold + 1e-9), 0.15, 1.0
            )
            mask[:, t, :] = base[:, None] * np.clip(
                mag[:, t, :] / (mag[:, t, :] + 0.5 * tail + 1e-9), 0.1, 1.0
            )
        out = istft_multi(spec * mask, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied coherence-guided late-reverb suppression for multichannel material.")

    elif slug == "multichannel_noise_psd_tracking":
        alpha = float(params.get("alpha", 1.0))
        floor = float(params.get("floor", 0.08))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        power = np.abs(spec) ** 2
        noise = np.minimum.accumulate(power, axis=1)
        noise = ndimage.minimum_filter1d(noise, size=11, axis=1)
        gain = np.clip((power - alpha * noise) / (power + 1e-12), floor, 1.0)
        out = istft_multi(spec * np.sqrt(gain), n_fft=2048, hop=512, length=work.shape[0])
        extras["mean_noise_power"] = float(np.mean(noise))
        notes.append("Tracked multichannel noise PSD and applied adaptive subtraction mask.")

    elif slug == "phase_consistent_multichannel_denoise":
        reduction_db = float(params.get("reduction_db", 10.0))
        floor = float(params.get("floor", 0.1))
        spec, _, _ = stft_multi(work, n_fft=2048, hop=512)
        mag = np.abs(spec)
        shared = np.mean(mag, axis=2, keepdims=True)
        noise = np.percentile(shared, 15, axis=1, keepdims=True)
        gain = np.clip(
            (shared - noise * (10.0 ** (reduction_db / 20.0))) / (shared + 1e-12), floor, 1.0
        )
        out = istft_multi(spec * gain, n_fft=2048, hop=512, length=work.shape[0])
        notes.append("Applied phase-consistent denoise mask shared across channels.")

    elif slug == "microphone_array_calibration_tones":
        tone_hz = params.get("tone_hz", [500.0, 1000.0, 2000.0])
        if isinstance(tone_hz, (int, float)):
            tones = [float(tone_hz)]
        else:
            tones = [float(v) for v in tone_hz] if tone_hz else [1000.0]
        apply_correction = bool(params.get("apply_correction", True))
        n = work.shape[0]
        win = np.hanning(n)[:, None]
        x_fft = np.fft.rfft(work * win, axis=0)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        gains = np.ones(work.shape[1], dtype=np.float64)
        delays = np.zeros(work.shape[1], dtype=np.float64)
        for ch in range(1, work.shape[1]):
            gain_est: list[float] = []
            delay_est: list[float] = []
            for tone in tones:
                idx = int(np.argmin(np.abs(freqs - tone)))
                freq_hz = max(1e-6, freqs[idx])
                ref_mag = np.abs(x_fft[idx, 0]) + 1e-12
                ch_mag = np.abs(x_fft[idx, ch]) + 1e-12
                gain_est.append(float(ref_mag / ch_mag))
                phase = np.angle(x_fft[idx, ch]) - np.angle(x_fft[idx, 0])
                delay_est.append(float((-phase / (2.0 * np.pi * freq_hz)) * sr))
            gains[ch] = float(np.median(gain_est)) if gain_est else 1.0
            delays[ch] = float(np.median(delay_est)) if delay_est else 0.0
        if apply_correction:
            out = _spatial_apply_delays(work, delays.tolist())
            out = out * gains[None, :]
        else:
            out = work.copy()
        extras["estimated_gain_db"] = (20.0 * np.log10(gains + 1e-12)).tolist()
        extras["estimated_delay_samples"] = delays.tolist()
        notes.append("Estimated microphone gain/delay mismatches from calibration tones.")

    elif slug == "cross_channel_click_pop_repair":
        spike_threshold = float(params.get("spike_threshold", 6.0))
        out = work.copy()
        for ch in range(out.shape[1]):
            x = out[:, ch]
            dx = np.abs(np.diff(x, prepend=x[0]))
            med = np.median(dx) + 1e-12
            bad = np.where(dx > spike_threshold * med)[0]
            for idx in bad:
                lo = max(0, idx - 1)
                hi = min(out.shape[0], idx + 2)
                out[idx, ch] = float(np.median(out[lo:hi, :]))
            out[:, ch] = signal.medfilt(out[:, ch], kernel_size=3)
        notes.append("Repaired click/pop outliers using cross-channel robust interpolation.")

    elif slug == "rotating_speaker_doppler_field":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        rotation_hz = float(params.get("rotation_hz", 0.25))
        depth_ms = float(params.get("depth_ms", 1.2))
        mono = np.mean(work, axis=1)
        t = np.arange(work.shape[0], dtype=np.float64) / float(sr)
        depth = depth_ms * 1e-3 * sr
        idx = np.arange(work.shape[0], dtype=np.float64)
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        for ch in range(output_channels):
            phase = 2.0 * np.pi * (rotation_hz * t + ch / max(1, output_channels))
            delay = depth * np.sin(phase)
            src = idx - delay
            wave = np.interp(src, idx, mono, left=mono[0], right=mono[-1])
            gain = 0.65 + 0.35 * np.cos(phase)
            out[:, ch] = wave * gain
        notes.append("Simulated rotating-speaker Doppler field over multichannel ring.")

    elif slug == "binaural_motion_trajectory_designer":
        trajectory = str(params.get("trajectory", "sine")).lower()
        trajectory_hz = float(params.get("trajectory_hz", 0.15))
        width = float(params.get("width", 1.0))
        itd_ms = float(params.get("itd_ms", 0.6))
        mono = np.mean(work, axis=1)
        t = np.arange(work.shape[0], dtype=np.float64) / float(sr)
        if trajectory == "saw":
            phase = t * trajectory_hz
            az = 90.0 * width * (2.0 * (phase - np.floor(phase + 0.5)))
        elif trajectory == "triangle":
            phase = (t * trajectory_hz) % 1.0
            az = 90.0 * width * (2.0 * np.abs(2.0 * phase - 1.0) - 1.0)
        else:
            az = 90.0 * width * np.sin(2.0 * np.pi * trajectory_hz * t)
        pan = np.clip((az + 90.0) / 180.0, 0.0, 1.0)
        g_l = np.cos(pan * np.pi * 0.5)
        g_r = np.sin(pan * np.pi * 0.5)
        itd = itd_ms * 1e-3 * sr * np.sin(np.deg2rad(az))
        idx = np.arange(work.shape[0], dtype=np.float64)
        left = np.interp(idx - np.maximum(itd, 0.0), idx, mono, left=mono[0], right=mono[-1])
        right = np.interp(idx + np.minimum(itd, 0.0), idx, mono, left=mono[0], right=mono[-1])
        out = np.stack([left * g_l, right * g_r], axis=1)
        notes.append("Designed dynamic binaural motion trajectory with time-varying pan and ITD.")

    elif slug == "stochastic_spatial_diffusion_cloud":
        output_channels = int(params.get("output_channels", 6))
        diffusion = float(params.get("diffusion", 0.8))
        max_delay_ms = float(params.get("max_delay_ms", 18.0))
        seed = int(params.get("seed", 1307))
        rng = np.random.default_rng(seed)
        mono = np.mean(work, axis=1)
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        max_delay = max_delay_ms * 1e-3 * sr
        for ch in range(output_channels):
            delay = rng.uniform(0.0, max_delay)
            y = _spatial_fractional_delay(mono, delay)
            a = np.clip(diffusion * rng.uniform(0.2, 0.9), 0.05, 0.95)
            y = signal.lfilter([a, 1.0], [1.0, a], y)
            y = signal.lfilter([1.0, -0.4 * a], [1.0], y)
            out[:, ch] = y * rng.uniform(0.6, 1.0)
        notes.append(
            "Generated stochastic spatial diffusion cloud with decorrelated delay/all-pass taps."
        )

    elif slug == "decorrelated_reverb_upmix":
        output_channels = int(params.get("output_channels", 6))
        decay_s = float(params.get("decay_s", 1.2))
        rir_length = int(params.get("rir_length", max(256, int(sr * min(3.0, decay_s * 2.0)))))
        mix = float(params.get("mix", 0.45))
        seed = int(params.get("seed", 1307))
        mono = np.mean(work, axis=1)
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        for ch in range(output_channels):
            rir = _spatial_synthetic_rir(rir_length, decay_s, sr, seed + 911, ch)
            wet = signal.fftconvolve(mono, rir, mode="full")[: work.shape[0]]
            dry = mono if ch % 2 == 0 else _spatial_fractional_delay(mono, float((ch + 1) * 2.0))
            out[:, ch] = (1.0 - mix) * dry + mix * wet
        notes.append("Upmixed source with decorrelated synthetic reverb field.")

    elif slug == "spectral_spatial_granulator":
        output_channels = int(params.get("output_channels", max(2, work.shape[1])))
        grain = int(params.get("grain", 1024))
        spread_semitones = float(params.get("spread_semitones", 5.0))
        density = float(params.get("density", 1.0))
        seed = int(params.get("seed", 1307))
        rng = np.random.default_rng(seed)
        mono = np.mean(work, axis=1)[:, None]
        out = np.zeros((work.shape[0], output_channels), dtype=np.float64)
        for ch in range(output_channels):
            stretch = float(np.clip(1.0 + rng.normal(0.0, 0.18 * max(0.1, density)), 0.5, 2.0))
            granulated = granular_time_stretch(
                mono, stretch=stretch, grain=grain, hop=max(64, grain // 4)
            )
            semitones = float(rng.normal(0.0, spread_semitones))
            granulated = pitch_shift(granulated, sr, semitones)
            granulated = ensure_length(granulated, work.shape[0])[:, 0]
            lfo = 0.7 + 0.3 * np.sin(
                2.0 * np.pi * (0.07 * (ch + 1)) * np.arange(work.shape[0]) / float(sr)
                + rng.uniform(0.0, 2.0 * np.pi)
            )
            out[:, ch] = granulated * lfo
        notes.append(
            "Rendered spectral-spatial granulator with per-channel stochastic pitch/grain trajectories."
        )

    elif slug == "spatial_freeze_resynthesis":
        output_channels = int(params.get("output_channels", work.shape[1]))
        frame_ratio = float(params.get("frame_ratio", 0.35))
        phase_drift = float(params.get("phase_drift", 0.03))
        src = _spatial_to_channels(work, output_channels)
        spec, _, _ = stft_multi(src, n_fft=2048, hop=512)
        idx = int(np.clip(round(frame_ratio * (spec.shape[1] - 1)), 0, max(0, spec.shape[1] - 1)))
        frozen = spec[:, idx, :]
        mag = np.abs(frozen)
        base_phase = np.angle(frozen)
        out_spec = np.zeros_like(spec)
        for t in range(spec.shape[1]):
            theta = 2.0 * np.pi * phase_drift * t
            for ch in range(output_channels):
                phase = base_phase[:, ch] + theta * (1.0 + 0.15 * ch)
                out_spec[:, t, ch] = mag[:, ch] * np.exp(1j * phase)
        out = istft_multi(out_spec, n_fft=2048, hop=512, length=work.shape[0])
        notes.append(
            "Resynthesized frozen spatial spectra with controlled per-channel phase drift."
        )

    else:
        out = work.copy()
        notes.append("Spatial fallback passthrough.")

    return normalize_peak(out), notes, extras
