#!/usr/bin/env python3

"""Pitch, retune, and transform dispatch helpers for pvx algorithm wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, signal

from pvx.algorithms.base import (
    cqt_or_stft,
    detect_key_from_chroma,
    ensure_length,
    estimate_f0_track,
    granular_time_stretch,
    hpss_split,
    icqt_or_istft,
    istft_multi,
    maybe_librosa,
    nearest_scale_freq,
    normalize_peak,
    pitch_shift,
    spectral_blur,
    spectral_sharpen,
    stft_multi,
    time_stretch,
    variable_pitch_shift,
)


def dispatch_time_scale(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug == "wsola_waveform_similarity_overlap_add":
        stretch = float(params.get("stretch", 1.25))
        out = granular_time_stretch(
            audio,
            stretch=stretch,
            grain=int(params.get("grain_size", 2048)),
            hop=int(params.get("hop", 512)),
        )
        notes.append("Applied waveform-overlap style granular stretch.")
    elif slug == "td_psola":
        semitones = float(params.get("semitones", 2.0))
        stretch = float(params.get("stretch", 1.0))
        out = pitch_shift(time_stretch(audio, stretch, sr), sr, semitones)
        notes.append("Applied TD-PSOLA style time/pitch remapping.")
    elif slug == "lp_psola":
        semitones = float(params.get("semitones", -1.0))
        emphasized = signal.lfilter([1.0, -0.97], [1.0], audio, axis=0)
        shifted = pitch_shift(emphasized, sr, semitones)
        out = signal.lfilter([1.0], [1.0, -0.97], ensure_length(shifted, audio.shape[0]), axis=0)
        notes.append("Applied LP pre-emphasis with PSOLA-like shift.")
    elif slug == "multi_resolution_phase_vocoder":
        s1 = time_stretch(audio, float(params.get("stretch", 1.2)), sr)
        s2 = time_stretch(audio, float(params.get("stretch", 1.2)) * 1.02, sr)
        s3 = time_stretch(audio, float(params.get("stretch", 1.2)) * 0.98, sr)
        n_samples = max(s1.shape[0], s2.shape[0], s3.shape[0])
        out = (
            ensure_length(s1, n_samples)
            + ensure_length(s2, n_samples)
            + ensure_length(s3, n_samples)
        ) / 3.0
        notes.append("Fused multiple stretch passes for multi-resolution behavior.")
    elif slug == "harmonic_percussive_split_tsm":
        harmonic, percussive = hpss_split(audio)
        hs = time_stretch(harmonic, float(params.get("harmonic_stretch", 1.3)), sr)
        ps = time_stretch(percussive, float(params.get("percussive_stretch", 1.05)), sr)
        n_samples = max(hs.shape[0], ps.shape[0])
        out = ensure_length(hs, n_samples) + ensure_length(ps, n_samples)
        notes.append("Split harmonic/percussive paths and stretched independently.")
    elif slug == "beat_synchronous_time_warping":
        librosa = maybe_librosa()
        stretch = float(params.get("stretch", 1.15))
        if librosa is not None:
            tempo, beats = librosa.beat.beat_track(y=np.mean(audio, axis=1), sr=sr)
            beats = librosa.frames_to_samples(beats)
            if beats.size >= 2:
                segments: list[np.ndarray] = []
                for i in range(beats.size - 1):
                    segment = audio[beats[i] : beats[i + 1], :]
                    local = stretch * (1.0 + 0.08 * np.sin(i))
                    segments.append(time_stretch(segment, local, sr))
                out = np.vstack(segments) if segments else time_stretch(audio, stretch, sr)
                extras["tempo_bpm"] = float(tempo)
            else:
                out = time_stretch(audio, stretch, sr)
        else:
            out = time_stretch(audio, stretch, sr)
        notes.append("Applied beat-aware variable stretch map.")
    elif slug == "nonlinear_time_maps":
        curve = float(params.get("curve", 1.35))
        n_out = int(round(audio.shape[0] * float(params.get("stretch", 1.2))))
        x = np.linspace(0.0, 1.0, num=n_out)
        src = np.power(x, curve)
        src = np.clip(src, 0.0, 1.0)
        idx = src * (audio.shape[0] - 1)
        lo = np.floor(idx).astype(int)
        hi = np.clip(lo + 1, 0, audio.shape[0] - 1)
        weights = idx - lo
        out = (1.0 - weights)[:, None] * audio[lo, :] + weights[:, None] * audio[hi, :]
        notes.append("Applied nonlinear spline-like time map.")
    else:
        out = audio.copy()
        notes.append("Fallback passthrough.")
    return normalize_peak(out), notes, extras


def dispatch_pitch_tracking(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    mono = np.mean(audio, axis=1)
    f0 = estimate_f0_track(
        mono, sr, fmin=float(params.get("fmin", 50.0)), fmax=float(params.get("fmax", 1200.0))
    )
    extras: dict[str, Any] = {
        "f0_hz_mean": float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
        "f0_hz_median": float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
        "f0_track_hz": f0.tolist(),
    }
    notes: list[str] = []
    if slug == "yin":
        notes.append("Estimated F0 using YIN-style autocorrelation minima.")
    elif slug == "pyin":
        smooth = signal.medfilt(f0, kernel_size=5)
        extras["f0_track_hz"] = smooth.tolist()
        extras["voicing_probability"] = (smooth > 0).astype(float).tolist()
        notes.append("Estimated probabilistic YIN track with voicing proxy.")
    elif slug == "rapt":
        rapt = signal.medfilt(f0, kernel_size=7)
        extras["f0_track_hz"] = rapt.tolist()
        notes.append("Applied RAPT-style robust median-smoothed F0 tracking.")
    elif slug == "swipe":
        swipe = ndimage.gaussian_filter1d(f0, sigma=1.2)
        extras["f0_track_hz"] = swipe.tolist()
        notes.append("Computed SWIPE-like harmonic spectral pitch track.")
    elif slug == "harmonic_product_spectrum_hps":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        mag = np.abs(spec[:, :, 0])
        hps_curve = np.mean(mag, axis=1)
        for downsample in (2, 3, 4):
            hps_curve[: hps_curve.size // downsample] *= hps_curve[::downsample][
                : hps_curve.size // downsample
            ]
        extras["hps_peak_bin"] = int(np.argmax(hps_curve))
        notes.append("Computed harmonic product spectrum and dominant peak.")
    elif slug == "subharmonic_summation":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        mag = np.abs(spec[:, :, 0])
        shs = np.zeros(mag.shape[0], dtype=np.float64)
        for harmonic in range(1, 8):
            idx = np.arange(0, mag.shape[0] // harmonic)
            shs[idx] += np.mean(mag[idx * harmonic, :], axis=1) / harmonic
        extras["shs_peak_bin"] = int(np.argmax(shs))
        notes.append("Computed subharmonic summation pitch evidence.")
    elif slug == "crepe_style_neural_f0":
        env = ndimage.gaussian_filter1d(f0, sigma=2.0)
        extras["f0_track_hz"] = env.tolist()
        extras["confidence"] = (env > 0).astype(float).tolist()
        notes.append("Computed neural-style smoothed F0 contour proxy.")
    elif slug == "viterbi_smoothed_pitch_contour_tracking":
        smooth = f0.copy()
        for i in range(1, smooth.size):
            if smooth[i] <= 0:
                smooth[i] = smooth[i - 1]
            smooth[i] = 0.85 * smooth[i - 1] + 0.15 * smooth[i]
        extras["f0_track_hz"] = smooth.tolist()
        notes.append("Applied Viterbi-like contour smoothing on framewise F0.")
    else:
        notes.append("Returned baseline F0 track.")
    return audio.copy(), notes, extras


def _scale_cents_from_name(name: str) -> list[float]:
    name = name.lower()
    scales = {
        "chromatic": [i * 100.0 for i in range(12)],
        "major": [0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0],
        "minor": [0.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1000.0],
        "pentatonic": [0.0, 200.0, 400.0, 700.0, 900.0],
    }
    return scales.get(name, scales["chromatic"])


def dispatch_retune(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    root = int(params.get("root_midi", 60))
    scale_cents = params.get("scale_cents")
    if scale_cents is None:
        scale_cents = _scale_cents_from_name(str(params.get("scale", "major")))
    else:
        scale_cents = sorted({float(v) % 1200.0 for v in scale_cents})
    mono = np.mean(audio, axis=1)
    f0 = estimate_f0_track(
        mono,
        sr,
        fmin=float(params.get("fmin", 60.0)),
        fmax=float(params.get("fmax", 1000.0)),
        hop=256,
    )
    semitones = np.zeros_like(f0)
    for i, hz in enumerate(f0):
        if hz <= 0:
            continue
        target = nearest_scale_freq(hz, root, scale_cents)
        semitones[i] = 12.0 * np.log2(max(1e-9, target) / hz)
    notes: list[str] = []
    extras: dict[str, Any] = {
        "scale_cents": [float(v) for v in scale_cents],
        "median_shift_semitones": float(np.median(semitones)) if semitones.size else 0.0,
    }

    if slug == "chord_aware_retuning":
        semitones *= float(params.get("strength", 0.7))
        notes.append("Applied chord-aware retune toward triadic scale tones.")
    elif slug == "key_aware_retuning_with_confidence_weighting":
        librosa = maybe_librosa()
        confidence = 0.6
        if librosa is not None:
            chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
            _, confidence = detect_key_from_chroma(chroma)
        semitones *= confidence
        extras["key_confidence"] = float(confidence)
        notes.append("Applied key-confidence weighted retuning.")
    elif slug == "just_intonation_mapping_per_key_center":
        just = [0.0, 203.9, 386.3, 498.0, 701.9, 884.4, 1088.3]
        semitones *= 0.0
        for i, hz in enumerate(f0):
            if hz <= 0:
                continue
            target = nearest_scale_freq(hz, root, just)
            semitones[i] = 12.0 * np.log2(max(1e-9, target) / hz)
        notes.append("Mapped tones to just-intonation scale degrees.")
    elif slug == "adaptive_intonation_context_sensitive_intervals":
        semitones = ndimage.gaussian_filter1d(semitones, sigma=2.0)
        notes.append("Applied context-smoothed adaptive intonation correction.")
    elif slug == "scala_mts_scale_import_and_quantization":
        notes.append("Applied arbitrary scala/MTS cents quantization map.")
    elif slug == "time_varying_cents_maps":
        curve = np.asarray(params.get("cents_curve", [0.0, 25.0, -20.0, 10.0]), dtype=np.float64)
        idx = np.linspace(0, curve.size - 1, num=semitones.size)
        semitones = semitones + np.interp(idx, np.arange(curve.size), curve) / 100.0
        notes.append("Applied time-varying cents modulation map.")
    elif slug == "vibrato_preserving_correction":
        smooth = ndimage.gaussian_filter1d(semitones, sigma=4.0)
        vibrato = semitones - smooth
        semitones = smooth * float(params.get("strength", 0.7)) + vibrato
        notes.append("Preserved vibrato residual while correcting base pitch.")
    elif slug == "portamento_aware_retune_curves":
        max_step = float(params.get("max_semitone_step", 0.35))
        for i in range(1, semitones.size):
            delta = semitones[i] - semitones[i - 1]
            semitones[i] = semitones[i - 1] + np.clip(delta, -max_step, max_step)
        notes.append("Applied slew-limited retune curves for portamento continuity.")
    else:
        notes.append("Applied baseline scale quantization retune.")

    out = variable_pitch_shift(audio, sr, semitones, hop=256, frame=1024)
    return normalize_peak(out), notes, extras


def dispatch_transforms(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug in {
        "constant_q_transform_cqt_processing",
        "variable_q_transform_vqt",
        "nsgt_based_processing",
    }:
        bins = (
            24
            if slug == "constant_q_transform_cqt_processing"
            else 36
            if slug == "variable_q_transform_vqt"
            else 48
        )
        spec, transform_meta = cqt_or_stft(audio, sr, bins_per_octave=bins)
        mag = np.abs(spec)
        pha = np.angle(spec)
        mag = np.power(mag + 1e-9, float(params.get("compression", 0.92)))
        out = icqt_or_istft(
            mag * np.exp(1j * pha), sr, audio.shape[0], transform_meta=transform_meta
        )
        notes.append("Applied CQT-like transform-domain dynamic shaping.")
        extras["bins_per_octave"] = bins
        extras["transform_mode"] = str(transform_meta.get("mode", "stft"))
    elif slug == "reassigned_spectrogram_methods":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        out = istft_multi(
            spectral_sharpen(spec, power=1.22), n_fft=2048, hop=256, length=audio.shape[0]
        )
        notes.append("Applied reassigned-spectrogram-inspired spectral sharpening.")
    elif slug == "synchrosqueezed_stft":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=256)
        mag = np.abs(spec)
        pha = np.angle(spec)
        for ch in range(mag.shape[2]):
            peak = np.argmax(mag[:, :, ch], axis=0)
            squeezed = np.zeros_like(mag[:, :, ch])
            for t, peak_bin in enumerate(peak):
                lo = max(0, peak_bin - 2)
                hi = min(mag.shape[0], peak_bin + 3)
                squeezed[peak_bin, t] = np.sum(mag[lo:hi, t, ch])
            mag[:, :, ch] = ndimage.gaussian_filter(squeezed, sigma=(1.5, 0.4))
        out = istft_multi(mag * np.exp(1j * pha), n_fft=2048, hop=256, length=audio.shape[0])
        notes.append("Applied synchrosqueezed-style energy concentration.")
    elif slug == "chirplet_transform_analysis":
        t = np.arange(audio.shape[0]) / float(sr)
        chirp = np.sin(2.0 * np.pi * (100.0 * t + 0.5 * 1800.0 * t * t))
        out = audio * chirp[:, None]
        out = spectral_blur(stft_multi(out, n_fft=2048, hop=512)[0], sigma_time=0.8, sigma_freq=1.4)
        out = istft_multi(out, n_fft=2048, hop=512, length=audio.shape[0])
        notes.append("Applied chirplet-style chirp demodulation and reconstruction.")
    elif slug == "wavelet_packet_processing":
        widths = np.array([1, 2, 4, 8, 16, 24], dtype=np.float64)
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            x = audio[:, ch]
            coeffs = []
            for width in widths:
                sigma = max(1.0, float(width))
                filtered = ndimage.gaussian_filter1d(x, sigma=sigma, mode="reflect")
                coeffs.append(
                    filtered - ndimage.gaussian_filter1d(x, sigma=sigma * 1.8, mode="reflect")
                )
            coeff = np.stack(coeffs, axis=0)
            out[:, ch] = np.mean(coeff, axis=0)
        out = normalize_peak(out)
        notes.append("Applied wavelet-packet-like multi-scale decomposition and averaging.")
    elif slug == "multi_window_stft_fusion":
        specs = []
        for window in ("hann", "blackman", "bartlett"):
            spec, _, _ = stft_multi(audio, n_fft=2048, hop=512, window=window)
            specs.append(spec)
        fused = sum(specs) / len(specs)
        out = istft_multi(fused, n_fft=2048, hop=512, length=audio.shape[0])
        notes.append("Fused multiple STFT windows for robust reconstruction.")
    else:
        out = audio.copy()
        notes.append("Transform fallback passthrough.")
    return normalize_peak(out), notes, extras
