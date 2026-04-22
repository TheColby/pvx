#!/usr/bin/env python3

"""Restoration, separation, and loudness dispatch helpers for pvx wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, signal

from pvx.algorithms.base import (
    dereverb_decay_subtract,
    dereverb_wpe_style,
    hpss_split,
    istft_multi,
    maybe_loudnorm,
    minimum_statistics_denoise,
    mmse_like_denoise,
    multiband_compression,
    normalize_peak,
    simple_declick,
    simple_declip,
    spectral_dynamics,
    spectral_gate,
    split_bands,
    stft_multi,
    transient_shaper,
    true_peak_limit,
    upward_compressor,
)


def dispatch_separation(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug == "rpca_hpss":
        harmonic, percussive = hpss_split(audio)
        out = normalize_peak(0.7 * harmonic + 0.3 * percussive)
        notes.append("Separated harmonic/percussive components and remixed.")
    elif slug == "nmf_decomposition":
        spec, _, _ = stft_multi(audio, n_fft=1024, hop=256)
        mag = np.abs(spec[:, :, 0])
        components = int(params.get("components", 4))
        rng = np.random.default_rng(1307)
        w = rng.random((mag.shape[0], components)) + 1e-4
        h = rng.random((components, mag.shape[1])) + 1e-4
        for _ in range(40):
            wh = w @ h + 1e-12
            h *= (w.T @ (mag / wh)) / (np.sum(w, axis=0)[:, None] + 1e-12)
            wh = w @ h + 1e-12
            w *= ((mag / wh) @ h.T) / (np.sum(h, axis=1)[None, :] + 1e-12)
        recon = w @ h
        pha = np.angle(spec[:, :, 0])
        out_mono = istft_multi(
            (recon * np.exp(1j * pha))[:, :, None], n_fft=1024, hop=256, length=audio.shape[0]
        )
        out = np.repeat(out_mono, audio.shape[1], axis=1)
        notes.append("Applied NMF decomposition on magnitude spectrogram.")
        extras["components"] = components
    elif slug == "ica_bss_for_multichannel_stems":
        if audio.shape[1] < 2:
            out = audio.copy()
            notes.append("ICA fallback: mono input, passthrough.")
        else:
            centered = audio - np.mean(audio, axis=0, keepdims=True)
            cov = (centered.T @ centered) / max(1, centered.shape[0])
            d, evecs = np.linalg.eigh(cov)
            whitening = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-9)))
            z = centered @ evecs @ whitening
            w = np.eye(z.shape[1])
            for _ in range(25):
                wz = z @ w.T
                g = np.tanh(wz)
                gp = 1.0 - g**2
                w = (g.T @ z) / z.shape[0] - np.diag(np.mean(gp, axis=0)) @ w
                u, _, vt = np.linalg.svd(w)
                w = u @ vt
            out = normalize_peak(z @ w.T)
            notes.append("Applied FastICA-style blind source separation.")
    elif slug == "sinusoidal_residual_transient_decomposition":
        harmonic, percussive = hpss_split(audio)
        residual = audio - harmonic - percussive
        out = normalize_peak(harmonic + 0.6 * percussive + 0.35 * residual)
        notes.append("Decomposed sinusoidal/residual/transient components.")
    elif slug == "demucs_style_stem_separation_backend":
        harmonic, percussive = hpss_split(audio)
        low, mid, high = split_bands(harmonic, sr)
        stems = [low, mid, high, percussive]
        mix = np.zeros_like(audio)
        for stem in stems:
            mix += normalize_peak(stem, target=0.25)
        out = normalize_peak(mix)
        notes.append("Produced pseudo-stems via multi-band + HPSS backend.")
    elif slug == "u_net_vocal_accompaniment_split":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
        mag = np.abs(spec)
        pha = np.angle(spec)
        freqs = np.linspace(0.0, sr * 0.5, num=mag.shape[0])
        vocal_band = (freqs >= 120.0) & (freqs <= 3500.0)
        mask = np.zeros_like(mag)
        mask[vocal_band, :, :] = 1.0
        mask = ndimage.gaussian_filter(mask, sigma=(2.0, 1.0, 0.0))
        vocal = istft_multi(
            mask * mag * np.exp(1j * pha), n_fft=2048, hop=512, length=audio.shape[0]
        )
        out = normalize_peak(vocal)
        notes.append("Applied U-Net-like spectral masking for vocal emphasis.")
    elif slug == "tensor_decomposition_cp_tucker":
        spec, _, _ = stft_multi(audio, n_fft=1024, hop=256)
        u, s, vh = np.linalg.svd(np.abs(spec[:, :, 0]), full_matrices=False)
        rank = int(params.get("rank", 16))
        rank = max(2, min(rank, s.size))
        recon = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        out_mono = istft_multi(
            (recon * np.exp(1j * np.angle(spec[:, :, 0])))[:, :, None],
            n_fft=1024,
            hop=256,
            length=audio.shape[0],
        )
        out = np.repeat(out_mono, audio.shape[1], axis=1)
        notes.append("Applied low-rank tensor-style decomposition.")
    elif slug == "probabilistic_latent_component_separation":
        spec, _, _ = stft_multi(audio, n_fft=1024, hop=256)
        mag = np.abs(spec)
        pha = np.angle(spec)
        prior = np.mean(mag, axis=1, keepdims=True)
        post = mag / (prior + 1e-9)
        soft = post / (1.0 + post)
        out = istft_multi(soft * mag * np.exp(1j * pha), n_fft=1024, hop=256, length=audio.shape[0])
        notes.append("Applied probabilistic soft-mask latent component separation.")
    else:
        out = audio.copy()
        notes.append("Separation fallback passthrough.")
    return normalize_peak(out), notes, extras


def dispatch_denoise(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "wiener_denoising":
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            out[:, ch] = signal.wiener(audio[:, ch], mysize=11)
        notes.append("Applied Wiener denoising.")
    elif slug == "mmse_stsa":
        out = mmse_like_denoise(audio, alpha=0.98, beta=0.12, log_domain=False)
        notes.append("Applied MMSE-STSA spectral estimator.")
    elif slug == "log_mmse":
        out = mmse_like_denoise(audio, alpha=0.985, beta=0.08, log_domain=True)
        notes.append("Applied log-MMSE spectral estimator.")
    elif slug == "minimum_statistics_noise_tracking":
        out = minimum_statistics_denoise(audio, floor=0.06)
        notes.append("Applied minimum-statistics noise tracking denoiser.")
    elif slug == "rnnoise_style_denoiser":
        hp_b, hp_a = signal.butter(2, 70.0 / (sr * 0.5), btype="high")
        hp = signal.lfilter(hp_b, hp_a, audio, axis=0)
        out = spectral_gate(hp, strength=1.35, floor=0.08)
        notes.append("Applied RNNoise-style high-pass + spectral gate denoiser.")
    elif slug == "diffusion_based_speech_audio_denoise":
        out = audio.copy()
        for _ in range(4):
            out = 0.65 * out + 0.35 * spectral_gate(out, strength=1.1, floor=0.12)
        notes.append("Applied iterative diffusion-like denoise refinement.")
    elif slug == "declip_via_sparse_reconstruction":
        out = simple_declip(audio, clip_threshold=float(params.get("clip_threshold", 0.97)))
        out = spectral_gate(out, strength=1.05, floor=0.1)
        notes.append("Applied clipped-sample interpolation + sparse spectral cleanup.")
    elif slug == "declick_decrackle_median_wavelet_interpolation":
        out = simple_declick(audio, threshold=float(params.get("spike_threshold", 6.0)))
        out = spectral_gate(out, strength=1.0, floor=0.12)
        notes.append("Applied declick/decrackle with median and interpolation cleanup.")
    else:
        out = audio.copy()
        notes.append("Denoise fallback passthrough.")
    return normalize_peak(out), notes, {}


def dispatch_dereverb(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "wpe_dereverberation":
        out = dereverb_wpe_style(
            audio, taps=int(params.get("taps", 4)), delay=int(params.get("delay", 2))
        )
        notes.append("Applied WPE-style late reflection prediction cancellation.")
    elif slug == "spectral_decay_subtraction":
        out = dereverb_decay_subtract(audio, strength=0.42, decay=0.90)
        notes.append("Applied spectral decay subtraction dereverberation.")
    elif slug == "late_reverb_suppression_via_coherence":
        if audio.shape[1] < 2:
            out = dereverb_decay_subtract(audio, strength=0.35, decay=0.92)
        else:
            mid = np.mean(audio[:, :2], axis=1, keepdims=True)
            side = audio[:, :1] - audio[:, 1:2]
            side = signal.lfilter([1.0, -0.85], [1.0], side, axis=0)
            out = np.hstack([mid + side, mid - side])
        notes.append("Suppressed late reverb via coherence-inspired mid/side processing.")
    elif slug == "room_impulse_inverse_filtering":
        rir = np.exp(-np.linspace(0, 8, num=1024))
        rir /= np.sum(rir)
        n_fft = audio.shape[0] + 1023
        inv = np.fft.rfft(rir, n=n_fft)
        inv = np.conj(inv) / (np.abs(inv) ** 2 + 1e-4)
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            x_fft = np.fft.rfft(audio[:, ch], n=n_fft)
            y = np.fft.irfft(x_fft * inv, n=n_fft)
            out[:, ch] = y[: audio.shape[0]]
        notes.append("Applied approximate inverse-filter room compensation.")
    elif slug == "multi_band_adaptive_deverb":
        low, mid, high = split_bands(audio, sr)
        out = (
            dereverb_decay_subtract(low, strength=0.25, decay=0.95)
            + dereverb_decay_subtract(mid, strength=0.40, decay=0.90)
            + dereverb_decay_subtract(high, strength=0.55, decay=0.84)
        )
        notes.append("Applied multi-band adaptive deverb strengths.")
    elif slug == "drr_guided_dereverb":
        early = signal.lfilter([1.0, -0.8], [1.0], audio, axis=0)
        late = audio - early
        drr = float(np.sum(early * early) / (np.sum(late * late) + 1e-12))
        mix = np.clip(drr / (drr + 1.0), 0.2, 0.9)
        out = mix * early + (1.0 - mix) * late * 0.4
        notes.append("Applied DRR-guided early/late rebalance dereverb.")
        return normalize_peak(out), notes, {"estimated_drr": drr}
    elif slug == "blind_deconvolution_dereverb":
        cep = np.fft.irfft(np.log(np.abs(np.fft.rfft(np.mean(audio, axis=1))) + 1e-9))
        cep[int(0.015 * sr) :] = 0.0
        ir = np.fft.irfft(np.exp(np.fft.rfft(cep)))
        ir = ir[: min(512, ir.size)]
        ir /= np.sum(np.abs(ir)) + 1e-12
        out = np.zeros_like(audio)
        for ch in range(audio.shape[1]):
            out[:, ch] = signal.fftconvolve(audio[:, ch], ir[::-1], mode="same")
        notes.append("Applied blind deconvolution via cepstral IR estimate.")
    elif slug == "neural_dereverb_module":
        out = audio.copy()
        for _ in range(3):
            out = 0.7 * out + 0.3 * dereverb_decay_subtract(out, strength=0.38, decay=0.91)
        notes.append("Applied neural-style iterative dereverb refinement module.")
    else:
        out = audio.copy()
        notes.append("Dereverb fallback passthrough.")
    return normalize_peak(out), notes, {}


def _lufs_estimate(audio: np.ndarray, sr: int) -> float:
    pyln = maybe_loudnorm()
    mono = np.mean(audio, axis=1)
    if pyln is not None:
        meter = pyln.Meter(sr)
        try:
            min_samples = int(np.ceil(float(meter.block_size) * float(sr))) + 1
        except (AttributeError, ValueError, TypeError):
            min_samples = int(0.4 * float(sr)) + 1
        if mono.size <= min_samples:
            pad = max(1, min_samples - int(mono.size) + 1)
            mono_eval = np.pad(mono, (0, pad), mode="edge")
        else:
            mono_eval = mono
        try:
            return float(meter.integrated_loudness(mono_eval))
        except (ValueError, RuntimeError):
            pass
    rms = np.sqrt(np.mean(mono * mono) + 1e-12)
    return float(20.0 * np.log10(rms + 1e-12))


def dispatch_dynamics(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    extras: dict[str, Any] = {}
    if slug == "ebu_r128_normalization":
        target = float(params.get("target_lufs", -16.0))
        current = _lufs_estimate(audio, sr)
        gain = 10.0 ** ((target - current) / 20.0)
        out = audio * gain
        notes.append("Applied EBU R128-style loudness normalization.")
        extras["input_lufs"] = current
        extras["target_lufs"] = target
    elif slug == "itu_bs_1770_loudness_measurement_gating":
        current = _lufs_estimate(audio, sr)
        gate = float(params.get("gate_lufs", -70.0))
        notes.append("Measured BS.1770 loudness with simple gating proxy.")
        extras.update({"integrated_lufs": current, "gate_lufs": gate})
        out = audio.copy()
    elif slug == "multi_band_compression":
        out = multiband_compression(audio, sr)
        notes.append("Applied three-band dynamic range compression.")
    elif slug == "upward_compression":
        out = upward_compressor(
            audio,
            threshold_db=float(params.get("threshold_db", -34.0)),
            ratio=float(params.get("ratio", 2.0)),
        )
        notes.append("Applied upward compression to low-level detail.")
    elif slug == "transient_shaping":
        out = transient_shaper(
            audio,
            attack_boost=float(params.get("attack_boost", 1.4)),
            sustain=float(params.get("sustain", 0.9)),
        )
        notes.append("Applied transient shaping envelope transfer.")
    elif slug == "spectral_dynamics_bin_wise_compressor_expander":
        out = spectral_dynamics(
            audio,
            threshold_db=float(params.get("threshold_db", -24.0)),
            ratio=float(params.get("ratio", 2.3)),
        )
        notes.append("Applied bin-wise spectral compression/expansion.")
    elif slug == "true_peak_limiting":
        out = true_peak_limit(audio, threshold=float(params.get("threshold", 0.95)))
        notes.append("Applied true-peak limiting.")
    elif slug == "lufs_target_mastering_chain":
        out = multiband_compression(audio, sr)
        out = transient_shaper(out, attack_boost=1.2, sustain=0.95)
        out = true_peak_limit(out, threshold=0.92)
        target = float(params.get("target_lufs", -14.0))
        current = _lufs_estimate(out, sr)
        out *= 10.0 ** ((target - current) / 20.0)
        out = true_peak_limit(out, threshold=0.92)
        notes.append("Applied LUFS-target mastering chain (compress->shape->limit->normalize).")
        extras["target_lufs"] = target
        extras["post_lufs"] = _lufs_estimate(out, sr)
    else:
        out = audio.copy()
        notes.append("Dynamics fallback passthrough.")
    return normalize_peak(out), notes, extras
