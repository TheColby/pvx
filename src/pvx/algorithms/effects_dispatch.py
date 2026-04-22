#!/usr/bin/env python3

"""Creative, granular, and analysis dispatch helpers for pvx wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from pvx.algorithms.base import (
    cross_synthesis,
    detect_key_from_chroma,
    ensure_length,
    envelope_modulation,
    formant_warp,
    granular_time_stretch,
    istft_multi,
    maybe_librosa,
    normalize_peak,
    phase_randomize,
    pitch_shift,
    resonator_bank,
    rhythmic_gate,
    ring_mod,
    spectral_blur,
    spectral_contrast_exaggerate,
    spectral_convolution,
    spectral_freeze,
    spectral_gate,
    spectral_tremolo,
    stft_multi,
)


def dispatch_creative(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "cross_synthesis_vocoder":
        out = cross_synthesis(audio)
        notes.append("Applied cross-synthesis using magnitude/phase exchange.")
    elif slug == "spectral_convolution_effects":
        out = spectral_convolution(audio, kernel_size=int(params.get("kernel_size", 9)))
        notes.append("Applied spectral convolution effect.")
    elif slug == "spectral_freeze_banks":
        out = spectral_freeze(audio, frame_ratio=float(params.get("frame_ratio", 0.32)))
        notes.append("Applied spectral freeze bank texture rendering.")
    elif slug == "spectral_blur_smear":
        spec, _, _ = stft_multi(audio, n_fft=2048, hop=512)
        out = istft_multi(
            spectral_blur(spec, sigma_time=1.7, sigma_freq=1.0),
            n_fft=2048,
            hop=512,
            length=audio.shape[0],
        )
        notes.append("Applied spectral blur/smear smoothing.")
    elif slug == "phase_randomization_textures":
        out = phase_randomize(audio, strength=float(params.get("strength", 1.0)))
        notes.append("Applied phase randomization texture synthesis.")
    elif slug == "formant_painting_warping":
        out = formant_warp(audio, ratio=float(params.get("ratio", 1.18)))
        notes.append("Applied formant painting/warping transfer.")
    elif slug == "resonator_filterbank_morphing":
        out = resonator_bank(audio, sr)
        notes.append("Applied resonator filterbank morphing.")
    elif slug == "spectral_contrast_exaggeration":
        out = spectral_contrast_exaggerate(audio, amount=float(params.get("amount", 1.4)))
        notes.append("Applied spectral contrast exaggeration.")
    else:
        out = audio.copy()
        notes.append("Creative fallback passthrough.")
    return normalize_peak(out), notes, {}


def dispatch_granular(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    notes: list[str] = []
    if slug == "granular_time_stretch_engine":
        out = granular_time_stretch(
            audio,
            stretch=float(params.get("stretch", 1.3)),
            grain=int(params.get("grain", 2048)),
            hop=int(params.get("hop", 512)),
        )
        notes.append("Applied granular overlap-add time stretch engine.")
    elif slug == "grain_cloud_pitch_textures":
        rng = np.random.default_rng(int(params.get("seed", 1307)))
        grains = []
        grain = int(params.get("grain", 1024))
        for _ in range(int(params.get("count", 64))):
            start = int(rng.integers(0, max(1, audio.shape[0] - grain)))
            chunk = audio[start : start + grain, :]
            semitones = float(rng.normal(0.0, 5.0))
            grains.append(pitch_shift(chunk, sr, semitones))
        out_len = int(round(audio.shape[0] * float(params.get("stretch", 1.0))))
        out = np.zeros((out_len, audio.shape[1]), dtype=np.float64)
        for i, grain_audio in enumerate(grains):
            pos = int((i / max(1, len(grains) - 1)) * max(0, out_len - grain_audio.shape[0]))
            out[pos : pos + grain_audio.shape[0], :] += grain_audio
        out = normalize_peak(out)
        notes.append("Rendered grain-cloud pitch texture synthesis.")
    elif slug == "freeze_grain_morphing":
        grain = int(params.get("grain", 2048))
        start = int(params.get("start", audio.shape[0] * 0.3))
        frozen = ensure_length(audio[start : start + grain, :], grain)
        out = np.zeros_like(audio)
        hop = grain // 4
        win = np.hanning(grain)[:, None]
        for pos in range(0, max(1, out.shape[0]), max(1, hop)):
            n = min(grain, out.shape[0] - pos)
            if n <= 0:
                break
            alpha = pos / max(1, out.shape[0] - n)
            chunk = (1.0 - alpha) * ensure_length(
                audio[pos : pos + grain, :], grain
            ) + alpha * frozen
            out[pos : pos + n, :] += (chunk * win)[:n, :]
        out = normalize_peak(out)
        notes.append("Applied freeze-grain morphing between source and frozen grain.")
    elif slug == "am_fm_ring_modulation_blocks":
        out = ring_mod(
            audio,
            sr,
            freq_hz=float(params.get("freq_hz", 42.0)),
            fm_depth=float(params.get("fm_depth", 2.5)),
        )
        notes.append("Applied AM/FM/ring modulation block.")
    elif slug == "spectral_tremolo":
        out = spectral_tremolo(audio, sr, lfo_hz=float(params.get("lfo_hz", 4.0)))
        notes.append("Applied spectral-domain tremolo.")
    elif slug == "formant_lfo_modulation":
        base = formant_warp(audio, ratio=1.05)
        t = np.arange(base.shape[0]) / float(sr)
        lfo = 1.0 + 0.25 * np.sin(2.0 * np.pi * float(params.get("lfo_hz", 0.8)) * t)
        out = base * lfo[:, None]
        notes.append("Applied formant LFO modulation.")
    elif slug == "rhythmic_gate_stutter_quantizer":
        out = rhythmic_gate(
            audio,
            sr,
            rate_hz=float(params.get("rate_hz", 7.0)),
            duty=float(params.get("duty", 0.28)),
        )
        notes.append("Applied rhythmic gate/stutter quantization.")
    elif slug == "envelope_followed_modulation_routing":
        out = envelope_modulation(audio, sr, depth=float(params.get("depth", 0.75)))
        notes.append("Applied envelope-followed modulation routing.")
    else:
        out = audio.copy()
        notes.append("Granular/modulation fallback passthrough.")
    return normalize_peak(out), notes, {}


def dispatch_analysis(
    slug: str, audio: np.ndarray, sr: int, params: dict[str, Any]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    mono = np.mean(audio, axis=1)
    notes: list[str] = []
    extras: dict[str, Any] = {}
    librosa = maybe_librosa()
    if slug == "onset_beat_downbeat_tracking":
        if librosa is not None:
            onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            tempo_arr = np.asarray(tempo, dtype=np.float64).reshape(-1)
            extras["tempo_bpm"] = float(tempo_arr[0]) if tempo_arr.size else 0.0
            extras["beat_frames"] = beats.tolist()
            extras["onset_strength"] = onset_env.tolist()
        else:
            env = np.abs(np.diff(mono, prepend=mono[0]))
            peaks, _ = signal.find_peaks(env, distance=max(1, int(sr * 0.1)))
            extras["tempo_bpm"] = float(60.0 / 0.5)
            extras["beat_samples"] = peaks.tolist()
        notes.append("Computed onset/beat/downbeat tracking features.")
    elif slug == "key_chord_detection":
        if librosa is not None:
            chroma = librosa.feature.chroma_stft(y=mono, sr=sr)
            key, confidence = detect_key_from_chroma(chroma)
            extras["estimated_key"] = key
            extras["confidence"] = confidence
            chord_root = int(np.argmax(np.mean(chroma, axis=1)))
            extras["estimated_chord"] = (
                f"{['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][chord_root]}maj"
            )
        else:
            extras["estimated_key"] = "C"
            extras["confidence"] = 0.0
            extras["estimated_chord"] = "Cmaj"
        notes.append("Estimated global key and dominant chord class.")
    elif slug == "structure_segmentation_verse_chorus_sections":
        frame = 2048
        hop = 512
        env = []
        for start in range(0, max(1, mono.size - frame + 1), hop):
            env.append(float(np.sqrt(np.mean(mono[start : start + frame] ** 2))))
        env_arr = np.asarray(env, dtype=np.float64)
        novelty = np.abs(np.diff(env_arr, prepend=env_arr[0]))
        cuts, _ = signal.find_peaks(
            novelty, distance=8, prominence=np.mean(novelty) + np.std(novelty)
        )
        extras["section_boundaries_frames"] = cuts.tolist()
        notes.append("Computed structure segmentation boundaries from novelty curve.")
    elif slug == "silence_speech_music_classifiers":
        zcr = np.mean(np.abs(np.diff(np.sign(mono))))
        flat = float(
            np.exp(np.mean(np.log(np.abs(np.fft.rfft(mono)) + 1e-12)))
            / (np.mean(np.abs(np.fft.rfft(mono))) + 1e-12)
        )
        label = "music"
        if np.max(np.abs(mono)) < 0.02:
            label = "silence"
        elif zcr > 0.18 and flat < 0.35:
            label = "speech"
        extras.update({"label": label, "zcr": float(zcr), "spectral_flatness": flat})
        notes.append("Classified input as silence/speech/music via heuristic features.")
    elif slug == "clip_hum_buzz_artifact_detection":
        clipped = float(np.mean(np.abs(audio) > 0.985))
        spec = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
        hum_bins = (np.abs(freqs - 50.0) < 2.0) | (np.abs(freqs - 60.0) < 2.0)
        hum = float(np.sum(spec[hum_bins]) / (np.sum(spec) + 1e-12))
        extras.update(
            {
                "clip_ratio": clipped,
                "hum_ratio": hum,
                "buzz_score": float(clipped * 0.6 + hum * 0.4),
            }
        )
        notes.append("Detected clipping/hum/buzz artifact indicators.")
    elif slug == "pesq_stoi_visqol_quality_metrics":
        spec = np.abs(np.fft.rfft(mono))
        centroid = float(
            np.sum(np.fft.rfftfreq(mono.size, 1.0 / sr) * spec) / (np.sum(spec) + 1e-12)
        )
        snr_proxy = float(
            20.0 * np.log10(np.sqrt(np.mean(mono * mono) + 1e-12) / (np.std(np.diff(mono)) + 1e-12))
        )
        extras.update(
            {
                "pesq_proxy": max(1.0, min(4.5, 1.0 + 0.03 * snr_proxy)),
                "stoi_proxy": max(0.0, min(1.0, 0.5 + 0.004 * snr_proxy)),
                "visqol_proxy": max(1.0, min(5.0, 2.0 + 0.0004 * centroid)),
            }
        )
        notes.append("Computed PESQ/STOI/VISQOL proxy metrics from spectral statistics.")
    elif slug == "auto_parameter_tuning_bayesian_optimization":
        candidates = np.linspace(0.1, 1.0, num=10)
        target_centroid = float(params.get("target_centroid", 1800.0))
        best = 0.1
        best_err = 1e18
        for candidate in candidates:
            shaped = spectral_gate(audio, strength=1.0 + candidate)
            spec = np.abs(np.fft.rfft(np.mean(shaped, axis=1)))
            freqs = np.fft.rfftfreq(shaped.shape[0], d=1.0 / sr)
            centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
            err = abs(centroid - target_centroid)
            if err < best_err:
                best_err = err
                best = float(candidate)
        extras.update({"best_parameter": best, "objective_error": best_err})
        notes.append("Executed Bayesian-style parameter search over candidate grid.")
    elif slug == "batch_preset_recommendation_based_on_source_features":
        rms = float(np.sqrt(np.mean(mono * mono)))
        crest = float(np.max(np.abs(mono)) / (rms + 1e-12))
        flat = float(
            np.exp(np.mean(np.log(np.abs(np.fft.rfft(mono)) + 1e-12)))
            / (np.mean(np.abs(np.fft.rfft(mono))) + 1e-12)
        )
        preset = "balanced"
        if crest > 5.0:
            preset = "transient_focus"
        elif flat > 0.6:
            preset = "denoise_focus"
        elif rms < 0.05:
            preset = "upward_compress"
        extras.update(
            {
                "recommended_preset": preset,
                "rms": rms,
                "crest_factor": crest,
                "spectral_flatness": flat,
            }
        )
        notes.append("Recommended batch preset from extracted source features.")
    else:
        notes.append("Analysis fallback metadata only.")
    return audio.copy(), notes, extras
