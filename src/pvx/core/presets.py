"""Preset definitions for pvx processing intent modes."""

from __future__ import annotations

from typing import Any


# Backward-compatible + new intent-focused preset names.
PRESET_CHOICES: tuple[str, ...] = (
    "none",
    "default",
    "vocal",
    "ambient",
    "extreme",
    "vocal_studio",
    "drums_safe",
    "extreme_ambient",
    "stereo_coherent",
)


PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "none": {},
    "default": {
        "quality_profile": "neutral",
        "transient_mode": "off",
        "stereo_mode": "independent",
        "coherence_strength": 0.0,
    },
    # Legacy presets retained for backward compatibility.
    "vocal": {
        "quality_profile": "speech",
        "pitch_mode": "formant-preserving",
        "phase_locking": "identity",
        "transient_preserve": True,
        "transient_mode": "reset",
    },
    "ambient": {
        "quality_profile": "ambient",
        "ambient_preset": True,
        "transient_mode": "hybrid",
        "transient_sensitivity": 0.45,
        "transient_protect_ms": 35.0,
        "transient_crossfade_ms": 14.0,
    },
    "extreme": {
        "quality_profile": "extreme",
        "stretch_mode": "multistage",
        "extreme_time_stretch": True,
        "transient_mode": "hybrid",
        "transient_sensitivity": 0.55,
        "transient_protect_ms": 36.0,
        "transient_crossfade_ms": 16.0,
    },
    # New intent presets.
    "vocal_studio": {
        "quality_profile": "speech",
        "pitch_mode": "formant-preserving",
        "phase_locking": "identity",
        "window": "hann",
        "n_fft": 4096,
        "win_length": 4096,
        "hop_size": 256,
        "transient_mode": "hybrid",
        "transient_sensitivity": 0.58,
        "transient_protect_ms": 32.0,
        "transient_crossfade_ms": 10.0,
        "stereo_mode": "ref_channel_lock",
        "coherence_strength": 0.45,
    },
    "drums_safe": {
        "quality_profile": "percussion",
        "phase_locking": "identity",
        "window": "kaiser",
        "kaiser_beta": 16.0,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_size": 128,
        "transient_mode": "wsola",
        "transient_sensitivity": 0.70,
        "transient_protect_ms": 24.0,
        "transient_crossfade_ms": 6.0,
        "stereo_mode": "ref_channel_lock",
        "coherence_strength": 0.70,
    },
    "extreme_ambient": {
        "quality_profile": "extreme",
        "phase_engine": "hybrid",
        "ambient_phase_mix": 0.45,
        "phase_locking": "identity",
        "window": "kaiser",
        "kaiser_beta": 20.0,
        "n_fft": 16384,
        "win_length": 16384,
        "hop_size": 1024,
        "stretch_mode": "multistage",
        "max_stage_stretch": 1.2,
        "onset_time_credit": True,
        "onset_credit_pull": 0.7,
        "onset_credit_max": 16.0,
        "transient_mode": "hybrid",
        "transient_sensitivity": 0.46,
        "transient_protect_ms": 42.0,
        "transient_crossfade_ms": 18.0,
        "stereo_mode": "mid_side_lock",
        "coherence_strength": 0.65,
    },
    "stereo_coherent": {
        "quality_profile": "music",
        "phase_locking": "identity",
        "transient_mode": "reset",
        "transient_sensitivity": 0.50,
        "transient_protect_ms": 30.0,
        "transient_crossfade_ms": 10.0,
        "stereo_mode": "mid_side_lock",
        "coherence_strength": 0.9,
    },
}
