#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""PVC-inspired harmonic/chord spectral mapping for pvx.

Phase 5 coverage:
- chordmapper
- inharmonator
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pvx.core.common import coerce_audio
from pvx.core.voc import VocoderConfig, istft, stft

HarmonyOperatorName = Literal["chordmapper", "inharmonator"]

CHORD_INTERVALS_SEMITONES: dict[str, tuple[int, ...]] = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "power": (0, 7),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
}


def chord_mapper_mask(
    freqs_hz: np.ndarray,
    *,
    root_hz: float,
    chord: str,
    tolerance_cents: float,
) -> np.ndarray:
    """Return [0,1] weight per bin for proximity to chord pitch classes."""
    f = np.asarray(freqs_hz, dtype=np.float64).reshape(-1)
    out = np.zeros_like(f)
    if f.size == 0:
        return out
    root = max(1e-6, float(root_hz))
    chord_key = str(chord).strip().lower()
    intervals = CHORD_INTERVALS_SEMITONES.get(chord_key, CHORD_INTERVALS_SEMITONES["major"])
    classes = np.asarray([(100.0 * semi) % 1200.0 for semi in intervals], dtype=np.float64)
    tol = max(1e-3, float(tolerance_cents))

    valid = f > 1e-6
    if not np.any(valid):
        return out
    cents = np.zeros_like(f)
    cents[valid] = 1200.0 * np.log2(f[valid] / root)
    cents_mod = np.mod(cents, 1200.0)

    for idx in np.flatnonzero(valid):
        # Gaussian distance in cents gives a smooth, musically intuitive attraction field.
        d = np.abs(cents_mod[idx] - classes)
        d = np.minimum(d, 1200.0 - d)
        nearest = float(np.min(d))
        out[idx] = float(np.exp(-0.5 * (nearest / tol) ** 2))
    return np.clip(out, 0.0, 1.0)


def _inharmonic_inverse_map(freqs_hz: np.ndarray, *, f0_hz: float, inharmonicity: float) -> np.ndarray:
    f_out = np.asarray(freqs_hz, dtype=np.float64).reshape(-1)
    f0 = max(1e-6, float(f0_hz))
    b = max(0.0, float(inharmonicity))
    if b <= 1e-12:
        return f_out.copy()

    a = b / (f0 * f0)
    # Solve u + a*u^2 = f_out^2, where u = f_in^2; this inverts stiff-string dispersion.
    rhs = np.square(f_out)
    disc = 1.0 + 4.0 * a * rhs
    u = np.maximum(0.0, (-1.0 + np.sqrt(disc)) / (2.0 * a))
    return np.sqrt(u)


def _interp_mag_phase_from_freq(
    mag: np.ndarray,
    pha: np.ndarray,
    freqs_hz: np.ndarray,
    src_freqs_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    src_f = np.asarray(src_freqs_hz, dtype=np.float64).reshape(-1)
    mag_src = np.asarray(mag, dtype=np.float64).reshape(-1)
    pha_src = np.asarray(pha, dtype=np.float64).reshape(-1)

    mag_dst = np.interp(freqs_hz, src_f, mag_src, left=mag_src[0], right=mag_src[-1])
    # Interpolate phase on the unit circle to avoid branch-wrap discontinuities.
    cos_src = np.cos(pha_src)
    sin_src = np.sin(pha_src)
    cos_dst = np.interp(freqs_hz, src_f, cos_src, left=cos_src[0], right=cos_src[-1])
    sin_dst = np.interp(freqs_hz, src_f, sin_src, left=sin_src[0], right=sin_src[-1])
    pha_dst = np.arctan2(sin_dst, cos_dst)
    return mag_dst, pha_dst


def process_harmony_operator(
    audio: np.ndarray,
    sample_rate: int,
    config: VocoderConfig,
    *,
    operator: HarmonyOperatorName,
    root_hz: float = 220.0,
    chord: str = "major",
    strength: float = 0.75,
    tolerance_cents: float = 35.0,
    boost_db: float = 6.0,
    attenuation: float = 0.45,
    inharmonicity: float = 1e-4,
    inharmonic_f0_hz: float = 220.0,
    inharmonic_mix: float = 1.0,
    dry_mix: float = 0.0,
) -> np.ndarray:
    """Apply chord-mapper or inharmonator operator in STFT domain."""
    work = coerce_audio(audio)
    out = np.zeros_like(work)
    wet_mix = np.clip(1.0 - float(dry_mix), 0.0, 1.0)
    dry_mix_clamped = np.clip(float(dry_mix), 0.0, 1.0)

    for ch in range(work.shape[1]):
        signal = work[:, ch]
        spec = stft(signal, config)
        mag = np.abs(spec)
        pha = np.angle(spec)
        n_bins, n_frames = mag.shape
        if n_bins <= 0 or n_frames <= 0:
            out[:, ch] = signal
            continue

        freqs = np.linspace(0.0, float(sample_rate) * 0.5, n_bins, dtype=np.float64)

        if operator == "chordmapper":
            mask = chord_mapper_mask(
                freqs,
                root_hz=float(root_hz),
                chord=str(chord),
                tolerance_cents=float(tolerance_cents),
            )
            chord_boost = float(10.0 ** (float(boost_db) / 20.0))
            s = np.clip(float(strength), 0.0, 1.0)
            a = np.clip(float(attenuation), 0.0, 1.0)
            gain = (1.0 - s * a) + (s * chord_boost * mask)
            out_mag = mag * np.maximum(gain[:, None], 1e-9)
            out_phase = pha
        elif operator == "inharmonator":
            inv_freq = _inharmonic_inverse_map(
                freqs,
                f0_hz=float(inharmonic_f0_hz),
                inharmonicity=float(inharmonicity),
            )
            warped_mag = np.zeros_like(mag)
            warped_phase = np.zeros_like(pha)
            # Warp each frame along the inharmonic frequency map, then blend by mix.
            for frame in range(n_frames):
                m_frame, p_frame = _interp_mag_phase_from_freq(
                    mag[:, frame],
                    pha[:, frame],
                    inv_freq,
                    freqs,
                )
                warped_mag[:, frame] = m_frame
                warped_phase[:, frame] = p_frame
            mix_val = np.clip(float(inharmonic_mix), 0.0, 1.0)
            out_mag = (1.0 - mix_val) * mag + mix_val * warped_mag
            out_phase = (1.0 - mix_val) * pha + mix_val * warped_phase
        else:
            raise ValueError(f"Unsupported harmony operator: {operator}")

        out_spec = out_mag * np.exp(1j * out_phase)
        wet = istft(out_spec, config, expected_length=signal.size)
        out[:, ch] = dry_mix_clamped * signal + wet_mix * wet

    return out
