# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

from __future__ import annotations

import numpy as np
from scipy import ndimage


def hpss_split(audio: np.ndarray, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-Percussive Source Separation."""
    from pvx.algorithms.base import istft_multi, maybe_librosa, stft_multi

    librosa = maybe_librosa()
    if librosa is not None:
        harm_channels: list[np.ndarray] = []
        perc_channels: list[np.ndarray] = []
        for ch in range(audio.shape[1]):
            st = librosa.stft(audio[:, ch], n_fft=n_fft, hop_length=hop)
            h, p = librosa.decompose.hpss(st)
            harm = librosa.istft(h, hop_length=hop, length=audio.shape[0])
            perc = librosa.istft(p, hop_length=hop, length=audio.shape[0])
            harm_channels.append(harm.astype(np.float64, copy=False))
            perc_channels.append(perc.astype(np.float64, copy=False))
        return np.stack(harm_channels, axis=1), np.stack(perc_channels, axis=1)

    spec, _, _ = stft_multi(audio, n_fft=n_fft, hop=hop)
    mag = np.abs(spec)
    harm = ndimage.median_filter(mag, size=(1, 17, 1))
    perc = ndimage.median_filter(mag, size=(17, 1, 1))
    denom = harm + perc + 1e-12
    mh = harm / denom
    mp = perc / denom
    h_spec = spec * mh
    p_spec = spec * mp
    return istft_multi(h_spec, n_fft=n_fft, hop=hop, length=audio.shape[0]), istft_multi(
        p_spec, n_fft=n_fft, hop=hop, length=audio.shape[0]
    )
