from __future__ import annotations

import numpy as np
from scipy import signal

def simple_declick(audio: np.ndarray, threshold: float = 6.0) -> np.ndarray:
    out = audio.copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        dx = np.abs(np.diff(x, prepend=x[0]))
        med = np.median(dx) + 1e-12
        bad = np.where(dx > threshold * med)[0]
        for idx in bad:
            lo = max(0, idx - 2)
            hi = min(x.size, idx + 3)
            x[idx] = np.median(x[lo:hi])
        out[:, ch] = signal.medfilt(x, kernel_size=5)
    return out
