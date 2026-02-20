# pvx Application Programming Interface (API) Overview

This guide shows how to use `pvx` as a Python library instead of shell commands.

## Acronym Primer

- application programming interface (API)
- command-line interface (CLI)
- path environment variable (PATH)
- digital signal processing (DSP)
- short-time Fourier transform (STFT)
- fast Fourier transform (FFT)
- central processing unit (CPU)
- graphics processing unit (GPU)
- Compute Unified Device Architecture (CUDA)
- comma-separated values (CSV)
- JavaScript Object Notation (JSON)
- fundamental frequency (F0)
- signal-to-noise ratio (SNR)

## 0) Quick Setup (Install + PATH)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
pvx --help
```

If `pvx` is not found, add the virtual environment `bin` directory to your path environment variable (`PATH`) (`zsh`):

```bash
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$(pwd)" >> ~/.zshrc
source ~/.zshrc
pvx --help
```

No-`PATH` fallback for CLI validation:

```bash
python3 pvx.py help voc
```

## 1) Import Paths

If installed with `pip install -e .`:

```python
from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch, resample_1d
```

From repository source without install:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch, resample_1d
```

## 2) Minimal Time-Stretch Example

```python
import numpy as np
import soundfile as sf
from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch

x, sr = sf.read("input.wav", always_2d=True)
mono = np.mean(x, axis=1)

cfg = VocoderConfig(
    n_fft=2048,
    win_length=2048,
    hop_size=512,
    window="hann",
    center=True,
    phase_locking="identity",
    transient_preserve=True,
    transient_threshold=2.0,
)

y = phase_vocoder_time_stretch(mono, 1.25, cfg)
sf.write("output_stretch.wav", y, sr)
```

## 3) Minimal Pitch-Shift (Duration-Preserving)

```python
import numpy as np
import soundfile as sf
from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch, resample_1d

x, sr = sf.read("input.wav", always_2d=True)
mono = np.mean(x, axis=1)

semitones = 3.0
ratio = 2 ** (semitones / 12.0)
internal_stretch = ratio

cfg = VocoderConfig(
    n_fft=2048,
    win_length=2048,
    hop_size=512,
    window="hann",
    center=True,
    phase_locking="identity",
    transient_preserve=True,
    transient_threshold=2.0,
)

stretched = phase_vocoder_time_stretch(mono, internal_stretch, cfg)
y = resample_1d(stretched, int(round(stretched.size / ratio)), mode="linear")
sf.write("output_pitch.wav", y, sr)
```

## 4) Jupyter Notebook Snippets

### Spectrogram visualization snippet

```python
import numpy as np
import matplotlib.pyplot as plt


def db_mag(sig, sr):
    n_fft = 2048
    hop = 512
    win = np.hanning(n_fft)
    frames = []
    for i in range(0, max(1, sig.size - n_fft), hop):
        chunk = sig[i:i+n_fft]
        if chunk.size < n_fft:
            break
        mag = np.abs(np.fft.rfft(chunk * win))
        frames.append(20 * np.log10(mag + 1e-9))
    return np.array(frames).T

S = db_mag(y, sr)
plt.figure(figsize=(10, 4))
plt.imshow(S, origin="lower", aspect="auto", cmap="magma")
plt.colorbar(label="dB")
plt.title("Output spectrogram")
plt.xlabel("Frame")
plt.ylabel("Bin")
plt.tight_layout()
```

### Compare phase locking modes

```python
from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch

cfg_free = VocoderConfig(
    n_fft=2048, win_length=2048, hop_size=512, window="hann",
    center=True, phase_locking="off", transient_preserve=True, transient_threshold=2.0
)
cfg_lock = VocoderConfig(
    n_fft=2048, win_length=2048, hop_size=512, window="hann",
    center=True, phase_locking="identity", transient_preserve=True, transient_threshold=2.0
)

y_free = phase_vocoder_time_stretch(mono, 1.25, cfg_free)
y_lock = phase_vocoder_time_stretch(mono, 1.25, cfg_lock)
```

## 5) Batch Processing Example (Python)

```python
from pathlib import Path
import numpy as np
import soundfile as sf
from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch

cfg = VocoderConfig(
    n_fft=2048, win_length=2048, hop_size=512,
    window="hann", center=True,
    phase_locking="identity", transient_preserve=True, transient_threshold=2.0
)

inp = Path("stems")
out = Path("out_stems")
out.mkdir(exist_ok=True)

for path in sorted(inp.glob("*.wav")):
    x, sr = sf.read(path, always_2d=True)
    y_channels = []
    for ch in range(x.shape[1]):
        y_channels.append(phase_vocoder_time_stretch(x[:, ch], 1.08, cfg))
    n = max(len(ch) for ch in y_channels)
    y = np.zeros((n, len(y_channels)), dtype=np.float64)
    for idx, ch in enumerate(y_channels):
        y[:len(ch), idx] = ch
    sf.write(out / f"{path.stem}_pv.wav", y, sr)
```

## 6) Segment Processing Example (Python)

```python
from dataclasses import dataclass
import numpy as np
import soundfile as sf
from pvx.core.voc import VocoderConfig, phase_vocoder_time_stretch, resample_1d

@dataclass
class Segment:
    start_sec: float
    end_sec: float
    stretch: float
    pitch_ratio: float

segments = [
    Segment(0.0, 1.0, 1.00, 1.00),
    Segment(1.0, 2.0, 1.20, 1.00),
    Segment(2.0, 3.0, 0.95, 2 ** (2/12)),
]

x, sr = sf.read("input.wav", always_2d=True)
mono = np.mean(x, axis=1)
cfg = VocoderConfig(
    n_fft=2048, win_length=2048, hop_size=512,
    window="hann", center=True,
    phase_locking="identity", transient_preserve=True, transient_threshold=2.0
)

parts = []
for seg in segments:
    s = int(round(seg.start_sec * sr))
    e = int(round(seg.end_sec * sr))
    if e <= s:
        continue
    block = mono[s:e]
    internal = seg.stretch * seg.pitch_ratio
    z = phase_vocoder_time_stretch(block, internal, cfg)
    if abs(seg.pitch_ratio - 1.0) > 1e-9:
        z = resample_1d(z, int(round(z.size / seg.pitch_ratio)), mode="linear")
    parts.append(z)

y = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
sf.write("segment_output.wav", y, sr)
```

## 7) Related CLI Equivalents

- `python3 pvxvoc.py ...` for direct production workflows.
- `python3 pvxvoc.py --explain-plan` to inspect resolved processing settings.
- `python3 pvxvoc.py --manifest-json ...` to log run metadata.
