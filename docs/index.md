# pvx

**Phase-vocoder DSP toolkit with a narrow alpha CLI contract and a broader experimental lab**

pvx is a Python toolkit for high-quality time and pitch processing using a phase-vocoder/STFT core, plus a composable audio augmentation library for machine learning.

## Start with the supported surface

For `0.1.0a1`, the stable scripted surface is intentionally narrow:

- `pvx`
- `pvxvoc`
- `pvxfreeze`
- `pvxwarp`
- `pvxformant`
- `pvxfilter`
- `pvxretune`
- `pvxanalysis`

Everything else in the docs tree is either beta, experimental, or reference inventory unless [SUPPORTED_SURFACE.md](SUPPORTED_SURFACE.md) says otherwise.

## Quick Start

```bash
pip install pvx
```

```python
from pvx.augment import Pipeline, AddNoise, GainPerturber, RoomSimulator

pipeline = Pipeline([
    GainPerturber(gain_db=(-3, 3), p=0.8),
    RoomSimulator(rt60_range=(0.1, 0.6), wet_range=(0.2, 0.7), p=0.5),
    AddNoise(snr_db=(15, 35), noise_type="pink", p=0.6),
], seed=42)

audio_aug, sr = pipeline(audio, sr)
```

## CLI Usage

```bash
# Time-stretch to 150%
pvx voc input.wav --stretch 1.5 --output stretched.wav

# Pitch-shift up 2 semitones
pvx voc input.wav --pitch 2 --output pitched.wav

# Room simulation with curated IR database
pvxrir speech.wav --ir-database echothief --category hall --output reverbed.wav
```

## Find what you need

Before diving into the long generated references, check the short contract docs first:

- [Supported Surface](SUPPORTED_SURFACE.md)
- [Getting Started](GETTING_STARTED.md)
- [Quality Guide](QUALITY_GUIDE.md)
- [Homebrew Install](HOMEBREW.md)

**I want to…**

| Goal | Start here |
| --- | --- |
| …install pvx and run my first command | [Getting Started](GETTING_STARTED.md) |
| …augment audio for ML training (PyTorch/HF/TF) | [ML Integration Guide](ML_INTEGRATION.md) |
| …pick a recipe for ASR / music / contrastive SSL | [Augmentation Cookbook](AUGMENTATION_COOKBOOK.md) |
| …run GPU-batched augmentation | [Pipeline Cookbook](PIPELINE_COOKBOOK.md) |
| …look up a CLI flag | [CLI Flags Reference](CLI_FLAGS_REFERENCE.md) |
| …understand phase-vocoder quality trade-offs | [Quality Guide](QUALITY_GUIDE.md) |
| …read the math behind the STFT core | [Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md) |
| …understand the system design | [Architecture](ARCHITECTURE.md) |
| …write a custom transform plugin | [Contributing](../CONTRIBUTING.md) |
| …browse the Python API | [API Reference](api/augment.md) |

New users: read **Getting Started** first. Reach for the generated references only when you need exhaustive flag or file inventory.
