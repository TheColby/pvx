<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# ML/DL Framework Integration Guide

> **⚠ Alpha release (0.1.0a1)** — `pvx.augment` is under active development.
> Public interfaces may change between minor versions until 1.0.
> Pin your dependency to an exact version (`pvx==0.1.0a1`) if you need
> stability, and please report issues on GitHub.

`pvx` ships a first-class Python API for audio data augmentation (`pvx.augment`) and thin,
zero-boilerplate adapters for the three most popular deep-learning ecosystems.

---

## Contents

1. [Installation](#installation)
2. [Quick Start — Pure Python](#quick-start--pure-python)
3. [PyTorch Integration](#pytorch-integration)
4. [HuggingFace Datasets Integration](#huggingface-datasets-integration)
5. [TensorFlow / tf.data Integration](#tensorflow--tfdata-integration)
6. [Determinism and Reproducibility](#determinism-and-reproducibility)
7. [Performance Tips](#performance-tips)
8. [GPU-Accelerated Transforms (PyTorch)](#gpu-accelerated-transforms-pytorch)
9. [Available Transforms](#available-transforms)

---

## Installation

```bash
# Core DSP + augmentation (no ML framework required)
pip install pvx

# With PyTorch
pip install "pvx[torch]"

# With HuggingFace
pip install "pvx[huggingface]"

# With TensorFlow
pip install "pvx[tensorflow]"

# All ML extras
pip install "pvx[ml]"
```

---

## Quick Start — Pure Python

```python
import soundfile as sf
from pvx.augment import (
    Pipeline,
    GainPerturber,
    RoomSimulator,
    AddNoise,
    SpecAugment,
    CodecDegradation,
)

# Load audio (any format supported by soundfile)
audio, sr = sf.read("speech.wav", dtype="float32", always_2d=False)

# Build an augmentation pipeline
pipeline = Pipeline([
    GainPerturber(gain_db=(-3, 3), p=0.8),
    RoomSimulator(rt60_range=(0.05, 0.8), wet_range=(0.1, 0.5), p=0.4),
    AddNoise(snr_db=(15, 35), noise_type="pink", p=0.5),
    CodecDegradation(codec="random", p=0.2),
    SpecAugment(freq_mask_param=27, time_mask_param=100, p=0.5),
], seed=42)

# Apply with a reproducible seed
audio_aug, sr_out = pipeline(audio, sr, seed=0)

# Save result
sf.write("speech_aug.wav", audio_aug, sr_out)
```

---

## Intent Preset Pipelines

Ready-made pipelines for common research use cases:

```python
from pvx.augment import asr_pipeline, music_pipeline, speech_enhancement_pipeline, contrastive_pipeline

# ASR robustness training
pipeline = asr_pipeline(seed=42)

# Music information retrieval
pipeline = music_pipeline(seed=42)

# Speech enhancement (noisy/reverberant input, clean target)
noisy_pipeline = speech_enhancement_pipeline(seed=42)

# Contrastive self-supervised learning (two correlated views)
pipeline_a, pipeline_b = contrastive_pipeline(seed=42)
clean_audio, sr = sf.read("speech.wav", dtype="float32")
view_a, _ = pipeline_a(clean_audio, sr, seed=idx)
view_b, _ = pipeline_b(clean_audio, sr, seed=idx)
```

---

## PyTorch Integration

### PvxAugmentDataset

```python
import torch
from torch.utils.data import DataLoader
from pvx.augment import Pipeline, AddNoise, RoomSimulator, SpecAugment
from pvx.integrations.pytorch import PvxAugmentDataset

pipeline = Pipeline([
    AddNoise(snr_db=(10, 30), p=0.5),
    RoomSimulator(rt60_range=(0.1, 0.8), p=0.4),
    SpecAugment(freq_mask_param=30, time_mask_param=50, p=0.5),
], seed=42)

file_list = ["data/train/speech_001.wav", "data/train/speech_002.wav", ...]
labels    = [0, 1, ...]  # optional

dataset = PvxAugmentDataset(
    file_list,
    pipeline=pipeline,
    sample_rate=16000,
    labels=labels,
    duration_s=3.0,       # crop/pad all clips to 3 seconds
    mono=True,
)

loader = DataLoader(
    dataset.as_torch_dataset(),
    batch_size=32,
    num_workers=4,
    collate_fn=PvxAugmentDataset.collate_fn,
    pin_memory=True,
)

for batch in loader:
    audio  = batch["audio"]   # (B, T) or (B, C, T)
    label  = batch["label"]   # (B,)
    lengths = batch["lengths"] # (B,) — true clip lengths before padding
    # ... training step ...
```

### PvxAugmentTransform (torchvision-style)

Use as a drop-in transform in any torchaudio-compatible dataset:

```python
from pvx.integrations.pytorch import PvxAugmentTransform

transform = PvxAugmentTransform(pipeline, sample_rate=16000)

# Use with torchaudio
import torchaudio
ds = torchaudio.datasets.LIBRISPEECH(
    root="data/", url="train-clean-100",
    download=True,
)
# Apply transform manually:
audio, sr, transcript, *_ = ds[0]
audio_aug = transform(audio)
```

### AudioCollator for variable-length batches

```python
from pvx.integrations.pytorch import AudioCollator

loader = DataLoader(
    dataset.as_torch_dataset(),
    batch_size=16,
    collate_fn=AudioCollator(return_lengths=True),
)

for audio_batch, lengths in loader:
    # audio_batch: (B, C, T_max) padded to longest clip
    # lengths: (B,) true lengths
    pass
```

---

## HuggingFace Datasets Integration

### Basic map() usage

```python
from datasets import load_dataset
from pvx.augment import Pipeline, AddNoise, RoomSimulator
from pvx.integrations.huggingface import make_augment_map_fn

ds = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train")

pipeline = Pipeline([
    AddNoise(snr_db=(10, 30), p=0.5),
    RoomSimulator(rt60_range=(0.1, 0.8), p=0.4),
], seed=42)

# HuggingFace Audio feature rows have format: {"array": np.ndarray, "sampling_rate": int}
augment_fn = make_augment_map_fn(
    pipeline,
    audio_column="audio",
    output_column="audio_aug",  # write to new column (keeps original)
    base_seed=42,
)

ds_aug = ds.map(
    augment_fn,
    batched=False,
    with_indices=True,  # required for per-row seed
    num_proc=8,
)
```

### Contrastive view generation (SSL)

```python
from pvx.integrations.huggingface import HFAugmentMapper
from pvx.augment import contrastive_pipeline

pipeline_a, pipeline_b = contrastive_pipeline(seed=42)

# Generate both views in a single map pass
mapper_a = HFAugmentMapper(pipeline_a, audio_column="audio", output_prefix="view_a")
mapper_b = HFAugmentMapper(pipeline_b, audio_column="audio", output_prefix="view_b")

# Or use built-in pair mode
from pvx.augment import music_pipeline
mapper = HFAugmentMapper(
    music_pipeline(seed=42),
    audio_column="audio",
    generate_pair=True,
    output_prefix="view",
)
ds_ssl = ds.map(mapper, batched=False, with_indices=True)
# ds_ssl["view_a"]["array"], ds_ssl["view_b"]["array"]
```

### One-liner convenience

```python
from pvx.integrations.huggingface import augment_dataset

ds_aug = augment_dataset(ds, pipeline, audio_column="audio", num_proc=8)
```

### Saving augmented datasets

```python
# Save to disk for fast reloading
ds_aug.save_to_disk("data/augmented_librispeech")

# Or push to HuggingFace Hub
ds_aug.push_to_hub("my-org/librispeech-augmented")
```

---

## TensorFlow / tf.data Integration

### Tensor map function

```python
import tensorflow as tf
from pvx.augment import Pipeline, AddNoise, RoomSimulator
from pvx.integrations.tensorflow import make_tf_augment_fn, pvx_augment_tf_dataset

pipeline = Pipeline([
    AddNoise(snr_db=(10, 30), p=0.5),
    RoomSimulator(rt60_range=(0.1, 0.8), p=0.4),
], seed=42)

# From raw tensor dataset
ds = tf.data.Dataset.from_tensor_slices(audio_array)  # shape: (N, T)
augment_fn = make_tf_augment_fn(pipeline, sample_rate=16000)
ds_aug = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

# From dict dataset (e.g. TF-Datasets)
import tensorflow_datasets as tfds
tfds_raw = tfds.load("speech_commands", split="train")
ds_aug = pvx_augment_tf_dataset(tfds_raw, pipeline, audio_key="audio")
```

### Dict-based map for keyed datasets

```python
from pvx.integrations.tensorflow import make_tf_augment_map_fn

fn = make_tf_augment_map_fn(
    pipeline,
    audio_key="audio",
    output_key="audio_aug",
    default_sr=16000,
)
ds_aug = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)
```

---

## Determinism and Reproducibility

All pvx transforms accept a `seed` parameter that fully determines their output.
The seed hierarchy is:

```
global_seed + item_index  →  per-item seed
per-item seed + transform_index  →  per-transform seed
```

This means:
- Same `seed` + same input → identical output (byte-for-byte reproducible)
- Different item indices produce different augmentations even with the same global seed
- Validation sets can be deterministically generated by fixing the seed

```python
# Training: random seed per epoch for variety
audio_aug, _ = pipeline(audio, sr, seed=epoch * n_items + item_idx)

# Validation: fixed seed for reproducible evaluation
audio_aug, _ = pipeline(audio, sr, seed=99999 + item_idx)
```

---

## Performance Tips

### Offline pre-computation (recommended for large datasets)

```bash
# Generate the augmented dataset once and save it
pvx augment data/train/*.wav \
  --output-dir data/train_aug \
  --variants-per-input 4 \
  --intent asr_robust \
  --seed 1337 \
  --workers 16
```

Then load pre-augmented files directly — eliminates augmentation overhead at training time.

### Online augmentation with worker processes

```python
# PyTorch: set num_workers >= 4 — each worker gets a separate process
loader = DataLoader(dataset.as_torch_dataset(), num_workers=8, ...)

# HuggingFace: use num_proc
ds_aug = ds.map(augment_fn, num_proc=8)

# TensorFlow: use AUTOTUNE
ds_aug = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)
```

### Caching expensive transforms

For transforms like `RoomSimulator` that are computationally intensive, cache
intermediate results:

```python
# HuggingFace — cache augmented dataset after first run
ds_aug = ds.map(augment_fn, num_proc=8, cache_file_name="data/cache/aug.arrow")
```

### TimeStretch and PitchShift — Engine Selection

`TimeStretch` and `PitchShift` support five processing engines via the
`engine` parameter:

| Engine | Requires | Notes |
|---|---|---|
| `"auto"` (default) | — | Prefers torchaudio > pytorch > pvx-cli |
| `"torchaudio"` | `pip install "pvx[torch]"` | Optimized C++ phase-vocoder kernel via torchaudio |
| `"pytorch"` | `pip install "pvx[torch]"` | Vectorized phase-vocoder in pure PyTorch |
| `"wavelet"` | built-in Haar; optional `pip install PyWavelets` | Experimental multiband wavelet backend |
| `"pvx-cli"` | pvx CLI on `PATH` | Full pvxvoc DSP stack via subprocess |

```python
from pvx.augment import TimeStretch, PitchShift

# Auto-select best available engine (recommended)
ts = TimeStretch(rate=(0.8, 1.2), engine="auto")

# Force torchaudio engine (fastest, requires torchaudio)
ts = TimeStretch(rate=(0.8, 1.2), engine="torchaudio")

# Force PyTorch engine (no torchaudio needed, just torch)
ts = TimeStretch(rate=(0.8, 1.2), engine="pytorch")

# Force experimental wavelet engine.
# wavelet="auto" uses PyWavelets db4 when available, otherwise built-in Haar.
ts = TimeStretch(rate=(0.8, 1.2), engine="wavelet", wavelet="auto", wavelet_levels=3)
ps = PitchShift(semitones=(-2, 2), engine="wavelet", wavelet="haar", wavelet_levels=4)

# Force pvx CLI subprocess (full DSP stack: transients, formants, stereo)
ts = TimeStretch(rate=(0.8, 1.2), engine="pvx-cli")
```

The CLI `pvx augment` command also accepts `--engine`:

```bash
pvx augment data/train/*.wav \
  --output-dir data/train_aug \
  --engine torchaudio \
  --intent asr_robust
```

To compare backend speed and simple pitch/duration smoke metrics:

```bash
python benchmarks/run_backend_compare.py --engines pytorch,pvx-cli,wavelet --wavelet auto
```

If no engine is available, `TimeStretch` and `PitchShift` will raise a
`RuntimeError` with a clear message — they will **not** silently return
unmodified audio.

For a lightweight, pure-Python pitch shift that requires no external engine,
use `PitchShiftSimple` (spectral bin interpolation, ±1–3 semitone
approximation).

### Streaming Augmentation for Long-Form Audio

For podcasts, audiobooks, and other long-form content, use the streaming
API to process files in chunks with bounded memory:

```python
from pvx.augment import stream_augment_file, Pipeline, AddNoise, GainPerturber

pipeline = Pipeline([
    GainPerturber(gain_db=(-3, 3), p=0.8),
    AddNoise(snr_db=(15, 35), noise_type="pink", p=0.5),
], seed=42)

# Process a 2-hour podcast in 30-second chunks
stream_augment_file(
    "podcast_2h.wav",
    "podcast_2h_aug.wav",
    pipeline=pipeline,
    chunk_duration_s=30.0,
    seed=42,
)
```

### Impulse Response Database

For physics-based room acoustics, use real impulse responses instead of
synthetic approximations:

```python
from pvx.augment import IRDatabase, ImpulseResponseConvolver

db = IRDatabase()
db.download("echothief")  # 115 real-world IRs, cached locally

# Use all IRs
aug = ImpulseResponseConvolver(db.ir_dir("echothief"), wet_range=(0.4, 1.0))

# Or filter by category
halls = db.filter("echothief", category="hall")
aug = ImpulseResponseConvolver(halls, wet_range=(0.5, 0.9))
```

### Mix online and offline augmentation

Use the CLI for heavy transforms (time-stretch, pitch-shift) and the Python API
for lightweight ones (gain, noise, SpecAugment) at runtime:

```python
# Heavy transforms pre-computed offline
ds_heavy = load_from_disk("data/train_heavy_aug")

# Lightweight transforms applied online
light_pipeline = Pipeline([
    GainPerturber(gain_db=(-3, 3), p=0.8),
    AddNoise(snr_db=(15, 35), noise_type="white", p=0.4),
    SpecAugment(freq_mask_param=27, time_mask_param=100, p=0.5),
], seed=42)

fn = make_augment_map_fn(light_pipeline, audio_column="audio")
ds_final = ds_heavy.map(fn, batched=False, with_indices=True, num_proc=4)
```

---

## GPU-Accelerated Transforms (PyTorch)

For maximum throughput during GPU training, `pvx.augment.gpu` provides native
PyTorch transforms that operate directly on `torch.Tensor` objects — no NumPy
round-tripping required. These support batched operation and run on CUDA, MPS,
or CPU.

```python
import torch
from pvx.augment.gpu import (
    TorchPipeline,
    TorchGainPerturber,
    TorchAddNoise,
    TorchEQPerturber,
    TorchSpecAugment,
    TorchNormalizer,
    TorchClippingSimulator,
    TorchTimeStretch,
    TorchPitchShift,
    TorchRoomSimulator,
    TorchMixup,
    NumpyTransformAdapter,  # wrap any NumPy transform for GPU use
)

# Build a GPU-native pipeline
gpu_pipeline = TorchPipeline([
    TorchGainPerturber(gain_db=(-6, 6), p=0.8),
    TorchAddNoise(snr_db=(10, 30), noise_type="pink", p=0.5),
    TorchEQPerturber(n_bands=4, gain_db_range=(-6, 6), p=0.4),
    TorchSpecAugment(freq_mask_param=27, time_mask_param=100, p=0.5),
], seed=42)

# Apply to a batch of tensors (B, C, T)
audio_batch = torch.randn(32, 1, 48000).cuda()
audio_aug = gpu_pipeline(audio_batch, sr=16000)
```

### Mixing GPU and NumPy transforms

Use `NumpyTransformAdapter` to wrap any NumPy-based pvx transform for use in a
`TorchPipeline`. The adapter moves data to CPU per-sample, so native `Torch*`
transforms are preferred for the hot path:

```python
from pvx.augment import CodecDegradation

gpu_pipeline = TorchPipeline([
    TorchGainPerturber(gain_db=(-6, 6)),
    NumpyTransformAdapter(CodecDegradation(codec="random", p=0.2)),
    TorchSpecAugment(freq_mask_param=27),
])
```

### Available GPU transforms

| Transform | Equivalent NumPy transform | Notes |
|---|---|---|
| `TorchGainPerturber` | `GainPerturber` | Batched random gain |
| `TorchAddNoise` | `AddNoise` | white/pink/brown via FFT on GPU |
| `TorchEQPerturber` | `EQPerturber` | STFT-domain parametric EQ |
| `TorchSpecAugment` | `SpecAugment` | Frequency + time masking |
| `TorchNormalizer` | `Normalizer` | Peak/RMS normalization |
| `TorchClippingSimulator` | `ClippingSimulator` | Hard/soft clipping |
| `TorchTimeStretch` | `TimeStretch` | Phase-vocoder time stretch on GPU |
| `TorchPitchShift` | `PitchShift` | Pitch shift via stretch + resample |
| `TorchRoomSimulator` | `RoomSimulator` | Synthetic reverb approximation |
| `TorchMixup` | — | Zhang et al. 2018 batch mixing |
| `NumpyTransformAdapter` | any `Transform` | CPU fallback wrapper |

---

## Available Transforms

| Transform | Module | Key Parameters | Use Case |
|---|---|---|---|
| `AddNoise` | `noise` | `snr_db`, `noise_type` | ASR, VAD, denoising |
| `BackgroundMixer` | `noise` | `background_sources`, `snr_db` | Real-world noise robustness |
| `ImpulseNoise` | `noise` | `rate`, `amplitude_range` | Transmission artifact simulation |
| `RoomSimulator` | `room` | `rt60_range`, `wet_range` | Synthetic reverb approximation for augmentation |
| `ImpulseResponseConvolver` | `room` | `ir_sources`, `wet_range` | Real IR convolution |
| `CodecDegradation` | `codec` | `codec` | Codec-robust ASR, telephony |
| `BitCrusher` | `codec` | `bits` | Lo-fi, retro effects |
| `BandwidthLimiter` | `codec` | `cutoff_hz` | Telephone/radio simulation |
| `SpecAugment` | `spectral` | `freq_mask_param`, `time_mask_param` | ASR, keyword spotting |
| `EQPerturber` | `spectral` | `n_bands`, `gain_db_range` | Microphone/room coloration |
| `SpectralNoise` | `spectral` | `noise_std_range` | Feature extractor robustness |
| `GainPerturber` | `time_domain` | `gain_db` | Loudness invariance |
| `Normalizer` | `time_domain` | `mode`, `target_db` | Pre-processing |
| `ClippingSimulator` | `time_domain` | `percentile`, `mode` | ADC artifact simulation |
| `TimeShift` | `time_domain` | `shift` | Temporal invariance |
| `Reverse` | `time_domain` | — | Data doubling |
| `Fade` | `time_domain` | `fade_in`, `fade_out` | Boundary artifact reduction |
| `TrimSilence` | `time_domain` | `threshold_db` | Pre-processing |
| `FixedLengthCrop` | `time_domain` | `duration_s` | Batching fixed-size inputs |
| `TimeStretch` | `time_domain` | `rate`, `preset`, `engine`, `wavelet`, `wavelet_levels` | Tempo invariance (auto/torchaudio/pytorch/wavelet/pvx-cli) |
| `PitchShift` | `time_domain` | `semitones`, `formant_mode`, `engine`, `wavelet`, `wavelet_levels` | Pitch invariance (auto/torchaudio/pytorch/wavelet/pvx-cli) |
| `stream_augment_file` | `streaming` | `chunk_duration_s`, `overlap_s` | Memory-bounded long-form processing |
| `IRDatabase` | `ir_database` | `cache_dir` | Real IR collection manager |
| `PitchShiftSimple` | `spectral` | `semitones` | Lightweight pitch approximation |
| `Pipeline` | `core` | `transforms`, `seed` | Sequential composition |
| `OneOf` | `core` | `transforms`, `weights` | Random selection |
| `SomeOf` | `core` | `transforms`, `k` | Random subset |
| `RandomApply` | `core` | `transform`, `p` | Stochastic application |

## Attribution
