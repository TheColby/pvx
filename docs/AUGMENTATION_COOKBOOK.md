<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# Audio Augmentation Cookbook

> **⚠ Alpha release (0.1.0a1)** — These recipes use `pvx.augment`, which is
> under active development.  APIs may change before 1.0.

Real-world augmentation recipes for common deep-learning audio tasks.
Each recipe includes the rationale, a pipeline configuration, and CLI equivalents.

---

## Contents

1. [ASR — Automatic Speech Recognition](#1-asr--automatic-speech-recognition)
2. [Keyword Spotting / Wake-Word Detection](#2-keyword-spotting--wake-word-detection)
3. [Speaker Verification / Identification](#3-speaker-verification--identification)
4. [Speech Enhancement and Denoising](#4-speech-enhancement-and-denoising)
5. [Music Classification and Tagging](#5-music-classification-and-tagging)
6. [Beat Tracking and Tempo Estimation](#6-beat-tracking-and-tempo-estimation)
7. [Pitch Estimation and Melody Extraction](#7-pitch-estimation-and-melody-extraction)
8. [Audio Event Detection / Sound Classification](#8-audio-event-detection--sound-classification)
9. [Self-Supervised / Contrastive Learning (SimCLR, MoCo, BYOL)](#9-self-supervised--contrastive-learning)
10. [Music Source Separation](#10-music-source-separation)

---

## 1. ASR — Automatic Speech Recognition

**Goal:** Train models robust to microphone quality, background noise, room acoustics,
and channel distortion.

**Augmentation rationale:**
- Room simulation: phones and meetings happen in reverberant spaces
- Background noise: cafes, offices, street noise are the real world
- Codec degradation: telephone and VoIP channels are common deployment targets
- Gain perturbation: microphone levels vary across recordings
- SpecAugment: proven to reduce WER on LibriSpeech, Common Voice

```python
from pvx.augment import Pipeline, GainPerturber, RoomSimulator, AddNoise, CodecDegradation, SpecAugment

asr_pipeline = Pipeline([
    GainPerturber(gain_db=(-6, 6), p=0.9),
    RoomSimulator(rt60_range=(0.05, 1.0), wet_range=(0.1, 0.7), p=0.5),
    AddNoise(snr_db=(5, 35), noise_type="pink", p=0.6),
    CodecDegradation(codec="random", p=0.2),
    SpecAugment(
        freq_mask_param=27,
        time_mask_param=100,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.8,
    ),
], seed=42)
```

**CLI:**
```bash
# Step 1: Heavy offline augmentation (time-stretch + pitch)
pvx augment data/train/*.wav \
  --output-dir data/train_aug \
  --variants-per-input 3 \
  --intent asr_robust \
  --engine auto \
  --seed 1337 \
  --workers 16

# Step 2: Add noise per file
for f in data/train_aug/*.wav; do
  pvxnoise "$f" --snr 5,30 --noise-type pink --output "$f"
done

# Step 3: Simulate codec degradation (optional)
for f in data/train_aug/*.wav; do
  pvxcodec "$f" --codec random --seed "$RANDOM" --output "$f"
done
```

**Note:** SpecAugment has been shown to improve WER in the original
[Park et al. 2019](https://arxiv.org/abs/1904.08779) paper. The impact of
the full pipeline above will depend on your model, data, and evaluation
conditions — we recommend benchmarking on your own held-out test set.

---

## 2. Keyword Spotting / Wake-Word Detection

**Goal:** Detect specific words under highly diverse acoustic conditions.

**Key considerations:**
- Label-preserving augmentation only — semantic content must be unchanged
- Aggressive time-stretch causes word timing drift; keep it mild
- Heavy SpecAugment helps greatly since models often rely on a few spectral cues

```python
from pvx.augment import Pipeline, GainPerturber, AddNoise, CodecDegradation, SpecAugment, TimeShift, Fade

kws_pipeline = Pipeline([
    GainPerturber(gain_db=(-6, 6), p=0.9),
    Fade(fade_in=(0.001, 0.02), fade_out=(0.001, 0.02), p=0.4),
    TimeShift(shift=(-0.05, 0.05), p=0.3),
    AddNoise(snr_db=(5, 30), noise_type="pink", p=0.7),
    CodecDegradation(codec="random", p=0.3),
    SpecAugment(
        freq_mask_param=30,
        time_mask_param=25,   # smaller T — keyword clips are short
        num_freq_masks=2,
        num_time_masks=2,
        p=0.7,
    ),
], seed=42)
```

**CLI:**
```bash
pvx augment keywords/*.wav \
  --output-dir keywords_aug \
  --variants-per-input 5 \
  --intent asr_robust \
  --label-policy preserve \   # keeps pitch/time perturbation mild
  --seed 7
```

---

## 3. Speaker Verification / Identification

**Goal:** Train embeddings invariant to microphone, room, and channel effects
but discriminative across speakers.

**Important:** Pitch perturbation should be minimal (large shifts can change perceived
speaker identity and confuse the model during training).

```python
from pvx.augment import Pipeline, GainPerturber, RoomSimulator, AddNoise, CodecDegradation, EQPerturber

sv_pipeline = Pipeline([
    GainPerturber(gain_db=(-6, 6), p=0.9),
    EQPerturber(n_bands=4, gain_db_range=(-8, 8), p=0.6),   # microphone colouration
    RoomSimulator(rt60_range=(0.05, 1.5), wet_range=(0.1, 0.8), p=0.6),
    AddNoise(snr_db=(10, 40), noise_type="pink", p=0.5),
    CodecDegradation(codec="random", p=0.25),
], seed=42)
```

**Contrastive learning setup (ECAPA-TDNN / ResNet style):**
```python
from pvx.augment import contrastive_pipeline

pipeline_a, pipeline_b = contrastive_pipeline(seed=42)

for audio, speaker_id in dataloader:
    view_a, _ = pipeline_a(audio, sr, seed=item_idx)
    view_b, _ = pipeline_b(audio, sr, seed=item_idx)
    loss = nt_xent_loss(embed(view_a), embed(view_b), labels=speaker_id)
```

---

## 4. Speech Enhancement and Denoising

**Goal:** Train models to map noisy/reverberant speech → clean speech.
Augmentation creates the *noisy input*; the clean original is the *target*.

```python
from pvx.augment import Pipeline, GainPerturber, RoomSimulator, AddNoise, ImpulseNoise, CodecDegradation, OneOf, Identity

# This pipeline degrades clean speech to create training pairs
degradation_pipeline = Pipeline([
    GainPerturber(gain_db=(-6, 6), p=1.0),
    RoomSimulator(rt60_range=(0.1, 2.5), wet_range=(0.2, 0.95), p=0.7),
    AddNoise(snr_db=(0, 20), noise_type="pink", p=0.8),
    OneOf([
        CodecDegradation(codec="voip_narrow"),
        CodecDegradation(codec="telephone"),
        Identity(),
    ], p=0.4),
    ImpulseNoise(rate=1.0, amplitude_range=(0.01, 0.1), p=0.1),
], seed=42)

# During training:
for clean_audio, sr in clean_dataset:
    noisy_audio, _ = degradation_pipeline(clean_audio, sr, seed=item_idx)
    predicted_clean = model(noisy_audio)
    loss = si_snr_loss(predicted_clean, clean_audio)
```

**Data generation CLI:**
```bash
# Parallel degradation of a clean corpus
pvxrir clean/*.wav --rt60 0.1,2.0 --wet 0.3,0.9 --output-dir reverberant/
pvxnoise reverberant/*.wav --snr 0,20 --noise-type pink --output-dir noisy/
pvxcodec noisy/*.wav --codec random --output-dir degraded/
```

---

## 5. Music Classification and Tagging

**Goal:** Train genre, mood, instrument, and tag classifiers robust to
diverse recording conditions and production styles.

```python
from pvx.augment import Pipeline, GainPerturber, EQPerturber, RoomSimulator, AddNoise, SpecAugment, OneOf, Identity, ImpulseNoise

music_cls_pipeline = Pipeline([
    GainPerturber(gain_db=(-6, 6), p=0.9),
    EQPerturber(n_bands=5, gain_db_range=(-8, 8), p=0.6),
    RoomSimulator(rt60_range=(0.1, 1.0), wet_range=(0.05, 0.3), p=0.3),
    OneOf([
        AddNoise(snr_db=(25, 45), noise_type="white"),
        ImpulseNoise(rate=0.5),
        Identity(),
    ], p=0.3),
    SpecAugment(
        freq_mask_param=30,
        time_mask_param=80,
        num_freq_masks=2,
        num_time_masks=2,
        p=0.5,
    ),
], seed=42)
```

**Offline augmentation with pvx augment:**
```bash
pvx augment music/*.wav \
  --output-dir music_aug \
  --variants-per-input 4 \
  --intent mir_music \
  --seed 42 \
  --workers 12
```

---

## 6. Beat Tracking and Tempo Estimation

**Goal:** Train tempo-invariant beat trackers. Time-stretch is the most critical augmentation.

**Important:** Time-stretch must preserve beat structure.  Use pvx's drum-safe preset
which uses hybrid transient mode to keep kick/snare clicks intact.

```python
from pvx.augment import Pipeline, GainPerturber, AddNoise, SpecAugment, TimeStretch

beat_pipeline = Pipeline([
    TimeStretch(
        rate=(0.8, 1.25),
        preserve_pitch=True,
        preset="drums_safe",
        engine="auto",   # uses PyTorch when available, else pvx CLI
        p=0.8,
    ),
    GainPerturber(gain_db=(-4, 4), p=0.8),
    AddNoise(snr_db=(25, 45), noise_type="white", p=0.3),
], seed=42)
```

**CLI — offline generation across a tempo range:**
```bash
for rate in 0.80 0.90 1.00 1.10 1.20; do
  pvxvoc drums.wav \
    --stretch $rate \
    --preset drums_safe \
    --transient-mode wsola \
    --output "drums_${rate}.wav"
done
```

---

## 7. Pitch Estimation and Melody Extraction

**Goal:** Train f0 estimators robust to background noise, vibrato, and pitch transpositions.

```python
from pvx.augment import Pipeline, GainPerturber, AddNoise, EQPerturber

pitch_pipeline = Pipeline([
    GainPerturber(gain_db=(-4, 4), p=0.8),
    EQPerturber(n_bands=4, gain_db_range=(-6, 6), p=0.5),
    AddNoise(snr_db=(15, 40), noise_type="pink", p=0.6),
], seed=42)

# For pitch shift augmentation use pvx voc (label must shift with pitch)
# pvxvoc vocal.wav --stretch 1.0 --pitch 3 --output vocal_up3.wav
```

**Offline pitch-shift augmentation with label tracking:**
```bash
for semitones in -4 -2 0 2 4; do
  pvxvoc vocal.wav \
    --stretch 1.0 \
    --pitch $semitones \
    --preset vocal_studio \
    --output "vocal_pitch${semitones}.wav"
  echo "$semitones" > "vocal_pitch${semitones}_label.txt"
done
```

---

## 8. Audio Event Detection / Sound Classification

**Goal:** Classify acoustic events (dogs barking, glass breaking, gunshots, etc.)
in diverse background conditions.

```python
from pvx.augment import (
    Pipeline, GainPerturber, AddNoise, BackgroundMixer,
    RoomSimulator, SpecAugment, Fade, TimeShift
)

sed_pipeline = Pipeline([
    GainPerturber(gain_db=(-6, 6), p=0.9),
    Fade(fade_in=(0.0, 0.1), fade_out=(0.0, 0.1), p=0.3),
    TimeShift(shift=(-0.2, 0.2), p=0.4),
    BackgroundMixer("backgrounds/", snr_db=(5, 20), p=0.6),
    RoomSimulator(rt60_range=(0.1, 1.5), wet_range=(0.1, 0.6), p=0.4),
    SpecAugment(freq_mask_param=30, time_mask_param=60, p=0.5),
], seed=42)
```

---

## 9. Self-Supervised / Contrastive Learning

**Goal:** Learn audio representations without labels using contrastive objectives
(SimCLR, MoCo, BYOL, VICReg, data2vec).

Two independently augmented views of the same clip form a positive pair.

```python
from pvx.augment import contrastive_pipeline
from pvx.integrations.huggingface import HFAugmentMapper

# Two strongly-correlated but statistically independent views
pipeline_a, pipeline_b = contrastive_pipeline(seed=42)

# HuggingFace
mapper = HFAugmentMapper(
    pipeline_a,   # pipeline_b handled internally via generate_pair=True
    generate_pair=True,
    output_prefix="view",
)
ds_ssl = ds.map(mapper, batched=False, with_indices=True, num_proc=8)
# ds_ssl columns: audio, view_a, view_b

# PyTorch training loop
for batch in loader:
    view_a = batch["view_a"]  # (B, T)
    view_b = batch["view_b"]  # (B, T)
    z_a = encoder(view_a)
    z_b = encoder(view_b)
    loss = nt_xent(z_a, z_b)
```

**Recommended SSL augmentation strength:**
- `p` values between 0.4–0.7 per transform
- Include at least: gain, room simulation, noise, SpecAugment
- Avoid transforms that destroy too much semantic content (extreme bit-crush, strong clipping)
- Two views should be recognizably related but perceptually distinct

---

## 10. Music Source Separation

**Goal:** Separate music into stems (vocals, drums, bass, other).
Augmentation creates harder mixtures and prevents over-fitting to recording conditions.

```python
from pvx.augment import Pipeline, GainPerturber, EQPerturber, AddNoise

# Per-stem augmentation (applied before mixing)
stem_pipeline = Pipeline([
    GainPerturber(gain_db=(-3, 3), p=0.9),
    EQPerturber(n_bands=3, gain_db_range=(-4, 4), p=0.5),
], seed=42)

# During training — augment each stem independently then remix
for vocal, drums, bass, other in stems:
    vocal_aug, _ = stem_pipeline(vocal, sr, seed=item_idx * 4 + 0)
    drums_aug, _ = stem_pipeline(drums, sr, seed=item_idx * 4 + 1)
    bass_aug, _ = stem_pipeline(bass, sr, seed=item_idx * 4 + 2)
    other_aug, _ = stem_pipeline(other, sr, seed=item_idx * 4 + 3)
    mixture = vocal_aug + drums_aug + bass_aug + other_aug
    predicted_stems = separator(mixture)
    loss = si_snr_loss(predicted_stems, [vocal_aug, drums_aug, bass_aug, other_aug])
```

---

## Augmentation Intensity Guide

| Strength | Stretch | Pitch | SNR | RT60 | Notes |
|---|---|---|---|---|---|
| Very mild (preserve label) | 0.97–1.03 | ±0.5 st | > 30 dB | < 0.3 s | KWS preserve mode |
| Mild | 0.92–1.12 | ±1.5 st | 20–35 dB | 0.1–0.6 s | ASR, SV |
| Moderate | 0.85–1.25 | ±3.0 st | 10–30 dB | 0.2–1.5 s | Music tagging, classification |
| Strong | 0.75–1.40 | ±5.0 st | 5–25 dB | 0.3–2.5 s | SSL, enhancement |
| Extreme | 0.50–2.00 | ±7.0 st | 0–15 dB | 0.5–4.0 s | Robustness stress tests |

## Attribution
