<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# pvx Benchmarks

_Generated from commit `[COMMIT_HASH]` (commit date: [COMMIT_DATE])._

Reproducible benchmark summary for core short-time Fourier transform/inverse short-time Fourier transform (STFT/ISTFT) path across central processing unit/Compute Unified Device Architecture/Apple-Silicon-native contexts.

## Quick Setup (Install + PATH)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
pvx --help
```

If `pvx` is not found, add the virtualenv binaries to your shell path (`zsh`):

```bash
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$(pwd)" >> ~/.zshrc
source ~/.zshrc
pvx --help
```

## Reproduce

```bash
python3 scripts/scripts_generate_docs_extras.py --run-benchmarks
```

## Benchmark Spec

- Sample rate: `48000` Hz
- Duration: `4.0` s
- Signal suite size: `4`
- Signal recipe: deterministic synthetic suite: tonal, speech-like, transient stereo, chirp/texture stereo
- STFT config: `n_fft=2048`, `win_length=2048`, `hop_size=512`, `window=hann`, `center=True`

| Signal | Channels | Duration (s) | Description |
| --- | ---: | ---: | --- |
| `tonal_ramp_mono` | 1 | 4.000 | Deterministic harmonic blend with slow ramp. |
| `speech_like_mono` | 1 | 4.000 | Speech-like harmonic formants with sparse glottal-like impulses. |
| `transient_drums_stereo` | 2 | 4.000 | Percussive burst train for onset retention and stereo timing drift. |
| `chirp_texture_stereo` | 2 | 4.000 | Wideband chirp-plus-texture to stress spectral and phase stability. |

## Host

- Platform: `macOS-15.1.1-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Python: `3.14.3`

## Backend Summary

| Backend | Status | Cases (ok/total) | Mean xRT | Mean elapsed (ms) | Peak host memory (MB) | Mean quality score (/100) | Mean SNR in (dB) | Mean SI-SDR (dB) | Mean LSD (dB) | Mean ModSpec | Mean Smear | Mean EnvCorr | Mean Coherence Drift | Mean Phasiness | Mean Musical Noise | Mean SNR vs CPU (dB) | Mean LSD vs CPU (dB) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpu | ok | 4/4 | 146.39 | 31.02 | 17.70 | 97.50 | 100.980 | 153.813 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0511 | 0.0000 | n/a | n/a |  |
| cuda | unavailable | 0/4 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package. |
| apple_silicon_native_cpu | ok | 4/4 | 147.28 | 30.58 | 17.70 | 97.50 | 100.980 | 153.813 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0511 | 0.0000 | 100.980 | 0.0000 |  |

## Per-Signal Results

| Backend | Signal | Channels | Duration (s) | Status | xRT | Elapsed (ms) | Quality (/100) | SNR in (dB) | SI-SDR (dB) | LSD (dB) | ModSpec | Smear | EnvCorr | Coherence Drift | Phasiness | Musical Noise | SNR vs CPU (dB) | LSD vs CPU (dB) | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| apple_silicon_native_cpu | `chirp_texture_stereo` | 2 | 4.000 | ok | 96.99 | 41.24 | 100.00 | 102.306 | 155.139 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 102.306 | 0.0000 |  |
| apple_silicon_native_cpu | `speech_like_mono` | 1 | 4.000 | ok | 197.66 | 20.24 | 100.00 | 99.925 | 152.758 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | n/a | 0.0000 | 0.0000 | 99.925 | 0.0000 |  |
| apple_silicon_native_cpu | `tonal_ramp_mono` | 1 | 4.000 | ok | 195.35 | 20.48 | 100.00 | 107.760 | 160.593 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | n/a | 0.0000 | 0.0000 | 107.760 | 0.0000 |  |
| apple_silicon_native_cpu | `transient_drums_stereo` | 2 | 4.000 | ok | 99.12 | 40.36 | 90.00 | 93.930 | 146.763 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2042 | 0.0000 | 93.930 | 0.0000 |  |
| cpu | `chirp_texture_stereo` | 2 | 4.000 | ok | 93.13 | 42.95 | 100.00 | 102.306 | 155.139 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a |  |
| cpu | `speech_like_mono` | 1 | 4.000 | ok | 198.94 | 20.11 | 100.00 | 99.925 | 152.758 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | n/a | 0.0000 | 0.0000 | n/a | n/a |  |
| cpu | `tonal_ramp_mono` | 1 | 4.000 | ok | 194.70 | 20.54 | 100.00 | 107.760 | 160.593 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | n/a | 0.0000 | 0.0000 | n/a | n/a |  |
| cpu | `transient_drums_stereo` | 2 | 4.000 | ok | 98.77 | 40.50 | 90.00 | 93.930 | 146.763 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.2042 | 0.0000 | n/a | n/a |  |
| cuda | `chirp_texture_stereo` | 2 | 4.000 | unavailable | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package. |
| cuda | `speech_like_mono` | 1 | 4.000 | unavailable | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package. |
| cuda | `tonal_ramp_mono` | 1 | 4.000 | unavailable | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package. |
| cuda | `transient_drums_stereo` | 2 | 4.000 | unavailable | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package. |

Interpretation notes:
- `xRT` (times real-time): higher is faster.
- `Quality (/100)` is a composite from core artifact metrics; use it as a quick ranking aid, not as a single acceptance criterion.
- Lower is better for LSD, ModSpec, Smear, Coherence Drift, Phasiness, and Musical Noise.
- Higher is better for SNR, SI-SDR, and Envelope Correlation.

Raw machine-readable benchmark output: `docs/benchmarks/latest.json`.

## Augmentation Benchmark Gate

For AI research/data augmentation workflows, run the profile suite:

```bash
python benchmarks/run_augment_profile_suite.py \
  --quick \
  --gate \
  --out-dir benchmarks/out_augment_profiles
```

This executes four task-specific profiles from `benchmarks/augment_profiles.json`:

- `speech`: automatic speech recognition (ASR)-oriented augmentation
- `music`: music information retrieval (MIR)-oriented augmentation
- `noisy`: noise-stress profile for robustness checks
- `stereo`: contrastive paired-view profile for stereo/coherence-sensitive workflows

To refresh profile baselines after an intentional benchmark change:

```bash
python benchmarks/run_augment_profile_suite.py \
  --quick \
  --refresh-baselines \
  --out-dir benchmarks/out_augment_profiles_refresh
```

Per-profile reports are written to:
- `benchmarks/out_augment_profiles/<profile>/report.json`
- `benchmarks/out_augment_profiles/<profile>/report.md`

Suite summary is written to:
- `benchmarks/out_augment_profiles/suite_report.json`
- `benchmarks/out_augment_profiles/suite_report.md`

Each profile applies:
- relative drift gating against its baseline (profile-specific `gate_tolerance`)
- absolute metric gates (for example, manifest integrity, clipping bounds, pair coverage)

Tracked metrics include:
- record/rendered counts
- required-field manifest validation errors
- split balance (`split_balance_l1`)
- augmentation diversity (`stretch_std`, `pitch_std`)
- safety/quality indicators (`clip_pct_max`, `peak_dbfs_p95`)
- pair completeness (`pair_coverage`) for contrastive two-view runs

## Attribution

Copyright (c) 2026 Colby Leider and contributors. See [`ATTRIBUTION.md`](../ATTRIBUTION.md).
