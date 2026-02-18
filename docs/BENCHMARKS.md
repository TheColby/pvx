# pvx Benchmarks

## Cycle-Consistency Runner (`benchmarks/run_bench.py`)

`pvx` includes a reproducible comparison runner against Rubber Band and librosa:

```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out
```

Defaults:
- `--pvx-bench-profile tuned` (deterministic, quality-focused profile)
- tiny public-domain synthetic set for CI stability

Benchmark interpretation policy:
- quality metrics are primary pass/fail signals
- runtime is tracked and improved as a secondary objective

Legacy profile (previous behavior):

```bash
python3 benchmarks/run_bench.py --quick --pvx-bench-profile legacy --out-dir benchmarks/out
```

Regression gate:

```bash
python3 benchmarks/run_bench.py \
  --quick \
  --out-dir benchmarks/out \
  --baseline benchmarks/baseline_small.json \
  --gate
```

Primary quality metrics:
- Log-spectral distance (LSD)
- Modulation spectrum distance
- Transient smear score
- Stereo coherence drift

Additional metrics now reported:
- SNR (dB)
- SI-SDR (dB)
- Spectral convergence
- Envelope correlation
- RMS level delta (dB)
- Crest-factor delta (dB)
- 95% spectral bandwidth delta (Hz)
- Zero-crossing-rate delta
- DC-offset delta
- Clipping-ratio delta
- PESQ MOS-LQO
- STOI / ESTOI
- ViSQOL MOS-LQO
- POLQA MOS-LQO
- PEAQ ODG
- Integrated LUFS delta
- Short-term LUFS delta
- Loudness range (LRA) delta
- True-peak delta (dBTP)
- F0 RMSE (cents)
- Voicing F1
- Harmonic-to-noise-ratio drift (dB)
- Onset precision / recall / F1
- Attack-time error (ms)
- ILD drift (dB)
- ITD drift (ms)
- Inter-channel phase deviation by band (low/mid/high/mean)
- Phasiness index
- Musical-noise index
- Pre-echo score

Perceptual fallback behavior:
- If external standards tooling is unavailable, pvx computes deterministic proxy values for PESQ/STOI/ESTOI/ViSQOL/POLQA/PEAQ.
- Markdown reports include `Proxy Fraction` to show how many perceptual metrics used proxy mode.
- External tool hooks:
  - `VISQOL_BIN`
  - `POLQA_BIN`
  - `PEAQ_BIN`

_Generated from commit `23925ec` (commit date: 2026-02-17T16:15:08-05:00)._

Reproducible benchmark summary for core STFT+ISTFT path across CPU/CUDA/Apple-Silicon-native contexts.

## Reproduce

```bash
python3 scripts_generate_docs_extras.py --run-benchmarks
```

## Benchmark Spec

- Sample rate: `48000` Hz
- Duration: `4.0` s
- Signal recipe: sum of 4 deterministic sinusoids with linear amplitude ramp
- STFT config: `n_fft=2048`, `win_length=2048`, `hop_size=512`, `window=hann`, `center=True`

## Host

- Platform: `macOS-15.1.1-arm64-arm-64bit-Mach-O`
- Machine: `arm64`
- Python: `3.14.3`

## Results

| Backend | Status | Elapsed (ms) | Peak host memory (MB) | SNR vs input (dB) | Spectral distance vs input (dB) | SNR vs CPU (dB) | Spectral distance vs CPU (dB) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpu | ok | 20.356 | 12.02 | 160.5928 | 0.0 | n/a | n/a |  |
| cuda | unavailable | n/a | n/a | n/a | n/a | n/a | n/a | CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package. |
| apple_silicon_native_cpu | ok | 20.276 | 12.02 | 160.5928 | 0.0 | 160.5928 | 0.0 |  |

Raw machine-readable benchmark output: `docs/benchmarks/latest.json`.
