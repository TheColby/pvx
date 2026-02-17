# PVX Benchmarks

_Generated from commit `58b1a2e` (commit date: 2026-02-17T09:29:32-05:00)._

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
