# pvx Benchmarks

_Generated from commit `dcc235a` (commit date: 2026-02-20T11:48:06-05:00)._

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
