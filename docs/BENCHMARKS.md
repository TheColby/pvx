# pvx Benchmarks

_Generated from commit `b4bed85` (commit date: 2026-02-19T13:53:32-05:00)._

## Acronym Primer

- application programming interface (API)
- command-line interface (CLI)
- path environment variable (PATH)
- digital signal processing (DSP)
- short-time Fourier transform (STFT)
- inverse short-time Fourier transform (ISTFT)
- fast Fourier transform (FFT)
- discrete Fourier transform (DFT)
- central processing unit (CPU)
- graphics processing unit (GPU)
- Compute Unified Device Architecture (CUDA)
- comma-separated values (CSV)
- JavaScript Object Notation (JSON)
- HyperText Markup Language (HTML)
- Portable Document Format (PDF)
- continuous integration (CI)
- fundamental frequency (F0)
- waveform similarity overlap-add (WSOLA)
- input/output (I/O)
- root-mean-square (RMS)
- loudness units relative to full scale (LUFS)
- signal-to-noise ratio (SNR)

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
