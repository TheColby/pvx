# pvx Architecture

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

System architecture for runtime processing, algorithm dispatch, and documentation pipelines.

## 1. Runtime and CLI Flow

```mermaid
flowchart LR
  A[User CLI Command] --> B[src/pvx/cli or pvxvoc parser]
  B --> C[Runtime Selection: auto/cpu/cuda]
  C --> D[Shared IO + Mastering Chain]
  D --> E[Core DSP in src/pvx/core/voc.py]
  E --> F[Output Writer / stdout stream]
```

## 2. Algorithm Registry and Dispatch

```mermaid
flowchart TD
  R[src/pvx/algorithms/registry.py] --> B[src/pvx/algorithms/base.py]
  B --> M1[time_scale_and_pitch_core/*]
  B --> M2[retune_and_intonation/*]
  B --> M3[dynamics_and_loudness/*]
  B --> M4[spatial_and_multichannel/*]
```

## 3. Documentation Build Graph

```mermaid
flowchart LR
  G1[scripts_generate_python_docs.py] --> D[docs/*]
  G2[scripts_generate_theory_docs.py] --> D
  G3[scripts_generate_docs_extras.py] --> D
  G4[scripts_generate_html_docs.py] --> H[docs/html/*]
  D --> H
```

## 4. CI + Pages

```mermaid
flowchart LR
  PR[Push / PR] --> CI[Doc and test workflow]
  CI --> S[Generation + drift checks]
  S --> T[Unit tests + docs coverage tests]
  T --> P[GitHub Pages deploy workflow]
  P --> SITE[Published docs/html site]
```
