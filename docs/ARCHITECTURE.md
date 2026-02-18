# pvx Architecture

_Generated from commit `23925ec` (commit date: 2026-02-17T16:15:08-05:00)._

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
