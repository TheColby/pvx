<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# PVC-to-pvx Parity Matrix

This document tracks functional parity work between Paul Koonce's historical PVC package and modern `pvx`.

Primary references:
- Princeton archive page (Paul Koonce): [https://www.cs.princeton.edu/courses/archive/spr99/cs325/koonce.html](https://www.cs.princeton.edu/courses/archive/spr99/cs325/koonce.html)
- Linux Audio package index entry: [https://wiki.linuxaudio.org/apps/all/pvc](https://wiki.linuxaudio.org/apps/all/pvc)

## Scope

- Goal: preserve useful PVC workflow primitives while keeping `pvx` quality-first and automation-friendly.
- Out of scope: byte-for-byte command-line compatibility with historical PVC binaries.

## Capability Matrix

| PVC capability | Why it matters | Current `pvx` status | Gap level |
| --- | --- | --- | --- |
| Persistent phase-vocoder analysis files (`pvanalysis`) | Reuse analysis across multiple processing passes | Implemented in phase 1/2 via PVXAN schema and `pvx analysis` | Closed (foundation) |
| Response-file workflows (`freqresponse`, `chordresponsemaker`) | Reusable filter/target curves | Implemented in phase 1/2 via PVXRF schema and `pvx response` | Closed (foundation) |
| Time-varying spectral filtering from response files (`tvfilter`) | Dynamic timbral trajectories | Implemented in phase 3 via `pvx tvfilter` + scalar map interpolation | Closed (phase 3) |
| Ring/resonator family (`ring`, `ringfilter`, `ringtvfilter`) | Controlled resonant coloration | Implemented in phase 4 via dedicated ring CLI family | Closed (phase 4) |
| Response-driven compander/noise/band emphasis (`compander`, `noisefilter`, `bandamp`) | Precision spectral dynamics | Implemented in phase 3 (`pvx spec-compander`, `pvx noisefilter`, `pvx bandamp`) | Closed (phase 3) |
| Harmonic/chord remapping (`harmonizer`, `chordmapper`, `inharmonator`) | Musically constrained spectral remap | Implemented in phase 5 via `pvx chordmapper` and `pvx inharmonator` | Closed (phase 5) |
| Spectral convolver with response semantics (`convolver`) | Controlled blend of source/target spectra | Partial overlap via morph/cross-synthesis | Medium |
| Generic function stream utilities (`envelope`, `reshape`) | Text-based control-signal authoring | Implemented in phase 6 via `pvx envelope` and `pvx reshape` | Closed (phase 6) |
| Parity benchmark scenarios and regression gates | Reproducible quality checks for PVC-inspired operators | Implemented in phase 7 via `benchmarks/run_pvc_parity.py` + CI workflow gate | Closed (phase 7) |
| Explicit resynthesis families and thresholds | Operator transparency and reproducibility | Partially available but not as PVC-style dedicated modes | Medium |

## Implemented in Phase 1 and Phase 2

1. Data-model and schema foundations:
   - `PVXAN` analysis artifact schema in `/Users/cleider/dev/pvx/src/pvx/core/analysis_store.py`
   - `PVXRF` response artifact schema in `/Users/cleider/dev/pvx/src/pvx/core/response_store.py`
2. User-facing command-line interface (CLI) foundations:
   - `pvx analysis create|inspect` in `/Users/cleider/dev/pvx/src/pvx/cli/pvxanalysis.py`
   - `pvx response create|inspect` in `/Users/cleider/dev/pvx/src/pvx/cli/pvxresponse.py`
   - Unified CLI registration in `/Users/cleider/dev/pvx/src/pvx/cli/pvx.py`

## Schema Notes

### PVXAN

- Container: compressed NumPy `.npz`
- Required members:
  - `meta_json`
  - `spectrum_real` (`channels x frames x bins`)
  - `spectrum_imag` (`channels x frames x bins`)
- Includes deterministic `sha256` digest function for reproducibility checks.

### PVXRF

- Container: compressed NumPy `.npz`
- Required members:
  - `meta_json`
  - `frequencies_hz` (`bins`)
  - `magnitude` (`channels x bins`)
  - `phase` (`channels x bins`)
- Includes deterministic `sha256` digest function for reproducibility checks.

## Implemented in Phase 3, Phase 4, and Phase 5

1. Response-driven spectral operators (`filter`, `tvfilter`, `noisefilter`, `bandamp`, `spec-compander`):
   - core: `/Users/cleider/dev/pvx/src/pvx/core/pvc_ops.py`
   - command-line interface (CLI): `/Users/cleider/dev/pvx/src/pvx/cli/pvxfilter.py` + wrappers
2. Ring/resonator family (`ring`, `ringfilter`, `ringtvfilter`):
   - core: `/Users/cleider/dev/pvx/src/pvx/core/pvc_resonators.py`
   - CLI: `/Users/cleider/dev/pvx/src/pvx/cli/pvxring.py` + wrappers
3. Harmonic/chord mapping family (`chordmapper`, `inharmonator`):
   - core: `/Users/cleider/dev/pvx/src/pvx/core/pvc_harmony.py`
   - CLI: `/Users/cleider/dev/pvx/src/pvx/cli/pvxharmmap.py` + wrappers

## Implemented in Phase 6 and Phase 7

1. Function-stream generators/transforms (`envelope`, `reshape`) interoperable with control-bus workflows:
   - core: `/Users/cleider/dev/pvx/src/pvx/core/pvc_functions.py`
   - CLI: `/Users/cleider/dev/pvx/src/pvx/cli/pvxenvelope.py`, `/Users/cleider/dev/pvx/src/pvx/cli/pvxreshape.py`
   - installed entry points: `pvx envelope`, `pvx reshape`
2. Parity benchmarks and regression gates:
   - benchmark runner: `/Users/cleider/dev/pvx/benchmarks/run_pvc_parity.py`
   - baseline: `/Users/cleider/dev/pvx/benchmarks/baseline_pvc_parity.json`
   - CI workflow: `/Users/cleider/dev/pvx/.github/workflows/pvc-parity-regression.yml`
   - expanded pipeline scenario: `analysis_response_function_chain` (analysis-derived response + function-stream modulation)

## Immediate Next Steps

1. Add explicit PVC-style resynthesis family toggles and thresholds where they improve auditability without fragmenting the core `pvx voc` interface.
2. Add additional parity scenarios that stress the same pipeline family with persisted artifact round-trips (`pvx analysis create` + `pvx response create`) and stricter drift gates.
3. Add the sorted top-level command roadmap below to stage user-facing workflow expansion in quality-first order.

### Sorted Top-Level Command Roadmap

| Phase | Priority | Proposed commands | Parity and quality rationale |
| --- | --- | --- | --- |
| Phase 1 | Highest | `doctor`, `inspect`, `validate`, `schema`, `preset`, `config` | Improves reproducibility and inspectability before introducing more operator complexity. |
| Phase 2 | High | `render`, `graph`, `queue`, `watch`, `cache` | Supports longer research- and production-oriented runs with less shell friction. |
| Phase 3 | Medium-High | `mod`, `derive`, `route`, `quantize`, `smooth` | Brings function-stream/control-signal workflows closer to classic PVC flexibility with modern parameter routing. |
| Phase 4 | Medium | `bench`, `compare`, `regress`, `abx`, `report` | Establishes objective parity and regression tracking as first-class command surfaces. |
| Phase 5 | Medium-Low | `align`, `match`, `stem`, `spatialize`, `live`, `serve` | Expands into advanced alignment, immersive, and deployment workflows after core parity and QA gates are stable. |

## Attribution
