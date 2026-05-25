<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# PVC Phase 6-7 Architecture

This document describes the architecture for the final two PVC-inspired phases:

1. Phase 6: function-stream utilities (`envelope`, `reshape`)
2. Phase 7: parity benchmarks and regression gates

## 1) Phase 6 Module Layout

Core module:

- `/Users/cleider/dev/pvx/src/pvx/core/pvc_functions.py`

Command-line interface (CLI) entry points:

- `/Users/cleider/dev/pvx/src/pvx/cli/pvxenvelope.py`
- `/Users/cleider/dev/pvx/src/pvx/cli/pvxreshape.py`

Unified registration:

- `/Users/cleider/dev/pvx/src/pvx/cli/pvx.py`

Packaging scripts:

- `/Users/cleider/dev/pvx/pyproject.toml`

### Design choices

1. Deterministic control-map generation and transformation with no random branches.
2. Text-first map formats (`csv` and `json`) for shell and version-control friendliness.
3. Interpolation compatibility with existing control-map conventions:
   - `none`, `stairstep`, `nearest`, `linear`, `cubic`, `exponential`, `s_curve`, `smootherstep`, `polynomial`
4. Direct interoperability with `pvx voc` and PVC-style operators by emitting explicit key columns (for example `stretch` or `pitch_ratio`).

## 2) Phase 7 Benchmark Layout

Benchmark runner:

- `/Users/cleider/dev/pvx/benchmarks/run_pvc_parity.py`

Baseline artifact:

- `/Users/cleider/dev/pvx/benchmarks/baseline_pvc_parity.json`

Continuous integration (CI) gate:

- `/Users/cleider/dev/pvx/.github/workflows/pvc-parity-regression.yml`

### Scenario set

The parity suite includes both identity-expected and effect-expected cases:

1. `filter_identity_flat`
2. `ring_identity_depth0`
3. `chordmapper_identity_strength0`
4. `inharmonator_identity_mix0`
5. `tvfilter_envelope_modulated`
6. `ringtvfilter_controlled`

### Objective metrics

Per-case metrics:

- signal-to-noise ratio (SNR, dB)
- log spectral distance (LSD)
- modulation spectrum distance
- envelope correlation
- output peak absolute value
- output root mean square (RMS)
- runtime seconds

Gate strategy:

1. Compare current report to committed baseline by case name.
2. Apply absolute drift tolerance per metric (`--gate-tolerance`).
3. Fail continuous integration (CI) if any case/metric drifts beyond tolerance.

## 3) Validation coverage

Tests:

- `/Users/cleider/dev/pvx/tests/test_pvc_phase6.py`
- `/Users/cleider/dev/pvx/tests/test_pvc_parity_benchmark.py`

Validation intent:

1. Correct envelope/control generation and reshape behavior.
2. Command-line interface (CLI) smoke coverage for new tools.
3. Benchmark report generation and baseline gate behavior.

## Attribution
