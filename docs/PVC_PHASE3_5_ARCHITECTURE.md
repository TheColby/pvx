<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# PVC Phase 3-5 Architecture

This document describes the architecture used to implement the next three PVC-inspired phases:

1. Phase 3: response-driven spectral operators
2. Phase 4: ring/resonator family
3. Phase 5: harmonic/chord mapping family

## 1) Module Layout

Core signal processing modules:

- `/Users/cleider/dev/pvx/src/pvx/core/pvc_ops.py`
- `/Users/cleider/dev/pvx/src/pvx/core/pvc_resonators.py`
- `/Users/cleider/dev/pvx/src/pvx/core/pvc_harmony.py`

Command-line interface (CLI) entry points:

- Response operators:
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxfilter.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxtvfilter.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxnoisefilter.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxbandamp.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxspeccompander.py`
- Ring/resonator operators:
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxring.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxringfilter.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxringtvfilter.py`
- Harmonic/chord operators:
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxharmmap.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxchordmapper.py`
  - `/Users/cleider/dev/pvx/src/pvx/cli/pvxinharmonator.py`

Unified dispatcher registration:

- `/Users/cleider/dev/pvx/src/pvx/cli/pvx.py`

## 2) Processing Contracts

All new operators follow the same signal contract:

1. Read audio (mono or multichannel) as floating point.
2. Process each channel deterministically.
3. Preserve sample rate and target output length.
4. Pass output through shared mastering/output policy.
5. Emit non-silent console metrics tables and status bars.

## 3) Phase 3 Design

`pvc_ops.py` provides five operators:

- `filter`: static response-curve spectral shaping
- `tvfilter`: time-varying response mix from CSV/JSON
- `noisefilter`: response-floor spectral attenuation
- `bandamp`: response-peak selective amplification
- `spec-compander`: response-referenced spectral companding

Key design choices:

- Response artifacts are loaded from PVXRF.
- Response curves are resized/shifted to match active STFT bins.
- Time-varying controls are scalar maps with interpolation modes:
  - `none`, `stairstep`, `nearest`, `linear`, `cubic`, `exponential`, `s_curve`, `smootherstep`, `polynomial`

## 4) Phase 4 Design

`pvc_resonators.py` provides:

- `ring`: ring modulation
- `ringfilter`: ring modulation + resonant peak filter
- `ringtvfilter`: time-varying ring controls + resonant filter

Key design choices:

- Sample-rate-domain modulation for phase continuity.
- Bounded feedback and bounded resonance decay for stability.
- Optional CSV/JSON control maps for `frequency_hz`, `depth`, and `mix`.

## 5) Phase 5 Design

`pvc_harmony.py` provides:

- `chordmapper`: chord-class spectral weighting (root + chord quality)
- `inharmonator`: inharmonic spectral frequency warp

Key design choices:

- Chord mapping uses circular pitch-class distance in cents.
- Inharmonator uses an analytic inverse warp from output-bin frequency to input-bin sampling frequency.
- Both operators are implemented in the short-time Fourier transform (STFT) domain with overlap-add resynthesis.

## 6) Backward Compatibility and UX

- Existing tools are untouched.
- New tools are additive and available both:
  - via unified command-line interface (CLI): `pvx <tool>`
  - via root wrappers: `python3 pvx<tool>.py ...`
- Shared conventions are reused: progress/status, verbosity, metrics table, mastering chain.

## Attribution


