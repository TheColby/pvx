# Development (DEV) Notes: `pvxvoc` Call Flow and Insertion Points (Phase 0 Recon)

This note documents the current processing path and exact insertion points for:
- hybrid transient engine (phase vocoder (PV) + waveform similarity overlap-add (WSOLA))
- stereo/multichannel coherence modes
- benchmark/regression infrastructure
- preset/help user experience (UX) refactor

## Acronym Primer

- development (DEV)
- command-line interface (CLI)
- user experience (UX)
- digital signal processing (DSP)
- phase vocoder (PV)
- waveform similarity overlap-add (WSOLA)
- short-time Fourier transform (STFT)
- inverse short-time Fourier transform (ISTFT)
- central processing unit (CPU)
- graphics processing unit (GPU)
- Compute Unified Device Architecture (CUDA)
- sample rate (SR)
- fundamental frequency (F0)

No digital signal processing (DSP) behavior changes are implemented in this phase.

## 1. Current `pvxvoc` Call Flow

Primary entrypoint:
- `/Users/cleider/dev/pvx/pvxvoc.py`
  - compatibility wrapper that forwards to `pvx.core.voc:main`

Core runtime flow:
1. `build_parser()` in `/Users/cleider/dev/pvx/src/pvx/core/voc.py:4015`
2. `main()` in `/Users/cleider/dev/pvx/src/pvx/core/voc.py:4545`
3. Input expansion/validation in `expand_inputs()` and validation block (`:4515`, `:4574`)
4. Runtime selection in `configure_runtime_from_args()` (`:4651`)
5. `VocoderConfig` creation (`:4664`)
6. Per-file processing via `process_file()` (`:3612`)
7. Per-file DSP via `process_audio_block()` (`:3264`)
8. Output write + optional manifest in `process_file()` (`:3805`, `:4754`)

`process_file()` high-level stages:
- read audio: `_read_audio_input()` (`:3453`)
- derive pitch/stretch: `choose_pitch_ratio()` + `resolve_base_stretch()` (`:2921`, `:3410`)
- optional map segmentation/checkpointing (`:3658` onward)
- DSP block path: `process_audio_block()` (`:3761`)
- optional output SR resample: `resample_multi()` (`:3780`, `:3873`)
- mastering chain: `apply_mastering_chain()` (`:3785`, `:2788`)
- write output: `_write_audio_output()` (`:3464`)

## 2. Existing STFT/PV and Transient/Phase Logic

Core STFT utilities:
- `stft()` / `istft()` in `/Users/cleider/dev/pvx/src/pvx/core/voc.py:1564` and `:1591`
- window generation in `make_window()` (`:1491`)

Time-stretch engines:
- classic PV: `phase_vocoder_time_stretch()` (`:1943`)
- Fourier-sync PV: `phase_vocoder_time_stretch_fourier_sync()` (`:2037`)
- multistage wrapper: `phase_vocoder_time_stretch_multistage()` (`:2220`)
- multires fusion wrapper: `phase_vocoder_time_stretch_multires_fusion()` (`:2374`)

Current transient handling:
- transient flags (spectral-flux threshold): `compute_transient_flags()` (`:1780`)
- onset-time-credit scheduler: `build_output_time_steps()` (`:1800`)
- transient-triggered phase reset inside PV loops (`:2015-:2019`, `:2141-:2145`)
- CLI controls today:
  - `--transient-preserve`
  - `--transient-threshold`
  - onset-time-credit flags (`--onset-time-credit`, etc.)

Current phase locking:
- identity phase locking kernel: `apply_identity_phase_locking()` (`:1922`)
- applied inside PV loops when `--phase-locking identity` (`:2021-:2025`, `:2147-:2151`)

## 3. Current Channel, Resample, and Mastering Behavior

Channel behavior:
- `process_audio_block()` currently processes channels independently in a for-loop (`:3326` onward)
- this is the main source of potential stereo image drift for large transforms

Resampling:
- per-channel pitch-duration correction uses `resample_1d()` (`:2486`)
- multi-channel output SR conversion uses `resample_multi()` (`:3873`)

Mastering:
- all mastering/dynamics/loudness is centralized in `apply_mastering_chain()` (`:2788`)
- called after time/pitch DSP and optional output sample-rate conversion (`:3785`)

Audio I/O:
- `pvxvoc`: `_read_audio_input()` / `_write_audio_output()` (`:3453`, `:3464`)
- shared helpers for other CLIs in `/Users/cleider/dev/pvx/src/pvx/core/common.py`
  - `read_audio()` (`:354`)
  - `write_output()` (`:369`)

## 4. Insertion Points for Planned Modules

### 4.1 Hybrid transient engine (`src/pvx/core/transients.py`, `src/pvx/core/wsola.py`)

Planned module boundaries:
- `src/pvx/core/transients.py`
  - feature extraction: spectral flux + HFC + broadbandness
  - onset picking + debounce/merge + min-duration regionization
  - output format: transient mask and region list
- `src/pvx/core/wsola.py`
  - deterministic WSOLA time-domain stretch primitive(s)
  - channel-safe interface for mono arrays

Primary integration hooks:
- parser/validation: `build_parser()` and `validate_args()` in `voc.py` (`:4015`, `:3880`)
- config propagation: `VocoderConfig` (`:131`) or a dedicated transient config object
- processing dispatch: `process_audio_block()` (`:3264`)
  - detect regions once per block
  - steady regions -> existing PV path
  - transient regions -> WSOLA path
  - stitch with region crossfades (reuse/extend `concat_audio_chunks()` logic at `:3476`)

Reset-only mode integration:
- retain a `reset` mode by routing to existing transient phase-reset path in PV loops (`:2015`, `:2141`)

Backward compatibility plan:
- keep `--transient-preserve/--transient-threshold` as aliases to new transient-mode behavior

### 4.2 Stereo/multichannel coherence modes

Primary integration hook:
- `process_audio_block()` channel loop (`:3326`)

Planned behavior split:
- `independent`: preserve current path
- `mid_side_lock`: preprocess M/S, coupled processing, decode back to L/R
- `ref_channel_lock`: maintain per-channel magnitudes while constraining phase increments to reference channel

Likely new helpers:
- `src/pvx/core/stereo.py` (processing-side coherence logic)
- `src/pvx/metrics/coherence.py` (analysis metrics API required for benchmarks/tests)

CLI hooks:
- add arguments in `build_parser()` groups (`:4128` onward)
- validate in `validate_args()` (`:3880`)
- include in explain-plan JSON (`:4695`)

### 4.3 Benchmarks and CI regression gates

Current state:
- no top-level `benchmarks/` runner package yet
- existing benchmark generation is doc-oriented in `scripts_generate_docs_extras.py`
- CI currently runs docs + unit tests in `.github/workflows/docs-ci.yml`

Planned additions:
- new top-level directory:
  - `/Users/cleider/dev/pvx/benchmarks/run_bench.py`
  - `/Users/cleider/dev/pvx/benchmarks/metrics.py`
  - `/Users/cleider/dev/pvx/benchmarks/baseline_small.json`
- tool adapters:
  - pvx CLI path
  - Rubber Band CLI (optional/skip when unavailable)
  - librosa baseline path
- CI:
  - new benchmark workflow or docs-ci extension with threshold gates

### 4.4 Presets and UX refactor

Current preset/help location:
- `_PRESET_OVERRIDES` and `_EXAMPLE_COMMANDS` in `voc.py` (`:288`, `:307`)
- argument grouping in `build_parser()` (`:4015`)

Planned refactor:
- move preset definitions to `src/pvx/core/presets.py`
- keep `--preset` backward compatible while adding new intent presets
- extend example mode to include short explanation text per example

## 5. Non-`pvxvoc` CLIs that reuse core DSP

Most command-specific CLIs in `/Users/cleider/dev/pvx/src/pvx/cli/` depend on:
- `build_vocoder_config()` in `core/common.py:281`
- `time_pitch_shift_channel()` / `time_pitch_shift_audio()` in `core/common.py:441` / `:464`

Implication:
- once hybrid/coherence logic is moved into shared core path(s), these tools can inherit improvements without breaking their CLI contracts.

## 6. Determinism Baseline

Current deterministic control points:
- force CPU path via `--device cpu` (`configure_runtime`, `voc.py:1178`)
- stochastic phase engines are deterministic only with `--phase-random-seed`

Planned deterministic requirement:
- WSOLA search, transient regionization, and coherence coupling must be deterministic in CPU mode for reproducible tests/benchmarks.

## Addendum: Stage 3 Implementation Snapshot (2026-02-19)

Implemented after this Phase 0 note:

- Shared output policy module added:
  - `/Users/cleider/dev/pvx/src/pvx/core/output_policy.py`
  - covers bit-depth policy, TPDF dithering, true-peak guard, and metadata sidecars
- Integrated into both write paths:
  - shared CLI writer: `/Users/cleider/dev/pvx/src/pvx/core/common.py`
  - `pvxvoc` writer: `/Users/cleider/dev/pvx/src/pvx/core/voc.py`
- Unified CLI helper commands added:
  - `pvx chain` for managed serial multi-stage chains
  - `pvx stream` for chunked wrapper over `pvx voc`
