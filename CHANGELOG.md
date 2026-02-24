# Changelog



## 2026-02-23

- Alpha release hardening and verification pass:
  - resolved CQT Nyquist overflow in transform dispatch by capping `n_bins` and carrying transform metadata through inverse synthesis.
  - hardened LUFS estimation for short clips by padding/fallback when `pyloudnorm` block-size requirements are not met.
  - fixed onset/beat analysis scalar extraction when tempo is returned as an array-like value.
  - fixed benchmark package import precedence so `src/pvx` is used before root-level compatibility wrappers.
  - completed full validation gate in `.venv`:
    - dependency sync
    - lint (`ruff`)
    - type check (`mypy`)
    - unit tests (107 tests)
    - coverage gate (`--fail-under=45`, observed 61%)
    - docs regeneration
    - quick benchmark run with report output.

- Expanded `pvx morph` into multi-mode cross-synthesis:
  - new blend modes in `src/pvx/cli/pvxmorph.py`:
    - `linear`, `geometric`
    - `magnitude_b_phase_a`, `magnitude_a_phase_b`
    - `carrier_a_envelope_b`, `carrier_b_envelope_a`
    - `carrier_a_mask_b`, `carrier_b_mask_a`
    - `product`, `max_mag`, `min_mag`
  - new controls:
    - `--blend-mode`
    - `--phase-mix`
    - `--mask-exponent`
    - `--envelope-lifter`
    - `--normalize-energy`
  - added CLI regression coverage in `tests/test_cli_regression.py` for envelope/mask cross-synthesis modes.
  - refreshed user docs examples in:
    - `README.md`
    - `docs/GETTING_STARTED.md`
    - `docs/EXAMPLES.md`
    - `docs/PIPELINE_COOKBOOK.md`

- Expanded feature-tracking/control-bus sidechain capabilities:
  - added frame-level feature extraction module:
    - `src/pvx/core/feature_tracking.py`
  - `pvx pitch-track` now emits broad feature vectors (configurable via `--feature-set` and `--mfcc-count`), including:
    - pitch/voicing, loudness/dynamics, spectral features, formants, rhythm, stereo cues, noise/artifact proxies
    - MFCC columns (`mfcc_01..mfcc_N`)
    - MPEG-7-style descriptors (`mpeg7_*`, including coarse audio spectrum envelope bands)
  - control-bus routes now support richer mapping operators:
    - `mul(source,factor)`, `add(source,offset)`, `affine(source,scale,bias)`, `clip(source,lo,hi)`
  - control-bus source resolution now supports any numeric CSV column name (not only fixed built-in names).
  - `pvx follow` now exposes tracker feature controls (`--feature-set`, `--mfcc-count`) for feature-driven sidechain workflows.

- Continued follow/control-bus UX phase with richer built-in examples:
  - expanded unified CLI examples in:
    - `src/pvx/cli/pvx.py`
    - new `pvx examples` entries:
      - `follow-feature`
      - `follow-formant`
      - `follow-noise-aware`
  - `pvx follow` now supports named example output:
    - `pvx follow --example` (basic)
    - `pvx follow --example all`
    - `pvx follow --example mfcc_flux|formant_onset|noise_aware|pitch`
  - added follow example regression coverage in:
    - `tests/test_cli_regression.py`
  - updated docs to surface feature-follow examples and `pvx follow --example` workflows:
    - `README.md`
    - `docs/GETTING_STARTED.md`
    - `docs/EXAMPLES.md`
    - `docs/FEATURE_SIDECHAIN_EXAMPLES.md`

- Completed remaining follow/control-bus rollout phases (Phase 3-6):
  - Phase 3 (one-command orchestrator):
    - added unified helper command `pvx follow` in:
      - `src/pvx/cli/pvx.py`
    - `pvx follow` now runs pitch tracking + control-map handoff + vocoder apply in one command.
    - supports:
      - tracker controls (`--backend`, `--fmin`, `--fmax`, `--frame-length`, `--hop-size`, `--emit`, `--stretch-*`)
      - vocoder map policy controls (`--pitch-conf-min`, `--pitch-lowconf-mode`, `--pitch-map-*`)
      - optional control routes (`--route ...`) and passthrough of additional `pvx voc` flags
  - Phase 4 (regression coverage):
    - added dedicated control-bus unit tests:
      - `tests/test_control_bus.py`
    - expanded CLI regression suite:
      - `tests/test_cli_regression.py`
      - added `pvx follow` success path, help target, and invalid passthrough rejection tests
  - Phase 5 (docs + UX migration guidance):
    - added migration guide:
      - `docs/FOLLOW_MIGRATION.md`
    - updated user docs to promote `pvx follow` as the shortest sidechain workflow:
      - `README.md`
      - `docs/GETTING_STARTED.md`
      - `docs/EXAMPLES.md`
      - `docs/PIPELINE_COOKBOOK.md`
  - Phase 6 (rollout artifacts):
    - updated reviewer checklist and validation flow:
      - `docs/HOW_TO_REVIEW.md`

## 2026-02-19

- Stage 3 (pipeline ergonomics + output policy hardening) implemented:
  - added shared output-policy core module:
    - `src/pvx/core/output_policy.py`
  - added deterministic output policy flags across audio-output tools:
    - `--bit-depth {inherit,16,24,32f}`
    - `--dither {none,tpdf}` and `--dither-seed`
    - `--true-peak-max-dbtp`
    - `--metadata-policy {none,sidecar,copy}`
    - retained explicit override `--subtype`
  - integrated output-policy processing into:
    - shared writer path `src/pvx/core/common.py`
    - `pvx voc` write path `src/pvx/core/voc.py`
  - added metadata sidecar emission (`*.metadata.json`) for reproducible output metadata.
  - added helper workflows in unified CLI:
    - `pvx chain` (managed multi-stage serial tool chains)
    - `pvx stream` now defaults to a stateful chunk engine with continuity context (`--mode stateful`)
    - legacy segmented behavior remains available via `pvx stream --mode wrapper`
  - updated non-`voc` tool writers to propagate source input metadata when sidecars are enabled.
  - added/updated tests:
    - `tests/test_output_policy.py`
    - `tests/test_cli_regression.py`
  - updated docs:
    - `README.md`
    - `docs/GETTING_STARTED.md`
    - `docs/EXAMPLES.md`
    - `docs/QUALITY_GUIDE.md`

- Stage 2 (quality validation + determinism) implemented:
  - expanded benchmark reproducibility controls in `benchmarks/run_bench.py`:
    - corpus manifest support (`--dataset-manifest`, `--refresh-manifest`, `--strict-corpus`)
    - deterministic CPU mode controls (`--deterministic-cpu`, `--no-deterministic-cpu`)
    - repeated determinism checks (`--determinism-runs`)
    - output signature capture and baseline signature gating (`--gate-signatures`)
    - row-level regression gating (`--gate-row-level`)
  - added multi-metric directional gate rules beyond the original 4-metric gate.
  - added automatic quality diagnostics in benchmark JSON/Markdown reports with remediation hints.
  - added corpus hash manifest `benchmarks/data/manifest.json`.
  - updated quick baseline to include row metrics and deterministic signatures:
    - `benchmarks/baseline_small.json`
  - updated CI benchmark gate workflow:
    - `.github/workflows/bench-regression.yml`
  - added benchmark Stage 2 unit coverage in:
    - `tests/test_benchmark_runner.py`
  - refreshed benchmark docs:
    - `docs/BENCHMARKS.md`

- Added unified `pvx` CLI entrypoint:
  - new module `src/pvx/cli/pvx.py`
  - new root compatibility wrapper `pvx.py`
  - new package script in `pyproject.toml`: `pvx = "pvx.cli.pvx:main"`
- Unified command dispatch now supports:
  - subcommands for all existing tools (`voc`, `freeze`, `harmonize`, `conform`, `morph`, `warp`, `formant`, `transient`, `unison`, `denoise`, `deverb`, `retune`, `layer`, `pitch-track`)
  - helper commands: `list`, `help`, `examples`, `guided`
  - default shorthand: `pvx <input.wav> ...` automatically routes to `pvx voc ...`
- Replaced legacy `main` navigator with a compatibility bridge to unified `pvx` CLI:
  - `src/pvx/cli/main.py`
- Standardized explicit single-file output support across common CLI tools:
  - added `--output` / `--out` in shared I/O args (`src/pvx/core/common.py`)
  - added validation for `--output` with multi-input/`--output-dir`/`--stdout` combinations
- Updated `pvxvoc` built-in examples/help text to show unified `pvx voc ...` commands.
- Updated beginner-facing docs for Stage 1 UX:
  - `README.md`
  - `docs/GETTING_STARTED.md`
  - `docs/EXAMPLES.md`
- Added CLI regression tests for unified command surface and explicit output path behavior:
  - `tests/test_cli_regression.py`

## 2026-02-18

- Added hybrid transient engine plumbing in `pvxvoc` with new modes:
  - `--transient-mode off|reset|hybrid|wsola`
  - `--transient-sensitivity`
  - `--transient-protect-ms`
  - `--transient-crossfade-ms`
- Added deterministic WSOLA core implementation in `src/pvx/core/wsola.py`.
- Added transient detection/segmentation module in `src/pvx/core/transients.py`.
- Added stereo/multichannel coherence controls:
  - `--stereo-mode independent|mid_side_lock|ref_channel_lock`
  - `--ref-channel`
  - `--coherence-strength`
- Added channel-coherence utilities and objective coherence metrics:
  - `src/pvx/core/stereo.py`
  - `src/pvx/metrics/coherence.py`
- Added intent preset registry in `src/pvx/core/presets.py` with new presets:
  - `default`, `vocal_studio`, `drums_safe`, `extreme_ambient`, `stereo_coherent`
  - legacy presets remain supported (`none`, `vocal`, `ambient`, `extreme`)
- Added benchmark suite:
  - `benchmarks/run_bench.py`
  - `benchmarks/metrics.py`
  - baseline gate file `benchmarks/baseline_small.json`
- Added CI benchmark regression workflow:
  - `.github/workflows/bench-regression.yml`
- Added/updated tests for transient/stereo behavior and new CLI paths.
- Refined CLI help taxonomy with explicit sections:
  - `I/O`, `Performance`, `Quality/Phase`, `Time/Pitch`, `Transients`, `Stereo`, `Output/Mastering`, `Debug`
  - consolidated duplicate `Time/Pitch` blocks into a single grouped section
- Added benchmark metric unit tests in `tests/test_benchmark_metrics.py`.
- Tuned benchmark runner defaults in `benchmarks/run_bench.py`:
  - new `--pvx-bench-profile {tuned,legacy}` switch
  - default `tuned` deterministic profile for stronger cycle-consistency quality metrics
- Expanded benchmark metrics in `benchmarks/metrics.py` and reporting:
  - SNR, SI-SDR, spectral convergence, envelope correlation
  - RMS/crest deltas, bandwidth(95%) delta, ZCR delta, DC delta, clipping-ratio delta
- Updated benchmark gate baseline in `benchmarks/baseline_small.json` to match tuned profile output.
- Added shared audio metrics table module `src/pvx/core/audio_metrics.py`.
- Added non-silent ASCII input/output metrics table printing across pvx CLIs and `pvxvoc`.
- Expanded documentation:
  - `docs/DEV_NOTES.md`
  - `docs/QUALITY_GUIDE.md`
  - `docs/RUBBERBAND_COMPARISON.md`
  - `docs/BENCHMARKS.md`
  - refreshed `README.md`, `docs/GETTING_STARTED.md`, `docs/EXAMPLES.md`

## Attribution

Copyright (c) 2026 Colby Leider and contributors. See [ATTRIBUTION.md](ATTRIBUTION.md).
