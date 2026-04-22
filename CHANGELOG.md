# Changelog

## 2026-03-15 (continued)

- Plugin system for custom transforms:
  - added `register_transform()`, `get_transform()`, `list_transforms()` to `pvx.augment.core`
  - supports decorator-based registration and auto-snake_case naming
  - added `_load_entry_point_plugins()` for third-party packages to register transforms via `pvx.augment.transforms` entry points
  - exported new functions from `pvx.augment.__init__`

- Batch GPU pipeline for multi-file processing:
  - added `batch_process_files()` to `pvx.augment.gpu`
  - loads multiple files, pads/truncates to uniform length, stacks into `(B, C, T)` tensor
  - processes entire batches through `TorchPipeline` on GPU with auto-device selection
  - writes results back with original-length trimming
  - exported from `pvx.augment.__init__`

- IRDatabase wired into pvxrir CLI:
  - added `--ir-database`, `--category`, `--list-databases`, `--cache-dir` flags to `pvxrir`
  - auto-downloads curated IR databases on first use (echothief, mit_kemar)
  - supports category filtering (hall, church, outdoor, room, large)
  - updated `docs/cli_flags_reference.json` with new flags

- Documentation site (mkdocs):
  - added `mkdocs.yml` with Material theme, search, code-copy, dark mode toggle
  - created `docs/index.md` landing page
  - added `docs` optional dependency group (`mkdocs`, `mkdocs-material`, `pymdown-extensions`)
  - updated `.github/workflows/pages.yml` to build mkdocs site alongside legacy docs

- Expanded ruff configuration:
  - added `[tool.ruff]` section to `pyproject.toml`
  - enabled 17 lint rule sets: E, W, F, I, N, UP, B, A, C4, SIM, TCH, RUF, PIE, PT, RET, ISC
  - configured isort with `pvx` as first-party, format with double quotes

- WER benchmark script validated:
  - `benchmarks/run_augment_wer_bench.py` confirmed parseable and functional
  - 5 augmentation conditions (baseline, gain, noise+gain, SpecAugment, full asr_pipeline)
  - supports --epochs, --max-train-samples, --device, GPU/CPU modes

- Test coverage expanded:
  - added `TestPluginSystem` class (5 tests): register, get, auto-naming, pipeline integration, error handling
  - full suite: 210 passed, 3 skipped

## 2026-03-15

- Alpha release readiness pass (research- and production-oriented):
  - cleaned packaging/test artifacts from the release flow (`build/`, `dist/`, `*.egg-info/`) and added ignore rules in `/Users/cleider/dev/pvx/.gitignore`.
  - expanded packaging metadata and distribution inputs:
    - `/Users/cleider/dev/pvx/pyproject.toml`
    - `/Users/cleider/dev/pvx/MANIFEST.in`
  - added a full multi-version continuous integration workflow:
    - `/Users/cleider/dev/pvx/.github/workflows/ci.yml`

- Augmentation benchmark depth expansion:
  - upgraded `/Users/cleider/dev/pvx/benchmarks/run_augment_bench.py` with:
    - named benchmark profiles (`--profile`, `--profiles-file`, `--list-profiles`)
    - profile-aware override logic
    - absolute metric gates in addition to baseline-relative drift gates
    - richer gate reporting in JSON and Markdown
  - added profile configuration:
    - `/Users/cleider/dev/pvx/benchmarks/augment_profiles.json`
  - added per-profile baselines:
    - `/Users/cleider/dev/pvx/benchmarks/baselines/augment_speech.json`
    - `/Users/cleider/dev/pvx/benchmarks/baselines/augment_music.json`
    - `/Users/cleider/dev/pvx/benchmarks/baselines/augment_noisy.json`
    - `/Users/cleider/dev/pvx/benchmarks/baselines/augment_stereo.json`
  - added suite runner for one-command matrix execution:
    - `/Users/cleider/dev/pvx/benchmarks/run_augment_profile_suite.py`
  - updated CI augmentation regression job to run profile suite gates:
    - `/Users/cleider/dev/pvx/.github/workflows/augment-bench-regression.yml`

- Augmentation framework/API expansion:
  - added optional long-form streaming augmentation:
    - `/Users/cleider/dev/pvx/src/pvx/augment/streaming.py`
  - added impulse-response database management for room simulation pipelines:
    - `/Users/cleider/dev/pvx/src/pvx/augment/ir_database.py`
  - added/expanded GPU path and augmentation transforms/tests/docs in:
    - `/Users/cleider/dev/pvx/src/pvx/augment/gpu.py`
    - `/Users/cleider/dev/pvx/src/pvx/augment/time_domain.py`
    - `/Users/cleider/dev/pvx/tests/test_augment_transforms.py`
    - `/Users/cleider/dev/pvx/tests/test_gpu_transforms.py`
    - `/Users/cleider/dev/pvx/docs/ML_INTEGRATION.md`
    - `/Users/cleider/dev/pvx/docs/AUGMENTATION_COOKBOOK.md`

- Validation:
  - full test suite pass in local environment: `205 passed, 3 skipped`.
  - augmentation profile suite quick gate pass:
    - `python benchmarks/run_augment_profile_suite.py --quick --gate`.

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
  - new blend modes in `/Users/cleider/dev/pvx/src/pvx/cli/pvxmorph.py`:
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
  - added CLI regression coverage in `/Users/cleider/dev/pvx/tests/test_cli_regression.py` for envelope/mask cross-synthesis modes.
  - refreshed user docs examples in:
    - `/Users/cleider/dev/pvx/README.md`
    - `/Users/cleider/dev/pvx/docs/GETTING_STARTED.md`
    - `/Users/cleider/dev/pvx/docs/EXAMPLES.md`
    - `/Users/cleider/dev/pvx/docs/PIPELINE_COOKBOOK.md`

- Expanded feature-tracking/control-bus sidechain capabilities:
  - added frame-level feature extraction module:
    - `/Users/cleider/dev/pvx/src/pvx/core/feature_tracking.py`
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
    - `/Users/cleider/dev/pvx/src/pvx/cli/pvx.py`
    - new `pvx examples` entries:
      - `follow-feature`
      - `follow-formant`
      - `follow-noise-aware`
  - `pvx follow` now supports named example output:
    - `pvx follow --example` (basic)
    - `pvx follow --example all`
    - `pvx follow --example mfcc_flux|formant_onset|noise_aware|pitch`
  - added follow example regression coverage in:
    - `/Users/cleider/dev/pvx/tests/test_cli_regression.py`
  - updated docs to surface feature-follow examples and `pvx follow --example` workflows:
    - `/Users/cleider/dev/pvx/README.md`
    - `/Users/cleider/dev/pvx/docs/GETTING_STARTED.md`
    - `/Users/cleider/dev/pvx/docs/EXAMPLES.md`
    - `/Users/cleider/dev/pvx/docs/FEATURE_SIDECHAIN_EXAMPLES.md`

- Completed remaining follow/control-bus rollout phases (Phase 3-6):
  - Phase 3 (one-command orchestrator):
    - added unified helper command `pvx follow` in:
      - `/Users/cleider/dev/pvx/src/pvx/cli/pvx.py`
    - `pvx follow` now runs pitch tracking + control-map handoff + vocoder apply in one command.
    - supports:
      - tracker controls (`--backend`, `--fmin`, `--fmax`, `--frame-length`, `--hop-size`, `--emit`, `--stretch-*`)
      - vocoder map policy controls (`--pitch-conf-min`, `--pitch-lowconf-mode`, `--pitch-map-*`)
      - optional control routes (`--route ...`) and passthrough of additional `pvx voc` flags
  - Phase 4 (regression coverage):
    - added dedicated control-bus unit tests:
      - `/Users/cleider/dev/pvx/tests/test_control_bus.py`
    - expanded CLI regression suite:
      - `/Users/cleider/dev/pvx/tests/test_cli_regression.py`
      - added `pvx follow` success path, help target, and invalid passthrough rejection tests
  - Phase 5 (docs + UX migration guidance):
    - added migration guide:
      - `/Users/cleider/dev/pvx/docs/FOLLOW_MIGRATION.md`
    - updated user docs to promote `pvx follow` as the shortest sidechain workflow:
      - `/Users/cleider/dev/pvx/README.md`
      - `/Users/cleider/dev/pvx/docs/GETTING_STARTED.md`
      - `/Users/cleider/dev/pvx/docs/EXAMPLES.md`
      - `/Users/cleider/dev/pvx/docs/PIPELINE_COOKBOOK.md`
  - Phase 6 (rollout artifacts):
    - updated reviewer checklist and validation flow:
      - `/Users/cleider/dev/pvx/docs/HOW_TO_REVIEW.md`

## 2026-02-19

- Stage 3 (pipeline ergonomics + output policy hardening) implemented:
  - added shared output-policy core module:
    - `/Users/cleider/dev/pvx/src/pvx/core/output_policy.py`
  - added deterministic output policy flags across audio-output tools:
    - `--bit-depth {inherit,16,24,32f}`
    - `--dither {none,tpdf}` and `--dither-seed`
    - `--true-peak-max-dbtp`
    - `--metadata-policy {none,sidecar,copy}`
    - retained explicit override `--subtype`
  - integrated output-policy processing into:
    - shared writer path `/Users/cleider/dev/pvx/src/pvx/core/common.py`
    - `pvx voc` write path `/Users/cleider/dev/pvx/src/pvx/core/voc.py`
  - added metadata sidecar emission (`*.metadata.json`) for reproducible output metadata.
  - added helper workflows in unified CLI:
    - `pvx chain` (managed multi-stage serial tool chains)
    - `pvx stream` now defaults to a stateful chunk engine with continuity context (`--mode stateful`)
    - legacy segmented behavior remains available via `pvx stream --mode wrapper`
  - updated non-`voc` tool writers to propagate source input metadata when sidecars are enabled.
  - added/updated tests:
    - `/Users/cleider/dev/pvx/tests/test_output_policy.py`
    - `/Users/cleider/dev/pvx/tests/test_cli_regression.py`
  - updated docs:
    - `/Users/cleider/dev/pvx/README.md`
    - `/Users/cleider/dev/pvx/docs/GETTING_STARTED.md`
    - `/Users/cleider/dev/pvx/docs/EXAMPLES.md`
    - `/Users/cleider/dev/pvx/docs/QUALITY_GUIDE.md`

- Stage 2 (quality validation + determinism) implemented:
  - expanded benchmark reproducibility controls in `/Users/cleider/dev/pvx/benchmarks/run_bench.py`:
    - corpus manifest support (`--dataset-manifest`, `--refresh-manifest`, `--strict-corpus`)
    - deterministic CPU mode controls (`--deterministic-cpu`, `--no-deterministic-cpu`)
    - repeated determinism checks (`--determinism-runs`)
    - output signature capture and baseline signature gating (`--gate-signatures`)
    - row-level regression gating (`--gate-row-level`)
  - added multi-metric directional gate rules beyond the original 4-metric gate.
  - added automatic quality diagnostics in benchmark JSON/Markdown reports with remediation hints.
  - added corpus hash manifest `/Users/cleider/dev/pvx/benchmarks/data/manifest.json`.
  - updated quick baseline to include row metrics and deterministic signatures:
    - `/Users/cleider/dev/pvx/benchmarks/baseline_small.json`
  - updated CI benchmark gate workflow:
    - `/Users/cleider/dev/pvx/.github/workflows/bench-regression.yml`
  - added benchmark Stage 2 unit coverage in:
    - `/Users/cleider/dev/pvx/tests/test_benchmark_runner.py`
  - refreshed benchmark docs:
    - `/Users/cleider/dev/pvx/docs/BENCHMARKS.md`

- Added unified `pvx` CLI entrypoint:
  - new module `/Users/cleider/dev/pvx/src/pvx/cli/pvx.py`
  - new root compatibility wrapper `/Users/cleider/dev/pvx/pvx.py`
  - new package script in `/Users/cleider/dev/pvx/pyproject.toml`: `pvx = "pvx.cli.pvx:main"`
- Unified command dispatch now supports:
  - subcommands for all existing tools (`voc`, `freeze`, `harmonize`, `conform`, `morph`, `warp`, `formant`, `transient`, `unison`, `denoise`, `deverb`, `retune`, `layer`, `pitch-track`)
  - helper commands: `list`, `help`, `examples`, `guided`
  - default shorthand: `pvx <input.wav> ...` automatically routes to `pvx voc ...`
- Replaced legacy `main` navigator with a compatibility bridge to unified `pvx` CLI:
  - `/Users/cleider/dev/pvx/src/pvx/cli/main.py`
- Standardized explicit single-file output support across common CLI tools:
  - added `--output` / `--out` in shared I/O args (`/Users/cleider/dev/pvx/src/pvx/core/common.py`)
  - added validation for `--output` with multi-input/`--output-dir`/`--stdout` combinations
- Updated `pvxvoc` built-in examples/help text to show unified `pvx voc ...` commands.
- Updated beginner-facing docs for Stage 1 UX:
  - `/Users/cleider/dev/pvx/README.md`
  - `/Users/cleider/dev/pvx/docs/GETTING_STARTED.md`
  - `/Users/cleider/dev/pvx/docs/EXAMPLES.md`
- Added CLI regression tests for unified command surface and explicit output path behavior:
  - `/Users/cleider/dev/pvx/tests/test_cli_regression.py`

## 2026-02-18

- Added hybrid transient engine plumbing in `pvxvoc` with new modes:
  - `--transient-mode off|reset|hybrid|wsola`
  - `--transient-sensitivity`
  - `--transient-protect-ms`
  - `--transient-crossfade-ms`
- Added deterministic WSOLA core implementation in `/Users/cleider/dev/pvx/src/pvx/core/wsola.py`.
- Added transient detection/segmentation module in `/Users/cleider/dev/pvx/src/pvx/core/transients.py`.
- Added stereo/multichannel coherence controls:
  - `--stereo-mode independent|mid_side_lock|ref_channel_lock`
  - `--ref-channel`
  - `--coherence-strength`
- Added channel-coherence utilities and objective coherence metrics:
  - `/Users/cleider/dev/pvx/src/pvx/core/stereo.py`
  - `/Users/cleider/dev/pvx/src/pvx/metrics/coherence.py`
- Added intent preset registry in `/Users/cleider/dev/pvx/src/pvx/core/presets.py` with new presets:
  - `default`, `vocal_studio`, `drums_safe`, `extreme_ambient`, `stereo_coherent`
  - legacy presets remain supported (`none`, `vocal`, `ambient`, `extreme`)
- Added benchmark suite:
  - `/Users/cleider/dev/pvx/benchmarks/run_bench.py`
  - `/Users/cleider/dev/pvx/benchmarks/metrics.py`
  - baseline gate file `/Users/cleider/dev/pvx/benchmarks/baseline_small.json`
- Added CI benchmark regression workflow:
  - `/Users/cleider/dev/pvx/.github/workflows/bench-regression.yml`
- Added/updated tests for transient/stereo behavior and new CLI paths.
- Refined CLI help taxonomy with explicit sections:
  - `I/O`, `Performance`, `Quality/Phase`, `Time/Pitch`, `Transients`, `Stereo`, `Output/Mastering`, `Debug`
  - consolidated duplicate `Time/Pitch` blocks into a single grouped section
- Added benchmark metric unit tests in `/Users/cleider/dev/pvx/tests/test_benchmark_metrics.py`.
- Tuned benchmark runner defaults in `/Users/cleider/dev/pvx/benchmarks/run_bench.py`:
  - new `--pvx-bench-profile {tuned,legacy}` switch
  - default `tuned` deterministic profile for stronger cycle-consistency quality metrics
- Expanded benchmark metrics in `/Users/cleider/dev/pvx/benchmarks/metrics.py` and reporting:
  - SNR, SI-SDR, spectral convergence, envelope correlation
  - RMS/crest deltas, bandwidth(95%) delta, ZCR delta, DC delta, clipping-ratio delta
- Updated benchmark gate baseline in `/Users/cleider/dev/pvx/benchmarks/baseline_small.json` to match tuned profile output.
- Added shared audio metrics table module `/Users/cleider/dev/pvx/src/pvx/core/audio_metrics.py`.
- Added non-silent ASCII input/output metrics table printing across pvx CLIs and `pvxvoc`.
- Expanded documentation:
  - `/Users/cleider/dev/pvx/docs/DEV_NOTES.md`
  - `/Users/cleider/dev/pvx/docs/QUALITY_GUIDE.md`
  - `/Users/cleider/dev/pvx/docs/RUBBERBAND_COMPARISON.md`
  - `/Users/cleider/dev/pvx/docs/BENCHMARKS.md`
  - refreshed `/Users/cleider/dev/pvx/README.md`, `/Users/cleider/dev/pvx/docs/GETTING_STARTED.md`, `/Users/cleider/dev/pvx/docs/EXAMPLES.md`

## Attribution
