<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# pvx Phasiness Implementation Plan

## Why this paper matters

`About this Phasiness Business` isolates why classic phase-vocoder output sounds
"phasier" under stretch: weak cross-bin phase coherence, unstable peak-region
phase behavior, and frame-to-frame phase inconsistency in dense harmonic
material.

For `pvx`, this gives a quality-first roadmap:

- strengthen local phase locking around spectral peaks
- preserve vertical coherence (within a frame) and horizontal coherence
  (across frames)
- reduce phase random walk in low-energy bins
- make coherence strategy content-aware (speech/tonal/percussive)

## Benefits we can extract for pvx

1. Better harmonic solidity at moderate and high stretch ratios.
2. Lower "swirl/smear" in sustained vocals and pads.
3. More stable stereo image when coherence is enforced consistently across channels.
4. Cleaner transient neighborhoods when hybrid transient handling and phase resets
   are coordinated.
5. Stronger objective benchmark results on modulation- and coherence-sensitive metrics.

## Phased implementation

### Phase 1: Coherence-aware phase propagation in core

Files:
- `src/pvx/core/voc.py`
- `src/pvx/core/stereo.py`

Work:
- Add explicit "peak-region phase-locking radius" control.
- Lock neighboring bins to anchor-bin phase increments near detected peaks.
- Add low-energy-bin damping to avoid unstable phase diffusion.
- Keep deterministic branch for central processing unit (CPU) mode.

CLI additions:
- `--phase-lock-radius <bins>`
- `--phase-lock-strength <0..1>`
- `--phase-noise-floor-db <dB>`

### Phase 2: Adaptive coherence policy

Files:
- `src/pvx/core/presets.py`
- `src/pvx/core/transients.py`
- `src/pvx/core/voc.py`

Work:
- Auto-select coherence mode by frame class:
  - tonal frames: stronger identity/peak lock
  - noisy/percussive frames: relaxed lock + transient protection
- Fuse with transient-mode handling (`reset`, `hybrid`, `wsola`).

CLI additions:
- `--phase-policy {static,adaptive}`
- `--phase-tonal-lock <0..1>`
- `--phase-noise-lock <0..1>`

### Phase 3: Stereo/multichannel coherence extension

Files:
- `src/pvx/core/stereo.py`
- `src/pvx/metrics/coherence.py`

Work:
- Apply lock decisions in mid/side (M/S) domain for stereo-preserving stretch.
- Improve reference-channel lock to retain inter-channel phase deltas near peaks.
- Add diagnostics for per-band coherence drift.

CLI additions:
- `--coherence-report-json <path>`

### Phase 4: Benchmark and regression gates

Files:
- `benchmarks/metrics.py`
- `benchmarks/run_bench.py`
- `.github/workflows/bench-regression.yml`

Work:
- Add phasiness-focused metrics:
  - inter-frame phase-deviation variance
  - peak-region phase-consistency score
  - modulation-spectrum stability score
- Add strict pass/fail thresholds for tonal and speech subsets.

### Phase 5: UX and docs

Files:
- [README.md](../README.md)
- [docs/QUALITY_GUIDE.md](QUALITY_GUIDE.md)
- [docs/EXAMPLES.md](EXAMPLES.md)

Work:
- Add "phasiness reduction" intent preset.
- Add examples:
  - vocal stretch quality profile
  - ambient pad high-ratio profile
  - stereo-coherent mix-bus profile
- Add troubleshooting matrix: artifact -> parameter moves.

## Validation targets

- Lower modulation-spectrum distance on sustained harmonic tests.
- Lower stereo coherence drift on stereo test corpus.
- Audible A/B improvement on speech and vocal sustained vowels.
- No regression in `stretch=1.0` identity reconstruction tolerance.

## Source paper and cited references to track

- J. Laroche; M. Dolson (1997), *About this Phasiness Business*.
- M. Dolson (1986), *The phase vocoder: A tutorial*.
- D. W. Griffin; J. S. Lim (1984), *Signal estimation from modified short-time Fourier transform*.
- J. Laroche; M. Dolson (1997), *Phase-vocoder: About this phasiness business* (submitted manuscript citation in paper).
- E. Moulines; J. Laroche (1995), *Non-parametric techniques for pitch-scale and time-scale modification of speech*.
- M. S. Puckette (1995), *Phase-locked vocoder*.

## Attribution


