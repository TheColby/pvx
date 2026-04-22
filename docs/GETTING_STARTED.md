<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# Getting Started with pvx




This guide is for first-time users who want to understand what `pvx` does, why it exists, and how to get useful results without treating digital signal processing (DSP) as magic. It is practical first, mystical later.

## 0. Quick Setup (Install + PATH)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
pvx --help
```

Same setup with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv run pvx --help
```

If `pvx` is not found, add the project virtualenv to your path environment variable (`PATH`) (`zsh`):

```bash
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$(pwd)" >> ~/.zshrc
source ~/.zshrc
pvx --help
```

No-`PATH` fallback:

```bash
.venv/bin/pvx voc input.wav --stretch 1.2 --output output.wav
# or
python3 -m pvx.cli.pvx voc input.wav --stretch 1.2 --output output.wav
```

No-`PATH` fallback with `uv`:

```bash
uv run pvx voc input.wav --stretch 1.2 --output output.wav
```

Man-page generation:

```bash
python3 scripts/scripts_install_man_pages.py
MANPATH="$(pwd)/man:$MANPATH" man pvx
```

## 0.3 Launch-Ready Helper Commands

Before announcing or demoing, run these:

```bash
pvx doctor
pvx quickstart input.wav --output output.wav
pvx safe input.wav --material mix --output output_safe.wav
pvx transforms
pvx smoke --output smoke_out.wav
pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337
pvx augment-manifest validate aug_out/augment_manifest.jsonl --strict
```

What each does:
- `pvx doctor`: checks environment, path environment variable (`PATH`), and optional dependencies.
- `pvx quickstart`: prints the minimal launch command sequence.
- `pvx safe`: runs `pvx voc` with conservative quality-first defaults.
- `pvx transforms`: explains transform options and runtime availability.
- `pvx smoke`: executes a fast synthetic end-to-end sanity render.
- `pvx augment`: builds deterministic augmentation datasets and writes manifests for machine-learning pipelines.
- `pvx augment-manifest`: validates, merges, and summarizes augmentation manifests.

## 0.2 Running Any Command with uv

`uv` can run every command in this guide without changing the DSP arguments.

- `pvx ...` -> `uv run pvx ...`
- `python3 some_script.py ...` -> `uv run python3 some_script.py ...`
- `python3 -m module.path ...` -> `uv run python3 -m module.path ...`

Example:

```bash
pvx voc input.wav --stretch 1.25 --output out.wav
uv run pvx voc input.wav --stretch 1.25 --output out.wav
```

If you are already muttering at your shell, that is normal.

## 0.1 First 3 Commands to Run

```bash
# 1) Stretch without pitch change
pvx voc input.wav --stretch 1.25 --output input_stretch.wav

# 2) Pitch change without duration change
pvx voc input.wav --stretch 1.0 --pitch -3 --output input_pitch.wav

# 3) Same pitch shift, but with formant protection
pvx voc input.wav --stretch 1.0 --pitch -3 --pitch-mode formant-preserving --output input_pitch_formant.wav
```

What to listen for:
- `input_stretch.wav`: longer timing, same note center
- `input_pitch.wav`: lower note center, possible timbral darkening
- `input_pitch_formant.wav`: lower note center with more stable vowel/timbre identity
- if all three sound identical, double-check that you did not typo the output path at 2 a.m.

## 1. What Problem pvx Solves

Audio workflows often need one or more of the following, with controllable artifacts:
- change duration without changing pitch
- change pitch without changing duration
- apply time-varying pitch/time trajectories from a map
- preserve transients and formants where possible
- process many files consistently from the command line

`pvx` gives you explicit control over those operations using phase-vocoder/short-time Fourier transform (STFT) methods plus specialized companion tools (freeze, harmonize, morph, retune, denoise, dereverb).

Design priority:
- quality first: preserve musical intent and minimize artifacts
- speed second: optimize runtime after quality targets are achieved
- translation: we would rather be right than merely fast

## Analog Tape Methods for Pitch/Time Shifting

Historically, reel-to-reel varispeed coupled time and pitch. Analog systems like the Eltro information rate changer and related Anton Springer regulator concepts were early attempts to separate them using segmented rotating-head playback methods instead of simple fixed-speed tape transport.

Why this matters for `pvx`:
- artifact control is largely a continuity problem (then: segment stitching; now: phase/transient/stereo continuity),
- high-quality time/pitch work usually benefits from conservative, staged processing.

Sources:
- [Eltro information rate changer (Wikipedia)](https://en.wikipedia.org/wiki/Eltro_information_rate_changer)
- [Vintage Technologies: The Eltro and the Voice of HAL (Wendy Carlos)](https://www.wendycarlos.com/other/Eltro-1967/)
- [Anton Springer and the Time and Pitch Regulator (Sound and Science)](https://soundandscience.net/contributor-essays/anton-springer-and-the-time-and-pitch-regulator/)

## 2. Basic DSP Terms (Plain Language)

- Sample rate: how many audio samples per second (e.g. 48,000 Hz).
- Frame: a short window of audio processed as one block.
- STFT: repeated short Fourier transforms over overlapping frames.
- Bin: one frequency slot in an STFT frame.
- Window: taper applied to each frame to reduce spectral leakage.
- Hop size: frame advance between adjacent STFT frames.
- Phase coherence: consistency of phase evolution across frames/bins.
- Transient: short attack event (drum hit, consonant burst).
- Formant: resonant spectral envelope that carries vowel/timbre identity.

## 2.1 Supported File Types

`pvx` audio I/O is provided by `soundfile/libsndfile`, so exact format support depends on your runtime build.

- Quick summary: `wav`, `flac`, `aiff`, `ogg`, `caf` are supported for stream output and are safe defaults.
- Full current format table: [FILE_TYPES.md](FILE_TYPES.md)

## 3. Stretch vs Pitch Shift

- Stretch controls duration.
  - `stretch = 2.0` means output is twice as long.
  - `stretch = 0.5` means output is half as long.
- Pitch shift controls musical frequency position.
  - `+12 semitones` = one octave up.
  - `-12 semitones` = one octave down.

In `pvx voc`, duration and pitch can be controlled independently:

```bash
pvx voc input.wav --stretch 1.25 --pitch 0 --output stretched.wav
pvx voc input.wav --stretch 1.0 --pitch 3 --output pitched.wav
```

## 3.1 Denoising with Phase-Consistent STFT Processing

`pvx denoise` uses short-time Fourier transform (STFT) spectral subtraction while preserving phase continuity in overlap-add resynthesis.

Starter profiles:

```bash
# Speech-safe (conservative)
pvx denoise speech.wav --noise-seconds 0.4 --reduction-db 5 --floor 0.2 --smooth 9 --output speech_clean.wav

# Music-safe (preserve more ambience/harmonics)
pvx denoise mix.wav --noise-seconds 0.3 --reduction-db 4 --floor 0.25 --smooth 7 --output mix_clean.wav
```

If you hear chirpy/watery artifacts:
- reduce `--reduction-db`
- increase `--smooth`
- raise `--floor` slightly
- use `--noise-file roomtone.wav` when the beginning of the source is not a clean noise-only segment

## 3.2 Time-Varying Parameter Control (CSV/JSON)

You can pass a control file directly to many core phase-vocoder numeric flags.

Examples:

```bash
pvx voc input.wav --stretch controls/stretch.csv --interp linear --output out.wav
pvx voc input.wav --pitch-shift-ratio controls/pitch.json --interp polynomial --order 3 --output out.wav
pvx voc input.wav --n-fft controls/nfft.csv --hop-size controls/hop.csv --output out.wav
```

Common flags that accept scalar values or control files (`.csv` / `.json`):
- time/pitch trajectory: `--stretch`, `--time-stretch`, `--pitch-shift-ratio`, `--pitch-shift-semitones`, `--pitch-shift-cents`
- frame/spectral resolution: `--n-fft`, `--win-length`, `--hop-size`, `--kaiser-beta`
- phase/transient/stereo shaping: `--ambient-phase-mix`, `--transient-threshold`, `--transient-sensitivity`, `--transient-protect-ms`, `--transient-crossfade-ms`, `--coherence-strength`
- multistage/Fourier-sync tuning: `--extreme-stretch-threshold`, `--max-stage-stretch`, `--fourier-sync-min-fft`, `--fourier-sync-max-fft`, `--fourier-sync-smooth`
- onset/formant shaping: `--onset-credit-pull`, `--onset-credit-max`, `--formant-lifter`, `--formant-strength`, `--formant-max-gain-db`

Control interpolation options:
- `--interp none` (stairstep / no interpolation)
- `--interp linear` (default)
- `--interp nearest`
- `--interp cubic`
- `--interp polynomial --order N` (any integer `N >= 1`, default `N=3`; effective degree is `min(N, control_points-1)`)

Polynomial order examples:
- `--interp polynomial --order 1`
- `--interp polynomial --order 2`
- `--interp polynomial --order 3`
- `--interp polynomial --order 5`

Point-style CSV:

```csv
time_sec,value
0.0,1.0
0.5,1.2
1.0,1.8
```

Segment-style CSV:

```csv
start_sec,end_sec,value
0.0,0.4,1.0
0.4,0.8,1.2
0.8,1.2,1.6
```

JSON points:

```json
{
  "interpolation": "linear",
  "points": [
    {"time_sec": 0.0, "value": 1.0},
    {"time_sec": 0.5, "value": 1.2},
    {"time_sec": 1.0, "value": 1.8}
  ]
}
```

JSON schema quick reference:

| Key | Required | Type | Meaning |
| --- | --- | --- | --- |
| `interpolation` / `interp` | no | string | Interpolation override (`none`, `linear`, `nearest`, `cubic`, `exponential`, `s_curve`, `smootherstep`, `polynomial`) |
| `order` | no | integer | Polynomial order for `polynomial` interpolation |
| `points` | yes (point mode) | array | Time/value points |
| `points[].time_sec` | yes | number | Timestamp in seconds |
| `points[].value` | yes | number/string | Parameter value |
| `segments` | yes (segment mode) | array | Piecewise constant control regions |
| `segments[].start_sec`, `segments[].end_sec` | yes | number | Segment boundaries in seconds |
| `segments[].value` | yes | number/string | Segment value |
| `parameters` | no | object | Multi-parameter container keyed by parameter name |

Notes:
- per-parameter dynamic controls cannot be combined with legacy `--pitch-map` / `--pitch-map-stdin` in the same command
- dynamic `--time-stretch` cannot be combined with `--target-duration`

Interpolation graph examples:

| Mode | Example curve |
| --- | --- |
| `none (stairstep)` | ![none interpolation](assets/interpolation/interp_none.svg) |
| `nearest` | ![nearest interpolation](assets/interpolation/interp_nearest.svg) |
| `linear` | ![linear interpolation](assets/interpolation/interp_linear.svg) |
| `cubic` | ![cubic interpolation](assets/interpolation/interp_cubic.svg) |
| `exponential` | ![exponential interpolation](assets/interpolation/interp_exponential.svg) |
| `s_curve (smoothstep)` | ![s_curve interpolation](assets/interpolation/interp_s_curve.svg) |
| `smootherstep` | ![smootherstep interpolation](assets/interpolation/interp_smootherstep.svg) |
| `polynomial order 1` | ![polynomial order 1](assets/interpolation/interp_polynomial_order_1.svg) |
| `polynomial order 2` | ![polynomial order 2](assets/interpolation/interp_polynomial_order_2.svg) |
| `polynomial order 3` | ![polynomial order 3](assets/interpolation/interp_polynomial_order_3.svg) |
| `polynomial order 5` | ![polynomial order 5](assets/interpolation/interp_polynomial_order_5.svg) |

Core processing function charts:

| Function family | Graph |
| --- | --- |
| Pitch ratio vs semitones | ![pitch ratio vs semitones](assets/functions/pitch_ratio_vs_semitones.svg) |
| Dynamics transfer curves | ![dynamics transfer curves](assets/functions/dynamics_transfer_curves.svg) |
| Soft clip transfer functions | ![softclip transfer functions](assets/functions/softclip_transfer_functions.svg) |
| Morph blend magnitude curves | ![morph blend curves](assets/functions/morph_blend_magnitude_curves.svg) |
| Mask exponent response | ![mask exponent response](assets/functions/mask_exponent_curves.svg) |

## 4. Visual Mental Model of STFT

```text
Time-domain signal
x[n] ------------------------------------------------------------>

Frame + windowing
         [w[n] * x[n:n+N]]
              [w[n] * x[n+H:n+H+N]]
                   [w[n] * x[n+2H:n+2H+N]]

Per-frame transform
         FFT -> X0[k]
         FFT -> X1[k]
         FFT -> X2[k]

Process in spectral domain
         modify magnitude/phase trajectories

Synthesis
         IFFT + overlap-add -> y[n]
```

Key tradeoff:
- larger `N` (FFT) -> better frequency resolution, worse time localization
- smaller `N` -> better time localization, rougher low-frequency precision
- yes, this is the classic engineering compromise you were hoping to avoid

## 5. Example Audio Scenarios

- Dialogue cleanup + mild timing correction: `pvx voc` + `pvx denoise` + `pvx deverb`.
- Ambient texture from short sample: `pvx voc --preset ambient --target-duration ...`.
- Vocal harmonies: `pvx harmonize` with interval and pan controls.
- Timeline-locked effects: `pvx conform` / `pvx warp` with CSV maps.

### 5.1 Use-Case Matrix (Where to start quickly)

| Use case | First command to try |
| --- | --- |
| Speech slowdown for transcription | `pvx voc speech.wav --preset vocal_studio --stretch 1.25 --output speech_slow.wav` |
| Podcast timing cleanup | `pvx voc voice.wav --stretch 0.95 --output voice_tight.wav` |
| Vocal pitch correction | `pvx retune vocal.wav --scale major --root C --output-dir out --suffix _retune` |
| Harmonic backing voices | `pvx harmonize lead.wav --intervals 0,4,7 --gains 1,0.8,0.65 --output-dir out` |
| Unison widen synth | `pvx unison synth.wav --voices 7 --detune-cents 16 --width 1.1 --output-dir out` |
| Long ambient from short source | `pvx voc oneshot.wav --preset extreme_ambient --target-duration 600 --output ambient.wav` |
| Drum-safe stretch | `pvx voc drums.wav --preset drums_safe --stretch 1.3 --output drums_stretch.wav` |
| Stereo image stability | `pvx voc mix.wav --stretch 1.15 --stereo-mode mid_side_lock --coherence-strength 0.9 --output mix_lock.wav` |
| Noise reduction pre-pass | `pvx denoise field.wav --reduction-db 8 --output-dir out --suffix _den` |
| Room tail reduction | `pvx deverb room.wav --strength 0.5 --output-dir out --suffix _dry` |
| CSV time/pitch choreography | `pvx conform source.wav --map map_conform.csv --output-dir out` |
| Automated quality comparison | `uv run python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out --gate --baseline benchmarks/baseline_small.json` |

## 6. Step-by-Step Walkthrough (One Small WAV)

Assume `sample.wav` is about 2–5 seconds long.

### Step A: Baseline stretch

```bash
pvx voc sample.wav --stretch 1.2 --output sample_stretch.wav
```

Expected:
- 20% longer duration
- same pitch
- slight smearing possible on sharp transients

### Step B: Add transient protection

```bash
pvx voc sample.wav --stretch 1.2 --transient-preserve --phase-locking identity --output sample_stretch_transient.wav
```

Expected:
- attacks should feel tighter than Step A
- fewer “swishy” transients

### Step C: Pitch shift with formant preservation

```bash
pvx voc sample.wav --stretch 1.0 --pitch -4 --pitch-mode formant-preserving --output sample_pitch_formant.wav
```

Expected:
- pitch lowered by 4 semitones
- vowel/timbre identity more stable than plain shift

### Step D: Auto profile planning

```bash
pvx voc sample.wav --auto-profile --auto-transform --explain-plan
```

Expected:
- JSON plan with resolved profile, transform, and core processing settings
- no audio output (plan only)

## 7. What Outputs Should Sound Like

- `sample_stretch.wav`: same pitch, longer phrase timing.
- `sample_stretch_transient.wav`: similar to above, but cleaner attack edges.
- `sample_pitch_formant.wav`: lower note center with less “character collapse.”

If artifacts are strong:
- lower ratio magnitude
- increase overlap (`--hop-size` smaller)
- try preset `vocal` for speech/singing
- test `--multires-fusion` for mixed content

## 8. New Quality Modes (Beginner Version)

### Hybrid transient mode (recommended on speech/drums)

```bash
pvx voc sample.wav \
  --transient-mode hybrid \
  --transient-sensitivity 0.6 \
  --transient-protect-ms 30 \
  --transient-crossfade-ms 10 \
  --stretch 1.25 \
  --output sample_hybrid.wav
```

What to expect:
- steadier harmonic regions from phase-vocoder processing
- cleaner attacks from WSOLA handling around detected onsets

### Stereo coherence lock (recommended for wide stereo sources)

```bash
pvx voc stereo_mix.wav \
  --stereo-mode mid_side_lock \
  --coherence-strength 0.9 \
  --stretch 1.2 \
  --output stereo_locked.wav
```

What to expect:
- less left/right image wobble
- more stable center image after heavy stretch

## 9. Intent Presets

Use presets when you do not want to tune many flags:

- `--preset vocal_studio`: formant-aware vocal defaults + transient hybrid handling
- `--preset drums_safe`: WSOLA-heavy transient safety for percussive content
- `--preset extreme_ambient`: extreme long-form ambient settings
- `--preset stereo_coherent`: stereo coupling defaults

Legacy presets remain available: `vocal`, `ambient`, `extreme`.

## 10. Simpler One-Line Pipelines

If you do not want long Unix pipe chains, use managed helpers:

```bash
pvx follow guide.wav target.wav --output target_follow.wav --emit pitch_to_stretch --pitch-conf-min 0.75
pvx chain sample.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output sample_chain.wav
pvx stream sample.wav --output sample_stream.wav --chunk-seconds 0.2 --time-stretch 2.0
pvx stream sample.wav --mode wrapper --output sample_stream_wrapper.wav --chunk-seconds 0.2 --time-stretch 2.0
pvx stretch-budget sample.wav --disk-budget 20GB --bit-depth 16 --requested-stretch 1000000
```

- `pvx follow` replaces long sidechain pipes for pitch/control-map-driven workflows.
- `pvx chain` runs serial stages with managed intermediate files.
- `pvx stream` defaults to a stateful chunk processor for smoother long-form continuity.
- `pvx stream --mode wrapper` keeps legacy segmented-wrapper behavior.
- `pvx stretch-budget` estimates maximum practical stretch before you commit to a long render.

## 10.1 Estimate Stretch Budget Before Extreme Jobs

Use this helper before very large ratios (`100x`, `1000x`, `1000000x`) so disk limits are explicit.

```bash
pvx stretch-budget input.wav --disk-budget 20GB --bit-depth 16
pvx stretch-budget input.wav --disk-budget 20GB --requested-stretch 1000000 --fail-if-exceeds --json
```

What it uses:
- input shape (frames/channels/sample rate)
- output storage assumption (`--output-format`, `--bit-depth` / `--subtype`)
- budget (`--disk-budget` or free space at `--budget-path`)
- headroom (`--safety-margin`, default `0.90`)

Recommendation:
- for production, prefer `--target-duration` over arbitrary huge ratios.
- if you run extreme jobs, combine `--stretch-mode multistage` with `--auto-segment-seconds`, `--checkpoint-dir`, and `--resume`.

## 10.2 Feature Tracking for Sidechain Control

`pvx pitch-track` now emits feature vectors (not just pitch map fields), including:
- pitch/voicing (`f0_hz`, `pitch_ratio`, `confidence`, `voicing_prob`, `pitch_stability`)
- loudness/dynamics (`rms`, `rms_db`, `short_lufs_db`, `crest_factor_db`)
- spectral features (`spectral_centroid_hz`, `spectral_flatness`, `spectral_flux`, `rolloff_hz`)
- formants and cepstra (`formant_f1_hz..formant_f3_hz`, `mfcc_01..mfcc_N`)
- rhythm markers (`tempo_bpm`, `beat_phase`, `downbeat_phase`, `onset_strength`, `transient_mask`)
- MPEG-7-style descriptors (`mpeg7_*` columns, including audio spectrum envelope bands)

Use with control-bus routes:

```bash
pvx pitch-track guide.wav --feature-set all --mfcc-count 13 --output - \
  | pvx voc target.wav --control-stdin \
      --route pitch_ratio=affine(mfcc_01,0.002,1.0) \
      --route pitch_ratio=clip(pitch_ratio,0.5,2.0) \
      --route stretch=affine(spectral_flux,0.03,1.0) \
      --route stretch=clip(stretch,0.85,1.5) \
      --output target_feature_follow.wav
```

For a larger gallery (single-feature, multi-feature, MFCC/MPEG-7 vector, and multi-guide workflows), see:
- [docs/FEATURE_SIDECHAIN_EXAMPLES.md](FEATURE_SIDECHAIN_EXAMPLES.md)

You can also print built-in command snippets directly from the command-line interface (CLI):

```bash
pvx follow --example
pvx follow --example all
pvx follow --example formant_onset
```

## 11. Output Policy Controls

All audio-output tools now share deterministic output policy flags:

- `--bit-depth {inherit,16,24,32f}`
- `--dither {none,tpdf}` and `--dither-seed`
- `--true-peak-max-dbtp`
- `--metadata-policy {none,sidecar,copy}`
- `--subtype` for explicit low-level output subtype override

## 12. Next Steps

- Run practical recipes and advanced use cases (72+): [docs/EXAMPLES.md](EXAMPLES.md)
- Learn architecture and DSP diagrams (26+): [docs/DIAGRAMS.md](DIAGRAMS.md)
- Study equation-level behavior (31 sections): [docs/MATHEMATICAL_FOUNDATIONS.md](MATHEMATICAL_FOUNDATIONS.md)
- Use Python API directly: [docs/API_OVERVIEW.md](API_OVERVIEW.md)
- Use benchmark runner: `benchmarks/run_bench.py`

## 13. Extra Beginner Recipes (Quick Wins)

```bash
# Speech slowdown
pvx voc speech.wav --preset vocal_studio --stretch 1.30 --output speech_slow.wav

# Drum-safe stretch
pvx voc drums.wav --preset drums_safe --stretch 1.20 --output drums_safe.wav

# Stereo coherence lock
pvx voc mix.wav --stretch 1.2 --stereo-mode mid_side_lock --coherence-strength 0.9 --output mix_lock.wav

# Freeze a transient into a pad
pvx freeze hit.wav --freeze-time 0.22 --duration 12 --output hit_pad.wav

# Morph two sources
pvx morph a.wav b.wav --alpha 0.45 --blend-mode carrier_a_envelope_b --output a_b_morph.wav

# True A->B trajectory morph in one command
pvx morph A.wav B.wav --alpha controls/alpha_curve.csv --interp linear --blend-mode linear --output A_to_B_morph.wav

# Major-scale retune
pvx retune vocal.wav --root C --scale major --strength 0.8 --output vocal_c_major.wav

# Alternate concert pitch retune (A4 = 432 Hz)
pvx retune vocal.wav --root A --scale minor --a4-reference-hz 432 --output vocal_a432.wav

# Explicit root fundamental retune (C4 ~= 261.6256 Hz)
pvx retune vocal.wav --root-hz 261.6256 --scale major --output vocal_c4_root.wav

# Auto-recommend root fundamental from the source
pvx retune vocal.wav --recommend-root --scale minor --output vocal_auto_root.wav

# Denoise then dereverb
pvx denoise noisy.wav --reduction-db 8 --stdout | pvx deverb - --strength 0.3 --output noisy_clean.wav

# Denoise then stretch
pvx denoise noisy.wav --reduction-db 6 --stdout | pvx voc - --stretch 2.0 --output noisy_clean_stretch.wav

# Generate an LFO control map (triangle) and apply it as time-varying stretch
pvx lfo --wave triangle --duration 12 --frequency-hz 0.4 --center 1.0 --amplitude 0.2 --key stretch --output controls/stretch_tri.csv
pvx voc input.wav --stretch controls/stretch_tri.csv --interp linear --output input_tri_stretch.wav
```

Run them in order, listen after each step, and resist changing ten parameters at once unless chaos is the objective.

## Attribution


