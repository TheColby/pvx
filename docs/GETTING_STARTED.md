# Getting Started with pvx

This guide is for first-time users who want to understand what `pvx` does, why it exists, and how to get useful results without treating DSP as magic.

Recommended command surface:
- use `pvx <tool>` for all new workflows (`pvx voc`, `pvx freeze`, `pvx harmonize`, etc.)
- legacy script wrappers (`python3 pvxvoc.py`, `python3 pvxfreeze.py`, ...) remain available for compatibility

## 1. What Problem pvx Solves

Audio workflows often need one or more of the following, with controllable artifacts:
- change duration without changing pitch
- change pitch without changing duration
- apply time-varying pitch/time trajectories from a map
- preserve transients and formants where possible
- process many files consistently from the command line

`pvx` gives you explicit control over those operations using phase-vocoder/STFT methods plus specialized companion tools (freeze, harmonize, morph, retune, denoise, dereverb).

Design priority:
- quality first: preserve musical intent and minimize artifacts
- speed second: optimize runtime after quality targets are achieved

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
- Full current format table: `FILE_TYPES.md`

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
| Automated quality comparison | `python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out --gate --baseline benchmarks/baseline_small.json` |

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
pvx chain sample.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output sample_chain.wav
pvx stream sample.wav --output sample_stream.wav --chunk-seconds 0.2 --time-stretch 2.0
```

- `pvx chain` runs serial stages with managed intermediate files.
- `pvx stream` wraps `pvx voc` chunk/segment controls for long renders.

## 11. Output Policy Controls

All audio-output tools now share deterministic output policy flags:

- `--bit-depth {inherit,16,24,32f}`
- `--dither {none,tpdf}` and `--dither-seed`
- `--true-peak-max-dbtp`
- `--metadata-policy {none,sidecar,copy}`
- `--subtype` for explicit low-level output subtype override

## 12. Next Steps

- Run practical recipes and advanced use cases (65+): `docs/EXAMPLES.md`
- Learn architecture and DSP diagrams (26+): `docs/DIAGRAMS.md`
- Study equation-level behavior (31 sections): `docs/MATHEMATICAL_FOUNDATIONS.md`
- Use Python API directly: `docs/API_OVERVIEW.md`
- Use benchmark runner: `benchmarks/run_bench.py`
