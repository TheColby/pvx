# pvx Quality Guide

This guide maps audible artifacts to concrete `pvx voc` fixes.

`pvx` operating principle:
- quality first: artifacts and coherence issues are the primary constraints
- speed second: runtime tuning is applied only after quality is acceptable

## 1. Fast Starting Presets

| Material | Recommended preset | Why |
| --- | --- | --- |
| Vocal/speech | `--preset vocal_studio` | formant-aware pitch + hybrid transient handling |
| Drums/percussion | `--preset drums_safe` | WSOLA-focused transient path |
| Wide stereo mix | `--preset stereo_coherent` | channel-coupled coherence |
| Extreme ambient | `--preset extreme_ambient` | multistage long-stretch strategy |

## 2. Artifact -> Fix Table

| What you hear | Likely cause | Primary fixes |
| --- | --- | --- |
| Attack smear / soft transients | transient regions processed purely in PV mode | `--transient-mode hybrid` or `--transient-mode wsola`; raise `--transient-sensitivity`; reduce `--transient-crossfade-ms` |
| Phasy tone / chorus blur | weak phase coherence | `--phase-locking identity`; consider `--stereo-mode ref_channel_lock` with `--coherence-strength 0.7` |
| Stereo image wobble | per-channel phase drift | `--stereo-mode mid_side_lock` or `--stereo-mode ref_channel_lock --ref-channel 0` |
| Robotic vocal timbre after pitch shift | formant drift | `--pitch-mode formant-preserving`; tune `--formant-lifter` and `--formant-strength` |
| Pumping/flat loudness | aggressive mastering chain | reduce compressor/limiter settings; disable unused mastering stages |
| Grainy extreme stretch | ratio too high for single-stage path | `--stretch-mode multistage`; lower `--max-stage-stretch`; optionally `--multires-fusion` |
| Unexpected overs/clipped delivery files | output policy not constrained | set `--true-peak-max-dbtp` and target `--bit-depth`; use `--dither tpdf` for PCM exports |

## 3. Practical Recipes

### Speech stretch with transient protection

```bash
pvx voc speech.wav \
  --preset vocal_studio \
  --transient-mode hybrid \
  --stretch 1.3 \
  --output speech_clean_stretch.wav
```

### Drum-safe stretch

```bash
pvx voc drums.wav \
  --preset drums_safe \
  --time-stretch 1.4 \
  --output drums_safe.wav
```

### Stereo coherence lock

```bash
pvx voc mix.wav \
  --stereo-mode ref_channel_lock \
  --ref-channel 0 \
  --coherence-strength 0.9 \
  --stretch 1.2 \
  --output mix_locked.wav
```

## 4. Parameter Ranges That Usually Work

| Parameter | Typical range | Notes |
| --- | --- | --- |
| `--transient-sensitivity` | `0.45 .. 0.75` | higher = more transient regions |
| `--transient-protect-ms` | `20 .. 45` | increase for noisy or percussive content |
| `--transient-crossfade-ms` | `6 .. 14` | larger values smooth boundaries |
| `--coherence-strength` | `0.5 .. 0.95` | higher = stronger channel coupling |
| `--max-stage-stretch` | `1.2 .. 1.8` | lower values are safer for extreme ratios |

## 5. Debug Workflow

1. Run `pvx voc --example all` to pick a close recipe.
2. Use `--dry-run --explain-plan` to inspect resolved settings.
3. Adjust one control at a time (`transient-mode`, then `coherence-strength`, then phase/window settings).
4. Compare A/B using short excerpts before full renders.
5. Only after quality is acceptable, optimize runtime (`--n-fft`, staging, segmentation/checkpoint options).
