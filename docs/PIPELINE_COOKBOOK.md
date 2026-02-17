# PVX Pipeline Cookbook

_Generated from commit `58b1a2e` (commit date: 2026-02-17T09:29:32-05:00)._

Curated one-line workflows for practical chaining, mastering, microtonal processing, and batch operation.

## Analysis/QA

### Quality metrics on processed speech

```bash
python3 -m pvx.algorithms.analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics clean.wav noisy.wav --output out/qa.json
```

Why: Collects objective quality indicators for regression tracking.

## Batch

### Batch stretch over folder

```bash
python3 pvxvoc.py stems/*.wav --time-stretch 1.08 --output-dir out/stems --overwrite
```

Why: Applies consistent transform to many files with one command.

### Dry-run output validation

```bash
python3 pvxdenoise.py takes/*.wav --reduction-db 8 --dry-run --output-dir out/preview
```

Why: Checks filename resolution and collisions without writing audio.

## Mastering

### Integrated loudness targeting with limiter

```bash
python3 pvxvoc.py mix.wav --time-stretch 1.0 --target-lufs -14 --compressor-threshold-db -20 --compressor-ratio 3 --limiter-threshold 0.98 --output-dir out --suffix _master
```

Why: Combines dynamics and loudness controls in shared mastering chain.

### Soft clip and hard safety ceiling

```bash
python3 pvxharmonize.py bus.wav --intervals 0,7,12 --mix 0.35 --soft-clip-level 0.92 --soft-clip-type tanh --hard-clip-level 0.99 --output-dir out
```

Why: Adds saturation while enforcing a strict final peak ceiling.

## Microtonal

### Custom cents map retune

```bash
python3 pvxretune.py vox.wav --root 60 --scale-cents 0,90,204,294,408,498,612,702,816,906,1020,1110 --strength 0.8 --output-dir out
```

Why: Maps incoming notes to a custom 12-degree microtonal scale.

### Conform CSV with per-segment ratios

```bash
python3 pvxconform.py solo.wav map_conform.csv --pitch-mode ratio --output-dir out --suffix _conform
```

Why: Applies timeline-specific time and pitch trajectories from CSV.

## Phase-vocoder core

### Moderate vocal stretch with formant preservation

```bash
python3 pvxvoc.py vocal.wav --time-stretch 1.15 --pitch-mode formant-preserving --output-dir out --suffix _pv
```

Why: Retains speech-like vowel envelope while stretching timing.

### Independent cents retune

```bash
python3 pvxvoc.py lead.wav --pitch-shift-cents -23 --time-stretch 1.0 --output-dir out --suffix _cents
```

Why: Applies precise microtonal offset without tempo change.

## Pipelines

### Time-stretch -> denoise -> dereverb in one pipe

```bash
python3 pvxvoc.py input.wav --time-stretch 1.25 --stdout | python3 pvxdenoise.py - --reduction-db 10 --stdout | python3 pvxdeverb.py - --strength 0.45 --output-dir out --suffix _clean
```

Why: Single-pass CLI chain for serial DSP in Unix pipes.

### Morph -> formant -> unison

```bash
python3 pvxmorph.py a.wav b.wav -o - | python3 pvxformant.py - --mode preserve --stdout | python3 pvxunison.py - --voices 5 --detune-cents 8 --output-dir out --suffix _morph_stack
```

Why: Builds a richer timbre chain with no intermediate files.

## Spatial

### VBAP adaptive panning via algorithm dispatcher

```bash
python3 -m pvx.algorithms.spatial_and_multichannel.imaging_and_panning.vbap_adaptive_panning input.wav --output-channels 6 --azimuth-deg 35 --width 0.8 --output out/vbap.wav
```

Why: Demonstrates algorithm-level spatial module invocation.

## Notes

- Use `--stdout`/`-` to chain tools without intermediate files.
- Add `--quiet` for script-driven runs; use default verbosity for live progress bars.
- For production mastering, validate true peaks and loudness after all nonlinear stages.
