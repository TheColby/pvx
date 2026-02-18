# pvx Pipeline Cookbook

_Generated from commit `23925ec` (commit date: 2026-02-17T16:15:08-05:00)._

Curated one-line workflows for practical chaining, mastering, microtonal processing, and batch operation.

## Analysis/QA

### Quality metrics on processed speech

```bash
python3 -m pvx.algorithms.analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics clean.wav noisy.wav --output out/qa.json
```

Why: Collects objective quality indicators for regression tracking.

## Automation

### A/B report generation

```bash
python3 scripts_ab_compare.py --input mix.wav --a-args "--time-stretch 1.1 --transform fft" --b-args "--time-stretch 1.1 --transform dct" --out-dir reports/ab --name fft_vs_dct
```

Why: Creates JSON/Markdown objective reports for fast algorithm and parameter comparisons.

### Benchmark matrix sweep

```bash
python3 scripts_benchmark_matrix.py --input mix.wav --transforms fft,dft,dct --windows hann,kaiser --n-ffts 1024,2048 --devices cpu --out-dir reports/bench
```

Why: Produces reproducible CSV/JSON runtime matrices across transform/window/FFT combinations.

### Quality regression check

```bash
python3 scripts_quality_regression.py --input mix.wav --output out/reg.wav --render-args "--time-stretch 1.2 --transform fft" --baseline-json reports/baseline.json --report-json reports/regression.json
```

Why: Compares current renders against baseline objective metrics with configurable tolerances.

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

### Extreme stretch with multistage strategy

```bash
python3 pvxvoc.py ambience.wav --target-duration 600 --ambient-preset --n-fft 16384 --win-length 16384 --hop-size 2048 --window kaiser --kaiser-beta 18 --output-dir out --suffix _ambient600x
```

Why: PaulStretch-style ambient profile for very large ratios using stochastic phase and onset time-credit controls.

### Ultra-smooth speech stretch (600x)

```bash
python3 pvxvoc.py speech.wav --target-duration 600 --stretch-mode standard --phase-engine propagate --phase-locking identity --n-fft 8192 --win-length 8192 --hop-size 256 --window hann --normalize peak --peak-dbfs -1 --compressor-threshold-db -30 --compressor-ratio 2.0 --compressor-attack-ms 25 --compressor-release-ms 250 --compressor-makeup-db 4 --limiter-threshold 0.95 --output-dir out --suffix _speech600x
```

Why: Prefers continuity and intelligibility over texture animation; avoids choppy stochastic artifacts on speech sources.

### Auto-profile plan preview

```bash
python3 pvxvoc.py input.wav --auto-profile --auto-transform --explain-plan
```

Why: Prints the resolved profile/config plan before long renders.

### Multi-resolution fusion stretch

```bash
python3 pvxvoc.py input.wav --multires-fusion --multires-ffts 1024,2048,4096 --multires-weights 0.2,0.35,0.45 --time-stretch 1.4 --output-dir out --suffix _multires
```

Why: Blends several FFT scales to reduce single-resolution bias on complex program material.

### Checkpointed long render with manifest

```bash
python3 pvxvoc.py long.wav --time-stretch 12 --auto-segment-seconds 0.5 --checkpoint-dir checkpoints --manifest-json reports/run_manifest.json --output-dir out --suffix _long
```

Why: Caches segment renders for resume workflows and writes run metadata for reproducibility.

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

### Pitch-follow sidechain map (A controls B)

```bash
python3 HPS-pitch-track.py A.wav | python3 pvxvoc.py B.wav --pitch-follow-stdin --pitch-conf-min 0.75 --pitch-lowconf-mode hold --time-stretch-factor 1.0 --output output.wav
```

Why: Tracks F0 contour from source A and applies it as a dynamic pitch-ratio control map on source B.

## Spatial

### VBAP adaptive panning via algorithm dispatcher

```bash
python3 -m pvx.algorithms.spatial_and_multichannel.imaging_and_panning.vbap_adaptive_panning input.wav --output-channels 6 --azimuth-deg 35 --width 0.8 --output out/vbap.wav
```

Why: Demonstrates algorithm-level spatial module invocation.

## Transform selection

### Default production backend (FFT + transient protection)

```bash
python3 pvxvoc.py mix.wav --transform fft --time-stretch 1.07 --transient-preserve --phase-locking identity --output-dir out --suffix _fft
```

Why: Use when you need the fastest and most stable general-purpose phase-vocoder path.

### Reference Fourier baseline using explicit DFT mode

```bash
python3 pvxvoc.py tone_sweep.wav --transform dft --time-stretch 1.00 --pitch-shift-semitones 0 --output-dir out --suffix _dft_ref
```

Why: Useful for parity checks and controlled transform-comparison experiments.

### Prime-size frame experiment with CZT backend

```bash
python3 pvxvoc.py archival_take.wav --transform czt --n-fft 1531 --win-length 1531 --hop-size 382 --time-stretch 1.03 --output-dir out --suffix _czt
```

Why: Alternative numerical path for awkward/prime frame sizes when validating edge cases.

### DCT timbral compaction for smooth harmonic material

```bash
python3 pvxvoc.py strings.wav --transform dct --pitch-shift-cents -18 --soft-clip-level 0.95 --output-dir out --suffix _dct
```

Why: Real-basis coefficients can emphasize envelope-like structure for creative reshaping.

### DST odd-symmetry color pass

```bash
python3 pvxvoc.py snare_loop.wav --transform dst --time-stretch 0.92 --phase-locking off --output-dir out --suffix _dst
```

Why: Provides an alternate real-basis artifact profile useful for creative percussive processing.

### Hartley real-basis exploratory render

```bash
python3 pvxvoc.py synth_pad.wav --transform hartley --time-stretch 1.30 --pitch-shift-semitones 3 --output-dir out --suffix _hartley
```

Why: Compares Hartley-domain behavior against complex FFT phase-vocoder output.

### A/B sweep of transform backends from shell loop

```bash
for t in fft dft czt dct dst hartley; do python3 pvxvoc.py voice.wav --transform "$t" --time-stretch 1.1 --output-dir out --suffix "_$t"; done
```

Why: Fast listening workflow for selecting the least-artifact transform on your source.

## Notes

- Use `--stdout`/`-` to chain tools without intermediate files.
- Add `--quiet` for script-driven runs; use default verbosity for live progress bars.
- For production mastering, validate true peaks and loudness after all nonlinear stages.
