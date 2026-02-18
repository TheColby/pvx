# pvx

`pvx` is a Python toolkit for phase-vocoder and STFT-based audio processing.

Primary project goal and differentiator:
- audio quality first (phase coherence, transient integrity, formant stability, stereo coherence)
- speed second (throughput/runtime tuning only after quality targets are met)

It includes focused CLI tools (`pvxvoc`, `pvxfreeze`, `pvxharmonize`, `pvxretune`, `pvxmorph`, etc.), shared mastering controls, CSV-driven automation paths, microtonal support, and optional GPU acceleration.

## 30-Second Quick Start

```bash
python3 -m pip install -e .
python3 pvxvoc.py input.wav --stretch 1.20 --output output.wav
```

What this does:
- reads `input.wav`
- stretches duration by 20%
- writes `output.wav`

## 5-Minute Tutorial (Single-File Workflow)

Use one file (`voice.wav`) and run three common operations.

1. Inspect available presets/examples:

```bash
python3 pvxvoc.py --example all
```

2. Time-stretch only:

```bash
python3 pvxvoc.py voice.wav --stretch 1.30 --output voice_stretch.wav
```

3. Pitch-shift only (duration unchanged):

```bash
python3 pvxvoc.py voice.wav --stretch 1.0 --pitch -3 --output voice_down3st.wav
```

4. Pitch-shift with formant preservation:

```bash
python3 pvxvoc.py voice.wav --stretch 1.0 --pitch -3 --pitch-mode formant-preserving --output voice_down3st_formant.wav
```

5. Quick A/B check:
- `voice_down3st.wav` should sound darker/slower-formant (“larger” vocal tract impression).
- `voice_down3st_formant.wav` should keep vowel identity more stable.

## Conceptual Overview: What Is a Phase Vocoder?

A phase vocoder processes short overlapping windows of audio in the frequency domain.

Analysis STFT:

$$
X_t[k] = \sum_{n=0}^{N-1} x[n+tH_a]w[n]e^{-j2\pi kn/N}
$$

Synthesis uses phase propagation to keep bins temporally coherent while changing the synthesis hop.

$$
\Delta\phi_t[k] = \mathrm{princarg}(\phi_t[k]-\phi_{t-1}[k]-\omega_kH_a)
$$

$$
\hat\phi_t[k] = \hat\phi_{t-1}[k] + \omega_kH_s + \Delta\phi_t[k]
$$

Pitch ratio mappings:

$$
r = 2^{\Delta s/12} = 2^{\Delta c/1200}
$$

Where:
- $N$: frame size (`--n-fft`)
- $H_a$: analysis hop
- $H_s$: synthesis hop (effective stretch control)
- $w[n]$: window (`--window`)

## When To Use Which Tool (Decision Tree)

```text
Start
 |
 +-- Need general time/pitch processing on one file or batch?
 |    -> pvxvoc.py
 |
 +-- Need sustained spectral drone from one instant?
 |    -> pvxfreeze.py
 |
 +-- Need stacked harmony voices from one source?
 |    -> pvxharmonize.py
 |
 +-- Need timeline-constrained pitch/time map from CSV?
 |    -> pvxconform.py / pvxwarp.py
 |
 +-- Need morphing between two sources?
 |    -> pvxmorph.py
 |
 +-- Need monophonic retune to scale/root?
 |    -> pvxretune.py
 |
 +-- Need denoise or dereverb cleanup?
      -> pvxdenoise.py / pvxdeverb.py
```

## Common Workflows

| Goal | Tool | Minimal command |
| --- | --- | --- |
| Vocal retune / timing correction | `pvxvoc.py` | `python3 pvxvoc.py vocal.wav --preset vocal --stretch 1.05 --pitch -1 --output vocal_fix.wav` |
| Sound-design freeze pad | `pvxfreeze.py` | `python3 pvxfreeze.py hit.wav --freeze-time 0.12 --duration 10 --output-dir out` |
| Tempo stretch with transient care | `pvxvoc.py` | `python3 pvxvoc.py drums.wav --stretch 1.2 --transient-preserve --phase-locking identity --output drums_120.wav` |
| Harmonic layering | `pvxharmonize.py` | `python3 pvxharmonize.py lead.wav --intervals 0,4,7 --gains 1,0.8,0.7 --output-dir out` |
| Cross-source morphing | `pvxmorph.py` | `python3 pvxmorph.py a.wav b.wav --alpha 0.5 --output morph.wav` |

More complete examples and use-case playbooks (65+ runnable recipes): `docs/EXAMPLES.md`

## Supported File Types

| Category | Supported types |
| --- | --- |
| Audio file input/output | All formats provided by the active `soundfile/libsndfile` build |
| Stream output (`--stdout`) | `wav`, `flac`, `aiff`/`aif`, `ogg`/`oga`, `caf` |
| Control maps | `csv` |
| Run manifests | `json` |
| Generated docs | `html`, `pdf` |

Full table of all currently supported audio container types: `docs/FILE_TYPES.md`

## Performance and GPU (Quality-First)

`pvx` is not tuned as a "fastest possible at any cost" engine. Start from quality-safe defaults, validate artifact levels, then reduce runtime where acceptable.

### CPU path
- default path is robust and portable
- use power-of-two FFT sizes first (`1024`, `2048`, `4096`, `8192`) for stable transform behavior and good throughput

### CUDA path

```bash
python3 pvxvoc.py input.wav --device cuda --stretch 1.1 --output out_cuda.wav
```

Short aliases:
- `--gpu` means `--device cuda`
- `--cpu` means `--device cpu`

### Quality-First tuning checklist
- start with quality controls first: `--phase-locking identity`, transient protection, stereo coherence mode
- choose larger `--n-fft` when low-frequency clarity matters; only reduce `--n-fft` when quality remains acceptable
- use `--multires-fusion` when it audibly improves content; disable only if quality is unchanged
- after artifact checks, optimize runtime via `--auto-segment-seconds` + `--checkpoint-dir` + `--resume`

## CLI Discoverability and UX

`pvxvoc` now includes beginner UX features:

- Intent presets:
  - Legacy: `--preset none|vocal|ambient|extreme`
  - New: `--preset default|vocal_studio|drums_safe|extreme_ambient|stereo_coherent`
- Example mode: `--example basic` (or `--example all`)
- Guided mode: `--guided` (interactive prompts)
- Grouped help sections for discoverability:
  - `I/O`, `Performance`, `Quality/Phase`, `Time/Pitch`, `Transients`, `Stereo`, `Output/Mastering`, `Debug`
- Beginner aliases:
  - `--stretch` -> `--time-stretch`
  - `--pitch` / `--semitones` -> `--pitch-shift-semitones`
  - `--cents` -> `--pitch-shift-cents`
  - `--ratio` -> `--pitch-shift-ratio`
  - `--out` -> `--output`
  - `--gpu` / `--cpu` -> device shortcut

Plan/debug aids:
- `--auto-profile`
- `--auto-transform`
- `--explain-plan`
- `--manifest-json`

New quality controls:
- Hybrid transient engine:
  - `--transient-mode off|reset|hybrid|wsola`
  - `--transient-sensitivity`
  - `--transient-protect-ms`
  - `--transient-crossfade-ms`
- Stereo/multichannel coherence:
  - `--stereo-mode independent|mid_side_lock|ref_channel_lock`
  - `--ref-channel`
  - `--coherence-strength`

Runtime metrics visibility:
- Unless `--silent` is used, pvx tools now print an ASCII metrics table for input/output audio
  - sample rate, channels, duration, peak/RMS/crest, DC offset, ZCR, clipping %, spectral centroid, 95% bandwidth
  - plus an input-vs-output comparison table with `input`, `output`, and `delta(out-in)` columns: SNR, SI-SDR, LSD, modulation distance, spectral convergence, envelope correlation, transient smear, loudness/true-peak, and stereo drift metrics

## Benchmarking (pvx vs Rubber Band vs librosa)

Run a tiny benchmark (cycle-consistency metrics):

```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out
```

This uses the tuned deterministic profile by default (`--pvx-bench-profile tuned`).
Use `--pvx-bench-profile legacy` to compare against the prior pvx benchmark settings.

Interpret benchmark priorities:
- quality metrics are primary acceptance criteria
- runtime is tracked as a secondary engineering metric

Reported metrics now include:
- LSD, modulation spectrum distance, transient smear, stereo coherence drift
- SNR, SI-SDR, spectral convergence, envelope correlation
- RMS delta, crest-factor delta, bandwidth(95%) delta, ZCR delta, DC delta, clipping-ratio delta
- Perceptual/intelligibility: PESQ, STOI, ESTOI, ViSQOL MOS-LQO, POLQA MOS-LQO, PEAQ ODG
- Loudness/mastering: integrated LUFS delta, short-term LUFS delta, LRA delta, true-peak delta
- Pitch/harmonic: F0 RMSE (cents), voicing F1, HNR drift
- Transient timing: onset precision/recall/F1, attack-time error
- Spatial/stereo: ILD drift, ITD drift, inter-channel phase deviation (low/mid/high/mean)
- Artifact-focused: phasiness index, musical-noise index, pre-echo score

Notes:
- Some perceptual standards require external/proprietary tools. When unavailable, pvx reports deterministic proxy estimates and includes a `Proxy Fraction` in benchmark markdown.
- External hooks are supported via environment variables:
  - `VISQOL_BIN`
  - `POLQA_BIN`
  - `PEAQ_BIN`

Run with regression gate against committed baseline:

```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out --baseline benchmarks/baseline_small.json --gate
```

## Visual Documentation

See `docs/DIAGRAMS.md` for:
- expanded architecture and DSP atlas (Mermaid + ASCII)
- quality-first tuning and metrics-flow diagrams
- STFT analysis/resynthesis timelines
- phase propagation and phase-locking diagrams
- hybrid transient/WSOLA/stitching diagrams
- stereo coherence mode diagrams
- map/segment and checkpoint/resume diagrams
- benchmark and CI gate flow diagrams
- mastering chain and troubleshooting decision trees

## Troubleshooting

### “No readable input files matched…”
- verify path and extension
- quote globs in shells if needed
- run `python3 pvxvoc.py --guided`

### Output sounds “phasier” or “smear-y”
- enable `--phase-locking identity`
- enable `--transient-preserve`
- reduce stretch ratio, or use `--stretch-mode multistage`

### Speech sounds robotic after pitch shift
- use `--pitch-mode formant-preserving`
- reduce semitone magnitude
- increase overlap (`--hop-size` smaller relative to `--win-length`)

### CUDA requested but falls back
- ensure CuPy install matches your CUDA runtime
- test with `--device cuda` to force explicit failure if unavailable

### Long extreme render interrupted
- rerun with `--checkpoint-dir ... --resume`
- consider `--auto-segment-seconds 0.25` to reduce recompute scope

## FAQ

### Can pvx time-stretch and time-compress?
Yes. `--stretch > 1` lengthens, `--stretch < 1` shortens.

### Can I shift pitch without changing duration?
Yes. Use pitch flags with `--stretch 1.0`, e.g. `--pitch`, `--cents`, or `--ratio`.

### Can I chain tools in one shell line?
Yes. Use `--stdout` and `-` input on downstream tools.

```bash
python3 pvxvoc.py input.wav --stretch 1.1 --stdout \
  | python3 pvxdenoise.py - --reduction-db 10 --stdout \
  | python3 pvxdeverb.py - --strength 0.4 --output cleaned.wav
```

### Does pvx support microtonal workflows?
Yes. Use ratio/cents/semitone controls and CSV map modes.

### Is every algorithm phase-vocoder-based?
No. The repo includes non-phase-vocoder modules too (analysis, denoise, dereverb, decomposition, etc.).

## Why This Matters

### How pvx differs from `librosa` and Rubber Band
- `librosa` is a broad analysis library; pvx is an operational CLI toolkit with production-style pipelines, shared mastering chain, map-based automation, and extensive command-line workflows.
- Rubber Band is a strong dedicated stretcher; pvx emphasizes inspectable Python implementations, explicit transform/window control, CSV-driven control maps, integrated multi-tool workflows, and a quality-first tuning philosophy.

### Why phase coherence matters
Unconstrained phase across bins/frames causes audible blur, chorus-like instability, and transient damage. Phase locking and transient-aware logic reduce these failures.

### When transient preservation matters
Most for drums, consonants, plosives, and percussive attacks. Less critical for smooth pads and static drones.

### When NOT to use a phase vocoder
- strong transient-critical material with very large ratio changes may prefer waveform/granular strategies
- extremely low-latency live paths may prefer simpler time-domain methods
- if your target is artifact-heavy texture, stochastic engines may be preferable to strict phase coherence

## Progressive Documentation Map

- Onboarding: `docs/GETTING_STARTED.md`
- Example cookbook (65+ runnable commands): `docs/EXAMPLES.md`
- Diagram atlas (26+ architecture/DSP diagrams): `docs/DIAGRAMS.md`
- Mathematical foundations (31 sections of equations + derivations): `docs/MATHEMATICAL_FOUNDATIONS.md`
- API usage from Python: `docs/API_OVERVIEW.md`
- File types and formats: `docs/FILE_TYPES.md`
- Quality troubleshooting guide: `docs/QUALITY_GUIDE.md`
- Rubber Band comparison notes: `docs/RUBBERBAND_COMPARISON.md`
- Benchmark guide: `docs/BENCHMARKS.md`
- Window reference: `docs/WINDOW_REFERENCE.md`
- Generated HTML docs: `docs/html/index.html`
- PDF bundle: `docs/pvx_documentation.pdf`

## Install

```bash
python3 -m pip install -e .
```

Optional CUDA:

```bash
python3 -m pip install cupy-cuda12x
```

## License

MIT
