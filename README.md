# pvx

`pvx` is a Python toolkit for high-quality time and pitch processing using a phase-vocoder/short-time Fourier transform (STFT) core.

Primary project goal and differentiator:
- audio quality first (phase coherence, transient integrity, formant stability, stereo coherence)
- speed second (throughput/runtime tuning only after quality targets are met)

It includes a unified command-line interface (CLI) (`pvx`) with focused subcommands (`voc`, `freeze`, `harmonize`, `retune`, `morph`, etc.), shared mastering controls, comma-separated values (CSV)-driven automation paths, microtonal support, and optional graphics processing unit (GPU) acceleration.

## Start Here: What pvx Solves

If you are new to digital signal processing (DSP)/audio engineering, think of `pvx` as a tool that lets you:
- make audio longer or shorter without changing musical key
- change key/pitch without changing playback speed
- preserve clarity while doing both (especially for vocals and stereo material)

`pvx` is built around the same class of methods used in modern time-stretch and pitch-shift workflows, but exposes them with practical CLI controls.

### Stretch vs Pitch Shift (Plain Language)

| Operation | What changes | What should stay the same |
| --- | --- | --- |
| Time stretch (`--stretch`) | Duration/tempo | Pitch/key |
| Pitch shift (`--pitch`, `--cents`, `--ratio`) | Pitch/key | Duration/tempo |
| Combined stretch + pitch | Both | Clarity, transients, stereo image (as much as possible) |

Example:
- `--stretch 2.0` means a 5-second sound becomes 10 seconds.
- `--pitch 12` means one octave up.
- `--pitch -12` means one octave down.

## What Is a Phase Vocoder? (No Math Version)

A phase vocoder is a way to process sound in very short overlapping slices.

For each slice:
1. It measures "how much of each frequency is present" and "where its phase is".
2. It modifies timing and/or pitch in that spectral representation.
3. It rebuilds audio from overlapping slices.

Why "phase" matters:
- If magnitudes are changed without consistent phase evolution, output can sound smeared, chorus-like, metallic, or unstable.
- Good phase handling keeps tones continuous across frames and improves naturalness.

In practical terms, `pvx` gives you controls for this quality layer:
- phase locking
- transient protection/hybrid modes
- stereo coherence modes
- formant-aware pitch workflows

## Mental Model (1 Minute)

Input waveform -> short overlapping frames -> frequency-domain edit -> overlap-add resynthesis -> output waveform

Useful intuition:
- window size (`--n-fft` / `--win-length`) trades time detail vs frequency detail
- hop size (`--hop-size`) controls frame overlap density
- larger windows often help low-frequency tonal stability
- transient handling is important for drums/plosives/onsets

## 30-Second Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
pvx voc input.wav --stretch 1.20 --output output.wav
```

If `pvx` is not found after install, add the virtualenv binaries to your shell path environment variable (`PATH`):

```bash
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$(pwd)" >> ~/.zshrc
source ~/.zshrc
pvx --help
```

If you do not want to modify the path environment variable (`PATH`), run the same command through the repository wrapper:

```bash
python3 pvx.py voc input.wav --stretch 1.20 --output output.wav
```

What this does:
- reads `input.wav`
- stretches duration by 20%
- writes `output.wav`

If you prefer direct script wrappers, legacy commands are still supported (`python3 pvxvoc.py ...`, `python3 pvxfreeze.py ...`, etc.).

## Unified CLI (Primary Entry Point)

`pvx` is now the recommended command surface for first-time users.

```bash
pvx list
pvx help voc
pvx examples basic
pvx guided
pvx chain --example
pvx stream --example
```

You can also use a convenience shortcut for the default vocoder path:

```bash
pvx input.wav --stretch 1.20 --output output.wav
```

This is equivalent to:

```bash
pvx voc input.wav --stretch 1.20 --output output.wav
```

## 5-Minute Tutorial (Single-File Workflow)

Use one file (`voice.wav`) and run three common operations.

1. Inspect available presets/examples:

```bash
pvx voc --example all
```

2. Time-stretch only:

```bash
pvx voc voice.wav --stretch 1.30 --output voice_stretch.wav
```

3. Pitch-shift only (duration unchanged):

```bash
pvx voc voice.wav --stretch 1.0 --pitch -3 --output voice_down3st.wav
```

4. Pitch-shift with formant preservation:

```bash
pvx voc voice.wav --stretch 1.0 --pitch -3 --pitch-mode formant-preserving --output voice_down3st_formant.wav
```

5. Quick A/B check:
- `voice_down3st.wav` should sound darker/slower-formant (“larger” vocal tract impression).
- `voice_down3st_formant.wav` should keep vowel identity more stable.

## Conceptual Overview: What Is a Phase Vocoder?

A phase vocoder uses the **short-time Fourier transform (STFT)** to repeatedly answer this question:
"What frequencies are present in this tiny time slice, and how do their phases evolve from one slice to the next?"

The core workflow is:
1. Split audio into overlapping frames.
2. Apply a window function to each frame.
3. Transform each frame into spectral bins (magnitude + phase).
4. Modify timing/pitch by controlling phase progression and synthesis hop.
5. Reconstruct audio by overlap-adding all processed frames.

If you are new to this, the key idea is that **phase continuity between frames** is what separates high-quality output from "phasiness" artifacts.

### 1) Analysis STFT

`pvx` analyzes each frame with:

$$
X_t[k] = \sum_{n=0}^{N-1} x[n+tH_a]w[n]e^{-j2\pi kn/N}
$$

where:
- $x[n]$ represents the input signal sample at index $n$
- $t$ represents frame index
- $k$ represents frequency-bin index
- $N$ represents frame size (`--n-fft`)
- $H_a$ represents analysis hop
- $w[n]$ represents the selected window (`--window`)

Plain-English meaning:
- each frame is windowed, then transformed
- output bin $X_t[k]$ is complex-valued (magnitude and phase)
- this gives the per-frame spectral state used by downstream processing

### 2) Phase-Vocoder Propagation

Time stretching is controlled by phase evolution:

$$
\Delta\phi_t[k] = \mathrm{princarg}(\phi_t[k]-\phi_{t-1}[k]-\omega_kH_a)
$$
$$
\hat\phi_t[k] = \hat\phi_{t-1}[k] + \omega_kH_s + \Delta\phi_t[k]
$$

where:
- $\phi_t[k]$ is observed phase at frame $t$, bin $k$
- $\hat\phi_t[k]$ is synthesized/output phase
- $\omega_k$ is nominal bin center frequency in radians/sample
- $H_s$ is synthesis hop (effective time-stretch control)
- $\mathrm{princarg}(\cdot)$ wraps phase to $(-\pi, \pi]$

Plain-English meaning:
- first estimate the true per-bin phase advance
- then re-accumulate phase using a new synthesis hop
- this lets duration change while preserving spectral continuity

### 3) Pitch Mapping

Pitch controls map musical intervals to ratio:

$$
r = 2^{\Delta s/12} = 2^{\Delta c/1200}
$$

where:
- $r$ is pitch ratio
- $\Delta s$ is semitone shift (`--pitch`)
- $\Delta c$ is cents shift (`--cents`)

Practical interpretation:
- $r > 1$: pitch up
- $r < 1$: pitch down
- formant options control whether vocal timbre shifts with pitch or is preserved

## When To Use Which Tool (Decision Tree)

```text
Start
 |
 +-- Need general time/pitch processing on one file or batch?
 |    -> pvx voc
 |
 +-- Need sustained spectral drone from one instant?
 |    -> pvx freeze
 |
 +-- Need stacked harmony voices from one source?
 |    -> pvx harmonize
 |
 +-- Need timeline-constrained pitch/time map from CSV?
 |    -> pvx conform / pvx warp
 |
 +-- Need morphing between two sources?
 |    -> pvx morph
 |
 +-- Need monophonic retune to scale/root?
 |    -> pvx retune
 |
 +-- Need denoise or dereverb cleanup?
      -> pvx denoise / pvx deverb
```

## Common Workflows

| Goal | Tool | Minimal command |
| --- | --- | --- |
| Vocal retune / timing correction | `pvx voc` | `pvx voc vocal.wav --preset vocal --stretch 1.05 --pitch -1 --output vocal_fix.wav` |
| Sound-design freeze pad | `pvx freeze` | `pvx freeze hit.wav --freeze-time 0.12 --duration 10 --output-dir out` |
| Tempo stretch with transient care | `pvx voc` | `pvx voc drums.wav --stretch 1.2 --transient-preserve --phase-locking identity --output drums_120.wav` |
| Harmonic layering | `pvx harmonize` | `pvx harmonize lead.wav --intervals 0,4,7 --gains 1,0.8,0.7 --output-dir out` |
| Cross-source morphing | `pvx morph` | `pvx morph a.wav b.wav --alpha 0.5 --output morph.wav` |

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
pvx voc input.wav --device cuda --stretch 1.1 --output out_cuda.wav
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

`pvx` now provides a single command surface for discovery (`pvx list`, `pvx help <tool>`, `pvx examples`, `pvx guided`), while `pvxvoc` retains advanced controls for detailed phase-vocoder workflows.

Additional helper workflows:
- `pvx chain`: managed multi-stage chains without manually wiring per-stage `--stdout` / `-` plumbing
- `pvx stream`: stateful chunk engine for long-form streaming workflows (`--mode stateful` default, `--mode wrapper` compatibility fallback)

`pvx voc` includes beginner UX features:

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
- Common output consistency:
  - shared tools now accept explicit single-file output via `--output` / `--out` in addition to `--output-dir` + `--suffix`

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

Output policy controls (shared across audio-output tools):
- `--bit-depth {inherit,16,24,32f}`
- `--dither {none,tpdf}` and `--dither-seed`
- `--true-peak-max-dbtp`
- `--metadata-policy {none,sidecar,copy}`
- `--subtype` remains available as explicit low-level override

## Benchmarking (pvx vs Rubber Band vs librosa)

Run a tiny benchmark (cycle-consistency metrics):

```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out
```

This uses the tuned deterministic profile by default (`--pvx-bench-profile tuned`).
Use `--pvx-bench-profile legacy` to compare against the prior pvx benchmark settings.

Stage 2 reproducibility controls:
- corpus manifest + hash validation: `--dataset-manifest`, `--strict-corpus`, `--refresh-manifest`
- deterministic CPU checks: `--deterministic-cpu`, `--determinism-runs`
- stronger gates: `--gate-row-level`, `--gate-signatures`
- automatic quality diagnostics are emitted in `report.md` and `report.json`

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
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out --strict-corpus --determinism-runs 2 --baseline benchmarks/baseline_small.json --gate --gate-row-level --gate-signatures
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
- run `pvx guided` (or `pvx voc --guided`)

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

For shorter one-liners without manual pipe wiring, use managed chain mode:

```bash
pvx chain input.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output output_chain.wav
```

For chunked long renders through the default stateful stream engine:

```bash
pvx stream input.wav --output output_stream.wav --chunk-seconds 0.2 --time-stretch 3.0
```

Compatibility fallback (legacy segmented-wrapper behavior):

```bash
pvx stream input.wav --mode wrapper --output output_stream.wav --chunk-seconds 0.2 --time-stretch 3.0
```

```bash
pvx voc input.wav --stretch 1.1 --stdout \
  | pvx denoise - --reduction-db 10 --stdout \
  | pvx deverb - --strength 0.4 --output cleaned.wav
```

### Does pvx support microtonal workflows?
Yes. Use ratio/cents/semitone controls and CSV map modes.

### Is every algorithm phase-vocoder-based?
No. The repo includes non-phase-vocoder modules too (analysis, denoise, dereverb, decomposition, etc.).

## Why This Matters

### How pvx differs from `librosa` and Rubber Band
- `librosa` is a broad analysis library; pvx is an operational CLI toolkit with research- and production-oriented pipelines, shared mastering chain, map-based automation, and extensive command-line workflows.
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
- Maintainer review checklist: `docs/HOW_TO_REVIEW.md`
- Generated HTML docs: `docs/html/index.html`
- PDF bundle: `docs/pvx_documentation.pdf`

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
pvx --help
```

Persist `pvx` on your shell path (`zsh`):

```bash
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$(pwd)" >> ~/.zshrc
source ~/.zshrc
pvx --help
```

Optional CUDA:

```bash
python3 -m pip install cupy-cuda12x
```

### Installation and Runtime Matrix

| Platform / Runtime | CPU mode | GPU/CUDA mode | Notes |
| --- | --- | --- | --- |
| Linux x86_64 | Supported | Supported (CUDA + CuPy) | Best choice for NVIDIA CUDA acceleration. |
| Windows x86_64 | Supported | Supported (CUDA + CuPy) | Match CuPy package to installed CUDA runtime. |
| macOS Intel | Supported | Not CUDA | Use CPU mode; Metal acceleration is not a CUDA path. |
| macOS Apple Silicon (M1/M2/M3/M4) | Supported (native arm64) | Not CUDA | Native Apple Silicon support in CPU path; prefer quality-focused profiles first. |

Primary command:

```bash
pvx voc input.wav --stretch 1.2 --output output.wav
```

Fallback without `PATH` updates:

```bash
python3 pvx.py voc input.wav --stretch 1.2 --output output.wav
```

Legacy wrappers remain available for backward compatibility.

## License

MIT
