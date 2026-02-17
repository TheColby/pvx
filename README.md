<p align="center"><br><br></p>

<p align="center">
  <img src="assets/pvx_logo.png" alt="pvx logo" width="360" />
</p>

<p align="center"><br><br></p>

# pvx

`pvx` is a [phase-vocoder](https://en.wikipedia.org/wiki/Phase_vocoder)-centric audio DSP toolkit focused on practical, production-style CLI workflows.
It combines [STFT](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)-based processing, microtonal retuning, optional GPU acceleration, and a large algorithm library behind consistent command-line interfaces.

## Why PVX Is Different

| Capability | What is unique in `pvx` |
| --- | --- |
| Native macOS architecture support | Designed for native builds on both [Apple Silicon](https://en.wikipedia.org/wiki/Apple_silicon) and Intel Macs (build each architecture separately for native binaries). |
| Optional GPU acceleration | [CUDA](https://developer.nvidia.com/cuda-toolkit) backend with `--device auto/cpu/cuda` and `--cuda-device`; automatic CPU fallback path when CUDA is unavailable. |
| Microtonal-ready workflows | Direct [cents](https://en.wikipedia.org/wiki/Cent_(music)), ratio, semitone, and custom `scale_cents` support across retune/formant/transient/conform pipelines. |
| Unified multi-tool UX | Shared CLI conventions, standardized I/O behavior, stream/pipeline support, and live status bars by default across tools. |
| Large, organized DSP catalog | Canonical `src/pvx/algorithms/` library with `123` themed algorithm modules plus generated parameter/docs indexes. |
| Backward compatibility | Root script wrappers and `pvxalgorithms` shim preserve legacy invocation/import paths while `src/pvx/...` remains canonical. |

Primary tool entry points:

- `pvxvoc.py`: full-feature phase vocoder (time/pitch/F0/formant/fourier-sync).
- `pvxfreeze.py`: spectral freeze / sustained texture generation.
- `pvxharmonize.py`: multi-voice harmonizer.
- `pvxconform.py`: time+pitch conforming via CSV map.
- `pvxmorph.py`: spectral morph between two files.
- `pvxwarp.py`: variable time warp via CSV map.
- `pvxformant.py`: [formant](https://en.wikipedia.org/wiki/Formant) shift/preserve processing.
- `pvxtransient.py`: transient-first time/pitch processing.
- `pvxunison.py`: stereo detuned unison thickener.
- `pvxdenoise.py`: spectral subtraction/[Wiener](https://en.wikipedia.org/wiki/Wiener_filter)-style denoising workflows.
- `pvxdeverb.py`: reverb-tail suppression.
- `pvxretune.py`: monophonic scale retune.
- `pvxlayer.py`: harmonic/percussive split and independent processing.
- `src/pvx/algorithms/`: canonical themed algorithm library with `123` DSP modules (time/pitch, transforms, separation, denoise, dereverb, dynamics, creative, granular, analysis, and spatial/multichannel).
- `pvxalgorithms/`: compatibility shim package so legacy imports still work (`pvxalgorithms.*` -> `pvx.algorithms.*`).

## Linked Technical References

`pvx` now includes a linked glossary and expanded research bibliography:

- [`docs/html/glossary.html`](docs/html/glossary.html): technical glossary with links for core DSP, phase-vocoder, MIR, and spatial-audio terms.
- [`docs/html/papers.html`](docs/html/papers.html): expanded bibliography (100+ papers/references) covering classic and modern methods.
- [`docs/html/index.html`](docs/html/index.html): grouped algorithm documentation with per-algorithm concept links.
- [`docs/html/README.md`](docs/html/README.md): notes on why GitHub shows HTML as source and how to view rendered HTML pages.
- [`docs/html/architecture.html`](docs/html/architecture.html): rendered architecture diagrams.
- [`docs/html/limitations.html`](docs/html/limitations.html): limitations/failure-mode guidance by algorithm.
- [`docs/html/benchmarks.html`](docs/html/benchmarks.html): reproducible benchmark report page.
- [`docs/html/cookbook.html`](docs/html/cookbook.html): command cookbook for practical pipelines.
- [`docs/html/cli_flags.html`](docs/html/cli_flags.html): full CLI flag reference rendered as HTML.
- [`docs/html/citations.html`](docs/html/citations.html): citation quality report and upgrade targets.
- [`docs/MATHEMATICAL_FOUNDATIONS.md`](docs/MATHEMATICAL_FOUNDATIONS.md): GitHub-renderable LaTeX derivations and plain-English DSP explanations.
- [`docs/WINDOW_REFERENCE.md`](docs/WINDOW_REFERENCE.md): complete 50-window mathematical reference with plain-English interpretation for every window.
- [`docs/ALGORITHM_LIMITATIONS.md`](docs/ALGORITHM_LIMITATIONS.md): assumptions, failure modes, and exclusion guidance per group/algorithm.
- [`docs/PIPELINE_COOKBOOK.md`](docs/PIPELINE_COOKBOOK.md): end-to-end one-liner command cookbook.
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): reproducible benchmark methodology and results.
- [`docs/CLI_FLAGS_REFERENCE.md`](docs/CLI_FLAGS_REFERENCE.md): parser-derived CLI flags reference.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Mermaid architecture diagrams in GitHub Markdown.
- [`docs/CITATION_QUALITY.md`](docs/CITATION_QUALITY.md): citation-link quality classification.
- [`docs/references.bib`](docs/references.bib): generated BibTeX export of the bibliography.
- [`docs/DOCS_CONTRACT.md`](docs/DOCS_CONTRACT.md): documentation contribution contract.

## Hosted Docs (GitHub Pages)

The repository includes an automated Pages workflow that publishes the generated `docs/` site.

- Workflow: `.github/workflows/pages.yml`
- Entry point in published artifact: `docs/index.html` (redirects to `docs/html/index.html`)
- Canonical local preview: `docs/html/index.html`

After enabling GitHub Pages in repository settings (GitHub Actions source), your docs URL is typically:
`https://<your-github-username>.github.io/pvx/html/index.html`

## Math and Windows in GitHub

GitHub repository view generally shows `.html` files as source text, not as rendered web pages.
For browser-rendered math directly inside GitHub, use:

- [`docs/MATHEMATICAL_FOUNDATIONS.md`](docs/MATHEMATICAL_FOUNDATIONS.md)
- [`docs/WINDOW_REFERENCE.md`](docs/WINDOW_REFERENCE.md)

Core equations used throughout PVX:

$$
X_t[k]=\sum_{n=0}^{N-1} x[n+tH_a]w[n]e^{-j2\pi kn/N}
$$

Plain English: STFT analysis takes overlapping, windowed frames and converts them into complex frequency bins.

$$
\Delta\phi_t[k]=\operatorname{princarg}\left(\phi_t[k]-\phi_{t-1}[k]-\omega_kH_a\right),\qquad
\hat{\phi}_t[k]=\hat{\phi}_{t-1}[k]+\hat{\omega}_t[k]H_s
$$

Plain English: the phase vocoder estimates true per-bin frequency from wrapped phase difference, then accumulates phase at the synthesis hop.

$$
r_{\text{pitch}}=2^{\Delta s/12}=2^{\Delta c/1200}
$$

Plain English: semitone and cent shifts are equivalent ways to express the same pitch ratio.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/pvx/core/` | Core DSP/runtime modules (`voc.py`, shared `common.py`). |
| `src/pvx/cli/` | Canonical CLI implementations for `pvx*` tools and `main.py` navigator. |
| `src/pvx/algorithms/` | Canonical algorithm package (`pvx.algorithms.*`). |
| `pvxvoc.py`, `pvxfreeze.py`, ... | Root compatibility wrappers forwarding to `src/pvx/...`. |
| `pvxalgorithms/` | Legacy package compatibility shim forwarding to `src/pvx/algorithms/`. |
| `docs/` | Generated documentation artifacts. |
| `tests/` | Unit and regression tests. |

## Install

### Option A: Requirements file

```bash
python3 -m pip install -r requirements.txt
```

### Option B: Editable package install

```bash
python3 -m pip install -e .
```

This exposes console commands from `pyproject.toml` (`pvxvoc`, `pvxfreeze`, etc.).

### Optional CUDA Acceleration

CUDA processing is available when CuPy is installed for your CUDA runtime:

```bash
python3 -m pip install cupy-cuda12x
```

Use `--device auto` (default), `--device cpu`, or `--device cuda` plus `--cuda-device <index>`.

## Quick Start

```bash
python3 pvxvoc.py test.wav --time-stretch 1.25 --pitch-shift-semitones 3
```

## Comprehensive Python Documentation and Help

`pvx` now ships a full, generated Python-file documentation set that covers every `.py` file in the repository.

Documentation artifacts:

- `docs/PYTHON_FILE_HELP.md`: exhaustive per-file reference and help index.
  - Current coverage: all `192` Python files in this repository.
  - Includes purpose, top-level symbols, help commands, and CLI help snapshots where applicable.
- `docs/PVX_ALGORITHM_PARAMS.md`: per-algorithm parameter keys consumed by `pvx.algorithms.base` dispatch.
- `docs/MATHEMATICAL_FOUNDATIONS.md`: full mathematical foundations in GitHub-renderable LaTeX plus plain-English interpretation.
- `docs/WINDOW_REFERENCE.md`: all supported windows defined mathematically and explained in plain English.
- `docs/ALGORITHM_LIMITATIONS.md`: assumptions/failure modes/when-not-to-use by group and algorithm.
- `docs/PIPELINE_COOKBOOK.md`: end-to-end one-line recipes (including Unix pipes).
- `docs/BENCHMARKS.md`: reproducible performance + quality benchmark report.
- `docs/CLI_FLAGS_REFERENCE.md`: parser-derived long-flag inventory with source mapping.
- `docs/ARCHITECTURE.md`: Mermaid architecture diagrams (GitHub-renderable).
- `docs/CITATION_QUALITY.md`: citation-link quality report and DOI/publisher upgrade targets.
- `docs/references.bib`: generated BibTeX export.
- `docs/DOCS_CONTRACT.md`: docs contribution contract/checklist.
- `ALGORITHM_INVENTORY.md`: themed inventory of all algorithm modules.
- [`docs/html/index.html`](docs/html/index.html): grouped HTML docs index (one page per algorithm folder/theme).
- [`docs/html/glossary.html`](docs/html/glossary.html): linked technical glossary with external references.
- [`docs/html/papers.html`](docs/html/papers.html): expanded bibliography (100+ papers/references) used in formulation of this codebase and related phase-vocoder DSP methods.
- [`docs/html/math.html`](docs/html/math.html): browser-rendered mathematical summary page.
- [`docs/html/windows.html`](docs/html/windows.html): browser-rendered complete window reference.
- [`docs/html/architecture.html`](docs/html/architecture.html): rendered architecture diagrams.
- [`docs/html/limitations.html`](docs/html/limitations.html): rendered limitations guidance.
- [`docs/html/benchmarks.html`](docs/html/benchmarks.html): rendered benchmark report.
- [`docs/html/cookbook.html`](docs/html/cookbook.html): rendered cookbook of command chains.
- [`docs/html/cli_flags.html`](docs/html/cli_flags.html): rendered CLI flag inventory.
- [`docs/html/citations.html`](docs/html/citations.html): rendered citation quality report.
- [`docs/html/README.md`](docs/html/README.md): HTML viewing guidance for GitHub/local browser/GitHub Pages.
- `docs/html/groups/*.html`: per-group algorithm pages listing IDs, module paths, dispatch parameter keys, and linked concepts.

Help access patterns:

| File type | How to get help |
| --- | --- |
| Main pvx tool CLIs (`pvxvoc.py`, `pvxfreeze.py`, etc.) | `python3 <tool>.py --help` |
| Top-level navigator | `python3 main.py --help` |
| Algorithm modules (`src/pvx/algorithms/.../*.py`) | `python3 src/pvx/algorithms/<theme>/<module>.py --help` or `python3 -m pvx.algorithms.<theme>.<module> --help` |
| Test modules | `python3 tests/<file>.py` or `python3 -m unittest ...` |

Legacy algorithm module path `python3 -m pvxalgorithms.<theme>.<module> --help` is still supported via compatibility shims.

Regenerate docs after code changes:

```bash
python3 scripts_generate_python_docs.py
python3 scripts_generate_theory_docs.py
python3 scripts_generate_docs_extras.py
python3 scripts_generate_html_docs.py
python3 scripts_generate_docs_pdf.py --output docs/pvx_documentation.pdf
```

PDF engine notes for `scripts_generate_docs_pdf.py`:

- Auto mode tries (in order): Chromium/Chrome, `wkhtmltopdf`, WeasyPrint, Playwright.
- If you need explicit setup:
  - Chromium/Chrome in PATH, or
  - `python3 -m pip install weasyprint`, or
  - `python3 -m pip install playwright && playwright install chromium`, or
  - system `wkhtmltopdf` with local-file access enabled.

## Documentation Contract

`pvx` uses a strict docs-sync policy for code changes:

- Contract file: [`docs/DOCS_CONTRACT.md`](docs/DOCS_CONTRACT.md)
- CI enforcement: [`.github/workflows/docs-ci.yml`](.github/workflows/docs-ci.yml)
- Scope: CLI flags, algorithm inventory, windows/math docs, citations, and HTML docs must be regenerated in the same change set.

<!-- BEGIN TECHNICAL GLOSSARY TABLE -->
## Technical Glossary Snapshot (Linked)

This table mirrors the generated glossary page and provides direct links for key DSP terms.
For the full version grouped by category, see [`docs/html/glossary.html`](docs/html/glossary.html).

### Core DSP Math

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Aliasing | Frequency-domain folding caused by undersampling. | [Open](docs/html/glossary.html#aliasing) | [Source](https://en.wikipedia.org/wiki/Aliasing) |
| Blackman window | Cosine-sum window with stronger sidelobe suppression. | [Open](docs/html/glossary.html#blackman-window) | [Source](https://en.wikipedia.org/wiki/Window_function) |
| Convolution | Linear filtering operation in time domain equivalent to multiplication in frequency domain. | [Open](docs/html/glossary.html#convolution) | [Source](https://en.wikipedia.org/wiki/Convolution) |
| Fast Fourier transform (FFT) | Efficient algorithm for computing the discrete Fourier transform. | [Open](docs/html/glossary.html#fast-fourier-transform-fft) | [Source](https://en.wikipedia.org/wiki/Fast_Fourier_transform) |
| Fourier transform | Transforms a signal from time domain to frequency domain. | [Open](docs/html/glossary.html#fourier-transform) | [Source](https://en.wikipedia.org/wiki/Fourier_transform) |
| Hamming window | Cosine-sum window with reduced nearest sidelobe amplitude. | [Open](docs/html/glossary.html#hamming-window) | [Source](https://en.wikipedia.org/wiki/Window_function) |
| Hann window | Raised cosine window commonly used in STFT analysis. | [Open](docs/html/glossary.html#hann-window) | [Source](https://en.wikipedia.org/wiki/Window_function) |
| Inverse FFT (IFFT) | Inverse operation that reconstructs time-domain samples from spectral bins. | [Open](docs/html/glossary.html#inverse-fft-ifft) | [Source](https://en.wikipedia.org/wiki/Fast_Fourier_transform) |
| Kaiser window | Parametric window controlled by beta parameter. | [Open](docs/html/glossary.html#kaiser-window) | [Source](https://en.wikipedia.org/wiki/Kaiser_window) |
| Linear phase | Filter phase response that preserves waveform shape of band-limited signals. | [Open](docs/html/glossary.html#linear-phase) | [Source](https://en.wikipedia.org/wiki/Linear_phase) |
| Minimum phase | System with minimum group delay for a given magnitude response. | [Open](docs/html/glossary.html#minimum-phase) | [Source](https://en.wikipedia.org/wiki/Minimum_phase) |
| Nyquist frequency | Half the sample rate; highest representable frequency in sampled audio. | [Open](docs/html/glossary.html#nyquist-frequency) | [Source](https://en.wikipedia.org/wiki/Nyquist_frequency) |
| Overlap-add (OLA) | Frame-based reconstruction by summing overlapped windowed segments. | [Open](docs/html/glossary.html#overlap-add-ola) | [Source](https://en.wikipedia.org/wiki/Overlap%E2%80%93add_method) |
| Spectral leakage | Energy spread across bins due to finite-duration analysis windows. | [Open](docs/html/glossary.html#spectral-leakage) | [Source](https://en.wikipedia.org/wiki/Spectral_leakage) |
| Window function | Tapering function applied before spectral analysis to control leakage. | [Open](docs/html/glossary.html#window-function) | [Source](https://en.wikipedia.org/wiki/Window_function) |

### Phase and Time-Scale DSP

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Formant preservation | Preservation of spectral envelope while pitch is changed. | [Open](docs/html/glossary.html#formant-preservation) | [Source](https://en.wikipedia.org/wiki/Formant) |
| Granular synthesis | Sound synthesis/modification using short grains. | [Open](docs/html/glossary.html#granular-synthesis) | [Source](https://en.wikipedia.org/wiki/Granular_synthesis) |
| HPSS | Harmonic-percussive source separation in TF domain. | [Open](docs/html/glossary.html#hpss) | [Source](https://scholar.google.com/scholar?q=harmonic+percussive+source+separation) |
| Identity phase locking | Phase-locking variant that follows dominant peaks. | [Open](docs/html/glossary.html#identity-phase-locking) | [Source](https://scholar.google.com/scholar?q=identity+phase+locking) |
| LP-PSOLA | Linear-prediction-assisted PSOLA variant. | [Open](docs/html/glossary.html#lp-psola) | [Source](https://scholar.google.com/scholar?q=LP-PSOLA) |
| Phase locking | Locking neighboring bin phases to preserve local waveform structure. | [Open](docs/html/glossary.html#phase-locking) | [Source](https://scholar.google.com/scholar?q=phase+locking+in+phase+vocoder) |
| Phase vocoder | STFT-based time-scale/pitch framework using phase propagation. | [Open](docs/html/glossary.html#phase-vocoder) | [Source](https://en.wikipedia.org/wiki/Phase_vocoder) |
| TD-PSOLA | Time-domain pitch-synchronous overlap-add technique. | [Open](docs/html/glossary.html#td-psola) | [Source](https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling) |
| Time warp map | Piecewise mapping from output time to input time. | [Open](docs/html/glossary.html#time-warp-map) | [Source](https://scholar.google.com/scholar?q=nonlinear+time+warping+audio) |
| Transient preservation | Strategies that protect attacks during spectral time modification. | [Open](docs/html/glossary.html#transient-preservation) | [Source](https://en.wikipedia.org/wiki/Transient_(acoustics)) |
| WSOLA | Waveform-similarity overlap-add time-scale modification. | [Open](docs/html/glossary.html#wsola) | [Source](https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling) |

### Pitch and Intonation

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Cents | Logarithmic pitch interval unit (1200 cents per octave). | [Open](docs/html/glossary.html#cents) | [Source](https://en.wikipedia.org/wiki/Cent_(music)) |
| Equal temperament | Tuning system dividing octave into equal semitone ratios. | [Open](docs/html/glossary.html#equal-temperament) | [Source](https://en.wikipedia.org/wiki/Equal_temperament) |
| Harmonic product spectrum | Downsample-and-multiply spectrum for F0 evidence. | [Open](docs/html/glossary.html#harmonic-product-spectrum) | [Source](https://scholar.google.com/scholar?q=harmonic+product+spectrum+pitch) |
| Just intonation | Tuning based on low-integer frequency ratios. | [Open](docs/html/glossary.html#just-intonation) | [Source](https://en.wikipedia.org/wiki/Just_intonation) |
| MIDI Tuning Standard | Specification for non-12TET tuning in MIDI systems. | [Open](docs/html/glossary.html#midi-tuning-standard) | [Source](https://en.wikipedia.org/wiki/MIDI_tuning_standard) |
| Portamento | Continuous slide between notes. | [Open](docs/html/glossary.html#portamento) | [Source](https://en.wikipedia.org/wiki/Portamento) |
| pYIN | Probabilistic extension of YIN with voicing model. | [Open](docs/html/glossary.html#pyin) | [Source](https://librosa.org/doc/main/generated/librosa.pyin.html) |
| RAPT | Robust algorithm for pitch tracking. | [Open](docs/html/glossary.html#rapt) | [Source](https://scholar.google.com/scholar?q=RAPT+pitch+tracking) |
| Scala scale format | Text format for microtonal scales. | [Open](docs/html/glossary.html#scala-scale-format) | [Source](https://www.huygens-fokker.org/scala/) |
| Subharmonic summation | Weighted subharmonic summation for F0 estimation. | [Open](docs/html/glossary.html#subharmonic-summation) | [Source](https://scholar.google.com/scholar?q=subharmonic+summation+pitch) |
| SWIPE | Sawtooth-waveform inspired pitch estimator. | [Open](docs/html/glossary.html#swipe) | [Source](https://doi.org/10.1121/1.2951592) |
| YIN | Fundamental-frequency estimator using difference-function minima. | [Open](docs/html/glossary.html#yin) | [Source](https://librosa.org/doc/main/generated/librosa.yin.html) |

### Time-Frequency Transforms

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Chirplet transform | Transform using chirped atoms. | [Open](docs/html/glossary.html#chirplet-transform) | [Source](https://en.wikipedia.org/wiki/Chirplet_transform) |
| Constant-Q transform (CQT) | Transform with logarithmic center frequencies and constant Q. | [Open](docs/html/glossary.html#constant-q-transform-cqt) | [Source](https://en.wikipedia.org/wiki/Constant-Q_transform) |
| Multiresolution analysis | Nested subspace structure for multi-scale representations. | [Open](docs/html/glossary.html#multiresolution-analysis) | [Source](https://en.wikipedia.org/wiki/Multiresolution_analysis) |
| NSGT | Nonstationary Gabor transform with invertible adaptive filter banks. | [Open](docs/html/glossary.html#nsgt) | [Source](https://scholar.google.com/scholar?q=nonstationary+gabor+transform) |
| Reassigned spectrogram | Sharper TF representation by moving energy to local centroids. | [Open](docs/html/glossary.html#reassigned-spectrogram) | [Source](https://en.wikipedia.org/wiki/Reassignment_method) |
| Synchrosqueezing | Post-processing that concentrates TF energy along ridges. | [Open](docs/html/glossary.html#synchrosqueezing) | [Source](https://en.wikipedia.org/wiki/Wavelet_transform#Synchrosqueezing_transform) |
| Variable-Q transform (VQT) | Generalized CQT with variable Q and bandwidth behavior. | [Open](docs/html/glossary.html#variable-q-transform-vqt) | [Source](https://scholar.google.com/scholar?q=variable-Q+transform+audio) |
| Wavelet packet | Hierarchical wavelet decomposition of both low/high branches. | [Open](docs/html/glossary.html#wavelet-packet) | [Source](https://en.wikipedia.org/wiki/Wavelet_packet_decomposition) |

### Separation and Decomposition

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| CP decomposition | Canonical polyadic tensor decomposition. | [Open](docs/html/glossary.html#cp-decomposition) | [Source](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) |
| Demucs | Deep waveform-domain music source separation architecture. | [Open](docs/html/glossary.html#demucs) | [Source](https://github.com/facebookresearch/demucs) |
| ICA | Blind source separation based on statistical independence. | [Open](docs/html/glossary.html#ica) | [Source](https://en.wikipedia.org/wiki/Independent_component_analysis) |
| NMF | Nonnegative factorization of matrices into parts-based components. | [Open](docs/html/glossary.html#nmf) | [Source](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) |
| RPCA | Low-rank + sparse decomposition robust to outliers. | [Open](docs/html/glossary.html#rpca) | [Source](https://en.wikipedia.org/wiki/Robust_principal_component_analysis) |
| Tensor decomposition | Higher-order extension of matrix factorization. | [Open](docs/html/glossary.html#tensor-decomposition) | [Source](https://en.wikipedia.org/wiki/Tensor_decomposition) |
| Tucker decomposition | Core tensor with mode matrices factorization. | [Open](docs/html/glossary.html#tucker-decomposition) | [Source](https://en.wikipedia.org/wiki/Tucker_decomposition) |
| U-Net | Encoder-decoder CNN with skip connections. | [Open](docs/html/glossary.html#u-net) | [Source](https://en.wikipedia.org/wiki/U-Net) |

### Denoising and Restoration

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Blind deconvolution | Joint estimation of source and room/filter response. | [Open](docs/html/glossary.html#blind-deconvolution) | [Source](https://en.wikipedia.org/wiki/Blind_deconvolution) |
| Declick | Removal of impulsive click artifacts. | [Open](docs/html/glossary.html#declick) | [Source](https://scholar.google.com/scholar?q=audio+declicking) |
| Declip | Reconstruction of clipped waveform segments. | [Open](docs/html/glossary.html#declip) | [Source](https://scholar.google.com/scholar?q=audio+declipping+sparse+reconstruction) |
| Dereverberation | Suppression of late reverberation components. | [Open](docs/html/glossary.html#dereverberation) | [Source](https://en.wikipedia.org/wiki/Reverberation) |
| Diffusion model | Generative model based on iterative denoising processes. | [Open](docs/html/glossary.html#diffusion-model) | [Source](https://en.wikipedia.org/wiki/Diffusion_model) |
| Direct-to-reverberant ratio (DRR) | Ratio of direct sound energy to reverberant energy. | [Open](docs/html/glossary.html#direct-to-reverberant-ratio-drr) | [Source](https://scholar.google.com/scholar?q=direct-to-reverberant+ratio) |
| Log-MMSE | Log-spectral MMSE enhancement estimator. | [Open](docs/html/glossary.html#log-mmse) | [Source](https://scholar.google.com/scholar?q=Log-MMSE+speech+enhancement) |
| Minimum statistics | Noise tracking from low percentile/minimum spectral statistics. | [Open](docs/html/glossary.html#minimum-statistics) | [Source](https://scholar.google.com/scholar?q=minimum+statistics+noise+estimation) |
| MMSE-STSA | MMSE short-time spectral amplitude estimator. | [Open](docs/html/glossary.html#mmse-stsa) | [Source](https://scholar.google.com/scholar?q=MMSE-STSA) |
| RNNoise | Hybrid recurrent neural + DSP speech denoiser. | [Open](docs/html/glossary.html#rnnoise) | [Source](https://github.com/xiph/rnnoise) |
| Spectral subtraction | Subtracts estimated noise spectrum from noisy signal. | [Open](docs/html/glossary.html#spectral-subtraction) | [Source](https://en.wikipedia.org/wiki/Spectral_subtraction) |
| Weighted prediction error (WPE) | Multi-channel dereverb by delayed linear prediction. | [Open](docs/html/glossary.html#weighted-prediction-error-wpe) | [Source](https://nara-wpe.readthedocs.io/en/latest/) |
| Wiener filter | MMSE linear estimator for noisy observations. | [Open](docs/html/glossary.html#wiener-filter) | [Source](https://en.wikipedia.org/wiki/Wiener_filter) |

### Dynamics and Mastering

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Compander | Combined compressor and expander behavior. | [Open](docs/html/glossary.html#compander) | [Source](https://en.wikipedia.org/wiki/Companding) |
| Compressor | Dynamic range processor that reduces gain above threshold. | [Open](docs/html/glossary.html#compressor) | [Source](https://en.wikipedia.org/wiki/Dynamic_range_compression) |
| EBU R128 | European loudness standard for production and broadcast. | [Open](docs/html/glossary.html#ebu-r128) | [Source](https://tech.ebu.ch/publications/r128) |
| Expander | Dynamic processor that reduces gain below threshold. | [Open](docs/html/glossary.html#expander) | [Source](https://en.wikipedia.org/wiki/Expander_(audio)) |
| Hard clipping | Abrupt saturation by truncating waveform beyond threshold. | [Open](docs/html/glossary.html#hard-clipping) | [Source](https://en.wikipedia.org/wiki/Clipping_(audio)) |
| ITU-R BS.1770 | International loudness measurement recommendation. | [Open](docs/html/glossary.html#itu-r-bs-1770) | [Source](https://www.itu.int/rec/R-REC-BS.1770) |
| Limiter | High-ratio compressor preventing peaks from exceeding threshold. | [Open](docs/html/glossary.html#limiter) | [Source](https://en.wikipedia.org/wiki/Dynamic_range_compression) |
| LUFS | Loudness unit relative to full scale. | [Open](docs/html/glossary.html#lufs) | [Source](https://en.wikipedia.org/wiki/LKFS) |
| Soft clipping | Smooth saturation curve reducing harsh distortion. | [Open](docs/html/glossary.html#soft-clipping) | [Source](https://en.wikipedia.org/wiki/Waveshaper) |
| True peak | Peak estimate including inter-sample overs. | [Open](docs/html/glossary.html#true-peak) | [Source](https://en.wikipedia.org/wiki/Peak_programme_meter) |

### Creative Spectral Effects

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Amplitude modulation (AM) | Modulating signal amplitude by a control waveform. | [Open](docs/html/glossary.html#amplitude-modulation-am) | [Source](https://en.wikipedia.org/wiki/Amplitude_modulation) |
| Cross synthesis | Combining spectral magnitude and phase or envelopes across signals. | [Open](docs/html/glossary.html#cross-synthesis) | [Source](https://en.wikipedia.org/wiki/Cross-synthesis) |
| Formant | Resonance region of spectral envelope. | [Open](docs/html/glossary.html#formant) | [Source](https://en.wikipedia.org/wiki/Formant) |
| Frequency modulation (FM) | Modulating oscillator frequency by another signal. | [Open](docs/html/glossary.html#frequency-modulation-fm) | [Source](https://en.wikipedia.org/wiki/Frequency_modulation_synthesis) |
| LFO | Low-frequency oscillator used for modulation. | [Open](docs/html/glossary.html#lfo) | [Source](https://en.wikipedia.org/wiki/Low-frequency_oscillation) |
| Resonator | Filter structure emphasizing a narrow band. | [Open](docs/html/glossary.html#resonator) | [Source](https://en.wikipedia.org/wiki/Resonance) |
| Ring modulation | Multiplication by a carrier to produce sidebands. | [Open](docs/html/glossary.html#ring-modulation) | [Source](https://en.wikipedia.org/wiki/Ring_modulation) |
| Spectral freezing | Holding a short-time spectrum and resynthesizing over time. | [Open](docs/html/glossary.html#spectral-freezing) | [Source](https://scholar.google.com/scholar?q=spectral+freeze+effect) |

### Spatial Audio and Multichannel

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Ambisonics | 3D audio representation using spherical harmonics. | [Open](docs/html/glossary.html#ambisonics) | [Source](https://en.wikipedia.org/wiki/Ambisonics) |
| Beamforming | Spatial filtering using multiple microphones/loudspeakers. | [Open](docs/html/glossary.html#beamforming) | [Source](https://en.wikipedia.org/wiki/Beamforming) |
| Binaural rendering | Headphone rendering preserving spatial localization cues. | [Open](docs/html/glossary.html#binaural-rendering) | [Source](https://en.wikipedia.org/wiki/Binaural_recording) |
| Crosstalk cancellation | Loudspeaker technique reducing opposite-ear leakage. | [Open](docs/html/glossary.html#crosstalk-cancellation) | [Source](https://scholar.google.com/scholar?q=transaural+crosstalk+cancellation) |
| DBAP | Distance-based amplitude panning. | [Open](docs/html/glossary.html#dbap) | [Source](https://scholar.google.com/scholar?q=distance+based+amplitude+panning+DBAP) |
| Delay-and-sum beamformer | Align-and-average array beamformer. | [Open](docs/html/glossary.html#delay-and-sum-beamformer) | [Source](https://en.wikipedia.org/wiki/Beamforming) |
| Diffuse field | Sound field with many uncorrelated incoming directions. | [Open](docs/html/glossary.html#diffuse-field) | [Source](https://scholar.google.com/scholar?q=diffuse+sound+field) |
| Direction of arrival (DoA) | Estimated source direction from array measurements. | [Open](docs/html/glossary.html#direction-of-arrival-doa) | [Source](https://en.wikipedia.org/wiki/Direction_of_arrival) |
| First-order ambisonics (FOA) | 4-channel WXYZ ambisonic format. | [Open](docs/html/glossary.html#first-order-ambisonics-foa) | [Source](https://en.wikipedia.org/wiki/Ambisonics) |
| GCC-PHAT | Time-delay estimation using phase transform weighting. | [Open](docs/html/glossary.html#gcc-phat) | [Source](https://scholar.google.com/scholar?q=GCC-PHAT) |
| Generalized sidelobe canceller (GSC) | Adaptive beamforming with blocking matrix branch. | [Open](docs/html/glossary.html#generalized-sidelobe-canceller-gsc) | [Source](https://scholar.google.com/scholar?q=generalized+sidelobe+canceller) |
| Higher-order ambisonics (HOA) | Ambisonics beyond first order for higher spatial resolution. | [Open](docs/html/glossary.html#higher-order-ambisonics-hoa) | [Source](https://en.wikipedia.org/wiki/Ambisonics) |
| HRTF | Head-related transfer function per source direction. | [Open](docs/html/glossary.html#hrtf) | [Source](https://en.wikipedia.org/wiki/Head-related_transfer_function) |
| Interaural level difference (ILD) | Level difference cue for localization. | [Open](docs/html/glossary.html#interaural-level-difference-ild) | [Source](https://en.wikipedia.org/wiki/Sound_localization) |
| Interaural time difference (ITD) | Arrival-time difference cue for localization. | [Open](docs/html/glossary.html#interaural-time-difference-itd) | [Source](https://en.wikipedia.org/wiki/Interaural_time_difference) |
| Mid/side processing | Stereo representation as sum (mid) and difference (side). | [Open](docs/html/glossary.html#mid-side-processing) | [Source](https://en.wikipedia.org/wiki/Joint_encoding#M/S_stereo_coding) |
| MVDR beamformer | Minimum variance distortionless response beamforming. | [Open](docs/html/glossary.html#mvdr-beamformer) | [Source](https://en.wikipedia.org/wiki/Minimum_variance_distortionless_response) |
| Room impulse response (RIR) | Acoustic response from source impulse to receiver. | [Open](docs/html/glossary.html#room-impulse-response-rir) | [Source](https://en.wikipedia.org/wiki/Impulse_response) |
| Superdirective beamforming | Beamforming with sharpened directivity often with regularization. | [Open](docs/html/glossary.html#superdirective-beamforming) | [Source](https://scholar.google.com/scholar?q=superdirective+beamforming) |
| VBAP | Vector-base amplitude panning for loudspeaker arrays. | [Open](docs/html/glossary.html#vbap) | [Source](https://scholar.google.com/scholar?q=VBAP+vector+base+amplitude+panning) |

### Analysis and QA

| Term | Description | Glossary link | External reference |
| --- | --- | --- | --- |
| Bayesian optimization | Global optimization of expensive objective functions. | [Open](docs/html/glossary.html#bayesian-optimization) | [Source](https://en.wikipedia.org/wiki/Bayesian_optimization) |
| Beat tracking | Estimating temporal pulse positions from audio. | [Open](docs/html/glossary.html#beat-tracking) | [Source](https://en.wikipedia.org/wiki/Beat_tracking) |
| Chord estimation | Inferring harmonic labels from tonal features. | [Open](docs/html/glossary.html#chord-estimation) | [Source](https://scholar.google.com/scholar?q=automatic+chord+estimation) |
| Onset detection | Detection of transient musical events. | [Open](docs/html/glossary.html#onset-detection) | [Source](https://en.wikipedia.org/wiki/Onset_(music)) |
| PESQ | Perceptual Evaluation of Speech Quality metric. | [Open](docs/html/glossary.html#pesq) | [Source](https://en.wikipedia.org/wiki/Perceptual_Evaluation_of_Speech_Quality) |
| STOI | Short-Time Objective Intelligibility measure. | [Open](docs/html/glossary.html#stoi) | [Source](https://scholar.google.com/scholar?q=STOI+speech+intelligibility) |
| Structure segmentation | Dividing music into sections like verse/chorus. | [Open](docs/html/glossary.html#structure-segmentation) | [Source](https://scholar.google.com/scholar?q=music+structure+segmentation) |
| ViSQOL | Virtual Speech Quality Objective Listener metric family. | [Open](docs/html/glossary.html#visqol) | [Source](https://scholar.google.com/scholar?q=ViSQOL) |
<!-- END TECHNICAL GLOSSARY TABLE -->

<!-- BEGIN ALGORITHM CATALOG -->
## Algorithm Catalog By Folder and Parameter Keys

This section enumerates every registered algorithm (123 total) grouped by folder.
`Parameter keys` are the exact `**params` keys consumed by `pvx.algorithms.base.run_algorithm()` dispatch.
Each row includes direct links to grouped HTML docs and glossary concepts.

| Folder | Theme | Algorithm count | Group page |
| --- | --- | ---: | --- |
| `time_scale_and_pitch_core` | Time-Scale and Pitch Core | 7 | [`docs/html/groups/time_scale_and_pitch_core.html`](docs/html/groups/time_scale_and_pitch_core.html) |
| `pitch_detection_and_tracking` | Pitch Detection and Tracking | 8 | [`docs/html/groups/pitch_detection_and_tracking.html`](docs/html/groups/pitch_detection_and_tracking.html) |
| `retune_and_intonation` | Retune and Intonation | 8 | [`docs/html/groups/retune_and_intonation.html`](docs/html/groups/retune_and_intonation.html) |
| `spectral_time_frequency_transforms` | Spectral and Time-Frequency Transforms | 8 | [`docs/html/groups/spectral_time_frequency_transforms.html`](docs/html/groups/spectral_time_frequency_transforms.html) |
| `separation_and_decomposition` | Separation and Decomposition | 8 | [`docs/html/groups/separation_and_decomposition.html`](docs/html/groups/separation_and_decomposition.html) |
| `denoise_and_restoration` | Denoise and Restoration | 8 | [`docs/html/groups/denoise_and_restoration.html`](docs/html/groups/denoise_and_restoration.html) |
| `dereverb_and_room_correction` | Dereverb and Room Correction | 8 | [`docs/html/groups/dereverb_and_room_correction.html`](docs/html/groups/dereverb_and_room_correction.html) |
| `dynamics_and_loudness` | Dynamics and Loudness | 8 | [`docs/html/groups/dynamics_and_loudness.html`](docs/html/groups/dynamics_and_loudness.html) |
| `creative_spectral_effects` | Creative Spectral Effects | 8 | [`docs/html/groups/creative_spectral_effects.html`](docs/html/groups/creative_spectral_effects.html) |
| `granular_and_modulation` | Granular and Modulation | 8 | [`docs/html/groups/granular_and_modulation.html`](docs/html/groups/granular_and_modulation.html) |
| `analysis_qa_and_automation` | Analysis, QA, and Automation | 8 | [`docs/html/groups/analysis_qa_and_automation.html`](docs/html/groups/analysis_qa_and_automation.html) |
| `spatial_and_multichannel` | Spatial and Multichannel | 36 | [`docs/html/groups/spatial_and_multichannel.html`](docs/html/groups/spatial_and_multichannel.html) |

### `time_scale_and_pitch_core`

Theme: **Time-Scale and Pitch Core**

Subgroups: `core` (7)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `time_scale_and_pitch_core.wsola_waveform_similarity_overlap_add` | WSOLA (Waveform Similarity Overlap-Add) | `src/pvx/algorithms/time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py` | `stretch`, `grain_size` | [WSOLA](docs/html/glossary.html#wsola) |
| `time_scale_and_pitch_core.td_psola` | TD-PSOLA | `src/pvx/algorithms/time_scale_and_pitch_core/td_psola.py` | `semitones`, `stretch` | [TD-PSOLA](docs/html/glossary.html#td-psola) |
| `time_scale_and_pitch_core.lp_psola` | LP-PSOLA | `src/pvx/algorithms/time_scale_and_pitch_core/lp_psola.py` | `semitones` | [LP-PSOLA](docs/html/glossary.html#lp-psola) |
| `time_scale_and_pitch_core.multi_resolution_phase_vocoder` | Multi-resolution phase vocoder | `src/pvx/algorithms/time_scale_and_pitch_core/multi_resolution_phase_vocoder.py` | `stretch` | [Phase vocoder](docs/html/glossary.html#phase-vocoder) |
| `time_scale_and_pitch_core.harmonic_percussive_split_tsm` | Harmonic/percussive split TSM | `src/pvx/algorithms/time_scale_and_pitch_core/harmonic_percussive_split_tsm.py` | `harmonic_stretch`, `percussive_stretch` | None |
| `time_scale_and_pitch_core.beat_synchronous_time_warping` | Beat-synchronous time warping | `src/pvx/algorithms/time_scale_and_pitch_core/beat_synchronous_time_warping.py` | `stretch` | None |
| `time_scale_and_pitch_core.nonlinear_time_maps` | Nonlinear time maps (curves, anchors, spline timing) | `src/pvx/algorithms/time_scale_and_pitch_core/nonlinear_time_maps.py` | `curve`, `stretch`, `fmin` | None |

### `pitch_detection_and_tracking`

Theme: **Pitch Detection and Tracking**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `pitch_detection_and_tracking.yin` | YIN | `src/pvx/algorithms/pitch_detection_and_tracking/yin.py` | None (generic/default path) | [YIN](docs/html/glossary.html#yin) |
| `pitch_detection_and_tracking.pyin` | pYIN | `src/pvx/algorithms/pitch_detection_and_tracking/pyin.py` | None (generic/default path) | [pYIN](docs/html/glossary.html#pyin) |
| `pitch_detection_and_tracking.rapt` | RAPT | `src/pvx/algorithms/pitch_detection_and_tracking/rapt.py` | None (generic/default path) | [RAPT](docs/html/glossary.html#rapt) |
| `pitch_detection_and_tracking.swipe` | SWIPE | `src/pvx/algorithms/pitch_detection_and_tracking/swipe.py` | None (generic/default path) | [SWIPE](docs/html/glossary.html#swipe) |
| `pitch_detection_and_tracking.harmonic_product_spectrum_hps` | Harmonic Product Spectrum (HPS) | `src/pvx/algorithms/pitch_detection_and_tracking/harmonic_product_spectrum_hps.py` | None (generic/default path) | [Harmonic product spectrum](docs/html/glossary.html#harmonic-product-spectrum) |
| `pitch_detection_and_tracking.subharmonic_summation` | Subharmonic summation | `src/pvx/algorithms/pitch_detection_and_tracking/subharmonic_summation.py` | None (generic/default path) | [Subharmonic summation](docs/html/glossary.html#subharmonic-summation) |
| `pitch_detection_and_tracking.crepe_style_neural_f0` | CREPE-style neural F0 | `src/pvx/algorithms/pitch_detection_and_tracking/crepe_style_neural_f0.py` | None (generic/default path) | None |
| `pitch_detection_and_tracking.viterbi_smoothed_pitch_contour_tracking` | Viterbi-smoothed pitch contour tracking | `src/pvx/algorithms/pitch_detection_and_tracking/viterbi_smoothed_pitch_contour_tracking.py` | `root_midi`, `scale_cents`, `scale`, `fmin` | None |

### `retune_and_intonation`

Theme: **Retune and Intonation**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `retune_and_intonation.chord_aware_retuning` | Chord-aware retuning | `src/pvx/algorithms/retune_and_intonation/chord_aware_retuning.py` | `strength` | None |
| `retune_and_intonation.key_aware_retuning_with_confidence_weighting` | Key-aware retuning with confidence weighting | `src/pvx/algorithms/retune_and_intonation/key_aware_retuning_with_confidence_weighting.py` | None (generic/default path) | None |
| `retune_and_intonation.just_intonation_mapping_per_key_center` | Just intonation mapping per key center | `src/pvx/algorithms/retune_and_intonation/just_intonation_mapping_per_key_center.py` | None (generic/default path) | [Just intonation](docs/html/glossary.html#just-intonation) |
| `retune_and_intonation.adaptive_intonation_context_sensitive_intervals` | Adaptive intonation (context-sensitive intervals) | `src/pvx/algorithms/retune_and_intonation/adaptive_intonation_context_sensitive_intervals.py` | None (generic/default path) | None |
| `retune_and_intonation.scala_mts_scale_import_and_quantization` | Scala/MTS scale import and quantization | `src/pvx/algorithms/retune_and_intonation/scala_mts_scale_import_and_quantization.py` | None (generic/default path) | None |
| `retune_and_intonation.time_varying_cents_maps` | Time-varying cents maps | `src/pvx/algorithms/retune_and_intonation/time_varying_cents_maps.py` | `cents_curve` | [Cents](docs/html/glossary.html#cents) |
| `retune_and_intonation.vibrato_preserving_correction` | Vibrato-preserving correction | `src/pvx/algorithms/retune_and_intonation/vibrato_preserving_correction.py` | `strength` | None |
| `retune_and_intonation.portamento_aware_retune_curves` | Portamento-aware retune curves | `src/pvx/algorithms/retune_and_intonation/portamento_aware_retune_curves.py` | `max_semitone_step`, `compression` | [Portamento](docs/html/glossary.html#portamento) |

### `spectral_time_frequency_transforms`

Theme: **Spectral and Time-Frequency Transforms**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `spectral_time_frequency_transforms.constant_q_transform_cqt_processing` | Constant-Q Transform (CQT) processing | `src/pvx/algorithms/spectral_time_frequency_transforms/constant_q_transform_cqt_processing.py` | None (generic/default path) | [Constant-Q transform (CQT)](docs/html/glossary.html#constant-q-transform-cqt) |
| `spectral_time_frequency_transforms.variable_q_transform_vqt` | Variable-Q Transform (VQT) | `src/pvx/algorithms/spectral_time_frequency_transforms/variable_q_transform_vqt.py` | None (generic/default path) | [Variable-Q transform (VQT)](docs/html/glossary.html#variable-q-transform-vqt) |
| `spectral_time_frequency_transforms.nsgt_based_processing` | NSGT-based processing | `src/pvx/algorithms/spectral_time_frequency_transforms/nsgt_based_processing.py` | None (generic/default path) | [NSGT](docs/html/glossary.html#nsgt) |
| `spectral_time_frequency_transforms.reassigned_spectrogram_methods` | Reassigned spectrogram methods | `src/pvx/algorithms/spectral_time_frequency_transforms/reassigned_spectrogram_methods.py` | None (generic/default path) | [Reassigned spectrogram](docs/html/glossary.html#reassigned-spectrogram) |
| `spectral_time_frequency_transforms.synchrosqueezed_stft` | Synchrosqueezed STFT | `src/pvx/algorithms/spectral_time_frequency_transforms/synchrosqueezed_stft.py` | None (generic/default path) | None |
| `spectral_time_frequency_transforms.chirplet_transform_analysis` | Chirplet transform analysis | `src/pvx/algorithms/spectral_time_frequency_transforms/chirplet_transform_analysis.py` | None (generic/default path) | [Chirplet transform](docs/html/glossary.html#chirplet-transform) |
| `spectral_time_frequency_transforms.wavelet_packet_processing` | Wavelet packet processing | `src/pvx/algorithms/spectral_time_frequency_transforms/wavelet_packet_processing.py` | None (generic/default path) | [Wavelet packet](docs/html/glossary.html#wavelet-packet) |
| `spectral_time_frequency_transforms.multi_window_stft_fusion` | Multi-window STFT fusion | `src/pvx/algorithms/spectral_time_frequency_transforms/multi_window_stft_fusion.py` | None (generic/default path) | None |

### `separation_and_decomposition`

Theme: **Separation and Decomposition**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `separation_and_decomposition.rpca_hpss` | RPCA HPSS | `src/pvx/algorithms/separation_and_decomposition/rpca_hpss.py` | None (generic/default path) | [HPSS](docs/html/glossary.html#hpss), [RPCA](docs/html/glossary.html#rpca) |
| `separation_and_decomposition.nmf_decomposition` | NMF decomposition | `src/pvx/algorithms/separation_and_decomposition/nmf_decomposition.py` | `components` | [NMF](docs/html/glossary.html#nmf) |
| `separation_and_decomposition.ica_bss_for_multichannel_stems` | ICA/BSS for multichannel stems | `src/pvx/algorithms/separation_and_decomposition/ica_bss_for_multichannel_stems.py` | None (generic/default path) | [ICA](docs/html/glossary.html#ica) |
| `separation_and_decomposition.sinusoidal_residual_transient_decomposition` | Sinusoidal+residual+transient decomposition | `src/pvx/algorithms/separation_and_decomposition/sinusoidal_residual_transient_decomposition.py` | None (generic/default path) | None |
| `separation_and_decomposition.demucs_style_stem_separation_backend` | Demucs-style stem separation backend | `src/pvx/algorithms/separation_and_decomposition/demucs_style_stem_separation_backend.py` | None (generic/default path) | [Demucs](docs/html/glossary.html#demucs) |
| `separation_and_decomposition.u_net_vocal_accompaniment_split` | U-Net vocal/accompaniment split | `src/pvx/algorithms/separation_and_decomposition/u_net_vocal_accompaniment_split.py` | None (generic/default path) | [U-Net](docs/html/glossary.html#u-net) |
| `separation_and_decomposition.tensor_decomposition_cp_tucker` | Tensor decomposition (CP/Tucker) | `src/pvx/algorithms/separation_and_decomposition/tensor_decomposition_cp_tucker.py` | `rank` | [Tensor decomposition](docs/html/glossary.html#tensor-decomposition) |
| `separation_and_decomposition.probabilistic_latent_component_separation` | Probabilistic latent component separation | `src/pvx/algorithms/separation_and_decomposition/probabilistic_latent_component_separation.py` | None (generic/default path) | None |

### `denoise_and_restoration`

Theme: **Denoise and Restoration**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `denoise_and_restoration.wiener_denoising` | Wiener denoising | `src/pvx/algorithms/denoise_and_restoration/wiener_denoising.py` | None (generic/default path) | None |
| `denoise_and_restoration.mmse_stsa` | MMSE-STSA | `src/pvx/algorithms/denoise_and_restoration/mmse_stsa.py` | None (generic/default path) | [MMSE-STSA](docs/html/glossary.html#mmse-stsa) |
| `denoise_and_restoration.log_mmse` | Log-MMSE | `src/pvx/algorithms/denoise_and_restoration/log_mmse.py` | None (generic/default path) | [Log-MMSE](docs/html/glossary.html#log-mmse) |
| `denoise_and_restoration.minimum_statistics_noise_tracking` | Minimum-statistics noise tracking | `src/pvx/algorithms/denoise_and_restoration/minimum_statistics_noise_tracking.py` | None (generic/default path) | [Minimum statistics](docs/html/glossary.html#minimum-statistics) |
| `denoise_and_restoration.rnnoise_style_denoiser` | RNNoise-style denoiser | `src/pvx/algorithms/denoise_and_restoration/rnnoise_style_denoiser.py` | None (generic/default path) | [RNNoise](docs/html/glossary.html#rnnoise) |
| `denoise_and_restoration.diffusion_based_speech_audio_denoise` | Diffusion-based speech/audio denoise | `src/pvx/algorithms/denoise_and_restoration/diffusion_based_speech_audio_denoise.py` | None (generic/default path) | None |
| `denoise_and_restoration.declip_via_sparse_reconstruction` | Declip via sparse reconstruction | `src/pvx/algorithms/denoise_and_restoration/declip_via_sparse_reconstruction.py` | `clip_threshold` | [Declip](docs/html/glossary.html#declip) |
| `denoise_and_restoration.declick_decrackle_median_wavelet_interpolation` | Declick/decrackle (median/wavelet + interpolation) | `src/pvx/algorithms/denoise_and_restoration/declick_decrackle_median_wavelet_interpolation.py` | `spike_threshold` | [Declick](docs/html/glossary.html#declick) |

### `dereverb_and_room_correction`

Theme: **Dereverb and Room Correction**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `dereverb_and_room_correction.wpe_dereverberation` | WPE dereverberation | `src/pvx/algorithms/dereverb_and_room_correction/wpe_dereverberation.py` | `taps` | [Dereverberation](docs/html/glossary.html#dereverberation) |
| `dereverb_and_room_correction.spectral_decay_subtraction` | Spectral decay subtraction | `src/pvx/algorithms/dereverb_and_room_correction/spectral_decay_subtraction.py` | None (generic/default path) | None |
| `dereverb_and_room_correction.late_reverb_suppression_via_coherence` | Late reverb suppression via coherence | `src/pvx/algorithms/dereverb_and_room_correction/late_reverb_suppression_via_coherence.py` | None (generic/default path) | None |
| `dereverb_and_room_correction.room_impulse_inverse_filtering` | Room impulse inverse filtering | `src/pvx/algorithms/dereverb_and_room_correction/room_impulse_inverse_filtering.py` | None (generic/default path) | None |
| `dereverb_and_room_correction.multi_band_adaptive_deverb` | Multi-band adaptive deverb | `src/pvx/algorithms/dereverb_and_room_correction/multi_band_adaptive_deverb.py` | None (generic/default path) | None |
| `dereverb_and_room_correction.drr_guided_dereverb` | DRR-guided dereverb | `src/pvx/algorithms/dereverb_and_room_correction/drr_guided_dereverb.py` | None (generic/default path) | None |
| `dereverb_and_room_correction.blind_deconvolution_dereverb` | Blind deconvolution dereverb | `src/pvx/algorithms/dereverb_and_room_correction/blind_deconvolution_dereverb.py` | None (generic/default path) | [Blind deconvolution](docs/html/glossary.html#blind-deconvolution) |
| `dereverb_and_room_correction.neural_dereverb_module` | Neural dereverb module | `src/pvx/algorithms/dereverb_and_room_correction/neural_dereverb_module.py` | None (generic/default path) | None |

### `dynamics_and_loudness`

Theme: **Dynamics and Loudness**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `dynamics_and_loudness.ebu_r128_normalization` | EBU R128 normalization | `src/pvx/algorithms/dynamics_and_loudness/ebu_r128_normalization.py` | `target_lufs` | [EBU R128](docs/html/glossary.html#ebu-r128) |
| `dynamics_and_loudness.itu_bs_1770_loudness_measurement_gating` | ITU BS.1770 loudness measurement/gating | `src/pvx/algorithms/dynamics_and_loudness/itu_bs_1770_loudness_measurement_gating.py` | `gate_lufs` | None |
| `dynamics_and_loudness.multi_band_compression` | Multi-band compression | `src/pvx/algorithms/dynamics_and_loudness/multi_band_compression.py` | None (generic/default path) | None |
| `dynamics_and_loudness.upward_compression` | Upward compression | `src/pvx/algorithms/dynamics_and_loudness/upward_compression.py` | `threshold_db` | None |
| `dynamics_and_loudness.transient_shaping` | Transient shaping | `src/pvx/algorithms/dynamics_and_loudness/transient_shaping.py` | `attack_boost` | None |
| `dynamics_and_loudness.spectral_dynamics_bin_wise_compressor_expander` | Spectral dynamics (bin-wise compressor/expander) | `src/pvx/algorithms/dynamics_and_loudness/spectral_dynamics_bin_wise_compressor_expander.py` | `threshold_db` | [Compressor](docs/html/glossary.html#compressor), [Expander](docs/html/glossary.html#expander) |
| `dynamics_and_loudness.true_peak_limiting` | True-peak limiting | `src/pvx/algorithms/dynamics_and_loudness/true_peak_limiting.py` | `threshold` | [True peak](docs/html/glossary.html#true-peak) |
| `dynamics_and_loudness.lufs_target_mastering_chain` | LUFS-target mastering chain | `src/pvx/algorithms/dynamics_and_loudness/lufs_target_mastering_chain.py` | `target_lufs` | [LUFS](docs/html/glossary.html#lufs) |

### `creative_spectral_effects`

Theme: **Creative Spectral Effects**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `creative_spectral_effects.cross_synthesis_vocoder` | Cross-synthesis vocoder | `src/pvx/algorithms/creative_spectral_effects/cross_synthesis_vocoder.py` | None (generic/default path) | [Cross synthesis](docs/html/glossary.html#cross-synthesis) |
| `creative_spectral_effects.spectral_convolution_effects` | Spectral convolution effects | `src/pvx/algorithms/creative_spectral_effects/spectral_convolution_effects.py` | `kernel_size` | [Convolution](docs/html/glossary.html#convolution) |
| `creative_spectral_effects.spectral_freeze_banks` | Spectral freeze banks | `src/pvx/algorithms/creative_spectral_effects/spectral_freeze_banks.py` | `frame_ratio` | None |
| `creative_spectral_effects.spectral_blur_smear` | Spectral blur/smear | `src/pvx/algorithms/creative_spectral_effects/spectral_blur_smear.py` | None (generic/default path) | None |
| `creative_spectral_effects.phase_randomization_textures` | Phase randomization textures | `src/pvx/algorithms/creative_spectral_effects/phase_randomization_textures.py` | `strength` | None |
| `creative_spectral_effects.formant_painting_warping` | Formant painting/warping | `src/pvx/algorithms/creative_spectral_effects/formant_painting_warping.py` | `ratio` | [Formant](docs/html/glossary.html#formant) |
| `creative_spectral_effects.resonator_filterbank_morphing` | Resonator/filterbank morphing | `src/pvx/algorithms/creative_spectral_effects/resonator_filterbank_morphing.py` | None (generic/default path) | [Resonator](docs/html/glossary.html#resonator) |
| `creative_spectral_effects.spectral_contrast_exaggeration` | Spectral contrast exaggeration | `src/pvx/algorithms/creative_spectral_effects/spectral_contrast_exaggeration.py` | `amount` | None |

### `granular_and_modulation`

Theme: **Granular and Modulation**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `granular_and_modulation.granular_time_stretch_engine` | Granular time-stretch engine | `src/pvx/algorithms/granular_and_modulation/granular_time_stretch_engine.py` | `stretch` | None |
| `granular_and_modulation.grain_cloud_pitch_textures` | Grain-cloud pitch textures | `src/pvx/algorithms/granular_and_modulation/grain_cloud_pitch_textures.py` | `seed`, `grain`, `count`, `stretch` | None |
| `granular_and_modulation.freeze_grain_morphing` | Freeze-grain morphing | `src/pvx/algorithms/granular_and_modulation/freeze_grain_morphing.py` | `grain`, `start` | None |
| `granular_and_modulation.am_fm_ring_modulation_blocks` | AM/FM/ring modulation blocks | `src/pvx/algorithms/granular_and_modulation/am_fm_ring_modulation_blocks.py` | `freq_hz` | [Ring modulation](docs/html/glossary.html#ring-modulation) |
| `granular_and_modulation.spectral_tremolo` | Spectral tremolo | `src/pvx/algorithms/granular_and_modulation/spectral_tremolo.py` | `lfo_hz` | None |
| `granular_and_modulation.formant_lfo_modulation` | Formant LFO modulation | `src/pvx/algorithms/granular_and_modulation/formant_lfo_modulation.py` | `lfo_hz` | [Formant](docs/html/glossary.html#formant), [LFO](docs/html/glossary.html#lfo) |
| `granular_and_modulation.rhythmic_gate_stutter_quantizer` | Rhythmic gate/stutter quantizer | `src/pvx/algorithms/granular_and_modulation/rhythmic_gate_stutter_quantizer.py` | `rate_hz` | None |
| `granular_and_modulation.envelope_followed_modulation_routing` | Envelope-followed modulation routing | `src/pvx/algorithms/granular_and_modulation/envelope_followed_modulation_routing.py` | `depth` | None |

### `analysis_qa_and_automation`

Theme: **Analysis, QA, and Automation**

Subgroups: `core` (8)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `analysis_qa_and_automation.onset_beat_downbeat_tracking` | Onset/beat/downbeat tracking | `src/pvx/algorithms/analysis_qa_and_automation/onset_beat_downbeat_tracking.py` | None (generic/default path) | None |
| `analysis_qa_and_automation.key_chord_detection` | Key/chord detection | `src/pvx/algorithms/analysis_qa_and_automation/key_chord_detection.py` | None (generic/default path) | None |
| `analysis_qa_and_automation.structure_segmentation_verse_chorus_sections` | Structure segmentation (verse/chorus/sections) | `src/pvx/algorithms/analysis_qa_and_automation/structure_segmentation_verse_chorus_sections.py` | None (generic/default path) | [Structure segmentation](docs/html/glossary.html#structure-segmentation) |
| `analysis_qa_and_automation.silence_speech_music_classifiers` | Silence/speech/music classifiers | `src/pvx/algorithms/analysis_qa_and_automation/silence_speech_music_classifiers.py` | None (generic/default path) | None |
| `analysis_qa_and_automation.clip_hum_buzz_artifact_detection` | Clip/hum/buzz artifact detection | `src/pvx/algorithms/analysis_qa_and_automation/clip_hum_buzz_artifact_detection.py` | None (generic/default path) | None |
| `analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics` | PESQ/STOI/VISQOL quality metrics | `src/pvx/algorithms/analysis_qa_and_automation/pesq_stoi_visqol_quality_metrics.py` | None (generic/default path) | [PESQ](docs/html/glossary.html#pesq), [STOI](docs/html/glossary.html#stoi), [ViSQOL](docs/html/glossary.html#visqol) |
| `analysis_qa_and_automation.auto_parameter_tuning_bayesian_optimization` | Auto-parameter tuning (Bayesian optimization) | `src/pvx/algorithms/analysis_qa_and_automation/auto_parameter_tuning_bayesian_optimization.py` | `target_centroid` | [Bayesian optimization](docs/html/glossary.html#bayesian-optimization) |
| `analysis_qa_and_automation.batch_preset_recommendation_based_on_source_features` | Batch preset recommendation based on source features | `src/pvx/algorithms/analysis_qa_and_automation/batch_preset_recommendation_based_on_source_features.py` | None (generic/default path) | None |

### `spatial_and_multichannel`

Theme: **Spatial and Multichannel**

Subgroups: `imaging_and_panning` (6), `beamforming_and_directionality` (6), `ambisonics_and_immersive` (6), `phase_vocoder_spatial` (6), `multichannel_restoration` (6), `creative_spatial_fx` (6)

| Algorithm ID | Algorithm name | Module Path | Parameter keys | Concepts |
| --- | --- | --- | --- | --- |
| `spatial_and_multichannel.vbap_adaptive_panning` | VBAP adaptive panning | `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/vbap_adaptive_panning.py` | `output_channels`, `azimuth_deg`, `width` | [VBAP](docs/html/glossary.html#vbap) |
| `spatial_and_multichannel.dbap_distance_based_amplitude_panning` | DBAP (distance-based amplitude panning) | `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/dbap_distance_based_amplitude_panning.py` | `output_channels`, `source_x`, `source_y`, `rolloff` | [DBAP](docs/html/glossary.html#dbap) |
| `spatial_and_multichannel.binaural_itd_ild_synthesis` | Binaural ITD/ILD synthesis | `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/binaural_itd_ild_synthesis.py` | `azimuth_deg`, `itd_max_ms`, `ild_db` | None |
| `spatial_and_multichannel.transaural_crosstalk_cancellation` | Transaural crosstalk cancellation | `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/transaural_crosstalk_cancellation.py` | `cancellation`, `delay_ms` | [Crosstalk cancellation](docs/html/glossary.html#crosstalk-cancellation) |
| `spatial_and_multichannel.stereo_width_frequency_dependent_control` | Stereo width (frequency-dependent control) | `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/stereo_width_frequency_dependent_control.py` | `width_low`, `width_high`, `crossover_hz` | None |
| `spatial_and_multichannel.phase_aligned_mid_side_field_rotation` | Phase-aligned mid/side field rotation | `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/phase_aligned_mid_side_field_rotation.py` | `rotation_deg` | None |
| `spatial_and_multichannel.delay_and_sum_beamforming` | Delay-and-sum beamforming | `src/pvx/algorithms/spatial_and_multichannel/beamforming_and_directionality/delay_and_sum_beamforming.py` | `steer_deg`, `spacing_m`, `sound_speed` | [Beamforming](docs/html/glossary.html#beamforming) |
| `spatial_and_multichannel.mvdr_beamformer_wideband` | MVDR beamformer (wideband) | `src/pvx/algorithms/spatial_and_multichannel/beamforming_and_directionality/mvdr_beamformer_wideband.py` | `steer_deg`, `spacing_m`, `sound_speed`, `diag_load` | [Beamforming](docs/html/glossary.html#beamforming), [MVDR beamformer](docs/html/glossary.html#mvdr-beamformer) |
| `spatial_and_multichannel.generalized_sidelobe_canceller` | Generalized sidelobe canceller | `src/pvx/algorithms/spatial_and_multichannel/beamforming_and_directionality/generalized_sidelobe_canceller.py` | `mu`, `steer_deg`, `spacing_m`, `sound_speed` | [Beamforming](docs/html/glossary.html#beamforming) |
| `spatial_and_multichannel.superdirective_beamformer` | Superdirective beamformer | `src/pvx/algorithms/spatial_and_multichannel/beamforming_and_directionality/superdirective_beamformer.py` | `steer_deg`, `spacing_m`, `sound_speed`, `aperture` | [Beamforming](docs/html/glossary.html#beamforming) |
| `spatial_and_multichannel.diffuse_field_coherence_masking` | Diffuse-field coherence masking | `src/pvx/algorithms/spatial_and_multichannel/beamforming_and_directionality/diffuse_field_coherence_masking.py` | `coherence_threshold`, `floor` | [Beamforming](docs/html/glossary.html#beamforming), [Diffuse field](docs/html/glossary.html#diffuse-field) |
| `spatial_and_multichannel.direction_of_arrival_grid_tracking` | Direction-of-arrival grid tracking | `src/pvx/algorithms/spatial_and_multichannel/beamforming_and_directionality/direction_of_arrival_grid_tracking.py` | `spacing_m`, `sound_speed`, `frame`, `hop` | [Beamforming](docs/html/glossary.html#beamforming) |
| `spatial_and_multichannel.first_order_ambisonic_encode_decode` | First-order ambisonic encode/decode | `src/pvx/algorithms/spatial_and_multichannel/ambisonics_and_immersive/first_order_ambisonic_encode_decode.py` | `azimuth_deg`, `elevation_deg` | [Ambisonics](docs/html/glossary.html#ambisonics) |
| `spatial_and_multichannel.higher_order_ambisonic_rotation` | Higher-order ambisonic rotation | `src/pvx/algorithms/spatial_and_multichannel/ambisonics_and_immersive/higher_order_ambisonic_rotation.py` | `yaw_deg` | [Ambisonics](docs/html/glossary.html#ambisonics) |
| `spatial_and_multichannel.ambisonic_binaural_rendering` | Ambisonic binaural rendering | `src/pvx/algorithms/spatial_and_multichannel/ambisonics_and_immersive/ambisonic_binaural_rendering.py` | `head_itd_ms`, `head_ild_db`, `azimuth_deg` | [Ambisonics](docs/html/glossary.html#ambisonics), [Binaural rendering](docs/html/glossary.html#binaural-rendering) |
| `spatial_and_multichannel.spherical_harmonic_diffuse_enhancement` | Spherical-harmonic diffuse enhancement | `src/pvx/algorithms/spatial_and_multichannel/ambisonics_and_immersive/spherical_harmonic_diffuse_enhancement.py` | `diffuse_mix` | [Ambisonics](docs/html/glossary.html#ambisonics) |
| `spatial_and_multichannel.hoa_order_truncation_and_upmix` | HOA order truncation and upmix | `src/pvx/algorithms/spatial_and_multichannel/ambisonics_and_immersive/hoa_order_truncation_and_upmix.py` | `target_channels` | [Ambisonics](docs/html/glossary.html#ambisonics) |
| `spatial_and_multichannel.spatial_room_impulse_convolution` | Spatial room impulse convolution | `src/pvx/algorithms/spatial_and_multichannel/ambisonics_and_immersive/spatial_room_impulse_convolution.py` | `decay_s`, `rir_length`, `seed` | [Convolution](docs/html/glossary.html#convolution), [Ambisonics](docs/html/glossary.html#ambisonics) |
| `spatial_and_multichannel.pvx_interchannel_phase_locking` | PVX interchannel phase locking | `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interchannel_phase_locking.py` | `lock_strength` | [Phase vocoder](docs/html/glossary.html#phase-vocoder), [Phase locking](docs/html/glossary.html#phase-locking) |
| `spatial_and_multichannel.pvx_spatial_transient_preservation` | PVX spatial transient preservation | `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_transient_preservation.py` | `transient_threshold`, `phase_smooth`, `preserve_amount` | [Phase vocoder](docs/html/glossary.html#phase-vocoder), [Transient preservation](docs/html/glossary.html#transient-preservation) |
| `spatial_and_multichannel.pvx_interaural_coherence_shaping` | PVX interaural coherence shaping | `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interaural_coherence_shaping.py` | `coherence_target` | [Phase vocoder](docs/html/glossary.html#phase-vocoder) |
| `spatial_and_multichannel.pvx_directional_spectral_warp` | PVX directional spectral warp | `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_directional_spectral_warp.py` | `warp_amount`, `azimuth_deg` | [Phase vocoder](docs/html/glossary.html#phase-vocoder) |
| `spatial_and_multichannel.pvx_multichannel_time_alignment` | PVX multichannel time alignment | `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_multichannel_time_alignment.py` | `max_lag` | [Phase vocoder](docs/html/glossary.html#phase-vocoder) |
| `spatial_and_multichannel.pvx_spatial_freeze_and_trajectory` | PVX spatial freeze and trajectory | `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_freeze_and_trajectory.py` | `frame_ratio`, `orbit_hz` | [Phase vocoder](docs/html/glossary.html#phase-vocoder) |
| `spatial_and_multichannel.multichannel_wiener_postfilter` | Multichannel Wiener postfilter | `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_wiener_postfilter.py` | `noise_floor` | None |
| `spatial_and_multichannel.coherence_based_dereverb_multichannel` | Coherence-based dereverb (multichannel) | `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/coherence_based_dereverb_multichannel.py` | `coherence_threshold`, `decay` | None |
| `spatial_and_multichannel.multichannel_noise_psd_tracking` | Multichannel noise PSD tracking | `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_noise_psd_tracking.py` | `alpha`, `floor` | None |
| `spatial_and_multichannel.phase_consistent_multichannel_denoise` | Phase-consistent multichannel denoise | `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/phase_consistent_multichannel_denoise.py` | `reduction_db`, `floor` | None |
| `spatial_and_multichannel.microphone_array_calibration_tones` | Microphone-array calibration tones | `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/microphone_array_calibration_tones.py` | `tone_hz`, `apply_correction` | None |
| `spatial_and_multichannel.cross_channel_click_pop_repair` | Cross-channel click/pop repair | `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/cross_channel_click_pop_repair.py` | `spike_threshold` | None |
| `spatial_and_multichannel.rotating_speaker_doppler_field` | Rotating-speaker Doppler field | `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/rotating_speaker_doppler_field.py` | `output_channels`, `rotation_hz`, `depth_ms` | None |
| `spatial_and_multichannel.binaural_motion_trajectory_designer` | Binaural motion trajectory designer | `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/binaural_motion_trajectory_designer.py` | `trajectory`, `trajectory_hz`, `width`, `itd_ms` | None |
| `spatial_and_multichannel.stochastic_spatial_diffusion_cloud` | Stochastic spatial diffusion cloud | `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/stochastic_spatial_diffusion_cloud.py` | `output_channels`, `diffusion`, `max_delay_ms`, `seed` | None |
| `spatial_and_multichannel.decorrelated_reverb_upmix` | Decorrelated reverb upmix | `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/decorrelated_reverb_upmix.py` | `output_channels`, `decay_s`, `rir_length`, `mix`, `seed` | None |
| `spatial_and_multichannel.spectral_spatial_granulator` | Spectral spatial granulator | `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spectral_spatial_granulator.py` | `output_channels`, `grain`, `spread_semitones`, `density`, `seed` | None |
| `spatial_and_multichannel.spatial_freeze_resynthesis` | Spatial freeze resynthesis | `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spatial_freeze_resynthesis.py` | `output_channels`, `frame_ratio`, `phase_drift` | None |
<!-- END ALGORITHM CATALOG -->


## Shared Conventions (Most Tools)

Most `pvx*` scripts are intentionally aligned around one CLI contract for batch audio workflows.  
`pvxmorph.py` remains the main exception (two explicit inputs and one explicit output path), but now also supports streaming pipes.

### I/O and Output Conventions

| Area | Flags | Behavior | Notes |
| --- | --- | --- | --- |
| Input selection | `inputs` (positional) | Accepts one or more paths/globs | If no paths resolve, the parser exits with an error. |
| Stdin input | `-` as input path | Reads audio bytes from stdin | Can be paired with `--stdout` (pipe through) or normal file output (writes `stdin_*`). |
| Output location | `-o`, `--output-dir` | Writes beside source file when omitted | Applies to tools using shared I/O args. |
| Stdout output | `--stdout` | Writes encoded audio bytes to stdout | Requires exactly one resolved input. |
| Output naming | `--suffix`, `--output-format` | Appends tool suffix and optional extension override | Examples: `_harm`, `_denoise`, `_retune`. |
| Overwrite policy | `--overwrite` | Allows replacing an existing output file | Without it, existing output is an error. |
| Dry-run mode | `--dry-run` | Validates and resolves outputs without writing audio | Useful for large batch verification. |
| File encoding | `--subtype` | Sets libsndfile subtype (`PCM_16`, `PCM_24`, `FLOAT`, etc.) | Leave unset for default output subtype. |

### Unix Pipe Chaining

You can chain tools in one line by piping stdout audio from one tool to stdin (`-`) of the next:

```bash
python3 pvxvoc.py input.wav --time-stretch 1.20 --stdout \
  | python3 pvxdenoise.py - --reduction-db 12 --stdout \
  | python3 pvxdeverb.py - --strength 0.45 --output-dir out --suffix _final
```

For tools with explicit output paths (for example `pvxmorph.py`), use `-o -` or `--stdout`:

```bash
python3 pvxmorph.py a.wav b.wav -o - \
  | python3 pvxformant.py - --mode preserve --stdout \
  | python3 pvxunison.py - --voices 5 --output-dir out --suffix _chain
```

### Console Output and Verbosity

Default behavior is a live status bar plus final summary lines.  
Status bars are shown unless quiet/silent mode is active.

| Control | Effective level | What you see |
| --- | --- | --- |
| No verbosity flags | `normal` | Status bar, final `[done]` summary, and errors (if any). |
| `-v` or `--verbosity verbose` | `verbose` | Normal output plus per-file `[ok]` details. |
| `-vv` or `--verbosity debug` | `debug` | Maximum detail level (currently most visible in `pvxvoc`). |
| `--quiet` or `--verbosity quiet` | `quiet` | No status bar and reduced non-error output. |
| `--silent` or `--verbosity silent` | `silent` | Suppresses all console output. |

Precedence notes:

- Repeating `-v` increases detail (`-vv`, `-vvv`), capped at `debug`.
- `--quiet`/`--silent` override higher verbosity requests.
- `pvxvoc.py` still accepts legacy `--no-progress` as a hidden alias mapped to quiet-style output.

### Mastering and Loudness Controls

| Area | Primary flags | Purpose |
| --- | --- | --- |
| Loudness target | `--target-lufs` | Apply integrated loudness targeting after dynamics/normalization. |
| Compressor | `--compressor-threshold-db`, `--compressor-ratio`, `--compressor-attack-ms`, `--compressor-release-ms`, `--compressor-makeup-db` | Downward compression above threshold. |
| Expander | `--expander-threshold-db`, `--expander-ratio`, `--expander-attack-ms`, `--expander-release-ms` | Downward expansion below threshold. |
| Compander | `--compander-threshold-db`, `--compander-compress-ratio`, `--compander-expand-ratio`, `--compander-attack-ms`, `--compander-release-ms`, `--compander-makeup-db` | Combined expansion/compression around one threshold. |
| Limiter | `--limiter-threshold` | Peak-limiter ceiling in linear full-scale. |
| Soft clipper | `--soft-clip-level`, `--soft-clip-type`, `--soft-clip-drive` | Nonlinear saturation with selectable transfer type (`tanh`, `arctan`, `cubic`). |
| Hard clipper | `--hard-clip-level`, `--clip` | Explicit hard-clip ceiling; `--clip` is legacy alias for `--hard-clip-level 1.0`. |
| Legacy normalization | `--normalize`, `--peak-dbfs`, `--rms-dbfs` | Optional peak/RMS normalization stage retained for compatibility. |

Processing order in the shared mastering chain:
`expander -> compressor -> compander -> normalize -> target LUFS -> limiter -> soft clip -> hard clip`.

### Microtonal Pitch Controls

| Tool | Microtonal inputs | Notes |
| --- | --- | --- |
| `pvxvoc.py` | `--pitch-shift-cents`, fractional `--pitch-shift-semitones`, `--pitch-shift-ratio` | One pitch selector at a time via mutual exclusivity. |
| `pvxtransient.py` | `--pitch-shift-cents`, fractional `--pitch-shift-semitones`, `--pitch-shift-ratio` | Cents are additive to semitones unless ratio is used. |
| `pvxformant.py` | `--pitch-shift-cents`, fractional `--pitch-shift-semitones` | Cents are additive to semitones. |
| `pvxharmonize.py` | fractional `--intervals`, `--intervals-cents` | Cents offsets are per voice and added to intervals. |
| `pvxlayer.py` | `--harmonic-pitch-cents`, `--percussive-pitch-cents` plus semitone controls | Harmonic/percussive paths are controlled independently. |
| `pvxretune.py` | `--scale-cents` | Custom octave degree map in cents relative to `--root`. |
| `pvxconform.py` | CSV `pitch_cents` / `pitch_ratio` / `pitch_semitones` | Each segment row provides exactly one pitch field. |

### STFT and Windowing Controls

| Area | Flags | Purpose |
| --- | --- | --- |
| FFT framing | `--n-fft`, `--win-length`, `--hop-size` | Controls frequency/time resolution and overlap. |
| Window shape | `--window` | Selects one of 50 available analysis windows. |
| Kaiser tuning | `--kaiser-beta` | Beta parameter when `--window kaiser` is selected. |
| Frame centering | `--no-center` | Disables centered framing. |

Window families include classic, cosine-sum, tapered, gaussian/generalized gaussian, exponential, cauchy, cosine-power, hann-poisson, and general-hamming variants.

All supported windows: `hann`, `hamming`, `blackman`, `blackmanharris`, `nuttall`, `flattop`, `blackman_nuttall`, `exact_blackman`, `sine`, `bartlett`, `boxcar`, `triangular`, `bartlett_hann`, `tukey`, `tukey_0p1`, `tukey_0p25`, `tukey_0p75`, `tukey_0p9`, `parzen`, `lanczos`, `welch`, `gaussian_0p25`, `gaussian_0p35`, `gaussian_0p45`, `gaussian_0p55`, `gaussian_0p65`, `general_gaussian_1p5_0p35`, `general_gaussian_2p0_0p35`, `general_gaussian_3p0_0p35`, `general_gaussian_4p0_0p35`, `exponential_0p25`, `exponential_0p5`, `exponential_1p0`, `cauchy_0p5`, `cauchy_1p0`, `cauchy_2p0`, `cosine_power_2`, `cosine_power_3`, `cosine_power_4`, `hann_poisson_0p5`, `hann_poisson_1p0`, `hann_poisson_2p0`, `general_hamming_0p50`, `general_hamming_0p60`, `general_hamming_0p70`, `general_hamming_0p80`, `bohman`, `cosine`, `kaiser`, `rect`.

Mathematical definitions and plain-English explanations for every window above are provided in:

- [`docs/WINDOW_REFERENCE.md`](docs/WINDOW_REFERENCE.md) (GitHub-renderable LaTeX)
- [`docs/html/windows.html`](docs/html/windows.html) (browser-rendered HTML with MathJax)
- [`docs/window_metrics.json`](docs/window_metrics.json) (machine-readable metrics for all windows)
- `docs/assets/windows/*` (generated SVG time-domain and spectrum plots per window)

Use `--help` on any `pvx*` tool to view the exact current window list.

### Runtime and Acceleration Controls

| Flag | Values | Behavior |
| --- | --- | --- |
| `--device` | `auto`, `cpu`, `cuda` | Chooses runtime backend for FFT/vocoder operations. |
| `--cuda-device` | non-negative integer | Selects CUDA GPU index when CUDA backend is active. |

### Exit Behavior

| Scenario | Exit code | Behavior |
| --- | --- | --- |
| All files processed successfully | `0` | Normal completion. |
| One or more files fail in batch mode | `1` | Tool keeps processing remaining files, then reports failures. |
| Invalid arguments/configuration | non-zero | Parser/runtime validation exits before processing. |

## Tool Reference

### 1. `pvxvoc.py`

Description:

`pvxvoc.py` is the canonical production engine in `pvx`. It performs STFT analysis, phase-coherent resynthesis, optional instantaneous-frequency correction, and output-domain post stages (normalization/mastering chain) in one CLI flow.
It is designed for both transparent utility transforms (small stretch/pitch adjustments) and aggressive creative transforms (large stretch, formant-preserving pitch movement, transient-aware processing).

Internally, this tool is where runtime acceleration and most shared DSP controls meet:

- CPU/CUDA backend selection (`--device` / `--cuda-device`).
- Full STFT/window controls (`--n-fft`, `--win-length`, `--hop-size`, `--window`, `--kaiser-beta`).
- Phase and transient controls (`--phase-locking`, `--transient-preserve`, threshold tuning).
- Output mastering pipeline (LUFS target, compressor/expander/compander/limiter/clipper).

Use `pvxvoc.py` when you want maximum control and one-pass consistency for batch processing, automation, or chained Unix pipelines.

Key capabilities:

- Multi-file + multi-channel processing.
- Time stretch by ratio (`--time-stretch`) or absolute length (`--target-duration`).
- Pitch by semitones, cents, ratio, or target F0 (`--target-f0`).
- Formant-preserving pitch mode (`--pitch-mode formant-preserving`).
- Transient-preserve + phase locking.
- Fourier-sync mode (`--fourier-sync`) with fundamental frame locking.
- Live status bar by default (`--quiet` or `--silent` suppresses it; legacy `--no-progress` is accepted as an alias).

Example:

```bash
python3 pvxvoc.py stems/*.wav \
  --time-stretch 1.2 \
  --pitch-shift-semitones -2 \
  --phase-locking identity \
  --transient-preserve \
  --normalize peak --peak-dbfs -1
```

Fourier-sync example:

```bash
python3 pvxvoc.py vocal.wav \
  --fourier-sync --n-fft 1500 --win-length 1500 --hop-size 375 \
  --f0-min 70 --f0-max 500 --time-stretch 1.25
```

### 2. `pvxfreeze.py`

Description:

`pvxfreeze.py` captures a selected STFT frame and resynthesizes it over time as a sustained texture.
Instead of pitch-correcting or time-warping the source globally, freeze mode locks spectral content around one moment and prolongs it, which is useful for ambient pads, transitions, drones, and time-suspended sound design.

The tool keeps shared output/metering conventions, so it can be inserted in the same batch and pipeline workflows as other `pvx*` utilities.

Key options:

- `--freeze-time`: time (seconds) where freeze frame is captured.
- `--duration`: length of rendered freeze.
- `--random-phase`: subtle animated phase drift.

Example:

```bash
python3 pvxfreeze.py pad.wav --freeze-time 0.42 --duration 8.0 --random-phase
```

### 3. `pvxharmonize.py`

Description:

`pvxharmonize.py` builds a multi-voice harmony stack from one source by spawning independent pitch-shifted voices, then mixing and optionally spatially distributing them.
Each voice can have its own semitone/cents offset, gain, and pan, making the tool suitable for tight interval doubling, choir-like widening, or synthetic ensemble textures.

For predictable musical results, treat `--intervals` as the harmonic design layer and `--gains`/`--pans` as arrangement/mix controls.

Key options:

- `--intervals`: semitone list per voice (fractional values allowed).
- `--intervals-cents`: optional per-voice cents offsets added to intervals.
- `--gains`: per-voice gains.
- `--pans`: per-voice pan in `[-1, 1]`.
- `--force-stereo`: force stereo render path.

Example:

```bash
python3 pvxharmonize.py lead.wav \
  --intervals 0,3,7 \
  --gains 1.0,0.8,0.7 \
  --pans -0.4,0,0.4 \
  --force-stereo
```

### 4. `pvxconform.py`

Description:

`pvxconform.py` applies timeline-specific time and pitch transformations from a map file, segment by segment.
It is intended for deterministic alignment workflows where transformation values must be explicit over time, such as ADR conform, vocal contour shaping from editorial maps, or experiment-driven microtonal score alignment.

Each segment can define independent stretch plus one pitch representation (`semitones`, `cents`, or direct ratio), allowing precise control while preserving batch repeatability.

Required CSV columns:

- `start_sec`
- `end_sec`
- `stretch`
- one pitch field per row: `pitch_semitones` or `pitch_cents` or `pitch_ratio`

Key options:

- `--map`: CSV path.
- `--crossfade-ms`: segment boundary smoothing.

Example CSV (`map_conform.csv`) variants:

`map_conform_basic_cents.csv` (small expressive offsets):

```csv
start_sec,end_sec,stretch,pitch_cents
0.0,1.0,1.00,0
1.0,2.4,1.20,35
2.4,3.1,0.90,-28
```

`map_conform_quarter_tone_24edo.csv` (24-EDO quarter-tone steps, 50 cents each):

```csv
start_sec,end_sec,stretch,pitch_cents
0.0,0.8,1.00,0
0.8,1.6,1.00,50
1.6,2.4,1.00,100
2.4,3.2,1.00,150
3.2,4.0,1.00,200
```

`map_conform_19edo.csv` (19-EDO steps, 1200/19 = 63.1579 cents):

```csv
start_sec,end_sec,stretch,pitch_cents
0.0,0.9,1.00,0.0000
0.9,1.8,1.00,63.1579
1.8,2.7,1.00,126.3158
2.7,3.6,1.00,189.4737
3.6,4.5,1.00,252.6316
```

`map_conform_just_intonation_ratios.csv` (JI pitch targets via direct ratios):

```csv
start_sec,end_sec,stretch,pitch_ratio
0.0,1.0,1.00,1.000000
1.0,2.0,1.00,1.066667
2.0,3.0,1.00,1.125000
3.0,4.0,1.00,1.200000
```

`map_conform_equal_temperament_semitones.csv` (standard semitone offsets):

```csv
start_sec,end_sec,stretch,pitch_semitones
0.0,1.0,1.00,0
1.0,2.0,1.00,2
2.0,3.0,0.95,-3
3.0,4.0,1.05,5
```

Example command:

```bash
python3 pvxconform.py vocal.wav --map map_conform_19edo.csv --crossfade-ms 10
```

### 5. `pvxmorph.py`

Description:

`pvxmorph.py` blends two sources in the spectral domain, combining short-time magnitude/phase characteristics according to `--alpha`.
This is not a simple crossfade: it performs time-frequency-domain interpolation, so spectral color, articulation, and timbral identity can migrate between inputs.

It is useful for timbral hybridization, transitional effects, and controlled spectral interpolation in sound design pipelines.

Key options:

- `input_a`, `input_b`: two source files.
- `--alpha`: morph blend (`0` = A, `1` = B).
- `-o/--output`: explicit output path.

Example:

```bash
python3 pvxmorph.py source_a.wav source_b.wav --alpha 0.35 -o hybrid.wav
```

### 6. `pvxwarp.py`

Description:

`pvxwarp.py` performs variable, segment-wise time deformation from CSV without applying pitch changes.
Use it when you need expressive or editorial timing reshaping (rubato-like warps, section alignment, phrasing correction) while preserving overall pitch structure.

Unlike global `--time-stretch`, map-driven warping lets each interval run faster/slower with explicit boundaries and optional crossfade smoothing at joins.

Required CSV columns:

- `start_sec`
- `end_sec`
- `stretch`

Key options:

- `--map`: CSV path.
- `--crossfade-ms`: segment joins.

Example CSV (`map_warp.csv`):

```csv
start_sec,end_sec,stretch
0.0,0.8,1.0
0.8,2.0,1.3
2.0,3.0,0.85
```

Example command:

```bash
python3 pvxwarp.py drums.wav --map map_warp.csv
```

### 7. `pvxformant.py`

Description:

`pvxformant.py` targets spectral-envelope behavior (formants) and is useful when pitch and perceived vocal/instrument body should be controlled separately.
In `preserve` mode it compensates formant drift during pitch movement; in `shift` mode it intentionally relocates formants for creative voice-shape effects.

This tool is especially relevant for vocal work where naive pitch shifting can introduce chipmunk/baritone artifacts that are primarily envelope-related, not fundamental-frequency-related.

Key options:

- `--pitch-shift-semitones`: optional pitch shift before formant stage.
- `--pitch-shift-cents`: microtonal cents offset added to semitone shift.
- `--formant-shift-ratio`: formant move factor.
- `--mode {shift,preserve}`:
  - `shift`: explicitly shift formants.
  - `preserve`: compensate formants when pitch shifting.
- `--formant-lifter`, `--formant-max-gain-db`: envelope extraction and gain limits.

Example:

```bash
python3 pvxformant.py voice.wav \
  --pitch-shift-semitones 4 \
  --mode preserve \
  --formant-lifter 32
```

### 8. `pvxtransient.py`

Description:

`pvxtransient.py` biases processing toward transient integrity by adjusting behavior around onset-like frames.
It is intended for material where attack clarity matters (drums, percussive instruments, plucked sources) and where standard phase-vocoder settings may smear sharp events under stretch.

Use it when you want time/pitch manipulation but with stronger onset protection than a default full-spectrum transform.

Key options:

- `--time-stretch` or `--target-duration`.
- `--pitch-shift-semitones`, `--pitch-shift-cents`, or `--pitch-shift-ratio`.
- `--transient-threshold`: transient detector sensitivity.

Example:

```bash
python3 pvxtransient.py percussion.wav --time-stretch 1.35 --transient-threshold 1.4
```

### 9. `pvxunison.py`

Description:

`pvxunison.py` creates detuned multi-voice layering around the source to increase width, density, and perceived scale.
It is conceptually similar to ensemble/unison synth effects, but exposed as deterministic CLI controls for voice count, detune spread, stereo width, and dry/wet balance.

Common use cases include mono lead widening, subtle stereo enhancement, and intentionally exaggerated chorus-like texture design.

Key options:

- `--voices`: number of unison voices.
- `--detune-cents`: overall detune span.
- `--width`: stereo spread control.
- `--dry-mix`: blend original signal.

Example:

```bash
python3 pvxunison.py synth.wav --voices 7 --detune-cents 18 --width 1.2 --dry-mix 0.15
```

### 10. `pvxdenoise.py`

Description:

`pvxdenoise.py` performs spectral noise reduction using a configurable noise estimate and subtraction profile.
The estimate can come from an initial segment of the target file or from an external noise-print recording, making the tool suitable for practical cleanup workflows with controlled capture conditions.

Core tuning parameters (`--reduction-db`, `--floor`, `--smooth`) trade off suppression strength vs artifact risk; moderate settings generally preserve timbre better in mixed-content material.

Key options:

- `--noise-seconds`: profile from beginning of target file.
- `--noise-file`: optional separate noise capture.
- `--reduction-db`: subtraction strength.
- `--floor`: residual floor factor.
- `--smooth`: temporal smoothing span.

Example:

```bash
python3 pvxdenoise.py noisy.wav --noise-file noise_print.wav --reduction-db 14 --smooth 7
```

### 11. `pvxdeverb.py`

Description:

`pvxdeverb.py` attenuates late reverberant energy and spectral tail persistence to improve clarity and articulation.
It is useful for overly live recordings, speech intelligibility recovery, and pre-processing before downstream pitch/transcription tasks that are sensitive to diffuse tails.

`--strength`, `--decay`, and `--floor` define the aggressiveness/memory/safety balance; overdriving these can make material unnaturally dry or hollow.

Key options:

- `--strength`: tail suppression amount.
- `--decay`: tail memory decay.
- `--floor`: per-bin floor safeguard.

Example:

```bash
python3 pvxdeverb.py room_take.wav --strength 0.5 --decay 0.9 --floor 0.12
```

### 12. `pvxretune.py`

Description:

`pvxretune.py` retunes monophonic content toward a specified tonal framework (preset scales or custom cents maps).
It is the primary scale-aware correction tool in `pvx`, with explicit support for microtonal degree definitions through `--scale-cents`.

Use this tool when correction should follow scale membership rather than fixed global transposition; `--strength` can be used to move between subtle guidance and near-hard quantization.

Key options:

- `--root`: tonic note (`C`, `C#`, ... `B`).
- `--scale`: `chromatic`, `major`, `minor`, `pentatonic`.
- `--scale-cents`: optional custom microtonal degree list (cents in one octave relative to `--root`).
- `--strength`: correction amount.
- `--chunk-ms`, `--overlap-ms`: analysis block design.
- `--f0-min`, `--f0-max`: F0 detection bounds.

Example:

```bash
python3 pvxretune.py vocal_mono.wav --root A --scale major --strength 0.9
```

### 13. `pvxlayer.py`

Description:

`pvxlayer.py` performs harmonic/percussive decomposition and processes each layer with independent time/pitch/gain controls before recombination.
This enables musically useful asymmetric edits, such as stretching harmonic sustain while keeping percussion tight, or retuning tonal content while leaving transient content nearly unchanged.

Because layer paths are independently controlled, this tool is often a better choice than a single global transform when source material combines stable pitched content and sharp transients.

Key options:

- Harmonic path: `--harmonic-stretch`, `--harmonic-pitch-semitones`, `--harmonic-pitch-cents`, `--harmonic-gain`.
- Percussive path: `--percussive-stretch`, `--percussive-pitch-semitones`, `--percussive-pitch-cents`, `--percussive-gain`.
- HPSS controls: `--harmonic-kernel`, `--percussive-kernel`.

Example:

```bash
python3 pvxlayer.py loop.wav \
  --harmonic-stretch 1.2 --harmonic-pitch-semitones 3 --harmonic-gain 0.9 \
  --percussive-stretch 1.0 --percussive-pitch-semitones 0 --percussive-gain 1.1
```

## Build Executables (All Major Desktop OSes)

`pvx` uses [PyInstaller](https://pyinstaller.org/) for executable builds.

Important notes:

- Build on each target OS directly.
- Cross-OS builds are not officially supported (Windows binaries should be built on Windows, etc.).
- For broad compatibility, build on the oldest platform you intend to support.

### Common Build Preparation

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pyinstaller
```

### Build One Tool

Example (`pvxvoc`):

```bash
python3 -m PyInstaller --onefile --name pvxvoc --collect-all soundfile pvxvoc.py
```

### Build All Tools (macOS/Linux shell)

```bash
for tool in pvxvoc pvxfreeze pvxharmonize pvxconform pvxmorph pvxwarp pvxformant pvxtransient pvxunison pvxdenoise pvxdeverb pvxretune pvxlayer; do
  python3 -m PyInstaller --onefile --name "$tool" --collect-all soundfile "$tool.py"
done
```

### Build All Tools (Windows PowerShell)

```powershell
$tools = @("pvxvoc","pvxfreeze","pvxharmonize","pvxconform","pvxmorph","pvxwarp","pvxformant","pvxtransient","pvxunison","pvxdenoise","pvxdeverb","pvxretune","pvxlayer")
foreach ($tool in $tools) {
  python -m PyInstaller --onefile --name $tool --collect-all soundfile "$tool.py"
}
```

### Platform-Specific Notes

#### Windows

- Use PowerShell or Command Prompt.
- Output binaries are `.exe` files in `dist\`.
- Install the Microsoft VC++ runtime if running on clean machines.

Run example:

```powershell
.\dist\pvxvoc.exe --help
```

#### macOS (Apple Silicon + Intel)

- Build separately on Apple Silicon and Intel for native binaries.
- Gatekeeper may quarantine unsigned binaries distributed outside your machine.
- For distribution, code-sign and notarize executables.

Run example:

```bash
./dist/pvxvoc --help
```

#### Linux

- Build on a baseline distro for best `glibc` compatibility.
- Prefer same distro family/version as deployment targets.
- Output is an ELF binary in `dist/`.

Run example:

```bash
./dist/pvxvoc --help
```

### Onefile vs Onedir

- `--onefile`: single executable, easier distribution, slower startup.
- `--onedir`: folder output, faster startup, easier debugging.

Onedir example:

```bash
python3 -m PyInstaller --onedir --name pvxvoc --collect-all soundfile pvxvoc.py
```

## Validation / Test

Run unit + regression tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Run CLI help checks for all tools:

```bash
for tool in pvxvoc pvxfreeze pvxharmonize pvxconform pvxmorph pvxwarp pvxformant pvxtransient pvxunison pvxdenoise pvxdeverb pvxretune pvxlayer; do
  python3 "$tool.py" --help > /dev/null
done
```
