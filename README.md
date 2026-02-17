<p align="center"><br><br></p>

<p align="center">
  <img src="assets/pvx_logo.png" alt="pvx logo" width="360" />
</p>

<p align="center"><br><br></p>

# pvx

`pvx` is a multi-tool audio DSP toolkit built around phase-vocoder workflows.

The repository currently includes:

- `pvxvoc.py`: full-feature phase vocoder (time/pitch/F0/formant/fourier-sync).
- `pvxfreeze.py`: spectral freeze / sustained texture generation.
- `pvxharmonize.py`: multi-voice harmonizer.
- `pvxconform.py`: time+pitch conforming via CSV map.
- `pvxmorph.py`: spectral morph between two files.
- `pvxwarp.py`: variable time warp via CSV map.
- `pvxformant.py`: formant shift/preserve processing.
- `pvxtransient.py`: transient-first time/pitch processing.
- `pvxunison.py`: stereo detuned unison thickener.
- `pvxdenoise.py`: spectral subtraction denoiser.
- `pvxdeverb.py`: reverb-tail suppression.
- `pvxretune.py`: monophonic scale retune.
- `pvxlayer.py`: harmonic/percussive split and independent processing.
- `pvxalgorithms/`: themed algorithm library with 95 DSP modules (time/pitch, transforms, separation, denoise, dereverb, dynamics, spatial, creative, granular, and analysis).

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

## Shared Conventions (Most Tools)

Most `pvx*` scripts are intentionally aligned around one CLI contract for batch audio workflows.  
`pvxmorph.py` is the main exception (two explicit inputs and one explicit output path).

### I/O and Output Conventions

| Area | Flags | Behavior | Notes |
| --- | --- | --- | --- |
| Input selection | `inputs` (positional) | Accepts one or more paths/globs | If no paths resolve, the parser exits with an error. |
| Output location | `-o`, `--output-dir` | Writes beside source file when omitted | Applies to tools using shared I/O args. |
| Output naming | `--suffix`, `--output-format` | Appends tool suffix and optional extension override | Examples: `_harm`, `_denoise`, `_retune`. |
| Overwrite policy | `--overwrite` | Allows replacing an existing output file | Without it, existing output is an error. |
| Dry-run mode | `--dry-run` | Validates and resolves outputs without writing audio | Useful for large batch verification. |
| File encoding | `--subtype` | Sets libsndfile subtype (`PCM_16`, `PCM_24`, `FLOAT`, etc.) | Leave unset for default output subtype. |

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

### Signal Level and Clipping Controls

| Flag | Values | Purpose |
| --- | --- | --- |
| `--normalize` | `none`, `peak`, `rms` | Applies output normalization strategy. |
| `--peak-dbfs` | float dBFS | Target peak when `--normalize peak`. |
| `--rms-dbfs` | float dBFS | Target RMS when `--normalize rms`. |
| `--clip` | boolean | Hard-limits output samples to `[-1.0, 1.0]`. |

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

Primary phase-vocoder engine for batch time/pitch processing.

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

Freezes one spectral snapshot and synthesizes a sustained output.

Key options:

- `--freeze-time`: time (seconds) where freeze frame is captured.
- `--duration`: length of rendered freeze.
- `--random-phase`: subtle animated phase drift.

Example:

```bash
python3 pvxfreeze.py pad.wav --freeze-time 0.42 --duration 8.0 --random-phase
```

### 3. `pvxharmonize.py`

Creates harmony voices by pitch-shifting and summing.

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

Conforms timing and pitch by segments from CSV.

Required CSV columns:

- `start_sec`
- `end_sec`
- `stretch`
- one pitch field per row: `pitch_semitones` or `pitch_cents` or `pitch_ratio`

Key options:

- `--map`: CSV path.
- `--crossfade-ms`: segment boundary smoothing.

Example CSV (`map_conform.csv`):

```csv
start_sec,end_sec,stretch,pitch_cents
0.0,1.0,1.0,0
1.0,2.4,1.2,35
2.4,3.1,0.9,-28
```

Example command:

```bash
python3 pvxconform.py vocal.wav --map map_conform.csv --crossfade-ms 10
```

### 5. `pvxmorph.py`

Morphs two inputs in STFT magnitude/phase space.

Key options:

- `input_a`, `input_b`: two source files.
- `--alpha`: morph blend (`0` = A, `1` = B).
- `-o/--output`: explicit output path.

Example:

```bash
python3 pvxmorph.py source_a.wav source_b.wav --alpha 0.35 -o hybrid.wav
```

### 6. `pvxwarp.py`

Time-only variable stretch from CSV map.

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

Formant-domain processing with optional pitch shift stage.

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

Transient-emphasis variant of phase-vocoder processing.

Key options:

- `--time-stretch` or `--target-duration`.
- `--pitch-shift-semitones`, `--pitch-shift-cents`, or `--pitch-shift-ratio`.
- `--transient-threshold`: transient detector sensitivity.

Example:

```bash
python3 pvxtransient.py percussion.wav --time-stretch 1.35 --transient-threshold 1.4
```

### 9. `pvxunison.py`

Stereo thickening via multiple detuned voices.

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

Spectral subtraction denoiser with optional external noise reference.

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

Reduces spectral tail smear (de-reverb style).

Key options:

- `--strength`: tail suppression amount.
- `--decay`: tail memory decay.
- `--floor`: per-bin floor safeguard.

Example:

```bash
python3 pvxdeverb.py room_take.wav --strength 0.5 --decay 0.9 --floor 0.12
```

### 12. `pvxretune.py`

Monophonic retuning toward a scale.

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

Splits audio into harmonic/percussive components, then applies independent pitch/time processing and recombines.

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
