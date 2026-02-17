<p align="center"><br><br></p>

<p align="center">
  <img src="assets/pvx_logo.png" alt="pvx logo" width="360" />
</p>

<p align="center"><br><br></p>

# pvx

`pvx` is a phase-vocoder-based DSP toolkit with multiple command-line processors:
`pvxvoc`, `pvxfreeze`, `pvxharmonize`, `pvxconform`, `pvxmorph`, `pvxwarp`,
`pvxformant`, `pvxtransient`, `pvxunison`, `pvxdenoise`, `pvxdeverb`, `pvxretune`, and `pvxlayer`.

This README focuses on `pvxvoc`, the core processor.

It supports:
- One or more input files in a single run
- Mono, stereo, or arbitrary channel counts
- FFT/window/hop controls
- Transient-preserving phase locking
- Fourier-sync mode with fundamental frame locking
- Time stretch by ratio or absolute target duration
- Pitch shift by semitones, ratio, or target fundamental frequency
- Formant-preserving pitch mode
- Optional output sample-rate conversion
- Optional peak/RMS normalization

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Basic Usage

```bash
python3 pvxvoc.py input.wav
```

Default output:
- File: `input_pv.wav`
- Location: same directory as input
- Processing: no time/pitch change (acts as a pass-through transform)
- A per-file processing progress bar is shown in interactive terminals

## Other Tools

- `python3 pvxfreeze.py <in.wav>`: spectral freeze / sustain.
- `python3 pvxharmonize.py <in.wav>`: multi-voice harmonizer.
- `python3 pvxconform.py <in.wav> --map map.csv`: pitch+timing map conform.
- `python3 pvxmorph.py <a.wav> <b.wav>`: spectral morphing.
- `python3 pvxwarp.py <in.wav> --map map.csv`: time-warp map processing.
- `python3 pvxformant.py <in.wav>`: formant shift/preserve processing.
- `python3 pvxtransient.py <in.wav>`: transient-first vocoder mode.
- `python3 pvxunison.py <in.wav>`: stereo micro-detune unison.
- `python3 pvxdenoise.py <in.wav>`: spectral denoise.
- `python3 pvxdeverb.py <in.wav>`: tail suppression / deverb.
- `python3 pvxretune.py <in.wav>`: monophonic retuning.
- `python3 pvxlayer.py <in.wav>`: harmonic/percussive layered processing.

## CLI

```bash
python3 pvxvoc.py [options] <input1> [<input2> ...]
```

### Common Examples

Time stretch 1.5x longer:

```bash
python3 pvxvoc.py song.wav --time-stretch 1.5
```

Pitch up by 3 semitones (preserving final duration unless overridden):

```bash
python3 pvxvoc.py vocals.wav --pitch-shift-semitones 3
```

Pitch by ratio and set exact output duration:

```bash
python3 pvxvoc.py take.wav --pitch-shift-ratio 0.8 --target-duration 12.0
```

Shift to a target F0 (auto-estimates input F0 first):

```bash
python3 pvxvoc.py voice.wav --target-f0 220 --f0-min 70 --f0-max 400
```

Batch process several files with custom STFT params:

```bash
python3 pvxvoc.py stems/*.wav \
  --n-fft 4096 --win-length 4096 --hop-size 1024 --window hann \
  --phase-locking identity --transient-preserve --transient-threshold 1.8 \
  --time-stretch 1.2 --pitch-shift-semitones -2
```

Fourier-sync mode (non-power-of-two FFT and frame locking to F0):

```bash
python3 pvxvoc.py vocal.wav \
  --fourier-sync --n-fft 1500 --win-length 1500 --hop-size 375 \
  --f0-min 70 --f0-max 500 --time-stretch 1.25
```

Write FLAC outputs into another directory:

```bash
python3 pvxvoc.py mixes/*.wav -o out --output-format flac --suffix _pv2 --overwrite
```

Convert output sample rate and normalize peaks:

```bash
python3 pvxvoc.py input.wav --target-sample-rate 48000 --normalize peak --peak-dbfs -1
```

Formant-preserving pitch shift:

```bash
python3 pvxvoc.py vocal.wav \
  --pitch-shift-semitones 4 \
  --pitch-mode formant-preserving \
  --formant-lifter 32 --formant-strength 1.0
```

Dry-run config validation (no files written):

```bash
python3 pvxvoc.py input.wav --time-stretch 0.85 --dry-run --verbose
```

## Option Summary

### Input / Output

- `inputs`: one or more input files (supports shell globs)
- `-o, --output-dir`: output directory (default: per-input directory)
- `--suffix`: filename suffix before extension (default: `_pv`)
- `--output-format`: output extension/format (e.g. `wav`, `flac`, `aiff`)
- `--overwrite`: replace existing outputs
- `--subtype`: libsndfile subtype (`PCM_16`, `PCM_24`, `FLOAT`, etc)
- `--dry-run`: plan processing without writing files
- `--verbose`: print per-file detailed processing diagnostics
- `--no-progress`: disable terminal progress bars

### Phase Vocoder / STFT

- `--n-fft`: FFT size (default `2048`)
- `--win-length`: analysis/synthesis window length (default `2048`)
- `--hop-size`: hop size in samples (default `512`)
- `--window`: `hann`, `hamming`, `blackman`, `rect`
- `--no-center`: disable centered frame padding
- `--phase-locking`: `off` or `identity` phase locking
- `--transient-preserve`: spectral-flux based transient phase resets
- `--transient-threshold`: transient detection sensitivity
- `--fourier-sync`: generic short-time Fourier mode with per-frame FFT locking to detected F0
- `--fourier-sync-min-fft`, `--fourier-sync-max-fft`: bounds for per-frame FFT sizes in sync mode
- `--fourier-sync-smooth`: smoothing span for prescanned F0 trajectory

### Time Controls

- `--time-stretch`: final duration multiplier (`2.0` = twice as long)
- `--target-duration`: absolute target duration in seconds (overrides `--time-stretch`)

### Pitch Controls

Exactly one of these may be used at a time:
- `--pitch-shift-semitones`
- `--pitch-shift-ratio`
- `--target-f0`

Related options:
- `--analysis-channel`: F0 analysis source for `--target-f0` (`mix` or `first`)
- `--f0-min`, `--f0-max`: F0 search bounds (Hz)
- `--pitch-mode`: `standard` or `formant-preserving`
- `--formant-lifter`: cepstral lifter size for envelope extraction
- `--formant-strength`: blend amount from `0.0` to `1.0`
- `--formant-max-gain-db`: formant correction gain limit

### Resampling / Level

- `--target-sample-rate`: output sample rate in Hz
- `--resample-mode`: `auto` (default), `fft`, `linear`
- `--normalize`: `none`, `peak`, `rms`
- `--peak-dbfs`: target peak dBFS for peak normalization
- `--rms-dbfs`: target RMS dBFS for RMS normalization
- `--clip`: hard-clip to `[-1, 1]` after processing

## Notes

- Multi-channel handling: each channel is processed independently with identical transform settings.
- Pitch shifting is implemented by combining phase-vocoder time scaling with resampling.
- `--target-f0` uses autocorrelation and works best on monophonic, pitched signals.
- Very transient-heavy material can benefit from larger FFT/hop tuning tradeoffs.
- `--fourier-sync` assumes mostly harmonic content and performs a prescan to lock frame sizes so detected F0 maps to integer spectral bins.

## Build Executables

Install dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

Build a single-file executable with PyInstaller:

```bash
python3 -m PyInstaller --onefile --name pvxvoc --collect-all soundfile pvxvoc.py
```

Build outputs:
- macOS/Linux: `dist/pvxvoc`
- Windows: `dist/pvxvoc.exe`

Run the executable:

```bash
./dist/pvxvoc --help
```

## Test Suite

Run all tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Included:
- DSP unit tests (stretch sizing, F0 estimation)
- Transient-preservation and phase-locking behavior checks
- Formant-preservation behavior checks
- CLI integration tests for multi-channel files
- Audio regression metric snapshot checks
