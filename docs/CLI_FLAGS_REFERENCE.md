# PVX CLI Flags Reference

_Generated from commit `58b1a2e` (commit date: 2026-02-17T09:29:32-05:00)._

This file enumerates long-form CLI flags discovered from argparse declarations in canonical PVX CLI sources.

Total tool+flag entries: **139**
Total unique long flags: **111**

## Unique Long Flags

`--alpha`, `--analysis-channel`, `--chunk-ms`, `--clip`, `--compander-attack-ms`, `--compander-compress-ratio`, `--compander-expand-ratio`, `--compander-makeup-db`, `--compander-release-ms`, `--compander-threshold-db`, `--compressor-attack-ms`, `--compressor-makeup-db`, `--compressor-ratio`, `--compressor-release-ms`, `--compressor-threshold-db`, `--crossfade-ms`, `--cuda-device`, `--decay`, `--detune-cents`, `--device`, `--dry-mix`, `--dry-run`, `--duration`, `--expander-attack-ms`, `--expander-ratio`, `--expander-release-ms`, `--expander-threshold-db`, `--f0-max`, `--f0-min`, `--floor`, `--force-stereo`, `--formant-lifter`, `--formant-max-gain-db`, `--formant-shift-ratio`, `--formant-strength`, `--fourier-sync`, `--fourier-sync-max-fft`, `--fourier-sync-min-fft`, `--fourier-sync-smooth`, `--freeze-time`, `--gains`, `--hard-clip-level`, `--harmonic-gain`, `--harmonic-kernel`, `--harmonic-pitch-cents`, `--harmonic-pitch-semitones`, `--harmonic-stretch`, `--hop-size`, `--intervals`, `--intervals-cents`, `--kaiser-beta`, `--limiter-threshold`, `--list-algorithm-package`, `--list-tools`, `--map`, `--mode`, `--n-fft`, `--no-center`, `--no-progress`, `--noise-file`, `--noise-seconds`, `--normalize`, `--output`, `--output-dir`, `--output-format`, `--overlap-ms`, `--overwrite`, `--pans`, `--peak-dbfs`, `--percussive-gain`, `--percussive-kernel`, `--percussive-pitch-cents`, `--percussive-pitch-semitones`, `--percussive-stretch`, `--phase-locking`, `--pitch-mode`, `--pitch-shift-cents`, `--pitch-shift-ratio`, `--pitch-shift-semitones`, `--quiet`, `--random-phase`, `--reduction-db`, `--resample-mode`, `--rms-dbfs`, `--root`, `--scale`, `--scale-cents`, `--show-docs`, `--silent`, `--smooth`, `--soft-clip-drive`, `--soft-clip-level`, `--soft-clip-type`, `--stdout`, `--strength`, `--subtype`, `--suffix`, `--target-duration`, `--target-f0`, `--target-lufs`, `--target-pitch-shift-semitones`, `--target-sample-rate`, `--time-stretch`, `--transient-preserve`, `--transient-threshold`, `--verbose`, `--verbosity`, `--voices`, `--width`, `--win-length`, `--window`

## `main.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--list-algorithm-package` | False | `` | `` | `store_true` | Print location of pvxalgorithms package and registry files | `src/pvx/cli/main.py` |
| `--list-tools` | False | `` | `` | `store_true` | Print the primary pvx CLI tools | `src/pvx/cli/main.py` |
| `--show-docs` | False | `` | `` | `store_true` | Print key generated documentation paths | `src/pvx/cli/main.py` |

## `pvxconform.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--crossfade-ms` | False | `8.0` | `` | `` | Segment crossfade in milliseconds | `src/pvx/cli/pvxconform.py` |
| `--map` | True | `` | `` | `` | CSV map path | `src/pvx/cli/pvxconform.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxconform.py` |

## `pvxdenoise.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--floor` | False | `0.1` | `` | `` | Noise floor multiplier | `src/pvx/cli/pvxdenoise.py` |
| `--noise-file` | False | `` | `` | `` | Optional external noise reference | `src/pvx/cli/pvxdenoise.py` |
| `--noise-seconds` | False | `0.35` | `` | `` | Noise profile duration from start | `src/pvx/cli/pvxdenoise.py` |
| `--reduction-db` | False | `12.0` | `` | `` | Reduction strength in dB | `src/pvx/cli/pvxdenoise.py` |
| `--smooth` | False | `5` | `` | `` | Temporal smoothing frames | `src/pvx/cli/pvxdenoise.py` |

## `pvxdeverb.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--decay` | False | `0.92` | `` | `` | Tail memory decay 0..1 | `src/pvx/cli/pvxdeverb.py` |
| `--floor` | False | `0.12` | `` | `` | Per-bin floor multiplier | `src/pvx/cli/pvxdeverb.py` |
| `--strength` | False | `0.45` | `` | `` | Tail suppression strength 0..1 | `src/pvx/cli/pvxdeverb.py` |

## `pvxformant.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--formant-lifter` | False | `32` | `` | `` |  | `src/pvx/cli/pvxformant.py` |
| `--formant-max-gain-db` | False | `12.0` | `` | `` |  | `src/pvx/cli/pvxformant.py` |
| `--formant-shift-ratio` | False | `1.0` | `` | `` | Formant ratio (>1 up, <1 down) | `src/pvx/cli/pvxformant.py` |
| `--mode` | False | `shift` | `shift, preserve` | `` |  | `src/pvx/cli/pvxformant.py` |
| `--pitch-shift-cents` | False | `0.0` | `` | `` | Additional microtonal pitch shift in cents before formant stage | `src/pvx/cli/pvxformant.py` |
| `--pitch-shift-semitones` | False | `0.0` | `` | `` | Optional pitch shift before formant stage | `src/pvx/cli/pvxformant.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxformant.py` |

## `pvxfreeze.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--duration` | False | `3.0` | `` | `` | Output freeze duration in seconds | `src/pvx/cli/pvxfreeze.py` |
| `--freeze-time` | False | `0.2` | `` | `` | Freeze anchor time in seconds | `src/pvx/cli/pvxfreeze.py` |
| `--random-phase` | False | `` | `` | `store_true` | Add subtle phase randomization per frame | `src/pvx/cli/pvxfreeze.py` |

## `pvxharmonize.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--force-stereo` | False | `` | `` | `store_true` | Mix result as stereo with panning | `src/pvx/cli/pvxharmonize.py` |
| `--gains` | False | `` | `` | `` | Optional comma-separated linear gain per voice | `src/pvx/cli/pvxharmonize.py` |
| `--intervals` | False | `0,4,7` | `` | `` | Comma-separated semitone intervals per voice (supports fractional values) | `src/pvx/cli/pvxharmonize.py` |
| `--intervals-cents` | False | `` | `` | `` | Optional cents offsets per voice, added to --intervals (e.g. 0,14,-12) | `src/pvx/cli/pvxharmonize.py` |
| `--pans` | False | `` | `` | `` | Optional comma-separated pan per voice [-1..1] | `src/pvx/cli/pvxharmonize.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxharmonize.py` |

## `pvxlayer.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--harmonic-gain` | False | `1.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--harmonic-kernel` | False | `31` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--harmonic-pitch-cents` | False | `0.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--harmonic-pitch-semitones` | False | `0.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--harmonic-stretch` | False | `1.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--percussive-gain` | False | `1.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--percussive-kernel` | False | `31` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--percussive-pitch-cents` | False | `0.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--percussive-pitch-semitones` | False | `0.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--percussive-stretch` | False | `1.0` | `` | `` |  | `src/pvx/cli/pvxlayer.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxlayer.py` |

## `pvxmorph.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--alpha` | False | `0.5` | `` | `` | Morph amount 0..1 (0=A, 1=B) | `src/pvx/cli/pvxmorph.py` |
| `--output` | False | `` | `` | `` | Output file path | `src/pvx/cli/pvxmorph.py` |
| `--output-format` | False | `` | `` | `` | Output extension/format; for --stdout defaults to wav | `src/pvx/cli/pvxmorph.py` |
| `--overwrite` | False | `` | `` | `store_true` |  | `src/pvx/cli/pvxmorph.py` |
| `--stdout` | False | `` | `` | `store_true` | Write processed audio to stdout stream (for piping); equivalent to -o - | `src/pvx/cli/pvxmorph.py` |
| `--subtype` | False | `` | `` | `` |  | `src/pvx/cli/pvxmorph.py` |

## `pvxretune.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--chunk-ms` | False | `80.0` | `` | `` | Analysis/process chunk duration in ms | `src/pvx/cli/pvxretune.py` |
| `--f0-max` | False | `1200.0` | `` | `` |  | `src/pvx/cli/pvxretune.py` |
| `--f0-min` | False | `60.0` | `` | `` |  | `src/pvx/cli/pvxretune.py` |
| `--overlap-ms` | False | `20.0` | `` | `` | Chunk overlap in ms | `src/pvx/cli/pvxretune.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxretune.py` |
| `--root` | False | `C` | `` | `` | Scale root note (C,C#,D,...,B) | `src/pvx/cli/pvxretune.py` |
| `--scale` | False | `chromatic` | `` | `` | Named scale for 12-TET quantization | `src/pvx/cli/pvxretune.py` |
| `--scale-cents` | False | `` | `` | `` | Optional comma-separated microtonal scale degrees in cents within one octave, relative to --root (example: 0,90,204,294,408,498,612,702,816,906,1020,1110) | `src/pvx/cli/pvxretune.py` |
| `--strength` | False | `0.85` | `` | `` | Correction strength 0..1 | `src/pvx/cli/pvxretune.py` |

## `pvxtransient.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--pitch-shift-cents` | False | `` | `` | `` | Optional microtonal pitch shift in cents (added to --pitch-shift-semitones) | `src/pvx/cli/pvxtransient.py` |
| `--pitch-shift-ratio` | False | `` | `` | `` |  | `src/pvx/cli/pvxtransient.py` |
| `--pitch-shift-semitones` | False | `0.0` | `` | `` |  | `src/pvx/cli/pvxtransient.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxtransient.py` |
| `--target-duration` | False | `` | `` | `` | Target duration in seconds | `src/pvx/cli/pvxtransient.py` |
| `--time-stretch` | False | `1.0` | `` | `` |  | `src/pvx/cli/pvxtransient.py` |
| `--transient-threshold` | False | `1.6` | `` | `` |  | `src/pvx/cli/pvxtransient.py` |

## `pvxunison.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--detune-cents` | False | `14.0` | `` | `` | Total detune span in cents | `src/pvx/cli/pvxunison.py` |
| `--dry-mix` | False | `0.2` | `` | `` | Dry signal mix amount | `src/pvx/cli/pvxunison.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxunison.py` |
| `--voices` | False | `5` | `` | `` | Number of unison voices | `src/pvx/cli/pvxunison.py` |
| `--width` | False | `1.0` | `` | `` | Stereo width multiplier 0..2 | `src/pvx/cli/pvxunison.py` |

## `pvxvoc.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--analysis-channel` | False | `mix` | `first, mix` | `` | Channel strategy for F0 estimation with --target-f0 (default: mix) | `src/pvx/core/voc.py` |
| `--clip` | False | `` | `` | `store_true` | Legacy alias: hard clip at +/-1.0 when set | `src/pvx/core/voc.py` |
| `--compander-attack-ms` | False | `8.0` | `` | `` | Compander attack time in ms | `src/pvx/core/voc.py` |
| `--compander-compress-ratio` | False | `3.0` | `` | `` | Compander compression ratio (>=1) | `src/pvx/core/voc.py` |
| `--compander-expand-ratio` | False | `1.8` | `` | `` | Compander expansion ratio (>=1) | `src/pvx/core/voc.py` |
| `--compander-makeup-db` | False | `0.0` | `` | `` | Compander makeup gain in dB | `src/pvx/core/voc.py` |
| `--compander-release-ms` | False | `120.0` | `` | `` | Compander release time in ms | `src/pvx/core/voc.py` |
| `--compander-threshold-db` | False | `` | `` | `` | Enable compander threshold in dBFS | `src/pvx/core/voc.py` |
| `--compressor-attack-ms` | False | `10.0` | `` | `` | Compressor attack time in ms | `src/pvx/core/voc.py` |
| `--compressor-makeup-db` | False | `0.0` | `` | `` | Compressor makeup gain in dB | `src/pvx/core/voc.py` |
| `--compressor-ratio` | False | `4.0` | `` | `` | Compressor ratio (>=1) | `src/pvx/core/voc.py` |
| `--compressor-release-ms` | False | `120.0` | `` | `` | Compressor release time in ms | `src/pvx/core/voc.py` |
| `--compressor-threshold-db` | False | `` | `` | `` | Enable compressor above threshold dBFS | `src/pvx/core/voc.py` |
| `--cuda-device` | False | `0` | `` | `` | CUDA device index used when --device is auto/cuda (default: 0) | `src/pvx/core/voc.py` |
| `--device` | False | `auto` | `auto, cpu, cuda` | `` | Compute device: auto (prefer CUDA), cpu, or cuda | `src/pvx/core/voc.py` |
| `--dry-run` | False | `` | `` | `store_true` | Resolve settings without writing files | `src/pvx/core/voc.py` |
| `--expander-attack-ms` | False | `5.0` | `` | `` | Expander attack time in ms | `src/pvx/core/voc.py` |
| `--expander-ratio` | False | `2.0` | `` | `` | Expander ratio (>=1) | `src/pvx/core/voc.py` |
| `--expander-release-ms` | False | `120.0` | `` | `` | Expander release time in ms | `src/pvx/core/voc.py` |
| `--expander-threshold-db` | False | `` | `` | `` | Enable downward expander below threshold dBFS | `src/pvx/core/voc.py` |
| `--f0-max` | False | `1000.0` | `` | `` | Maximum F0 search bound in Hz (default: 1000) | `src/pvx/core/voc.py` |
| `--f0-min` | False | `50.0` | `` | `` | Minimum F0 search bound in Hz (default: 50) | `src/pvx/core/voc.py` |
| `--formant-lifter` | False | `32` | `` | `` | Cepstral lifter cutoff for formant envelope extraction (default: 32) | `src/pvx/core/voc.py` |
| `--formant-max-gain-db` | False | `12.0` | `` | `` | Max per-bin formant correction gain in dB (default: 12) | `src/pvx/core/voc.py` |
| `--formant-strength` | False | `1.0` | `` | `` | Formant correction blend 0..1 when pitch mode is formant-preserving (default: 1.0) | `src/pvx/core/voc.py` |
| `--fourier-sync` | False | `` | `` | `store_true` | Enable fundamental frame locking. Uses generic short-time Fourier transforms with per-frame FFT sizes locked to detected F0. | `src/pvx/core/voc.py` |
| `--fourier-sync-max-fft` | False | `8192` | `` | `` | Maximum frame FFT size for --fourier-sync (default: 8192) | `src/pvx/core/voc.py` |
| `--fourier-sync-min-fft` | False | `256` | `` | `` | Minimum frame FFT size for --fourier-sync (default: 256) | `src/pvx/core/voc.py` |
| `--fourier-sync-smooth` | False | `5` | `` | `` | Smoothing span (frames) for prescanned F0 track in --fourier-sync (default: 5) | `src/pvx/core/voc.py` |
| `--hard-clip-level` | False | `` | `` | `` | Hard clip level in linear full-scale | `src/pvx/core/voc.py` |
| `--hop-size` | False | `512` | `` | `` | Hop size in samples (default: 512) | `src/pvx/core/voc.py` |
| `--kaiser-beta` | False | `14.0` | `` | `` | Kaiser window beta parameter used when --window kaiser (default: 14.0) | `src/pvx/core/voc.py` |
| `--limiter-threshold` | False | `` | `` | `` | Peak limiter threshold in linear full-scale | `src/pvx/core/voc.py` |
| `--n-fft` | False | `2048` | `` | `` | FFT size (default: 2048) | `src/pvx/core/voc.py` |
| `--no-center` | False | `` | `` | `store_true` | Disable center padding in STFT/ISTFT | `src/pvx/core/voc.py` |
| `--no-progress` | False | `` | `` | `store_true` |  | `src/pvx/core/voc.py` |
| `--normalize` | False | `none` | `none, peak, rms` | `` | Output normalization mode | `src/pvx/core/voc.py` |
| `--output-dir` | False | `` | `` | `` | Directory for output files (default: same directory as each input) | `src/pvx/core/voc.py` |
| `--output-format` | False | `` | `` | `` | Output format/extension (e.g. wav, flac, aiff). Default: keep input extension. | `src/pvx/core/voc.py` |
| `--overwrite` | False | `` | `` | `store_true` | Overwrite existing outputs | `src/pvx/core/voc.py` |
| `--peak-dbfs` | False | `` | `` | `` | Target peak dBFS when --normalize peak | `src/pvx/core/voc.py` |
| `--phase-locking` | False | `identity` | `off, identity` | `` | Inter-bin phase locking mode for transient fidelity (default: identity) | `src/pvx/core/voc.py` |
| `--pitch-mode` | False | `standard` | `standard, formant-preserving` | `` | Pitch mode: standard shift or formant-preserving correction (default: standard) | `src/pvx/core/voc.py` |
| `--pitch-shift-cents` | False | `` | `` | `` | Pitch shift in cents (+1200 is one octave up) | `src/pvx/core/voc.py` |
| `--pitch-shift-ratio` | False | `` | `` | `` | Pitch ratio (>1 up, <1 down) | `src/pvx/core/voc.py` |
| `--pitch-shift-semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up) | `src/pvx/core/voc.py` |
| `--quiet` | False | `` | `` | `store_true` | Reduce output and hide status bars | `src/pvx/core/voc.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` | Resampling engine (auto=fft if scipy available, else linear) | `src/pvx/core/voc.py` |
| `--rms-dbfs` | False | `` | `` | `` | Target RMS dBFS when --normalize rms | `src/pvx/core/voc.py` |
| `--silent` | False | `` | `` | `store_true` | Suppress all console output | `src/pvx/core/voc.py` |
| `--soft-clip-drive` | False | `1.0` | `` | `` | Soft clip drive amount (>0) | `src/pvx/core/voc.py` |
| `--soft-clip-level` | False | `` | `` | `` | Soft clip output ceiling in linear full-scale | `src/pvx/core/voc.py` |
| `--soft-clip-type` | False | `tanh` | `tanh, arctan, cubic` | `` | Soft clip transfer type | `src/pvx/core/voc.py` |
| `--stdout` | False | `` | `` | `store_true` | Write processed audio to stdout stream (for piping); requires exactly one input | `src/pvx/core/voc.py` |
| `--subtype` | False | `` | `` | `` | Output file subtype for soundfile (e.g. PCM_16, PCM_24, FLOAT) | `src/pvx/core/voc.py` |
| `--suffix` | False | `_pv` | `` | `` | Suffix appended to output filename stem (default: _pv) | `src/pvx/core/voc.py` |
| `--target-duration` | False | `` | `` | `` | Absolute target duration in seconds (overrides --time-stretch) | `src/pvx/core/voc.py` |
| `--target-f0` | False | `` | `` | `` | Target fundamental frequency in Hz. Auto-estimates source F0 per file. | `src/pvx/core/voc.py` |
| `--target-lufs` | False | `` | `` | `` | Integrated loudness target in LUFS | `src/pvx/core/voc.py` |
| `--target-pitch-shift-semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up) | `src/pvx/core/voc.py` |
| `--target-sample-rate` | False | `` | `` | `` | Output sample rate in Hz (default: keep input rate) | `src/pvx/core/voc.py` |
| `--time-stretch` | False | `1.0` | `` | `` | Final duration multiplier (1.0=unchanged, 2.0=2x longer) | `src/pvx/core/voc.py` |
| `--transient-preserve` | False | `` | `` | `store_true` | Enable transient phase resets based on spectral flux | `src/pvx/core/voc.py` |
| `--transient-threshold` | False | `2.0` | `` | `` | Spectral-flux multiplier for transient detection (default: 2.0) | `src/pvx/core/voc.py` |
| `--verbose` | False | `0` | `` | `count` | Increase verbosity (repeat for extra detail) | `src/pvx/core/voc.py` |
| `--verbosity` | False | `normal` | `` | `` | Console verbosity level | `src/pvx/core/voc.py` |
| `--win-length` | False | `2048` | `` | `` | Window length in samples (default: 2048) | `src/pvx/core/voc.py` |
| `--window` | False | `hann` | `` | `` | Window type (default: hann) | `src/pvx/core/voc.py` |

## `pvxwarp.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--crossfade-ms` | False | `8.0` | `` | `` |  | `src/pvx/cli/pvxwarp.py` |
| `--map` | True | `` | `` | `` | CSV map with start_sec,end_sec,stretch | `src/pvx/cli/pvxwarp.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxwarp.py` |
