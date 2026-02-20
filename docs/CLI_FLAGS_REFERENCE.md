# pvx Command-Line Interface (CLI) Flags Reference

_Generated from commit `dcc235a` (commit date: 2026-02-20T11:48:06-05:00)._

This file enumerates long-form CLI flags discovered from argparse declarations in canonical pvx CLI sources.

Total tool+flag entries: **217**
Total unique long flags: **181**

## Unique Long Flags

`--alpha`, `--ambient-phase-mix`, `--ambient-preset`, `--analysis-channel`, `--auto-profile`, `--auto-profile-lookahead-seconds`, `--auto-segment-seconds`, `--auto-transform`, `--backend`, `--bit-depth`, `--cents`, `--checkpoint-dir`, `--checkpoint-id`, `--chunk-ms`, `--chunk-seconds`, `--clip`, `--coherence-strength`, `--compander-attack-ms`, `--compander-compress-ratio`, `--compander-expand-ratio`, `--compander-makeup-db`, `--compander-release-ms`, `--compander-threshold-db`, `--compressor-attack-ms`, `--compressor-makeup-db`, `--compressor-ratio`, `--compressor-release-ms`, `--compressor-threshold-db`, `--confidence-floor`, `--context-ms`, `--cpu`, `--crossfade-ms`, `--cuda-device`, `--decay`, `--detune-cents`, `--device`, `--dither`, `--dither-seed`, `--dry-mix`, `--dry-run`, `--duration`, `--example`, `--expander-attack-ms`, `--expander-ratio`, `--expander-release-ms`, `--expander-threshold-db`, `--explain-plan`, `--extreme-stretch-threshold`, `--extreme-time-stretch`, `--f0-max`, `--f0-min`, `--floor`, `--fmax`, `--fmin`, `--force-stereo`, `--formant-lifter`, `--formant-max-gain-db`, `--formant-shift-ratio`, `--formant-strength`, `--fourier-sync`, `--fourier-sync-max-fft`, `--fourier-sync-min-fft`, `--fourier-sync-smooth`, `--frame-length`, `--freeze-time`, `--gains`, `--gpu`, `--guided`, `--hard-clip-level`, `--harmonic-gain`, `--harmonic-kernel`, `--harmonic-pitch-cents`, `--harmonic-pitch-semitones`, `--harmonic-stretch`, `--hop-size`, `--intervals`, `--intervals-cents`, `--kaiser-beta`, `--keep-intermediate`, `--limiter-threshold`, `--manifest-append`, `--manifest-json`, `--map`, `--max-stage-stretch`, `--metadata-policy`, `--mode`, `--multires-ffts`, `--multires-fusion`, `--multires-weights`, `--n-fft`, `--no-center`, `--no-onset-realign`, `--no-progress`, `--noise-file`, `--noise-seconds`, `--normalize`, `--onset-credit-max`, `--onset-credit-pull`, `--onset-time-credit`, `--out`, `--output`, `--output-dir`, `--output-format`, `--overlap-ms`, `--overwrite`, `--pans`, `--peak-dbfs`, `--percussive-gain`, `--percussive-kernel`, `--percussive-pitch-cents`, `--percussive-pitch-semitones`, `--percussive-stretch`, `--phase-engine`, `--phase-locking`, `--phase-random-seed`, `--pipeline`, `--pitch`, `--pitch-conf-min`, `--pitch-follow-stdin`, `--pitch-lowconf-mode`, `--pitch-map`, `--pitch-map-crossfade-ms`, `--pitch-map-smooth-ms`, `--pitch-map-stdin`, `--pitch-mode`, `--pitch-shift-cents`, `--pitch-shift-ratio`, `--pitch-shift-semitones`, `--preset`, `--quality-profile`, `--quiet`, `--random-phase`, `--ratio`, `--ratio-max`, `--ratio-min`, `--ratio-reference`, `--reduction-db`, `--ref-channel`, `--reference-hz`, `--resample-mode`, `--resume`, `--rms-dbfs`, `--root`, `--scale`, `--scale-cents`, `--semitones`, `--silent`, `--smooth`, `--smooth-frames`, `--soft-clip-drive`, `--soft-clip-level`, `--soft-clip-type`, `--stdout`, `--stereo-mode`, `--strength`, `--stretch`, `--stretch-mode`, `--subtype`, `--suffix`, `--target-duration`, `--target-f0`, `--target-lufs`, `--target-pitch-shift-semitones`, `--target-sample-rate`, `--time-stretch`, `--time-stretch-factor`, `--transform`, `--transient-crossfade-ms`, `--transient-mode`, `--transient-preserve`, `--transient-protect-ms`, `--transient-sensitivity`, `--transient-threshold`, `--true-peak-max-dbtp`, `--verbose`, `--verbosity`, `--voices`, `--width`, `--win-length`, `--window`, `--work-dir`

## `hps_pitch_track.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--backend` | False | `auto` | `auto, pyin, acf` | `` | Pitch backend (default: auto -> pyin if available, else acf) | `src/pvx/cli/hps_pitch_track.py` |
| `--confidence-floor` | False | `0.0` | `` | `` | Set confidence below this floor to 0.0 (default: 0.0). | `src/pvx/cli/hps_pitch_track.py` |
| `--fmax` | False | `1200.0` | `` | `` | Maximum F0 in Hz (default: 1200) | `src/pvx/cli/hps_pitch_track.py` |
| `--fmin` | False | `50.0` | `` | `` | Minimum F0 in Hz (default: 50) | `src/pvx/cli/hps_pitch_track.py` |
| `--frame-length` | False | `2048` | `` | `` | Frame length in samples (default: 2048) | `src/pvx/cli/hps_pitch_track.py` |
| `--hop-size` | False | `256` | `` | `` | Hop size in samples (default: 256) | `src/pvx/cli/hps_pitch_track.py` |
| `--output` | False | `` | `` | `` | Output CSV path (default: '-' for stdout) | `src/pvx/cli/hps_pitch_track.py` |
| `--ratio-max` | False | `4.0` | `` | `` | Upper clamp for emitted pitch_ratio (default: 4.0). | `src/pvx/cli/hps_pitch_track.py` |
| `--ratio-min` | False | `0.25` | `` | `` | Lower clamp for emitted pitch_ratio (default: 0.25). | `src/pvx/cli/hps_pitch_track.py` |
| `--ratio-reference` | False | `median` | `median, mean, first, hz` | `` | Reference for emitted pitch_ratio values (default: median voiced f0). | `src/pvx/cli/hps_pitch_track.py` |
| `--reference-hz` | False | `` | `` | `` | Reference frequency in Hz when --ratio-reference hz. | `src/pvx/cli/hps_pitch_track.py` |
| `--smooth-frames` | False | `5` | `` | `` | Smoothing window for pitch_ratio frames (default: 5). | `src/pvx/cli/hps_pitch_track.py` |
| `--stretch` | False | `1.0` | `` | `` | Emit constant stretch column value (default: 1.0). | `src/pvx/cli/hps_pitch_track.py` |

## `pvx.py`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--chunk-seconds` | False | `0.25` | `` | `` | Chunk/segment duration for `--auto-segment-seconds` (default: 0.25) | `src/pvx/cli/pvx.py` |
| `--context-ms` | False | `` | `` | `` | Optional stateful context window in milliseconds (default: auto from window/hop) | `src/pvx/cli/pvx.py` |
| `--crossfade-ms` | False | `0.0` | `` | `` | Crossfade used for segment assembly in milliseconds (default: 0.0) | `src/pvx/cli/pvx.py` |
| `--example` | False | `` | `` | `store_true` | Print a copy-paste chain example and exit | `src/pvx/cli/pvx.py` |
| `--keep-intermediate` | False | `` | `` | `store_true` | Keep intermediate stage files after successful completion | `src/pvx/cli/pvx.py` |
| `--mode` | False | `stateful` | `stateful, wrapper` | `` | Stream engine: stateful chunk processor (default) or wrapper compatibility mode | `src/pvx/cli/pvx.py` |
| `--out` | True | `` | `` | `` | Final output path (or '-') | `src/pvx/cli/pvx.py` |
| `--output` | True | `` | `` | `` | Final output path (or '-') | `src/pvx/cli/pvx.py` |
| `--pipeline` | True | `` | `` | `` | Pipeline string with stages separated by '|'. Example: "voc --stretch 1.2 | formant --mode preserve" | `src/pvx/cli/pvx.py` |
| `--work-dir` | False | `` | `` | `` | Optional directory for intermediate stage files | `src/pvx/cli/pvx.py` |

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
| `--pitch-shift-ratio` | False | `` | `` | `` | Pitch ratio override. Accepts decimals (1.5), integer ratios (3/2), and expressions (2^(1/12)). | `src/pvx/cli/pvxtransient.py` |
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
| `--ambient-phase-mix` | False | `0.5` | `` | `` | Random-phase blend when --phase-engine hybrid (0.0=propagated only, 1.0=random only; default: 0.5). | `src/pvx/core/voc.py` |
| `--ambient-preset` | False | `` | `` | `store_true` | Convenience preset for ambient extreme stretch (random phase engine, onset-time-credit, transient preserve, conservative staging). | `src/pvx/core/voc.py` |
| `--analysis-channel` | False | `mix` | `first, mix` | `` | Channel strategy for F0 estimation with --target-f0 (default: mix) | `src/pvx/core/voc.py` |
| `--auto-profile` | False | `` | `` | `store_true` | Analyze input and choose a profile automatically (speech/music/percussion/ambient/extreme). | `src/pvx/core/voc.py` |
| `--auto-profile-lookahead-seconds` | False | `6.0` | `` | `` | Seconds of audio used when estimating --auto-profile (default: 6.0). | `src/pvx/core/voc.py` |
| `--auto-segment-seconds` | False | `0.0` | `` | `` | Optional segment size in seconds for long jobs. When >0, processing runs per segment with crossfade assembly. | `src/pvx/core/voc.py` |
| `--auto-transform` | False | `` | `` | `store_true` | Allow automatic transform selection when --transform is not explicitly set. | `src/pvx/core/voc.py` |
| `--bit-depth` | False | `inherit` | `` | `` | Output bit-depth policy (default: inherit). Ignored when --subtype is set. | `src/pvx/core/voc.py` |
| `--cents` | False | `` | `` | `` | Pitch shift in cents (+1200 is one octave up) | `src/pvx/core/voc.py` |
| `--checkpoint-dir` | False | `` | `` | `` | Directory used to cache per-segment checkpoint chunks for resume workflows. | `src/pvx/core/voc.py` |
| `--checkpoint-id` | False | `` | `` | `` | Optional checkpoint run identifier (default: hash of input/settings). | `src/pvx/core/voc.py` |
| `--clip` | False | `` | `` | `store_true` | Legacy alias: hard clip at +/-1.0 when set | `src/pvx/core/voc.py` |
| `--coherence-strength` | False | `0.0` | `` | `` | Coherence lock strength in [0,1] (0=off, 1=full lock). | `src/pvx/core/voc.py` |
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
| `--cpu` | False | `` | `` | `store_true` | Alias for --device cpu. | `src/pvx/core/voc.py` |
| `--cuda-device` | False | `0` | `` | `` | CUDA device index used when --device is auto/cuda (default: 0) | `src/pvx/core/voc.py` |
| `--device` | False | `auto` | `auto, cpu, cuda` | `` | Compute device: auto (prefer CUDA), cpu, or cuda | `src/pvx/core/voc.py` |
| `--dither` | False | `none` | `` | `` | Dither policy before quantized writes (default: none) | `src/pvx/core/voc.py` |
| `--dither-seed` | False | `` | `` | `` | Deterministic RNG seed for dithering (default: random seed) | `src/pvx/core/voc.py` |
| `--dry-run` | False | `` | `` | `store_true` | Resolve settings without writing files | `src/pvx/core/voc.py` |
| `--example` | False | `` | `` | `` | Print copy-paste example command(s) and exit. | `src/pvx/core/voc.py` |
| `--expander-attack-ms` | False | `5.0` | `` | `` | Expander attack time in ms | `src/pvx/core/voc.py` |
| `--expander-ratio` | False | `2.0` | `` | `` | Expander ratio (>=1) | `src/pvx/core/voc.py` |
| `--expander-release-ms` | False | `120.0` | `` | `` | Expander release time in ms | `src/pvx/core/voc.py` |
| `--expander-threshold-db` | False | `` | `` | `` | Enable downward expander below threshold dBFS | `src/pvx/core/voc.py` |
| `--explain-plan` | False | `` | `` | `store_true` | Print resolved processing plan JSON and exit without rendering audio. | `src/pvx/core/voc.py` |
| `--extreme-stretch-threshold` | False | `2.0` | `` | `` | Auto-mode threshold for multistage activation (default: 2.0). | `src/pvx/core/voc.py` |
| `--extreme-time-stretch` | False | `` | `` | `store_true` | Force multistage strategy even when ratio is moderate. | `src/pvx/core/voc.py` |
| `--f0-max` | False | `1000.0` | `` | `` | Maximum F0 search bound in Hz (default: 1000) | `src/pvx/core/voc.py` |
| `--f0-min` | False | `50.0` | `` | `` | Minimum F0 search bound in Hz (default: 50) | `src/pvx/core/voc.py` |
| `--formant-lifter` | False | `32` | `` | `` | Cepstral lifter cutoff for formant envelope extraction (default: 32) | `src/pvx/core/voc.py` |
| `--formant-max-gain-db` | False | `12.0` | `` | `` | Max per-bin formant correction gain in dB (default: 12) | `src/pvx/core/voc.py` |
| `--formant-strength` | False | `1.0` | `` | `` | Formant correction blend 0..1 when pitch mode is formant-preserving (default: 1.0) | `src/pvx/core/voc.py` |
| `--fourier-sync` | False | `` | `` | `store_true` | Enable fundamental frame locking. Uses generic short-time Fourier transforms with per-frame FFT sizes locked to detected F0. | `src/pvx/core/voc.py` |
| `--fourier-sync-max-fft` | False | `8192` | `` | `` | Maximum frame FFT size for --fourier-sync (default: 8192) | `src/pvx/core/voc.py` |
| `--fourier-sync-min-fft` | False | `256` | `` | `` | Minimum frame FFT size for --fourier-sync (default: 256) | `src/pvx/core/voc.py` |
| `--fourier-sync-smooth` | False | `5` | `` | `` | Smoothing span (frames) for prescanned F0 track in --fourier-sync (default: 5) | `src/pvx/core/voc.py` |
| `--gpu` | False | `` | `` | `store_true` | Alias for --device cuda. | `src/pvx/core/voc.py` |
| `--guided` | False | `` | `` | `store_true` | Interactive guided mode for first-time users. | `src/pvx/core/voc.py` |
| `--hard-clip-level` | False | `` | `` | `` | Hard clip level in linear full-scale | `src/pvx/core/voc.py` |
| `--hop-size` | False | `512` | `` | `` | Hop size in samples (default: 512) | `src/pvx/core/voc.py` |
| `--kaiser-beta` | False | `14.0` | `` | `` | Kaiser window beta parameter used when --window kaiser (default: 14.0) | `src/pvx/core/voc.py` |
| `--limiter-threshold` | False | `` | `` | `` | Peak limiter threshold in linear full-scale | `src/pvx/core/voc.py` |
| `--manifest-append` | False | `` | `` | `store_true` | Append entries to an existing --manifest-json file instead of replacing it. | `src/pvx/core/voc.py` |
| `--manifest-json` | False | `` | `` | `` | Write processing manifest JSON with per-file settings and outcomes. | `src/pvx/core/voc.py` |
| `--max-stage-stretch` | False | `1.8` | `` | `` | Maximum per-stage ratio used in multistage mode (default: 1.8). | `src/pvx/core/voc.py` |
| `--metadata-policy` | False | `none` | `` | `` | Output metadata policy: none, sidecar, or copy (sidecar implementation) | `src/pvx/core/voc.py` |
| `--multires-ffts` | False | `1024,2048,4096` | `` | `` | Comma-separated FFT sizes for --multires-fusion (default: 1024,2048,4096) | `src/pvx/core/voc.py` |
| `--multires-fusion` | False | `` | `` | `store_true` | Blend multiple FFT resolutions for each channel before pitch resampling. | `src/pvx/core/voc.py` |
| `--multires-weights` | False | `` | `` | `` | Comma-separated fusion weights for --multires-fusion (defaults to equal weights). | `src/pvx/core/voc.py` |
| `--n-fft` | False | `2048` | `` | `` | FFT size (default: 2048) | `src/pvx/core/voc.py` |
| `--no-center` | False | `` | `` | `store_true` | Disable center padding in STFT/ISTFT | `src/pvx/core/voc.py` |
| `--no-onset-realign` | False | `` | `` | `store_true` | Disable fractional read-position realignment on onsets when --onset-time-credit is enabled. | `src/pvx/core/voc.py` |
| `--no-progress` | False | `` | `` | `store_true` |  | `src/pvx/core/voc.py` |
| `--normalize` | False | `none` | `none, peak, rms` | `` | Output normalization mode | `src/pvx/core/voc.py` |
| `--onset-credit-max` | False | `8.0` | `` | `` | Maximum accumulated onset time credit in analysis-frame units (default: 8.0). | `src/pvx/core/voc.py` |
| `--onset-credit-pull` | False | `0.5` | `` | `` | Fraction of per-frame read advance removable while onset credit exists (0.0..1.0, default: 0.5). | `src/pvx/core/voc.py` |
| `--onset-time-credit` | False | `` | `` | `store_true` | Enable onset-triggered time-credit scheduling to reduce transient smear during extreme stretching. | `src/pvx/core/voc.py` |
| `--out` | False | `` | `` | `` | Explicit output file path (single-input mode only). Alias: --out | `src/pvx/core/voc.py` |
| `--output` | False | `` | `` | `` | Explicit output file path (single-input mode only). Alias: --out | `src/pvx/core/voc.py` |
| `--output-dir` | False | `` | `` | `` | Directory for output files (default: same directory as each input) | `src/pvx/core/voc.py` |
| `--output-format` | False | `` | `` | `` | Output format/extension (e.g. wav, flac, aiff). Default: keep input extension. | `src/pvx/core/voc.py` |
| `--overwrite` | False | `` | `` | `store_true` | Overwrite existing outputs | `src/pvx/core/voc.py` |
| `--peak-dbfs` | False | `` | `` | `` | Target peak dBFS when --normalize peak | `src/pvx/core/voc.py` |
| `--phase-engine` | False | `propagate` | `` | `` | Phase synthesis engine: propagate (classic phase vocoder), hybrid (propagated + stochastic blend), random (ambient stochastic phase). | `src/pvx/core/voc.py` |
| `--phase-locking` | False | `identity` | `off, identity` | `` | Inter-bin phase locking mode for transient fidelity (default: identity) | `src/pvx/core/voc.py` |
| `--phase-random-seed` | False | `` | `` | `` | Optional deterministic seed for random/hybrid phase generation. | `src/pvx/core/voc.py` |
| `--pitch` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up) | `src/pvx/core/voc.py` |
| `--pitch-conf-min` | False | `0.0` | `` | `` | Minimum accepted map confidence (default: 0 disables gating). | `src/pvx/core/voc.py` |
| `--pitch-follow-stdin` | False | `` | `` | `store_true` | Shortcut for --pitch-map-stdin (sidechain pitch-follow workflows). | `src/pvx/core/voc.py` |
| `--pitch-lowconf-mode` | False | `hold` | `hold, unity, interp` | `` | Low-confidence map handling mode (default: hold). | `src/pvx/core/voc.py` |
| `--pitch-map` | False | `` | `` | `` | CSV control map for time-varying stretch/pitch. Columns: start_sec,end_sec plus optional stretch,pitch_ratio/pitch_cents/pitch_semitones,confidence. Use '-' to read from stdin. | `src/pvx/core/voc.py` |
| `--pitch-map-crossfade-ms` | False | `8.0` | `` | `` | Crossfade between processed map segments in milliseconds (default: 8.0). | `src/pvx/core/voc.py` |
| `--pitch-map-smooth-ms` | False | `0.0` | `` | `` | Moving-average smoothing over map pitch ratios in milliseconds. | `src/pvx/core/voc.py` |
| `--pitch-map-stdin` | False | `` | `` | `store_true` | Read control-map CSV from stdin. | `src/pvx/core/voc.py` |
| `--pitch-mode` | False | `standard` | `standard, formant-preserving` | `` | Pitch mode: standard shift or formant-preserving correction (default: standard) | `src/pvx/core/voc.py` |
| `--pitch-shift-cents` | False | `` | `` | `` | Pitch shift in cents (+1200 is one octave up) | `src/pvx/core/voc.py` |
| `--pitch-shift-ratio` | False | `` | `` | `` | Pitch ratio (>1 up, <1 down). Accepts decimals (1.5), integer ratios (3/2), and expressions (2^(1/12)). | `src/pvx/core/voc.py` |
| `--pitch-shift-semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up) | `src/pvx/core/voc.py` |
| `--preset` | False | `none` | `` | `` | High-level intent preset. Legacy: none/vocal/ambient/extreme. New: default/vocal_studio/drums_safe/extreme_ambient/stereo_coherent. | `src/pvx/core/voc.py` |
| `--quality-profile` | False | `neutral` | `` | `` | Named tuning profile for vocoder defaults (default: neutral) | `src/pvx/core/voc.py` |
| `--quiet` | False | `` | `` | `store_true` | Reduce output and hide status bars | `src/pvx/core/voc.py` |
| `--ratio` | False | `` | `` | `` | Pitch ratio (>1 up, <1 down). Accepts decimals (1.5), integer ratios (3/2), and expressions (2^(1/12)). | `src/pvx/core/voc.py` |
| `--ref-channel` | False | `0` | `` | `` | Reference channel index used by --stereo-mode ref_channel_lock (default: 0). | `src/pvx/core/voc.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` | Resampling engine (auto=fft if scipy available, else linear) | `src/pvx/core/voc.py` |
| `--resume` | False | `` | `` | `store_true` | Reuse existing checkpoint chunks from --checkpoint-dir when available. | `src/pvx/core/voc.py` |
| `--rms-dbfs` | False | `` | `` | `` | Target RMS dBFS when --normalize rms | `src/pvx/core/voc.py` |
| `--semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up) | `src/pvx/core/voc.py` |
| `--silent` | False | `` | `` | `store_true` | Suppress all console output | `src/pvx/core/voc.py` |
| `--soft-clip-drive` | False | `1.0` | `` | `` | Soft clip drive amount (>0) | `src/pvx/core/voc.py` |
| `--soft-clip-level` | False | `` | `` | `` | Soft clip output ceiling in linear full-scale | `src/pvx/core/voc.py` |
| `--soft-clip-type` | False | `tanh` | `tanh, arctan, cubic` | `` | Soft clip transfer type | `src/pvx/core/voc.py` |
| `--stdout` | False | `` | `` | `store_true` | Write processed audio to stdout stream (for piping); requires exactly one input | `src/pvx/core/voc.py` |
| `--stereo-mode` | False | `independent` | `independent, mid_side_lock, ref_channel_lock` | `` | Channel coherence strategy: independent (legacy), mid_side_lock (M/S-coupled), ref_channel_lock (phase-lock to reference channel). | `src/pvx/core/voc.py` |
| `--stretch` | False | `` | `` | `` | Alias for --time-stretch. | `src/pvx/core/voc.py` |
| `--stretch-mode` | False | `auto` | `auto, standard, multistage` | `` | Stretch strategy: standard (single pass), multistage (chained moderate passes), or auto (multistage only for extreme ratios; default: auto). | `src/pvx/core/voc.py` |
| `--subtype` | False | `` | `` | `` | Explicit libsndfile output subtype override (e.g., PCM_16, PCM_24, FLOAT) | `src/pvx/core/voc.py` |
| `--suffix` | False | `_pv` | `` | `` | Suffix appended to output filename stem (default: _pv) | `src/pvx/core/voc.py` |
| `--target-duration` | False | `` | `` | `` | Absolute target duration in seconds (overrides --time-stretch) | `src/pvx/core/voc.py` |
| `--target-f0` | False | `` | `` | `` | Target fundamental frequency in Hz. Auto-estimates source F0 per file. | `src/pvx/core/voc.py` |
| `--target-lufs` | False | `` | `` | `` | Integrated loudness target in LUFS | `src/pvx/core/voc.py` |
| `--target-pitch-shift-semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up) | `src/pvx/core/voc.py` |
| `--target-sample-rate` | False | `` | `` | `` | Output sample rate in Hz (default: keep input rate) | `src/pvx/core/voc.py` |
| `--time-stretch` | False | `1.0` | `` | `` | Final duration multiplier (1.0=unchanged, 2.0=2x longer) | `src/pvx/core/voc.py` |
| `--time-stretch-factor` | False | `1.0` | `` | `` | Final duration multiplier (1.0=unchanged, 2.0=2x longer) | `src/pvx/core/voc.py` |
| `--transform` | False | `fft` | `` | `` | Per-frame transform backend for STFT/ISTFT paths (default: fft; options: fft, dft, czt, dct, dst, hartley) | `src/pvx/core/voc.py` |
| `--transient-crossfade-ms` | False | `10.0` | `` | `` | Crossfade duration for transient/steady stitching (default: 10 ms). | `src/pvx/core/voc.py` |
| `--transient-mode` | False | `off` | `off, reset, hybrid, wsola` | `` | Transient handling mode: off (none), reset (phase reset), hybrid (PV steady + WSOLA transients), or wsola (time-domain transient-safe path). | `src/pvx/core/voc.py` |
| `--transient-preserve` | False | `` | `` | `store_true` | Enable transient phase resets based on spectral flux | `src/pvx/core/voc.py` |
| `--transient-protect-ms` | False | `30.0` | `` | `` | Transient protection width in milliseconds (default: 30). | `src/pvx/core/voc.py` |
| `--transient-sensitivity` | False | `0.5` | `` | `` | Transient detector sensitivity in [0,1] (higher catches more onsets). | `src/pvx/core/voc.py` |
| `--transient-threshold` | False | `2.0` | `` | `` | Spectral-flux multiplier for transient detection (default: 2.0) | `src/pvx/core/voc.py` |
| `--true-peak-max-dbtp` | False | `` | `` | `` | Apply output gain trim to enforce max true-peak in dBTP | `src/pvx/core/voc.py` |
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
