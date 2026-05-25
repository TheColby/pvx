<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# pvx Command-Line Interface (CLI) Flags Reference

_Generated from commit `35e9761` (commit date: 2026-05-25T08:14:42-04:00)._

This file enumerates long-form CLI flags discovered from argparse declarations in canonical pvx CLI sources.

Total tool+flag entries: **476**
Total unique long flags: **338**

## Unique Long Flags

`--a4-reference-hz`, `--alpha`, `--ambient-phase-mix`, `--ambient-preset`, `--amplitude`, `--analysis-channel`, `--append-manifest`, `--attack-sec`, `--attenuation`, `--audit-metrics`, `--auto-profile`, `--auto-profile-lookahead-seconds`, `--auto-segment-seconds`, `--auto-transform`, `--backend`, `--background-dir`, `--band`, `--band-gain-db`, `--band-width-bins`, `--batch-size`, `--bit-depth`, `--bits`, `--blend-mode`, `--boost-db`, `--budget-path`, `--cache-dir`, `--category`, `--center`, `--cents`, `--checkpoint-dir`, `--checkpoint-id`, `--chord`, `--chunk-ms`, `--chunk-seconds`, `--clip`, `--codec`, `--coherence-strength`, `--comp-ratio`, `--comp-threshold-db`, `--compander-attack-ms`, `--compander-compress-ratio`, `--compander-expand-ratio`, `--compander-makeup-db`, `--compander-release-ms`, `--compander-threshold-db`, `--compressor-attack-ms`, `--compressor-makeup-db`, `--compressor-ratio`, `--compressor-release-ms`, `--compressor-threshold-db`, `--confidence-floor`, `--context-ms`, `--control-stdin`, `--coord-system`, `--cpu`, `--crossfade-ms`, `--cuda-device`, `--cycles`, `--decay`, `--decay-sec`, `--dedupe-by`, `--depth`, `--detune-cents`, `--device`, `--disk-budget`, `--distance-law`, `--dither`, `--dither-seed`, `--drr`, `--dry`, `--dry-mix`, `--dry-run`, `--duration`, `--duty-cycle`, `--emit`, `--end`, `--engine`, `--envelope-lifter`, `--example`, `--exp-curve`, `--expand-ratio`, `--expander-attack-ms`, `--expander-ratio`, `--expander-release-ms`, `--expander-threshold-db`, `--explain-plan`, `--exponent`, `--extreme-stretch-threshold`, `--extreme-time-stretch`, `--f0-max`, `--f0-min`, `--factor`, `--fail-if-exceeds`, `--feature-set`, `--feedback`, `--fill`, `--floor`, `--fmax`, `--fmin`, `--force`, `--force-stereo`, `--formant-lifter`, `--formant-max-gain-db`, `--formant-shift-ratio`, `--formant-strength`, `--format`, `--fourier-sync`, `--fourier-sync-max-fft`, `--fourier-sync-min-fft`, `--fourier-sync-smooth`, `--frame-length`, `--freeze-time`, `--freq`, `--freq-mask`, `--frequency-hz`, `--gain`, `--gains`, `--gpu`, `--group-separator`, `--grouping`, `--guided`, `--hard-clip-level`, `--harmonic-gain`, `--harmonic-kernel`, `--harmonic-pitch-cents`, `--harmonic-pitch-semitones`, `--harmonic-stretch`, `--hop-length`, `--hop-size`, `--inharmonic-f0-hz`, `--inharmonic-mix`, `--inharmonicity`, `--input-format`, `--intent`, `--interp`, `--intervals`, `--intervals-cents`, `--ir`, `--ir-database`, `--ir-dir`, `--ir-file`, `--json`, `--kaiser-beta`, `--keep-intermediate`, `--key`, `--label-policy`, `--labels-csv`, `--limiter-threshold`, `--list-databases`, `--manifest-append`, `--manifest-csv`, `--manifest-json`, `--manifest-jsonl`, `--map`, `--mask-exponent`, `--material`, `--max`, `--max-length-s`, `--max-stage-stretch`, `--metadata-policy`, `--method`, `--mfcc-count`, `--min`, `--mix`, `--mode`, `--multires-ffts`, `--multires-fusion`, `--multires-weights`, `--n-fft`, `--no-audit-metrics`, `--no-center`, `--no-normalize-gains`, `--no-onset-realign`, `--no-progress`, `--no-trim`, `--noise-file`, `--noise-floor`, `--noise-seconds`, `--noise-type`, `--normalize`, `--normalize-energy`, `--normalize-gains`, `--num-freq-masks`, `--num-time-masks`, `--offset`, `--onset-credit-max`, `--onset-credit-pull`, `--onset-time-credit`, `--operation`, `--operator`, `--order`, `--out`, `--output`, `--output-csv`, `--output-dir`, `--output-format`, `--output-jsonl`, `--output-key`, `--output-subtype`, `--output-suffix`, `--overlap-ms`, `--overwrite`, `--pair-mode`, `--pans`, `--peak`, `--peak-count`, `--peak-dbfs`, `--percussive-gain`, `--percussive-kernel`, `--percussive-pitch-cents`, `--percussive-pitch-semitones`, `--percussive-stretch`, `--phase`, `--phase-engine`, `--phase-locking`, `--phase-mix`, `--phase-mode`, `--phase-rad`, `--phase-random-seed`, `--pipeline`, `--pipeline-config`, `--pitch`, `--pitch-conf-min`, `--pitch-follow-stdin`, `--pitch-lowconf-mode`, `--pitch-map`, `--pitch-map-crossfade-ms`, `--pitch-map-smooth-ms`, `--pitch-map-stdin`, `--pitch-mode`, `--pitch-shift-cents`, `--pitch-shift-ratio`, `--pitch-shift-semitones`, `--policy`, `--pre-delay-ms`, `--preset`, `--quality-profile`, `--quiet`, `--random-phase`, `--rate`, `--ratio`, `--ratio-max`, `--ratio-min`, `--ratio-reference`, `--recommend-root`, `--reduction-db`, `--ref-channel`, `--reference-hz`, `--release-sec`, `--requested-stretch`, `--resample-mode`, `--resonance-decay`, `--resonance-hz`, `--resonance-mix`, `--resonance-q`, `--response`, `--response-gain-db`, `--response-mix`, `--resume`, `--rms-dbfs`, `--root`, `--root-hz`, `--route`, `--rt60`, `--safety-margin`, `--sample-rate`, `--scale`, `--scale-cents`, `--seed`, `--semitones`, `--shift-bins`, `--silent`, `--sine-cycles`, `--sine-phase-rad`, `--smooth`, `--smooth-frames`, `--smoothing-bins`, `--snr`, `--soft-clip-drive`, `--soft-clip-level`, `--soft-clip-type`, `--speaker-angles`, `--split`, `--split-mode`, `--sr`, `--start`, `--stdout`, `--stereo-mode`, `--strength`, `--stretch`, `--stretch-from`, `--stretch-max`, `--stretch-min`, `--stretch-mode`, `--stretch-scale`, `--strict`, `--strict-manifest`, `--subtype`, `--suffix`, `--summary-json`, `--sustain`, `--target`, `--target-duration`, `--target-f0`, `--target-lufs`, `--target-max`, `--target-min`, `--target-pitch-shift-semitones`, `--target-sample-rate`, `--time-mask`, `--time-stretch`, `--time-stretch-factor`, `--tolerance-cents`, `--trajectory-shape`, `--transform`, `--transient-crossfade-ms`, `--transient-mode`, `--transient-preserve`, `--transient-protect-ms`, `--transient-sensitivity`, `--transient-threshold`, `--transpose-semitones`, `--true-peak-max-dbtp`, `--tv-interp`, `--tv-key`, `--tv-map`, `--tv-order`, `--variants-per-input`, `--verbose`, `--verbosity`, `--voices`, `--wave`, `--wet`, `--width`, `--win-length`, `--window`, `--work-dir`, `--workers`

## `hps-pitch-track`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--backend` | False | `auto` | `auto, pyin, acf` | `` | Pitch backend (default: auto -> pyin if available, else acf) | `src/pvx/cli/hps_pitch_track.py` |
| `--confidence-floor` | False | `0.0` | `` | `` | Set confidence below this floor to 0.0 (default: 0.0). | `src/pvx/cli/hps_pitch_track.py` |
| `--emit` | False | `pitch_map` | `` | `` | Output mode: pitch_map (default), stretch_map, or pitch_to_stretch. | `src/pvx/cli/hps_pitch_track.py` |
| `--feature-set` | False | `all` | `none, basic, advanced, all` | `` | Feature tracking preset emitted as extra CSV columns. none/basic/advanced/all (default: all). | `src/pvx/cli/hps_pitch_track.py` |
| `--fmax` | False | `1200.0` | `` | `` | Maximum F0 in Hz (default: 1200) | `src/pvx/cli/hps_pitch_track.py` |
| `--fmin` | False | `50.0` | `` | `` | Minimum F0 in Hz (default: 50) | `src/pvx/cli/hps_pitch_track.py` |
| `--frame-length` | False | `2048` | `` | `` | Frame length in samples (default: 2048) | `src/pvx/cli/hps_pitch_track.py` |
| `--hop-size` | False | `256` | `` | `` | Hop size in samples (default: 256) | `src/pvx/cli/hps_pitch_track.py` |
| `--mfcc-count` | False | `13` | `` | `` | Number of MFCC columns (mfcc_01..mfcc_N) when feature-set is advanced/all (default: 13). | `src/pvx/cli/hps_pitch_track.py` |
| `--output` | False | `` | `` | `` | Output CSV path (default: '-' for stdout) | `src/pvx/cli/hps_pitch_track.py` |
| `--ratio-max` | False | `4.0` | `` | `` | Upper clamp for emitted pitch_ratio (default: 4.0). | `src/pvx/cli/hps_pitch_track.py` |
| `--ratio-min` | False | `0.25` | `` | `` | Lower clamp for emitted pitch_ratio (default: 0.25). | `src/pvx/cli/hps_pitch_track.py` |
| `--ratio-reference` | False | `median` | `median, mean, first, hz` | `` | Reference for emitted pitch_ratio values (default: median voiced f0). | `src/pvx/cli/hps_pitch_track.py` |
| `--reference-hz` | False | `` | `` | `` | Reference frequency in Hz when --ratio-reference hz. | `src/pvx/cli/hps_pitch_track.py` |
| `--smooth-frames` | False | `5` | `` | `` | Smoothing window for pitch_ratio frames (default: 5). | `src/pvx/cli/hps_pitch_track.py` |
| `--stretch` | False | `1.0` | `` | `` | Emit constant stretch column value for --emit pitch_map (default: 1.0). | `src/pvx/cli/hps_pitch_track.py` |
| `--stretch-from` | False | `pitch_ratio` | `` | `` | Source signal used to derive stretch in stretch-oriented emit modes (default: pitch_ratio). | `src/pvx/cli/hps_pitch_track.py` |
| `--stretch-max` | False | `4.0` | `` | `` | Upper clamp for emitted stretch in stretch-oriented modes (default: 4.0). | `src/pvx/cli/hps_pitch_track.py` |
| `--stretch-min` | False | `0.25` | `` | `` | Lower clamp for emitted stretch in stretch-oriented modes (default: 0.25). | `src/pvx/cli/hps_pitch_track.py` |
| `--stretch-scale` | False | `1.0` | `` | `` | Scale factor for derived stretch tracks (default: 1.0). | `src/pvx/cli/hps_pitch_track.py` |

## `pvx`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--append-manifest` | False | `` | `` | `store_true` | Append/merge with existing manifest instead of replacing it | `src/pvx/cli/pvx_augment.py` |
| `--audit-metrics` | False | `True` | `` | `store_true` | Compute per-output audit metrics (default: on) | `src/pvx/cli/pvx_augment.py` |
| `--backend` | False | `auto` | `auto, pyin, acf` | `` | Pitch tracker backend | `src/pvx/cli/pvx_pipeline.py` |
| `--batch-size` | False | `16` | `` | `` | Number of files processed simultaneously on GPU (default: 16) | `src/pvx/cli/pvx_augment.py` |
| `--bit-depth` | False | `inherit` | `inherit, 16, 24, 32f` | `` | Bit-depth assumption when --subtype is not set (default: inherit from input subtype). | `src/pvx/cli/pvx_helpers.py` |
| `--budget-path` | False | `` | `` | `` | Path used to query free space when --disk-budget is omitted (default: current directory). | `src/pvx/cli/pvx_helpers.py` |
| `--chunk-seconds` | False | `0.25` | `` | `` | Chunk/segment duration for `--auto-segment-seconds` (default: 0.25) | `src/pvx/cli/pvx_pipeline.py` |
| `--confidence-floor` | False | `0.0` | `` | `` | Minimum tracker confidence | `src/pvx/cli/pvx_pipeline.py` |
| `--context-ms` | False | `` | `` | `` | Optional stateful context window in milliseconds (default: auto from window/hop) | `src/pvx/cli/pvx_pipeline.py` |
| `--crossfade-ms` | False | `0.0` | `` | `` | Crossfade used for segment assembly in milliseconds (default: 0.0) | `src/pvx/cli/pvx_pipeline.py` |
| `--dedupe-by` | False | `output_path` | `` | `` | Deduplication key (default: output_path) | `src/pvx/cli/pvx_augment.py` |
| `--device` | False | `auto` | `auto, cuda, mps, cpu` | `` | Compute device (default: auto — CUDA > MPS > CPU) | `src/pvx/cli/pvx_augment.py` |
| `--disk-budget` | False | `` | `` | `` | Total budget size (e.g., 500MB, 20GB, 2TiB). If omitted, use free space at --budget-path. | `src/pvx/cli/pvx_helpers.py` |
| `--dry-run` | False | `` | `` | `store_true` | Plan manifest and filenames without rendering audio | `src/pvx/cli/pvx_augment.py` |
| `--duration` | False | `0.3` | `` | `` | Synthetic input duration seconds (default: 0.30) | `src/pvx/cli/pvx_helpers.py` |
| `--emit` | False | `pitch_to_stretch` | `pitch_map, stretch_map, pitch_to_stretch` | `` | Control map emit mode for the guide track (default: pitch_to_stretch) | `src/pvx/cli/pvx_pipeline.py` |
| `--engine` | False | `auto` | `auto, pytorch, torchaudio, wavelet, pvx-cli` | `` | DSP engine for time-stretch and pitch-shift transforms. 'auto' prefers torchaudio > pytorch > pvx-cli. 'torchaudio' uses torchaudio.functional.phase_vocoder. 'pytorch' uses native PyTorch phase vocoder. 'wavelet' uses the experimental wavelet backend. 'pvx-cli' always uses subprocess. (default: auto) | `src/pvx/cli/pvx_augment.py` |
| `--example` | False | `` | `` | `` | Print follow example command(s) and exit. Use `--example` for basic or `--example all` for the full set. | `src/pvx/cli/pvx_pipeline.py` |
| `--fail-if-exceeds` | False | `` | `` | `store_true` | Return non-zero when --requested-stretch does not fit the usable budget. | `src/pvx/cli/pvx_helpers.py` |
| `--feature-set` | False | `all` | `none, basic, advanced, all` | `` | Feature columns emitted by pitch tracker (default: all) | `src/pvx/cli/pvx_pipeline.py` |
| `--fmax` | False | `1200.0` | `` | `` | Maximum tracked f0 in Hz | `src/pvx/cli/pvx_pipeline.py` |
| `--fmin` | False | `50.0` | `` | `` | Minimum tracked f0 in Hz | `src/pvx/cli/pvx_pipeline.py` |
| `--frame-length` | False | `2048` | `` | `` | Tracker frame length in samples | `src/pvx/cli/pvx_pipeline.py` |
| `--group-separator` | False | `__` | `` | `` | Separator used by stem-prefix grouping (default: '__') | `src/pvx/cli/pvx_augment.py` |
| `--grouping` | False | `stem-prefix` | `none, stem-prefix` | `` | Split-grouping strategy (default: stem-prefix). `stem-prefix` keeps variants from similarly named sources in one split. | `src/pvx/cli/pvx_augment.py` |
| `--hop-size` | False | `256` | `` | `` | Tracker hop size in samples | `src/pvx/cli/pvx_pipeline.py` |
| `--intent` | False | `asr_robust` | `asr_robust, mir_music, ssl_contrastive` | `` | Built-in augmentation intent profile when --pipeline-config is not given (default: asr_robust) | `src/pvx/cli/pvx_augment.py` |
| `--json` | False | `` | `` | `store_true` | Emit JSON output | `src/pvx/cli/pvx_augment.py` |
| `--keep-intermediate` | False | `` | `` | `store_true` | Keep intermediate stage files after successful completion | `src/pvx/cli/pvx_pipeline.py` |
| `--label-policy` | False | `allow_alter` | `allow_alter, preserve` | `` | Label perturbation policy (default: allow_alter) | `src/pvx/cli/pvx_augment.py` |
| `--labels-csv` | False | `` | `` | `` | Optional metadata CSV with path/label/speaker columns for balanced split modes | `src/pvx/cli/pvx_augment.py` |
| `--manifest-csv` | False | `` | `` | `` | Optional CSV manifest path (default: <output-dir>/augment_manifest.csv) | `src/pvx/cli/pvx_augment.py` |
| `--manifest-jsonl` | False | `` | `` | `` | Optional JSONL manifest path (default: <output-dir>/augment_manifest.jsonl) | `src/pvx/cli/pvx_augment.py` |
| `--material` | False | `mix` | `mix, speech, vocal, drums, ambient` | `` | Material profile for `pvx safe` command generation | `src/pvx/cli/pvx_helpers.py` |
| `--max-length-s` | False | `30.0` | `` | `` | Maximum file duration in seconds (default: 30.0) | `src/pvx/cli/pvx_augment.py` |
| `--mfcc-count` | False | `13` | `` | `` | MFCC column count emitted by pitch tracker (default: 13) | `src/pvx/cli/pvx_pipeline.py` |
| `--mode` | False | `stateful` | `stateful, wrapper` | `` | Stream engine: stateful chunk processor (default) or wrapper compatibility mode | `src/pvx/cli/pvx_pipeline.py` |
| `--no-audit-metrics` | False | `` | `` | `store_false` | Disable audit metric computation | `src/pvx/cli/pvx_augment.py` |
| `--no-progress` | False | `` | `` | `store_true` | Suppress progress messages | `src/pvx/cli/pvx_augment.py` |
| `--out` | True | `` | `` | `` | Output audio path | `src/pvx/cli/pvx_helpers.py` |
| `--output` | False | `output.wav` | `` | `` | Output audio path (default: output.wav) | `src/pvx/cli/pvx_helpers.py` |
| `--output-csv` | False | `` | `` | `` | Optional merged CSV output path | `src/pvx/cli/pvx_augment.py` |
| `--output-dir` | True | `` | `` | `` | Output directory for augmented files | `src/pvx/cli/pvx_augment.py` |
| `--output-format` | False | `wav` | `wav, flac, aiff, ogg, caf` | `` | Output format container (default: wav) | `src/pvx/cli/pvx_augment.py` |
| `--output-jsonl` | True | `` | `` | `` | Merged JSONL output path | `src/pvx/cli/pvx_augment.py` |
| `--output-subtype` | False | `PCM_16` | `PCM_16, PCM_24, PCM_32, FLOAT` | `` | Output sample format (default: PCM_16) | `src/pvx/cli/pvx_augment.py` |
| `--output-suffix` | False | `_aug` | `` | `` | Suffix appended before extension (default: _aug) | `src/pvx/cli/pvx_augment.py` |
| `--overwrite` | False | `` | `` | `store_true` | Overwrite existing outputs | `src/pvx/cli/pvx_augment.py` |
| `--pair-mode` | False | `off` | `off, contrastive2` | `` | Pair-view mode: off or two-view contrastive output (default: off) | `src/pvx/cli/pvx_augment.py` |
| `--pipeline` | True | `` | `` | `` | Pipeline string with stages separated by '|'. Example: "voc --stretch 1.2 | formant --mode preserve" | `src/pvx/cli/pvx_pipeline.py` |
| `--pipeline-config` | False | `` | `` | `` | Optional YAML/JSON pipeline manifest (see pvx.augment.config.load_pipeline) | `src/pvx/cli/pvx_augment.py` |
| `--pitch` | False | `0.0` | `` | `` | Smoke render pitch semitones (default: 0.0) | `src/pvx/cli/pvx_helpers.py` |
| `--pitch-conf-min` | False | `0.75` | `` | `` | Minimum accepted map confidence for pvx voc (default: 0.75) | `src/pvx/cli/pvx_pipeline.py` |
| `--pitch-lowconf-mode` | False | `hold` | `hold, unity, interp` | `` | Low-confidence handling mode in pvx voc (default: hold) | `src/pvx/cli/pvx_pipeline.py` |
| `--pitch-map-crossfade-ms` | False | `20.0` | `` | `` | Map segment crossfade in pvx voc (milliseconds, default: 20) | `src/pvx/cli/pvx_pipeline.py` |
| `--pitch-map-smooth-ms` | False | `0.0` | `` | `` | Additional map smoothing in pvx voc (milliseconds) | `src/pvx/cli/pvx_pipeline.py` |
| `--policy` | False | `` | `` | `` | Optional JSON policy file with augmentation defaults and bounds/choices overrides | `src/pvx/cli/pvx_augment.py` |
| `--quiet` | False | `` | `` | `store_true` | Reduce logs | `src/pvx/cli/pvx_augment.py` |
| `--ratio-max` | False | `4.0` | `` | `` | Maximum pitch_ratio clamp | `src/pvx/cli/pvx_pipeline.py` |
| `--ratio-min` | False | `0.25` | `` | `` | Minimum pitch_ratio clamp | `src/pvx/cli/pvx_pipeline.py` |
| `--ratio-reference` | False | `median` | `median, mean, first, hz` | `` | Reference mode for pitch_ratio derivation in tracking | `src/pvx/cli/pvx_pipeline.py` |
| `--reference-hz` | False | `` | `` | `` | Reference Hz when --ratio-reference hz | `src/pvx/cli/pvx_pipeline.py` |
| `--requested-stretch` | False | `` | `` | `` | Optional stretch ratio to evaluate against the computed budget. | `src/pvx/cli/pvx_helpers.py` |
| `--resume` | False | `` | `` | `store_true` | Skip already-rendered outputs found in existing manifest/output-dir | `src/pvx/cli/pvx_augment.py` |
| `--route` | False | `[]` | `` | `append` | Optional pvx voc control route expression. Repeat to chain. Example: --route stretch=pitch_ratio --route pitch_ratio=const(1.0) | `src/pvx/cli/pvx_pipeline.py` |
| `--safety-margin` | False | `0.9` | `` | `` | Usable fraction of budget in (0,1]; default: 0.90 (10%% headroom). | `src/pvx/cli/pvx_helpers.py` |
| `--sample-rate` | False | `24000` | `` | `` | Synthetic input sample rate (default: 24000) | `src/pvx/cli/pvx_helpers.py` |
| `--seed` | False | `1337` | `` | `` | Deterministic base seed (default: 1337) | `src/pvx/cli/pvx_augment.py` |
| `--silent` | False | `` | `` | `store_true` | Suppress logs | `src/pvx/cli/pvx_augment.py` |
| `--smooth-frames` | False | `5` | `` | `` | Smoothing window in frames | `src/pvx/cli/pvx_pipeline.py` |
| `--split` | False | `0.8,0.1,0.1` | `` | `` | train,val,test split ratios (default: 0.8,0.1,0.1) | `src/pvx/cli/pvx_augment.py` |
| `--split-mode` | False | `random` | `random, label_balanced, speaker_balanced` | `` | Split assignment mode (default: random) | `src/pvx/cli/pvx_augment.py` |
| `--sr` | False | `16000` | `` | `` | Target sample rate (default: 16000) | `src/pvx/cli/pvx_augment.py` |
| `--stretch` | False | `1.25` | `` | `` | Smoke render stretch factor (default: 1.25) | `src/pvx/cli/pvx_helpers.py` |
| `--stretch-from` | False | `pitch_ratio` | `pitch_ratio, inv_pitch_ratio, f0_hz` | `` | Source for deriving stretch in stretch-oriented emit modes | `src/pvx/cli/pvx_pipeline.py` |
| `--stretch-max` | False | `4.0` | `` | `` | Upper clamp for derived stretch | `src/pvx/cli/pvx_pipeline.py` |
| `--stretch-min` | False | `0.25` | `` | `` | Lower clamp for derived stretch | `src/pvx/cli/pvx_pipeline.py` |
| `--stretch-scale` | False | `1.0` | `` | `` | Scale factor for derived stretch track | `src/pvx/cli/pvx_pipeline.py` |
| `--strict` | False | `` | `` | `store_true` | Return non-zero on validation errors | `src/pvx/cli/pvx_augment.py` |
| `--strict-manifest` | False | `` | `` | `store_true` | Return non-zero if manifest validation errors are found | `src/pvx/cli/pvx_augment.py` |
| `--subtype` | False | `` | `` | `` | Explicit libsndfile subtype assumption (e.g., PCM_16, PCM_24, FLOAT). | `src/pvx/cli/pvx_helpers.py` |
| `--variants-per-input` | False | `3` | `` | `` | Number of augmented outputs per input file (default: 3) | `src/pvx/cli/pvx_augment.py` |
| `--work-dir` | False | `` | `` | `` | Optional directory for intermediate stage files | `src/pvx/cli/pvx_pipeline.py` |
| `--workers` | False | `1` | `` | `` | Parallel worker count for rendering (default: 1) | `src/pvx/cli/pvx_augment.py` |

## `pvxanalysis`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--analysis-channel` | False | `` | `` | `` | Channel index to analyze; -1 means all channels (default: -1) | `src/pvx/cli/pvxanalysis.py` |
| `--cuda-device` | False | `0` | `` | `` | CUDA device index for --device cuda/auto (default: 0) | `src/pvx/cli/pvxanalysis.py` |
| `--device` | False | `auto` | `auto, cpu, cuda` | `` | Compute device selection (default: auto) | `src/pvx/cli/pvxanalysis.py` |
| `--hop-size` | False | `512` | `` | `` | Hop size (default: 512) | `src/pvx/cli/pvxanalysis.py` |
| `--json` | False | `` | `` | `store_true` | Print summary as JSON | `src/pvx/cli/pvxanalysis.py` |
| `--kaiser-beta` | False | `14.0` | `` | `` | Kaiser beta when --window kaiser (default: 14.0) | `src/pvx/cli/pvxanalysis.py` |
| `--n-fft` | False | `2048` | `` | `` | STFT FFT size (default: 2048) | `src/pvx/cli/pvxanalysis.py` |
| `--no-center` | False | `` | `` | `store_true` | Disable centered framing | `src/pvx/cli/pvxanalysis.py` |
| `--output` | False | `` | `` | `` | Output PVXAN path (default: <input>.pvxan.npz) | `src/pvx/cli/pvxanalysis.py` |
| `--summary-json` | False | `` | `` | `` | Optional path to write summary JSON | `src/pvx/cli/pvxanalysis.py` |
| `--transform` | False | `fft` | `` | `` | Transform backend (default: fft) | `src/pvx/cli/pvxanalysis.py` |
| `--win-length` | False | `2048` | `` | `` | Window length (default: 2048) | `src/pvx/cli/pvxanalysis.py` |
| `--window` | False | `hann` | `` | `` | Window type | `src/pvx/cli/pvxanalysis.py` |

## `pvxcodec`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--bit-depth` | False | `24` | `16, 24, 32, float` | `` | Output file bit depth (default: 24) | `src/pvx/cli/pvxcodec.py` |
| `--bits` | False | `` | `` | `` | Bit depth for raw bit-crushing (4-16). Applied after codec if both specified. | `src/pvx/cli/pvxcodec.py` |
| `--codec` | False | `` | `mp3_low, mp3_medium, telephone, voip_narrow, voip_wide, am_radio, lo_fi, random` | `` | Codec preset to simulate (default: none — use --bits for raw bit-crush) | `src/pvx/cli/pvxcodec.py` |
| `--output` | True | `` | `` | `` | Output audio file | `src/pvx/cli/pvxcodec.py` |
| `--seed` | False | `0` | `` | `` | Random seed (default: 0) | `src/pvx/cli/pvxcodec.py` |

## `pvxconform`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--crossfade-ms` | False | `8.0` | `` | `` | Segment crossfade in milliseconds | `src/pvx/cli/pvxconform.py` |
| `--map` | True | `` | `` | `` | CSV map path | `src/pvx/cli/pvxconform.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxconform.py` |

## `pvxdenoise`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--floor` | False | `0.1` | `` | `` | Noise floor multiplier | `src/pvx/cli/pvxdenoise.py` |
| `--noise-file` | False | `` | `` | `` | Optional external noise reference | `src/pvx/cli/pvxdenoise.py` |
| `--noise-seconds` | False | `0.35` | `` | `` | Noise profile duration from start | `src/pvx/cli/pvxdenoise.py` |
| `--reduction-db` | False | `12.0` | `` | `` | Reduction strength in dB | `src/pvx/cli/pvxdenoise.py` |
| `--smooth` | False | `5` | `` | `` | Temporal smoothing frames | `src/pvx/cli/pvxdenoise.py` |

## `pvxdeverb`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--decay` | False | `0.92` | `` | `` | Tail memory decay 0..1 | `src/pvx/cli/pvxdeverb.py` |
| `--floor` | False | `0.12` | `` | `` | Per-bin floor multiplier | `src/pvx/cli/pvxdeverb.py` |
| `--strength` | False | `0.45` | `` | `` | Tail suppression strength 0..1 | `src/pvx/cli/pvxdeverb.py` |

## `pvxenvelope`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--amplitude` | False | `1.0` | `` | `` | Peak/depth/amplitude value (mode-dependent; periodic modes use this as waveform amplitude) | `src/pvx/cli/pvxenvelope.py` |
| `--attack-sec` | False | `0.1` | `` | `` | ADSR attack in seconds | `src/pvx/cli/pvxenvelope.py` |
| `--center` | False | `0.0` | `` | `` | Start/center value (mode-dependent; for periodic modes this is the center/DC offset) | `src/pvx/cli/pvxenvelope.py` |
| `--cycles` | False | `` | `` | `` | Waveform cycles across full duration (periodic modes) | `src/pvx/cli/pvxenvelope.py` |
| `--decay-sec` | False | `0.2` | `` | `` | ADSR decay in seconds | `src/pvx/cli/pvxenvelope.py` |
| `--duration` | True | `` | `` | `` | Envelope duration in seconds | `src/pvx/cli/pvxenvelope.py` |
| `--duty-cycle` | False | `0.5` | `` | `` | Duty cycle in (0,1) for square mode (default: 0.5) | `src/pvx/cli/pvxenvelope.py` |
| `--end` | False | `0.0` | `` | `` | End value for ADSR/ramp/exp modes | `src/pvx/cli/pvxenvelope.py` |
| `--exp-curve` | False | `4.0` | `` | `` | Exponential curve steepness for exp mode | `src/pvx/cli/pvxenvelope.py` |
| `--format` | False | `auto` | `csv, json, auto` | `` | Output format (default: auto from extension or csv for stdout) | `src/pvx/cli/pvxenvelope.py` |
| `--freq` | False | `` | `` | `` | Waveform frequency in Hz for periodic modes (converted to cycles = frequency * duration) | `src/pvx/cli/pvxenvelope.py` |
| `--frequency-hz` | False | `` | `` | `` | Waveform frequency in Hz for periodic modes (converted to cycles = frequency * duration) | `src/pvx/cli/pvxenvelope.py` |
| `--key` | False | `value` | `` | `` | Output control column/key name (default: value) | `src/pvx/cli/pvxenvelope.py` |
| `--max` | False | `` | `` | `` | Optional value clamp maximum | `src/pvx/cli/pvxenvelope.py` |
| `--min` | False | `` | `` | `` | Optional value clamp minimum | `src/pvx/cli/pvxenvelope.py` |
| `--mode` | False | `adsr` | `` | `` | Envelope/LFO waveform mode | `src/pvx/cli/pvxenvelope.py` |
| `--out` | False | `` | `` | `` | Output path. Default: stdout. | `src/pvx/cli/pvxenvelope.py` |
| `--output` | False | `` | `` | `` | Output path. Default: stdout. | `src/pvx/cli/pvxenvelope.py` |
| `--peak` | False | `1.0` | `` | `` | Peak/depth/amplitude value (mode-dependent; periodic modes use this as waveform amplitude) | `src/pvx/cli/pvxenvelope.py` |
| `--phase` | False | `0.0` | `` | `` | Initial waveform phase in radians for periodic modes | `src/pvx/cli/pvxenvelope.py` |
| `--phase-rad` | False | `0.0` | `` | `` | Initial waveform phase in radians for periodic modes | `src/pvx/cli/pvxenvelope.py` |
| `--rate` | False | `20.0` | `` | `` | Control points per second (default: 20) | `src/pvx/cli/pvxenvelope.py` |
| `--release-sec` | False | `0.2` | `` | `` | ADSR release in seconds | `src/pvx/cli/pvxenvelope.py` |
| `--sine-cycles` | False | `` | `` | `` | Waveform cycles across full duration (periodic modes) | `src/pvx/cli/pvxenvelope.py` |
| `--sine-phase-rad` | False | `0.0` | `` | `` | Initial waveform phase in radians for periodic modes | `src/pvx/cli/pvxenvelope.py` |
| `--start` | False | `0.0` | `` | `` | Start/center value (mode-dependent; for periodic modes this is the center/DC offset) | `src/pvx/cli/pvxenvelope.py` |
| `--stdout` | False | `` | `` | `store_true` | Write map to stdout | `src/pvx/cli/pvxenvelope.py` |
| `--sustain` | False | `0.7` | `` | `` | Sustain value for ADSR mode | `src/pvx/cli/pvxenvelope.py` |
| `--wave` | False | `adsr` | `` | `` | Envelope/LFO waveform mode | `src/pvx/cli/pvxenvelope.py` |

## `pvxfilter`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--band-gain-db` | False | `6.0` | `` | `` | Band boost amount for bandamp (dB) | `src/pvx/cli/pvxfilter.py` |
| `--band-width-bins` | False | `6` | `` | `` | Band width in bins for bandamp | `src/pvx/cli/pvxfilter.py` |
| `--comp-ratio` | False | `2.0` | `` | `` | Compression ratio for spec-compander | `src/pvx/cli/pvxfilter.py` |
| `--comp-threshold-db` | False | `` | `` | `` | Relative threshold for spec-compander | `src/pvx/cli/pvxfilter.py` |
| `--dry-mix` | False | `0.0` | `` | `` | Dry signal mix amount in [0,1] (default: 0.0) | `src/pvx/cli/pvxfilter.py` |
| `--expand-ratio` | False | `1.2` | `` | `` | Expansion ratio for spec-compander | `src/pvx/cli/pvxfilter.py` |
| `--noise-floor` | False | `1.0` | `` | `` | Threshold scale for noisefilter | `src/pvx/cli/pvxfilter.py` |
| `--operator` | False | `` | `` | `` |  | `src/pvx/cli/pvxfilter.py` |
| `--peak-count` | False | `8` | `` | `` | Number of response peaks for bandamp | `src/pvx/cli/pvxfilter.py` |
| `--response` | True | `` | `` | `` | Input PVXRF response artifact | `src/pvx/cli/pvxfilter.py` |
| `--response-gain-db` | False | `0.0` | `` | `` | Global response gain in dB (default: 0) | `src/pvx/cli/pvxfilter.py` |
| `--response-mix` | False | `1.0` | `` | `` | Wet response mix amount (default: 1.0) | `src/pvx/cli/pvxfilter.py` |
| `--shift-bins` | False | `0` | `` | `` | Shift response curve by FFT bins | `src/pvx/cli/pvxfilter.py` |
| `--transpose-semitones` | False | `0.0` | `` | `` | Transpose response curve in semitones | `src/pvx/cli/pvxfilter.py` |
| `--tv-interp` | False | `linear` | `` | `` | TV interpolation mode | `src/pvx/cli/pvxfilter.py` |
| `--tv-key` | False | `response_mix` | `` | `` | Column/key in --tv-map (default: response_mix) | `src/pvx/cli/pvxfilter.py` |
| `--tv-map` | False | `` | `` | `` | Optional CSV/JSON map for time-varying response_mix | `src/pvx/cli/pvxfilter.py` |
| `--tv-order` | False | `3` | `` | `` | Polynomial order for --tv-interp polynomial | `src/pvx/cli/pvxfilter.py` |

## `pvxformant`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--formant-lifter` | False | `32` | `` | `` |  | `src/pvx/cli/pvxformant.py` |
| `--formant-max-gain-db` | False | `12.0` | `` | `` |  | `src/pvx/cli/pvxformant.py` |
| `--formant-shift-ratio` | False | `1.0` | `` | `` | Formant ratio (>1 up, <1 down) | `src/pvx/cli/pvxformant.py` |
| `--mode` | False | `shift` | `shift, preserve` | `` |  | `src/pvx/cli/pvxformant.py` |
| `--pitch-shift-cents` | False | `0.0` | `` | `` | Additional microtonal pitch shift in cents before formant stage | `src/pvx/cli/pvxformant.py` |
| `--pitch-shift-semitones` | False | `0.0` | `` | `` | Optional pitch shift before formant stage | `src/pvx/cli/pvxformant.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxformant.py` |

## `pvxfreeze`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--duration` | False | `3.0` | `` | `` | Output freeze duration in seconds | `src/pvx/cli/pvxfreeze.py` |
| `--freeze-time` | False | `0.2` | `` | `` | Freeze anchor time in seconds | `src/pvx/cli/pvxfreeze.py` |
| `--phase-mode` | False | `instantaneous` | `instantaneous, bin, hold` | `` | Phase progression mode inside the frozen segment: instantaneous (default, least flutter), bin (bin-center), hold (no advance). | `src/pvx/cli/pvxfreeze.py` |
| `--random-phase` | False | `` | `` | `store_true` | Add subtle phase randomization per frame | `src/pvx/cli/pvxfreeze.py` |

## `pvxgain`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--bit-depth` | False | `24` | `16, 24, 32, float` | `` | Output bit depth (default: 24) | `src/pvx/cli/pvxgain.py` |
| `--gain` | False | `` | `` | `` | Gain in dB. Single value (fixed) or 'min,max' (random range). Mutually exclusive with --normalize. | `src/pvx/cli/pvxgain.py` |
| `--normalize` | False | `` | `peak, rms` | `` | Normalization mode. Mutually exclusive with --gain. | `src/pvx/cli/pvxgain.py` |
| `--output` | True | `` | `` | `` | Output audio file | `src/pvx/cli/pvxgain.py` |
| `--seed` | False | `0` | `` | `` | Random seed (default: 0) | `src/pvx/cli/pvxgain.py` |
| `--target` | False | `` | `` | `` | Target level in dBFS for normalization (default: -1.0) | `src/pvx/cli/pvxgain.py` |

## `pvxharmmap`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--attenuation` | False | `0.45` | `` | `` | Chordmapper out-of-chord attenuation in [0,1] | `src/pvx/cli/pvxharmmap.py` |
| `--boost-db` | False | `6.0` | `` | `` | Chordmapper in-chord boost in dB | `src/pvx/cli/pvxharmmap.py` |
| `--chord` | False | `major` | `` | `` | Chord quality for chordmapper | `src/pvx/cli/pvxharmmap.py` |
| `--dry-mix` | False | `0.0` | `` | `` | Dry signal mix in [0,1] | `src/pvx/cli/pvxharmmap.py` |
| `--inharmonic-f0-hz` | False | `220.0` | `` | `` | Reference f0 for inharmonator | `src/pvx/cli/pvxharmmap.py` |
| `--inharmonic-mix` | False | `1.0` | `` | `` | Inharmonator mix in [0,1] | `src/pvx/cli/pvxharmmap.py` |
| `--inharmonicity` | False | `0.0001` | `` | `` | Inharmonicity coefficient B | `src/pvx/cli/pvxharmmap.py` |
| `--operator` | False | `` | `` | `` |  | `src/pvx/cli/pvxharmmap.py` |
| `--root-hz` | False | `220.0` | `` | `` | Chord root frequency in Hz | `src/pvx/cli/pvxharmmap.py` |
| `--strength` | False | `0.75` | `` | `` | Chordmapper emphasis strength in [0,1] | `src/pvx/cli/pvxharmmap.py` |
| `--tolerance-cents` | False | `35.0` | `` | `` | Chordmapper tolerance in cents | `src/pvx/cli/pvxharmmap.py` |

## `pvxharmonize`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--force-stereo` | False | `` | `` | `store_true` | Mix result as stereo with panning | `src/pvx/cli/pvxharmonize.py` |
| `--gains` | False | `` | `` | `` | Optional comma-separated linear gain per voice | `src/pvx/cli/pvxharmonize.py` |
| `--intervals` | False | `0,4,7` | `` | `` | Comma-separated semitone intervals per voice (supports fractional values) | `src/pvx/cli/pvxharmonize.py` |
| `--intervals-cents` | False | `` | `` | `` | Optional cents offsets per voice, added to --intervals (e.g. 0,14,-12) | `src/pvx/cli/pvxharmonize.py` |
| `--pans` | False | `` | `` | `` | Optional comma-separated pan per voice [-1..1] | `src/pvx/cli/pvxharmonize.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxharmonize.py` |

## `pvxlayer`

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

## `pvxmorph`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--alpha` | False | `0.5` | `` | `` | Morph amount 0..1 (0=A, 1=B). Accepts scalar or control file (.csv/.json) for time-varying A->B trajectory morphing. | `src/pvx/cli/pvxmorph.py` |
| `--blend-mode` | False | `linear` | `` | `` | Cross-synthesis blend style. linear/geometric are symmetric blends; carrier_* modes transfer envelope/mask from modulator to carrier. | `src/pvx/cli/pvxmorph.py` |
| `--envelope-lifter` | False | `32` | `` | `` | Cepstral lifter cutoff for carrier_*_envelope_* modes (default: 32). | `src/pvx/cli/pvxmorph.py` |
| `--interp` | False | `linear` | `` | `` | Interpolation mode for --alpha/--phase-mix control files (default: linear). | `src/pvx/cli/pvxmorph.py` |
| `--mask-exponent` | False | `1.0` | `` | `` | Exponent used by carrier_*_mask_* modes (default: 1.0). | `src/pvx/cli/pvxmorph.py` |
| `--normalize-energy` | False | `` | `` | `store_true` | Normalize each output channel RMS toward alpha-blended input RMS. | `src/pvx/cli/pvxmorph.py` |
| `--order` | False | `3` | `` | `` | Polynomial order for --interp polynomial (default: 3). Accepts any integer >= 1; effective fit degree is min(order, control_points-1). | `src/pvx/cli/pvxmorph.py` |
| `--output` | False | `` | `` | `` | Output file path | `src/pvx/cli/pvxmorph.py` |
| `--output-format` | False | `` | `` | `` | Output extension/format; for --stdout defaults to wav | `src/pvx/cli/pvxmorph.py` |
| `--overwrite` | False | `` | `` | `store_true` |  | `src/pvx/cli/pvxmorph.py` |
| `--phase-mix` | False | `` | `` | `` | Phase blend in [0,1]. If omitted, mode-specific defaults apply (A-phase for *_phase_a/carrier_a_*, B-phase for *_phase_b/carrier_b_*, alpha for symmetric modes). Accepts scalar or control file (.csv/.json). | `src/pvx/cli/pvxmorph.py` |
| `--stdout` | False | `` | `` | `store_true` | Write processed audio to stdout stream (for piping); equivalent to -o - | `src/pvx/cli/pvxmorph.py` |

## `pvxnoise`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--background-dir` | False | `` | `` | `` | Directory of background audio files to mix instead of synthetic noise | `src/pvx/cli/pvxnoise.py` |
| `--band` | False | `300,4000` | `` | `` | Low,high frequency band in Hz for bandlimited noise (default: 300,4000) | `src/pvx/cli/pvxnoise.py` |
| `--bit-depth` | False | `24` | `16, 24, 32, float` | `` | Output bit depth (default: 24) | `src/pvx/cli/pvxnoise.py` |
| `--noise-type` | False | `white` | `white, pink, brown, gaussian, bandlimited` | `` | Noise type (default: white) | `src/pvx/cli/pvxnoise.py` |
| `--output` | True | `` | `` | `` | Output audio file | `src/pvx/cli/pvxnoise.py` |
| `--seed` | False | `0` | `` | `` | Random seed (default: 0) | `src/pvx/cli/pvxnoise.py` |
| `--snr` | False | `20` | `` | `` | Signal-to-noise ratio in dB. Single value for fixed SNR or 'min,max' for uniform sampling. (default: 20) | `src/pvx/cli/pvxnoise.py` |

## `pvxreshape`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--exponent` | False | `1.0` | `` | `` | Exponent for pow operation | `src/pvx/cli/pvxreshape.py` |
| `--factor` | False | `1.0` | `` | `` | Scale factor for scale/time-scale ops | `src/pvx/cli/pvxreshape.py` |
| `--format` | False | `auto` | `auto, csv, json` | `` | Output format (default: auto from --output suffix or csv for stdout) | `src/pvx/cli/pvxreshape.py` |
| `--input-format` | False | `auto` | `auto, csv, json` | `` | Input format override (default: auto by suffix or csv for stdin) | `src/pvx/cli/pvxreshape.py` |
| `--interp` | False | `linear` | `` | `` | Interpolation mode | `src/pvx/cli/pvxreshape.py` |
| `--key` | False | `value` | `` | `` | Control column/key to read (default: value) | `src/pvx/cli/pvxreshape.py` |
| `--max` | False | `` | `` | `` | Maximum clamp value | `src/pvx/cli/pvxreshape.py` |
| `--min` | False | `` | `` | `` | Minimum clamp value | `src/pvx/cli/pvxreshape.py` |
| `--offset` | False | `0.0` | `` | `` | Offset for offset/time-shift ops | `src/pvx/cli/pvxreshape.py` |
| `--operation` | True | `` | `` | `` | Reshape operation | `src/pvx/cli/pvxreshape.py` |
| `--order` | False | `3` | `` | `` | Polynomial order for --interp polynomial | `src/pvx/cli/pvxreshape.py` |
| `--out` | False | `` | `` | `` | Output file path. Default: stdout. | `src/pvx/cli/pvxreshape.py` |
| `--output` | False | `` | `` | `` | Output file path. Default: stdout. | `src/pvx/cli/pvxreshape.py` |
| `--output-key` | False | `` | `` | `` | Output control key. Default: same as --key | `src/pvx/cli/pvxreshape.py` |
| `--rate` | False | `20.0` | `` | `` | Resample control rate (Hz) for resample operation | `src/pvx/cli/pvxreshape.py` |
| `--stdout` | False | `` | `` | `store_true` | Write reshaped map to stdout | `src/pvx/cli/pvxreshape.py` |
| `--target-max` | False | `1.0` | `` | `` | Target maximum for normalize operation | `src/pvx/cli/pvxreshape.py` |
| `--target-min` | False | `0.0` | `` | `` | Target minimum for normalize operation | `src/pvx/cli/pvxreshape.py` |
| `--window` | False | `5` | `` | `` | Window size for smooth operation | `src/pvx/cli/pvxreshape.py` |

## `pvxresponse`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--json` | False | `` | `` | `store_true` | Print summary as JSON | `src/pvx/cli/pvxresponse.py` |
| `--method` | False | `median` | `` | `` | Magnitude aggregation across frames (default: median) | `src/pvx/cli/pvxresponse.py` |
| `--normalize` | False | `peak` | `` | `` | Magnitude normalization mode (default: peak) | `src/pvx/cli/pvxresponse.py` |
| `--output` | False | `` | `` | `` | Output PVXRF path (default: <analysis>.pvxrf.npz) | `src/pvx/cli/pvxresponse.py` |
| `--phase-mode` | False | `mean` | `` | `` | Phase aggregation strategy (default: mean) | `src/pvx/cli/pvxresponse.py` |
| `--smoothing-bins` | False | `1` | `` | `` | Moving-average smoothing width in bins (default: 1) | `src/pvx/cli/pvxresponse.py` |
| `--summary-json` | False | `` | `` | `` | Optional path to write summary JSON | `src/pvx/cli/pvxresponse.py` |

## `pvxretune`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--a4-reference-hz` | False | `440.0` | `` | `` | Concert-pitch reference for A4 in Hz (default: 440.0) | `src/pvx/cli/pvxretune.py` |
| `--chunk-ms` | False | `80.0` | `` | `` | Analysis/process chunk duration in ms | `src/pvx/cli/pvxretune.py` |
| `--f0-max` | False | `1200.0` | `` | `` |  | `src/pvx/cli/pvxretune.py` |
| `--f0-min` | False | `60.0` | `` | `` |  | `src/pvx/cli/pvxretune.py` |
| `--overlap-ms` | False | `20.0` | `` | `` | Chunk overlap in ms | `src/pvx/cli/pvxretune.py` |
| `--recommend-root` | False | `` | `` | `store_true` | Estimate and use a per-file root fundamental from tracked F0. | `src/pvx/cli/pvxretune.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxretune.py` |
| `--root` | False | `C` | `` | `` | Scale root note (C,C#,D,...,B) | `src/pvx/cli/pvxretune.py` |
| `--root-hz` | False | `` | `` | `` | Optional root fundamental in Hz (overrides --root when provided). | `src/pvx/cli/pvxretune.py` |
| `--scale` | False | `chromatic` | `` | `` | Named scale for 12-TET quantization | `src/pvx/cli/pvxretune.py` |
| `--scale-cents` | False | `` | `` | `` | Optional comma-separated microtonal scale degrees in cents within one octave, relative to --root (example: 0,90,204,294,408,498,612,702,816,906,1020,1110) | `src/pvx/cli/pvxretune.py` |
| `--strength` | False | `0.85` | `` | `` | Correction strength 0..1 | `src/pvx/cli/pvxretune.py` |

## `pvxring`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--depth` | False | `1.0` | `` | `` | Ring modulation depth in [0,1] | `src/pvx/cli/pvxring.py` |
| `--feedback` | False | `0.0` | `` | `` | Feedback amount in [0,1) | `src/pvx/cli/pvxring.py` |
| `--frequency-hz` | False | `40.0` | `` | `` | Carrier frequency in Hz | `src/pvx/cli/pvxring.py` |
| `--mix` | False | `1.0` | `` | `` | Ring wet mix in [0,1] | `src/pvx/cli/pvxring.py` |
| `--operator` | False | `` | `` | `` |  | `src/pvx/cli/pvxring.py` |
| `--resonance-decay` | False | `0.2` | `` | `` | Resonator feedback/decay in [0,1) | `src/pvx/cli/pvxring.py` |
| `--resonance-hz` | False | `1200.0` | `` | `` | Resonance center frequency (Hz) | `src/pvx/cli/pvxring.py` |
| `--resonance-mix` | False | `0.35` | `` | `` | Resonator wet mix in [0,1] | `src/pvx/cli/pvxring.py` |
| `--resonance-q` | False | `8.0` | `` | `` | Resonance quality factor Q | `src/pvx/cli/pvxring.py` |
| `--tv-interp` | False | `linear` | `` | `` | TV interpolation mode | `src/pvx/cli/pvxring.py` |
| `--tv-map` | False | `` | `` | `` | CSV/JSON time-varying map for ringtvfilter | `src/pvx/cli/pvxring.py` |
| `--tv-order` | False | `3` | `` | `` | Polynomial order for --tv-interp polynomial | `src/pvx/cli/pvxring.py` |

## `pvxrir`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--bit-depth` | False | `24` | `16, 24, 32, float` | `` | Output bit depth (default: 24) | `src/pvx/cli/pvxrir.py` |
| `--cache-dir` | False | `` | `` | `` | Cache directory for downloaded IR databases (default: ~/.pvx/ir_cache) | `src/pvx/cli/pvxrir.py` |
| `--category` | False | `` | `` | `` | Filter IRs by category when using --ir-database (e.g., hall, church, outdoor, room, large) | `src/pvx/cli/pvxrir.py` |
| `--drr` | False | `3,12` | `` | `` | Direct-to-reverb ratio in dB. Single value or 'min,max' range (default: 3,12) | `src/pvx/cli/pvxrir.py` |
| `--ir-database` | False | `` | `` | `` | Use IRs from a curated database (auto-downloads on first use). Available: echothief, mit_kemar | `src/pvx/cli/pvxrir.py` |
| `--ir-dir` | False | `` | `` | `` | Directory of impulse response files (one is chosen randomly) | `src/pvx/cli/pvxrir.py` |
| `--ir-file` | False | `` | `` | `` | Path to an impulse response WAV file (overrides synthetic generation) | `src/pvx/cli/pvxrir.py` |
| `--list-databases` | False | `False` | `` | `store_true` | List available IR databases and exit | `src/pvx/cli/pvxrir.py` |
| `--no-trim` | False | `` | `` | `store_true` | Do not trim output to input length (include reverb tail) | `src/pvx/cli/pvxrir.py` |
| `--output` | True | `` | `` | `` | Output audio file | `src/pvx/cli/pvxrir.py` |
| `--pre-delay-ms` | False | `5.0` | `` | `` | Pre-delay in ms before the reverberant tail (default: 5.0) | `src/pvx/cli/pvxrir.py` |
| `--rt60` | False | `0.2,1.5` | `` | `` | Reverberation time in seconds. Single value or 'min,max' range (default: 0.2,1.5) | `src/pvx/cli/pvxrir.py` |
| `--seed` | False | `0` | `` | `` | Random seed (default: 0) | `src/pvx/cli/pvxrir.py` |
| `--wet` | False | `0.3,0.7` | `` | `` | Wet/dry mix 0-1. Single value or 'min,max' range (default: 0.3,0.7) | `src/pvx/cli/pvxrir.py` |

## `pvxspecaugment`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--bit-depth` | False | `24` | `16, 24, 32, float` | `` | Output bit depth (default: 24) | `src/pvx/cli/pvxspecaugment.py` |
| `--fill` | False | `0` | `` | `` | Fill value for masked regions: 0, mean, or a float (default: 0) | `src/pvx/cli/pvxspecaugment.py` |
| `--freq-mask` | False | `27` | `` | `` | Maximum frequency mask width F in bins (default: 27) | `src/pvx/cli/pvxspecaugment.py` |
| `--hop-length` | False | `128` | `` | `` | STFT hop length in samples (default: 128) | `src/pvx/cli/pvxspecaugment.py` |
| `--n-fft` | False | `512` | `` | `` | FFT size (default: 512) | `src/pvx/cli/pvxspecaugment.py` |
| `--num-freq-masks` | False | `2` | `` | `` | Number of frequency masks (default: 2) | `src/pvx/cli/pvxspecaugment.py` |
| `--num-time-masks` | False | `2` | `` | `` | Number of time masks (default: 2) | `src/pvx/cli/pvxspecaugment.py` |
| `--output` | True | `` | `` | `` | Output audio file | `src/pvx/cli/pvxspecaugment.py` |
| `--seed` | False | `0` | `` | `` | Random seed (default: 0) | `src/pvx/cli/pvxspecaugment.py` |
| `--time-mask` | False | `100` | `` | `` | Maximum time mask width T in frames (default: 100) | `src/pvx/cli/pvxspecaugment.py` |

## `pvxtrajectoryreverb`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--coord-system` | False | `cartesian` | `cartesian, spherical` | `` | Coordinate format for --start/--end (default: cartesian) | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--distance-law` | False | `inverse` | `none, inverse, inverse-square` | `` | Distance gain law applied across trajectory (default: inverse) | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--dry` | False | `0.0` | `` | `` | Dry direct mix scalar (default: 0.0) | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--end` | True | `` | `` | `` | End position. Cartesian: x,y,z. Spherical: az_deg,el_deg,r. | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--ir` | True | `` | `` | `` | Multichannel impulse response path (for example 4-channel room capture). | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--no-normalize-gains` | False | `` | `` | `store_false` | Disable per-sample gain normalization. | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--normalize-gains` | False | `True` | `` | `store_true` | Normalize per-sample channel gains before distance weighting (default: on) | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--speaker-angles` | False | `` | `` | `` | Optional speaker layout azimuth/elevation list in degrees as "az,el;az,el;...". Count must match impulse-response channels. | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--start` | True | `` | `` | `` | Start position. Cartesian: x,y,z. Spherical: az_deg,el_deg,r. | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--trajectory-shape` | False | `linear` | `linear, ease-in, ease-out, ease-in-out` | `` | Interpolation shape from start to end (default: linear) | `src/pvx/cli/pvxtrajectoryreverb.py` |
| `--wet` | False | `1.0` | `` | `` | Wet reverb mix scalar (default: 1.0) | `src/pvx/cli/pvxtrajectoryreverb.py` |

## `pvxtransient`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--pitch-shift-cents` | False | `` | `` | `` | Optional microtonal pitch shift in cents (added to --pitch-shift-semitones) | `src/pvx/cli/pvxtransient.py` |
| `--pitch-shift-ratio` | False | `` | `` | `` | Pitch ratio override. Accepts decimals (1.5), integer ratios (3/2), and expressions (2^(1/12)). | `src/pvx/cli/pvxtransient.py` |
| `--pitch-shift-semitones` | False | `0.0` | `` | `` |  | `src/pvx/cli/pvxtransient.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxtransient.py` |
| `--target-duration` | False | `` | `` | `` | Target duration in seconds | `src/pvx/cli/pvxtransient.py` |
| `--time-stretch` | False | `1.0` | `` | `` |  | `src/pvx/cli/pvxtransient.py` |
| `--transient-threshold` | False | `1.6` | `` | `` |  | `src/pvx/cli/pvxtransient.py` |

## `pvxunison`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--detune-cents` | False | `14.0` | `` | `` | Total detune span in cents | `src/pvx/cli/pvxunison.py` |
| `--dry-mix` | False | `0.2` | `` | `` | Dry signal mix amount | `src/pvx/cli/pvxunison.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxunison.py` |
| `--voices` | False | `5` | `` | `` | Number of unison voices | `src/pvx/cli/pvxunison.py` |
| `--width` | False | `1.0` | `` | `` | Stereo width multiplier 0..2 | `src/pvx/cli/pvxunison.py` |

## `pvxvoc`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--ambient-phase-mix` | False | `0.5` | `` | `` | Random-phase blend when --phase-engine hybrid (0.0=propagated only, 1.0=random only; default: 0.5). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--ambient-preset` | False | `` | `` | `store_true` | Convenience preset for ambient extreme stretch (random phase engine, onset-time-credit, transient preserve, conservative staging). | `src/pvx/core/voc_parser.py` |
| `--analysis-channel` | False | `mix` | `first, mix` | `` | Channel strategy for F0 estimation with --target-f0 (default: mix) | `src/pvx/core/voc_parser.py` |
| `--auto-profile` | False | `` | `` | `store_true` | Analyze input and choose a profile automatically (speech/music/percussion/ambient/extreme). | `src/pvx/core/voc_parser.py` |
| `--auto-profile-lookahead-seconds` | False | `6.0` | `` | `` | Seconds of audio used when estimating --auto-profile (default: 6.0). | `src/pvx/core/voc_parser.py` |
| `--auto-segment-seconds` | False | `0.0` | `` | `` | Optional segment size in seconds for long jobs. When >0, processing runs per segment with crossfade assembly. | `src/pvx/core/voc_parser.py` |
| `--auto-transform` | False | `` | `` | `store_true` | Allow automatic transform selection when --transform is not explicitly set. | `src/pvx/core/voc_parser.py` |
| `--bit-depth` | False | `inherit` | `` | `` | Output bit-depth policy (default: inherit). Ignored when --subtype is set. | `src/pvx/core/voc_parser.py` |
| `--cents` | False | `` | `` | `` | Pitch shift in cents (+1200 is one octave up). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--checkpoint-dir` | False | `` | `` | `` | Directory used to cache per-segment checkpoint chunks for resume workflows. | `src/pvx/core/voc_parser.py` |
| `--checkpoint-id` | False | `` | `` | `` | Optional checkpoint run identifier (default: hash of input/settings). | `src/pvx/core/voc_parser.py` |
| `--clip` | False | `` | `` | `store_true` | Legacy alias: hard clip at +/-1.0 when set | `src/pvx/core/voc.py` |
| `--coherence-strength` | False | `0.0` | `` | `` | Coherence lock strength in [0,1] (0=off, 1=full lock). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
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
| `--control-stdin` | False | `` | `` | `store_true` | Alias for --pitch-map-stdin (canonical control-bus CSV stdin path). | `src/pvx/core/voc_parser.py` |
| `--cpu` | False | `` | `` | `store_true` | Alias for --device cpu. | `src/pvx/core/voc_parser.py` |
| `--cuda-device` | False | `0` | `` | `` | CUDA device index used when --device is auto/cuda (default: 0) | `src/pvx/core/voc.py` |
| `--device` | False | `auto` | `auto, cpu, cuda` | `` | Compute device: auto (prefer CUDA), cpu, or cuda | `src/pvx/core/voc.py` |
| `--dither` | False | `none` | `` | `` | Dither policy before quantized writes (default: none) | `src/pvx/core/voc_parser.py` |
| `--dither-seed` | False | `` | `` | `` | Deterministic RNG seed for dithering (default: random seed) | `src/pvx/core/voc_parser.py` |
| `--dry-run` | False | `` | `` | `store_true` | Resolve settings without writing files | `src/pvx/core/voc_parser.py` |
| `--example` | False | `` | `` | `` | Print copy-paste example command(s) and exit. | `src/pvx/core/voc_parser.py` |
| `--expander-attack-ms` | False | `5.0` | `` | `` | Expander attack time in ms | `src/pvx/core/voc.py` |
| `--expander-ratio` | False | `2.0` | `` | `` | Expander ratio (>=1) | `src/pvx/core/voc.py` |
| `--expander-release-ms` | False | `120.0` | `` | `` | Expander release time in ms | `src/pvx/core/voc.py` |
| `--expander-threshold-db` | False | `` | `` | `` | Enable downward expander below threshold dBFS | `src/pvx/core/voc.py` |
| `--explain-plan` | False | `` | `` | `store_true` | Print resolved processing plan JSON and exit without rendering audio. | `src/pvx/core/voc_parser.py` |
| `--extreme-stretch-threshold` | False | `2.0` | `` | `` | Auto-mode threshold for multistage activation (default: 2.0). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--extreme-time-stretch` | False | `` | `` | `store_true` | Force multistage strategy even when ratio is moderate. | `src/pvx/core/voc_parser.py` |
| `--f0-max` | False | `1000.0` | `` | `` | Maximum F0 search bound in Hz (default: 1000) | `src/pvx/core/voc_parser.py` |
| `--f0-min` | False | `50.0` | `` | `` | Minimum F0 search bound in Hz (default: 50) | `src/pvx/core/voc_parser.py` |
| `--force` | False | `` | `` | `store_true` | Overwrite existing outputs and other write targets | `src/pvx/core/voc_parser.py` |
| `--formant-lifter` | False | `32` | `` | `` | Cepstral lifter cutoff for formant envelope extraction (default: 32). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--formant-max-gain-db` | False | `12.0` | `` | `` | Max per-bin formant correction gain in dB (default: 12). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--formant-strength` | False | `1.0` | `` | `` | Formant correction blend 0..1 when pitch mode is formant-preserving (default: 1.0). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--fourier-sync` | False | `` | `` | `store_true` | Enable fundamental frame locking. Uses generic short-time Fourier transforms with per-frame FFT sizes locked to detected F0. | `src/pvx/core/voc_parser.py` |
| `--fourier-sync-max-fft` | False | `8192` | `` | `` | Maximum frame FFT size for --fourier-sync (default: 8192). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--fourier-sync-min-fft` | False | `256` | `` | `` | Minimum frame FFT size for --fourier-sync (default: 256). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--fourier-sync-smooth` | False | `5` | `` | `` | Smoothing span (frames) for prescanned F0 track in --fourier-sync (default: 5). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--gpu` | False | `` | `` | `store_true` | Alias for --device cuda. | `src/pvx/core/voc_parser.py` |
| `--guided` | False | `` | `` | `store_true` | Interactive guided mode for first-time users. | `src/pvx/core/voc_parser.py` |
| `--hard-clip-level` | False | `` | `` | `` | Hard clip level in linear full-scale | `src/pvx/core/voc.py` |
| `--hop-size` | False | `512` | `` | `` | Hop size in samples (default: 512). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--interp` | False | `linear` | `` | `` | Interpolation mode for time-varying control signals loaded from CSV/JSON (default: linear). | `src/pvx/core/voc_parser.py` |
| `--kaiser-beta` | False | `14.0` | `` | `` | Kaiser window beta parameter used when --window kaiser (default: 14.0). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--limiter-threshold` | False | `` | `` | `` | Peak limiter threshold in linear full-scale | `src/pvx/core/voc.py` |
| `--manifest-append` | False | `` | `` | `store_true` | Append entries to an existing --manifest-json file instead of replacing it. | `src/pvx/core/voc_parser.py` |
| `--manifest-json` | False | `` | `` | `` | Write processing manifest JSON with per-file settings and outcomes. | `src/pvx/core/voc_parser.py` |
| `--max-stage-stretch` | False | `1.8` | `` | `` | Maximum per-stage ratio used in multistage mode (default: 1.8). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--metadata-policy` | False | `none` | `` | `` | Output metadata policy: none, sidecar, or copy (sidecar implementation) | `src/pvx/core/voc_parser.py` |
| `--multires-ffts` | False | `1024,2048,4096` | `` | `` | Comma-separated FFT sizes for --multires-fusion (default: 1024,2048,4096) | `src/pvx/core/voc_parser.py` |
| `--multires-fusion` | False | `` | `` | `store_true` | Blend multiple FFT resolutions for each channel before pitch resampling. | `src/pvx/core/voc_parser.py` |
| `--multires-weights` | False | `` | `` | `` | Comma-separated fusion weights for --multires-fusion (defaults to equal weights). | `src/pvx/core/voc_parser.py` |
| `--n-fft` | False | `2048` | `` | `` | FFT size (default: 2048). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--no-center` | False | `` | `` | `store_true` | Disable center padding in STFT/ISTFT | `src/pvx/core/voc_parser.py` |
| `--no-onset-realign` | False | `` | `` | `store_true` | Disable fractional read-position realignment on onsets when --onset-time-credit is enabled. | `src/pvx/core/voc_parser.py` |
| `--no-progress` | False | `` | `` | `store_true` |  | `src/pvx/core/voc_console.py` |
| `--normalize` | False | `none` | `none, peak, rms` | `` | Output normalization mode | `src/pvx/core/voc.py` |
| `--onset-credit-max` | False | `8.0` | `` | `` | Maximum accumulated onset time credit in analysis-frame units (default: 8.0). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--onset-credit-pull` | False | `0.5` | `` | `` | Fraction of per-frame read advance removable while onset credit exists (0.0..1.0, default: 0.5). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--onset-time-credit` | False | `` | `` | `store_true` | Enable onset-triggered time-credit scheduling to reduce transient smear during extreme stretching. | `src/pvx/core/voc_parser.py` |
| `--order` | False | `3` | `` | `` | Polynomial order for --interp polynomial (default: 3). Accepts any integer >= 1; effective fit degree is min(order, control_points-1). | `src/pvx/core/voc_parser.py` |
| `--out` | False | `` | `` | `` | Explicit output file path (single-input mode only). Alias: --out | `src/pvx/core/voc_parser.py` |
| `--output` | False | `` | `` | `` | Explicit output file path (single-input mode only). Alias: --out | `src/pvx/core/voc_parser.py` |
| `--output-dir` | False | `` | `` | `` | Directory for output files (default: same directory as each input) | `src/pvx/core/voc_parser.py` |
| `--output-format` | False | `` | `` | `` | Output format/extension (e.g. wav, flac, aiff). Default: keep input extension. | `src/pvx/core/voc_parser.py` |
| `--overwrite` | False | `` | `` | `store_true` | Overwrite existing outputs and other write targets | `src/pvx/core/voc_parser.py` |
| `--peak-dbfs` | False | `` | `` | `` | Target peak dBFS when --normalize peak | `src/pvx/core/voc.py` |
| `--phase-engine` | False | `propagate` | `` | `` | Phase synthesis engine: propagate (classic phase vocoder), hybrid (propagated + stochastic blend), random (ambient stochastic phase). | `src/pvx/core/voc_parser.py` |
| `--phase-locking` | False | `identity` | `off, identity` | `` | Inter-bin phase locking mode for transient fidelity (default: identity) | `src/pvx/core/voc_parser.py` |
| `--phase-random-seed` | False | `` | `` | `` | Optional deterministic seed for random/hybrid phase generation. | `src/pvx/core/voc_parser.py` |
| `--pitch` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--pitch-conf-min` | False | `0.0` | `` | `` | Minimum accepted map confidence (default: 0 disables gating). | `src/pvx/core/voc_parser.py` |
| `--pitch-follow-stdin` | False | `` | `` | `store_true` | Shortcut for --pitch-map-stdin (sidechain pitch-follow workflows). | `src/pvx/core/voc_parser.py` |
| `--pitch-lowconf-mode` | False | `hold` | `hold, unity, interp` | `` | Low-confidence map handling mode (default: hold). | `src/pvx/core/voc_parser.py` |
| `--pitch-map` | False | `` | `` | `` | CSV control map for time-varying stretch/pitch. Columns: start_sec,end_sec plus optional stretch,pitch_ratio/pitch_cents/pitch_semitones,confidence. Use '-' to read from stdin. | `src/pvx/core/voc_parser.py` |
| `--pitch-map-crossfade-ms` | False | `8.0` | `` | `` | Crossfade between processed map segments in milliseconds (default: 8.0). | `src/pvx/core/voc_parser.py` |
| `--pitch-map-smooth-ms` | False | `0.0` | `` | `` | Moving-average smoothing over map pitch ratios in milliseconds. | `src/pvx/core/voc_parser.py` |
| `--pitch-map-stdin` | False | `` | `` | `store_true` | Read control-map CSV from stdin. | `src/pvx/core/voc_parser.py` |
| `--pitch-mode` | False | `standard` | `standard, formant-preserving` | `` | Pitch mode: standard shift or formant-preserving correction (default: standard) | `src/pvx/core/voc_parser.py` |
| `--pitch-shift-cents` | False | `` | `` | `` | Pitch shift in cents (+1200 is one octave up). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--pitch-shift-ratio` | False | `` | `` | `` | Pitch ratio (>1 up, <1 down). Accepts decimals (1.5), integer ratios (3/2), expressions (2^(1/12)), or a control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--pitch-shift-semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--preset` | False | `none` | `` | `` | High-level intent preset. Legacy: none/vocal/ambient/extreme. New: default/vocal_studio/drums_safe/extreme_ambient/stereo_coherent. | `src/pvx/core/voc_parser.py` |
| `--quality-profile` | False | `neutral` | `` | `` | Named tuning profile for vocoder defaults (default: neutral) | `src/pvx/core/voc_parser.py` |
| `--quiet` | False | `` | `` | `store_true` | Reduce output and hide status bars | `src/pvx/core/voc_console.py` |
| `--ratio` | False | `` | `` | `` | Pitch ratio (>1 up, <1 down). Accepts decimals (1.5), integer ratios (3/2), expressions (2^(1/12)), or a control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--ref-channel` | False | `0` | `` | `` | Reference channel index used by --stereo-mode ref_channel_lock (default: 0). | `src/pvx/core/voc_parser.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` | Resampling engine (auto=fft if scipy available, else linear) | `src/pvx/core/voc_parser.py` |
| `--resume` | False | `` | `` | `store_true` | Reuse existing checkpoint chunks from --checkpoint-dir when available. | `src/pvx/core/voc_parser.py` |
| `--rms-dbfs` | False | `` | `` | `` | Target RMS dBFS when --normalize rms | `src/pvx/core/voc.py` |
| `--route` | False | `[]` | `` | `append` | Control-bus routing expression for map rows. Repeat flag to chain routes. Syntax: target=source, target=const(v), target=inv(source), target=pow(source,exp), target=mul(source,factor), target=add(source,offset), target=affine(source,scale,bias), target=clip(source,lo,hi). Targets: stretch,pitch_ratio. Sources: any numeric column present in the control-map CSV. | `src/pvx/core/voc_parser.py` |
| `--semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--silent` | False | `` | `` | `store_true` | Suppress all console output | `src/pvx/core/voc_console.py` |
| `--soft-clip-drive` | False | `1.0` | `` | `` | Soft clip drive amount (>0) | `src/pvx/core/voc.py` |
| `--soft-clip-level` | False | `` | `` | `` | Soft clip output ceiling in linear full-scale | `src/pvx/core/voc.py` |
| `--soft-clip-type` | False | `tanh` | `tanh, arctan, cubic` | `` | Soft clip transfer type | `src/pvx/core/voc.py` |
| `--stdout` | False | `` | `` | `store_true` | Write processed audio to stdout stream (for piping); requires exactly one input | `src/pvx/core/voc_parser.py` |
| `--stereo-mode` | False | `independent` | `independent, mid_side_lock, ref_channel_lock` | `` | Channel coherence strategy: independent (legacy), mid_side_lock (M/S-coupled), ref_channel_lock (phase-lock to reference channel). | `src/pvx/core/voc_parser.py` |
| `--stretch` | False | `` | `` | `` | Alias for --time-stretch. Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--stretch-mode` | False | `auto` | `auto, standard, multistage` | `` | Stretch strategy: standard (single pass), multistage (chained moderate passes), or auto (multistage only for extreme ratios; default: auto). | `src/pvx/core/voc_parser.py` |
| `--subtype` | False | `` | `` | `` | Explicit libsndfile output subtype override (e.g., PCM_16, PCM_24, FLOAT) | `src/pvx/core/voc_parser.py` |
| `--suffix` | False | `_pv` | `` | `` | Suffix appended to output filename stem (default: _pv) | `src/pvx/core/voc_parser.py` |
| `--target-duration` | False | `` | `` | `` | Absolute target duration in seconds (overrides --time-stretch) | `src/pvx/core/voc_parser.py` |
| `--target-f0` | False | `` | `` | `` | Target fundamental frequency in Hz. Auto-estimates source F0 per file. | `src/pvx/core/voc_parser.py` |
| `--target-lufs` | False | `` | `` | `` | Integrated loudness target in LUFS | `src/pvx/core/voc.py` |
| `--target-pitch-shift-semitones` | False | `` | `` | `` | Pitch shift in semitones (+12 is one octave up). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--target-sample-rate` | False | `` | `` | `` | Output sample rate in Hz (default: keep input rate) | `src/pvx/core/voc_parser.py` |
| `--time-stretch` | False | `1.0` | `` | `` | Final duration multiplier (1.0=unchanged, 2.0=2x longer). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--time-stretch-factor` | False | `1.0` | `` | `` | Final duration multiplier (1.0=unchanged, 2.0=2x longer). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--transform` | False | `fft` | `` | `` | Per-frame transform backend for STFT/ISTFT paths (default: fft; options: fft, dft, czt, dct, dst, hartley) | `src/pvx/core/voc_parser.py` |
| `--transient-crossfade-ms` | False | `10.0` | `` | `` | Crossfade duration for transient/steady stitching (default: 10 ms). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--transient-mode` | False | `off` | `off, reset, hybrid, wsola` | `` | Transient handling mode: off (none), reset (phase reset), hybrid (PV steady + WSOLA transients), or wsola (time-domain transient-safe path). | `src/pvx/core/voc_parser.py` |
| `--transient-preserve` | False | `` | `` | `store_true` | Enable transient phase resets based on spectral flux | `src/pvx/core/voc_parser.py` |
| `--transient-protect-ms` | False | `30.0` | `` | `` | Transient protection width in milliseconds (default: 30). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--transient-sensitivity` | False | `0.5` | `` | `` | Transient detector sensitivity in [0,1] (higher catches more onsets). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--transient-threshold` | False | `2.0` | `` | `` | Spectral-flux multiplier for transient detection (default: 2.0). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--true-peak-max-dbtp` | False | `` | `` | `` | Apply output gain trim to enforce max true-peak in dBTP | `src/pvx/core/voc_parser.py` |
| `--verbose` | False | `0` | `` | `count` | Increase verbosity (repeat for extra detail) | `src/pvx/core/voc_console.py` |
| `--verbosity` | False | `normal` | `` | `` | Console verbosity level | `src/pvx/core/voc_console.py` |
| `--win-length` | False | `2048` | `` | `` | Window length in samples (default: 2048). Accepts scalar or control file (.csv/.json). | `src/pvx/core/voc_parser.py` |
| `--window` | False | `hann` | `` | `` | Window type (default: hann) | `src/pvx/core/voc_parser.py` |

## `pvxwarp`

| Flag | Required | Default | Choices | Action | Description | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `--crossfade-ms` | False | `8.0` | `` | `` |  | `src/pvx/cli/pvxwarp.py` |
| `--map` | True | `` | `` | `` | CSV map with start_sec,end_sec,stretch | `src/pvx/cli/pvxwarp.py` |
| `--resample-mode` | False | `auto` | `auto, fft, linear` | `` |  | `src/pvx/cli/pvxwarp.py` |

## Attribution

 See [`ATTRIBUTION.md`](../ATTRIBUTION.md).
