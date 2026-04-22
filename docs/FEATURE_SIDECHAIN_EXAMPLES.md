<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# Feature-Driven Sidechain Examples




This guide shows copy-paste command-line interface (CLI) recipes where one file (the guide) controls another file (the target) using tracked audio features.

- Guide file: `guide.wav` (feature source).
- Target file: `target.wav` (sound being transformed).
- Output folder in examples: `out/`.
- Most recipes use `pvx follow` (shortest path) and `--route` formulas.

Create an output folder once:

```bash
mkdir -p out maps
```

Quick built-in recipe discovery:

```bash
pvx follow --example
pvx follow --example all
pvx follow --example mfcc_flux
```

## 1) Quick Start

Basic pitch-to-stretch sidechain:

```bash
pvx follow guide.wav target.wav --emit pitch_to_stretch --pitch-conf-min 0.75 --output out/follow_basic.wav
```

Direct pitch-follow (guide pitch contour drives target pitch contour):

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 --output out/follow_pitch.wav
```

Track first, render later (decoupled analysis/render):

```bash
pvx pitch-track guide.wav --feature-set all --mfcc-count 13 --output maps/guide_features.csv
pvx voc target.wav --pitch-map maps/guide_features.csv \
  --route stretch=pitch_ratio \
  --route pitch_ratio=const(1.0) \
  --output out/track_then_render.wav
```

## 2) Inspect Feature Columns Before Routing

Emit all tracked columns:

```bash
pvx pitch-track guide.wav --feature-set all --mfcc-count 20 --output maps/guide_all.csv
```

Print the CSV header as one column per line:

```bash
head -n 1 maps/guide_all.csv | tr ',' '\n'
```

Useful columns include:
- pitch/voicing: `f0_hz`, `pitch_ratio`, `confidence`, `voicing_prob`, `pitch_stability`
- spectral: `spectral_centroid_hz`, `spectral_flux`, `spectral_flatness`, `rolloff_hz`
- dynamics: `rms`, `rms_db`, `short_lufs_db`, `transientness`, `crest_factor_db`
- timbre vectors: `mfcc_01..mfcc_N`, `mpeg7_*`, `mpeg7_audio_spectrum_envelope_01..10`

## 3) Single-Feature Recipes

### 3.1 `f0_hz` -> pitch ratio

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(f0_hz,0.002272727,0.0) \
  --route pitch_ratio=clip(pitch_ratio,0.5,2.0) \
  --output out/f0_to_pitch.wav
```

### 3.2 `pitch_ratio` -> stretch (inverse motion)

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=inv(pitch_ratio) \
  --route stretch=clip(stretch,0.7,1.8) \
  --route pitch_ratio=const(1.0) \
  --output out/pitch_to_stretch_inverse.wav
```

### 3.3 `confidence` -> subtle pitch bend depth

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(confidence,0.25,0.9) \
  --route pitch_ratio=clip(pitch_ratio,0.85,1.15) \
  --output out/confidence_to_pitch.wav
```

### 3.4 `voicing_prob` -> stretch

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=affine(voicing_prob,0.6,0.8) \
  --route stretch=clip(stretch,0.8,1.4) \
  --route pitch_ratio=const(1.0) \
  --output out/voicing_to_stretch.wav
```

### 3.5 `rms_norm` -> stretch

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=affine(rms_norm,1.2,0.6) \
  --route stretch=clip(stretch,0.8,1.8) \
  --route pitch_ratio=const(1.0) \
  --output out/rms_to_stretch.wav
```

### 3.6 `spectral_flux` -> stretch

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=affine(spectral_flux,0.04,1.0) \
  --route stretch=clip(stretch,0.85,1.6) \
  --route pitch_ratio=const(1.0) \
  --output out/flux_to_stretch.wav
```

### 3.7 `onset_strength` -> attack-reactive stretch

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=affine(onset_strength,0.05,0.95) \
  --route stretch=clip(stretch,0.75,1.7) \
  --route pitch_ratio=const(1.0) \
  --output out/onset_to_stretch.wav
```

### 3.8 `transientness` -> transient-safe stretch variation

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=affine(transientness,-0.4,1.2) \
  --route stretch=clip(stretch,0.8,1.2) \
  --route pitch_ratio=const(1.0) \
  --output out/transientness_to_stretch.wav
```

### 3.9 `spectral_centroid_hz` -> brightness-linked pitch

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(spectral_centroid_hz,0.00025,0.7) \
  --route pitch_ratio=clip(pitch_ratio,0.7,1.6) \
  --output out/centroid_to_pitch.wav
```

### 3.10 `spectral_flatness` -> noisy/tonal pitch response

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(spectral_flatness,-0.5,1.25) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.25) \
  --output out/flatness_to_pitch.wav
```

### 3.11 `harmonic_ratio` -> harmonicity-linked pitch

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(harmonic_ratio,0.3,0.85) \
  --route pitch_ratio=clip(pitch_ratio,0.85,1.2) \
  --output out/harmonic_ratio_to_pitch.wav
```

### 3.12 `inharmonicity` -> anti-harmonic pitch detune

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(inharmonicity,-0.8,1.2) \
  --route pitch_ratio=clip(pitch_ratio,0.75,1.2) \
  --output out/inharmonicity_to_pitch.wav
```

### 3.13 `rolloff_norm` -> pitch

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(rolloff_norm,0.45,0.8) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.3) \
  --output out/rolloff_to_pitch.wav
```

### 3.14 `pitch_stability` -> micro-variation amount

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(pitch_stability,0.2,0.9) \
  --route pitch_ratio=clip(pitch_ratio,0.9,1.12) \
  --output out/pitch_stability_to_pitch.wav
```

## 4) Multi-Feature Recipes (Different Features -> Different Targets)

### 4.1 MFCC + MPEG-7 spectral flux

```bash
pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(mfcc_01,0.002,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.5,2.0) \
  --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) \
  --route stretch=clip(stretch,0.85,1.6) \
  --output out/mfcc_flux_dual.wav
```

### 4.2 Formant + onset

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(formant_f1_hz,0.0016,0.2) \
  --route pitch_ratio=clip(pitch_ratio,0.7,1.5) \
  --route stretch=affine(onset_norm,-0.35,1.2) \
  --route stretch=clip(stretch,0.8,1.3) \
  --output out/formant_onset_dual.wav
```

### 4.3 Beat phase + downbeat phase

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(beat_phase,0.12,0.94) \
  --route pitch_ratio=clip(pitch_ratio,0.88,1.12) \
  --route stretch=affine(downbeat_phase,0.3,0.85) \
  --route stretch=clip(stretch,0.85,1.25) \
  --output out/beat_downbeat_dual.wav
```

### 4.4 Tempo + centroid

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route stretch=affine(tempo_bpm,0.003,0.6) \
  --route stretch=clip(stretch,0.8,1.4) \
  --route pitch_ratio=affine(centroid_norm,0.4,0.8) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.3) \
  --output out/tempo_centroid_dual.wav
```

### 4.5 Stereo cues: interaural level difference + interaural time difference

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(ild_db,0.02,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.9,1.1) \
  --route stretch=affine(itd_ms,0.12,1.0) \
  --route stretch=clip(stretch,0.9,1.15) \
  --output out/ild_itd_dual.wav
```

### 4.6 Noise-aware: hiss + hum

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route stretch=affine(hiss_ratio,-0.6,1.2) \
  --route stretch=clip(stretch,0.8,1.2) \
  --route pitch_ratio=affine(hum_60_ratio,-0.4,1.15) \
  --route pitch_ratio=clip(pitch_ratio,0.9,1.2) \
  --output out/noise_aware_dual.wav
```

### 4.7 Speech vs music probabilities

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route stretch=affine(speech_prob,0.5,0.8) \
  --route stretch=clip(stretch,0.8,1.35) \
  --route pitch_ratio=affine(music_prob,0.2,0.9) \
  --route pitch_ratio=clip(pitch_ratio,0.9,1.2) \
  --output out/speech_music_dual.wav
```

### 4.8 Formant 2 + spectral spread

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(formant_f2_hz,0.0008,0.5) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.5) \
  --route stretch=affine(spectral_spread_hz,0.00008,0.85) \
  --route stretch=clip(stretch,0.8,1.4) \
  --output out/formant2_spread_dual.wav
```

### 4.9 Loudness proxy + transient mask

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route stretch=affine(short_lufs_db,0.02,1.4) \
  --route stretch=clip(stretch,0.75,1.4) \
  --route pitch_ratio=affine(transient_mask,0.1,0.95) \
  --route pitch_ratio=clip(pitch_ratio,0.95,1.08) \
  --output out/lufs_transientmask_dual.wav
```

### 4.10 MPEG-7 centroid + MPEG-7 spread

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(mpeg7_spectral_centroid_hz,0.00022,0.75) \
  --route pitch_ratio=clip(pitch_ratio,0.7,1.5) \
  --route stretch=affine(mpeg7_spectral_spread_hz,0.00008,0.9) \
  --route stretch=clip(stretch,0.8,1.5) \
  --output out/mpeg7_centroid_spread.wav
```

### 4.11 MPEG-7 attack-time + MFCC driver

```bash
pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 \
  --route stretch=affine(mpeg7_log_attack_time_s,0.4,1.0) \
  --route stretch=clip(stretch,0.85,1.3) \
  --route pitch_ratio=affine(mfcc_03,0.0015,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.25) \
  --output out/mpeg7_attack_mfcc.wav
```

## 5) Route Operator Patterns (`const`, `inv`, `pow`, `mul`, `add`, `affine`, `clip`)

### 5.1 Start from a feature, then scale and bias

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=pitch_ratio \
  --route pitch_ratio=mul(pitch_ratio,0.85) \
  --route pitch_ratio=add(pitch_ratio,0.15) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.4) \
  --output out/route_mul_add.wav
```

### 5.2 Exaggerate movement with power law

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=affine(flux_norm,1.0,0.0) \
  --route stretch=pow(stretch,1.8) \
  --route stretch=affine(stretch,0.9,0.7) \
  --route stretch=clip(stretch,0.8,1.8) \
  --route pitch_ratio=const(1.0) \
  --output out/route_pow_flux.wav
```

### 5.3 Invert and clamp

```bash
pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 \
  --route stretch=inv(rms_norm) \
  --route stretch=clip(stretch,0.8,1.5) \
  --route pitch_ratio=const(1.0) \
  --output out/route_inv_rms.wav
```

## 6) Feature-Vector Recipes (MFCC + MPEG-7)

### 6.1 MFCC second coefficient driver

```bash
pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(mfcc_02,0.0018,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.7,1.4) \
  --output out/mfcc02_pitch.wav
```

### 6.2 MFCC + MPEG-7 audio spectrum envelope band

```bash
pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(mfcc_04,0.0015,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.75,1.35) \
  --route stretch=affine(mpeg7_audio_spectrum_envelope_03,0.08,1.0) \
  --route stretch=clip(stretch,0.85,1.4) \
  --output out/mfcc_envband_dual.wav
```

### 6.3 MPEG-7 spectral crest and decrease

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(mpeg7_spectral_crest,0.03,0.9) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.25) \
  --route stretch=affine(mpeg7_spectral_decrease,-3.0,1.0) \
  --route stretch=clip(stretch,0.85,1.2) \
  --output out/mpeg7_crest_decrease.wav
```

### 6.4 Increase MFCC dimensionality to 20

```bash
pvx follow guide.wav target.wav --feature-set all --mfcc-count 20 --emit pitch_map --stretch 1.0 \
  --route pitch_ratio=affine(mfcc_10,0.0012,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.25) \
  --route stretch=affine(mfcc_12,0.03,1.0) \
  --route stretch=clip(stretch,0.9,1.2) \
  --output out/mfcc10_mfcc12_dual.wav
```

### 6.5 Build custom vector features, then route them

```bash
pvx pitch-track guide.wav --feature-set all --mfcc-count 20 --output maps/guide_vec.csv
python3 - <<'PY'
import csv
from pathlib import Path
src = Path('maps/guide_vec.csv')
dst = Path('maps/guide_vec_aug.csv')
with src.open() as f_in, dst.open('w', newline='') as f_out:
    r = csv.DictReader(f_in)
    fn = list(r.fieldnames or []) + ['mfcc_energy_1_4', 'mfcc_tilt_1_8']
    w = csv.DictWriter(f_out, fieldnames=fn)
    w.writeheader()
    for row in r:
        m1 = float(row.get('mfcc_01', 0.0) or 0.0)
        m2 = float(row.get('mfcc_02', 0.0) or 0.0)
        m3 = float(row.get('mfcc_03', 0.0) or 0.0)
        m4 = float(row.get('mfcc_04', 0.0) or 0.0)
        m8 = float(row.get('mfcc_08', 0.0) or 0.0)
        row['mfcc_energy_1_4'] = 0.25 * (m1 + m2 + m3 + m4)
        row['mfcc_tilt_1_8'] = m1 - m8
        w.writerow(row)
PY
pvx voc target.wav --pitch-map maps/guide_vec_aug.csv \
  --route pitch_ratio=affine(mfcc_tilt_1_8,0.0018,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.75,1.4) \
  --route stretch=affine(mfcc_energy_1_4,0.05,1.0) \
  --route stretch=clip(stretch,0.85,1.4) \
  --output out/vector_augmented.wav
```

## 7) Multiple Guide Files (A and B both influence target)

### 7.1 Blend pitch behavior from two guides

```bash
pvx pitch-track guide_A.wav --feature-set all --output maps/guide_A.csv
pvx pitch-track guide_B.wav --feature-set all --output maps/guide_B.csv
python3 - <<'PY'
import csv
from pathlib import Path

a = Path('maps/guide_A.csv')
b = Path('maps/guide_B.csv')
out = Path('maps/guide_blend.csv')
with a.open() as fa, b.open() as fb, out.open('w', newline='') as fo:
    ra = csv.DictReader(fa)
    rb = csv.DictReader(fb)
    rows_b = list(rb)
    fields = list(ra.fieldnames or [])
    if 'pitch_ratio' not in fields:
        fields.append('pitch_ratio')
    w = csv.DictWriter(fo, fieldnames=fields)
    w.writeheader()
    rows_b_len = max(1, len(rows_b))
    for i, row_a in enumerate(ra):
        row_b = rows_b[min(i, rows_b_len - 1)]
        pa = float(row_a.get('pitch_ratio', 1.0) or 1.0)
        pb = float(row_b.get('pitch_ratio', 1.0) or 1.0)
        row_a['pitch_ratio'] = 0.6 * pa + 0.4 * pb
        w.writerow(row_a)
PY
pvx voc target.wav --pitch-map maps/guide_blend.csv \
  --route pitch_ratio=clip(pitch_ratio,0.7,1.5) \
  --route stretch=const(1.0) \
  --output out/two_guide_pitch_blend.wav
```

### 7.2 Guide A controls pitch, guide B controls stretch

```bash
pvx pitch-track guide_A.wav --feature-set all --output maps/guide_A.csv
pvx pitch-track guide_B.wav --feature-set all --output maps/guide_B.csv
python3 - <<'PY'
import csv
from pathlib import Path

a = Path('maps/guide_A.csv')
b = Path('maps/guide_B.csv')
out = Path('maps/guide_Apitch_Bstretch.csv')
with a.open() as fa, b.open() as fb, out.open('w', newline='') as fo:
    ra = csv.DictReader(fa)
    rb = csv.DictReader(fb)
    rows_b = list(rb)
    fields = list(dict.fromkeys((ra.fieldnames or []) + ['stretch']))
    w = csv.DictWriter(fo, fieldnames=fields)
    w.writeheader()
    rows_b_len = max(1, len(rows_b))
    for i, row_a in enumerate(ra):
        row_b = rows_b[min(i, rows_b_len - 1)]
        flux_b = float(row_b.get('spectral_flux', 0.0) or 0.0)
        row_a['stretch'] = max(0.8, min(1.5, 1.0 + 0.03 * flux_b))
        w.writerow(row_a)
PY
pvx voc target.wav --pitch-map maps/guide_Apitch_Bstretch.csv \
  --route pitch_ratio=clip(pitch_ratio,0.75,1.35) \
  --route stretch=clip(stretch,0.8,1.5) \
  --output out/two_guide_split_roles.wav
```

## 8) Manual Pipe Variants

Pitch-track to `pvx voc` in one pipe:

```bash
pvx pitch-track guide.wav --feature-set all --mfcc-count 13 --output - \
  | pvx voc target.wav --control-stdin \
      --route pitch_ratio=affine(mfcc_01,0.002,1.0) \
      --route pitch_ratio=clip(pitch_ratio,0.5,2.0) \
      --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) \
      --route stretch=clip(stretch,0.85,1.6) \
      --output out/manual_pipe_feature_follow.wav
```

Time-varying control map followed by denoise/deverb in one chain:

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --route stretch=affine(spectral_flux,0.04,1.0) \
  --route stretch=clip(stretch,0.9,1.4) \
  --route pitch_ratio=affine(mfcc_01,0.0015,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.85,1.2) \
  --output out/follow_stage.wav
pvx denoise out/follow_stage.wav --reduction-db 8 --output out/follow_stage_dn.wav
pvx deverb out/follow_stage_dn.wav --strength 0.35 --output out/follow_stage_dn_dr.wav
```

## 9) Reuse a Saved Feature Map for Fast Iteration

```bash
pvx pitch-track guide.wav --feature-set all --mfcc-count 13 --output maps/guide_full.csv
pvx voc target.wav --pitch-map maps/guide_full.csv \
  --route pitch_ratio=affine(mfcc_01,0.002,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.5,2.0) \
  --route stretch=affine(spectral_flux,0.04,1.0) \
  --route stretch=clip(stretch,0.85,1.6) \
  --output out/reused_map_follow.wav

pvx voc target.wav --pitch-map maps/guide_full.csv \
  --route pitch_ratio=affine(formant_f1_hz,0.0012,0.6) \
  --route pitch_ratio=clip(pitch_ratio,0.8,1.4) \
  --route stretch=affine(onset_norm,-0.3,1.2) \
  --route stretch=clip(stretch,0.8,1.3) \
  --output out/reused_map_alt_formula.wav
```

## 10) Compact Feature Set (Smaller CSV)

```bash
pvx pitch-track guide.wav --feature-set basic --output maps/guide_basic.csv
pvx voc target.wav --pitch-map maps/guide_basic.csv \
  --route pitch_ratio=affine(spectral_centroid_hz,0.0002,0.8) \
  --route pitch_ratio=clip(pitch_ratio,0.75,1.4) \
  --output out/basic_features_follow.wav
```

## 11) Safe Starting Ranges

When designing formulas, start conservative and widen only if needed:

- `pitch_ratio`: `0.85 .. 1.15`
- `stretch`: `0.8 .. 1.4`
- `confidence floor`: `0.65 .. 0.85`
- map smoothing: `--pitch-map-smooth-ms 10..40`

Example conservative profile:

```bash
pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 \
  --pitch-conf-min 0.8 --pitch-lowconf-mode hold --pitch-map-smooth-ms 20 \
  --route pitch_ratio=affine(mfcc_01,0.001,1.0) \
  --route pitch_ratio=clip(pitch_ratio,0.9,1.1) \
  --route stretch=affine(spectral_flux,0.02,1.0) \
  --route stretch=clip(stretch,0.9,1.2) \
  --output out/conservative_sidechain.wav
```

## Attribution
