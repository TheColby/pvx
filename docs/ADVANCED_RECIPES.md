# Advanced Session Recipes

This chapter collects complete `pvx` sessions rather than isolated commands. Each recipe begins with a production or compositional goal, creates any required control or analysis artifacts, performs the render, and names the main failure modes to audition. The examples assume the current directory contains `controls`, `maps`, `artifacts`, `renders`, `checkpoints`, and `reports` directories.

```bash
mkdir -p controls maps artifacts renders checkpoints reports
```

The recipes deliberately overlap in technique. Repetition at this level is useful because a command acquires a different meaning when it appears before restoration, after harmonic mapping, inside a sidechain system, or at the end of a checkpointed long render. Treat filenames as a session plan and preserve the intermediate artifacts until the result has been approved.

## Restoration and delivery

The first group treats phase-vocoder work as part of a larger production chain. Conservative values are intentional. Restoration artifacts accumulate quickly when denoising, dereverberation, stretching, and loudness control are all applied to the same signal.

### Recipe 1: dialogue reconstruction from matched room tone

Build a spectral noise profile from isolated room tone, apply moderate profile-driven suppression, reduce the remaining room tail, then slow the dialogue slightly and finish it at a controlled delivery level.

```bash
pvx analysis create roomtone.wav --output artifacts/roomtone.pvxan.npz
pvx response create artifacts/roomtone.pvxan.npz \
  --output artifacts/roomtone.pvxrf.npz --method median --normalize peak
pvx noisefilter dialog.wav \
  --response artifacts/roomtone.pvxrf.npz \
  --noise-floor 1.2 --response-mix 1.0 --dry-mix 0.05 \
  --output renders/dialog_profiled.wav
pvx deverb renders/dialog_profiled.wav --strength 0.30 --stdout \
| pvx voc - --preset vocal_studio --stretch 1.06 \
    --target-lufs -16 --limiter-threshold -1.0 \
    --output renders/dialog_delivery.wav
```

Audition sibilants, breaths, and word endings before judging the noise floor. If the voice becomes hollow, reduce profile influence before reducing more reverberation.

### Recipe 2: archival speech with two-stage timing repair

Use a gentle first pass to improve intelligibility, then apply a small pacing correction in a separate render. Separate stages make it easier to determine whether an artifact came from restoration or time modification.

```bash
pvx denoise archive.wav --reduction-db 5 --smooth 7 --stdout \
| pvx deverb - --strength 0.25 --output renders/archive_clean.wav
pvx voc renders/archive_clean.wav \
  --stretch 1.12 --phase-locking identity \
  --n-fft 2048 --hop-size 256 \
  --target-lufs -18 --limiter-threshold -1.5 \
  --manifest-json reports/archive.json --manifest-append \
  --output renders/archive_retimed.wav
```

Compare consonant edges against `archive_clean.wav`. The second pass should alter pace without making the first pass sound retrospectively overprocessed.

### Recipe 3: field recording preservation with multiresolution stretch

Clean a field recording lightly, then combine three analysis scales so low-frequency ambience and short foreground events do not depend on one FFT size.

```bash
pvx denoise field.wav --reduction-db 4 --smooth 9 --stdout \
| pvx deverb - --strength 0.20 --output renders/field_clean.wav
pvx voc renders/field_clean.wav \
  --stretch 1.35 --multires-fusion \
  --multires-ffts 1024,2048,4096 \
  --multires-weights 0.25,0.35,0.40 \
  --target-lufs -20 --limiter-threshold -1.5 \
  --output renders/field_multires.wav
```

Listen separately to insects, wind, distant traffic, and nearby impacts. A weight set that improves one layer can blur another, so retain the cleaned intermediate for comparison.

### Recipe 4: coherent multichannel restoration

For a multichannel recording, remove modest noise first and anchor phase evolution to a reference channel during the time adjustment. The reference should contain a stable version of the events shared across channels.

```bash
pvx denoise surround.wav --reduction-db 4 --smooth 7 \
  --output renders/surround_clean.wav
pvx voc renders/surround_clean.wav \
  --stretch 1.10 \
  --stereo-mode ref_channel_lock --ref-channel 2 \
  --coherence-strength 0.95 \
  --bit-depth 24 --dither tpdf --dither-seed 41 \
  --metadata-policy sidecar \
  --output renders/surround_coherent.wav
```

Check channel pairs, mono compatibility, and diffuse ambience. Excessive coherence can pull legitimately decorrelated material toward the reference channel.

## Automation as composition

The next recipes treat control files as reusable musical objects. Generate them deterministically, inspect them as data, and reshape them independently of the sound render.

### Recipe 5: exponential expansion with a smoothed alternate take

Create one strongly curved stretch trajectory and a smoothed derivative, then render both against identical source and phase settings.

```bash
pvx envelope --mode exp --duration 10 --rate 30 \
  --start 0.7 --end 1.8 --exp-curve 6 \
  --key stretch --output controls/stretch_exp.csv
pvx reshape controls/stretch_exp.csv \
  --key stretch --operation smooth --window 11 \
  --output controls/stretch_exp_smooth.csv
pvx voc input.wav --stretch controls/stretch_exp.csv \
  --interp linear --output renders/stretch_exp.wav
pvx voc input.wav --stretch controls/stretch_exp_smooth.csv \
  --interp linear --output renders/stretch_exp_smooth.wav
```

The useful comparison is not simply smooth versus rough. Decide whether the sharper trajectory contributes phrasing or merely exposes control-rate discontinuity.

### Recipe 6: polynomial pitch arc with formant protection

Drive pitch from JSON control points, use cubic-order polynomial interpolation, and preserve vocal identity during the excursion.

```json
{
  "points": [
    {"time_sec": 0.0, "value": 1.0},
    {"time_sec": 1.5, "value": 1.122462048},
    {"time_sec": 3.0, "value": 1.334839854},
    {"time_sec": 4.5, "value": 0.890898718},
    {"time_sec": 6.0, "value": 1.0}
  ]
}
```

Save the object above as `controls/pitch_arc.json`, then render it:

```bash
pvx voc vocal.wav \
  --pitch-shift-ratio controls/pitch_arc.json \
  --interp polynomial --order 3 \
  --pitch-mode formant-preserving \
  --stereo-mode mid_side_lock --coherence-strength 0.90 \
  --output renders/vocal_pitch_arc.wav
```

Listen for polynomial overshoot between points, vowel-size drift, and center-image movement. Add points or reduce interpolation order before adding more smoothing.

### Recipe 7: rhythmic sample-and-hold time folding

Use explicit time regions and sample-and-hold interpolation to alternate compression and expansion. This is intentionally discontinuous and works best when boundaries align with musical events.

```csv
start_sec,end_sec,value
0.0,0.5,1.0
0.5,1.0,1.5
1.0,1.5,0.75
1.5,2.0,2.0
2.0,3.0,1.0
```

Save the map as `controls/fold.csv`, then render it:

```bash
pvx voc loop.wav \
  --stretch controls/fold.csv --interp none \
  --transient-mode hybrid \
  --output renders/loop_time_fold.wav
```

Move one boundary by a few milliseconds and compare. The experiment reveals whether the gesture depends on event alignment or on the ratio sequence alone.

### Recipe 8: section-adaptive analysis resolution

Change FFT and hop sizes over time so transient sections use shorter analysis while sustained sections receive greater frequency resolution.

```csv
time_sec,value
0.0,1024
4.0,1024
4.1,4096
12.0,4096
12.1,2048
```

Save that map as `controls/nfft.csv`. Create a matching hop map named `controls/hop.csv` with values `128`, `128`, `512`, `512`, and `256` at the same timestamps.

```bash
pvx voc arrangement.wav \
  --n-fft controls/nfft.csv --hop-size controls/hop.csv \
  --stretch 1.25 --interp linear \
  --output renders/arrangement_adaptive.wav
```

Audition the two transition regions in isolation. Resolution changes can become timbral edits even when the requested stretch remains constant.

## Sidechain and feature routing

Feature routing allows one file to shape another without replacing its spectrum directly. Always clamp derived values. A guide feature can contain outliers that are harmless as measurements but destructive as stretch or pitch ratios.

### Recipe 9: inverse melodic breathing

Use the guide's pitch contour to compress target time at higher notes and expand it at lower notes while leaving target pitch unchanged.

```bash
pvx follow melody.wav drone.wav \
  --emit pitch_map --stretch 1.0 --pitch-conf-min 0.75 \
  --route 'stretch=inv(pitch_ratio)' \
  --route 'stretch=clip(stretch,0.70,1.80)' \
  --route 'pitch_ratio=const(1.0)' \
  --output renders/drone_inverse_melody.wav
```

The result follows relative pitch movement, not note duration or score position. Listen for unstable regions where the guide becomes unvoiced.

### Recipe 10: MFCC pitch with flux-driven time

Route one timbral coefficient to target pitch and MPEG-7 spectral flux to target stretch. This produces coupled motion from two different aspects of the guide.

```bash
pvx follow guide.wav target.wav \
  --feature-set all --mfcc-count 13 \
  --emit pitch_map --stretch 1.0 \
  --route 'pitch_ratio=affine(mfcc_01,0.002,1.0)' \
  --route 'pitch_ratio=clip(pitch_ratio,0.5,2.0)' \
  --route 'stretch=affine(mpeg7_spectral_flux,0.05,1.0)' \
  --route 'stretch=clip(stretch,0.85,1.6)' \
  --output renders/target_mfcc_flux.wav
```

MFCC values are source-dependent. Treat the coefficients as control signals whose scaling must be calibrated, not as universal perceptual units.

### Recipe 11: transient-protected feature following

Use loudness to create broad timing motion while the renderer's hybrid transient mode protects attacks. Track first so the feature map can be inspected before rendering.

```bash
pvx pitch-track percussion_guide.wav \
  --feature-set all --output maps/percussion_features.csv
pvx voc pad.wav --pitch-map maps/percussion_features.csv \
  --route 'stretch=affine(rms_norm,1.0,0.7)' \
  --route 'stretch=clip(stretch,0.80,1.60)' \
  --route 'pitch_ratio=const(1.0)' \
  --transient-mode hybrid \
  --output renders/pad_percussion_motion.wav
```

Inspect the routed map if attacks still bloom. Feature-derived control and the renderer's transient mode solve related but different problems.

### Recipe 12: two guides with separate musical roles

Let one guide supply pitch and another guide supply spectral-flux timing. The short Python stage merges two inspected feature files into one explicit control artifact.

```bash
pvx pitch-track pitch_guide.wav --feature-set all --output maps/pitch_guide.csv
pvx pitch-track rhythm_guide.wav --feature-set all --output maps/rhythm_guide.csv
python3 - <<'PY'
import csv
from pathlib import Path

a = Path("maps/pitch_guide.csv")
b = Path("maps/rhythm_guide.csv")
out = Path("maps/two_guide.csv")
with a.open() as fa, b.open() as fb, out.open("w", newline="") as fo:
    pitch_rows = list(csv.DictReader(fa))
    rhythm_rows = list(csv.DictReader(fb))
    fields = list(dict.fromkeys(list(pitch_rows[0]) + ["stretch"]))
    writer = csv.DictWriter(fo, fieldnames=fields)
    writer.writeheader()
    for index, row in enumerate(pitch_rows):
        rhythm = rhythm_rows[min(index, len(rhythm_rows) - 1)]
        flux = float(rhythm.get("spectral_flux", 0.0) or 0.0)
        row["stretch"] = max(0.8, min(1.5, 1.0 + 0.03 * flux))
        writer.writerow(row)
PY
pvx voc target.wav --pitch-map maps/two_guide.csv \
  --route 'pitch_ratio=clip(pitch_ratio,0.75,1.35)' \
  --route 'stretch=clip(stretch,0.80,1.50)' \
  --output renders/target_two_guides.wav
```

This merge aligns rows rather than musical events. For guides with different durations or analysis rates, resample them to a shared timeline before combining features.

## Persistent spectral artifacts

These recipes use PVXAN and PVXRF files as session assets. Keep their metadata with the project and name them after both source and derivation method.

### Recipe 13: reusable timbral imprint with an automated entrance

Derive a median response from one source, generate a slow response-mix curve, and introduce that spectral identity into another source over twelve seconds.

```bash
pvx analysis create color.wav --output artifacts/color.pvxan.npz \
  --n-fft 4096 --win-length 4096 --hop-size 256 --window hann
pvx response create artifacts/color.pvxan.npz \
  --output artifacts/color_median.pvxrf.npz \
  --method median --phase-mode mean --normalize peak --smoothing-bins 3
pvx envelope --mode ramp --duration 12 --rate 30 \
  --start 0.0 --end 1.0 --key response_mix \
  --output controls/color_mix.csv
pvx tvfilter carrier.wav \
  --response artifacts/color_median.pvxrf.npz \
  --tv-map controls/color_mix.csv --tv-key response_mix \
  --tv-interp s_curve --tv-order 3 \
  --output renders/carrier_color_entrance.wav
```

Compare median response against an RMS-derived response. The change can be more consequential than the automation curve.

### Recipe 14: response-informed noise removal before extreme stretch

Use a noise-only excerpt to build a profile, clean the source, then perform a resumable fifteen-minute stretch. Cleaning first prevents stationary noise from becoming the dominant long-form texture.

```bash
pvx analysis create noise_only.wav --output artifacts/noise.pvxan.npz
pvx response create artifacts/noise.pvxan.npz \
  --output artifacts/noise.pvxrf.npz --method median --normalize peak
pvx noisefilter seed.wav --response artifacts/noise.pvxrf.npz \
  --noise-floor 1.1 --response-mix 0.8 --dry-mix 0.10 \
  --output renders/seed_clean.wav
pvx voc renders/seed_clean.wav \
  --preset extreme_ambient --target-duration 900 \
  --auto-segment-seconds 0.25 \
  --checkpoint-dir checkpoints/seed_15min \
  --manifest-json reports/seed_15min.json --manifest-append \
  --output renders/seed_15min.wav
```

Also render ten seconds from the untreated seed. Extreme stretching makes profile mistakes easy to hear and difficult to diagnose after a long render.

### Recipe 15: morph, freeze, and finish a hybrid pad

Create a balanced spectral morph, freeze a stable point, remove a small amount of accumulated haze, and set a restrained final loudness.

```bash
pvx morph bells.wav choir.wav --alpha 0.5 \
  --output renders/bells_choir.wav
pvx freeze renders/bells_choir.wav \
  --freeze-time 0.8 --duration 40 \
  --output renders/bells_choir_frozen.wav
pvx denoise renders/bells_choir_frozen.wav \
  --reduction-db 3 --smooth 9 --stdout \
| pvx voc - --stretch 1.0 --target-lufs -20 \
    --limiter-threshold -1.5 \
    --output renders/bells_choir_pad.wav
```

Choose the freeze point by listening to the morph first. A technically stable frame can still have an unhelpful vowel, beating pattern, or stereo balance.

### Recipe 16: response-shaped dynamics and resonant afterimage

Use one response artifact for spectral companding, then add a restrained tuned resonance as a separate, inspectable stage.

```bash
pvx analysis create reference.wav --output artifacts/reference.pvxan.npz
pvx response create artifacts/reference.pvxan.npz \
  --output artifacts/reference.pvxrf.npz --method rms --normalize rms
pvx spec-compander source.wav \
  --response artifacts/reference.pvxrf.npz \
  --comp-threshold-db -18 --comp-ratio 2.5 --expand-ratio 1.3 \
  --output renders/source_speccomp.wav
pvx ringfilter renders/source_speccomp.wav \
  --frequency-hz 43 --depth 0.35 --mix 0.35 \
  --resonance-hz 860 --resonance-q 7 \
  --resonance-mix 0.25 --resonance-decay 0.18 \
  --output renders/source_afterimage.wav
```

The first stage changes dynamic relationships among frequency regions; the second adds persistence. If the result masks attacks, reduce resonance before weakening the compander.

## Harmonic, inharmonic, and spatial design

The following recipes deliberately stack pitch-class operations, spectral remapping, and stereo processing. Render intermediate stages because harmonic mistakes become difficult to attribute after widening and resonance.

### Recipe 17: retuned harmony choir with controlled width

Retune the lead, construct a triadic stack, then add modest unison width. Each operation receives its own file so tuning, voicing, and spatial density can be approved separately.

```bash
pvx retune lead.wav --scale major --root D \
  --output renders/lead_retuned.wav
pvx harmonize renders/lead_retuned.wav \
  --intervals 0,4,7 --gains 1,0.75,0.65 \
  --output renders/lead_harmony.wav
pvx unison renders/lead_harmony.wav \
  --voices 5 --detune-cents 9 --width 0.9 \
  --output renders/lead_choir.wav
```

Approve octave tracking in the retuned file before judging the choir. Unison thickening can conceal tuning errors without correcting them.

### Recipe 18: chord-constrained resonant cloud

Map an ambience recording toward a minor-seventh collection, stretch it, then add a high-Q resonant layer at low wet level.

```bash
pvx chordmapper ambience.wav \
  --root-hz 110 --chord min7 --strength 0.80 \
  --tolerance-cents 40 --boost-db 6 --attenuation 0.40 \
  --output renders/ambience_min7.wav
pvx voc renders/ambience_min7.wav \
  --stretch 3.0 --multires-fusion \
  --multires-ffts 1024,2048,4096 \
  --multires-weights 0.20,0.35,0.45 \
  --output renders/ambience_min7_stretched.wav
pvx ringfilter renders/ambience_min7_stretched.wav \
  --frequency-hz 27.5 --depth 0.25 --mix 0.25 \
  --resonance-hz 1320 --resonance-q 12 \
  --resonance-mix 0.18 --resonance-decay 0.30 \
  --output renders/ambience_min7_cloud.wav
```

Narrow chord tolerance and high resonance Q can fight each other. Widen the mapping before increasing resonant energy.

### Recipe 19: inharmonic bell transformed into a frozen bed

Warp a bell toward stiff-string partial spacing, stretch the result, and freeze a moment after the initial strike.

```bash
pvx inharmonator bell.wav \
  --inharmonic-f0-hz 220 --inharmonicity 0.0002 \
  --inharmonic-mix 1.0 --dry-mix 0.1 \
  --output renders/bell_inharmonic.wav
pvx voc renders/bell_inharmonic.wav \
  --stretch 4.0 --transient-mode hybrid \
  --output renders/bell_inharmonic_stretched.wav
pvx freeze renders/bell_inharmonic_stretched.wav \
  --freeze-time 1.2 --duration 60 \
  --output renders/bell_inharmonic_bed.wav
```

The freeze point should occur after the protected strike but before the evolving partials lose the interval structure that motivated the recipe.

### Recipe 20: stereo vocal transposition and harmony with image control

Shift the stereo vocal with formant preservation and mid-side locking, then harmonize and apply restrained unison width.

```bash
pvx voc stereo_vocal.wav \
  --stretch 1.0 --pitch 2 \
  --pitch-mode formant-preserving \
  --stereo-mode mid_side_lock --coherence-strength 0.90 \
  --output renders/stereo_vocal_up2.wav
pvx harmonize renders/stereo_vocal_up2.wav \
  --intervals 0,3,7 --gains 1,0.60,0.50 \
  --output renders/stereo_vocal_harmony.wav
pvx unison renders/stereo_vocal_harmony.wav \
  --voices 3 --detune-cents 6 --width 0.65 \
  --output renders/stereo_vocal_harmony_wide.wav
```

Check the center after each stage. Width should be the final decoration, not a remedy for phase instability introduced earlier.

## Long renders, comparison, and reproducibility

Complex sessions need failure recovery and evidence. The final group emphasizes checkpoints, manifests, deterministic exports, and matched alternatives.

### Recipe 21: four-hour resumable installation render

Render an extreme duration through small checkpointable segments and preserve the resolved session in a manifest. The same command resumes after interruption.

```bash
pvx voc installation_seed.wav \
  --target-duration 14400 \
  --preset extreme_ambient \
  --checkpoint-dir checkpoints/installation \
  --auto-segment-seconds 0.25 --resume \
  --manifest-json reports/installation.json --manifest-append \
  --output renders/installation_4h.wav
```

Test the first minute before committing hours of computation. Recovery protects time, but it does not protect against an unattractive parameter choice.

### Recipe 22: deterministic delivery and null-test pair

Render the same transform twice with an explicit dither seed. Matching outputs provide a quick reproducibility check for the supported deterministic path.

```bash
pvx voc master.wav --stretch 1.08 \
  --bit-depth 24 --dither tpdf --dither-seed 123 \
  --manifest-json reports/deterministic.json --manifest-append \
  --output renders/master_seed123_a.wav
pvx voc master.wav --stretch 1.08 \
  --bit-depth 24 --dither tpdf --dither-seed 123 \
  --manifest-json reports/deterministic.json --manifest-append \
  --output renders/master_seed123_b.wav
```

Compare hashes and perform a null test in an audio editor. A mismatch is evidence to investigate, not automatically proof of an audible regression.

### Recipe 23: matched transform and resolution suite

Create four controlled alternatives while changing only one family of analysis choices. Consistent naming turns the output directory into a compact listening experiment.

```bash
pvx voc source.wav --stretch 2.0 --transform fft \
  --n-fft 1024 --output renders/source_fft_1024.wav
pvx voc source.wav --stretch 2.0 --transform fft \
  --n-fft 4096 --output renders/source_fft_4096.wav
pvx voc source.wav --stretch 2.0 --transform hartley \
  --n-fft 1024 --output renders/source_hartley_1024.wav
pvx voc source.wav --stretch 2.0 --transform hartley \
  --n-fft 4096 --output renders/source_hartley_4096.wav
```

Level-match and randomize playback order. Without that discipline, file naming and render time can bias the comparison before listening begins.

### Recipe 24: complex session with plan, render, and regression gate

Resolve an automatic plan without rendering, perform a manifested checkpointed render, then run the focused PVC-inspired parity benchmark. This closes the loop from intention to artifact to repository health.

```bash
pvx voc source.wav \
  --auto-profile --auto-transform \
  --manifest-json reports/complex_plan.json \
  --dry-run --explain-plan
pvx voc source.wav \
  --auto-profile --auto-transform \
  --target-duration 1200 \
  --checkpoint-dir checkpoints/complex_session \
  --auto-segment-seconds 0.5 --resume \
  --manifest-json reports/complex_render.json --manifest-append \
  --output renders/complex_session.wav
python3 benchmarks/run_pvc_parity.py \
  --quick --out-dir benchmarks/out_pvc_parity \
  --baseline benchmarks/baseline_pvc_parity.json \
  --gate --gate-tolerance 0.20
```

The benchmark does not certify the artistic render. It verifies that the implementation still satisfies a focused set of deterministic regression expectations after the session's tools have been exercised.

## Adapting the recipes

Complexity should be added one auditable stage at a time. Keep every intermediate while developing a recipe, compare at matched loudness, and write down the source region used for judgment. Once the chain is understood, remove intermediates only as an operational optimization.

A reliable adaptation changes one class of decision at a time. First establish the source and analysis settings. Next establish automation. Then add restoration, harmonic processing, or spatial policy. Finish with loudness, bit depth, dither, metadata, and manifests. This order is not mandatory, but it makes failures easier to locate and successful gestures easier to preserve.
