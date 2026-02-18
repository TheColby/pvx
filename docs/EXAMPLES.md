# pvx Example Cookbook

All commands are designed to be copy-paste runnable from the repository root.

Assumptions:
- input files exist in your working directory
- output format defaults to input extension unless overridden
- tools using `--output-dir` and `--suffix` write inferred filenames

---

## 1) Slow Down Speech

**Command**
```bash
python3 pvxvoc.py speech.wav --preset vocal --stretch 1.35 --output speech_slow.wav
```

**Explanation**
- Uses vocal preset and moderate stretch to increase intelligibility review time.

**Before/After**
- Before: normal speech rate.
- After: slower phrase timing, mostly same pitch.

**Parameters that matter most**
- `--stretch`
- `--preset vocal`

**Artifacts to listen for**
- consonant smearing
- subtle flutter in sustained vowels

---

## 2) Time-Compress Speech (Faster Playback)

**Command**
```bash
python3 pvxvoc.py speech.wav --preset vocal --stretch 0.85 --output speech_fast.wav
```

**Explanation**
- Compresses duration while trying to preserve speech quality.

**Before/After**
- Before: original pace.
- After: shorter/faster delivery.

**Parameters that matter most**
- `--stretch < 1`
- `--phase-locking identity`

**Artifacts to listen for**
- transient roughness on plosives
- “grainy” tails on fricatives

---

## 3) Raise Pitch Without Changing Speed

**Command**
```bash
python3 pvxvoc.py vocal.wav --stretch 1.0 --pitch 3 --output vocal_up3.wav
```

**Explanation**
- Raises pitch by 3 semitones with duration unchanged.

**Before/After**
- Before: original key center.
- After: same timing, higher pitch.

**Parameters that matter most**
- `--pitch`
- `--stretch 1.0`

**Artifacts to listen for**
- timbral thinning
- phasey upper harmonics at larger shifts

---

## 4) Lower Pitch With Formant Preservation

**Command**
```bash
python3 pvxvoc.py vocal.wav --stretch 1.0 --pitch -4 --pitch-mode formant-preserving --output vocal_down4_formant.wav
```

**Explanation**
- Applies downward pitch shift while compensating formant envelope drift.

**Before/After**
- Before: natural vocal size.
- After: lower notes with less “giant voice” coloration.

**Parameters that matter most**
- `--pitch`
- `--pitch-mode formant-preserving`
- `--formant-lifter`

**Artifacts to listen for**
- residual “hollow” vowels
- over-correction if formant settings are aggressive

---

## 5) Stretch + Preserve Formants

**Command**
```bash
python3 pvxvoc.py vocal.wav --stretch 1.2 --pitch -2 --pitch-mode formant-preserving --output vocal_stretch_formant.wav
```

**Explanation**
- Combined timing and pitch modification tuned for voice.

**Before/After**
- Before: original timing and key.
- After: slower and lower while retaining more vowel identity.

**Parameters that matter most**
- `--stretch`
- `--pitch`
- `--pitch-mode`

**Artifacts to listen for**
- transient blur at phrase starts
- comb-like coloration on loud sustained notes

---

## 6) Extreme Time Stretch Ambient Pad

**Command**
```bash
python3 pvxvoc.py one_shot.wav --preset ambient --target-duration 600 --output one_shot_10min.wav
```

**Explanation**
- Uses ambient preset for very large stretch ratios.

**Before/After**
- Before: short transient-rich source.
- After: long evolving drone texture.

**Parameters that matter most**
- `--preset ambient`
- `--target-duration`
- `--n-fft`, `--hop-size`

**Artifacts to listen for**
- low-level flutter on bright components
- transient residue if source is very percussive

---

## 7) Freeze a Harmonic Moment

**Command**
```bash
python3 pvxfreeze.py guitar_chord.wav --freeze-time 0.45 --duration 12 --output-dir out --suffix _freeze
```

**Explanation**
- Captures one spectral slice and resynthesizes sustained texture.

**Before/After**
- Before: normal decay envelope.
- After: held spectral “snapshot.”

**Parameters that matter most**
- `--freeze-time`
- `--duration`

**Artifacts to listen for**
- static texture if freeze point is too narrow-band
- periodic shimmer if random phase is enabled heavily

---

## 8) Retune Vocal to A440

**Command**
```bash
python3 pvxvoc.py vocal_note.wav --target-f0 440 --pitch-mode formant-preserving --output vocal_A440.wav
```

**Explanation**
- Estimates source F0 and applies a global ratio so the tracked center lands at A4 = 440 Hz.

**Before/After**
- Before: note center may sit sharp/flat vs A440 reference.
- After: center frequency aligns to 440 Hz.

**Parameters that matter most**
- `--target-f0`
- `--pitch-mode`

**Artifacts to listen for**
- metallic edges on strong consonants
- formant drift if `--pitch-mode standard` is used

---

## 9) Morph Two Sounds

**Command**
```bash
python3 pvxmorph.py source_a.wav source_b.wav --alpha 0.35 --output morph_35.wav --overwrite
```

**Explanation**
- Interpolates spectral content between A and B.

**Before/After**
- Before: two distinct sources.
- After: hybrid timbre weighted toward A at `alpha=0.35`.

**Parameters that matter most**
- `--alpha`

**Artifacts to listen for**
- phasing if sources differ strongly in timing
- hollow midrange from spectral mismatch

---

## 10) Create Chorus/Unison Thickness

**Command**
```bash
python3 pvxunison.py synth.wav --voices 7 --detune-cents 18 --width 1.2 --dry-mix 0.15 --output-dir out --suffix _uni7
```

**Explanation**
- Generates detuned multi-voice spread.

**Before/After**
- Before: narrow mono/stereo lead.
- After: wider, denser ensemble-like body.

**Parameters that matter most**
- `--voices`
- `--detune-cents`
- `--width`

**Artifacts to listen for**
- excessive beating at large detune
- low-end smearing if source is bass-heavy

---

## 11) De-Noise Field Recording

**Command**
```bash
python3 pvxdenoise.py field.wav --noise-seconds 0.5 --reduction-db 10 --smooth 7 --output-dir out --suffix _den
```

**Explanation**
- Builds a noise estimate from the opening region and subtracts adaptively.

**Before/After**
- Before: broadband hiss/air/noise floor.
- After: lower noise floor with possible detail loss.

**Parameters that matter most**
- `--noise-seconds`
- `--reduction-db`
- `--smooth`

**Artifacts to listen for**
- musical noise
- dullness in high frequencies

---

## 12) De-Reverb a Room Recording

**Command**
```bash
python3 pvxdeverb.py room_voice.wav --strength 0.55 --decay 0.90 --floor 0.12 --output-dir out --suffix _dry
```

**Explanation**
- Suppresses late reverberant tails in spectral domain.

**Before/After**
- Before: long room decay and reduced articulation.
- After: drier presentation and clearer direct sound.

**Parameters that matter most**
- `--strength`
- `--decay`
- `--floor`

**Artifacts to listen for**
- pumping on sustained vowels
- “underwater” tails if too aggressive

---

## 13) Segment-Based Dynamic Stretch via CSV

**Command**
```bash
python3 pvxvoc.py phrase.wav --pitch-map map_warp.csv --output phrase_map.wav
```

**Example `map_warp.csv`**
```csv
start_sec,end_sec,stretch
0.0,0.8,1.00
0.8,1.6,1.25
1.6,2.2,0.90
2.2,3.0,1.10
```

**Explanation**
- Applies piecewise timing control across segments.

**Before/After**
- Before: fixed timing.
- After: phrase-level temporal reshaping.

**Parameters that matter most**
- CSV `stretch`
- segment boundaries
- `--pitch-map-crossfade-ms`

**Artifacts to listen for**
- boundary clicks if crossfade is too short
- unnatural tempo discontinuities

---

## 14) Sidechain Pitch Trajectory (A Controls B)

**Command**
```bash
python3 HPS-pitch-track.py A.wav | python3 pvxvoc.py B.wav --pitch-follow-stdin --pitch-conf-min 0.75 --output B_follow.wav
```

**Explanation**
- Tracks F0 in A and applies contour as ratio map to B.

**Before/After**
- Before: B has its own pitch motion.
- After: B follows A’s pitch trajectory where confidence is sufficient.

**Parameters that matter most**
- tracker backend and frame settings
- `--pitch-conf-min`
- `--pitch-lowconf-mode`

**Artifacts to listen for**
- warbling from noisy tracker segments
- abrupt ratio jumps at low-confidence transitions

---

## 15) GPU Acceleration

**Command**
```bash
python3 pvxvoc.py mix.wav --gpu --stretch 1.12 --output mix_gpu.wav
```

**Explanation**
- Uses CUDA path if available (`--gpu` alias for `--device cuda`).

**Before/After**
- Before: CPU render time baseline.
- After: same algorithmic output path with potentially lower render time.

**Parameters that matter most**
- `--device` / `--gpu`
- FFT size and hop

**Artifacts to listen for**
- normally identical class of artifacts to CPU; compare for parity in edge cases

---

## 16) Batch Processing Folder

**Command**
```bash
python3 pvxvoc.py stems/*.wav --preset vocal --stretch 1.05 --output-dir out/stems --overwrite
```

**Explanation**
- Applies consistent processing across many files.

**Before/After**
- Before: original folder material.
- After: each file rendered with `_pv` suffix in target folder.

**Parameters that matter most**
- wildcard expansion
- `--output-dir`
- `--overwrite`

**Artifacts to listen for**
- per-file content mismatch under one global setting

---

## 17) Pipe Multiple Tools in One Line

**Command**
```bash
python3 pvxvoc.py input.wav --stretch 1.15 --stdout \
  | python3 pvxdenoise.py - --reduction-db 8 --stdout \
  | python3 pvxdeverb.py - --strength 0.4 --stdout > cleaned.wav
```

**Explanation**
- Builds a linear chain without intermediate files.

**Before/After**
- Before: raw source.
- After: stretched + denoised + dereverberated result.

**Parameters that matter most**
- each stage intensity
- pipe ordering

**Artifacts to listen for**
- cumulative over-processing
- clipped peaks if mastering limits are omitted

---

## 18) Auto Profile + Auto Transform (Planning)

**Command**
```bash
python3 pvxvoc.py input.wav --auto-profile --auto-transform --explain-plan
```

**Explanation**
- Prints selected processing plan JSON without rendering audio.

**Before/After**
- Before: no run metadata.
- After: explicit resolved settings.

**Parameters that matter most**
- `--auto-profile-lookahead-seconds`

**Artifacts to listen for**
- n/a (plan-only mode)

---

## 19) Multi-Resolution Fusion Render

**Command**
```bash
python3 pvxvoc.py input.wav --multires-fusion --multires-ffts 1024,2048,4096 --multires-weights 0.2,0.35,0.45 --stretch 1.25 --output input_multires.wav
```

**Explanation**
- Blends multiple FFT scales to reduce single-resolution bias.

**Before/After**
- Before: single-resolution render.
- After: potentially smoother balance between transient sharpness and tonal stability.

**Parameters that matter most**
- `--multires-ffts`
- `--multires-weights`

**Artifacts to listen for**
- slight chorus/phasing if fusion weights over-emphasize conflicting scales

---

## 20) Checkpoint + Resume Long Render

**Command (first pass)**
```bash
python3 pvxvoc.py long_source.wav --preset extreme --auto-segment-seconds 0.5 --checkpoint-dir checkpoints --manifest-json reports/run_manifest.json --output long_pass1.wav
```

**Command (resume)**
```bash
python3 pvxvoc.py long_source.wav --preset extreme --auto-segment-seconds 0.5 --checkpoint-dir checkpoints --resume --manifest-json reports/run_manifest.json --manifest-append --output long_pass2.wav
```

**Explanation**
- Stores per-segment chunks and reuses completed ones.

**Before/After**
- Before: interrupted run loses progress.
- After: restart reuses cached segments.

**Parameters that matter most**
- `--auto-segment-seconds`
- `--checkpoint-dir`
- `--resume`

**Artifacts to listen for**
- segment boundary smoothness (tune crossfade)

---

## 21) Harmonize Triad Stack

**Command**
```bash
python3 pvxharmonize.py lead.wav --intervals 0,4,7 --intervals-cents 0,4,-3 --gains 1.0,0.85,0.78 --pans -0.35,0.0,0.35 --force-stereo --output-dir out --suffix _triad
```

**Explanation**
- Builds triad with subtle micro-detune offsets.

**Before/After**
- Before: single voice.
- After: harmonized stack with stereo spread.

**Parameters that matter most**
- `--intervals`
- `--intervals-cents`
- `--pans`

**Artifacts to listen for**
- beating and clutter if gains are too high

---

## 22) Timeline Warp with `pvxwarp`

**Command**
```bash
python3 pvxwarp.py drums.wav --map map_warp.csv --crossfade-ms 10 --output-dir out --suffix _warp
```

**Explanation**
- Applies time-only map to drive phrase-level groove edits.

**Before/After**
- Before: fixed groove.
- After: section-dependent push/pull timing.

**Parameters that matter most**
- map values
- `--crossfade-ms`

**Artifacts to listen for**
- timing discontinuities at map boundaries

---

## 23) Layer Harmonic and Percussive Paths

**Command**
```bash
python3 pvxlayer.py full_mix.wav \
  --harmonic-stretch 1.15 --harmonic-pitch-semitones 2 --harmonic-gain 0.95 \
  --percussive-stretch 1.00 --percussive-pitch-semitones 0 --percussive-gain 1.05 \
  --output-dir out --suffix _layered
```

**Explanation**
- Separates harmonic/percussive content and processes each path differently.

**Before/After**
- Before: one global process for all content.
- After: tonal and transient content treated independently.

**Parameters that matter most**
- harmonic/percussive stretch and gain
- HPSS kernels

**Artifacts to listen for**
- separation bleed
- phase cancellation between paths if gains are extreme

---

## 24) JSON Manifest + Automation-Friendly Logging

**Command**
```bash
python3 pvxvoc.py input.wav --stretch 1.08 --pitch -1 \
  --manifest-json reports/pvx_runs.json --manifest-append \
  --output out/input_take1.wav
```

**Explanation**
- Writes machine-readable run metadata for CI jobs, dataset pipelines, or batch monitoring.

**Before/After**
- Before: render exists but settings/provenance are not captured.
- After: output audio plus JSON metadata with resolved parameters and timing.

**Parameters that matter most**
- `--manifest-json`
- `--manifest-append`
- stable output naming convention

**Artifacts to listen for**
- not audio-specific; validate manifest integrity and path consistency

---

## 25) Speech Natural Stretch (Hybrid Transients)

**Command**
```bash
python3 pvxvoc.py speech.wav --preset vocal_studio --transient-mode hybrid --stretch 1.25 --output speech_natural.wav
```

**Explanation**
- Uses speech-focused preset plus hybrid transient handling to reduce consonant smear.

**Before/After**
- Before: original speech rate.
- After: slower pacing with cleaner consonants vs pure PV.

**Parameters that matter most**
- `--preset vocal_studio`
- `--transient-mode hybrid`
- `--transient-sensitivity`

**Artifacts to listen for**
- residual metallic tone on strong fricatives

---

## 26) Drums Safe Stretch

**Command**
```bash
python3 pvxvoc.py drums.wav --preset drums_safe --time-stretch 1.4 --output drums_safe.wav
```

**Explanation**
- Uses WSOLA-forward transient handling for percussive attacks.

**Before/After**
- Before: tight attack transients.
- After: longer groove with fewer softened hits than basic PV.

**Parameters that matter most**
- `--preset drums_safe`
- `--transient-mode wsola`

**Artifacts to listen for**
- repetitive texture on sustained cymbal tails

---

## 27) Stereo Coherent Stretch

**Command**
```bash
python3 pvxvoc.py wide_mix.wav --preset stereo_coherent --stretch 1.2 --output wide_mix_coherent.wav
```

**Explanation**
- Uses mid/side coherence coupling to reduce image wobble.

**Before/After**
- Before: original stereo image.
- After: stretched image with more stable center and side phase relation.

**Parameters that matter most**
- `--stereo-mode`
- `--coherence-strength`

**Artifacts to listen for**
- slight narrowing if coherence is set too high

---

## 28) Extreme Ambient Stretch (10-minute target)

**Command**
```bash
python3 pvxvoc.py one_shot.wav --preset extreme_ambient --target-duration 600 --output one_shot_ambient_10min.wav
```

**Explanation**
- Long-form multistage stretch preset with transient-aware blending.

**Before/After**
- Before: short source event.
- After: long ambient texture.

**Parameters that matter most**
- `--preset extreme_ambient`
- `--target-duration`
- `--max-stage-stretch`

**Artifacts to listen for**
- low-level swirl if source is very broadband

---

## 29) Pitch Shift with Formant Preservation

**Command**
```bash
python3 pvxvoc.py vocal.wav --stretch 1.0 --pitch -4 --pitch-mode formant-preserving --output vocal_down4_formant.wav
```

**Explanation**
- Separates pitch movement from formant envelope correction.

**Before/After**
- Before: original pitch center.
- After: lower notes with better vowel identity retention.

**Parameters that matter most**
- `--pitch`
- `--pitch-mode formant-preserving`
- `--formant-strength`

**Artifacts to listen for**
- hollow timbre if over-corrected

---

## 30) Hybrid Transient Mode (Manual Tuning)

**Command**
```bash
python3 pvxvoc.py source.wav \
  --transient-mode hybrid \
  --transient-sensitivity 0.65 \
  --transient-protect-ms 26 \
  --transient-crossfade-ms 8 \
  --time-stretch 1.3 \
  --output source_hybrid_tuned.wav
```

**Explanation**
- Explicitly tunes detector and stitch behavior instead of relying on preset defaults.

**Before/After**
- Before: single global stretch strategy.
- After: transient windows handled differently from steady-state regions.

**Parameters that matter most**
- `--transient-sensitivity`
- `--transient-protect-ms`
- `--transient-crossfade-ms`

**Artifacts to listen for**
- boundary roughness if crossfade is too short

---

## 31) Benchmark Command (pvx vs Rubber Band vs librosa)

**Command**
```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out --baseline benchmarks/baseline_small.json --gate
```

**Explanation**
- Runs cycle-consistency benchmark and fails if `pvx` regresses against baseline.

**Before/After**
- Before: no objective quality snapshot.
- After: JSON/Markdown benchmark report plus regression verdict.

**Parameters that matter most**
- `--quick`
- `--baseline`
- `--gate`

**Artifacts to listen for**
- n/a (objective benchmark command)
