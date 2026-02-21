# pvx Example Cookbook

All commands are designed to be copy-paste runnable from the repository root.

## Quick Setup (Install + PATH)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
pvx --help
```

If `pvx` is not found, add the virtualenv `bin` directory to your path environment variable (`PATH`) (`zsh`):

```bash
printf 'export PATH="%s/.venv/bin:$PATH"\n' "$(pwd)" >> ~/.zshrc
source ~/.zshrc
pvx --help
```

No-`PATH` fallback for every example command:

```bash
python3 pvx.py <tool> ...
```

Preferred invocation:
- use unified subcommands via `pvx` (for example: `pvx voc`, `pvx freeze`, `pvx morph`)
- legacy wrappers remain valid (`python3 pvxvoc.py`, `python3 pvxfreeze.py`, etc.)
- quick conversion rule: replace `python3 pvxvoc.py` with `pvx voc`, `python3 pvxharmonize.py` with `pvx harmonize`, and so on

Assumptions:
- input files exist in your working directory
- output format defaults to input extension unless overridden
- tools using `--output-dir` and `--suffix` write inferred filenames

## Starter Path (Run These First)

If you are new to `pvx`, run these in order:

```bash
pvx voc input.wav --stretch 1.2 --output step1_stretch.wav
pvx voc input.wav --stretch 1.0 --pitch -3 --output step2_pitch.wav
pvx voc input.wav --stretch 1.0 --pitch -3 --pitch-mode formant-preserving --output step3_pitch_formant.wav
pvx voc drums.wav --preset drums_safe --stretch 1.25 --output step4_drums.wav
pvx voc mix.wav --stretch 1.2 --stereo-mode mid_side_lock --coherence-strength 0.9 --output step5_stereo.wav
```

Expected audible progression:
- `step1`: timing changes only
- `step2`: pitch changes only
- `step3`: pitch change with improved vocal timbre stability
- `step4`: better attack retention on percussive material
- `step5`: more stable stereo image during stretch

## Use-Case Index (By Theme)

| Theme | Start here | Typical tools |
| --- | --- | --- |
| Speech intelligibility and timing | 1, 2, 25, 61 | `pvx voc` |
| Musical pitch and formants | 3, 4, 5, 29, 37, 48 | `pvx voc`, `pvx retune` |
| Extreme ambient and drones | 6, 28, 45, 50, 65 | `pvx voc`, `pvx freeze` |
| Transient-safe processing | 26, 30, 32, 33, 52 | `pvx voc` |
| Stereo/multichannel coherence | 27, 34, 35, 57, 58 | `pvx voc` |
| Morphing and hybrid sound design | 9, 23, 49, 55 | `pvx morph`, `pvx layer`, `pvx harmonize` |
| Cleanup/restoration | 11, 12, 50, 53, 64 | `pvx denoise`, `pvx deverb` |
| Automation and reproducibility | 13, 14, 24, 31, 44, 47, 63 | `pvx conform`, `pvx voc`, benchmarks |
| Transform and backend research | 40, 41, 62 | `pvx voc` |
| Live-style piping workflows | 17, 46, 59 | `pvx voc` + downstream tools |

---

## 1) Slow Down Speech

**Command**
```bash
pvx voc speech.wav --preset vocal --stretch 1.35 --output speech_slow.wav
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
pvx voc speech.wav --preset vocal --stretch 0.85 --output speech_fast.wav
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
pvx voc vocal.wav --stretch 1.0 --pitch 3 --output vocal_up3.wav
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
pvx voc vocal.wav --stretch 1.0 --pitch -4 --pitch-mode formant-preserving --output vocal_down4_formant.wav
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
pvx voc vocal.wav --stretch 1.2 --pitch -2 --pitch-mode formant-preserving --output vocal_stretch_formant.wav
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
pvx voc one_shot.wav --preset ambient --target-duration 600 --output one_shot_10min.wav
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
pvx freeze guitar_chord.wav --freeze-time 0.45 --duration 12 --output-dir out --suffix _freeze
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
pvx voc vocal_note.wav --target-f0 440 --pitch-mode formant-preserving --output vocal_A440.wav
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
pvx morph source_a.wav source_b.wav --alpha 0.35 --output morph_35.wav --overwrite
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
pvx unison synth.wav --voices 7 --detune-cents 18 --width 1.2 --dry-mix 0.15 --output-dir out --suffix _uni7
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
pvx denoise field.wav --noise-seconds 0.5 --reduction-db 10 --smooth 7 --output-dir out --suffix _den
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
pvx deverb room_voice.wav --strength 0.55 --decay 0.90 --floor 0.12 --output-dir out --suffix _dry
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
pvx voc phrase.wav --pitch-map map_warp.csv --output phrase_map.wav
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
pvx pitch-track A.wav | pvx voc B.wav --pitch-follow-stdin --pitch-conf-min 0.75 --output B_follow.wav
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
pvx voc mix.wav --gpu --stretch 1.12 --output mix_gpu.wav
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
pvx voc stems/*.wav --preset vocal --stretch 1.05 --output-dir out/stems --overwrite
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
pvx voc input.wav --stretch 1.15 --stdout \
  | pvx denoise - --reduction-db 8 --stdout \
  | pvx deverb - --strength 0.4 --stdout > cleaned.wav
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
pvx voc input.wav --auto-profile --auto-transform --explain-plan
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
pvx voc input.wav --multires-fusion --multires-ffts 1024,2048,4096 --multires-weights 0.2,0.35,0.45 --stretch 1.25 --output input_multires.wav
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
pvx voc long_source.wav --preset extreme --auto-segment-seconds 0.5 --checkpoint-dir checkpoints --manifest-json reports/run_manifest.json --output long_pass1.wav
```

**Command (resume)**
```bash
pvx voc long_source.wav --preset extreme --auto-segment-seconds 0.5 --checkpoint-dir checkpoints --resume --manifest-json reports/run_manifest.json --manifest-append --output long_pass2.wav
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
pvx harmonize lead.wav --intervals 0,4,7 --intervals-cents 0,4,-3 --gains 1.0,0.85,0.78 --pans -0.35,0.0,0.35 --force-stereo --output-dir out --suffix _triad
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
pvx warp drums.wav --map map_warp.csv --crossfade-ms 10 --output-dir out --suffix _warp
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
pvx layer full_mix.wav \
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
pvx voc input.wav --stretch 1.08 --pitch -1 \
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
pvx voc speech.wav --preset vocal_studio --transient-mode hybrid --stretch 1.25 --output speech_natural.wav
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
pvx voc drums.wav --preset drums_safe --time-stretch 1.4 --output drums_safe.wav
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
pvx voc wide_mix.wav --preset stereo_coherent --stretch 1.2 --output wide_mix_coherent.wav
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
pvx voc one_shot.wav --preset extreme_ambient --target-duration 600 --output one_shot_ambient_10min.wav
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
pvx voc vocal.wav --stretch 1.0 --pitch -4 --pitch-mode formant-preserving --output vocal_down4_formant.wav
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
pvx voc source.wav \
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

---

## 32) Transient Reset Mode for Cleaner Attacks

**Command**
```bash
pvx voc drums.wav --stretch 1.22 --transient-mode reset --transient-sensitivity 0.62 --output drums_reset.wav
```

**Explanation**
- Applies phase resets around detected onsets while leaving steady regions in PV flow.

**Before/After**
- Before: transient-rich grooves may soften at larger stretches.
- After: snare/kick attack definition is usually sharper.

**Parameters that matter most**
- `--transient-mode reset`
- `--transient-sensitivity`

**Artifacts to listen for**
- onset clicks if sensitivity is too high

---

## 33) WSOLA-Only Transient Handling

**Command**
```bash
pvx voc percussion.wav --stretch 1.35 --transient-mode wsola --transient-protect-ms 35 --output percussion_wsola.wav
```

**Explanation**
- Prioritizes time-domain overlap/similarity handling for transient regions.

**Before/After**
- Before: attack smearing in strictly spectral paths.
- After: stronger transient solidity with different texture in sustains.

**Parameters that matter most**
- `--transient-mode wsola`
- `--transient-protect-ms`

**Artifacts to listen for**
- grain boundaries if crossfade/protect windows are mismatched

---

## 34) Mid/Side Stereo Coherence Stretch

**Command**
```bash
pvx voc stereo_mix.wav --stretch 1.18 --stereo-mode mid_side_lock --coherence-strength 0.9 --output stereo_ms_lock.wav
```

**Explanation**
- Preserves stereo image geometry by constraining M/S phase behavior.

**Before/After**
- Before: image may wobble on sustained sections.
- After: center and width usually feel more stable.

**Parameters that matter most**
- `--stereo-mode mid_side_lock`
- `--coherence-strength`

**Artifacts to listen for**
- over-constrained width if coherence is set too high

---

## 35) Reference-Channel Lock for Multichannel Content

**Command**
```bash
pvx voc multichannel.wav --stretch 1.12 --stereo-mode ref_channel_lock --ref-channel 0 --coherence-strength 0.85 --output multichannel_ref_lock.wav
```

**Explanation**
- Uses one channel as phase anchor to reduce inter-channel decorrelation.

**Before/After**
- Before: surround image drift on strong harmonics.
- After: tighter channel relationship across processing.

**Parameters that matter most**
- `--stereo-mode ref_channel_lock`
- `--ref-channel`
- `--coherence-strength`

**Artifacts to listen for**
- channel dominance bias if reference channel is atypical

---

## 36) Irrational Pitch Ratio Input

**Command**
```bash
pvx voc tone.wav --stretch 1.0 --pitch-ratio "2^(1/12)" --output tone_up_1semitone_expr.wav
```

**Explanation**
- Uses expression parsing for irrational ratio input directly on CLI.

**Before/After**
- Before: original fundamental.
- After: +1 equal-tempered semitone.

**Parameters that matter most**
- `--pitch-ratio` expression
- `--stretch 1.0`

**Artifacts to listen for**
- none specific for small shifts; monitor formant drift on voice

---

## 37) Just-Ratio Pitch Input

**Command**
```bash
pvx voc tone.wav --stretch 1.0 --pitch-ratio 3/2 --output tone_perfect_fifth.wav
```

**Explanation**
- Uses just-intonation ratio syntax for exact rational interval control.

**Before/After**
- Before: root pitch.
- After: perfect fifth above (3:2).

**Parameters that matter most**
- `--pitch-ratio 3/2`

**Artifacts to listen for**
- beating vs reference if mixed with equal-tempered material

---

## 38) N-TET CSV Map (19-TET) with `pvxconform`

**Command**
```bash
pvx conform lead.wav --map maps/map_19tet.csv --output-dir out --suffix _19tet
```

**Explanation**
- Applies segmentwise pitch/time map from CSV using 19-tone equal temperament ratios.

**Before/After**
- Before: 12-TET phrases.
- After: 19-TET trajectory according to map.

**Parameters that matter most**
- `--map`
- CSV `pitch_ratio` values

**Artifacts to listen for**
- boundary roughness if map segments are too short

---

## 39) Just-Intonation CSV Conform Map

**Command**
```bash
pvx conform choir.wav --map maps/map_just_intonation.csv --output-dir out --suffix _ji
```

**Explanation**
- Retunes with segment ratios such as `5/4`, `6/5`, `7/4`, `9/8`.

**Before/After**
- Before: equal-tempered intonation center.
- After: just-intonation harmonic color.

**Parameters that matter most**
- map segment durations
- map `pitch_ratio`

**Artifacts to listen for**
- abrupt color shifts between unrelated just ratios

---

## 40) Transform A/B: FFT vs DCT

**Command**
```bash
pvx voc source.wav --stretch 1.08 --transform fft --output fft_out.wav
pvx voc source.wav --stretch 1.08 --transform dct --output dct_out.wav
```

**Explanation**
- Compares complex-phase Fourier path against real cosine basis path.

**Before/After**
- Before: one baseline source.
- After: two different transform-character outputs.

**Parameters that matter most**
- `--transform`
- identical stretch/pitch settings for fair comparison

**Artifacts to listen for**
- phase character and transient envelope differences

---

## 41) CZT for Non-Power-of-Two Frame Setup

**Command**
```bash
pvx voc source.wav --transform czt --n-fft 1536 --win-length 1536 --hop-size 384 --stretch 1.1 --output czt_1536.wav
```

**Explanation**
- Uses chirp-z backend for a non-power-of-two frame strategy.

**Before/After**
- Before: default FFT path.
- After: alternate numerical behavior for same macro task.

**Parameters that matter most**
- `--transform czt`
- `--n-fft`, `--win-length`, `--hop-size`

**Artifacts to listen for**
- subtle high-frequency texture differences vs FFT

---

## 42) Full Mastering Chain in One Render

**Command**
```bash
pvx voc mix.wav --stretch 1.03 --target-lufs -14 --compressor-threshold-db -18 --compressor-ratio 2.2 --limiter-threshold -0.9 --soft-clip-level 0.96 --output mix_mastered.wav
```

**Explanation**
- Demonstrates integrated DSP + mastering path for one-step render.

**Before/After**
- Before: un-leveled output with higher crest variance.
- After: controlled loudness with safety limiting/clipping.

**Parameters that matter most**
- `--target-lufs`
- compressor and limiter thresholds
- `--soft-clip-level`

**Artifacts to listen for**
- pumping, flattened transients, clip harshness

---

## 43) Safety Limiter + Hard Clip Guard

**Command**
```bash
pvx voc loud_source.wav --stretch 1.0 --limiter-threshold -1.0 --hard-clip-level 0.98 --output loud_safe.wav
```

**Explanation**
- Adds a limiter stage followed by a hard guard ceiling.

**Before/After**
- Before: possible overs and intersample spikes.
- After: bounded peaks with predictable ceiling.

**Parameters that matter most**
- `--limiter-threshold`
- `--hard-clip-level`

**Artifacts to listen for**
- transient flattening if limiter is overworked

---

## 44) Plan-Only Diagnostic (No Render)

**Command**
```bash
pvx voc source.wav --auto-profile --auto-transform --manifest-json plan.json --dry-run --explain-plan
```

**Explanation**
- Produces resolved strategy without writing output audio.

**Before/After**
- Before: no visibility into automatic decision path.
- After: JSON and console explanation of chosen settings.

**Parameters that matter most**
- `--auto-profile`
- `--auto-transform`
- `--dry-run`

**Artifacts to listen for**
- n/a (planning command)

---

## 45) Checkpointed Long Render Workflow

**Command**
```bash
pvx voc long_source.wav --target-duration 1200 --auto-segment-seconds 0.5 --checkpoint-dir .pvx_ckpt --output long_out.wav
pvx voc long_source.wav --target-duration 1200 --auto-segment-seconds 0.5 --checkpoint-dir .pvx_ckpt --resume --output long_out.wav
```

**Explanation**
- First command starts segmented checkpointed render; second resumes if interrupted.

**Before/After**
- Before: full rerender after interruption.
- After: resumed render from saved chunk state.

**Parameters that matter most**
- `--checkpoint-dir`
- `--auto-segment-seconds`
- `--resume`

**Artifacts to listen for**
- chunk-boundary discontinuities if segmenting is too coarse

---

## 46) Multi-Tool Pipe with Streaming I/O

**Command**
```bash
pvx voc input.wav --stretch 1.1 --stdout \
| pvx denoise - --reduction-db 8 --stdout \
| pvx deverb - --strength 0.35 --output cleaned_chain.wav
```

**Explanation**
- Streams PCM data through multiple tools in one line.

**Before/After**
- Before: separate intermediate files.
- After: one-pass chained processing.

**Parameters that matter most**
- upstream `--stdout`
- downstream `-` stdin input

**Artifacts to listen for**
- cumulative artifact stacking from multiple stages

---

## 47) Batch Tree Processing with `find` + `xargs`

**Command**
```bash
find sessions -name "*.wav" -print0 | xargs -0 -I{} pvx voc "{}" --stretch 1.05 --output-dir renders --suffix _x105 --overwrite
```

**Explanation**
- Applies a consistent transform to an entire tree of WAV files.

**Before/After**
- Before: manual per-file invocation.
- After: reproducible batch render set.

**Parameters that matter most**
- stable suffix conventions
- overwrite policy

**Artifacts to listen for**
- content classes that need per-file presets instead of one global recipe

---

## 48) Vocal Retune by Scale/Root (`pvxretune`)

**Command**
```bash
pvx retune vocal.wav --scale minor --root A --f0-min 70 --f0-max 1000 --output-dir out --suffix _retune_amin
```

**Explanation**
- Tracks monophonic F0 and snaps segments to requested tonal set.

**Before/After**
- Before: natural pitch drift.
- After: scale-constrained melody line.

**Parameters that matter most**
- `--scale`, `--root`
- `--f0-min`, `--f0-max`

**Artifacts to listen for**
- octave errors on noisy consonants

---

## 49) Harmonize Then Widen in One Pipeline

**Command**
```bash
pvx harmonize lead.wav --intervals 0,4,7 --gains 1,0.75,0.65 --stdout \
| pvx unison - --voices 5 --detune-cents 10 --width 1.0 --output harmonized_wide.wav
```

**Explanation**
- Builds harmony stack first, then adds unison detuned width.

**Before/After**
- Before: single melodic line.
- After: harmonized and spatially widened texture.

**Parameters that matter most**
- harmonic intervals/gains
- unison voice count and detune

**Artifacts to listen for**
- dense masking in low-mid region

---

## 50) Ambient Macro-Chain (Stretch + Cleanup + Loudness)

**Command**
```bash
pvx voc seed.wav --preset extreme_ambient --target-duration 600 --stdout \
| pvx denoise - --reduction-db 4 --smooth 9 --stdout \
| pvx deverb - --strength 0.25 --stdout \
| pvx voc - --stretch 1.0 --target-lufs -18 --output ambient_final.wav
```

**Explanation**
- Demonstrates long-form ambient generation plus light restoration and final loudness targeting.

**Before/After**
- Before: short seed sample.
- After: long-form ambient render with controlled output level.

**Parameters that matter most**
- ambient preset stretch controls
- cleanup strength values
- final `--target-lufs`

**Artifacts to listen for**
- accumulated haze from over-processing

---

## 51) Broadcast-Style Speech Finishing (Stretch + Loudness + Safety)

**Command**
```bash
pvx voc narration.wav --preset vocal_studio --stretch 1.08 --target-lufs -16 --limiter-threshold -1.0 --soft-clip-level 0.98 --hard-clip-level 0.995 --output narration_finished.wav
```

**Explanation**
- Applies subtle timing expansion and a conservative mastering finish in one pass.

**Before/After**
- Before: untreated narration.
- After: slightly slower pacing with controlled final loudness and peaks.

**Parameters that matter most**
- `--stretch`
- `--target-lufs`
- limiter/clip safety thresholds

**Artifacts to listen for**
- pumping from aggressive limiting
- clipped consonant edges if clip levels are too low

---

## 52) Extreme Stretch with Checkpointed Recovery

**Command**
```bash
pvx voc short_source.wav --preset extreme_ambient --target-duration 900 --auto-segment-seconds 0.25 --checkpoint-dir checkpoints/short_source --output short_source_15min.wav
```

**Explanation**
- Runs a very large render with small recoverable segments and persistent checkpoints.

**Before/After**
- Before: short source gesture.
- After: long ambient evolution with resumable rendering.

**Parameters that matter most**
- `--target-duration`
- `--auto-segment-seconds`
- `--checkpoint-dir`

**Artifacts to listen for**
- timbral drift from repeated stage boundaries

---

## 53) Room Recording Rehab (De-Reverb Then Stretch)

**Command**
```bash
pvx deverb room_take.wav --strength 0.45 --stdout \
| pvx voc - --preset vocal_studio --stretch 1.15 --output room_take_rehab.wav
```

**Explanation**
- Reduces reverberant smear first, then performs moderate timing stretch on cleaner material.

**Before/After**
- Before: dense room reflections.
- After: clearer direct signal with slower timing.

**Parameters that matter most**
- `pvx deverb --strength`
- `pvx voc --stretch`

**Artifacts to listen for**
- metallic tails from over-dereverb
- new blur if stretch is too large

---

## 54) Scale-Quantized Retune Followed by Unison Widening

**Command**
```bash
pvx retune lead.wav --scale major --root D --output-dir out --suffix _retuned \
&& pvx unison out/lead_retuned.wav --voices 5 --detune-cents 9 --width 0.9 --output-dir out --suffix _retuned_unison
```

**Explanation**
- First constrains melody to key, then adds controlled chorus/unison width.

**Before/After**
- Before: dry monophonic lead.
- After: tuned and widened lead texture.

**Parameters that matter most**
- retune scale/root
- unison voices and detune

**Artifacts to listen for**
- tuning jumps on weak F0 regions
- phasey image when width/detune are too high

---

## 55) Morph Then Freeze for Hybrid Pads

**Command**
```bash
pvx morph bells.wav choir.wav --alpha 0.5 --output bells_choir_morph.wav \
&& pvx freeze bells_choir_morph.wav --freeze-time 0.8 --duration 20 --output-dir out --suffix _frozen
```

**Explanation**
- Builds a hybrid timbre first, then captures a stable harmonic moment as a sustained pad.

**Before/After**
- Before: two independent sources.
- After: one blended frozen harmonic texture.

**Parameters that matter most**
- morph `--alpha`
- freeze capture time and duration

**Artifacts to listen for**
- static texture if freeze time is poorly chosen

---

## 56) CUDA Attempt with Deterministic CPU Fallback Workflow

**Command**
```bash
pvx voc mix.wav --device cuda --stretch 1.12 --output mix_cuda.wav || pvx voc mix.wav --device cpu --stretch 1.12 --output mix_cpu.wav
```

**Explanation**
- Tries GPU first, then falls back to CPU using the same core settings.

**Before/After**
- Before: original mix timing.
- After: consistent stretched output regardless of device path.

**Parameters that matter most**
- `--device`
- stretch/quality controls shared between both paths

**Artifacts to listen for**
- no expected quality change between CPU and CUDA modes

---

## 57) 5.1/Multichannel Reference-Lock Stretch

**Command**
```bash
pvx voc surround.wav --stretch 1.1 --stereo-mode ref_channel_lock --ref-channel 2 --coherence-strength 0.95 --output surround_lock.wav
```

**Explanation**
- Anchors phase evolution to a chosen reference channel to reduce inter-channel drift.

**Before/After**
- Before: original multichannel phase relationships.
- After: stretched output with better preserved channel coupling.

**Parameters that matter most**
- `--stereo-mode ref_channel_lock`
- `--ref-channel`
- `--coherence-strength`

**Artifacts to listen for**
- narrowed width if coherence is overly strong for decorrelated ambience

---

## 58) Mid/Side Lock with Formant-Preserving Pitch Shift

**Command**
```bash
pvx voc stereo_vocal.wav --stretch 1.0 --pitch 2 --pitch-mode formant-preserving --stereo-mode mid_side_lock --coherence-strength 0.9 --output stereo_vocal_up2_lock.wav
```

**Explanation**
- Combines vocal formant handling with stereo image preservation.

**Before/After**
- Before: natural stereo vocal.
- After: upshifted vocal with tighter image stability.

**Parameters that matter most**
- `--pitch`, `--pitch-mode`
- `--stereo-mode`, `--coherence-strength`

**Artifacts to listen for**
- center blur if coherence strength is too low

---

## 59) Confidence-Gated External Pitch Control via Pipe

**Command**
```bash
pvx pitch-track A.wav | pvx voc B.wav --pitch-follow-stdin --pitch-conf-min 0.75 --pitch-lowconf-mode hold --pitch-map-crossfade-ms 20 --output B_pitch_follow.wav
```

**Explanation**
- Tracks pitch trajectory from `A.wav`, then applies only high-confidence pitch control to `B.wav`.

**Before/After**
- Before: independent `B.wav` pitch contour.
- After: `B.wav` follows trusted sections of `A.wav` pitch shape.

**Parameters that matter most**
- `--pitch-conf-min`
- `--pitch-lowconf-mode`
- `--pitch-map-crossfade-ms`

**Artifacts to listen for**
- contour lag if crossfade is too long
- pitch jitter if confidence threshold is too low

---

## 60) Microtonal Just-Ratio Map Playback

**Command**
```bash
pvx voc mono_line.wav --pitch-map maps/just_ratios.csv --pitch-map-crossfade-ms 15 --output mono_line_ji.wav
```

**Explanation**
- Applies time-varying just-ratio pitch targets from CSV.

**Before/After**
- Before: fixed tuning.
- After: trajectory-constrained microtonal tuning.

**Parameters that matter most**
- `--pitch-map`
- `--pitch-map-crossfade-ms`

**Artifacts to listen for**
- step audibility if map density is too sparse

---

## 61) Speech Clarity Pass (Identity Lock + Conservative Window/Hop)

**Command**
```bash
pvx voc lecture.wav --stretch 1.18 --phase-locking identity --n-fft 2048 --hop-size 256 --output lecture_clear_stretch.wav
```

**Explanation**
- Uses a quality-focused speech profile manually tuned for lower phase blur.

**Before/After**
- Before: original lecture pace.
- After: slower and clearer for note-taking.

**Parameters that matter most**
- `--phase-locking identity`
- `--n-fft`, `--hop-size`

**Artifacts to listen for**
- mild transient smear if hop becomes too large

---

## 62) Transform Research A/B (FFT vs Hartley)

**Command**
```bash
pvx voc source.wav --stretch 1.1 --transform fft --output source_fft.wav \
&& pvx voc source.wav --stretch 1.1 --transform hartley --output source_hartley.wav
```

**Explanation**
- Produces matched renders to compare transform-domain behavior.

**Before/After**
- Before: one reference source.
- After: two algorithmic variants for blind listening tests.

**Parameters that matter most**
- `--transform`
- all other parameters held constant

**Artifacts to listen for**
- fine-grain differences in high-frequency texture

---

## 63) Multi-Resolution Fusion With Explicit Weights

**Command**
```bash
pvx voc texture.wav --stretch 1.4 --multires-fusion --multires-ffts 1024,2048,4096 --multires-weights 0.2,0.35,0.45 --output texture_multires_weighted.wav
```

**Explanation**
- Fuses multiple FFT scales with explicit weighting rather than implicit defaults.

**Before/After**
- Before: single-scale spectral processing.
- After: blended resolution emphasizing low-frequency stability.

**Parameters that matter most**
- `--multires-ffts`
- `--multires-weights`

**Artifacts to listen for**
- diffuse attacks if large-FFT weight is too dominant

---

## 64) Conservative Field-Restoration Chain

**Command**
```bash
pvx denoise field.wav --reduction-db 5 --smooth 7 --stdout \
| pvx deverb - --strength 0.3 --stdout \
| pvx voc - --stretch 1.0 --target-lufs -20 --limiter-threshold -1.5 --output field_restored.wav
```

**Explanation**
- Applies light denoise + dereverb and ends with conservative loudness normalization.

**Before/After**
- Before: noisy reverberant field capture.
- After: cleaner and level-aligned output.

**Parameters that matter most**
- denoise reduction amount
- dereverb strength
- final loudness/limiter settings

**Artifacts to listen for**
- chirpy residuals from over-denoise

---

## 65) Reproducible Ambient Session With Manifest Logging

**Command**
```bash
pvx voc seed.wav --preset extreme_ambient --target-duration 600 --manifest-json sessions/ambient_manifest.json --manifest-append --output ambient_take.wav
```

**Explanation**
- Writes run metadata into a JSON manifest for repeatable, auditable experiments.

**Before/After**
- Before: short seed sample.
- After: long ambient render plus machine-readable run record.

**Parameters that matter most**
- `--manifest-json`
- `--manifest-append`
- chosen preset/stretch strategy

**Artifacts to listen for**
- broad smearing from overly aggressive stretch factors

---

## 66) Managed One-Line Chain (No Manual Pipe Wiring)

**Command**
```bash
pvx chain source.wav --pipeline "voc --stretch 1.2 | formant --mode preserve --formant-shift-ratio 1.0" --output source_chain.wav
```

**Explanation**
- Runs serial stages with managed intermediate files.
- Useful when you want multi-stage processing without writing long shell pipelines.

**Before/After**
- Before: single source file.
- After: time-stretched + formant-preserved render in one managed command.

**Parameters that matter most**
- `--pipeline` stage order and settings
- per-stage quality controls (inside the pipeline string)

**Artifacts to listen for**
- cumulative coloration when too many strong stages are chained

---

## 67) Chunked Stream Wrapper + Output Policy Controls

**Command**
```bash
pvx stream source.wav --output source_stream.wav --chunk-seconds 0.2 --time-stretch 2.0 --bit-depth 24 --dither tpdf --dither-seed 7 --metadata-policy sidecar
```

**Explanation**
- Uses `pvx stream` stateful mode (default) for chunked long renders with continuity context.
- Adds deterministic output policy controls and metadata sidecar emission.

**Before/After**
- Before: original source.
- After: chunked long-form render with explicit output-depth/dither policy and sidecar metadata.

**Parameters that matter most**
- `--chunk-seconds`
- `--bit-depth`
- `--dither`, `--dither-seed`
- `--metadata-policy`

**Artifacts to listen for**
- over-short chunks may increase boundary coloration

Legacy compatibility mode:
```bash
pvx stream source.wav --mode wrapper --output source_stream_wrapper.wav --chunk-seconds 0.2 --time-stretch 2.0
```

---

## 68) Dynamic Stretch via Direct CSV Flag Input

**Command**
```bash
pvx voc source.wav --stretch controls/stretch.csv --interp linear --output source_dyn_stretch.wav
```

**Example `controls/stretch.csv`**
```csv
time_sec,value
0.0,1.0
1.0,1.2
2.0,1.6
3.0,2.0
```

**Explanation**
- Uses a time-varying control-rate signal directly on `--stretch` (no legacy `--pitch-map` wrapper needed).
- Interpolates control points with `--interp linear` (default mode).

**Before/After**
- Before: fixed global stretch ratio.
- After: continuously changing stretch trajectory over time.

**Parameters that matter most**
- `--stretch <csv/json>`
- `--interp`
- `--order` (if polynomial)

**Artifacts to listen for**
- excessive local tempo curvature from over-aggressive control points
- boundary coloration if your control points force rapid ratio changes

---

## 69) Dynamic Pitch Ratio via JSON + Polynomial Interpolation

**Command**
```bash
pvx voc vocal.wav --pitch-shift-ratio controls/pitch_curve.json --interp polynomial --order 3 --output vocal_dyn_pitch.wav
```

**Example `controls/pitch_curve.json`**
```json
{
  "points": [
    {"time_sec": 0.0, "value": 1.0},
    {"time_sec": 0.8, "value": 1.122462048},
    {"time_sec": 1.6, "value": 1.334839854},
    {"time_sec": 2.4, "value": 1.0}
  ]
}
```

**Explanation**
- Drives pitch from a JSON control-rate signal attached directly to `--pitch-shift-ratio`.
- `--interp polynomial --order 3` fits a smooth polynomial trajectory through points.

**Before/After**
- Before: static transposition.
- After: continuous pitch contour over time.

**Parameters that matter most**
- `--pitch-shift-ratio <csv/json>`
- `--interp`
- `--order`

**Artifacts to listen for**
- over/undershoot from high-order polynomial fitting
- formant drift if large excursions are used without formant-preserving mode

---

## 70) Stairstep (Sample-and-Hold) Control Mode

**Command**
```bash
pvx voc phrase.wav --stretch controls/step_stretch.csv --interp none --output phrase_step_control.wav
```

**Example `controls/step_stretch.csv`**
```csv
start_sec,end_sec,value
0.0,0.5,1.0
0.5,1.0,1.25
1.0,1.5,0.9
1.5,2.0,1.1
```

**Explanation**
- Uses stairstep control (`--interp none`) with piecewise constant parameter regions.
- Best when you explicitly want quantized timeline regions.

**Before/After**
- Before: smooth or fixed trajectory.
- After: discrete region-based time changes.

**Parameters that matter most**
- `--interp none`
- segment boundaries
- control `value`

**Artifacts to listen for**
- audible discontinuities if regions are too short or jumps are too large

---

## 71) Dynamic Analysis Resolution (N-FFT + Hop Size Maps)

**Command**
```bash
pvx voc source.wav --n-fft controls/nfft.csv --hop-size controls/hop.csv --stretch 1.2 --output source_dyn_resolution.wav
```

**Example `controls/nfft.csv`**
```csv
time_sec,value
0.0,1024
2.0,2048
4.0,4096
```

**Example `controls/hop.csv`**
```csv
time_sec,value
0.0,128
2.0,256
4.0,512
```

**Explanation**
- Time-varying spectral resolution can trade transient precision and tonal stability across song sections.
- Useful for research workflows where you want section-dependent analysis behavior.

**Before/After**
- Before: one fixed analysis resolution for the whole render.
- After: section-adaptive STFT resolution.

**Parameters that matter most**
- `--n-fft <csv/json>`
- `--hop-size <csv/json>`
- `--win-length` (static or dynamic)

**Artifacts to listen for**
- timbral shifts at parameter-transition boundaries
- mismatched hop/window trajectories causing roughness

---

## 72) Stateful Stream Mode with Dynamic Stretch CSV

**Command**
```bash
pvx stream input.wav --output output_stream.wav --chunk-seconds 0.2 --stretch controls/stretch.csv --interp linear
```

**Example `controls/stretch.csv`**
```csv
time_sec,value
0.0,1.0
3.0,1.4
6.0,2.0
```

**Explanation**
- Uses chunked stateful stream processing with the same control-file interface as `pvx voc`.
- The stretch value is sampled from the interpolated trajectory for each processed chunk.

**Before/After**
- Before: one global stretch factor.
- After: evolving stretch profile over time, while staying in stream mode.

**Parameters that matter most**
- `--chunk-seconds`
- `--stretch <csv/json>`
- `--interp`

**Artifacts to listen for**
- chunk-boundary texture changes if `--chunk-seconds` is too short
- abrupt pacing changes from steep control curves
