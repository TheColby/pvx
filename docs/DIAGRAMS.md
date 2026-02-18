# pvx Visual Diagrams (ASCII)

## 1) STFT Analysis / Resynthesis

```text
Input x[n]
  |
  v
+----------------------------+
| Frame + Window (N samples) |
+----------------------------+
  |
  v
+----------------------------+
| FFT -> X_t[k]              |
| magnitude + phase          |
+----------------------------+
  |
  v
+----------------------------+
| Spectral Processing         |
| (stretch/pitch/retune/etc.) |
+----------------------------+
  |
  v
+----------------------------+
| IFFT -> x_t[n]             |
+----------------------------+
  |
  v
+----------------------------+
| Overlap-Add Reconstruction |
+----------------------------+
  |
  v
Output y[n]
```

## 2) Phase Locking vs Free Phase

```text
Per-bin phase trajectory over frames

Free phase propagation:
frame:   t0   t1   t2   t3   t4
bin k:   o----o----o----o----o
bin k+1: o--o-----o--o-----o---  (relative drift)

Identity phase locking:
frame:   t0   t1   t2   t3   t4
peak bin:   O----O----O----O----O
neighbor:   o----o----o----o----o  (locked to local peak reference)

Effect:
- Free phase can increase blur/chorus artifacts.
- Locked phase preserves local waveform structure around peaks.
```

## 3) pvx Processing Pipeline

```text
CLI args
  |
  +--> parser + validation
  |
  +--> optional profile/preset/auto-plan
  |
  +--> audio read
  |
  +--> per-channel process
  |      |
  |      +--> STFT/vocoder core
  |      +--> optional multires fusion
  |      +--> optional map/segment logic
  |
  +--> optional mastering chain
  |
  +--> write output / stdout
```

## 4) Segment CSV Flow Example

```text
CSV rows
(start_sec, end_sec, stretch, pitch_ratio, confidence)
  |
  v
parse + validate
  |
  +--> confidence policy (hold/unity/interp)
  |
  +--> optional ratio smoothing
  |
  +--> fill timeline gaps with defaults
  |
  +--> segment loop
         |
         +--> process each segment
         +--> optional checkpoint cache
  |
  +--> crossfade segment boundaries
  |
  v
final assembled output
```

## 5) Mastering Chain Block Diagram

```text
Processed signal
  |
  v
[Expander] -> [Compressor] -> [Compander]
  |
  v
[Normalization / Target LUFS]
  |
  v
[Limiter]
  |
  v
[Soft Clip]
  |
  v
[Hard Clip Safety]
  |
  v
Final output
```

Notes:
- Each stage is optional and controlled by CLI flags.
- Stage order affects audible behavior and should be tuned conservatively.
