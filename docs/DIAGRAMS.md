# pvx Diagram Atlas

This document provides expanded architecture and DSP flow diagrams for pvx.
Diagrams are a mix of Mermaid (GitHub-rendered) and ASCII (terminal-friendly).

## 1) End-to-End pvxvoc Flow

```mermaid
flowchart TD
    A["CLI args"] --> B["Parse + validate"]
    B --> C["Preset / auto-profile / auto-transform"]
    C --> D["Input decode"]
    D --> E["Per-channel processing loop"]
    E --> F["STFT analysis"]
    F --> G["Phase-vocoder core"]
    G --> H["Transient logic (off/reset/hybrid/wsola)"]
    H --> I["Stereo coherence mode"]
    I --> J["ISTFT synthesis"]
    J --> K["Optional mastering chain"]
    K --> L["Metrics tables"]
    L --> M["Write output / stdout"]
```

## 2) STFT Analysis / Synthesis Timeline

```text
signal: x[n] -------------------------------------------------------------->

windows:
          |----N----|
               |----N----|
                    |----N----|
hop:          <---Ha--->

analysis:
  frame t0 -> FFT -> X0[k]
  frame t1 -> FFT -> X1[k]
  frame t2 -> FFT -> X2[k]

processing:
  magnitude trajectory + phase trajectory updates

synthesis:
  IFFT(frame0), IFFT(frame1), IFFT(frame2) + overlap-add with Hs
```

```mermaid
flowchart LR
    X["x[n]"] --> W["Window + frame"]
    W --> S["FFT"]
    S --> P["Spectral processing"]
    P --> I["IFFT"]
    I --> O["Overlap-add"]
    O --> Y["y[n]"]
```

## 3) Phase Propagation and Locking

```mermaid
flowchart TD
    A["Input phase φ_t[k]"] --> B["Compute wrapped delta"]
    B --> C["Instantaneous frequency estimate"]
    C --> D["Synthesis phase accumulation"]
    D --> E{"Phase locking mode"}
    E -->|off| F["Free-bin phase"]
    E -->|identity| G["Lock neighbors to local peak"]
    F --> H["Output phase"]
    G --> H
```

```text
Frame index:    t0      t1      t2      t3
Peak bin:       O-------O-------O-------O
Neighbor free:  o--o------o---o-----o-----   (drift can increase blur)
Neighbor lock:  o-------o-------o-------o     (relative peak shape preserved)
```

## 4) Hybrid Transient Engine

```mermaid
flowchart LR
    A["Input audio"] --> B["Transient features"]
    B --> C["Onset/region mask"]
    C --> D{"Region type"}
    D -->|steady| E["Phase-vocoder path"]
    D -->|transient| F["WSOLA path"]
    E --> G["Crossfade stitch"]
    F --> G
    G --> H["Output audio"]
```

```text
time ------------------------------------------------------------>
mask 000000111100000011110000000
     steady^^^trans^^steady^^trans^^steady
```

## 5) Transient Feature Stack

```mermaid
flowchart TD
    A["STFT magnitudes"] --> B["Spectral flux"]
    A --> C["High-frequency content"]
    A --> D["Broadbandness"]
    B --> E["Weighted onset score"]
    C --> E
    D --> E
    E --> F["Debounce + min-duration merge"]
    F --> G["Transient region list"]
```

## 6) WSOLA Similarity Search

```text
reference analysis frame:      [========R========]

candidate search window:
                         [c0][c1][c2][c3][c4][c5][c6]
                           \    \    \    \    \    \
similarity score:           s0   s1   s2   s3   s4   s5

choose argmax(similarity) -> best offset -> overlap-add with crossfade
```

```mermaid
flowchart TD
    A["Current synthesis position"] --> B["Search candidate offsets"]
    B --> C["Compute similarity metric"]
    C --> D["Pick best offset"]
    D --> E["Windowed overlap-add"]
```

## 7) Stereo / Multichannel Coherence Modes

```mermaid
flowchart LR
    A["Input channels"] --> B{"stereo-mode"}
    B -->|independent| C["Per-channel PV"]
    B -->|mid_side_lock| D["L/R -> M/S -> constrained PV -> L/R"]
    B -->|ref_channel_lock| E["Lock increments to reference channel"]
    C --> F["Output channels"]
    D --> F
    E --> F
```

```text
independent:      L and R evolve freely
mid_side_lock:    preserve center/side coherence envelope
ref_channel_lock: keep phase relation anchored to one channel
```

## 8) Microtonal Map Processing

```mermaid
flowchart TD
    A["CSV rows start/end/stretch/pitch_ratio/confidence"] --> B["Parse + validate"]
    B --> C["Confidence policy"]
    C --> D["Optional smoothing"]
    D --> E["Fill timeline gaps"]
    E --> F["Segment render loop"]
    F --> G["Boundary crossfades"]
    G --> H["Assembled output"]
```

## 9) Extreme Stretch: Multistage Strategy

```mermaid
flowchart LR
    A["Requested stretch ratio"] --> B{"Ratio > stage threshold?"}
    B -->|no| C["Single stage"]
    B -->|yes| D["Split into stage factors"]
    D --> E["Stage 1 render"]
    E --> F["Stage 2 render"]
    F --> G["..."]
    G --> H["Final stage output"]
```

```text
example total stretch = 600x
600 = 5 * 5 * 4.8 * 5
render sequentially with safer stage factors
```

## 10) Transform Backend Decision Path

```mermaid
flowchart TD
    A["Requested transform"] --> B{"Backend available?"}
    B -->|yes| C["Use requested transform"]
    B -->|no| D["Raise clear CLI error"]
    C --> E["One-sided bins + vocoder path"]
```

```text
preferred default: fft
research/verification: dft/czt/dct/dst/hartley
```

## 11) Mastering Chain Block Diagram

```mermaid
flowchart LR
    A["Processed signal"] --> B["Expander (optional)"]
    B --> C["Compressor (optional)"]
    C --> D["Compander (optional)"]
    D --> E["Target loudness / normalize"]
    E --> F["Limiter"]
    F --> G["Soft clip"]
    G --> H["Hard clip guard"]
    H --> I["Final output"]
```

## 12) Metrics Printing Path

```mermaid
flowchart TD
    A["Input summary metrics"] --> C["ASCII summary table"]
    B["Input vs output compare metrics"] --> D["ASCII comparison table"]
    C --> E["Console output (unless --silent)"]
    D --> E
```

## 13) Benchmark Runner + CI Gate

```mermaid
flowchart TD
    A["Generate tiny dataset"] --> B["Run pvx cycle consistency"]
    B --> C["Run librosa baseline"]
    C --> D{"Rubber Band installed?"}
    D -->|yes| E["Run Rubber Band cycle consistency"]
    D -->|no| F["Mark RB unavailable"]
    E --> G["Compute objective metrics"]
    F --> G
    G --> H["Write report.json + report.md"]
    H --> I{"Gate enabled?"}
    I -->|yes| J["Compare to baseline JSON"]
    J --> K["Pass or fail CI"]
    I -->|no| L["Informational run"]
```

## 14) Pipe Chaining Pattern

```text
python3 pvxvoc.py input.wav --stdout \
| python3 pvxdenoise.py - --stdout \
| python3 pvxdeverb.py - --output final.wav

rule:
- producer uses --stdout
- consumer reads '-' as stdin
```

## 15) Checkpoint / Resume State Machine

```mermaid
stateDiagram-v2
    [*] --> Start
    Start --> ScanCheckpoint
    ScanCheckpoint --> LoadChunk: chunk exists and --resume
    ScanCheckpoint --> RenderChunk: chunk missing
    LoadChunk --> NextChunk
    RenderChunk --> SaveChunk
    SaveChunk --> NextChunk
    NextChunk --> ScanCheckpoint: more chunks
    NextChunk --> Assemble: done
    Assemble --> [*]
```

## 16) Troubleshooting Decision Tree

```mermaid
flowchart TD
    A["Artifact heard"] --> B{"What kind?"}
    B -->|transient smear| C["Increase transient protection / hybrid mode"]
    B -->|phasey blur| D["Enable identity phase lock, adjust window/hop"]
    B -->|stereo wobble| E["Set stereo-mode lock + coherence strength"]
    B -->|loudness pumping| F["Relax mastering thresholds"]
    C --> G["Re-render short excerpt"]
    D --> G
    E --> G
    F --> G
```

## 17) Data Product Map

```text
markdown docs  -> docs/*.md
html docs      -> docs/html/*.html
pdf bundle     -> docs/pvx_documentation.pdf
benchmark json -> benchmarks/out/report.json
benchmark md   -> benchmarks/out/report.md
```

## 18) Window/Overlap Tradeoff Map

```mermaid
flowchart TD
    A["Choose window type"] --> B{"Need strong sidelobe suppression?"}
    B -->|yes| C["Use Blackman-Harris / Nuttall family"]
    B -->|no| D["Use Hann/Hamming family"]
    C --> E["Increase overlap for smoother continuity"]
    D --> F["Moderate overlap often sufficient"]
    E --> G["Check transient sharpness"]
    F --> G
```

```text
larger window + higher overlap  -> smoother/cleaner sustained content
smaller window + lower overlap  -> sharper time localization, rougher bass resolution
```

## 19) Phase Reset vs Hybrid Boundary Behavior

```text
time -------------------------------------------------------->
transient mask:      000011100000001110000

mode=reset:
phase tracks   ----->|reset|--------->|reset|------->

mode=hybrid:
pv steady      -----[PV]-----     -----[PV]-----
wsola transient      [WSOLA]           [WSOLA]
stitch               <xfade>           <xfade>
```

## 20) Multi-Resolution Fusion Fan-In

```mermaid
flowchart LR
    A["Frame"] --> B["FFT 1024 path"]
    A --> C["FFT 2048 path"]
    A --> D["FFT 4096 path"]
    B --> E["Weighted fusion"]
    C --> E
    D --> E
    E --> F["Unified spectrum for synthesis"]
```

## 21) Pitch-Map Confidence Gate

```mermaid
flowchart TD
    A["Pitch control sample (ratio, confidence)"] --> B{"confidence >= threshold?"}
    B -->|yes| C["Apply target ratio"]
    B -->|no + hold| D["Hold previous trusted ratio"]
    B -->|no + neutral| E["Fallback to ratio=1.0"]
    C --> F["Crossfade-smoothed control stream"]
    D --> F
    E --> F
    F --> G["Pitch/time engine"]
```

## 22) Microtonal CSV Interpolation Flow

```mermaid
flowchart LR
    A["CSV rows (time, pitch_ratio)"] --> B["Parse + sort by time"]
    B --> C{"Gap between rows?"}
    C -->|small| D["Linear interpolation"]
    C -->|large| E["Hold + guard crossfade"]
    D --> F["Per-frame ratio stream"]
    E --> F
    F --> G["Render segments"]
```

## 23) Audio Metrics Table Generation

```text
input.wav  -> basic metrics (sr,ch,duration,peak,rms,crest,dc,zcr,bw95,clip%)
output.wav -> basic metrics (same fields)
paired cmp -> compare metrics (snr,si-sdr,lsd,mod-dist,envelope corr,...)

console:
  table A: input summary
  table B: output summary
  table C: input vs output (input / output / delta)
```

## 24) Quality-First Optimization Loop

```mermaid
flowchart TD
    A["Target use-case"] --> B["Pick quality-safe preset"]
    B --> C["Render short excerpt"]
    C --> D["Listen + inspect metrics"]
    D --> E{"Artifacts acceptable?"}
    E -->|no| F["Adjust quality controls first"]
    F --> C
    E -->|yes| G["Tune runtime/perf knobs"]
    G --> H["Full render"]
```

## 25) Benchmark Metric Taxonomy

```mermaid
flowchart TD
    A["Metric families"] --> B["Spectral: LSD, convergence, modulation"]
    A --> C["Time/transient: onset F1, smear, attack error"]
    A --> D["Level/loudness: LUFS, true-peak, crest"]
    A --> E["Speech/perceptual: STOI/ESTOI/PESQ proxies"]
    A --> F["Spatial: ILD/ITD/IPD drift, coherence"]
    B --> G["Regression gate"]
    C --> G
    D --> G
    E --> G
    F --> G
```

## 26) Documentation Build Pipeline

```mermaid
flowchart LR
    A["docs/*.md"] --> B["scripts_generate_html_docs.py"]
    B --> C["docs/html/*.html"]
    C --> D["scripts_generate_docs_pdf.py"]
    D --> E["docs/pvx_documentation.pdf"]
```

Use this atlas together with:
- `docs/MATHEMATICAL_FOUNDATIONS.md` for equation-level details
- `docs/EXAMPLES.md` for copy-paste command recipes
