# Lessons from Paul Koonce's PVC for pvx

Paul Koonce's PVC is useful to `pvx` for reasons deeper than a shared interest in phase-vocoder processing. PVC treated spectral transformation as a collection of focused Unix tools, persistent analysis files, reusable response data, and time-varying control functions. Its manual documented not only what each routine did, but how a musician was expected to assemble routines into a repeatable working method.

This chapter studies that method rather than proposing literal command compatibility. Historical PVC and modern `pvx` differ in file formats, performance expectations, interface conventions, and implementation architecture. The aim is to preserve the durable ideas: make representations explicit, make automation ordinary, keep commands composable, expose consequential processing choices, and distinguish stable tools from experiments.

The principal historical source is Koonce's archived PVC manual. A secondary catalog record helps establish the package's circulation in the Linux-audio community.

- [Paul Koonce, PVC manual, Princeton archive](https://www.cs.princeton.edu/courses/archive/spr99/cs325/koonce.html)
- [PVC package entry, Linux Audio applications index](https://wiki.linuxaudio.org/apps/all/pvc)

## PVC as a working environment

Koonce described PVC 1.0 as a collection of phase-vocoder signal-processing routines and shell scripts written in C for Unix. The package was built in the spirit of work by Eric Lyon and Chris Penrose and pointed readers toward F. Richard Moore and Mark Dolson for the phase-vocoder foundations. This genealogy matters because PVC was not presented as an isolated invention. It was a working layer built from inherited code, published explanations, local research, and the practical needs of a composer-programmer.

The collection ranged from a generic phase vocoder to specialized tools for time warping, filtering, companding, harmonic remapping, convolution, resonance, envelope extraction, and function reshaping. Several commands first produced an analysis or response artifact that another command consumed. Others read parameter functions from external files. Shell scripts held the many values needed for a processing pass and connected preparatory analysis to resynthesis.

This design made the command line a compositional environment. A user did not merely choose an effect and move a few controls. The user selected a representation, prepared data, transformed it, chose a resynthesis path, and preserved the decisions in a script. That sequence remains a strong model for serious offline audio work.

| PVC family | Persistent or controlling object | Musical purpose |
| --- | --- | --- |
| `plainpv`, `twarp` | spectral frames and time functions | change time, pitch, spectral position, and amplitude |
| `pvanalysis` | reusable complex analysis data | prepare a source for several later readings |
| `freqresponse`, `chordresponsemaker` | measured or synthesized response | construct spectral templates |
| `filter`, `tvfilter`, `convolver` | static or evolving spectra | filter and cross-synthesize sounds |
| `harmonizer`, `chordmapper` | frequency replication and mapping rules | construct harmonies and spectral chords |
| `ring`, `ringfilter`, `ringtvfilter` | spectral feedback and response data | create source-shaped resonances |
| `envelope`, `reshape` | scalar function streams | extract and transform automation data |

The table shows that PVC's real unit of design was not the effect name. It was the relationship among a sound, an intermediate representation, a control stream, and a resynthesis method. That distinction guides every lesson that follows.

## Lesson one: small commands can form a broad language

PVC exposed focused routines such as `plainpv`, `twarp`, `filter`, `harmonizer`, and `ringfilter`. Each routine had a recognizable purpose even when it offered many parameters. The collection became broad through composition rather than through one universal command with every possible flag.

Small commands improve more than discoverability. They establish boundaries for testing, documentation, failure reporting, and saved examples. A user can learn what kind of artifact enters a command and what kind leaves it. A developer can change one operator while preserving shared conventions for input, output, automation, and diagnostics.

`pvx` carries this idea forward through subcommands including `voc`, `freeze`, `formant`, `filter`, `retune`, `analysis`, `response`, `tvfilter`, `ringfilter`, `chordmapper`, `envelope`, and `reshape`. The `chain` command provides managed serial composition when a workflow should remain one reproducible invocation.

```bash
pvx chain vocal.wav \
  --pipeline "denoise --reduction-db 6 | voc --stretch 1.3 | formant --mode preserve" \
  --output vocal_prepared.wav
```

The lesson has a limit. A project can fragment itself into dozens of commands whose conventions disagree. Focused tools remain useful only when they share naming, map formats, channel behavior, output rules, error messages, and help structure. The command boundary should clarify an operation, not merely relocate complexity.

## Lesson two: time-varying control belongs in the foundation

PVC allowed parameters marked as functions to read headerless streams of 32-bit floating-point values. The package fitted a function stream to the requested duration and interpolated between values. CMUSIC generation tools supplied control data, while `envelope` and `reshape` could derive or transform functions. This made automation part of ordinary command-line practice rather than an optional graphical layer.

A simple linear interpretation of adjacent function samples is:

$$
u(t)=(1-\lambda)u_i+\lambda u_{i+1}
$$

where $u(t)$ is the parameter value at time $t$, $u_i$ and $u_{i+1}$ are adjacent control values, and $\lambda$ is the normalized position between their timestamps in the interval from zero to one.

The equation is modest, but its architectural consequence is large. A parameter should not need a separate implementation for scalar and time-varying use. The renderer should be able to ask for the value at the current time while the control layer decides whether that value came from a constant, a CSV column, a JSON object, a generated envelope, or a routed analysis feature.

Modern `pvx` uses named, inspectable control maps rather than PVC's anonymous binary float streams. A control trajectory can be generated, reshaped, and then applied to a render:

```bash
pvx envelope \
  --mode adsr --duration 8 --rate 20 \
  --attack-sec 0.2 --decay-sec 0.6 --sustain 1.1 --release-sec 1.0 \
  --key stretch --output controls/stretch_env.csv

pvx reshape controls/stretch_env.csv \
  --key stretch --operation resample --rate 50 \
  --interp polynomial --order 5 \
  --output controls/stretch_dense.csv

pvx voc input.wav \
  --stretch controls/stretch_dense.csv --interp linear \
  --output input_envdriven.wav
```

The durable lesson is not that every parameter should move constantly. It is that constant values and trajectories should participate in one coherent control system. A scalar is simply the least complicated trajectory.

## Lesson three: intermediate artifacts support musical memory

PVC's `pvanalysis` files separated analysis from transformation. One analysis could supply time warping, convolution, or time-varying filtering without requiring the source to be analyzed again. Static response files similarly represented measured or synthesized spectral shapes for filtering, companding, and resonance.

Persistent artifacts change the creative process. They make analysis reusable, but they also make it comparable. A user can inspect metadata, derive several responses from one analysis, archive a useful filter profile, or rerun only the stage that changed. A long render becomes a sequence of named decisions rather than one opaque calculation.

The modern equivalent must carry more context than PVC's raw working files could reliably preserve. Transform size, window, hop, sample rate, channel policy, normalization, units, software version, and integrity information determine whether stored spectral data remains meaningful. `pvx` therefore uses structured PVXAN analysis artifacts and PVXRF response artifacts.

```bash
pvx analysis create source.wav \
  --output source.pvxan.npz \
  --n-fft 4096 --win-length 4096 --hop-size 256 --window hann

pvx analysis inspect source.pvxan.npz

pvx response create source.pvxan.npz \
  --output source.pvxrf.npz \
  --method median --phase-mode mean --normalize peak --smoothing-bins 3

pvx response inspect source.pvxrf.npz
```

This is not persistence for its own sake. An artifact earns its place when it can be inspected, validated, reused, and connected to the source and parameters that produced it. Otherwise, caching merely creates another opaque file.

## Lesson four: expose the processing choice that changes the sound

Koonce's manual distinguished overlap-add and oscillator-bank resynthesis. Magnitude-only changes could preserve enough spectral structure for the faster overlap-add route. Frequency changes disturbed that structure and required sinusoidal oscillator-bank resynthesis. A threshold could suppress low-amplitude oscillators to reduce cost.

The exact division belongs to PVC's implementation, but the interface lesson remains current. When an algorithmic choice changes transients, phase coherence, noise texture, or computational cost, it should not disappear behind a vague quality label. Users need an intelligible model of why two modes sound different and when one is appropriate.

Modern processors can choose among direct inverse transforms, phase propagation, peak locking, sinusoidal models, transient replacement, waveform-aligned segments, and hybrid methods. Not every internal switch deserves a public flag. The ones that alter the audible contract should appear as stable concepts with meaningful defaults and listening guidance.

This principle also discourages false equivalence. Two modes are not necessarily points on a single better-to-worse scale. One may preserve harmonic continuity while another protects attacks. A third may embrace diffusion as a useful texture. Documentation should describe the trade rather than hide it behind words such as high quality.

## Lesson five: scripts are part of the instrument

PVC shipped shell scripts that Koonce described as a practical substitute for a graphical interface. The scripts stored numerous parameters, ran preliminary analyses, and invoked the main command. Their value was not visual polish. They turned a successful experiment into a procedure that could be read, revised, and repeated.

This remains one of the command line's strongest musical properties. A command can function as a score for a transformation. Version control can show how that score changed. Batch execution can apply it to a corpus. A collaborator can inspect the exact operation rather than infer it from a filename such as `final_really_final.wav`.

Modern workflow support should improve on raw shell history without making the process less transparent. Manifests, checkpoints, deterministic modes, explicit seeds, managed chains, and machine-readable reports can preserve intention around a command. The script remains valuable because it is both executable and legible.

## Lesson six: stable releases need an experimental edge

Koonce wrote that PVC 1.0 included routines he considered stable, useful, and moderately transparent, while other experimental routines remained outside the release. That statement captures a mature release discipline. Musical software needs room for speculative processing, but a user also needs to know which surfaces can support a composition, a class, or a production pipeline.

For `pvx`, the appropriate response is not to stop experimentation. It is to mark status honestly. Supported commands need tests, stable help, documented formats, and migration care. Experimental commands need clear warnings, disposable outputs, and freedom to change. The distinction protects both reliability and invention.

The lesson is especially important for a large command collection. Adding a name to top-level help creates an expectation. A narrow alpha surface can be more useful than a broad inventory whose behaviors and file contracts cannot yet be maintained.

## What pvx should not copy

Historical respect does not require historical reenactment. Several PVC constraints arose from its time and should not define a modern implementation.

PVC accepted NeXT/Sun 16-bit sound files and used conventions suited to the Unix systems on which it ran. Modern software should accept well-defined contemporary formats, preserve channel and sample-rate metadata, and avoid making one workstation's native representation the user's problem.

PVC's binary function streams were efficient and simple, but a headerless vector does not identify its rate, units, key, interpolation, duration policy, or provenance. Human-readable CSV and structured JSON are better interchange formats for control data, while compact binary storage can remain an internal optimization.

The package's shell scripts managed commands with very large flag sets. Scripts remain valuable, but the modern answer to parameter growth also includes presets, schemas, grouped help, validation, manifests, and focused subcommands. A script should preserve a decision, not compensate indefinitely for an interface that no one can understand.

Finally, parity should not mean reproducing every artifact or implementation limitation. `pvx` should preserve the musical operation and the inspectable workflow while using current knowledge about phase coherence, transient handling, multichannel processing, numerical precision, and testing.

## Three PVC-inspired laboratories

The following laboratories turn the historical lessons into small, reproducible `pvx` exercises. Each begins with a representation and ends with a listening question. Use short, legally obtained source files and keep the unprocessed source for comparison.

### Laboratory one: a reusable spectral imprint

This laboratory follows the PVC pattern of analysis, response construction, and time-varying filtering. The source supplies a durable spectral profile, while a separate control map determines how strongly that profile enters the target.

```bash
pvx analysis create source.wav --output source.pvxan.npz
pvx response create source.pvxan.npz \
  --output source.pvxrf.npz --method median --normalize peak
pvx envelope \
  --mode ramp --duration 12 --rate 30 \
  --start 0.0 --end 1.0 --key response_mix \
  --output controls/response_mix.csv
pvx tvfilter target.wav \
  --response source.pvxrf.npz \
  --tv-map controls/response_mix.csv --tv-key response_mix \
  --tv-interp s_curve --output target_imprinted.wav
```

Listen for the point at which the source profile becomes identifiable. Compare that moment with the control-map value. Then change only the response method or smoothing and ask whether identity arrives through broad envelope shape, narrow resonances, or noise coloration.

### Laboratory two: control data as compositional material

This laboratory treats an envelope as an independent object. Generate it once, create several transformed versions, and apply each version to the same source. The comparison isolates the musical effect of control shaping from the underlying phase-vocoder settings.

```bash
pvx envelope \
  --mode exp --duration 10 --rate 30 \
  --start 0.7 --end 1.8 --exp-curve 6 \
  --key stretch --output controls/stretch_exp.csv
pvx reshape controls/stretch_exp.csv \
  --key stretch --operation smooth --window 11 \
  --output controls/stretch_smooth.csv
pvx voc input.wav \
  --stretch controls/stretch_exp.csv \
  --output input_exp.wav
pvx voc input.wav \
  --stretch controls/stretch_smooth.csv \
  --output input_smooth.wav
```

Listen at moments where the stretch changes most rapidly. The question is not simply which render is cleaner. Ask whether smoothing improves the intended gesture or removes an articulation that made the trajectory musically clear.

### Laboratory three: spectral resonance and harmonic mapping

PVC used analyzed and synthetic responses to shape resonance, while its harmonic tools replicated or remapped spectral components. This laboratory compares those two ways of imposing pitch organization on a sound.

```bash
pvx ringfilter input.wav \
  --frequency-hz 60 --depth 0.8 --mix 0.9 \
  --resonance-hz 1200 --resonance-q 9 \
  --resonance-mix 0.4 --resonance-decay 0.2 \
  --output input_ringfilter.wav

pvx chordmapper input.wav \
  --root-hz 220 --chord min7 \
  --strength 0.75 --tolerance-cents 35 \
  --boost-db 6 --attenuation 0.45 \
  --output input_chordmapped.wav
```

The resonator emphasizes persistence near selected frequencies; the chord mapper reorganizes energy according to a harmonic model. Compare source recognition, attack clarity, tonal center, and the behavior of noise. The two renders may suggest a useful chain, but first learn what each operation contributes alone.

## A continuing design checklist

PVC's example suggests a compact checklist for future `pvx` work. The checklist should be applied before an advanced operator is considered complete.

1. Give the operator one clear musical purpose and shared input-output conventions.
2. Document scalar and time-varying control separately, including interpolation and units.
3. Make persistent artifacts self-describing, inspectable, and versioned.
4. Explain any resynthesis choice that materially changes sound or cost.
5. Provide one conservative example, one expressive example, and one warning about unsuitable material.
6. Add focused tests for the operator and an end-to-end fixture for its normal workflow.
7. Record whether the command is supported, provisional, or experimental.
8. Preserve commands, manifests, seeds, and relevant metadata for reproducible renders.

PVC's enduring lesson is that algorithms, representations, controls, and working procedures should be designed together. This chapter paraphrases Koonce's *PVC* manual in Princeton University's spring 1999 CS325 archive, with the Linux Audio index as a secondary record. No direct source-code descent from PVC to `pvx` is claimed.
