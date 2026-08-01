<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# AI Research and Data Augmentation with pvx

`pvx augment` provides deterministic, manifest-driven audio augmentation for machine-learning workflows.

Primary use cases:
- automatic speech recognition (ASR) robustness experiments
- music information retrieval (MIR) augmentation studies
- self-supervised learning (SSL) contrastive view generation
- ablation studies where exact augmentation parameters must be replayed

## Core Command

```bash
pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337
```

What this does:
- resolves input files/globs/directories
- renders `N` deterministic variants per input (`--variants-per-input`)
- writes per-variant metadata to:
  - `augment_manifest.jsonl`
  - `augment_manifest.csv`

Manifest tool companion:

```bash
pvx augment-manifest validate aug_out/augment_manifest.jsonl --strict
```

## Intent Profiles

| Intent | Typical target | Behavior |
| --- | --- | --- |
| `asr_robust` | Speech pipelines | Mild time/pitch perturbation with conservative vocal/formant defaults |
| `mir_music` | Music analytics | Moderate timing/pitch/spectral variation suited for musical content |
| `ssl_contrastive` | Contrastive representation learning | Wider perturbation envelope for view diversity while remaining reproducible |

## Reproducibility Controls

| Option | Meaning |
| --- | --- |
| `--seed` | Global deterministic seed for all sampled parameters |
| `--split train,val,test` | Deterministic split assignment ratios written into manifest |
| `--split-mode` | Split strategy: `random`, `label_balanced`, `speaker_balanced` |
| `--labels-csv` | Metadata table with `path`/`label`/`speaker` fields for balanced split modes |
| `--grouping` | Split assignment grouping strategy (`stem-prefix` or `none`) |
| `--group-separator` | Prefix separator for grouping (default: `__`) |
| `--pair-mode` | Pair-view mode: `off` or `contrastive2` |
| `--label-policy` | Perturbation policy: `allow_alter` or `preserve` |
| `--policy` | JSON policy file with reproducible defaults and parameter bounds |
| `--workers` | Parallel render workers with deterministic seed behavior |
| `--resume` | Skip previously rendered outputs using existing manifests/output files |
| `--append-manifest` | Merge with existing manifests rather than replace |
| `--dry-run` | Plan outputs and write manifests without rendering audio |
| `--manifest-jsonl` / `--manifest-csv` | Explicit manifest output paths |

## Manifest Fields

Each JSON Lines (JSONL) row contains:

- `source_path`
- `output_path`
- `intent`
- `seed`
- `split` (`train`, `val`, `test`)
- `group_key` (split-group identifier)
- `status` (`planned`, `rendered`, or `error:<code>`)
- `source_sha256`, `output_sha256`
- `params` object, including:
  - `stretch`
  - `pitch`
  - `preset`
  - `window`
  - `transform`
  - `formant_strength`
  - `transient_sensitivity`
  - `target_lufs`
- optional `audit` object:
  - `duration_sec`
  - `peak_dbfs`
  - `rms_dbfs`
  - `clip_pct`
  - `zcr`

CSV manifest includes the same essential fields in tabular form.

## Example Workflows

### 1) Speech Robustness Set

```bash
pvx augment corpus/speech/*.wav \
  --output-dir data_aug/speech \
  --variants-per-input 6 \
  --intent asr_robust \
  --grouping stem-prefix \
  --group-separator "__" \
  --split 0.8,0.1,0.1 \
  --seed 1337
```

### 2) Music Information Retrieval Set

```bash
pvx augment corpus/music/**/*.wav \
  --output-dir data_aug/music \
  --variants-per-input 4 \
  --intent mir_music \
  --split-mode label_balanced \
  --labels-csv labels.csv \
  --split 0.7,0.2,0.1 \
  --seed 2026
```

### 3) Contrastive Planning-Only Pass

```bash
pvx augment corpus/*.wav \
  --output-dir data_aug/plan_only \
  --variants-per-input 3 \
  --intent ssl_contrastive \
  --pair-mode contrastive2 \
  --dry-run \
  --seed 42
```

### 4) With explicit manifest paths

```bash
pvx augment corpus/*.wav \
  --output-dir data_aug/run_a \
  --variants-per-input 5 \
  --intent asr_robust \
  --label-policy preserve \
  --policy policies/asr_policy.json \
  --manifest-jsonl reports/run_a_manifest.jsonl \
  --manifest-csv reports/run_a_manifest.csv \
  --seed 9001
```

### 5) Resume and append

```bash
pvx augment corpus/*.wav \
  --output-dir data_aug/run_resume \
  --variants-per-input 5 \
  --intent mir_music \
  --resume \
  --append-manifest \
  --seed 9001
```

### 6) Manifest merge + stats

```bash
pvx augment-manifest merge \
  reports/run_a_manifest.jsonl reports/run_b_manifest.jsonl \
  --output-jsonl reports/merged_manifest.jsonl \
  --output-csv reports/merged_manifest.csv

pvx augment-manifest stats reports/merged_manifest.jsonl
```

## Policy File Template

```json
{
  "augment": {
    "intent": "asr_robust",
    "variants_per_input": 4,
    "split": "0.8,0.1,0.1",
    "split_mode": "speaker_balanced",
    "grouping": "stem-prefix",
    "group_separator": "__",
    "pair_mode": "off",
    "label_policy": "preserve",
    "workers": 4,
    "device": "cpu",
    "output_format": "wav",
    "bounds": {
      "stretch": [0.94, 1.08],
      "pitch": [-1.0, 1.0],
      "formant_strength": [0.60, 0.92],
      "transient_sensitivity": [0.50, 0.75],
      "target_lufs": [-24.0, -16.0]
    },
    "choices": {
      "window": ["hann", "hamming"],
      "transform": ["fft", "dft"],
      "preset": ["vocal_studio"]
    }
  }
}
```

## Why Augmentation Works

Audio augmentation changes the training distribution without requiring a new recording session. Its purpose is not to make a corpus look large on disk. Its purpose is to expose the model to variations that should not change the desired decision, while retaining the acoustic evidence that should change that decision. This distinction separates principled augmentation from arbitrary signal damage.

For supervised learning, a useful starting objective is:

$$
\mathcal{R}_{\mathrm{aug}}(\theta)
=
\mathbb{E}_{(x,y)\sim\mathcal{D}}
\mathbb{E}_{\tau\sim q(\tau)}
\left[\ell\left(f_{\theta}(\tau(x)),g_{\tau}(y)\right)\right].
$$

where $\mathcal{D}$ is the empirical training distribution, $x$ is an audio example, $y$ is its target, $\tau$ is a sampled transformation, $q(\tau)$ is the augmentation policy, $g_{\tau}$ is the corresponding target transformation, $f_{\theta}$ is the model with parameters $\theta$, and $\ell$ is the training loss. For a label-preserving transform, $g_{\tau}(y)=y$. For an equivariant task, such as pitch estimation under transposition, $g_{\tau}$ must alter the target.

The two expectations express two different sources of variation. Sampling examples from $\mathcal{D}$ exposes the model to the recorded corpus. Sampling transforms from $q$ exposes it to a designed neighborhood around each recording. If that neighborhood resembles plausible deployment conditions, augmentation can improve robustness. If it crosses semantic boundaries or exaggerates irrelevant artifacts, the model learns the wrong invariances.

## Invariance, Equivariance, and Invalid Transformations

Every augmentation policy should begin with a statement about labels. An invariant label stays unchanged after transformation. An equivariant label changes in a known way. An invalid transformation destroys or obscures the evidence needed to define the target. Treating all three cases as label preserving is one of the most common causes of disappointing augmentation experiments.

The following examples show how the same operation can have different meanings in different tasks.

| Transformation | Label-preserving example | Label-changing or invalid example |
| --- | --- | --- |
| Time stretch | Speaker identity under mild stretching | Beat timestamps and event boundaries must move |
| Pitch shift | Many environmental sound classes | Fundamental-frequency and key labels must transpose |
| Gain change | ASR transcript or genre label | Absolute loudness prediction target must change |
| Polarity reversal | Most monaural classification | Some waveform-generation and spatial tasks may be sensitive |
| Channel swap | Stereo event classification when orientation is irrelevant | Left-right localization labels must swap |
| Room simulation | Speech transcript | Dry-target enhancement pairs require careful target design |
| Additive noise | Transcript at intelligible SNR | Clean-signal reconstruction target remains unmodified |
| Cropping | Clip-level class present throughout | Event labels outside the crop must be removed |

A policy review should classify every transform before training begins. For a scalar target $y$ and transform parameter $a$, the transformed target can be written as:

$$
y' = g_a(y).
$$

where $y'$ is the target attached to the transformed audio, $g_a$ is the task-specific label mapping, and $a$ is the sampled transform parameter. For pitch in hertz shifted by $s$ semitones, $g_s(y)=y2^{s/12}$. For an event time stretched by factor $r$, $g_r(y)=ry$ when the implementation defines $r$ as output duration divided by input duration. Verify the rate convention before transforming annotations.

## Designing the Augmentation Distribution

Choosing a transform class is only the beginning. The probability of applying it, the parameter distribution, its order relative to other transforms, and correlations among parameters determine the distribution the model actually sees. A policy that says only "add noise and reverb" is underspecified.

For a transform with application probability $p$, the effective training distribution is a mixture:

$$
p_{\mathrm{train}}(z)
=
(1-p)p_{\mathrm{clean}}(z)
+
p\int p(z\mid x,\tau)q(\tau)\,d\tau.
$$

where $z$ is the example presented to the model, $p_{\mathrm{clean}}$ is the clean-example distribution, $p(z\mid x,\tau)$ describes the result of applying $\tau$ to source $x$, and $q(\tau)$ is the parameter distribution. Keeping $p<1$ retains clean anchors. Setting every transform to $p=1$ can unintentionally remove the original domain from training.

Uniform sampling is convenient but not automatically perceptually uniform. A uniform distribution in amplitude does not produce a uniform distribution in decibels. A uniform distribution in frequency ratio does not produce a uniform distribution in semitones. Define ranges in the coordinate system that corresponds to the intended perceptual or operational variation.

The signal-to-noise ratio used by additive-noise transforms is:

$$
\mathrm{SNR}_{\mathrm{dB}}
=
10\log_{10}\left(\frac{P_x}{P_n}\right),
$$

where $P_x$ is the average signal power and $P_n$ is the average added-noise power over the measured region. Silence handling matters because estimating $P_x$ over long silent margins can produce an augmentation that is much louder than expected during active speech. Compare active-region and whole-file measurements when validating an SNR policy.

The following policy questions are worth answering explicitly before a large render:

- What deployment variation is each transform intended to approximate?
- Which labels are invariant, which are transformed, and which examples become invalid?
- What percentage of training examples remains clean?
- Are transform parameters sampled in perceptually meaningful coordinates?
- Are parameter combinations independent, correlated, or mutually exclusive?
- Does transform order reproduce a plausible acoustic process?
- Which held-out conditions will test the policy's intended benefit?

## Transform Order Is Part of the Model

Audio operations generally do not commute. Adding noise and then applying room simulation reverberates both signal and noise. Applying room simulation and then adding noise models a reverberant source recorded with noise introduced closer to the microphone or electronics. Both can be useful, but they represent different worlds.

Let $A$ and $B$ be two transforms. In general:

$$
A(B(x)) \neq B(A(x)).
$$

where $x$ is the source waveform, $A$ is one augmentation operation, and $B$ is another. The inequality can arise from nonlinear processing, time variation, resampling, clipping, or simply from a different physical interpretation.

A defensible speech pipeline often follows an approximate causal sequence: source variation, room propagation, microphone or channel coloration, transmission codec, and sensor noise. Creative MIR and self-supervised policies may deliberately violate that order, but the violation should be intentional rather than accidental.

```python
from pvx.augment import (
    AddNoise,
    CodecDegradation,
    EQPerturber,
    GainPerturber,
    Pipeline,
    RoomSimulator,
)

causal_recording_model = Pipeline(
    [
        GainPerturber(gain_db=(-4, 4), p=0.8),
        RoomSimulator(rt60_range=(0.15, 0.9), wet_range=(0.1, 0.55), p=0.5),
        EQPerturber(n_bands=4, gain_db_range=(-5, 5), p=0.4),
        CodecDegradation(codec="random", p=0.2),
        AddNoise(snr_db=(8, 35), noise_type="pink", p=0.6),
    ],
    seed=1337,
)
```

## Phase-Vocoder Augmentation as a Special Case

Time stretching and pitch shifting are unusually powerful because they alter musical or linguistic structure while leaving many other properties recognizable. They are also unusually easy to misuse. A phase vocoder can introduce transient blur, vertical phase incoherence, stereo-image changes, and formant displacement. These artifacts may become shortcuts that a model associates with a class or data source.

The synthesis hop in a basic time-stretch schedule is:

$$
H_s = rH_a,
$$

where $H_s$ is the synthesis hop, $H_a$ is the analysis hop, and $r$ is the requested output-duration ratio. The ratio changes frame placement, while phase propagation attempts to preserve sinusoidal continuity. At large $r$, attacks occupy more output time unless transient handling resets or reroutes them.

Pitch shifting by $s$ semitones uses the frequency ratio:

$$
\rho = 2^{s/12},
$$

where $\rho$ is the resampling or spectral frequency ratio and $s$ is the signed semitone shift. A pitch-label target must be multiplied by $\rho$. A key label must be transposed modulo twelve. A speaker-identity label may remain unchanged only for a narrow shift range that does not undermine identity.

Phase-vocoder augmentation deserves its own audit. Listen for artifacts at policy extrema, not only at median settings. Compare several source classes, including isolated attacks, voiced speech, dense mixtures, bass instruments, stereo ambience, and quiet tails. If a model can detect the processing engine more easily than the intended acoustic variation, the policy may improve training loss while reducing real-world validity.

The following settings are conservative starting points rather than universal truths:

| Task | Time range | Pitch range | Principal caution |
| --- | --- | --- | --- |
| ASR | 0.94 to 1.08 | -1 to +1 semitone | Preserve intelligibility and speaker cues |
| Speaker recognition | 0.97 to 1.03 | 0 or very narrow | Avoid changing perceived identity |
| Music tagging | 0.85 to 1.18 | -2 to +2 semitones | Check tag invariance and transient quality |
| Beat tracking | 0.75 to 1.30 | Usually 0 | Transform beat timestamps with duration |
| Pitch tracking | Mild or none | Task-dependent | Transform frequency labels exactly |
| Contrastive SSL | Wider, paired by policy | Wider, paired by policy | Avoid trivially identifiable processing signatures |

## Severity, Coverage, and Curriculum

More severe augmentation is not necessarily better. Mild policies may fail to cover deployment conditions, while extreme policies can increase label noise and optimization difficulty. Validation error often follows a shallow basin rather than a monotonic curve. A curriculum can begin near the clean distribution and gradually widen the parameter ranges.

\begin{figure}[H]
\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[x=0.8cm,y=0.75cm]
\begin{scope}
\draw[->] (0,0) -- (5.4,0) node[right] {augmentation severity (normalized)};
\draw[->] (0,0) -- (0,3.6) node[above] {validation error (normalized)};
\draw[thick] plot[smooth] coordinates {(0.2,2.8) (1.2,1.8) (2.3,1.25) (3.2,1.35) (4.2,2.0) (5.0,3.0)};
\node[align=center] at (2.5,0.65) {useful region};
\end{scope}
\begin{scope}[xshift=9cm]
\draw[->] (0,0) -- (5.4,0) node[right] {training progress (\%)};
\draw[->] (0,0) -- (0,3.6) node[above] {maximum severity (normalized)};
\draw[thick] plot[smooth] coordinates {(0.2,0.55) (1.2,0.75) (2.2,1.25) (3.2,2.0) (4.2,2.65) (5.0,3.0)};
\node[align=center] at (3.1,1.0) {curriculum schedule};
\end{scope}
\end{tikzpicture}
}
\caption{Two policy-design concepts. The left graph motivates a measured severity search. The right graph shows one possible curriculum that widens the allowed range during training.}
\end{figure}

A linear severity curriculum can be written as:

$$
a(e)=a_0+\frac{\min(e,E)}{E}(a_1-a_0),
$$

where $a(e)$ is the maximum augmentation severity at epoch $e$, $a_0$ is the initial severity, $a_1$ is the final severity, and $E$ is the number of epochs over which the range expands. The parameter $a$ may control one range or a coordinated family of ranges. Measure whether curriculum scheduling helps rather than assuming it will.

Coverage should be checked empirically from manifests. If a transform is configured with probability $p$ but conditional failures, clipping rejection, or incompatible inputs reduce its realized use, the actual rate can differ from $p$. Count rendered parameter combinations and compare them with the intended policy.

## Determinism and Seed Architecture

Reproducibility requires more than recording one global seed. Parallel workers, shuffled data loaders, resumed jobs, and changes in corpus ordering can all perturb a naive random-number stream. A stable design derives each example seed from immutable identifiers.

A conceptual seed derivation is:

$$
s_i = \operatorname{Hash}(s_0, u_i, v_i, k) \bmod 2^{31},
$$

where $s_i$ is the child seed for one transform, $s_0$ is the experiment seed, $u_i$ is the stable source identifier, $v_i$ is the variant number, and $k$ is the transform position or name. A cryptographic hash is useful for stable distribution, but the exact hash and serialization must be recorded if independent implementations need bitwise agreement.

The pvx manifest should be treated as an experimental artifact. Preserve it beside model checkpoints and evaluation reports. Hashes establish which waveforms were used, while parameters explain how they were made. A seed without transform versions, source hashes, and policy bounds is not enough to recreate a dataset years later.

The following provenance fields are especially valuable:

- pvx version and source commit
- operating system, architecture, and relevant dependency versions
- source and output hashes
- policy-file hash
- global seed and derived per-example seed
- transform order and sampled parameters
- engine, preset, window, and phase-handling settings
- render status and error code
- label transformation version
- corpus license and source identifier

## Split Hygiene and Family Leakage

Augmented siblings must not cross data splits. If one source recording appears in training while a stretched or noisy sibling appears in validation, the validation score measures memorization of source-specific details as well as generalization. File-name differences do not make examples independent.

Assign splits before augmentation and group all descendants by a stable source identity. For speech, the group may need to be the speaker rather than the utterance. For music, it may need to be the composition, performance, session, album, or artist depending on the scientific claim. For bioacoustics, recording site and date can matter as much as species.

The leakage unit should match the strongest nuisance correlation that the model could exploit. A speaker-independent ASR experiment requires speaker-disjoint evaluation. A cover-song retrieval experiment requires composition-aware splits. A room-robustness study may require room-disjoint evaluation even when speakers overlap.

```bash
pvx augment corpus/train/*.wav \
  --output-dir derived/train \
  --variants-per-input 5 \
  --split-mode speaker_balanced \
  --labels-csv metadata/train_labels.csv \
  --grouping stem-prefix \
  --group-separator "__" \
  --seed 1337
```

After rendering, inspect the manifest for group overlap rather than trusting naming conventions. A strict validation step should fail the pipeline when one `group_key` occurs in more than one split.

```bash
pvx augment-manifest validate derived/train/augment_manifest.jsonl --strict
pvx augment-manifest stats derived/train/augment_manifest.jsonl
```

## Online, Offline, and Hybrid Augmentation

Offline augmentation renders finite variants before training. It provides auditability, stable throughput, and easy listening inspection at the cost of storage and limited diversity. Online augmentation samples transformations during training. It provides effectively unbounded variation but consumes training compute and makes exact replay more demanding. A hybrid system precomputes expensive deterministic transforms and applies inexpensive random transforms in the data loader.

The approximate offline storage requirement is:

$$
B \approx MVSRC,
$$

where $B$ is the number of stored bytes, $M$ is the number of source clips, $V$ is the number of variants per source, $S$ is mean duration in seconds, $R$ is sample rate, and $C$ is bytes per multichannel sample frame. Container overhead is omitted. Lossless compression changes the realized total but not the basic scaling.

The approximate online compute fraction is:

$$
\eta_{\mathrm{aug}}=\frac{T_{\mathrm{aug}}}{T_{\mathrm{load}}+T_{\mathrm{aug}}+T_{\mathrm{model}}},
$$

where $\eta_{\mathrm{aug}}$ is the fraction of step time consumed by augmentation, $T_{\mathrm{load}}$ is data-loading time, $T_{\mathrm{aug}}$ is transform time, and $T_{\mathrm{model}}$ is forward and backward model time. Measure this fraction with realistic worker counts and storage, because a policy that starves the accelerator may cost more than an extra offline corpus copy.

The following division is often practical:

| Precompute offline | Sample online |
| --- | --- |
| High-quality phase-vocoder stretching | Gain perturbation |
| Expensive impulse-response convolution | Cropping and time shift |
| Codec round trips | Lightweight additive noise |
| Source separation or stem processing | Spectral masks |
| Audited label-equivariant renders | Mixup when targets support it |

## Contrastive and Self-Supervised Views

Contrastive learning needs two views that preserve the identity or content relation the objective is meant to capture. If the views are nearly identical, the task can be trivial. If they no longer share the intended content, the positive pair becomes false. The policy must sit between those failures.

For two views $x_a=\tau_a(x)$ and $x_b=\tau_b(x)$, a common normalized temperature-scaled loss is:

$$
\ell_i=-\log
\frac{\exp(\operatorname{sim}(z_i,z_i^+)/T)}
{\sum_{j\neq i}\exp(\operatorname{sim}(z_i,z_j)/T)},
$$

where $z_i$ is the embedding of one view, $z_i^+$ is the embedding of its paired positive view, $z_j$ are candidate embeddings in the denominator, $\operatorname{sim}$ is a similarity function, and $T$ is the temperature. Augmentation defines which information the loss encourages the representation to discard.

Independent view sampling is not always appropriate. Two aggressive crops may contain no common event. Two large pitch shifts may alter identity for a speaker objective. Coordinated policies can enforce minimum temporal overlap, shared source regions, bounded relative pitch, or one weak and one strong view.

```bash
pvx augment corpus/unlabeled/*.wav \
  --output-dir derived/ssl_views \
  --variants-per-input 2 \
  --intent ssl_contrastive \
  --pair-mode contrastive2 \
  --manifest-jsonl reports/ssl_views.jsonl \
  --seed 4242
```

Audit positive pairs by listening to random, median-severity, and maximum-severity examples. Also train a small classifier to predict augmentation type from embeddings. Very high transform predictability can reveal that the representation is organizing itself around processing signatures rather than useful content.

## Paired and Structured Targets

Enhancement, separation, source localization, beat tracking, alignment, and sequence labeling require coordinated changes to audio and targets. The safest implementation represents an example as a structured record rather than applying waveform transforms in isolation.

For a waveform $x(t)$ with event intervals $[a_j,b_j]$, a constant time stretch $r$ gives:

$$
x'(t)=x(t/r),\qquad [a'_j,b'_j]=[ra_j,rb_j].
$$

where $x'(t)$ is the stretched waveform, $r$ is output duration divided by input duration, and $[a'_j,b'_j]$ is the transformed interval for event $j$. Cropping by an offset $c$ then maps retained boundaries to $a''_j=a'_j-c$ and $b''_j=b'_j-c$, with clipping to the crop interval.

For enhancement training, the noisy input and clean target should usually remain sample aligned. If room simulation or resampling changes latency, record and compensate for that latency before calculating a sample-domain loss. A transform applied only to the input represents corruption. A geometric transform applied to both input and target represents a new aligned pair. Confusing these cases teaches delay, filtering, or pitch errors as if they were desired output.

## Task-Specific Policy Reasoning

Different tasks reward different invariances. A single global "audio augmentation" recipe cannot be optimal across ASR, music transcription, speaker recognition, enhancement, and generative modeling. The policy should follow the task's target semantics and deployment environment.

### Automatic speech recognition

ASR usually treats moderate room, noise, channel, gain, and speaking-rate variation as label preserving. Large pitch changes, severe time compression, aggressive cropping, or low SNR can make the transcript ambiguous. Measure word error rate separately on clean, noisy, reverberant, accented, and channel-degraded subsets so one aggregate number does not hide regressions.

### Speaker and paralinguistic modeling

Speaker identity, age, emotion, pathology, and prosody depend on cues that ordinary ASR may discard. Formant shifts and pitch transforms can directly alter those cues. Prefer environmental and channel augmentation first, then introduce voice transformations through controlled ablations. Report performance by demographic and recording-condition strata when metadata and consent permit.

### Music information retrieval

Music tags differ in transposition and tempo invariance. Instrument labels may survive moderate pitch shifts, while key labels must transpose and tuning labels may not survive. Genre labels may tolerate modest tempo changes, while beat and downbeat timestamps must move. Chord roots transpose but chord qualities usually remain. Use task-specific target adapters rather than one file-level label policy.

### Pitch and melody estimation

Pitch shifting is valuable precisely because it requires exact label equivariance. Keep a high-precision mapping between sample time and target frames, account for resampling conventions, and verify frequencies after rendering. Formant-preserving shifts may be preferable for voice realism, but the model should not learn that one formant algorithm uniquely identifies shifted examples.

### Enhancement and source separation

The clean target defines what counts as distortion. Additive noise, reverberation, clipping, and codecs may be applied to the input alone when removal is desired. Gain and timing changes may need to affect both input and target. For source separation, transforms applied independently to stems increase mixture diversity, but the stems must be remixed exactly and their sum checked against the mixture.

### Generative and neural-vocoder training

Waveform and spectrogram generators are sensitive to phase, sample alignment, loudness, and bandwidth. An augmentation that is harmless for classification may create inconsistent conditioning-target pairs for generation. Decide whether transforms belong before feature extraction, after feature extraction, on the waveform target, or on both sides of the pair.

## Class Imbalance and Conditional Policies

Augmentation can increase the number of minority-class examples, but replication alone does not create new semantic coverage. If every rare example produces many near-identical children, the model may memorize the source family. Vary nuisance conditions while preserving group-aware splits, and monitor performance per original source rather than per rendered file.

A class-dependent sampling weight can be written as:

$$
w_c=\frac{(n_c+\epsilon)^{-\alpha}}
{\sum_j(n_j+\epsilon)^{-\alpha}},
$$

where $w_c$ is the probability of selecting class $c$, $n_c$ is its source-example count, $\epsilon$ prevents division by zero, and $\alpha$ controls the strength of balancing. Setting $\alpha=0$ gives uniform class-independent sampling over the existing selection process, while larger $\alpha$ increasingly favors rare classes. Keep source selection and augmentation severity as separate decisions.

Conditional policies can also reflect known deployment structure. A telephone codec belongs on telephone-like speech, not necessarily on every music clip. Very long reverberation may be plausible for distant speech but implausible for close-miked studio labels. Document these conditions so the policy does not become an opaque set of class-specific interventions.

## Ablation Studies

An augmentation result is difficult to interpret when ten transforms are introduced at once. Begin with a clean baseline, evaluate individual transform families, then test combinations and severity ranges. The purpose of an ablation is not only to find the best score. It reveals which variation the model lacked and whether transforms interact.

A minimum experiment matrix should include the following runs:

| Run | Policy | Question answered |
| --- | --- | --- |
| A | No augmentation | What is the clean baseline? |
| B | Gain and crop only | Does cheap geometric variation help? |
| C | Noise only | Is additive robustness missing? |
| D | Room only | Is reverberation the limiting condition? |
| E | Time and pitch only | Does phase-vocoder variation help? |
| F | Full policy | Do the families combine constructively? |
| G | Full policy without one family | Which family contributes marginal value? |
| H | Full policy at half and double severity | Is the chosen range near a useful operating point? |

Run multiple training seeds when the expected improvement is close to ordinary training variance. Record mean, dispersion, and the individual runs. A single favorable seed is not evidence that the policy is robust.

For metric $m$ measured on an augmented system and baseline, report:

$$
\Delta m=m_{\mathrm{aug}}-m_{\mathrm{base}},
$$

where $\Delta m$ is the absolute metric change, $m_{\mathrm{aug}}$ is the score after training with augmentation, and $m_{\mathrm{base}}$ is the baseline score under the same evaluation protocol. Also report relative change when it is meaningful, but do not replace the absolute result with a percentage that hides scale.

## Evaluation Beyond One Aggregate Score

An augmentation policy should be evaluated on clean data and on targeted stress conditions. Clean performance reveals whether invariance came at the cost of useful information. Stress subsets reveal whether the intended robustness actually improved. A synthetic stress test is useful for diagnosis, but it should not be the only evidence because matching training and test transformations can reward engine-specific artifacts.

The evaluation suite should include these perspectives:

- clean in-domain performance
- natural out-of-domain recordings
- controlled synthetic severity sweeps
- task-specific subgroup metrics
- calibration and confidence under corruption
- worst-group or low-percentile performance
- compute, latency, and storage cost
- random listening inspection and artifact annotation

Plot metrics against severity rather than reporting only one endpoint. Every graph should label the transform parameter and unit on the horizontal axis and the metric and direction on the vertical axis. Include confidence intervals or run dispersion when multiple seeds are available.

## Detecting Shortcuts and Augmentation Artifacts

Models exploit stable details that researchers may not notice. A resampler's passband, a codec delay, deterministic file padding, a phase-vocoder texture, or a naming-dependent split can become predictive. Shortcut checks should therefore be part of policy validation.

Useful diagnostics include the following experiments:

- Train a classifier to predict whether and how an example was augmented.
- Compare multiple DSP engines for the same nominal transform.
- Randomize output encoding and metadata independently of class.
- Inspect spectrograms at policy boundaries and around transients.
- Evaluate on naturally occurring versions of the targeted condition.
- Shuffle augmentation parameters across labels and confirm performance collapses when it should.
- Check whether source duration, padding, or output length leaks a target.
- Verify that failed renders are not concentrated in particular classes.

A high augmentation-detector score is not automatically fatal because transformations necessarily leave evidence. It is a warning to test whether the downstream model relies on that evidence. Engine diversity and natural evaluation recordings are stronger safeguards than attempting to make every synthetic transform undetectable.

## Quality Control Before Training

Large augmentation jobs should have a release gate just like software. Validate manifests, inspect distributions, listen to stratified samples, and reject outputs that violate technical limits. The gate should be deterministic and should produce a report that can travel with the dataset.

The following checks catch many failures cheaply:

1. Confirm that output count matches the dry-run plan.
2. Validate that every output opens and has the expected channel count and sample rate.
3. Check peak level, clipping percentage, duration, silence fraction, and nonfinite samples.
4. Compare realized parameter histograms with policy bounds.
5. Confirm that source groups occur in one split only.
6. Verify label transforms on hand-calculated examples.
7. Listen to random samples and all extrema.
8. Compare source and output hashes for accidental duplicates.
9. Record render failures and investigate class-dependent failure rates.
10. Freeze manifests and policy files before model training begins.

```bash
pvx augment corpus/*.wav \
  --output-dir candidate_aug \
  --variants-per-input 4 \
  --intent mir_music \
  --dry-run \
  --manifest-jsonl reports/candidate_plan.jsonl \
  --seed 2026

pvx augment-manifest validate reports/candidate_plan.jsonl --strict
pvx augment-manifest stats reports/candidate_plan.jsonl
```

## A Reproducible Experiment Layout

A predictable directory structure reduces the chance that policies, manifests, and model results become separated. Treat derived audio as replaceable, but treat its recipe and provenance as durable research records.

```text
experiment/
  README.md
  environment.txt
  policies/
    baseline.json
    full.json
    no_room.json
  manifests/
    sources.jsonl
    full_train.jsonl
    no_room_train.jsonl
  reports/
    augmentation_qc.json
    evaluation_by_condition.csv
  checkpoints/
  logs/
  derived_audio/
```

The experiment README should define the task, corpus version, split unit, label semantics, primary metric, expected deployment conditions, and stopping rule. Write these choices before reading test results. This makes later policy changes easier to recognize as exploratory rather than confirmatory.

## Complex Policy Example: Robust Speech with Auditable Views

The following example combines deterministic offline phase-vocoder variants with online environmental corruption. The expensive structural variation is frozen in a manifest. Noise and gain remain dynamic during training so each epoch sees fresh conditions.

```bash
pvx augment speech/train/*.wav \
  --output-dir derived_audio/speech_structure \
  --variants-per-input 3 \
  --intent asr_robust \
  --label-policy preserve \
  --split-mode speaker_balanced \
  --labels-csv metadata/speakers.csv \
  --grouping stem-prefix \
  --group-separator "__" \
  --manifest-jsonl manifests/speech_structure.jsonl \
  --manifest-csv manifests/speech_structure.csv \
  --workers 8 \
  --seed 314159
```

The online stage can then add recording-condition variation without changing transcripts.

```python
from pvx.augment import AddNoise, GainPerturber, Pipeline, RoomSimulator

online_environment = Pipeline(
    [
        GainPerturber(gain_db=(-5.0, 5.0), p=0.8),
        RoomSimulator(
            rt60_range=(0.08, 0.85),
            wet_range=(0.08, 0.50),
            p=0.45,
        ),
        AddNoise(snr_db=(8.0, 35.0), noise_type="pink", p=0.60),
    ],
    seed=271828,
)

def augment_training_item(audio, sample_rate, stable_item_seed):
    return online_environment(audio, sample_rate, seed=stable_item_seed)
```

Evaluate this system against clean speech, naturally noisy speech, naturally reverberant speech, and synthetic sweeps made with a different noise and room collection. The different evaluation generator reduces the chance that success reflects exact matching to the training engine.

## Complex Policy Example: Equivariant Music Labels

A music pipeline often needs to transform several labels together. Suppose each example contains audio, beat times, a key class, and framewise fundamental frequency. A stretch factor $r$ multiplies beat times, while a pitch shift $s$ multiplies frequency by $2^{s/12}$ and transposes key by $s$ semitones when $s$ is integral.

The combined mapping is:

$$
g_{r,s}(b_j,k,f_t)
=
\left(rb_j,(k+s)\bmod 12,2^{s/12}f_t\right),
$$

where $b_j$ is beat time $j$, $k$ is the pitch-class key label, $f_t$ is fundamental frequency at frame $t$, $r$ is the output-duration ratio, and $s$ is the semitone shift. Frame timestamps must also be stretched before the transformed $f_t$ sequence is aligned to output audio.

This mapping illustrates why a generic `label_policy=preserve` switch cannot express every research task. Render parameters in the pvx manifest, then run a task-specific annotation transformer that consumes those exact parameters. Validate the transformed annotation against a small set of synthetic tones and click tracks before applying it to a corpus.

## Fairness, Consent, and Dataset Governance

Augmentation does not repair missing populations. Pitch shifting one voice does not create a new speaker, accent, age group, language, room, microphone practice, or cultural context. It can reduce sensitivity to selected nuisance variables, but it cannot replace representative collection and careful governance.

Policies may also affect groups differently. A denoising or time-scaling transform can alter high-frequency consonants, breathiness, vocal tremor, or low-energy speech in ways that are uneven across speakers. Evaluate subgroup performance where ethically collected metadata permits it. When metadata is unavailable or inappropriate, state the limitation rather than treating the aggregate score as universal.

Respect source licenses, consent terms, and restrictions on derivative data. A manifest should retain lineage from output to source without exposing private paths or identities in public artifacts. Hashes can support integrity checks, but hashes do not anonymize a small or identifiable corpus.

## Publication and Handoff Checklist

A published augmentation study should provide enough information for another researcher to reconstruct both the policy and the evaluation. The following checklist is a useful minimum:

- corpus versions, licenses, and inclusion criteria
- split assignment unit and leakage checks
- pvx version or commit and installation method
- complete policy files with parameter distributions and transform order
- seeds and seed-derivation method
- variants per source and realized transform frequencies
- label-invariance or label-equivariance rules
- clean and stress-test metrics with multiple training seeds
- compute, storage, and rendering-failure summaries
- manifest schema and hashes for released artifacts
- listening or quality-control procedure
- known artifacts, failed conditions, and excluded examples

## Research Questions Worth Testing

The chapter's methods support several deeper experiments. These questions are more informative than asking whether augmentation helps in the abstract:

- Does phase-vocoder engine diversity improve transfer to natural tempo variation?
- At what severity does a label-preserving pitch shift begin to alter perceived speaker identity?
- Do curriculum policies outperform a fixed mixture when total transform exposure is held constant?
- Does preserving a clean anchor fraction improve calibration under in-domain conditions?
- Which transform interactions are constructive, redundant, or destructive?
- Can a model predict the processing engine from learned embeddings?
- Does group-balanced splitting change conclusions drawn from ordinary random splitting?
- How much apparent robustness survives evaluation with independently generated corruptions?
- Are minority-class gains due to nuisance diversity or simple oversampling?
- Which manifest statistics best predict a policy's downstream benefit?

## Benchmarking Augmentation Pipelines

Use the profile suite benchmark (speech/music/noisy/stereo):

```bash
python benchmarks/run_augment_profile_suite.py \
  --quick \
  --gate \
  --out-dir benchmarks/out_augment_profiles
```

Refresh all per-profile baselines after intentional benchmark changes:

```bash
python benchmarks/run_augment_profile_suite.py \
  --quick \
  --refresh-baselines \
  --out-dir benchmarks/out_augment_profiles_refresh
```

Outputs:
- Per-profile reports:
  - `benchmarks/out_augment_profiles/<profile>/report.json`
  - `benchmarks/out_augment_profiles/<profile>/report.md`
- Suite report:
  - `benchmarks/out_augment_profiles/suite_report.json`
  - `benchmarks/out_augment_profiles/suite_report.md`

## Running with uv

```bash
uv run pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337
```

## Research Notes

- Keep original clean file IDs in your training metadata so you can group augmented siblings.
- Avoid split leakage: use `--grouping stem-prefix` with a stable naming convention (for example `speaker42__take3.wav`).
- Prefer fixed seeds for published experiments; change seeds only for explicit variance studies.
- Use `--dry-run` before large jobs to validate output counts and manifest content.

## Attribution
