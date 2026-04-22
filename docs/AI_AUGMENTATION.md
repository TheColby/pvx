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


