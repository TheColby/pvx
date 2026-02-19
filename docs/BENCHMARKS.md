# pvx Benchmarks

`pvx` uses a deterministic quality-first benchmark harness in `benchmarks/run_bench.py`.

Stage 2 focus:
- objective + perceptual proxy metrics
- deterministic CPU reproducibility checks
- corpus hash validation via manifest
- CI regression gating with aggregate + row-level checks
- automatic diagnostics with recommended remediation flags

## Quick CI Run

```bash
python3 benchmarks/run_bench.py \
  --quick \
  --out-dir benchmarks/out \
  --strict-corpus \
  --determinism-runs 2 \
  --baseline benchmarks/baseline_small.json \
  --gate \
  --gate-row-level \
  --gate-signatures
```

Outputs:
- `benchmarks/out/report.json`
- `benchmarks/out/report.md`
- optional `benchmarks/out/summary.png` with `--plots`

## Corpus And Manifest

The benchmark corpus is generated deterministically in `benchmarks/data/` and tracked by:
- `benchmarks/data/manifest.json`

Useful commands:

```bash
# Rebuild manifest from corpus
python3 benchmarks/run_bench.py --quick --refresh-manifest

# Enforce manifest/hash consistency
python3 benchmarks/run_bench.py --quick --strict-corpus
```

## Determinism Controls

- `--deterministic-cpu` (default on):
  - forces `--phase-random-seed 0` for pvx benchmark renders
  - applies stable environment controls (`PYTHONHASHSEED=0`, single-thread BLAS hints)
- `--determinism-runs N`:
  - repeats pvx render cycle per case
  - records hash sequences and max absolute sample delta
  - gate fails if mismatches are detected

## Gate Behavior

`--gate` compares current report vs baseline report (`--baseline`).

Aggregate metric directions:
- lower-is-better metrics fail if current > baseline + tolerance
- higher-is-better metrics fail if current < baseline - tolerance

User-tunable tolerances:
- `--tol-lsd`
- `--tol-modspec`
- `--tol-smear`
- `--tol-coherence`

Additional checks:
- `--gate-row-level`: per-case row regressions
- `--gate-signatures`: exact hash match for pvx output signatures

## Diagnostics

Each method now emits diagnostics in JSON and Markdown reports, for example:
- transient smear mitigation (`--transient-mode hybrid|wsola`, hop adjustments)
- phase/coherence fixes (`--phase-locking identity`, stereo lock modes)
- pitch/formant recommendations (`--pitch-mode formant-preserving`)
- perceptual-tooling notes when proxy metrics are used

These diagnostics are intended to reduce iteration time when quality regresses.
