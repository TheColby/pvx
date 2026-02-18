# pvx vs Rubber Band (Technical Comparison)

This document is intentionally non-marketing. Both tools are strong; each has tradeoffs.

## 1. Scope and Design Philosophy

| Topic | pvx | Rubber Band |
| --- | --- | --- |
| Primary form | Python toolkit + multiple CLIs + inspectable DSP code | Highly optimized dedicated stretcher/pitch-shifter library/CLI |
| Priority order | Audio quality first, speed second | High-quality defaults with strong optimization maturity |
| Extensibility | Easy to modify in Python (`src/pvx/core/*`) | Strong core engine, but deeper custom algorithm edits are lower-level |
| Workflow focus | Processing pipelines, CSV control maps, integrated mastering and docs | Best-in-class focused stretch/pitch engine for many production workflows |

## 2. Areas Where pvx Now Adds Distinct Value

1. Hybrid transient engine (`--transient-mode hybrid|wsola`), mixing PV steady-state with WSOLA transient handling.
2. Stereo/multichannel coherence controls (`--stereo-mode` + `--coherence-strength`).
3. Built-in cycle-consistency benchmark runner (`benchmarks/run_bench.py`) with regression gate support.
4. Rich CLI ergonomics for new users (`--preset`, `--example`, `--guided`).

## 3. Areas Where Rubber Band Remains Strong

1. Mature optimized implementation tuned over many years.
2. Widely integrated in DAWs and production tools.
3. Excellent default quality without much parameter tuning.

## 4. Practical Guidance

Use `pvx` when:
- you need transparent, inspectable DSP implementation details
- you want to compose custom pipelines and control maps quickly
- you need deterministic CPU regression testing in-repo
- you want to bias decisions toward quality preservation before speed

Use Rubber Band when:
- you need a very mature drop-in engine with minimal tuning
- integration context already standardizes on Rubber Band

## 5. Reproducible Comparison in This Repo

```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out
```

If Rubber Band is installed, it is included automatically; if not, it is skipped with a clear note in the report.

## 6. Important Caveat About "Better"

"Better" is material- and task-dependent:
- vocals vs drums
- mild vs extreme stretch
- mono vs wide stereo
- objective metric score vs subjective musical preference

Use the benchmark report as a directional signal, then always audition critical renders.
