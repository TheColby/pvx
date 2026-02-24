<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# How To Review (Stage 4)




Use this checklist to review follow/control-bus rollout changes methodically.

## 1. Command-Line Interface (CLI) Surface

Run:

```bash
pvx list
pvx help voc
pvx help follow
pvx follow --example all
pvx chain --help
pvx stream --help
```

Check:
- `follow`, `chain`, and `stream` helper commands appear in `pvx list`.
- `pvx help follow` points users to `pvx follow --help`.
- `pvx follow --example all` prints built-in feature-follow recipes.
- `pvx stream --help` shows `--mode {stateful,wrapper}`.
- `pvx voc` help includes output-policy flags:
  - `--bit-depth`
  - `--dither`
  - `--dither-seed`
  - `--true-peak-max-dbtp`
  - `--metadata-policy`

## 2. Backward Compatibility

Run:

```bash
python3 pvxvoc.py --help
python3 pvx.py help voc
```

Check:
- legacy wrappers still work.
- unified CLI forwarding still resolves `voc` help and options.

## 3. Managed Follow/Chain/Stream Workflows

Run:

```bash
pvx follow test_guide.wav test_target.wav --emit pitch_to_stretch --pitch-conf-min 0.75 --output /tmp/pvx_follow_review.wav --backend acf --quiet
pvx chain assets/test.wav --pipeline "voc --time-stretch 1.02 | formant --mode preserve" --output /tmp/pvx_chain_review.wav
pvx stream assets/test.wav --output /tmp/pvx_stream_review.wav --chunk-seconds 0.10 --time-stretch 1.02
pvx stream assets/test.wav --mode wrapper --output /tmp/pvx_stream_wrapper_review.wav --chunk-seconds 0.10 --time-stretch 1.02
```

Check:
- all commands complete successfully.
- `pvx follow` writes output without requiring explicit shell pipes.
- output files are written.

## 4. Output Policy Behavior

Run:

```bash
pvx voc assets/test.wav --time-stretch 1.0 --bit-depth 16 --dither tpdf --dither-seed 7 --metadata-policy sidecar --output /tmp/pvx_policy_review.wav --overwrite
```

Check:
- `/tmp/pvx_policy_review.wav` subtype is PCM_16.
- metadata sidecar exists at:
  - `/tmp/pvx_policy_review.wav.metadata.json`
- sidecar includes policy fields (`bit_depth`, `dither`, `true_peak_max_dbtp`, etc.).

## 5. Regression Coverage

Run:

```bash
python3 -m unittest tests.test_control_bus tests.test_cli_regression
```

Check:
- control-route parsing and CSV mapping tests pass.
- `pvx follow` regression tests pass (including invalid passthrough rejection).

## 6. Benchmark/Regression

Run:

```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out_stage4_validation
```

Check:
- report files generated:
  - `benchmarks/out_stage4_validation/report.json`
  - `benchmarks/out_stage4_validation/report.md`

## 7. Full Test Suite

Run:

```bash
python3 -m unittest discover -s tests
```

Check:
- all tests pass.
- includes new coverage for `pvx follow` and control-bus route helpers.

## Attribution

Copyright (c) 2026 Colby Leider and contributors. See [ATTRIBUTION.md](../ATTRIBUTION.md).
