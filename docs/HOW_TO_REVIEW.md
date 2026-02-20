# How To Review (Stage 3)

Use this checklist to review Stage 3 changes methodically.

## Acronym Primer

- command-line interface (CLI)
- digital signal processing (DSP)
- comma-separated values (CSV)
- JavaScript Object Notation (JSON)
- true peak decibels relative to full scale (dBTP)

## 1. Command-Line Interface (CLI) Surface

Run:

```bash
pvx list
pvx help voc
pvx chain --help
pvx stream --help
```

Check:
- `chain` and `stream` helper commands appear in `pvx list`.
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

## 3. Managed Chain/Stream Workflows

Run:

```bash
pvx chain test.wav --pipeline "voc --time-stretch 1.02 | formant --mode preserve" --output /tmp/pvx_chain_review.wav
pvx stream test.wav --output /tmp/pvx_stream_review.wav --chunk-seconds 0.10 --time-stretch 1.02
pvx stream test.wav --mode wrapper --output /tmp/pvx_stream_wrapper_review.wav --chunk-seconds 0.10 --time-stretch 1.02
```

Check:
- both commands complete successfully.
- output files are written.

## 4. Output Policy Behavior

Run:

```bash
pvx voc test.wav --time-stretch 1.0 --bit-depth 16 --dither tpdf --dither-seed 7 --metadata-policy sidecar --output /tmp/pvx_policy_review.wav --overwrite
```

Check:
- `/tmp/pvx_policy_review.wav` subtype is PCM_16.
- metadata sidecar exists at:
  - `/tmp/pvx_policy_review.wav.metadata.json`
- sidecar includes policy fields (`bit_depth`, `dither`, `true_peak_max_dbtp`, etc.).

## 5. Benchmark/Regression

Run:

```bash
python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out_stage3_validation
```

Check:
- report files generated:
  - `benchmarks/out_stage3_validation/report.json`
  - `benchmarks/out_stage3_validation/report.md`

## 6. Test Suite

Run:

```bash
python3 -m unittest discover -s tests
```

Check:
- all tests pass.
- includes new coverage for output policy and unified CLI chain/stream paths.
