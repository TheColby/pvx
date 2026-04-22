<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# Follow Workflow Migration




This note describes how to migrate from long pitch-follow shell pipelines to the unified `pvx follow` helper.

## Why This Change

- shorter commands for common sidechain workflows
- fewer shell-specific quoting and pipe errors
- same control-map architecture under the hood (`pitch-track` + `voc --control-stdin`)
- backward compatibility for existing pipelines

## Old vs New

Old (manual pipe):

```bash
pvx pitch-track A.wav --emit pitch_to_stretch --output - \
  | pvx voc B.wav --control-stdin --pitch-conf-min 0.75 --output B_follow.wav
```

New (one command):

```bash
pvx follow A.wav B.wav --emit pitch_to_stretch --pitch-conf-min 0.75 --output B_follow.wav
```

## When to Keep Manual Pipes

Use explicit pipes when you need:

- custom route rewrites (`--route stretch=pitch_ratio`, `--route pitch_ratio=const(1.0)`)
- external non-pvx processing between producer and consumer
- shell orchestration with additional non-audio transforms

Equivalent advanced manual path:

```bash
pvx pitch-track A.wav --output - \
  | pvx voc B.wav --control-stdin \
      --route stretch=pitch_ratio \
      --route pitch_ratio=const(1.0) \
      --output B_manual.wav
```

## Rollout Guidance

1. For new scripts, default to `pvx follow`.
2. Keep existing manual pipelines unchanged until they are touched for other reasons.
3. Add regression tests for representative follow jobs (already covered in `tests/test_cli_regression.py`).
4. Keep manual route examples in documentation for advanced control-bus users.

## Attribution


