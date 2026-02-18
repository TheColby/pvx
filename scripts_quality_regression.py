#!/usr/bin/env python3
"""Render a pvx case and compare objective metrics against a baseline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _metrics(path: Path) -> dict[str, float]:
    audio, sr = sf.read(path, always_2d=True)
    mono = np.mean(np.asarray(audio, dtype=np.float64), axis=1)
    if mono.size == 0:
        return {
            "sample_rate": float(sr),
            "samples": 0.0,
            "duration_s": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "centroid_hz": 0.0,
            "flatness": 0.0,
            "jump_count_gt_0p10": 0.0,
        }
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    peak = float(np.max(np.abs(mono)))
    win = np.hanning(mono.size)
    spec = np.abs(np.fft.rfft(mono * win))
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
    centroid = float(np.sum(freqs * spec) / max(1e-12, float(np.sum(spec))))
    flatness = float(np.exp(np.mean(np.log(spec + 1e-12))) / (np.mean(spec) + 1e-12))
    jump_count = int(np.sum(np.abs(np.diff(mono)) > 0.10))
    return {
        "sample_rate": float(sr),
        "samples": float(mono.size),
        "duration_s": float(mono.size / sr),
        "rms": rms,
        "peak": peak,
        "centroid_hz": centroid,
        "flatness": flatness,
        "jump_count_gt_0p10": float(jump_count),
    }


def _compare(current: dict[str, float], baseline: dict[str, float], tolerances: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for key, tol in tolerances.items():
        if key not in baseline:
            continue
        a = float(current.get(key, 0.0))
        b = float(baseline[key])
        if abs(a - b) > float(tol):
            failures.append(f"{key}: current={a:.6f} baseline={b:.6f} tol={float(tol):.6f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pvx quality regression checker.")
    parser.add_argument("--input", type=Path, required=True, help="Input audio file")
    parser.add_argument("--output", type=Path, required=True, help="Rendered output path")
    parser.add_argument("--tool", default="python3 pvxvoc.py", help="Render command (default: 'python3 pvxvoc.py')")
    parser.add_argument("--render-args", required=True, help="Argument string appended after input path")
    parser.add_argument("--report-json", type=Path, default=Path("reports/quality_regression.json"))
    parser.add_argument("--baseline-json", type=Path, default=None, help="Optional baseline metrics JSON")
    parser.add_argument("--tol-rms", type=float, default=0.05, help="Absolute RMS tolerance")
    parser.add_argument("--tol-peak", type=float, default=0.08, help="Absolute peak tolerance")
    parser.add_argument("--tol-centroid-hz", type=float, default=400.0, help="Absolute centroid tolerance")
    parser.add_argument("--tol-flatness", type=float, default=0.08, help="Absolute flatness tolerance")
    parser.add_argument("--tol-jumps", type=float, default=1200.0, help="Absolute jump-count tolerance")
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    tool_cmd = shlex.split(args.tool)
    render_args = shlex.split(args.render_args)
    cmd = [
        *tool_cmd,
        str(input_path),
        *render_args,
        "--output",
        str(output_path),
        "--overwrite",
        "--quiet",
    ]
    proc = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)

    current = _metrics(output_path)
    payload: dict[str, object] = {
        "input": str(input_path),
        "output": str(output_path),
        "tool": args.tool,
        "render_args": args.render_args,
        "metrics": current,
        "pass": True,
        "failures": [],
    }

    if args.baseline_json is not None:
        baseline_path = args.baseline_json.resolve()
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = baseline_payload.get("metrics", baseline_payload)
        tolerances = {
            "rms": float(args.tol_rms),
            "peak": float(args.tol_peak),
            "centroid_hz": float(args.tol_centroid_hz),
            "flatness": float(args.tol_flatness),
            "jump_count_gt_0p10": float(args.tol_jumps),
        }
        failures = _compare(current, baseline, tolerances)
        payload["baseline"] = baseline
        payload["tolerances"] = tolerances
        payload["failures"] = failures
        payload["pass"] = not failures

    report_path = args.report_json.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    if not bool(payload["pass"]):
        for failure in payload["failures"]:  # type: ignore[index]
            print(f"[FAIL] {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
