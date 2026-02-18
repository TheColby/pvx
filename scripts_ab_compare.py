#!/usr/bin/env python3
"""Run an A/B pvx render comparison and emit metrics reports."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
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
            "crest": 0.0,
            "centroid_hz": 0.0,
            "flatness": 0.0,
            "jump_count_gt_0p10": 0.0,
        }

    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    peak = float(np.max(np.abs(mono)))
    crest = peak / max(rms, 1e-12)

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
        "crest": crest,
        "centroid_hz": centroid,
        "flatness": flatness,
        "jump_count_gt_0p10": float(jump_count),
    }


def _render(
    *,
    label: str,
    tool_cmd: list[str],
    input_path: Path,
    extra_args: list[str],
    output_path: Path,
    cwd: Path,
) -> tuple[dict[str, float], float]:
    cmd = [
        *tool_cmd,
        str(input_path),
        *extra_args,
        "--output",
        str(output_path),
        "--overwrite",
        "--quiet",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = float(time.perf_counter() - t0)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} render failed: {proc.stderr.strip()}")
    if not output_path.exists():
        raise RuntimeError(f"{label} render produced no output: {output_path}")
    return _metrics(output_path), elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B compare pvx renders with basic objective metrics.")
    parser.add_argument("--input", type=Path, required=True, help="Input audio file")
    parser.add_argument(
        "--tool",
        default="python3 pvxvoc.py",
        help="Base command used for both renders (default: 'python3 pvxvoc.py')",
    )
    parser.add_argument("--a-args", required=True, help="Additional argument string for render A")
    parser.add_argument("--b-args", required=True, help="Additional argument string for render B")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/ab"), help="Report/output directory")
    parser.add_argument("--name", default="ab_compare", help="Report basename")
    args = parser.parse_args(argv)

    cwd = Path.cwd()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_a = out_dir / f"{args.name}_A.wav"
    out_b = out_dir / f"{args.name}_B.wav"

    tool_cmd = shlex.split(args.tool)
    a_args = shlex.split(args.a_args)
    b_args = shlex.split(args.b_args)

    metrics_a, elapsed_a = _render(
        label="A",
        tool_cmd=tool_cmd,
        input_path=input_path,
        extra_args=a_args,
        output_path=out_a,
        cwd=cwd,
    )
    metrics_b, elapsed_b = _render(
        label="B",
        tool_cmd=tool_cmd,
        input_path=input_path,
        extra_args=b_args,
        output_path=out_b,
        cwd=cwd,
    )

    audio_a, sr_a = sf.read(out_a, always_2d=True)
    audio_b, sr_b = sf.read(out_b, always_2d=True)
    if sr_a != sr_b:
        raise RuntimeError(f"Sample-rate mismatch: A={sr_a}, B={sr_b}")
    n = min(audio_a.shape[0], audio_b.shape[0])
    d = np.mean(audio_a[:n, :], axis=1) - np.mean(audio_b[:n, :], axis=1)
    diff_rms = float(np.sqrt(np.mean(d * d) + 1e-12))
    snr_db = float(20.0 * np.log10(np.sqrt(np.mean(np.mean(audio_a[:n, :], axis=1) ** 2) + 1e-12) / diff_rms))

    payload = {
        "input": str(input_path),
        "tool": args.tool,
        "a_args": args.a_args,
        "b_args": args.b_args,
        "elapsed_seconds": {"A": elapsed_a, "B": elapsed_b},
        "metrics": {"A": metrics_a, "B": metrics_b},
        "difference": {"aligned_samples": int(n), "diff_rms": diff_rms, "snr_db_vs_A": snr_db},
    }

    json_path = out_dir / f"{args.name}.json"
    md_path = out_dir / f"{args.name}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_lines = [
        f"# A/B Report: {args.name}",
        "",
        f"- Input: `{input_path}`",
        f"- Command A: `{args.tool} {input_path.name} {args.a_args}`",
        f"- Command B: `{args.tool} {input_path.name} {args.b_args}`",
        f"- Elapsed A/B: `{elapsed_a:.3f}s` / `{elapsed_b:.3f}s`",
        f"- Diff RMS: `{diff_rms:.6f}`",
        f"- SNR (B vs A): `{snr_db:.2f} dB`",
        "",
        "## Metrics",
        "",
        "| Metric | A | B |",
        "| --- | ---: | ---: |",
    ]
    for key in sorted(metrics_a.keys()):
        md_lines.append(f"| `{key}` | {metrics_a[key]:.6f} | {metrics_b[key]:.6f} |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
