#!/usr/bin/env python3
"""Benchmark matrix runner for pvxvoc transform/window/device combinations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shlex
import subprocess
import time
from pathlib import Path


def _parse_csv_tokens(value: str) -> list[str]:
    return [token.strip() for token in str(value).split(",") if token.strip()]


def _run_case(
    *,
    tool_cmd: list[str],
    input_path: Path,
    output_path: Path,
    transform: str,
    window: str,
    n_fft: int,
    device: str,
    time_stretch: float,
    repeats: int,
    cwd: Path,
) -> dict[str, float | str | int]:
    cmd = [
        *tool_cmd,
        str(input_path),
        "--transform",
        transform,
        "--window",
        window,
        "--n-fft",
        str(n_fft),
        "--win-length",
        str(n_fft),
        "--hop-size",
        str(max(1, n_fft // 4)),
        "--device",
        device,
        "--time-stretch",
        str(time_stretch),
        "--output",
        str(output_path),
        "--overwrite",
        "--quiet",
    ]

    elapsed_runs: list[float] = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        elapsed = float(time.perf_counter() - t0)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Benchmark case failed [{transform}/{window}/n_fft={n_fft}/device={device}]: {proc.stderr.strip()}"
            )
        elapsed_runs.append(elapsed)

    avg = sum(elapsed_runs) / len(elapsed_runs)
    return {
        "transform": transform,
        "window": window,
        "n_fft": int(n_fft),
        "device": device,
        "time_stretch": float(time_stretch),
        "repeats": int(repeats),
        "elapsed_mean_s": float(avg),
        "elapsed_min_s": float(min(elapsed_runs)),
        "elapsed_max_s": float(max(elapsed_runs)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pvxvoc benchmark sweeps and emit CSV/JSON.")
    parser.add_argument("--input", type=Path, required=True, help="Input audio file")
    parser.add_argument(
        "--tool",
        default="python3 pvxvoc.py",
        help="Command used to invoke pvxvoc (default: 'python3 pvxvoc.py')",
    )
    parser.add_argument("--transforms", default="fft,dft,czt,dct,dst,hartley", help="CSV transform list")
    parser.add_argument("--windows", default="hann,kaiser,blackmanharris", help="CSV window list")
    parser.add_argument("--n-ffts", default="1024,2048,4096", help="CSV FFT-size list")
    parser.add_argument("--devices", default="cpu", help="CSV device list (cpu,cuda,auto)")
    parser.add_argument("--time-stretch", type=float, default=1.25, help="Stretch factor (default: 1.25)")
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions per case (default: 1)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/benchmarks"), help="Output directory")
    parser.add_argument("--name", default="benchmark_matrix", help="Output basename")
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    transforms = _parse_csv_tokens(args.transforms)
    windows = _parse_csv_tokens(args.windows)
    devices = _parse_csv_tokens(args.devices)
    n_ffts = [int(token) for token in _parse_csv_tokens(args.n_ffts)]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_output = out_dir / f"{args.name}_tmp.wav"
    tool_cmd = shlex.split(args.tool)
    cwd = Path.cwd()

    rows: list[dict[str, float | str | int]] = []
    for transform, window, n_fft, device in itertools.product(transforms, windows, n_ffts, devices):
        row = _run_case(
            tool_cmd=tool_cmd,
            input_path=input_path,
            output_path=tmp_output,
            transform=transform,
            window=window,
            n_fft=int(n_fft),
            device=device,
            time_stretch=float(args.time_stretch),
            repeats=int(args.repeats),
            cwd=cwd,
        )
        rows.append(row)

    csv_path = out_dir / f"{args.name}.csv"
    json_path = out_dir / f"{args.name}.json"

    fieldnames = [
        "transform",
        "window",
        "n_fft",
        "device",
        "time_stretch",
        "repeats",
        "elapsed_mean_s",
        "elapsed_min_s",
        "elapsed_max_s",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "input": str(input_path),
        "tool": args.tool,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
