#!/usr/bin/env python3

"""Run all augmentation benchmark profiles and summarize regression status."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "run_augment_bench.py"
DEFAULT_PROFILES = ROOT / "benchmarks" / "augment_profiles.json"


def _load_profiles(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {}) if isinstance(payload, dict) else {}
    if not isinstance(profiles, dict):
        return []
    return sorted(str(k) for k in profiles)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all augmentation benchmark profiles and gates."
    )
    parser.add_argument("--profiles-file", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/out_augment_profiles"))
    parser.add_argument("--quick", action="store_true", help="Use quick per-profile runs")
    parser.add_argument("--gate", action="store_true", help="Enable profile gate checks")
    parser.add_argument(
        "--refresh-baselines", action="store_true", help="Refresh each profile baseline"
    )
    args = parser.parse_args(argv)

    profiles_file = Path(args.profiles_file).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = _load_profiles(profiles_file)
    if not profiles:
        print("No profiles found.", file=sys.stderr)
        return 2

    summary: list[dict[str, Any]] = []
    overall_ok = True

    for profile in profiles:
        profile_out = out_dir / profile
        cmd = [
            sys.executable,
            str(SCRIPT),
            "--profiles-file",
            str(profiles_file),
            "--profile",
            profile,
            "--out-dir",
            str(profile_out),
        ]
        if bool(args.quick):
            cmd.append("--quick")
        if bool(args.gate):
            cmd.append("--gate")
        if bool(args.refresh_baselines):
            cmd.append("--refresh-baseline")

        proc = subprocess.run(cmd, cwd=ROOT)
        report_json = profile_out / "report.json"
        payload = _load_json(report_json) if report_json.exists() else {}
        gate = payload.get("gate", {}) if isinstance(payload, dict) else {}
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}

        gate_ok = bool(gate.get("ok", False)) if isinstance(gate, dict) else False
        row = {
            "profile": profile,
            "exit_code": int(proc.returncode),
            "gate_ok": gate_ok,
            "records_total": metrics.get("records_total"),
            "required_field_errors": metrics.get("required_field_errors"),
            "split_balance_l1": metrics.get("split_balance_l1"),
            "clip_pct_max": metrics.get("clip_pct_max"),
            "pair_coverage": metrics.get("pair_coverage"),
        }
        summary.append(row)
        if proc.returncode != 0:
            overall_ok = False

    suite = {
        "meta": {
            "profiles_file": str(profiles_file),
            "quick": bool(args.quick),
            "gate": bool(args.gate),
            "refresh_baselines": bool(args.refresh_baselines),
            "profiles": profiles,
        },
        "summary": summary,
        "ok": bool(overall_ok),
    }

    suite_json = out_dir / "suite_report.json"
    suite_md = out_dir / "suite_report.md"
    suite_json.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# pvx Augmentation Profile Suite")
    lines.append("")
    lines.append(f"- Profiles file: `{profiles_file}`")
    lines.append(f"- Gate enabled: `{bool(args.gate)}`")
    lines.append(f"- Quick mode: `{bool(args.quick)}`")
    lines.append("")
    lines.append(
        "| Profile | Exit | Gate | Records | Req Errors | Split L1 | Clip Max | Pair Coverage |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary:
        lines.append(
            "| {profile} | {exit_code} | {gate_ok} | {records_total} | {required_field_errors} | {split_balance_l1} | {clip_pct_max} | {pair_coverage} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(f"- Overall pass: `{bool(overall_ok)}`")
    suite_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[augment-suite] report json -> {suite_json}")
    print(f"[augment-suite] report md   -> {suite_md}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
