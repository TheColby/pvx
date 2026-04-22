#!/usr/bin/env python3

"""Create and inspect reusable frequency-response artifacts (PVXRF)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pvx.core.analysis_store import load_analysis_artifact
from pvx.core.common import (
    add_console_args,
    build_examples_epilog,
    build_status_bar,
    is_silent,
    log_error,
    log_message,
)
from pvx.core.response_store import (
    NormalizeMode,
    PhaseMode,
    ResponseMethod,
    load_response_artifact,
    response_from_analysis,
    save_response_artifact,
    summarize_response_artifact,
)

RESPONSE_METHOD_CHOICES: tuple[ResponseMethod, ...] = ("median", "mean", "rms", "max")
PHASE_MODE_CHOICES: tuple[PhaseMode, ...] = ("mean", "zero", "first")
NORMALIZE_CHOICES: tuple[NormalizeMode, ...] = ("none", "peak", "rms")


def _normalize_argv(argv: list[str] | None) -> list[str]:
    tokens = list(argv or [])
    if not tokens:
        return tokens
    if tokens[0] in {"create", "inspect", "--help", "-h"}:
        return tokens
    return ["create", *tokens]


def _default_output_path(analysis_path: Path) -> Path:
    return analysis_path.with_suffix(".pvxrf.npz")


def _print_summary(summary: dict[str, object]) -> None:
    width = max(len(key) for key in summary) if summary else 0
    for key, value in summary.items():
        print(f"{key:<{width}} : {value}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvx response",
        description="Create and inspect PVXRF frequency-response artifacts derived from PVXAN analyses.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx response create input.pvxan.npz --output input.pvxrf.npz --method median --normalize peak",
                "pvx response inspect input.pvxrf.npz",
                "pvxresponse input.pvxan.npz --phase-mode zero --normalize rms",
            ],
            notes=[
                "When command is omitted, create mode is assumed for convenience.",
                "PVXRF stores per-channel response magnitude/phase vectors and frequency bins.",
            ],
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create",
        help="Derive response artifact from analysis artifact",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    create_parser.add_argument("analysis", type=Path, help="Input PVXAN artifact")
    create_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PVXRF path (default: <analysis>.pvxrf.npz)",
    )
    create_parser.add_argument(
        "--method",
        choices=list(RESPONSE_METHOD_CHOICES),
        default="median",
        help="Magnitude aggregation across frames (default: median)",
    )
    create_parser.add_argument(
        "--phase-mode",
        choices=list(PHASE_MODE_CHOICES),
        default="mean",
        help="Phase aggregation strategy (default: mean)",
    )
    create_parser.add_argument(
        "--normalize",
        choices=list(NORMALIZE_CHOICES),
        default="peak",
        help="Magnitude normalization mode (default: peak)",
    )
    create_parser.add_argument(
        "--smoothing-bins",
        type=int,
        default=1,
        help="Moving-average smoothing width in bins (default: 1)",
    )
    create_parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    add_console_args(create_parser)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect existing PVXRF artifact",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    inspect_parser.add_argument("artifact", type=Path, help="PVXRF artifact path")
    inspect_parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    inspect_parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    add_console_args(inspect_parser)

    return parser


def _run_create(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if int(args.smoothing_bins) <= 0:
        parser.error("--smoothing-bins must be >= 1")

    status = build_status_bar(args, "pvxresponse", total=4)
    analysis_path = Path(args.analysis).expanduser().resolve()
    analysis = load_analysis_artifact(analysis_path)
    status.step(1, "load-analysis")

    response = response_from_analysis(
        analysis,
        method=str(args.method),  # type: ignore[arg-type]
        phase_mode=str(args.phase_mode),  # type: ignore[arg-type]
        normalize=str(args.normalize),  # type: ignore[arg-type]
        smoothing_bins=int(args.smoothing_bins),
    )
    status.step(2, "derive")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(analysis_path)
    )
    output_path = save_response_artifact(output_path, response)
    status.step(3, "save")

    summary = summarize_response_artifact(response)
    status.step(4, "summarize")
    status.finish("done")

    if args.summary_json is not None:
        _write_json(Path(args.summary_json).expanduser().resolve(), summary)
        log_message(
            args,
            f"[ok] wrote summary JSON -> {Path(args.summary_json).expanduser().resolve()}",
            min_level="normal",
        )
    if not is_silent(args):
        print("PVXRF summary")
        _print_summary(summary)
    log_message(args, f"[ok] saved response -> {output_path}", min_level="normal")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    status = build_status_bar(args, "pvxresponse-inspect", total=2)
    artifact = load_response_artifact(Path(args.artifact).expanduser().resolve())
    status.step(1, "load")
    summary = summarize_response_artifact(artifact)
    status.step(2, "inspect")
    status.finish("done")
    if args.summary_json is not None:
        _write_json(Path(args.summary_json).expanduser().resolve(), summary)
        log_message(
            args,
            f"[ok] wrote summary JSON -> {Path(args.summary_json).expanduser().resolve()}",
            min_level="normal",
        )
    if not is_silent(args):
        if bool(args.json):
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print("PVXRF summary")
            _print_summary(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(_normalize_argv(tokens))
    try:
        if args.command == "create":
            return _run_create(args, parser)
        if args.command == "inspect":
            return _run_inspect(args)
    except (OSError, ValueError, RuntimeError) as exc:
        log_error(args, f"[error] pvxresponse: {exc}")
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
