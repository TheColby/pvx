#!/usr/bin/env python3

"""PVC-style function-stream reshaper for control maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pvx.core.common import (
    add_console_args,
    build_examples_epilog,
    build_status_bar,
    log_message,
)
from pvx.core.pvc_functions import (
    INTERP_CHOICES,
    RESHAPE_OPERATIONS,
    dump_control_points_csv,
    dump_control_points_json,
    load_control_points,
    parse_control_points_payload,
    reshape_control_points,
)


def build_parser(prog: str = "pvx reshape") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Transform control-rate CSV/JSON maps for pvx routing and modulation workflows.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx reshape stretch_env.csv --key stretch --operation scale --factor 1.2 --output stretch_scaled.csv",
                "pvx reshape map.csv --key pitch_ratio --operation clip --min 0.5 --max 2.0 --stdout | pvx voc input.wav --pitch-map-stdin --route pitch_ratio=pitch_ratio --output out.wav",
                "pvx reshape alpha_curve.csv --operation resample --rate 50 --interp polynomial --order 5 --output alpha_curve_dense.csv",
                "pvx reshape stretch_env.csv --operation smooth --window 9 --output stretch_smooth.csv",
            ],
            notes=[
                "Use --key to select which control column to reshape.",
                "For stdin input, pass '-' as INPUT and optionally set --input-format.",
            ],
        ),
    )
    parser.add_argument("input", help="Input control map path or '-' for stdin")
    parser.add_argument(
        "--key", default="value", help="Control column/key to read (default: value)"
    )
    parser.add_argument(
        "--output-key",
        default=None,
        help="Output control key. Default: same as --key",
    )
    parser.add_argument(
        "--operation", choices=list(RESHAPE_OPERATIONS), required=True, help="Reshape operation"
    )
    parser.add_argument(
        "--factor", type=float, default=1.0, help="Scale factor for scale/time-scale ops"
    )
    parser.add_argument(
        "--offset", type=float, default=0.0, help="Offset for offset/time-shift ops"
    )
    parser.add_argument(
        "--min", dest="min_value", type=float, default=None, help="Minimum clamp value"
    )
    parser.add_argument(
        "--max", dest="max_value", type=float, default=None, help="Maximum clamp value"
    )
    parser.add_argument("--exponent", type=float, default=1.0, help="Exponent for pow operation")
    parser.add_argument("--window", type=int, default=5, help="Window size for smooth operation")
    parser.add_argument(
        "--target-min", type=float, default=0.0, help="Target minimum for normalize operation"
    )
    parser.add_argument(
        "--target-max", type=float, default=1.0, help="Target maximum for normalize operation"
    )
    parser.add_argument(
        "--rate", type=float, default=20.0, help="Resample control rate (Hz) for resample operation"
    )
    parser.add_argument(
        "--interp", choices=list(INTERP_CHOICES), default="linear", help="Interpolation mode"
    )
    parser.add_argument(
        "--order", type=int, default=3, help="Polynomial order for --interp polynomial"
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "csv", "json"],
        default="auto",
        help="Input format override (default: auto by suffix or csv for stdin)",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "json"],
        default="auto",
        help="Output format (default: auto from --output suffix or csv for stdout)",
    )
    parser.add_argument(
        "--output", "--out", type=Path, default=None, help="Output file path. Default: stdout."
    )
    parser.add_argument("--stdout", action="store_true", help="Write reshaped map to stdout")
    add_console_args(parser)
    return parser


def _resolve_output_format(args: argparse.Namespace) -> str:
    fmt = str(getattr(args, "format", "auto")).strip().lower()
    if fmt in {"csv", "json"}:
        return fmt
    out = getattr(args, "output", None)
    if out is None:
        return "csv"
    return "json" if Path(out).suffix.lower() == ".json" else "csv"


def _resolve_input_format(args: argparse.Namespace, input_token: str) -> str:
    fmt = str(getattr(args, "input_format", "auto")).strip().lower()
    if fmt in {"csv", "json"}:
        return fmt
    if input_token == "-":
        return "csv"
    path = Path(input_token)
    return "json" if path.suffix.lower() == ".json" else "csv"


def _render_payload(times, values, *, key: str, output_format: str) -> str:
    if output_format == "json":
        return dump_control_points_json(times, values, key=key)
    return dump_control_points_csv(times, values, key=key)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if str(args.key).strip() == "":
        parser.error("--key must not be empty")
    output_key = str(args.output_key or args.key).strip()
    if output_key == "":
        parser.error("--output-key must not be empty")
    if int(args.window) < 1:
        parser.error("--window must be >= 1")
    if int(args.order) < 1:
        parser.error("--order must be >= 1")
    if str(args.operation) == "normalize" and float(args.target_max) < float(args.target_min):
        parser.error("--target-max must be >= --target-min")
    if str(args.operation) == "resample" and float(args.rate) <= 0.0:
        parser.error("--rate must be > 0 for resample operation")

    input_token = str(args.input).strip()
    input_format = _resolve_input_format(args, input_token)
    output_format = _resolve_output_format(args)
    write_stdout = bool(args.stdout or args.output is None)

    status = build_status_bar(args, "pvxreshape", 2)
    if input_token == "-":
        payload = sys.stdin.read()
        times, values = parse_control_points_payload(
            payload,
            key=str(args.key),
            source_label="stdin",
            fmt=input_format,
        )
    else:
        times, values = load_control_points(
            Path(input_token),
            key=str(args.key),
            fmt=input_format,
        )
    status.step(1, "loaded")

    out_t, out_v = reshape_control_points(
        times,
        values,
        operation=str(args.operation),  # type: ignore[arg-type]
        factor=float(args.factor),
        offset=float(args.offset),
        min_value=args.min_value,
        max_value=args.max_value,
        exponent=float(args.exponent),
        window=int(args.window),
        target_min=float(args.target_min),
        target_max=float(args.target_max),
        interp=str(args.interp),  # type: ignore[arg-type]
        order=int(args.order),
        resample_rate_hz=float(args.rate),
    )
    status.step(2, "reshaped")
    status.finish("done")

    payload_out = _render_payload(out_t, out_v, key=output_key, output_format=output_format)
    if write_stdout:
        sys.stdout.write(payload_out)
        sys.stdout.flush()
        log_message(
            args,
            (
                f"[done] pvxreshape operation={args.operation} "
                f"points_in={int(times.size)} points_out={int(out_t.size)} "
                f"key={output_key} output=stdout"
            ),
            min_level="verbose",
        )
        return 0

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload_out, encoding="utf-8")
    log_message(
        args,
        (
            f"[done] pvxreshape operation={args.operation} "
            f"points_in={int(times.size)} points_out={int(out_t.size)} "
            f"key={output_key} output={out_path}"
        ),
        min_level="normal",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
