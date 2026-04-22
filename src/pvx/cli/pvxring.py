#!/usr/bin/env python3

"""PVC-inspired ring and resonator operators."""

from __future__ import annotations

import argparse
from pathlib import Path

from pvx.core.common import (
    add_common_io_args,
    add_vocoder_args,
    build_examples_epilog,
    build_status_bar,
    default_output_path,
    ensure_runtime,
    finalize_audio,
    log_error,
    log_message,
    print_input_output_metrics_table,
    read_audio,
    resolve_inputs,
    validate_vocoder_args,
    write_output,
)
from pvx.core.pvc_resonators import InterpMode, RingOperatorName, process_ring_operator

OPERATOR_CHOICES: tuple[RingOperatorName, ...] = ("ring", "ringfilter", "ringtvfilter")
INTERP_CHOICES: tuple[InterpMode, ...] = (
    "none",
    "stairstep",
    "nearest",
    "linear",
    "cubic",
    "polynomial",
    "exponential",
    "s_curve",
    "smootherstep",
)


def build_parser(
    default_operator: RingOperatorName = "ring", prog: str = "pvx ring"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Ring modulation and resonator filtering with PVC-style controls.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx ring input.wav --frequency-hz 43 --depth 0.9 --mix 0.8 --output ring.wav",
                "pvx ringfilter input.wav --frequency-hz 60 --resonance-hz 1200 --resonance-q 9 --output ringfilter.wav",
                "pvx ringtvfilter input.wav --tv-map ring_map.csv --tv-interp linear --output ringtv.wav",
            ]
        ),
    )
    add_common_io_args(parser, default_suffix="_ring")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument(
        "--operator",
        choices=list(OPERATOR_CHOICES),
        default=default_operator,
        help=f"Ring operator (default: {default_operator})",
    )
    parser.add_argument("--frequency-hz", type=float, default=40.0, help="Carrier frequency in Hz")
    parser.add_argument("--depth", type=float, default=1.0, help="Ring modulation depth in [0,1]")
    parser.add_argument("--mix", type=float, default=1.0, help="Ring wet mix in [0,1]")
    parser.add_argument("--feedback", type=float, default=0.0, help="Feedback amount in [0,1)")
    parser.add_argument(
        "--resonance-hz", type=float, default=1200.0, help="Resonance center frequency (Hz)"
    )
    parser.add_argument("--resonance-q", type=float, default=8.0, help="Resonance quality factor Q")
    parser.add_argument(
        "--resonance-mix", type=float, default=0.35, help="Resonator wet mix in [0,1]"
    )
    parser.add_argument(
        "--resonance-decay", type=float, default=0.2, help="Resonator feedback/decay in [0,1)"
    )
    parser.add_argument(
        "--tv-map", type=Path, default=None, help="CSV/JSON time-varying map for ringtvfilter"
    )
    parser.add_argument(
        "--tv-interp", choices=list(INTERP_CHOICES), default="linear", help="TV interpolation mode"
    )
    parser.add_argument(
        "--tv-order", type=int, default=3, help="Polynomial order for --tv-interp polynomial"
    )
    return parser


def run_ring_cli(
    argv: list[str] | None = None,
    *,
    default_operator: RingOperatorName = "ring",
    prog: str = "pvx ring",
) -> int:
    parser = build_parser(default_operator=default_operator, prog=prog)
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)

    if not (0.0 <= float(args.depth) <= 1.0):
        parser.error("--depth must be in [0,1]")
    if not (0.0 <= float(args.mix) <= 1.0):
        parser.error("--mix must be in [0,1]")
    if not (0.0 <= float(args.feedback) < 1.0):
        parser.error("--feedback must be in [0,1)")
    if float(args.resonance_q) <= 0.0:
        parser.error("--resonance-q must be > 0")
    if not (0.0 <= float(args.resonance_mix) <= 1.0):
        parser.error("--resonance-mix must be in [0,1]")
    if not (0.0 <= float(args.resonance_decay) < 1.0):
        parser.error("--resonance-decay must be in [0,1)")
    if int(args.tv_order) < 1:
        parser.error("--tv-order must be >= 1")

    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, f"pvx{args.operator!s}", len(paths))
    failures = 0

    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            out = process_ring_operator(
                audio,
                sr,
                operator=str(args.operator),  # type: ignore[arg-type]
                frequency_hz=float(args.frequency_hz),
                depth=float(args.depth),
                mix=float(args.mix),
                feedback=float(args.feedback),
                resonance_hz=float(args.resonance_hz),
                resonance_q=float(args.resonance_q),
                resonance_mix=float(args.resonance_mix),
                resonance_decay=float(args.resonance_decay),
                tv_map_path=Path(args.tv_map).expanduser().resolve()
                if args.tv_map is not None
                else None,
                tv_interp=str(args.tv_interp),  # type: ignore[arg-type]
                tv_order=int(args.tv_order),
            )
            out = finalize_audio(out, sr, args)
            out_path = default_output_path(path, args)
            print_input_output_metrics_table(
                args,
                input_label=str(path),
                input_audio=audio,
                input_sr=sr,
                output_label=str(out_path),
                output_audio=out,
                output_sr=sr,
            )
            write_output(out_path, out, sr, args, input_path=path)
            log_message(args, f"[ok] {path} -> {out_path}", min_level="verbose")
        except (OSError, ValueError, RuntimeError) as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)

    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(
        args, f"[done] pvxring processed={len(paths)} failed={failures}", min_level="normal"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    return run_ring_cli(argv, default_operator="ring", prog="pvx ring")


if __name__ == "__main__":
    raise SystemExit(main())
