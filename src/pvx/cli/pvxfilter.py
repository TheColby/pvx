#!/usr/bin/env python3

"""PVC-inspired response-driven spectral operators."""

from __future__ import annotations

import argparse
from pathlib import Path

from pvx.core.common import (
    add_common_io_args,
    add_vocoder_args,
    build_examples_epilog,
    build_status_bar,
    build_vocoder_config,
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
from pvx.core.pvc_ops import (
    InterpMode,
    OperatorName,
    load_scalar_control_points,
    process_response_operator,
)
from pvx.core.response_store import load_response_artifact

OPERATOR_CHOICES: tuple[OperatorName, ...] = (
    "filter",
    "tvfilter",
    "noisefilter",
    "bandamp",
    "spec-compander",
)
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
    default_operator: OperatorName = "filter", prog: str = "pvx filter"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Response-driven spectral processing using PVXRF artifacts.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx filter input.wav --response input.pvxrf.npz --response-mix 1.0 --output filtered.wav",
                "pvx tvfilter input.wav --response input.pvxrf.npz --tv-map mix_map.csv --tv-interp linear --output tvfiltered.wav",
                "pvx noisefilter noisy.wav --response noise_profile.pvxrf.npz --noise-floor 1.2 --output denoised.wav",
                "pvx bandamp source.wav --response profile.pvxrf.npz --band-gain-db 8 --peak-count 10 --output emphasized.wav",
                "pvx spec-compander source.wav --response profile.pvxrf.npz --comp-threshold-db -18 --comp-ratio 2.5 --output shaped.wav",
            ]
        ),
    )
    add_common_io_args(parser, default_suffix="_filter")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument(
        "--operator",
        choices=list(OPERATOR_CHOICES),
        default=default_operator,
        help=f"Processing operator (default: {default_operator})",
    )
    parser.add_argument(
        "--response", type=Path, required=True, help="Input PVXRF response artifact"
    )
    parser.add_argument(
        "--response-mix", type=float, default=1.0, help="Wet response mix amount (default: 1.0)"
    )
    parser.add_argument(
        "--dry-mix", type=float, default=0.0, help="Dry signal mix amount in [0,1] (default: 0.0)"
    )
    parser.add_argument(
        "--response-gain-db",
        type=float,
        default=0.0,
        help="Global response gain in dB (default: 0)",
    )
    parser.add_argument(
        "--transpose-semitones",
        type=float,
        default=0.0,
        help="Transpose response curve in semitones",
    )
    parser.add_argument(
        "--shift-bins", type=int, default=0, help="Shift response curve by FFT bins"
    )

    parser.add_argument(
        "--tv-map",
        type=Path,
        default=None,
        help="Optional CSV/JSON map for time-varying response_mix",
    )
    parser.add_argument(
        "--tv-key", default="response_mix", help="Column/key in --tv-map (default: response_mix)"
    )
    parser.add_argument(
        "--tv-interp", choices=list(INTERP_CHOICES), default="linear", help="TV interpolation mode"
    )
    parser.add_argument(
        "--tv-order", type=int, default=3, help="Polynomial order for --tv-interp polynomial"
    )

    parser.add_argument(
        "--noise-floor", type=float, default=1.0, help="Threshold scale for noisefilter"
    )
    parser.add_argument(
        "--band-gain-db", type=float, default=6.0, help="Band boost amount for bandamp (dB)"
    )
    parser.add_argument(
        "--band-width-bins", type=int, default=6, help="Band width in bins for bandamp"
    )
    parser.add_argument(
        "--peak-count", type=int, default=8, help="Number of response peaks for bandamp"
    )
    parser.add_argument(
        "--comp-threshold-db",
        type=float,
        default=-18.0,
        help="Relative threshold for spec-compander",
    )
    parser.add_argument(
        "--comp-ratio", type=float, default=2.0, help="Compression ratio for spec-compander"
    )
    parser.add_argument(
        "--expand-ratio", type=float, default=1.2, help="Expansion ratio for spec-compander"
    )
    return parser


def run_filter_cli(
    argv: list[str] | None = None,
    *,
    default_operator: OperatorName = "filter",
    prog: str = "pvx filter",
) -> int:
    parser = build_parser(default_operator=default_operator, prog=prog)
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)

    if not (0.0 <= float(args.dry_mix) <= 1.0):
        parser.error("--dry-mix must be in [0, 1]")
    if float(args.noise_floor) <= 0.0:
        parser.error("--noise-floor must be > 0")
    if int(args.band_width_bins) <= 0:
        parser.error("--band-width-bins must be > 0")
    if int(args.peak_count) <= 0:
        parser.error("--peak-count must be > 0")
    if float(args.comp_ratio) < 1.0:
        parser.error("--comp-ratio must be >= 1")
    if float(args.expand_ratio) < 1.0:
        parser.error("--expand-ratio must be >= 1")
    if int(args.tv_order) < 1:
        parser.error("--tv-order must be >= 1")

    response = load_response_artifact(Path(args.response).expanduser().resolve())
    tv_t, tv_v = load_scalar_control_points(
        Path(args.tv_map).expanduser().resolve() if args.tv_map is not None else None,
        key=str(args.tv_key),
        default_value=float(args.response_mix),
    )
    config = build_vocoder_config(args, phase_locking="off", transient_preserve=False)
    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, f"pvx{str(args.operator).replace('-', '')}", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            out = process_response_operator(
                audio,
                sr,
                config,
                response,
                operator=str(args.operator),  # type: ignore[arg-type]
                response_mix=float(args.response_mix),
                dry_mix=float(args.dry_mix),
                response_gain_db=float(args.response_gain_db),
                transpose_semitones=float(args.transpose_semitones),
                shift_bins=int(args.shift_bins),
                tv_points_t=tv_t,
                tv_points_v=tv_v,
                tv_interp=str(args.tv_interp),  # type: ignore[arg-type]
                tv_order=int(args.tv_order),
                noise_floor=float(args.noise_floor),
                band_gain_db=float(args.band_gain_db),
                band_width_bins=int(args.band_width_bins),
                peak_count=int(args.peak_count),
                comp_threshold_db=float(args.comp_threshold_db),
                comp_ratio=float(args.comp_ratio),
                expand_ratio=float(args.expand_ratio),
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
        args, f"[done] pvxfilter processed={len(paths)} failed={failures}", min_level="normal"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    return run_filter_cli(argv, default_operator="filter", prog="pvx filter")


if __name__ == "__main__":
    raise SystemExit(main())
