#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Trajectory-aware multichannel convolution reverb for mono sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvx.core.common import (
    add_common_io_args,
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
    write_output,
)
from pvx.core.spatial_reverb import (
    apply_multichannel_trajectory_reverb,
    compute_trajectory_gains,
    default_speaker_angles,
    parse_coordinate,
    parse_speaker_angles,
    resample_audio_linear,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convolve mono source with multichannel impulse response and move source from A to B.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                (
                    "pvx trajectory-reverb source.wav --ir room_4ch.wav "
                    "--coord-system cartesian --start -1,0,1 --end 1,0,1 --output flythrough.wav"
                ),
                (
                    "pvx trajectory-reverb voice.wav --ir ambeo_ir.wav "
                    "--coord-system spherical --start -90,0,1.5 --end 90,0,1.5 --trajectory-shape ease-in-out "
                    "--output voice_orbit.wav"
                ),
                (
                    "pvx trajectory-reverb source.wav --ir room_4ch.wav "
                    "--speaker-angles \"-45,0;45,0;135,0;-135,0\" --wet 0.9 --dry 0.2 "
                    "--stdout | pvx deverb - --strength 0.25 --output flythrough_clean.wav"
                ),
            ],
            notes=[
                "Use cartesian coordinates as x,y,z or spherical as azimuth_deg,elevation_deg,radius.",
                "This workflow expects mono source input and multichannel impulse response.",
            ],
        ),
    )
    add_common_io_args(parser, default_suffix="_trajrev")
    parser.add_argument(
        "--ir",
        type=Path,
        required=True,
        help="Multichannel impulse response path (for example 4-channel room capture).",
    )
    parser.add_argument(
        "--coord-system",
        choices=["cartesian", "spherical"],
        default="cartesian",
        help="Coordinate format for --start/--end (default: cartesian)",
    )
    parser.add_argument(
        "--start",
        nargs="?",
        required=True,
        help="Start position. Cartesian: x,y,z. Spherical: az_deg,el_deg,r.",
    )
    parser.add_argument(
        "--end",
        nargs="?",
        required=True,
        help="End position. Cartesian: x,y,z. Spherical: az_deg,el_deg,r.",
    )
    parser.add_argument(
        "--speaker-angles",
        default=None,
        help=(
            "Optional speaker layout azimuth/elevation list in degrees as "
            "\"az,el;az,el;...\". Count must match impulse-response channels."
        ),
    )
    parser.add_argument(
        "--trajectory-shape",
        choices=["linear", "ease-in", "ease-out", "ease-in-out"],
        default="linear",
        help="Interpolation shape from start to end (default: linear)",
    )
    parser.add_argument(
        "--distance-law",
        choices=["none", "inverse", "inverse-square"],
        default="inverse",
        help="Distance gain law applied across trajectory (default: inverse)",
    )
    parser.add_argument(
        "--normalize-gains",
        action="store_true",
        default=True,
        help="Normalize per-sample channel gains before distance weighting (default: on)",
    )
    parser.add_argument(
        "--no-normalize-gains",
        dest="normalize_gains",
        action="store_false",
        help="Disable per-sample gain normalization.",
    )
    parser.add_argument("--wet", type=float, default=1.0, help="Wet reverb mix scalar (default: 1.0)")
    parser.add_argument("--dry", type=float, default=0.0, help="Dry direct mix scalar (default: 0.0)")
    return parser


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    return np.mean(arr, axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)

    if not (0.0 <= float(args.wet) <= 2.0):
        parser.error("--wet must be in [0, 2]")
    if not (0.0 <= float(args.dry) <= 2.0):
        parser.error("--dry must be in [0, 2]")

    try:
        start_xyz = parse_coordinate(str(args.start), str(args.coord_system))
        end_xyz = parse_coordinate(str(args.end), str(args.coord_system))
    except Exception as exc:
        parser.error(f"Invalid trajectory coordinates: {exc}")

    ir_audio, ir_sr = read_audio(Path(args.ir).expanduser().resolve())
    ir_ch = int(ir_audio.shape[1]) if ir_audio.ndim == 2 else 1
    if ir_ch <= 0:
        parser.error("--ir must contain at least one channel")

    if args.speaker_angles:
        try:
            speaker_angles = parse_speaker_angles(str(args.speaker_angles), ir_ch)
        except Exception as exc:
            parser.error(f"Invalid --speaker-angles: {exc}")
    else:
        speaker_angles = default_speaker_angles(ir_ch)

    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, "pvxtrajectoryreverb", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            src_audio, src_sr = read_audio(path)
            mono = _to_mono(src_audio)
            ir = np.asarray(ir_audio, dtype=np.float64)
            if int(ir_sr) != int(src_sr):
                ir = resample_audio_linear(ir, int(ir_sr), int(src_sr))
                log_message(
                    args,
                    f"[info] resampled IR from {ir_sr} Hz to {src_sr} Hz for {path.name}",
                    min_level="verbose",
                )

            gains = compute_trajectory_gains(
                mono.size,
                ir.shape[1] if ir.ndim == 2 else 1,
                start_xyz=start_xyz,
                end_xyz=end_xyz,
                speaker_angles_deg=speaker_angles,
                shape=str(args.trajectory_shape),
                distance_law=str(args.distance_law),
                normalize_per_sample=bool(args.normalize_gains),
            )
            out = apply_multichannel_trajectory_reverb(
                source_mono=mono,
                impulse_response=ir,
                gains=gains,
                wet=float(args.wet),
                dry=float(args.dry),
            )
            out = finalize_audio(out, int(src_sr), args)
            out_path = default_output_path(path, args)
            print_input_output_metrics_table(
                args,
                input_label=str(path),
                input_audio=src_audio,
                input_sr=int(src_sr),
                output_label=str(out_path),
                output_audio=out,
                output_sr=int(src_sr),
            )
            write_output(out_path, out, int(src_sr), args, input_path=path)
            log_message(
                args,
                (
                    f"[ok] {path} -> {out_path} | ir_ch={ir.shape[1] if ir.ndim == 2 else 1} "
                    f"shape={args.trajectory_shape} law={args.distance_law}"
                ),
                min_level="verbose",
            )
        except Exception as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)
    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(args, f"[done] pvxtrajectoryreverb processed={len(paths)} failed={failures}", min_level="normal")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
