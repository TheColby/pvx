#!/usr/bin/env python3

"""PVC-inspired harmonic/chord spectral mapping CLI."""

from __future__ import annotations

import argparse

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
from pvx.core.pvc_harmony import (
    CHORD_INTERVALS_SEMITONES,
    HarmonyOperatorName,
    process_harmony_operator,
)

OPERATOR_CHOICES: tuple[HarmonyOperatorName, ...] = ("chordmapper", "inharmonator")


def build_parser(
    default_operator: HarmonyOperatorName = "chordmapper", prog: str = "pvx chordmapper"
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Harmonic/chord mapping and inharmonic spectral warping.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx chordmapper input.wav --root-hz 220 --chord minor --strength 0.75 --output chordmapped.wav",
                "pvx inharmonator input.wav --inharmonic-f0-hz 220 --inharmonicity 0.0002 --inharmonic-mix 1.0 --output inharm.wav",
            ]
        ),
    )
    add_common_io_args(parser, default_suffix="_harm")
    add_vocoder_args(parser, default_n_fft=4096, default_win_length=4096, default_hop_size=512)
    parser.add_argument(
        "--operator",
        choices=list(OPERATOR_CHOICES),
        default=default_operator,
        help=f"Harmony operator (default: {default_operator})",
    )

    parser.add_argument("--root-hz", type=float, default=220.0, help="Chord root frequency in Hz")
    parser.add_argument(
        "--chord",
        choices=sorted(CHORD_INTERVALS_SEMITONES),
        default="major",
        help="Chord quality for chordmapper",
    )
    parser.add_argument(
        "--strength", type=float, default=0.75, help="Chordmapper emphasis strength in [0,1]"
    )
    parser.add_argument(
        "--tolerance-cents", type=float, default=35.0, help="Chordmapper tolerance in cents"
    )
    parser.add_argument(
        "--boost-db", type=float, default=6.0, help="Chordmapper in-chord boost in dB"
    )
    parser.add_argument(
        "--attenuation",
        type=float,
        default=0.45,
        help="Chordmapper out-of-chord attenuation in [0,1]",
    )

    parser.add_argument(
        "--inharmonic-f0-hz", type=float, default=220.0, help="Reference f0 for inharmonator"
    )
    parser.add_argument(
        "--inharmonicity", type=float, default=1e-4, help="Inharmonicity coefficient B"
    )
    parser.add_argument(
        "--inharmonic-mix", type=float, default=1.0, help="Inharmonator mix in [0,1]"
    )
    parser.add_argument("--dry-mix", type=float, default=0.0, help="Dry signal mix in [0,1]")
    return parser


def run_harmmap_cli(
    argv: list[str] | None = None,
    *,
    default_operator: HarmonyOperatorName = "chordmapper",
    prog: str = "pvx chordmapper",
) -> int:
    parser = build_parser(default_operator=default_operator, prog=prog)
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)

    if float(args.root_hz) <= 0.0:
        parser.error("--root-hz must be > 0")
    if not (0.0 <= float(args.strength) <= 1.0):
        parser.error("--strength must be in [0,1]")
    if float(args.tolerance_cents) <= 0.0:
        parser.error("--tolerance-cents must be > 0")
    if not (0.0 <= float(args.attenuation) <= 1.0):
        parser.error("--attenuation must be in [0,1]")
    if float(args.inharmonic_f0_hz) <= 0.0:
        parser.error("--inharmonic-f0-hz must be > 0")
    if float(args.inharmonicity) < 0.0:
        parser.error("--inharmonicity must be >= 0")
    if not (0.0 <= float(args.inharmonic_mix) <= 1.0):
        parser.error("--inharmonic-mix must be in [0,1]")
    if not (0.0 <= float(args.dry_mix) <= 1.0):
        parser.error("--dry-mix must be in [0,1]")

    config = build_vocoder_config(args, phase_locking="identity", transient_preserve=False)
    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, f"pvx{args.operator!s}", len(paths))
    failures = 0

    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            out = process_harmony_operator(
                audio,
                sr,
                config,
                operator=str(args.operator),  # type: ignore[arg-type]
                root_hz=float(args.root_hz),
                chord=str(args.chord),
                strength=float(args.strength),
                tolerance_cents=float(args.tolerance_cents),
                boost_db=float(args.boost_db),
                attenuation=float(args.attenuation),
                inharmonicity=float(args.inharmonicity),
                inharmonic_f0_hz=float(args.inharmonic_f0_hz),
                inharmonic_mix=float(args.inharmonic_mix),
                dry_mix=float(args.dry_mix),
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
        args, f"[done] pvxharmmap processed={len(paths)} failed={failures}", min_level="normal"
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    return run_harmmap_cli(argv, default_operator="chordmapper", prog="pvx chordmapper")


if __name__ == "__main__":
    raise SystemExit(main())
