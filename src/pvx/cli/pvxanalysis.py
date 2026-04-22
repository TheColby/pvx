#!/usr/bin/env python3

"""Persist and inspect reusable phase-vocoder analysis artifacts (PVXAN)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pvx.core.analysis_store import (
    analyze_audio,
    load_analysis_artifact,
    save_analysis_artifact,
    summarize_analysis_artifact,
)
from pvx.core.common import (
    add_console_args,
    build_examples_epilog,
    build_status_bar,
    ensure_runtime,
    is_silent,
    log_error,
    log_message,
    read_audio,
)
from pvx.core.voc import (
    TRANSFORM_CHOICES,
    WINDOW_CHOICES,
    VocoderConfig,
    validate_transform_available,
)


def _normalize_argv(argv: list[str] | None) -> list[str]:
    tokens = list(argv or [])
    if not tokens:
        return tokens
    if tokens[0] in {"create", "inspect", "--help", "-h"}:
        return tokens
    return ["create", *tokens]


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".pvxan.npz")


def _print_summary(summary: dict[str, object]) -> None:
    width = max(len(key) for key in summary) if summary else 0
    for key, value in summary.items():
        print(f"{key:<{width}} : {value}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvx analysis",
        description="Create and inspect PVXAN analysis artifacts for reusable phase-vocoder workflows.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx analysis create input.wav --output input.pvxan.npz --n-fft 4096 --hop-size 256",
                "pvx analysis inspect input.pvxan.npz",
                "pvxanalysis input.wav --output input.pvxan.npz",
            ],
            notes=[
                "When command is omitted, create mode is assumed for convenience.",
                "PVXAN stores complex STFT payloads (channels x frames x bins) in compressed NPZ format.",
            ],
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create",
        help="Analyze audio and save PVXAN artifact",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    create_parser.add_argument("input", type=Path, help="Input audio path")
    create_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PVXAN path (default: <input>.pvxan.npz)",
    )
    create_parser.add_argument(
        "--analysis-channel",
        type=int,
        default=-1,
        help="Channel index to analyze; -1 means all channels (default: -1)",
    )
    create_parser.add_argument(
        "--n-fft", type=int, default=2048, help="STFT FFT size (default: 2048)"
    )
    create_parser.add_argument(
        "--win-length", type=int, default=2048, help="Window length (default: 2048)"
    )
    create_parser.add_argument("--hop-size", type=int, default=512, help="Hop size (default: 512)")
    create_parser.add_argument(
        "--window", choices=list(WINDOW_CHOICES), default="hann", help="Window type"
    )
    create_parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=14.0,
        help="Kaiser beta when --window kaiser (default: 14.0)",
    )
    create_parser.add_argument(
        "--transform",
        choices=list(TRANSFORM_CHOICES),
        default="fft",
        help="Transform backend (default: fft)",
    )
    create_parser.add_argument("--no-center", action="store_true", help="Disable centered framing")
    create_parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    add_console_args(create_parser)
    create_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device selection (default: auto)",
    )
    create_parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index for --device cuda/auto (default: 0)",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect existing PVXAN artifact",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    inspect_parser.add_argument("artifact", type=Path, help="PVXAN artifact path")
    inspect_parser.add_argument("--json", action="store_true", help="Print summary as JSON")
    inspect_parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write summary JSON",
    )
    add_console_args(inspect_parser)

    return parser


def _validate_create_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if int(args.n_fft) <= 0:
        parser.error("--n-fft must be > 0")
    if int(args.win_length) <= 0:
        parser.error("--win-length must be > 0")
    if int(args.hop_size) <= 0:
        parser.error("--hop-size must be > 0")
    if int(args.win_length) > int(args.n_fft):
        parser.error("--win-length must be <= --n-fft")
    if int(args.hop_size) > int(args.win_length):
        parser.error("--hop-size must be <= --win-length")
    if float(args.kaiser_beta) < 0:
        parser.error("--kaiser-beta must be >= 0")
    if int(args.analysis_channel) < -1:
        parser.error("--analysis-channel must be -1 or a non-negative index")
    validate_transform_available(str(args.transform), parser)


def _run_create(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    _validate_create_args(args, parser)
    ensure_runtime(args, parser)
    status = build_status_bar(args, "pvxanalysis", total=4)

    input_path = Path(args.input).expanduser().resolve()
    audio, sample_rate = read_audio(input_path)
    status.step(1, "read")

    channel_index = int(args.analysis_channel)
    source_audio = np.asarray(audio, dtype=np.float64)
    source_path = str(input_path)
    if channel_index >= 0:
        if channel_index >= source_audio.shape[1]:
            parser.error(
                f"--analysis-channel={channel_index} out of range for input with {source_audio.shape[1]} channels"
            )
        source_audio = source_audio[:, channel_index : channel_index + 1]
        source_path = f"{input_path}#ch{channel_index}"
    status.step(2, "prep")

    config = VocoderConfig(
        n_fft=int(args.n_fft),
        win_length=int(args.win_length),
        hop_size=int(args.hop_size),
        window=str(args.window),
        center=not bool(args.no_center),
        phase_locking="off",
        transient_preserve=False,
        transient_threshold=2.0,
        kaiser_beta=float(args.kaiser_beta),
        transform=str(args.transform),
    )
    artifact = analyze_audio(
        source_audio,
        sample_rate,
        config,
        source_path=source_path,
    )
    status.step(3, "analyze")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path)
    )
    output_path = save_analysis_artifact(output_path, artifact)
    summary = summarize_analysis_artifact(artifact)
    status.step(4, "save")
    status.finish("done")

    if args.summary_json is not None:
        _write_json(Path(args.summary_json).expanduser().resolve(), summary)
        log_message(
            args,
            f"[ok] wrote summary JSON -> {Path(args.summary_json).expanduser().resolve()}",
            min_level="normal",
        )
    if not is_silent(args):
        if bool(getattr(args, "json", False)):
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print("PVXAN summary")
            _print_summary(summary)
    log_message(args, f"[ok] saved analysis -> {output_path}", min_level="normal")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    status = build_status_bar(args, "pvxanalysis-inspect", total=2)
    artifact = load_analysis_artifact(Path(args.artifact).expanduser().resolve())
    status.step(1, "load")
    summary = summarize_analysis_artifact(artifact)
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
            print("PVXAN summary")
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
        log_error(args, f"[error] pvxanalysis: {exc}")
        return 1
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
