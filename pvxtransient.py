#!/usr/bin/env python3
"""Transient-aware time/pitch processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvxcommon import (
    add_common_io_args,
    add_vocoder_args,
    build_vocoder_config,
    default_output_path,
    ensure_runtime,
    finalize_audio,
    read_audio,
    resolve_inputs,
    semitone_to_ratio,
    time_pitch_shift_channel,
    validate_vocoder_args,
    write_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transient-preserving phase-vocoder processor")
    add_common_io_args(parser, default_suffix="_trans")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=256)
    parser.add_argument("--time-stretch", type=float, default=1.0)
    parser.add_argument("--target-duration", type=float, default=None, help="Target duration in seconds")
    parser.add_argument("--pitch-shift-semitones", type=float, default=0.0)
    parser.add_argument("--pitch-shift-ratio", type=float, default=None)
    parser.add_argument("--transient-threshold", type=float, default=1.6)
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.time_stretch <= 0:
        parser.error("--time-stretch must be > 0")
    if args.target_duration is not None and args.target_duration <= 0:
        parser.error("--target-duration must be > 0")
    if args.pitch_shift_ratio is not None and args.pitch_shift_ratio <= 0:
        parser.error("--pitch-shift-ratio must be > 0")
    if args.transient_threshold <= 0:
        parser.error("--transient-threshold must be > 0")

    pitch_ratio = args.pitch_shift_ratio if args.pitch_shift_ratio is not None else semitone_to_ratio(args.pitch_shift_semitones)
    config = build_vocoder_config(
        args,
        phase_locking="identity",
        transient_preserve=True,
        transient_threshold=args.transient_threshold,
    )
    paths = resolve_inputs(args.inputs, parser)

    failures = 0
    for path in paths:
        try:
            audio, sr = read_audio(path)
            stretch = args.time_stretch
            if args.target_duration is not None:
                stretch = (args.target_duration * sr) / max(1, audio.shape[0])

            channels: list[np.ndarray] = []
            for ch in range(audio.shape[1]):
                channels.append(
                    time_pitch_shift_channel(
                        audio[:, ch],
                        stretch=stretch,
                        pitch_ratio=pitch_ratio,
                        config=config,
                        resample_mode=args.resample_mode,
                    )
                )
            out_len = max(ch.size for ch in channels)
            out = np.zeros((out_len, len(channels)), dtype=np.float64)
            for idx, ch in enumerate(channels):
                out[: ch.size, idx] = ch
            out = finalize_audio(out, args)

            out_path = default_output_path(path, args)
            write_output(out_path, out, sr, args)
            if args.verbose:
                print(f"[ok] {path} -> {out_path} | stretch={stretch:.4f} pitch={pitch_ratio:.4f}")
        except Exception as exc:
            failures += 1
            print(f"[error] {path}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

