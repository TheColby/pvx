#!/usr/bin/env python3
"""Multi-voice harmonizer built from phase-vocoder pitch shifts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvx.core.common import (
    add_common_io_args,
    add_vocoder_args,
    build_status_bar,
    build_vocoder_config,
    default_output_path,
    ensure_runtime,
    finalize_audio,
    log_error,
    log_message,
    parse_float_list,
    read_audio,
    resolve_inputs,
    semitone_to_ratio,
    time_pitch_shift_audio,
    validate_vocoder_args,
    write_output,
)


def pan_stereo(signal: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    left = np.sqrt(0.5 * (1.0 - pan))
    right = np.sqrt(0.5 * (1.0 + pan))
    mono = np.mean(signal, axis=1)
    return np.stack([mono * left, mono * right], axis=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate harmonized voices from an input")
    add_common_io_args(parser, default_suffix="_harm")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--intervals", default="0,4,7", help="Comma-separated semitone intervals per voice (supports fractional values)")
    parser.add_argument(
        "--intervals-cents",
        default="",
        help="Optional cents offsets per voice, added to --intervals (e.g. 0,14,-12)",
    )
    parser.add_argument("--gains", default="", help="Optional comma-separated linear gain per voice")
    parser.add_argument("--pans", default="", help="Optional comma-separated pan per voice [-1..1]")
    parser.add_argument("--force-stereo", action="store_true", help="Mix result as stereo with panning")
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)

    intervals = parse_float_list(args.intervals)
    if not intervals:
        parser.error("--intervals requires at least one value")
    interval_cents = parse_float_list(args.intervals_cents, allow_empty=True) if args.intervals_cents else [0.0] * len(intervals)
    if len(interval_cents) < len(intervals):
        interval_cents = interval_cents + [0.0] * (len(intervals) - len(interval_cents))
    interval_cents = interval_cents[: len(intervals)]
    pitch_steps = [semi + (cent / 100.0) for semi, cent in zip(intervals, interval_cents)]

    gains = parse_float_list(args.gains, allow_empty=True) if args.gains else [1.0] * len(intervals)
    if len(gains) < len(intervals):
        gains = gains + [gains[-1] if gains else 1.0] * (len(intervals) - len(gains))
    gains = gains[: len(intervals)]

    pans = parse_float_list(args.pans, allow_empty=True) if args.pans else list(np.linspace(-0.7, 0.7, num=len(intervals)))
    if len(pans) < len(intervals):
        pans = pans + [0.0] * (len(intervals) - len(pans))
    pans = [float(np.clip(v, -1.0, 1.0)) for v in pans[: len(intervals)]]

    config = build_vocoder_config(args, phase_locking="identity", transient_preserve=True, transient_threshold=2.0)
    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, "pvxharmonize", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            voices: list[np.ndarray] = []
            for semi, gain in zip(pitch_steps, gains):
                ratio = semitone_to_ratio(semi)
                shifted = time_pitch_shift_audio(audio, 1.0, ratio, config, resample_mode=args.resample_mode)
                voices.append(shifted * gain)

            out_len = max(v.shape[0] for v in voices)
            channel_count = 2 if args.force_stereo or audio.shape[1] == 1 else audio.shape[1]
            out = np.zeros((out_len, channel_count), dtype=np.float64)

            for voice, pan in zip(voices, pans):
                if channel_count == 2:
                    stereo = pan_stereo(voice, pan) if voice.shape[1] != 2 else voice
                    out[: stereo.shape[0], :2] += stereo[:, :2]
                else:
                    out[: voice.shape[0], : voice.shape[1]] += voice

            out = finalize_audio(out, sr, args)
            out_path = default_output_path(path, args)
            write_output(out_path, out, sr, args)
            log_message(args, f"[ok] {path} -> {out_path} | voices={len(intervals)}, ch={out.shape[1]}", min_level="verbose")
        except Exception as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)
    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(args, f"[done] pvxharmonize processed={len(paths)} failed={failures}", min_level="normal")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
