#!/usr/bin/env python3
"""Create unison width via micro-detuned phase-vocoder voices."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from pvxcommon import (
    add_common_io_args,
    add_vocoder_args,
    build_status_bar,
    build_vocoder_config,
    default_output_path,
    ensure_runtime,
    finalize_audio,
    log_error,
    log_message,
    read_audio,
    resolve_inputs,
    time_pitch_shift_audio,
    validate_vocoder_args,
    write_output,
)


def cents_to_ratio(cents: float) -> float:
    return float(2.0 ** (cents / 1200.0))


def pan_gains(pan: float) -> tuple[float, float]:
    p = float(np.clip(pan, -1.0, 1.0))
    return math.sqrt(0.5 * (1.0 - p)), math.sqrt(0.5 * (1.0 + p))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stereo unison thickener")
    add_common_io_args(parser, default_suffix="_unison")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--voices", type=int, default=5, help="Number of unison voices")
    parser.add_argument("--detune-cents", type=float, default=14.0, help="Total detune span in cents")
    parser.add_argument("--width", type=float, default=1.0, help="Stereo width multiplier 0..2")
    parser.add_argument("--dry-mix", type=float, default=0.2, help="Dry signal mix amount")
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.voices <= 0:
        parser.error("--voices must be > 0")
    if args.detune_cents < 0:
        parser.error("--detune-cents must be >= 0")
    if args.width < 0:
        parser.error("--width must be >= 0")
    if args.dry_mix < 0:
        parser.error("--dry-mix must be >= 0")

    config = build_vocoder_config(args, phase_locking="identity", transient_preserve=True, transient_threshold=2.0)
    paths = resolve_inputs(args.inputs, parser)
    status = build_status_bar(args, "pvxunison", len(paths))

    failures = 0
    rng = np.random.default_rng(1307)
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            base = np.mean(audio, axis=1, keepdims=True)
            voice_offsets = np.linspace(-args.detune_cents * 0.5, args.detune_cents * 0.5, num=args.voices)
            voice_offsets += rng.normal(scale=max(0.0, args.detune_cents * 0.04), size=args.voices)
            pans = np.linspace(-1.0, 1.0, num=args.voices) * np.clip(args.width, 0.0, 2.0) * 0.8

            out_len = base.shape[0]
            mix = np.zeros((out_len, 2), dtype=np.float64)
            for cents, pan in zip(voice_offsets, pans):
                ratio = cents_to_ratio(float(cents))
                voice = time_pitch_shift_audio(base, stretch=1.0, pitch_ratio=ratio, config=config, resample_mode=args.resample_mode)
                left, right = pan_gains(float(pan))
                tmp = np.zeros((out_len, 2), dtype=np.float64)
                tmp[: voice.shape[0], 0] = voice[:, 0] * left
                tmp[: voice.shape[0], 1] = voice[:, 0] * right
                mix += tmp

            dry = np.zeros((out_len, 2), dtype=np.float64)
            dry[:, 0] = base[:, 0]
            dry[:, 1] = base[:, 0]
            out = mix + (args.dry_mix * dry)
            out = finalize_audio(out, args)

            out_path = default_output_path(path, args)
            write_output(out_path, out, sr, args)
            log_message(args, f"[ok] {path} -> {out_path} | voices={args.voices}", min_level="verbose")
        except Exception as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)
    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(args, f"[done] pvxunison processed={len(paths)} failed={failures}", min_level="normal")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
