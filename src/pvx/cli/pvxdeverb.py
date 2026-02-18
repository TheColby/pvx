#!/usr/bin/env python3
"""Spectral tail suppression for dereverberation-like cleanup."""

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
    read_audio,
    resolve_inputs,
    validate_vocoder_args,
    write_output,
    print_input_output_metrics_table,
)
from pvx.core.voc import istft, stft


def deverb_channel(
    signal: np.ndarray,
    config,
    *,
    strength: float,
    decay: float,
    floor: float,
) -> np.ndarray:
    spec = stft(signal, config)
    mag = np.abs(spec)
    pha = np.angle(spec)
    if mag.shape[1] <= 1:
        return signal

    out_mag = np.empty_like(mag)
    reverb = np.zeros(mag.shape[0], dtype=np.float64)
    out_mag[:, 0] = mag[:, 0]
    for t in range(1, mag.shape[1]):
        reverb = np.maximum(decay * reverb, mag[:, t - 1])
        cleaned = mag[:, t] - strength * reverb
        min_floor = floor * mag[:, t]
        out_mag[:, t] = np.maximum(cleaned, min_floor)

    out = out_mag * np.exp(1j * pha)
    return istft(out, config, expected_length=signal.size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce spectral tails / reverberant smear")
    add_common_io_args(parser, default_suffix="_deverb")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--strength", type=float, default=0.45, help="Tail suppression strength 0..1")
    parser.add_argument("--decay", type=float, default=0.92, help="Tail memory decay 0..1")
    parser.add_argument("--floor", type=float, default=0.12, help="Per-bin floor multiplier")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if not (0.0 <= args.strength <= 1.0):
        parser.error("--strength must be between 0 and 1")
    if not (0.0 < args.decay < 1.0):
        parser.error("--decay must be in (0, 1)")
    if args.floor <= 0:
        parser.error("--floor must be > 0")

    config = build_vocoder_config(args, phase_locking="off", transient_preserve=False)
    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, "pvxdeverb", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            out = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                out[:, ch] = deverb_channel(
                    audio[:, ch],
                    config,
                    strength=args.strength,
                    decay=args.decay,
                    floor=args.floor,
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
            write_output(out_path, out, sr, args)
            log_message(args, f"[ok] {path} -> {out_path}", min_level="verbose")
        except Exception as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)
    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(args, f"[done] pvxdeverb processed={len(paths)} failed={failures}", min_level="normal")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
