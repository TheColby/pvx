#!/usr/bin/env python3
"""Spectral freeze tool built on pvx phase-vocoder primitives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvxcommon import (
    add_common_io_args,
    add_vocoder_args,
    build_vocoder_config,
    default_output_path,
    finalize_audio,
    read_audio,
    resolve_inputs,
    validate_vocoder_args,
    write_output,
    ensure_runtime,
)
from pvxvoc import istft, stft


def freeze_channel(
    signal: np.ndarray,
    sr: int,
    freeze_time_s: float,
    duration_s: float,
    config,
    random_phase: bool,
) -> np.ndarray:
    spectrum = stft(signal, config)
    bins, frames = spectrum.shape
    if frames <= 0:
        return np.zeros(max(1, int(round(duration_s * sr))), dtype=np.float64)

    frame_at_time = int(round((freeze_time_s * sr) / config.hop_size))
    frame_idx = int(np.clip(frame_at_time, 0, frames - 1))
    out_frames = max(1, int(round((duration_s * sr) / config.hop_size)))

    mag = np.abs(spectrum[:, frame_idx])
    phase = np.angle(spectrum[:, frame_idx]).copy()
    omega = 2.0 * np.pi * config.hop_size * np.arange(bins, dtype=np.float64) / config.n_fft
    out = np.zeros((bins, out_frames), dtype=np.complex128)

    rng = np.random.default_rng(12345)
    for idx in range(out_frames):
        if random_phase:
            phase = phase + omega + rng.uniform(-0.03, 0.03, size=bins)
        else:
            phase = phase + omega
        out[:, idx] = mag * np.exp(1j * phase)

    expected_len = max(1, int(round(duration_s * sr)))
    return istft(out, config, expected_length=expected_len)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze a spectral slice into a sustained output")
    add_common_io_args(parser, default_suffix="_freeze")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=256)
    parser.add_argument("--freeze-time", type=float, default=0.2, help="Freeze anchor time in seconds")
    parser.add_argument("--duration", type=float, default=3.0, help="Output freeze duration in seconds")
    parser.add_argument("--random-phase", action="store_true", help="Add subtle phase randomization per frame")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)

    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.freeze_time < 0:
        parser.error("--freeze-time must be >= 0")

    config = build_vocoder_config(args, phase_locking="off", transient_preserve=False)
    paths = resolve_inputs(args.inputs, parser)

    failures = 0
    for path in paths:
        try:
            audio, sr = read_audio(path)
            out_channels: list[np.ndarray] = []
            for ch in range(audio.shape[1]):
                out_channels.append(
                    freeze_channel(
                        audio[:, ch],
                        sr,
                        args.freeze_time,
                        args.duration,
                        config,
                        random_phase=args.random_phase,
                    )
                )
            out_len = max(ch.size for ch in out_channels)
            out = np.zeros((out_len, len(out_channels)), dtype=np.float64)
            for idx, ch in enumerate(out_channels):
                out[: ch.size, idx] = ch

            out = finalize_audio(out, args)
            out_path = default_output_path(path, args)
            write_output(out_path, out, sr, args)
            if args.verbose:
                print(f"[ok] {path} -> {out_path} | ch={out.shape[1]}, dur={out.shape[0]/sr:.3f}s")
        except Exception as exc:
            failures += 1
            print(f"[error] {path}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

