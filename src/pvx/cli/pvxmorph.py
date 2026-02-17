#!/usr/bin/env python3
"""Spectral morphing between two input files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvx.core.common import (
    add_console_args,
    add_vocoder_args,
    build_status_bar,
    build_vocoder_config,
    ensure_runtime,
    log_error,
    log_message,
    read_audio,
    validate_vocoder_args,
    write_output,
)
from pvx.core.voc import add_mastering_args, apply_mastering_chain, force_length, istft, resample_1d, stft


def match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] > channels:
        return audio[:, :channels]
    reps = channels - audio.shape[1]
    extra = np.repeat(audio[:, -1:], reps, axis=1)
    return np.hstack([audio, extra])


def morph_pair(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    config,
    alpha: float,
) -> np.ndarray:
    max_len = max(a.shape[0], b.shape[0])
    channels = max(a.shape[1], b.shape[1])
    a2 = np.zeros((max_len, channels), dtype=np.float64)
    b2 = np.zeros((max_len, channels), dtype=np.float64)
    a_adj = match_channels(a, channels)
    b_adj = match_channels(b, channels)
    a2[: a_adj.shape[0], :] = a_adj
    b2[: b_adj.shape[0], :] = b_adj

    out = np.zeros_like(a2)
    for ch in range(channels):
        sa = stft(a2[:, ch], config)
        sb = stft(b2[:, ch], config)
        n_bins = max(sa.shape[0], sb.shape[0])
        n_frames = max(sa.shape[1], sb.shape[1])
        sa2 = np.zeros((n_bins, n_frames), dtype=np.complex128)
        sb2 = np.zeros((n_bins, n_frames), dtype=np.complex128)
        sa2[: sa.shape[0], : sa.shape[1]] = sa
        sb2[: sb.shape[0], : sb.shape[1]] = sb

        mag = (1.0 - alpha) * np.abs(sa2) + alpha * np.abs(sb2)
        pha = np.angle((1.0 - alpha) * np.exp(1j * np.angle(sa2)) + alpha * np.exp(1j * np.angle(sb2)))
        sm = mag * np.exp(1j * pha)
        out[:, ch] = istft(sm, config, expected_length=max_len)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Morph two audio files in the STFT domain")
    parser.add_argument("input_a", type=Path, help="Input A path or '-' for stdin")
    parser.add_argument("input_b", type=Path, help="Input B path or '-' for stdin")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file path")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write processed audio to stdout stream (for piping); equivalent to -o -",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="Output extension/format; for --stdout defaults to wav",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Morph amount 0..1 (0=A, 1=B)")
    add_mastering_args(parser)
    parser.add_argument("--subtype", default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_console_args(parser)
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if not (0.0 <= args.alpha <= 1.0):
        parser.error("--alpha must be between 0 and 1")
    if str(args.input_a) == "-" and str(args.input_b) == "-":
        parser.error("At most one morph input may be '-' (stdin)")
    if args.output is not None and str(args.output) == "-":
        args.stdout = True
    if args.stdout and args.output is not None and str(args.output) != "-":
        parser.error("Use either --stdout (or -o -) or an explicit output file path, not both")

    config = build_vocoder_config(args, phase_locking="off", transient_preserve=False)
    status = build_status_bar(args, "pvxmorph", 1)
    try:
        a, sr_a = read_audio(args.input_a)
        b, sr_b = read_audio(args.input_b)
        if sr_b != sr_a:
            target = max(1, int(round(b.shape[0] * sr_a / sr_b)))
            rb = np.zeros((target, b.shape[1]), dtype=np.float64)
            for ch in range(b.shape[1]):
                rb[:, ch] = resample_1d(b[:, ch], target, "auto")
            b = rb

        out = morph_pair(a, b, sr_a, config, args.alpha)
        out = apply_mastering_chain(out, sr_a, args)

        if args.stdout:
            out_path = Path("-")
        elif args.output is not None:
            out_path = args.output
        else:
            base = Path("stdin.wav") if str(args.input_a) == "-" else args.input_a
            out_path = base.with_name(f"{base.stem}_morph.wav")
        write_output(out_path, out, sr_a, args)
        log_message(args, f"[ok] {args.input_a} + {args.input_b} -> {out_path}", min_level="verbose")
        status.step(1, out_path.name)
        status.finish("done")
        log_message(args, "[done] pvxmorph processed=1 failed=0", min_level="normal")
        return 0
    except Exception as exc:
        log_error(args, f"[error] {args.input_a} + {args.input_b}: {exc}")
        status.step(1, "error")
        status.finish("errors=1")
        log_message(args, "[done] pvxmorph processed=1 failed=1", min_level="normal")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
