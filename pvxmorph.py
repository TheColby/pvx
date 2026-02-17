#!/usr/bin/env python3
"""Spectral morphing between two input files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvxcommon import (
    add_console_args,
    add_vocoder_args,
    build_status_bar,
    build_vocoder_config,
    ensure_runtime,
    log_error,
    log_message,
    read_audio,
    validate_vocoder_args,
)
from pvxvoc import force_length, istft, normalize_audio, resample_1d, stft
import soundfile as sf


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
    parser.add_argument("input_a", type=Path, help="Input A")
    parser.add_argument("input_b", type=Path, help="Input B")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output file path")
    parser.add_argument("--alpha", type=float, default=0.5, help="Morph amount 0..1 (0=A, 1=B)")
    parser.add_argument("--normalize", choices=["none", "peak", "rms"], default="none")
    parser.add_argument("--peak-dbfs", type=float, default=-1.0)
    parser.add_argument("--rms-dbfs", type=float, default=-18.0)
    parser.add_argument("--clip", action="store_true")
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
        out = normalize_audio(out, args.normalize, args.peak_dbfs, args.rms_dbfs)
        if args.clip:
            out = np.clip(out, -1.0, 1.0)

        out_path = args.output if args.output is not None else args.input_a.with_name(f"{args.input_a.stem}_morph.wav")
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_path} (use --overwrite)")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), out, sr_a, subtype=args.subtype)
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
