#!/usr/bin/env python3
"""Layered harmonic/percussive processing with independent controls."""

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
    time_pitch_shift_audio,
    validate_vocoder_args,
    write_output,
)
from pvxvoc import istft, stft

try:
    from scipy.ndimage import median_filter
except Exception:  # pragma: no cover
    median_filter = None


def hpss_masks(
    magnitude: np.ndarray,
    harmonic_kernel: int,
    percussive_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    if median_filter is None:
        harm = np.maximum(magnitude, 1e-12)
        perc = np.maximum(magnitude.mean(axis=0, keepdims=True), 1e-12)
        perc = np.repeat(perc, magnitude.shape[0], axis=0)
    else:
        harm = median_filter(magnitude, size=(1, max(3, harmonic_kernel)))
        perc = median_filter(magnitude, size=(max(3, percussive_kernel), 1))

    denom = harm + perc + 1e-12
    mh = harm / denom
    mp = perc / denom
    return mh, mp


def split_hpss(signal: np.ndarray, config, hk: int, pk: int) -> tuple[np.ndarray, np.ndarray]:
    spec = stft(signal, config)
    mag = np.abs(spec)
    mh, mp = hpss_masks(mag, hk, pk)
    harmonic = istft(spec * mh, config, expected_length=signal.size)
    percussive = istft(spec * mp, config, expected_length=signal.size)
    return harmonic, percussive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split into harmonic/percussive layers and process independently")
    add_common_io_args(parser, default_suffix="_layer")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--harmonic-stretch", type=float, default=1.0)
    parser.add_argument("--harmonic-pitch-semitones", type=float, default=0.0)
    parser.add_argument("--percussive-stretch", type=float, default=1.0)
    parser.add_argument("--percussive-pitch-semitones", type=float, default=0.0)
    parser.add_argument("--harmonic-gain", type=float, default=1.0)
    parser.add_argument("--percussive-gain", type=float, default=1.0)
    parser.add_argument("--harmonic-kernel", type=int, default=31)
    parser.add_argument("--percussive-kernel", type=int, default=31)
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.harmonic_stretch <= 0 or args.percussive_stretch <= 0:
        parser.error("stretch factors must be > 0")
    if args.harmonic_kernel <= 0 or args.percussive_kernel <= 0:
        parser.error("kernel sizes must be > 0")

    config = build_vocoder_config(args, phase_locking="identity", transient_preserve=True, transient_threshold=1.7)
    paths = resolve_inputs(args.inputs, parser)

    harm_ratio = semitone_to_ratio(args.harmonic_pitch_semitones)
    perc_ratio = semitone_to_ratio(args.percussive_pitch_semitones)

    failures = 0
    for path in paths:
        try:
            audio, sr = read_audio(path)
            harm_channels: list[np.ndarray] = []
            perc_channels: list[np.ndarray] = []
            for ch in range(audio.shape[1]):
                h, p = split_hpss(audio[:, ch], config, args.harmonic_kernel, args.percussive_kernel)
                harm_channels.append(h)
                perc_channels.append(p)

            hsig = np.stack(harm_channels, axis=1)
            psig = np.stack(perc_channels, axis=1)

            hproc = time_pitch_shift_audio(
                hsig,
                stretch=args.harmonic_stretch,
                pitch_ratio=harm_ratio,
                config=config,
                resample_mode=args.resample_mode,
            ) * args.harmonic_gain
            pproc = time_pitch_shift_audio(
                psig,
                stretch=args.percussive_stretch,
                pitch_ratio=perc_ratio,
                config=config,
                resample_mode=args.resample_mode,
            ) * args.percussive_gain

            out_len = max(hproc.shape[0], pproc.shape[0])
            out = np.zeros((out_len, audio.shape[1]), dtype=np.float64)
            out[: hproc.shape[0], :] += hproc
            out[: pproc.shape[0], :] += pproc

            out = finalize_audio(out, args)
            out_path = default_output_path(path, args)
            write_output(out_path, out, sr, args)
            if args.verbose:
                print(f"[ok] {path} -> {out_path} | out_dur={out.shape[0]/sr:.3f}s")
        except Exception as exc:
            failures += 1
            print(f"[error] {path}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

