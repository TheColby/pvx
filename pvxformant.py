#!/usr/bin/env python3
"""Formant processing tool with optional pitch shifting."""

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
from pvxvoc import cepstral_envelope, db_to_amplitude, istft, stft


def shift_envelope(env: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("Formant ratio must be > 0")
    n = env.size
    x = np.linspace(0.0, 1.0, num=n, endpoint=True)
    src = np.clip(x / ratio, 0.0, 1.0)
    return np.interp(src, x, env).astype(np.float64)


def formant_process_channel(
    signal: np.ndarray,
    config,
    *,
    lifter: int,
    ratio: float,
    max_gain_db: float,
) -> np.ndarray:
    spec = stft(signal, config)
    mag = np.abs(spec)
    pha = np.angle(spec)

    limit = db_to_amplitude(max_gain_db)
    out = np.empty_like(spec)
    for idx in range(spec.shape[1]):
        env = cepstral_envelope(mag[:, idx], lifter)
        env_shift = shift_envelope(env, ratio)
        gain = env_shift / np.maximum(env, 1e-12)
        gain = np.clip(gain, 1.0 / limit, limit)
        out_mag = mag[:, idx] * gain
        out[:, idx] = out_mag * np.exp(1j * pha[:, idx])
    return istft(out, config, expected_length=signal.size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Formant shift/preserve processor")
    add_common_io_args(parser, default_suffix="_formant")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--pitch-shift-semitones", type=float, default=0.0, help="Optional pitch shift before formant stage")
    parser.add_argument("--formant-shift-ratio", type=float, default=1.0, help="Formant ratio (>1 up, <1 down)")
    parser.add_argument("--mode", choices=["shift", "preserve"], default="shift")
    parser.add_argument("--formant-lifter", type=int, default=32)
    parser.add_argument("--formant-max-gain-db", type=float, default=12.0)
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.formant_lifter < 0:
        parser.error("--formant-lifter must be >= 0")
    if args.formant_shift_ratio <= 0:
        parser.error("--formant-shift-ratio must be > 0")
    if args.formant_max_gain_db <= 0:
        parser.error("--formant-max-gain-db must be > 0")

    config = build_vocoder_config(args, phase_locking="identity", transient_preserve=False)
    pitch_ratio = semitone_to_ratio(args.pitch_shift_semitones)
    paths = resolve_inputs(args.inputs, parser)

    failures = 0
    for path in paths:
        try:
            audio, sr = read_audio(path)
            out = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                shifted = time_pitch_shift_channel(
                    audio[:, ch],
                    stretch=1.0,
                    pitch_ratio=pitch_ratio,
                    config=config,
                    resample_mode=args.resample_mode,
                )
                ratio = args.formant_shift_ratio
                if args.mode == "preserve" and abs(pitch_ratio - 1.0) > 1e-12:
                    ratio = ratio / pitch_ratio
                out_ch = formant_process_channel(
                    shifted,
                    config,
                    lifter=args.formant_lifter,
                    ratio=ratio,
                    max_gain_db=args.formant_max_gain_db,
                )
                if out_ch.size != out.shape[0]:
                    pad = np.zeros(max(out.shape[0], out_ch.size), dtype=np.float64)
                    pad[: out_ch.size] = out_ch
                    if pad.size != out.shape[0]:
                        tmp = np.zeros((pad.size, audio.shape[1]), dtype=np.float64)
                        tmp[: out.shape[0], :] = out
                        out = tmp
                    out[:, ch] = pad[: out.shape[0]]
                else:
                    out[:, ch] = out_ch

            out = finalize_audio(out, args)
            out_path = default_output_path(path, args)
            write_output(out_path, out, sr, args)
            if args.verbose:
                print(f"[ok] {path} -> {out_path} | mode={args.mode}")
        except Exception as exc:
            failures += 1
            print(f"[error] {path}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

