#!/usr/bin/env python3
"""Phase-consistent spectral denoiser."""

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
from pvx.core.voc import db_to_amplitude, istft, stft


def smooth_mask(mask: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return mask
    kernel = np.ones(span, dtype=np.float64) / span
    padded = np.pad(mask, ((0, 0), (span // 2, span // 2)), mode="edge")
    out = np.empty_like(mask)
    for b in range(mask.shape[0]):
        out[b, :] = np.convolve(padded[b, :], kernel, mode="valid")[: mask.shape[1]]
    return out


def denoise_channel(
    signal: np.ndarray,
    sr: int,
    config,
    *,
    noise_ref: np.ndarray | None,
    noise_seconds: float,
    reduction_db: float,
    floor: float,
    smooth: int,
) -> np.ndarray:
    spec = stft(signal, config)
    mag = np.abs(spec)
    pha = np.angle(spec)

    if noise_ref is None:
        noise_frames = max(1, int(round((noise_seconds * sr) / config.hop_size)))
        ref = np.median(mag[:, : min(noise_frames, mag.shape[1])], axis=1)
    else:
        ref_mag = np.abs(stft(noise_ref, config))
        ref = np.median(ref_mag, axis=1)

    subtract = db_to_amplitude(reduction_db)
    clean = np.maximum(mag - subtract * ref[:, None], floor * ref[:, None])
    gain = clean / np.maximum(mag, 1e-12)
    gain = smooth_mask(gain, smooth)
    out = (mag * gain) * np.exp(1j * pha)
    return istft(out, config, expected_length=signal.size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spectral subtraction denoiser")
    add_common_io_args(parser, default_suffix="_denoise")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--noise-seconds", type=float, default=0.35, help="Noise profile duration from start")
    parser.add_argument("--noise-file", type=Path, default=None, help="Optional external noise reference")
    parser.add_argument("--reduction-db", type=float, default=12.0, help="Reduction strength in dB")
    parser.add_argument("--floor", type=float, default=0.1, help="Noise floor multiplier")
    parser.add_argument("--smooth", type=int, default=5, help="Temporal smoothing frames")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.noise_seconds <= 0:
        parser.error("--noise-seconds must be > 0")
    if args.reduction_db < 0:
        parser.error("--reduction-db must be >= 0")
    if args.floor <= 0:
        parser.error("--floor must be > 0")
    if args.smooth <= 0:
        parser.error("--smooth must be > 0")

    config = build_vocoder_config(args, phase_locking="off", transient_preserve=False)
    paths = resolve_inputs(args.inputs, parser, args)

    noise_ref = None
    if args.noise_file is not None:
        noise_ref, _ = read_audio(args.noise_file)
    status = build_status_bar(args, "pvxdenoise", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            out = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                ref_ch = None
                if noise_ref is not None:
                    ref_idx = min(ch, noise_ref.shape[1] - 1)
                    ref_ch = noise_ref[:, ref_idx]
                out[:, ch] = denoise_channel(
                    audio[:, ch],
                    sr,
                    config,
                    noise_ref=ref_ch,
                    noise_seconds=args.noise_seconds,
                    reduction_db=args.reduction_db,
                    floor=args.floor,
                    smooth=args.smooth,
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
    log_message(args, f"[done] pvxdenoise processed={len(paths)} failed={failures}", min_level="normal")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
