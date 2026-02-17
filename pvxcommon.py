#!/usr/bin/env python3
"""Shared helpers for pvx DSP command-line tools."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from pvxvoc import (
    VocoderConfig,
    WINDOW_CHOICES,
    add_runtime_args,
    configure_runtime_from_args,
    compute_output_path,
    ensure_runtime_dependencies,
    expand_inputs,
    force_length,
    normalize_audio,
    phase_vocoder_time_stretch,
    resample_1d,
)


@dataclass(frozen=True)
class SegmentSpec:
    start_s: float
    end_s: float
    stretch: float = 1.0
    pitch_ratio: float = 1.0


def add_common_io_args(parser: argparse.ArgumentParser, default_suffix: str) -> None:
    parser.add_argument("inputs", nargs="+", help="Input files or glob patterns")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--suffix", default=default_suffix, help=f"Output filename suffix (default: {default_suffix})")
    parser.add_argument("--output-format", default=None, help="Output extension/format")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print, but do not write files")
    parser.add_argument("--verbose", action="store_true", help="Print per-file diagnostics")
    parser.add_argument("--normalize", choices=["none", "peak", "rms"], default="none", help="Output normalization")
    parser.add_argument("--peak-dbfs", type=float, default=-1.0, help="Target peak dBFS for peak normalization")
    parser.add_argument("--rms-dbfs", type=float, default=-18.0, help="Target RMS dBFS for RMS normalization")
    parser.add_argument("--clip", action="store_true", help="Hard clip output to [-1, 1]")
    parser.add_argument("--subtype", default=None, help="libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)")


def add_vocoder_args(
    parser: argparse.ArgumentParser,
    *,
    default_n_fft: int = 2048,
    default_win_length: int = 2048,
    default_hop_size: int = 512,
) -> None:
    parser.add_argument("--n-fft", type=int, default=default_n_fft, help=f"FFT size (default: {default_n_fft})")
    parser.add_argument(
        "--win-length",
        type=int,
        default=default_win_length,
        help=f"Window length in samples (default: {default_win_length})",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=default_hop_size,
        help=f"Hop size in samples (default: {default_hop_size})",
    )
    parser.add_argument("--window", choices=list(WINDOW_CHOICES), default="hann", help="Window type")
    parser.add_argument(
        "--kaiser-beta",
        type=float,
        default=14.0,
        help="Kaiser window beta parameter used when --window kaiser (default: 14.0)",
    )
    parser.add_argument("--no-center", action="store_true", help="Disable centered framing")
    add_runtime_args(parser)


def build_vocoder_config(
    args: argparse.Namespace,
    *,
    phase_locking: str = "identity",
    transient_preserve: bool = False,
    transient_threshold: float = 2.0,
) -> VocoderConfig:
    return VocoderConfig(
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_size=args.hop_size,
        window=args.window,
        center=not args.no_center,
        phase_locking=phase_locking,
        transient_preserve=transient_preserve,
        transient_threshold=transient_threshold,
        kaiser_beta=args.kaiser_beta,
    )


def validate_vocoder_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.n_fft <= 0:
        parser.error("--n-fft must be > 0")
    if args.win_length <= 0:
        parser.error("--win-length must be > 0")
    if args.hop_size <= 0:
        parser.error("--hop-size must be > 0")
    if args.win_length > args.n_fft:
        parser.error("--win-length must be <= --n-fft")
    if args.hop_size > args.win_length:
        parser.error("--hop-size should be <= --win-length")
    if args.kaiser_beta < 0:
        parser.error("--kaiser-beta must be >= 0")
    if args.cuda_device < 0:
        parser.error("--cuda-device must be >= 0")


def resolve_inputs(patterns: Iterable[str], parser: argparse.ArgumentParser) -> list[Path]:
    paths = expand_inputs(patterns)
    if not paths:
        parser.error("No readable files found from provided inputs")
    return paths


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    return audio.astype(np.float64, copy=False), int(sr)


def finalize_audio(audio: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = normalize_audio(audio, args.normalize, args.peak_dbfs, args.rms_dbfs)
    if args.clip:
        out = np.clip(out, -1.0, 1.0)
    return out


def write_output(path: Path, audio: np.ndarray, sr: int, args: argparse.Namespace) -> None:
    if path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists: {path} (use --overwrite to replace)")
    if args.dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype=args.subtype)


def default_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    return compute_output_path(input_path, output_dir, args.suffix, args.output_format)


def parse_float_list(value: str, *, allow_empty: bool = False) -> list[float]:
    items = [chunk.strip() for chunk in value.split(",")]
    if not items:
        return []
    out: list[float] = []
    for item in items:
        if not item:
            if allow_empty:
                continue
            raise ValueError("Empty numeric element in list")
        out.append(float(item))
    return out


def semitone_to_ratio(semitones: float) -> float:
    return float(2.0 ** (semitones / 12.0))


def time_pitch_shift_channel(
    signal: np.ndarray,
    stretch: float,
    pitch_ratio: float,
    config: VocoderConfig,
    *,
    resample_mode: str = "auto",
) -> np.ndarray:
    if stretch <= 0.0:
        raise ValueError("stretch must be > 0")
    if pitch_ratio <= 0.0:
        raise ValueError("pitch_ratio must be > 0")

    internal_stretch = stretch * pitch_ratio
    shifted = phase_vocoder_time_stretch(signal, internal_stretch, config)
    if abs(pitch_ratio - 1.0) > 1e-10:
        target_samples = max(1, int(round(shifted.size / pitch_ratio)))
        shifted = resample_1d(shifted, target_samples, resample_mode)

    target_length = max(1, int(round(signal.size * stretch)))
    return force_length(shifted, target_length)


def time_pitch_shift_audio(
    audio: np.ndarray,
    stretch: float,
    pitch_ratio: float,
    config: VocoderConfig,
    *,
    resample_mode: str = "auto",
) -> np.ndarray:
    channels: list[np.ndarray] = []
    for idx in range(audio.shape[1]):
        channels.append(time_pitch_shift_channel(audio[:, idx], stretch, pitch_ratio, config, resample_mode=resample_mode))
    out_len = max(ch.size for ch in channels)
    out = np.zeros((out_len, len(channels)), dtype=np.float64)
    for ch, values in enumerate(channels):
        out[: values.size, ch] = values
    return out


def read_segment_csv(path: Path, *, has_pitch: bool) -> list[SegmentSpec]:
    segments: list[SegmentSpec] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"start_sec", "end_sec", "stretch"}
        if has_pitch:
            required.add("pitch_semitones")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row in reader:
            start_s = float(row["start_sec"])
            end_s = float(row["end_sec"])
            stretch = float(row["stretch"])
            if end_s <= start_s:
                continue
            if stretch <= 0:
                raise ValueError("stretch must be positive in all CSV rows")
            pitch_ratio = 1.0
            if has_pitch:
                pitch_ratio = semitone_to_ratio(float(row["pitch_semitones"]))
            segments.append(SegmentSpec(start_s=start_s, end_s=end_s, stretch=stretch, pitch_ratio=pitch_ratio))

    segments.sort(key=lambda seg: seg.start_s)
    return segments


def concat_with_crossfade(chunks: list[np.ndarray], sr: int, crossfade_ms: float = 8.0) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 1), dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]

    fade = max(0, int(round(sr * crossfade_ms / 1000.0)))
    out = chunks[0]
    for nxt in chunks[1:]:
        if fade <= 0 or out.shape[0] < fade or nxt.shape[0] < fade:
            out = np.vstack([out, nxt])
            continue
        w = np.linspace(0.0, 1.0, num=fade, endpoint=True)[:, None]
        tail = out[-fade:, :] * (1.0 - w) + nxt[:fade, :] * w
        out = np.vstack([out[:-fade, :], tail, nxt[fade:, :]])
    return out


def ensure_runtime(
    args: argparse.Namespace | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    ensure_runtime_dependencies()
    if args is not None:
        configure_runtime_from_args(args, parser)
