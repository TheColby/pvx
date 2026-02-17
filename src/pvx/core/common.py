#!/usr/bin/env python3
"""Shared helpers for pvx DSP command-line tools."""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from pvx.core.voc import (
    TRANSFORM_CHOICES,
    VocoderConfig,
    WINDOW_CHOICES,
    add_mastering_args,
    add_runtime_args,
    apply_mastering_chain,
    configure_runtime_from_args,
    compute_output_path,
    ensure_runtime_dependencies,
    expand_inputs,
    force_length,
    parse_pitch_ratio_value,
    phase_vocoder_time_stretch,
    resample_1d,
    validate_transform_available,
    validate_mastering_args,
)


@dataclass(frozen=True)
class SegmentSpec:
    start_s: float
    end_s: float
    stretch: float = 1.0
    pitch_ratio: float = 1.0


VERBOSITY_LEVELS = ("silent", "quiet", "normal", "verbose", "debug")
_VERBOSITY_TO_LEVEL = {name: idx for idx, name in enumerate(VERBOSITY_LEVELS)}


def add_console_args(
    parser: argparse.ArgumentParser,
    *,
    include_no_progress_alias: bool = False,
) -> None:
    parser.add_argument(
        "--verbosity",
        choices=list(VERBOSITY_LEVELS),
        default="normal",
        help="Console verbosity level",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeat for extra detail)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output and hide status bars")
    parser.add_argument("--silent", action="store_true", help="Suppress all console output")
    if include_no_progress_alias:
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help=argparse.SUPPRESS,
        )


def console_level(args: argparse.Namespace) -> int:
    cached = getattr(args, "_console_level_cache", None)
    if cached is not None:
        return int(cached)

    base_level = _VERBOSITY_TO_LEVEL.get(str(getattr(args, "verbosity", "normal")), _VERBOSITY_TO_LEVEL["normal"])
    verbose_count = int(getattr(args, "verbose", 0) or 0)
    level = min(_VERBOSITY_TO_LEVEL["debug"], base_level + verbose_count)
    if bool(getattr(args, "no_progress", False)):
        level = min(level, _VERBOSITY_TO_LEVEL["quiet"])
    if bool(getattr(args, "quiet", False)):
        level = min(level, _VERBOSITY_TO_LEVEL["quiet"])
    if bool(getattr(args, "silent", False)):
        level = _VERBOSITY_TO_LEVEL["silent"]

    setattr(args, "_console_level_cache", level)
    return level


def is_quiet(args: argparse.Namespace) -> bool:
    return console_level(args) <= _VERBOSITY_TO_LEVEL["quiet"]


def is_silent(args: argparse.Namespace) -> bool:
    return console_level(args) == _VERBOSITY_TO_LEVEL["silent"]


def log_message(
    args: argparse.Namespace,
    message: str,
    *,
    min_level: str = "normal",
    error: bool = False,
) -> None:
    required = _VERBOSITY_TO_LEVEL[min_level]
    if console_level(args) < required:
        return
    stream_to_stdout = bool(getattr(args, "stdout", False))
    stream = sys.stderr if error or stream_to_stdout else sys.stdout
    print(message, file=stream)


def log_error(args: argparse.Namespace, message: str) -> None:
    if is_silent(args):
        return
    print(message, file=sys.stderr)


class StatusBar:
    def __init__(self, label: str, total: int, *, enabled: bool, width: int = 32) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.enabled = enabled
        self.width = max(10, int(width))
        self._last_fraction = -1.0
        self._last_ts = 0.0
        self._finished = False
        if self.enabled:
            self.set(0.0, "start")

    def set(self, fraction: float, detail: str = "") -> None:
        if not self.enabled or self._finished:
            return

        now = time.time()
        frac = min(1.0, max(0.0, float(fraction)))
        should_render = (
            frac >= 1.0
            or self._last_fraction < 0.0
            or (frac - self._last_fraction) >= 0.01
            or (now - self._last_ts) >= 0.2
        )
        if not should_render:
            return

        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {detail}" if detail else ""
        sys.stderr.write(f"\r[{bar}] {frac * 100:6.2f}% {self.label}{suffix}")
        sys.stderr.flush()
        self._last_fraction = frac
        self._last_ts = now
        if frac >= 1.0:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._finished = True

    def step(self, current: int, detail: str = "") -> None:
        self.set(current / self.total, detail)

    def finish(self, detail: str = "done") -> None:
        self.set(1.0, detail)


def build_status_bar(args: argparse.Namespace, label: str, total: int) -> StatusBar:
    return StatusBar(label=label, total=total, enabled=not is_quiet(args))


def add_common_io_args(parser: argparse.ArgumentParser, default_suffix: str) -> None:
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files/globs or '-' for stdin",
    )
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--suffix", default=default_suffix, help=f"Output filename suffix (default: {default_suffix})")
    parser.add_argument("--output-format", default=None, help="Output extension/format")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write processed audio to stdout stream (for piping); requires exactly one input",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print, but do not write files")
    add_console_args(parser)
    add_mastering_args(parser)
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
    parser.add_argument(
        "--transform",
        choices=list(TRANSFORM_CHOICES),
        default="fft",
        help=(
            "Per-frame transform backend for STFT/ISTFT paths "
            "(default: fft; options: fft, dft, czt, dct, dst, hartley)"
        ),
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
        transform=args.transform,
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
    validate_transform_available(args.transform, parser)
    validate_mastering_args(args, parser)


def resolve_inputs(
    patterns: Iterable[str],
    parser: argparse.ArgumentParser,
    args: argparse.Namespace | None = None,
) -> list[Path]:
    paths = expand_inputs(patterns)
    if not paths:
        parser.error("No readable files found from provided inputs")
    stdin_count = sum(1 for p in paths if str(p) == "-")
    if stdin_count > 1:
        parser.error("Input '-' (stdin) may only be specified once")
    if stdin_count and len(paths) != 1:
        parser.error("Input '-' (stdin) cannot be combined with additional input files")
    if args is not None and bool(getattr(args, "stdout", False)) and len(paths) != 1:
        parser.error("--stdout requires exactly one resolved input")
    return paths


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    if str(path) == "-":
        payload = sys.stdin.buffer.read()
        if not payload:
            raise ValueError("No audio bytes received on stdin")
        audio, sr = sf.read(io.BytesIO(payload), always_2d=True)
    else:
        audio, sr = sf.read(str(path), always_2d=True)
    return audio.astype(np.float64, copy=False), int(sr)


def finalize_audio(audio: np.ndarray, sample_rate: int, args: argparse.Namespace) -> np.ndarray:
    return apply_mastering_chain(audio, int(sample_rate), args)


def write_output(path: Path, audio: np.ndarray, sr: int, args: argparse.Namespace) -> None:
    to_stdout = bool(getattr(args, "stdout", False)) or str(path) == "-"
    dry_run = bool(getattr(args, "dry_run", False))
    overwrite = bool(getattr(args, "overwrite", False))
    subtype = getattr(args, "subtype", None)
    output_format = getattr(args, "output_format", None)

    if to_stdout:
        if dry_run:
            return
        stream_fmt = _stream_format_name(output_format)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format=stream_fmt, subtype=subtype)
        sys.stdout.buffer.write(buffer.getvalue())
        sys.stdout.buffer.flush()
        return

    if path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output exists: {path} (use --overwrite to replace)")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype=subtype)


def default_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    output_dir = args.output_dir.resolve() if args.output_dir is not None else None
    source = Path("stdin.wav") if str(input_path) == "-" else input_path
    return compute_output_path(source, output_dir, args.suffix, args.output_format)


def _stream_format_name(output_format: str | None) -> str:
    ext = (output_format or "wav").lower().lstrip(".")
    mapping = {
        "wav": "WAV",
        "flac": "FLAC",
        "aif": "AIFF",
        "aiff": "AIFF",
        "ogg": "OGG",
        "oga": "OGG",
        "caf": "CAF",
    }
    if ext in mapping:
        return mapping[ext]
    raise ValueError(
        f"Unsupported stream output format '{output_format}'. "
        "Use --output-format with one of: wav, flac, aiff, ogg, caf."
    )


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


def cents_to_ratio(cents: float) -> float:
    return float(2.0 ** (cents / 1200.0))


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
        pitch_columns = ("pitch_ratio", "pitch_cents", "pitch_semitones")
        if has_pitch:
            headers = set(reader.fieldnames or [])
            if not any(name in headers for name in pitch_columns):
                raise ValueError(f"Missing CSV pitch column. Provide one of: {list(pitch_columns)}")
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        for row_idx, row in enumerate(reader, start=2):
            start_s = float(row["start_sec"])
            end_s = float(row["end_sec"])
            stretch = float(row["stretch"])
            if end_s <= start_s:
                continue
            if stretch <= 0:
                raise ValueError("stretch must be positive in all CSV rows")
            pitch_ratio = 1.0
            if has_pitch:
                ratio_text = str(row.get("pitch_ratio", "")).strip()
                cents_text = str(row.get("pitch_cents", "")).strip()
                semitones_text = str(row.get("pitch_semitones", "")).strip()
                populated = int(bool(ratio_text)) + int(bool(cents_text)) + int(bool(semitones_text))
                if populated == 0:
                    raise ValueError(
                        f"CSV row {row_idx}: missing pitch value. Provide pitch_ratio, pitch_cents, or pitch_semitones."
                    )
                if populated > 1:
                    raise ValueError(
                        f"CSV row {row_idx}: multiple pitch fields set. Use only one of pitch_ratio, pitch_cents, pitch_semitones."
                    )
                if ratio_text:
                    pitch_ratio = parse_pitch_ratio_value(
                        ratio_text,
                        context=f"CSV row {row_idx} pitch_ratio",
                    )
                elif cents_text:
                    pitch_ratio = cents_to_ratio(float(cents_text))
                else:
                    pitch_ratio = semitone_to_ratio(float(semitones_text))
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
