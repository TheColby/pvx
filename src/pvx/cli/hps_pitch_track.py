#!/usr/bin/env python3
"""Track F0 and emit a pvx control-map CSV for pitch-follow pipelines."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pvx.core.common import add_console_args, build_status_bar, log_message


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    if str(path) == "-":
        payload = sys.stdin.buffer.read()
        if not payload:
            raise ValueError("No audio bytes received on stdin")
        audio, sr = sf.read(io.BytesIO(payload), always_2d=True)
    else:
        audio, sr = sf.read(str(path), always_2d=True)
    return audio.astype(np.float64, copy=False), int(sr)


def _acf_pitch_and_confidence(frame: np.ndarray, sr: int, fmin: float, fmax: float) -> tuple[float, float]:
    work = np.asarray(frame, dtype=np.float64)
    work = work - np.mean(work)
    if work.size < 8:
        return 0.0, 0.0
    energy = float(np.dot(work, work))
    if energy <= 1e-12:
        return 0.0, 0.0

    corr = np.correlate(work, work, mode="full")[work.size - 1 :]
    if corr.size < 3:
        return 0.0, 0.0

    min_lag = max(1, int(sr / max(fmax, 1e-9)))
    max_lag = min(corr.size - 1, int(sr / max(fmin, 1e-9)))
    if max_lag <= min_lag:
        return 0.0, 0.0

    window = corr[min_lag : max_lag + 1]
    lag = int(np.argmax(window)) + min_lag
    peak = float(corr[lag])
    if lag <= 0 or peak <= 0.0:
        return 0.0, 0.0
    f0 = float(sr / lag)
    confidence = float(np.clip(peak / max(1e-12, corr[0]), 0.0, 1.0))
    return f0, confidence


def _estimate_reference_hz(f0_hz: np.ndarray, confidence: np.ndarray, mode: str, reference_hz: float | None) -> float:
    voiced = (f0_hz > 0.0) & np.isfinite(f0_hz)
    if confidence.size == voiced.size:
        voiced &= np.isfinite(confidence)

    if mode == "hz":
        if reference_hz is None or reference_hz <= 0.0:
            raise ValueError("--reference-hz must be > 0 when --ratio-reference hz")
        return float(reference_hz)
    if not np.any(voiced):
        return float(reference_hz if reference_hz and reference_hz > 0.0 else 440.0)
    values = f0_hz[voiced]
    if mode == "mean":
        return float(np.mean(values))
    if mode == "first":
        return float(values[0])
    return float(np.median(values))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    if window % 2 == 0:
        window += 1
    window = min(window, max(3, values.size | 1))
    if window <= 1:
        return values
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _track_pyin(
    mono: np.ndarray,
    sr: int,
    *,
    fmin: float,
    fmax: float,
    frame_length: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import librosa  # type: ignore
    except Exception as exc:
        raise RuntimeError("librosa is required for backend=pyin") from exc

    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
        mono,
        sr=sr,
        fmin=float(fmin),
        fmax=float(fmax),
        frame_length=int(frame_length),
        hop_length=int(hop_size),
    )
    f0 = np.nan_to_num(np.asarray(f0_hz, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(np.asarray(voiced_prob, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if voiced_flag is not None:
        voiced = np.asarray(voiced_flag, dtype=bool)
        conf = np.where(voiced, conf, 0.0)
        f0 = np.where(voiced, f0, 0.0)
    return f0, conf


def _track_acf(
    mono: np.ndarray,
    sr: int,
    *,
    fmin: float,
    fmax: float,
    frame_length: int,
    hop_size: int,
    status: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = max(1, int(np.ceil(mono.size / max(1, hop_size))))
    f0 = np.zeros(n_frames, dtype=np.float64)
    conf = np.zeros(n_frames, dtype=np.float64)
    for idx in range(n_frames):
        start = idx * hop_size
        frame = mono[start : start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size))
        hz, c = _acf_pitch_and_confidence(frame, sr, fmin, fmax)
        f0[idx] = hz
        conf[idx] = c
        if status is not None:
            status.step(idx + 1, "acf")
    return f0, conf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HPS/pyin-style pitch tracker that emits pvx control-map CSV to stdout.",
    )
    parser.add_argument("input", type=Path, help="Input audio file path or '-' for stdin audio")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("-"),
        help="Output CSV path (default: '-' for stdout)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pyin", "acf"],
        default="auto",
        help="Pitch backend (default: auto -> pyin if available, else acf)",
    )
    parser.add_argument("--fmin", type=float, default=50.0, help="Minimum F0 in Hz (default: 50)")
    parser.add_argument("--fmax", type=float, default=1200.0, help="Maximum F0 in Hz (default: 1200)")
    parser.add_argument("--frame-length", type=int, default=2048, help="Frame length in samples (default: 2048)")
    parser.add_argument("--hop-size", type=int, default=256, help="Hop size in samples (default: 256)")
    parser.add_argument(
        "--ratio-reference",
        choices=["median", "mean", "first", "hz"],
        default="median",
        help="Reference for emitted pitch_ratio values (default: median voiced f0).",
    )
    parser.add_argument(
        "--reference-hz",
        type=float,
        default=None,
        help="Reference frequency in Hz when --ratio-reference hz.",
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=0.25,
        help="Lower clamp for emitted pitch_ratio (default: 0.25).",
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=4.0,
        help="Upper clamp for emitted pitch_ratio (default: 4.0).",
    )
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=5,
        help="Smoothing window for pitch_ratio frames (default: 5).",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=0.0,
        help="Set confidence below this floor to 0.0 (default: 0.0).",
    )
    parser.add_argument(
        "--stretch",
        type=float,
        default=1.0,
        help="Emit constant stretch column value (default: 1.0).",
    )
    add_console_args(parser)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.fmin <= 0.0 or args.fmax <= 0.0 or args.fmin >= args.fmax:
        parser.error("--fmin and --fmax must satisfy 0 < fmin < fmax")
    if args.frame_length <= 0:
        parser.error("--frame-length must be > 0")
    if args.hop_size <= 0:
        parser.error("--hop-size must be > 0")
    if args.ratio_min <= 0.0:
        parser.error("--ratio-min must be > 0")
    if args.ratio_max <= args.ratio_min:
        parser.error("--ratio-max must be > --ratio-min")
    if args.smooth_frames < 1:
        parser.error("--smooth-frames must be >= 1")
    if args.confidence_floor < 0.0:
        parser.error("--confidence-floor must be >= 0")
    if args.stretch <= 0.0:
        parser.error("--stretch must be > 0")
    if args.reference_hz is not None and args.reference_hz <= 0.0:
        parser.error("--reference-hz must be > 0")


def _emit_csv(
    *,
    output: Path,
    sample_rate: int,
    hop_size: int,
    input_samples: int,
    f0_hz: np.ndarray,
    confidence: np.ndarray,
    pitch_ratio: np.ndarray,
    stretch: float,
) -> int:
    stream: io.TextIOBase
    close_stream = False
    if str(output) == "-":
        stream = sys.stdout
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = output.open("w", encoding="utf-8", newline="")
        close_stream = True

    row_count = 0
    writer = csv.writer(stream)
    writer.writerow(["start_sec", "end_sec", "stretch", "pitch_ratio", "confidence", "f0_hz"])
    frame_dur = hop_size / float(sample_rate)
    total_dur = input_samples / float(sample_rate)
    for idx in range(f0_hz.size):
        start_sec = idx * frame_dur
        end_sec = min(total_dur, (idx + 1) * frame_dur)
        if end_sec <= start_sec:
            end_sec = start_sec + frame_dur
        writer.writerow(
            [
                f"{start_sec:.9f}",
                f"{end_sec:.9f}",
                f"{stretch:.9f}",
                f"{pitch_ratio[idx]:.9f}",
                f"{confidence[idx]:.9f}",
                f"{f0_hz[idx]:.9f}",
            ]
        )
        row_count += 1

    if close_stream:
        stream.close()
    return row_count


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    stream_mode = str(args.output) == "-"
    setattr(args, "stdout", stream_mode)

    audio, sr = _read_audio(args.input)
    if audio.shape[0] == 0:
        raise ValueError("Input has no audio samples")
    mono = np.mean(audio, axis=1)

    status = build_status_bar(args, "hps-pitch-track", 1)
    status.step(0, "analyze")

    backend = str(args.backend)
    used_backend = backend
    f0_hz: np.ndarray
    confidence: np.ndarray

    if backend in {"auto", "pyin"}:
        try:
            f0_hz, confidence = _track_pyin(
                mono,
                sr,
                fmin=args.fmin,
                fmax=args.fmax,
                frame_length=args.frame_length,
                hop_size=args.hop_size,
            )
            used_backend = "pyin"
        except Exception:
            if backend == "pyin":
                raise
            used_backend = "acf"
            status = build_status_bar(args, "hps-pitch-track", max(1, int(np.ceil(mono.size / args.hop_size))))
            f0_hz, confidence = _track_acf(
                mono,
                sr,
                fmin=args.fmin,
                fmax=args.fmax,
                frame_length=args.frame_length,
                hop_size=args.hop_size,
                status=status,
            )
    else:
        used_backend = "acf"
        status = build_status_bar(args, "hps-pitch-track", max(1, int(np.ceil(mono.size / args.hop_size))))
        f0_hz, confidence = _track_acf(
            mono,
            sr,
            fmin=args.fmin,
            fmax=args.fmax,
            frame_length=args.frame_length,
            hop_size=args.hop_size,
            status=status,
        )

    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    if args.confidence_floor > 0.0:
        confidence = np.where(confidence >= args.confidence_floor, confidence, 0.0)

    reference_hz = _estimate_reference_hz(
        f0_hz,
        confidence,
        mode=args.ratio_reference,
        reference_hz=args.reference_hz,
    )
    pitch_ratio = np.ones_like(f0_hz, dtype=np.float64)
    voiced = (f0_hz > 0.0) & np.isfinite(f0_hz)
    pitch_ratio[voiced] = f0_hz[voiced] / max(1e-9, reference_hz)
    pitch_ratio = np.clip(pitch_ratio, args.ratio_min, args.ratio_max)
    pitch_ratio = _smooth(pitch_ratio, int(args.smooth_frames))
    pitch_ratio = np.clip(pitch_ratio, args.ratio_min, args.ratio_max)

    rows = _emit_csv(
        output=args.output,
        sample_rate=sr,
        hop_size=args.hop_size,
        input_samples=mono.size,
        f0_hz=f0_hz,
        confidence=confidence,
        pitch_ratio=pitch_ratio,
        stretch=float(args.stretch),
    )
    status.finish("done")

    voiced_count = int(np.sum(voiced))
    log_message(
        args,
        (
            f"[done] hps-pitch-track backend={used_backend}, frames={f0_hz.size}, voiced={voiced_count}, "
            f"ref_hz={reference_hz:.3f}, rows={rows}"
        ),
        min_level="normal",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
