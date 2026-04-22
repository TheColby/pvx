#!/usr/bin/env python3

"""Monophonic retuning with phase-vocoder segment processing."""

from __future__ import annotations

import argparse
import math

import numpy as np

from pvx.core.common import (
    add_common_io_args,
    add_vocoder_args,
    build_examples_epilog,
    build_status_bar,
    build_vocoder_config,
    default_output_path,
    ensure_runtime,
    finalize_audio,
    log_error,
    log_message,
    parse_float_list,
    print_input_output_metrics_table,
    read_audio,
    resolve_inputs,
    time_pitch_shift_audio,
    validate_vocoder_args,
    write_output,
)
from pvx.core.voc import estimate_f0_autocorrelation

NOTE_TO_CLASS = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}

CLASS_TO_NOTE = [
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
]

SCALES = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
}


def freq_to_midi(freq: float, *, a4_reference_hz: float = 440.0) -> float:
    return 69.0 + 12.0 * math.log2(freq / float(a4_reference_hz))


def midi_to_freq(midi: float, *, a4_reference_hz: float = 440.0) -> float:
    return float(a4_reference_hz) * (2.0 ** ((midi - 69.0) / 12.0))


def normalize_octave_cents(values: list[float]) -> list[float]:
    unique = sorted({round(float(v) % 1200.0, 6) for v in values})
    return [float(v) for v in unique]


def nearest_scale_freq(
    freq: float,
    root: str,
    scale_name: str,
    *,
    custom_scale_cents: list[float] | None = None,
    a4_reference_hz: float = 440.0,
    root_hz: float | None = None,
) -> float:
    midi = freq_to_midi(freq, a4_reference_hz=float(a4_reference_hz))
    cents = midi * 100.0
    if root_hz is None:
        root_class = NOTE_TO_CLASS[root.upper()]
        root_cents = float(root_class * 100)
    else:
        root_cents = freq_to_midi(float(root_hz), a4_reference_hz=float(a4_reference_hz)) * 100.0
    if custom_scale_cents is None:
        degree_cents = [float(interval * 100.0) for interval in SCALES[scale_name]]
    else:
        degree_cents = custom_scale_cents
    center_octave = int(round((cents - root_cents) / 1200.0))
    best = cents
    best_err = float("inf")
    for octave in range(center_octave - 6, center_octave + 7):
        base = root_cents + (octave * 1200.0)
        for degree in degree_cents:
            cand_cents = base + degree
            err = abs(cand_cents - cents)
            if err < best_err:
                best_err = err
                best = cand_cents
    return midi_to_freq(best / 100.0, a4_reference_hz=float(a4_reference_hz))


def overlap_add(chunks: list[np.ndarray], starts: list[int], total_len: int) -> np.ndarray:
    channels = chunks[0].shape[1]
    out = np.zeros((total_len, channels), dtype=np.float64)
    weight = np.zeros((total_len, 1), dtype=np.float64)
    for chunk, start in zip(chunks, starts):
        n = chunk.shape[0]
        w = np.hanning(n)
        if n < 3:
            w = np.ones(n, dtype=np.float64)
        s = start
        e = min(total_len, start + n)
        if e <= s:
            continue
        wn = w[: e - s, None]
        out[s:e, :] += chunk[: e - s, :] * wn
        weight[s:e, :] += wn
    nz = weight[:, 0] > 1e-9
    out[nz, :] /= weight[nz, :]
    return out


def collect_f0_values(
    mono: np.ndarray,
    sr: int,
    *,
    chunk: int,
    step: int,
    f0_min: float,
    f0_max: float,
) -> list[float]:
    values: list[float] = []
    for start in range(0, mono.shape[0], step):
        end = min(mono.shape[0], start + chunk)
        piece = mono[start:end]
        if piece.shape[0] < 8:
            continue
        try:
            f0 = float(estimate_f0_autocorrelation(piece, sr, f0_min, f0_max))
        except (ValueError, ArithmeticError, RuntimeError):
            continue
        if np.isfinite(f0) and f0 > 0.0:
            values.append(f0)
    return values


def recommend_root_hz(
    f0_values: list[float], *, a4_reference_hz: float = 440.0
) -> tuple[float, str] | None:
    if not f0_values:
        return None
    midi_values = np.array(
        [freq_to_midi(v, a4_reference_hz=float(a4_reference_hz)) for v in f0_values],
        dtype=np.float64,
    )
    pitch_classes = np.mod(np.rint(midi_values).astype(np.int64), 12)
    pitch_class_counts = np.bincount(pitch_classes, minlength=12)
    root_class = int(np.argmax(pitch_class_counts))
    median_midi = float(np.median(midi_values))
    root_octave = int(round((median_midi - float(root_class)) / 12.0))
    root_midi = float(root_class + (12 * root_octave))
    root_hz = midi_to_freq(root_midi, a4_reference_hz=float(a4_reference_hz))
    return float(root_hz), CLASS_TO_NOTE[root_class]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monophonic retune toward a musical scale",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx retune vocal.wav --root C --scale major --strength 0.8 --output vocal_major.wav",
                "pvx retune flute.wav --root D --scale-cents 0,112,204,316,498,702,884,1088 --output flute_microtonal.wav",
                "pvx retune vocal.wav --root A --scale minor --a4-reference-hz 432 --output vocal_a432.wav",
                "pvx retune voice.wav --root-hz 261.6256 --scale major --output voice_c4_anchor.wav",
                "pvx retune vocal.wav --scale minor --recommend-root --output vocal_auto_root.wav",
                "pvx retune lead.wav --root A --scale minor --stdout | pvx unison - --voices 5 --detune-cents 10 --output lead_retune_unison.wav",
            ],
            notes=[
                "Use --scale-cents for custom microtonal scales within one octave.",
                "Use --a4-reference-hz for alternate concert pitch (for example 432 Hz).",
                "Use --root-hz to anchor tuning to a specific root fundamental frequency.",
                "Use --recommend-root to estimate a root fundamental from the input file.",
                "Increase --chunk-ms for steadier pitch decisions, reduce for faster note changes.",
            ],
        ),
    )
    add_common_io_args(parser, default_suffix="_retune")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--root", default="C", help="Scale root note (C,C#,D,...,B)")
    parser.add_argument(
        "--root-hz",
        type=float,
        default=None,
        help="Optional root fundamental in Hz (overrides --root when provided).",
    )
    parser.add_argument(
        "--recommend-root",
        action="store_true",
        help="Estimate and use a per-file root fundamental from tracked F0.",
    )
    parser.add_argument(
        "--scale",
        choices=sorted(SCALES.keys()),
        default="chromatic",
        help="Named scale for 12-TET quantization",
    )
    parser.add_argument(
        "--scale-cents",
        default=None,
        help=(
            "Optional comma-separated microtonal scale degrees in cents within one octave, "
            "relative to --root (example: 0,90,204,294,408,498,612,702,816,906,1020,1110)"
        ),
    )
    parser.add_argument("--strength", type=float, default=0.85, help="Correction strength 0..1")
    parser.add_argument(
        "--chunk-ms", type=float, default=80.0, help="Analysis/process chunk duration in ms"
    )
    parser.add_argument("--overlap-ms", type=float, default=20.0, help="Chunk overlap in ms")
    parser.add_argument(
        "--a4-reference-hz",
        type=float,
        default=440.0,
        help="Concert-pitch reference for A4 in Hz (default: 440.0)",
    )
    parser.add_argument("--f0-min", type=float, default=60.0)
    parser.add_argument("--f0-max", type=float, default=1200.0)
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.root.upper() not in NOTE_TO_CLASS:
        parser.error("--root must be a valid note name")
    custom_scale_cents: list[float] | None = None
    if args.scale_cents:
        cents = parse_float_list(args.scale_cents)
        if not cents:
            parser.error("--scale-cents requires at least one numeric value")
        custom_scale_cents = normalize_octave_cents(cents)
        if not custom_scale_cents:
            parser.error("--scale-cents resolves to an empty set")
    if not (0.0 <= args.strength <= 1.0):
        parser.error("--strength must be between 0 and 1")
    if args.chunk_ms <= 5.0:
        parser.error("--chunk-ms must be > 5")
    if args.overlap_ms < 0:
        parser.error("--overlap-ms must be >= 0")
    if not np.isfinite(float(args.a4_reference_hz)) or float(args.a4_reference_hz) <= 0.0:
        parser.error("--a4-reference-hz must be a finite value > 0")
    if args.root_hz is not None and (
        not np.isfinite(float(args.root_hz)) or float(args.root_hz) <= 0.0
    ):
        parser.error("--root-hz must be a finite value > 0")
    if args.recommend_root and args.root_hz is not None:
        parser.error("--recommend-root and --root-hz are mutually exclusive")
    if args.f0_min <= 0 or args.f0_max <= 0 or args.f0_min >= args.f0_max:
        parser.error("0 < --f0-min < --f0-max required")

    config = build_vocoder_config(
        args, phase_locking="identity", transient_preserve=True, transient_threshold=1.8
    )
    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, "pvxretune", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            mono = np.mean(audio, axis=1)

            chunk = max(32, int(round(sr * args.chunk_ms / 1000.0)))
            overlap = int(round(sr * args.overlap_ms / 1000.0))
            step = max(8, chunk - overlap)
            root_hz = float(args.root_hz) if args.root_hz is not None else None
            if args.recommend_root:
                recommended = recommend_root_hz(
                    collect_f0_values(
                        mono,
                        sr,
                        chunk=chunk,
                        step=step,
                        f0_min=float(args.f0_min),
                        f0_max=float(args.f0_max),
                    ),
                    a4_reference_hz=float(args.a4_reference_hz),
                )
                if recommended is None:
                    log_message(
                        args,
                        f"[warn] {path.name}: unable to estimate root fundamental; falling back to --root {args.root.upper()}",
                        min_level="normal",
                    )
                else:
                    root_hz, root_note = recommended
                    log_message(
                        args,
                        f"[info] {path.name}: recommended root={root_note} ({root_hz:.3f} Hz)",
                        min_level="normal",
                    )

            chunks: list[np.ndarray] = []
            starts: list[int] = []
            ratios: list[float] = []
            for start in range(0, audio.shape[0], step):
                end = min(audio.shape[0], start + chunk)
                piece = audio[start:end, :]
                mono_piece = mono[start:end]
                if piece.shape[0] < 8:
                    continue
                ratio = 1.0
                try:
                    f0 = estimate_f0_autocorrelation(mono_piece, sr, args.f0_min, args.f0_max)
                    target = nearest_scale_freq(
                        f0,
                        args.root,
                        args.scale,
                        custom_scale_cents=custom_scale_cents,
                        a4_reference_hz=float(args.a4_reference_hz),
                        root_hz=root_hz,
                    )
                    ratio = target / f0
                except (ValueError, ArithmeticError, RuntimeError):
                    ratio = 1.0
                ratio = 1.0 + (ratio - 1.0) * args.strength
                shifted = time_pitch_shift_audio(
                    piece,
                    stretch=1.0,
                    pitch_ratio=ratio,
                    config=config,
                    resample_mode=args.resample_mode,
                )
                chunks.append(shifted[: piece.shape[0], :])
                starts.append(start)
                ratios.append(ratio)

            if not chunks:
                out = audio
            else:
                out = overlap_add(chunks, starts, audio.shape[0])

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
            write_output(out_path, out, sr, args, input_path=path)
            if ratios:
                log_message(
                    args,
                    f"[ok] {path} -> {out_path} | median_ratio={float(np.median(ratios)):.4f}",
                    min_level="verbose",
                )
            else:
                log_message(args, f"[ok] {path} -> {out_path}", min_level="verbose")
        except (OSError, ValueError, RuntimeError) as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)
    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(
        args, f"[done] pvxretune processed={len(paths)} failed={failures}", min_level="normal"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
