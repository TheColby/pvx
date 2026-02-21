#!/usr/bin/env python3
"""Conform timing and pitch to a user-provided segment map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvx.core.common import (
    SegmentSpec,
    add_common_io_args,
    add_vocoder_args,
    build_examples_epilog,
    build_status_bar,
    build_vocoder_config,
    concat_with_crossfade,
    default_output_path,
    ensure_runtime,
    finalize_audio,
    log_error,
    log_message,
    read_audio,
    read_segment_csv,
    resolve_inputs,
    time_pitch_shift_audio,
    validate_vocoder_args,
    write_output,
    print_input_output_metrics_table,
)


def expand_segments(segments: list[SegmentSpec], total_s: float) -> list[SegmentSpec]:
    merged: list[SegmentSpec] = []
    cursor = 0.0
    for seg in segments:
        start = max(0.0, min(total_s, seg.start_s))
        end = max(0.0, min(total_s, seg.end_s))
        if end <= start:
            continue
        if start > cursor:
            merged.append(SegmentSpec(start_s=cursor, end_s=start, stretch=1.0, pitch_ratio=1.0))
        merged.append(SegmentSpec(start_s=start, end_s=end, stretch=seg.stretch, pitch_ratio=seg.pitch_ratio))
        cursor = max(cursor, end)
    if cursor < total_s:
        merged.append(SegmentSpec(start_s=cursor, end_s=total_s, stretch=1.0, pitch_ratio=1.0))
    return merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Conform audio to a CSV map. CSV columns: start_sec,end_sec,stretch,"
            " and one pitch field: pitch_semitones or pitch_cents or pitch_ratio "
            "(pitch_ratio accepts decimals, fractions like 3/2, or expressions like 2^(1/12))"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx conform input.wav --map map_conform.csv --output conformed.wav",
                "pvx conform vocal.wav --map map_just_intonation.csv --crossfade-ms 12 --output vocal_ji.wav",
                "pvx conform source.wav --map map_warp.csv --stdout | pvx denoise - --reduction-db 6 --output source_conform_clean.wav",
            ],
            notes=[
                "Map CSV requires start_sec,end_sec,stretch plus one pitch column.",
                "pitch_ratio accepts decimals, integer ratios (3/2), and expressions (2^(1/12)).",
            ],
        ),
    )
    add_common_io_args(parser, default_suffix="_conform")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--map", required=True, type=Path, help="CSV map path")
    parser.add_argument("--crossfade-ms", type=float, default=8.0, help="Segment crossfade in milliseconds")
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.crossfade_ms < 0:
        parser.error("--crossfade-ms must be >= 0")

    map_segments = read_segment_csv(args.map, has_pitch=True)
    if not map_segments:
        parser.error("Map has no valid segments")

    config = build_vocoder_config(args, phase_locking="identity", transient_preserve=True, transient_threshold=2.0)
    paths = resolve_inputs(args.inputs, parser, args)
    status = build_status_bar(args, "pvxconform", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            total_s = audio.shape[0] / sr
            segments = expand_segments(map_segments, total_s)

            chunks: list[np.ndarray] = []
            for seg in segments:
                start = int(round(seg.start_s * sr))
                end = int(round(seg.end_s * sr))
                if end <= start:
                    continue
                piece = audio[start:end, :]
                shifted = time_pitch_shift_audio(
                    piece,
                    seg.stretch,
                    seg.pitch_ratio,
                    config,
                    resample_mode=args.resample_mode,
                )
                chunks.append(shifted)

            out = concat_with_crossfade(chunks, sr, crossfade_ms=args.crossfade_ms)
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
            log_message(args, f"[ok] {path} -> {out_path} | segs={len(segments)}, dur={out.shape[0]/sr:.3f}s", min_level="verbose")
        except Exception as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)

    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(args, f"[done] pvxconform processed={len(paths)} failed={failures}", min_level="normal")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
