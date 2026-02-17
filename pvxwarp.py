#!/usr/bin/env python3
"""Time-warp an input according to a user-provided stretch map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pvxcommon import (
    SegmentSpec,
    add_common_io_args,
    add_vocoder_args,
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
)


def fill_stretch_segments(segments: list[SegmentSpec], total_s: float) -> list[SegmentSpec]:
    out: list[SegmentSpec] = []
    cursor = 0.0
    for seg in segments:
        start = max(0.0, min(total_s, seg.start_s))
        end = max(0.0, min(total_s, seg.end_s))
        if end <= start:
            continue
        if start > cursor:
            out.append(SegmentSpec(start_s=cursor, end_s=start, stretch=1.0, pitch_ratio=1.0))
        out.append(SegmentSpec(start_s=start, end_s=end, stretch=seg.stretch, pitch_ratio=1.0))
        cursor = max(cursor, end)
    if cursor < total_s:
        out.append(SegmentSpec(start_s=cursor, end_s=total_s, stretch=1.0, pitch_ratio=1.0))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply variable time-stretch map from CSV")
    add_common_io_args(parser, default_suffix="_warp")
    add_vocoder_args(parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512)
    parser.add_argument("--map", required=True, type=Path, help="CSV map with start_sec,end_sec,stretch")
    parser.add_argument("--crossfade-ms", type=float, default=8.0)
    parser.add_argument("--resample-mode", choices=["auto", "fft", "linear"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if args.crossfade_ms < 0:
        parser.error("--crossfade-ms must be >= 0")

    segments = read_segment_csv(args.map, has_pitch=False)
    if not segments:
        parser.error("Map has no valid rows")

    config = build_vocoder_config(args, phase_locking="identity", transient_preserve=True, transient_threshold=2.0)
    paths = resolve_inputs(args.inputs, parser)
    status = build_status_bar(args, "pvxwarp", len(paths))

    failures = 0
    for idx, path in enumerate(paths, start=1):
        try:
            audio, sr = read_audio(path)
            full = fill_stretch_segments(segments, audio.shape[0] / sr)

            pieces: list[np.ndarray] = []
            for seg in full:
                s = int(round(seg.start_s * sr))
                e = int(round(seg.end_s * sr))
                if e <= s:
                    continue
                part = audio[s:e, :]
                pieces.append(
                    time_pitch_shift_audio(part, seg.stretch, 1.0, config, resample_mode=args.resample_mode)
                )

            out = concat_with_crossfade(pieces, sr, crossfade_ms=args.crossfade_ms)
            out = finalize_audio(out, args)
            out_path = default_output_path(path, args)
            write_output(out_path, out, sr, args)
            log_message(args, f"[ok] {path} -> {out_path} | segs={len(full)} dur={out.shape[0]/sr:.3f}s", min_level="verbose")
        except Exception as exc:
            failures += 1
            log_error(args, f"[error] {path}: {exc}")
        status.step(idx, path.name)

    status.finish("done" if failures == 0 else f"errors={failures}")
    log_message(args, f"[done] pvxwarp processed={len(paths)} failed={failures}", min_level="normal")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
