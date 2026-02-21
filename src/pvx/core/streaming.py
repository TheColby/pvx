#!/usr/bin/env python3
"""Stateful chunked streaming helpers for the unified pvx CLI."""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pvx.core.audio_metrics import (
    render_audio_comparison_table,
    render_audio_metrics_table,
    summarize_audio_metrics,
)
from pvx.core.output_policy import prepare_output_audio, write_metadata_sidecar
from pvx.core import voc as voc_core


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    if str(path) == "-":
        payload = sys.stdin.buffer.read()
        if not payload:
            raise ValueError("No audio bytes received on stdin")
        audio, sr = sf.read(io.BytesIO(payload), always_2d=True)
    else:
        audio, sr = sf.read(str(path), always_2d=True)
    return np.asarray(audio, dtype=np.float64), int(sr)


def _build_config(args: argparse.Namespace) -> voc_core.VocoderConfig:
    return voc_core.VocoderConfig(
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_size=args.hop_size,
        window=args.window,
        kaiser_beta=args.kaiser_beta,
        transform=voc_core.normalize_transform_name(args.transform),
        center=not args.no_center,
        phase_locking=args.phase_locking,
        phase_engine=args.phase_engine,
        ambient_phase_mix=args.ambient_phase_mix,
        phase_random_seed=args.phase_random_seed,
        transient_preserve=args.transient_preserve,
        transient_threshold=args.transient_threshold,
        onset_time_credit=args.onset_time_credit,
        onset_credit_pull=args.onset_credit_pull,
        onset_credit_max=args.onset_credit_max,
        onset_realign=not args.no_onset_realign,
    )


def _resolve_voc_args_for_stream(
    *,
    input_token: str,
    output_token: str,
    passthrough: list[str],
) -> tuple[argparse.Namespace, argparse.ArgumentParser, list[Path], set[str]]:
    voc_argv = [input_token, *passthrough]
    parser = voc_core.build_parser()
    cli_flags = voc_core.collect_cli_flags(voc_argv)
    args = parser.parse_args(voc_argv)
    args._cli_flags = cli_flags

    if ("--transient-mode" not in cli_flags) and bool(getattr(args, "transient_preserve", False)):
        args.transient_mode = "reset"

    voc_core.validate_args(args, parser)
    input_paths = voc_core.expand_inputs(args.inputs)
    if not input_paths:
        parser.error("No readable input files matched the provided paths/patterns.")
    if len(input_paths) != 1:
        parser.error("Stateful stream mode requires exactly one resolved input")
    if bool(args.pitch_map is not None) or bool(args.pitch_map_stdin):
        parser.error("Stateful stream mode does not support --pitch-map / --pitch-map-stdin")
    if bool(args.resume) or bool(args.checkpoint_dir is not None):
        parser.error("Stateful stream mode does not support checkpoint/resume flags")
    if bool(args.explain_plan):
        parser.error("Stateful stream mode does not support --explain-plan")
    if bool(args.auto_segment_seconds > 0.0):
        parser.error("Stateful stream mode manages chunking directly; remove --auto-segment-seconds")

    preset_changes = voc_core.apply_named_preset(
        args,
        preset=str(args.preset),
        provided_flags=cli_flags,
    )

    active_profile = str(args.quality_profile)
    auto_features: dict[str, float] | None = None
    # In stream mode we support auto-profile for both file and stdin sources.
    preview_audio, preview_sr = _read_audio(input_paths[0])
    if preview_audio.shape[0] == 0:
        parser.error("Input has no audio samples")
    if args.auto_profile:
        stretch_estimate = voc_core.resolve_base_stretch(args, preview_audio.shape[0], preview_sr)
        auto_features = voc_core.estimate_content_features(
            preview_audio,
            preview_sr,
            channel_mode=str(args.analysis_channel),
            lookahead_seconds=float(args.auto_profile_lookahead_seconds),
        )
        active_profile = voc_core.suggest_quality_profile(stretch_ratio=stretch_estimate, features=auto_features)

    args._active_quality_profile = active_profile
    profile_changes = voc_core.apply_quality_profile_overrides(
        args,
        profile=active_profile,
        provided_flags=cli_flags,
    )
    profile_changes = list(preset_changes) + profile_changes

    if args.auto_transform:
        resolved_transform = voc_core.resolve_transform_auto(
            requested_transform=str(args.transform),
            profile=active_profile,
            n_fft=int(args.n_fft),
            provided_flags=cli_flags,
        )
        if resolved_transform != args.transform:
            args.transform = resolved_transform
            profile_changes.append("transform")

    if args.ambient_preset:
        args.phase_engine = "random"
        args.transient_preserve = True
        args.onset_time_credit = True
        if str(args.stretch_mode) == "auto":
            args.stretch_mode = "multistage"
        args.max_stage_stretch = min(float(args.max_stage_stretch), 1.35)
        if args._active_quality_profile == "neutral":
            args._active_quality_profile = "ambient"

    voc_core.validate_args(args, parser)
    voc_core.configure_runtime_from_args(args, parser)

    if voc_core.console_level(args) >= 3:
        info = (
            f"[info] stream profile={args._active_quality_profile}, "
            f"auto_profile={'on' if args.auto_profile else 'off'}, "
            f"auto_transform={'on' if args.auto_transform else 'off'}, "
            f"transform={args.transform}"
        )
        if profile_changes:
            info += f", overrides={','.join(sorted(set(profile_changes)))}"
        voc_core.log_message(args, info, min_level="verbose")
        if auto_features is not None:
            voc_core.log_message(args, f"[info] auto-profile features={auto_features}", min_level="debug")

    if str(output_token) == "-":
        args.stdout = True
        args.output = None
    else:
        args.stdout = False
        args.output = Path(output_token).expanduser().resolve()
    args.output_dir = None

    return args, parser, input_paths, cli_flags


def _chunk_core_extract(
    *,
    full_audio: np.ndarray,
    sr: int,
    chunk_start: int,
    chunk_end: int,
    context_samples: int,
    stretch: float,
    pitch_ratio: float,
    args: argparse.Namespace,
    config: voc_core.VocoderConfig,
) -> tuple[np.ndarray, int]:
    seg_start = max(0, chunk_start - context_samples)
    seg_end = min(full_audio.shape[0], chunk_end + context_samples)
    seg = np.asarray(full_audio[seg_start:seg_end], dtype=np.float64)

    block = voc_core.process_audio_block(
        seg,
        sr,
        args,
        config,
        stretch=stretch,
        pitch_ratio=pitch_ratio,
    )

    seg_out = np.asarray(block.audio, dtype=np.float64)
    rel_start = chunk_start - seg_start
    rel_end = chunk_end - seg_start
    out_start = int(round(rel_start * stretch))
    out_end = int(round(rel_end * stretch))
    out_start = max(0, min(out_start, seg_out.shape[0]))
    out_end = max(out_start, min(out_end, seg_out.shape[0]))

    core = seg_out[out_start:out_end, :]
    target_len = max(1, int(round((chunk_end - chunk_start) * stretch)))
    core = voc_core.force_length_multi(core, target_len)
    return core, int(block.stage_count)


def _concat_exact(chunks: list[np.ndarray], channels: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, channels), dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]
    return np.vstack(chunks)


def run_stateful_stream(
    *,
    input_token: str,
    output_token: str,
    passthrough: list[str],
    chunk_seconds: float,
    context_ms: float | None,
    crossfade_ms: float,
) -> int:
    args, parser, input_paths, _ = _resolve_voc_args_for_stream(
        input_token=input_token,
        output_token=output_token,
        passthrough=passthrough,
    )

    input_path = input_paths[0]
    audio, sr = _read_audio(input_path)
    if audio.shape[0] == 0:
        raise ValueError("Input file has no audio samples")

    pitch = voc_core.choose_pitch_ratio(args, audio, sr)
    base_stretch = voc_core.resolve_base_stretch(args, audio.shape[0], sr)
    if base_stretch <= 0.0:
        parser.error("--time-stretch/--target-duration resolved to non-positive value")
    dynamic_refs: dict[str, voc_core.DynamicControlRef] = dict(getattr(args, "_dynamic_control_refs", {}) or {})

    config = _build_config(args)
    chunk_samples = max(1, int(round(float(chunk_seconds) * sr)))
    computed_context = max(config.win_length * 2, config.hop_size * 8)
    if context_ms is None:
        context_samples = int(computed_context)
    else:
        context_samples = max(0, int(round(float(context_ms) * sr / 1000.0)))
    if crossfade_ms > 0.0:
        # Stateful mode uses context-window overlap and core extraction;
        # interpret crossfade request as additional context instead of length-changing overlap.
        requested = int(round(float(crossfade_ms) * sr / 1000.0))
        context_samples = max(context_samples, requested)

    total_samples = int(audio.shape[0])
    total_chunks = max(1, math.ceil(total_samples / chunk_samples))
    dynamic_signals: dict[str, voc_core.DynamicControlSignal] = {}
    if dynamic_refs:
        total_seconds = float(total_samples) / float(sr)
        for parameter, ref in dynamic_refs.items():
            signal = voc_core.load_dynamic_control_signal(ref, total_seconds=total_seconds)
            if signal.parameter != parameter:
                signal = voc_core.DynamicControlSignal(
                    parameter=parameter,
                    interpolation=signal.interpolation,
                    order=signal.order,
                    times_sec=signal.times_sec,
                    values=signal.values,
                )
            dynamic_signals[parameter] = signal

    voc_core.log_message(
        args,
        (
            f"[stream] mode=stateful, chunks={total_chunks}, chunk_seconds={float(chunk_seconds):.4f}, "
            f"context_samples={context_samples}, stretch={base_stretch:.6f}, pitch_ratio={pitch.ratio:.6f}, "
            f"controls={'dynamic' if dynamic_signals else 'static'}"
        ),
        min_level="normal",
    )

    chunk_outputs: list[np.ndarray] = []
    stage_count = 1
    for idx in range(total_chunks):
        start = idx * chunk_samples
        end = min(total_samples, start + chunk_samples)
        if end <= start:
            continue

        chunk_args = args
        chunk_config = config
        chunk_stretch = base_stretch
        chunk_pitch_ratio = pitch.ratio
        if dynamic_signals:
            mid_sec = 0.5 * ((start + end) / float(sr))
            sample_t = np.asarray([mid_sec], dtype=np.float64)
            overrides: dict[str, float] = {}
            for parameter, signal in dynamic_signals.items():
                value = float(voc_core._sample_dynamic_signal(signal, sample_t)[0])
                if parameter == "time_stretch":
                    chunk_stretch = value
                elif parameter == "pitch_ratio":
                    chunk_pitch_ratio = value
                else:
                    overrides[parameter] = value
            chunk_stretch, chunk_pitch_ratio, clean_overrides = voc_core._finalize_dynamic_segment_values(
                args=args,
                stretch=chunk_stretch,
                pitch_ratio=chunk_pitch_ratio,
                overrides=overrides,
            )
            if clean_overrides:
                chunk_args = voc_core.clone_args_namespace(args)
                for key, value in clean_overrides.items():
                    setattr(chunk_args, key, value)
                chunk_config = voc_core.build_vocoder_config_from_args(chunk_args)

        core, stages = _chunk_core_extract(
            full_audio=audio,
            sr=sr,
            chunk_start=start,
            chunk_end=end,
            context_samples=context_samples,
            stretch=chunk_stretch,
            pitch_ratio=chunk_pitch_ratio,
            args=chunk_args,
            config=chunk_config,
        )
        stage_count = max(stage_count, stages)
        chunk_outputs.append(core)
        if voc_core.console_level(args) >= 3:
            voc_core.log_message(
                args,
                f"[stream] processed chunk {idx + 1}/{total_chunks}",
                min_level="verbose",
            )

    out_audio = _concat_exact(chunk_outputs, channels=audio.shape[1])
    if dynamic_signals:
        expected_len = max(1, int(sum(chunk.shape[0] for chunk in chunk_outputs)))
    else:
        expected_len = max(1, int(round(total_samples * base_stretch)))
    out_audio = voc_core.force_length_multi(out_audio, expected_len)

    out_sr = sr
    if args.target_sample_rate is not None and args.target_sample_rate != sr:
        new_len = max(1, int(round(out_audio.shape[0] * args.target_sample_rate / sr)))
        out_audio = voc_core.resample_multi(out_audio, new_len, args.resample_mode)
        out_sr = int(args.target_sample_rate)

    out_audio = voc_core.apply_mastering_chain(out_audio, out_sr, args)
    out_audio, resolved_subtype = prepare_output_audio(
        out_audio,
        int(out_sr),
        args,
        explicit_subtype=getattr(args, "subtype", None),
    )

    output_path = Path("-") if bool(getattr(args, "stdout", False)) else Path(args.output)
    if (not args.stdout) and output_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

    metrics_table = render_audio_metrics_table(
        [
            (f"in:{input_path}", summarize_audio_metrics(audio, int(sr))),
            (f"out:{output_path}", summarize_audio_metrics(out_audio, int(out_sr))),
        ],
        title="Audio Metrics",
        include_delta_from_first=True,
    )
    compare_table = render_audio_comparison_table(
        reference_label=f"in:{input_path}",
        reference_audio=audio,
        reference_sr=int(sr),
        candidate_label=f"out:{output_path}",
        candidate_audio=out_audio,
        candidate_sr=int(out_sr),
        title="Audio Compare Metrics",
    )
    voc_core.log_message(args, f"{metrics_table}\n{compare_table}", min_level="quiet")

    if not args.dry_run:
        voc_core._write_audio_output(output_path, out_audio, out_sr, args, subtype=resolved_subtype)
        sidecar = write_metadata_sidecar(
            output_path=output_path,
            input_path=(None if str(input_path) == "-" else input_path),
            audio=out_audio,
            sample_rate=int(out_sr),
            subtype=resolved_subtype,
            args=args,
            extra={
                "stream_mode": "stateful",
                "chunk_seconds": float(chunk_seconds),
                "context_samples": int(context_samples),
                "chunks": int(total_chunks),
                "stages": int(stage_count),
                "quality_profile": str(getattr(args, "_active_quality_profile", "neutral")),
                "transform": str(config.transform),
                "window": str(config.window),
                "phase_engine": str(config.phase_engine),
                "transient_mode": str(args.transient_mode),
                "stereo_mode": str(args.stereo_mode),
                "coherence_strength": float(args.coherence_strength),
                "dynamic_controls": [
                    {
                        "parameter": ref.parameter,
                        "path": str(ref.path),
                        "value_kind": str(ref.value_kind),
                        "interp": str(ref.interpolation),
                        "order": int(ref.order),
                    }
                    for ref in dynamic_refs.values()
                ],
            },
        )
        if sidecar is not None:
            voc_core.log_message(args, f"[info] metadata sidecar -> {sidecar}", min_level="verbose")

    voc_core.log_message(
        args,
        (
            f"[stream] done -> {output_path} | chunks={total_chunks}, stages={stage_count}, "
            f"dur={audio.shape[0]/sr:.3f}s->{out_audio.shape[0]/out_sr:.3f}s"
        ),
        min_level="normal",
    )
    return 0
