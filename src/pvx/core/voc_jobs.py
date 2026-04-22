#!/usr/bin/env python3

"""Job orchestration helpers for pvx voc renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from pvx.core import voc as voc_core
from pvx.core.audio_metrics import (
    render_audio_comparison_table,
    render_audio_metrics_table,
    summarize_audio_metrics,
)
from pvx.core.output_policy import prepare_output_audio, write_metadata_sidecar


def compute_output_path(
    input_path: Path,
    output_dir: Path | None,
    suffix: str,
    output_format: str | None,
) -> Path:
    base_dir = output_dir if output_dir is not None else input_path.parent
    ext = output_format.lower().lstrip(".") if output_format else input_path.suffix.lstrip(".")
    if not ext:
        ext = "wav"
    return base_dir / f"{input_path.stem}{suffix}.{ext}"


def _stream_format_name(output_format: str | None, output_path: Path | None = None) -> str:
    if output_format:
        ext = output_format.lower().lstrip(".")
    elif output_path is not None and str(output_path) != "-" and output_path.suffix:
        ext = output_path.suffix.lower().lstrip(".")
    else:
        ext = "wav"
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


def _hash_file_contents(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def _checkpoint_chunk_meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _read_audio_input(input_path: Path) -> tuple[np.ndarray, int]:
    if str(input_path) == "-":
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
            bytes_written = shutil.copyfileobj(sys.stdin.buffer, tmp)
            tmp.flush()
            if bytes_written is None:
                size = tmp.tell()
            else:
                size = int(bytes_written)
            if size <= 0:
                raise ValueError("No audio bytes received on stdin")
            audio, sr = sf.read(tmp.name, always_2d=True)
    else:
        audio, sr = sf.read(str(input_path), always_2d=True)
    return audio.astype(np.float64, copy=False), int(sr)


def _write_audio_output(
    output_path: Path,
    audio: np.ndarray,
    sr: int,
    args: argparse.Namespace,
    *,
    subtype: str | None = None,
) -> None:
    if bool(getattr(args, "stdout", False)) or str(output_path) == "-":
        stream_fmt = _stream_format_name(
            getattr(args, "output_format", None), output_path=output_path
        )
        ext = stream_fmt.lower()
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
            sf.write(tmp.name, audio, sr, format=stream_fmt, subtype=subtype)
            tmp.flush()
            tmp.seek(0)
            shutil.copyfileobj(tmp, sys.stdout.buffer)
            sys.stdout.buffer.flush()
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype=subtype)


def concat_audio_chunks(chunks: list[np.ndarray], *, sr: int, crossfade_ms: float) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 1), dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]

    fade = max(0, int(round(sr * max(0.0, crossfade_ms) / 1000.0)))
    out = chunks[0]
    for nxt in chunks[1:]:
        if fade <= 0 or out.shape[0] < fade or nxt.shape[0] < fade:
            out = np.vstack([out, nxt])
            continue
        w = np.linspace(0.0, 1.0, num=fade, endpoint=True)[:, None]
        blend = out[-fade:, :] * (1.0 - w) + nxt[:fade, :] * w
        out = np.vstack([out[:-fade, :], blend, nxt[fade:, :]])
    return out


def build_uniform_control_segments(
    *,
    total_seconds: float,
    segment_seconds: float,
    stretch: float,
    pitch_ratio: float,
) -> list[voc_core.ControlSegment]:
    total = max(0.0, float(total_seconds))
    seg = max(1e-3, float(segment_seconds))
    if total <= 0.0:
        return []

    out: list[voc_core.ControlSegment] = []
    cursor = 0.0
    while cursor < total:
        end = min(total, cursor + seg)
        out.append(
            voc_core.ControlSegment(
                start_sec=cursor,
                end_sec=end,
                stretch=float(stretch),
                pitch_ratio=float(pitch_ratio),
                confidence=1.0,
            )
        )
        cursor = end
    return out


def _checkpoint_job_id(
    *,
    input_path: Path,
    args: argparse.Namespace,
    base_stretch: float,
    pitch_ratio: float,
) -> str:
    return _render_fingerprint(
        input_path=input_path,
        args=args,
        base_stretch=base_stretch,
        pitch_ratio=pitch_ratio,
    )[:16]


def _render_fingerprint(
    *,
    input_path: Path,
    args: argparse.Namespace,
    base_stretch: float,
    pitch_ratio: float,
    audio: np.ndarray | None = None,
    control_payload: str | None = None,
    dynamic_payloads: dict[str, str] | None = None,
) -> str:
    dynamic_refs = dict(getattr(args, "_dynamic_control_refs", {}) or {})
    checkpoint_inputs = {
        "pitch_map": None,
        "pitch_map_hash": None,
        "dynamic_controls": [],
        "routes": list(getattr(args, "route", []) or []),
    }
    pitch_map = getattr(args, "pitch_map", None)
    if pitch_map is not None and str(pitch_map) != "-":
        pitch_map_path = Path(pitch_map)
        checkpoint_inputs["pitch_map"] = str(pitch_map_path.resolve())
        if control_payload is not None:
            checkpoint_inputs["pitch_map_hash"] = hashlib.sha1(
                control_payload.encode("utf-8")
            ).hexdigest()
        elif pitch_map_path.exists():
            checkpoint_inputs["pitch_map_hash"] = _hash_file_contents(pitch_map_path)
    checkpoint_inputs["dynamic_controls"] = [
        {
            "parameter": ref.parameter,
            "path": str(ref.path.resolve()),
            "hash": (
                hashlib.sha1(dynamic_payloads[ref.parameter].encode("utf-8")).hexdigest()
                if dynamic_payloads is not None and ref.parameter in dynamic_payloads
                else _hash_file_contents(ref.path)
            ),
            "value_kind": ref.value_kind,
            "interp": ref.interpolation,
            "order": int(ref.order),
        }
        for ref in dynamic_refs.values()
    ]
    payload = {
        "input": str(input_path),
        "input_hash": None,
        "time_stretch": float(base_stretch),
        "pitch_ratio": float(pitch_ratio),
        "target_duration": getattr(args, "target_duration", None),
        "n_fft": int(getattr(args, "n_fft", 0)),
        "win_length": int(getattr(args, "win_length", 0)),
        "hop_size": int(getattr(args, "hop_size", 0)),
        "window": str(getattr(args, "window", "hann")),
        "phase_engine": str(getattr(args, "phase_engine", "propagate")),
        "transform": str(getattr(args, "transform", "fft")),
        "profile": str(getattr(args, "_active_quality_profile", "neutral")),
        "processing": {
            "phase_locking": str(getattr(args, "phase_locking", "off")),
            "stretch_mode": str(getattr(args, "stretch_mode", "standard")),
            "pitch_mode": str(getattr(args, "pitch_mode", "standard")),
            "transient_mode": str(getattr(args, "transient_mode", "off")),
            "transient_sensitivity": float(getattr(args, "transient_sensitivity", 0.5)),
            "transient_protect_ms": float(getattr(args, "transient_protect_ms", 30.0)),
            "transient_crossfade_ms": float(getattr(args, "transient_crossfade_ms", 10.0)),
            "stereo_mode": str(getattr(args, "stereo_mode", "independent")),
            "ref_channel": int(getattr(args, "ref_channel", 0)),
            "coherence_strength": float(getattr(args, "coherence_strength", 0.0)),
            "auto_segment_seconds": float(getattr(args, "auto_segment_seconds", 0.0)),
            "multires_fusion": bool(getattr(args, "multires_fusion", False)),
            "multires_ffts": list(getattr(args, "_multires_ffts", []) or []),
            "multires_weights": list(getattr(args, "_multires_weights", []) or []),
            "ambient_phase_mix": float(getattr(args, "ambient_phase_mix", 0.5)),
            "formant_lifter": str(getattr(args, "formant_lifter", 32)),
            "formant_strength": str(getattr(args, "formant_strength", 1.0)),
            "formant_max_gain_db": str(getattr(args, "formant_max_gain_db", 12.0)),
            "pitch_conf_min": float(getattr(args, "pitch_conf_min", 0.0)),
            "pitch_lowconf_mode": str(getattr(args, "pitch_lowconf_mode", "hold")),
            "pitch_map_smooth_ms": float(getattr(args, "pitch_map_smooth_ms", 0.0)),
            "pitch_map_crossfade_ms": float(getattr(args, "pitch_map_crossfade_ms", 0.0)),
        },
        "control_inputs": checkpoint_inputs,
    }
    if audio is not None:
        payload["input_hash"] = hashlib.sha1(
            np.ascontiguousarray(audio).view(np.uint8).tobytes()
        ).hexdigest()
    elif str(input_path) != "-" and input_path.exists():
        payload["input_hash"] = _hash_file_contents(input_path)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def resolve_checkpoint_context(
    *,
    input_path: Path,
    args: argparse.Namespace,
    base_stretch: float,
    pitch_ratio: float,
) -> tuple[str, Path] | None:
    checkpoint_root = getattr(args, "checkpoint_dir", None)
    if checkpoint_root is None:
        return None
    cp_root = Path(checkpoint_root).resolve()
    cp_id = str(getattr(args, "checkpoint_id", "") or "").strip()
    if not cp_id:
        cp_id = _checkpoint_job_id(
            input_path=input_path,
            args=args,
            base_stretch=base_stretch,
            pitch_ratio=pitch_ratio,
        )
    cp_dir = cp_root / cp_id
    cp_dir.mkdir(parents=True, exist_ok=True)
    return cp_id, cp_dir


def load_checkpoint_chunk(
    path: Path,
    *,
    expected_meta: dict[str, Any] | None = None,
) -> np.ndarray:
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Checkpoint chunk has invalid shape: {path}")
    meta_path = _checkpoint_chunk_meta_path(path)
    if expected_meta is not None:
        if not meta_path.exists():
            raise ValueError(f"Checkpoint metadata missing for chunk: {path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, expected in expected_meta.items():
            actual = meta.get(key)
            if actual != expected:
                raise ValueError(
                    f"Checkpoint chunk metadata mismatch for {path.name}: "
                    f"{key} expected {expected!r}, got {actual!r}"
                )
        if int(meta.get("output_samples", values.shape[0])) != int(values.shape[0]):
            raise ValueError(f"Checkpoint chunk sample count mismatch for: {path}")
        if int(meta.get("channels", values.shape[1])) != int(values.shape[1]):
            raise ValueError(f"Checkpoint chunk channel count mismatch for: {path}")
    return values


def save_checkpoint_chunk(
    path: Path,
    values: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float64)
    np.save(path, arr, allow_pickle=False)
    if metadata is not None:
        payload = dict(metadata)
        payload.setdefault("output_samples", int(arr.shape[0]))
        payload.setdefault("channels", int(arr.shape[1] if arr.ndim == 2 else 1))
        _checkpoint_chunk_meta_path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def write_manifest(
    path: Path,
    entries: list[dict[str, Any]],
    *,
    append: bool,
) -> None:
    payload_entries: list[dict[str, Any]] = []
    if append and path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                payload_entries.extend(list(existing.get("entries", [])))
            elif isinstance(existing, list):
                payload_entries.extend(existing)
        except (json.JSONDecodeError, OSError, ValueError):
            raise ValueError(f"Cannot append to invalid manifest: {path}") from None
    payload_entries.extend(entries)

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entry_count": len(payload_entries),
        "entries": payload_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    if args.stdout:
        return Path("-")
    if args.output is not None:
        return args.output
    source_path = Path("stdin.wav") if str(input_path) == "-" else input_path
    return compute_output_path(source_path, args.output_dir, args.suffix, args.output_format)


def _ensure_writable_target(path: Path, args: argparse.Namespace, *, label: str) -> None:
    if str(path) == "-" or bool(getattr(args, "dry_run", False)):
        return
    if path.exists() and not bool(getattr(args, "overwrite", False)):
        raise FileExistsError(f"{label} exists: {path}. Use --overwrite/--force to replace it.")


def _preflight_file_writes(
    *,
    input_path: Path,
    args: argparse.Namespace,
    checkpoint_dir: Path | None,
    checkpoint_id: str | None,
    segment_count: int,
) -> Path:
    output_path = _resolve_output_path(input_path, args)
    _ensure_writable_target(output_path, args, label="Output")

    if str(getattr(args, "metadata_policy", "none")).lower() != "none" and str(output_path) != "-":
        _ensure_writable_target(_metadata_sidecar_path(output_path), args, label="Metadata sidecar")

    if checkpoint_dir is not None and checkpoint_id is not None and not bool(getattr(args, "resume", False)):
        _ensure_writable_target(checkpoint_dir / "state.json", args, label="Checkpoint state")
        for seg_idx in range(segment_count):
            chunk_path = checkpoint_dir / f"segment_{seg_idx:05d}.npy"
            _ensure_writable_target(chunk_path, args, label="Checkpoint chunk")
            _ensure_writable_target(
                _checkpoint_chunk_meta_path(chunk_path),
                args,
                label="Checkpoint metadata",
            )
    return output_path


def _consume_prefetched_audio(
    input_path: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int] | None:
    cache = dict(getattr(args, "_prefetched_audio_cache", {}) or {})
    key = str(input_path)
    if key not in cache:
        return None
    payload = cache.pop(key)
    args._prefetched_audio_cache = cache
    audio, sr = payload
    return np.asarray(audio, dtype=np.float64, copy=False), int(sr)


def _preflight_checkpoint_targets(args: argparse.Namespace) -> None:
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint_dir is None or bool(getattr(args, "resume", False)) or bool(getattr(args, "dry_run", False)):
        return
    cp_root = Path(checkpoint_dir)
    state_path = cp_root / "state.json"
    _ensure_writable_target(state_path, args, label="Checkpoint state")
    for pattern, label in (("segment_*.npy", "Checkpoint chunk"), ("segment_*.json", "Checkpoint metadata")):
        for path in cp_root.glob(pattern):
            _ensure_writable_target(path, args, label=label)


def process_file(
    input_path: Path,
    args: argparse.Namespace,
    config: voc_core.VocoderConfig,
    file_index: int = 0,
    file_total: int = 1,
) -> voc_core.JobResult:
    progress_enabled = not voc_core.is_quiet(args)
    progress = voc_core.ProgressBar(
        label=f"{input_path.name} [{file_index + 1}/{file_total}]",
        enabled=progress_enabled,
    )

    def make_progress_callback(
        start: float, end: float, detail: str
    ) -> voc_core.ProgressCallback | None:
        if not progress_enabled:
            return None

        span = max(0.0, end - start)

        def _callback(done: int, total: int) -> None:
            denom = max(1, total)
            progress.set(start + span * (done / denom), detail)

        return _callback

    output_path = _preflight_file_writes(
        input_path=input_path,
        args=args,
        checkpoint_dir=None,
        checkpoint_id=None,
        segment_count=0,
    )

    _preflight_checkpoint_targets(args)

    progress.set(0.02, "read")
    prefetched = _consume_prefetched_audio(input_path, args)
    if prefetched is None:
        audio, sr = _read_audio_input(input_path)
    else:
        audio, sr = prefetched

    if audio.shape[0] == 0:
        raise ValueError("Input file has no audio samples")

    progress.set(0.08, "analyze")
    pitch = voc_core.choose_pitch_ratio(args, audio, sr)
    base_stretch = voc_core.resolve_base_stretch(args, audio.shape[0], sr)
    use_dynamic_controls = bool(getattr(args, "_dynamic_control_refs", {}))
    use_control_map = bool(args.pitch_map is not None) or bool(args.pitch_map_stdin)
    auto_segment_seconds = float(getattr(args, "auto_segment_seconds", 0.0))
    use_auto_segments = (
        (not use_control_map) and (not use_dynamic_controls) and (auto_segment_seconds > 0.0)
    )
    segment_mode = use_control_map or use_auto_segments or use_dynamic_controls
    map_segments: list[voc_core.ControlSegment] = []
    internal_stretch = base_stretch * pitch.ratio
    sync_plan: voc_core.FourierSyncPlan | None = None
    stage_count = 1
    checkpoint_id: str | None = None
    checkpoint_dir: Path | None = None
    checkpoint_state_path: Path | None = None
    render_fingerprint: str | None = None

    if segment_mode:
        progress.set(0.10, "map")
        total_seconds = audio.shape[0] / float(sr)
        if use_dynamic_controls:
            map_segments = voc_core.build_dynamic_control_segments(
                args=args,
                sr=sr,
                total_seconds=total_seconds,
                base_stretch=base_stretch,
                base_pitch_ratio=pitch.ratio,
            )
        elif use_control_map:
            raw_segments = voc_core.load_control_segments(
                args,
                default_stretch=base_stretch,
                default_pitch_ratio=pitch.ratio,
            )
            map_segments = voc_core.expand_control_segments(
                raw_segments,
                total_seconds=total_seconds,
                default_stretch=base_stretch,
                default_pitch_ratio=pitch.ratio,
            )
        else:
            map_segments = build_uniform_control_segments(
                total_seconds=total_seconds,
                segment_seconds=auto_segment_seconds,
                stretch=base_stretch,
                pitch_ratio=pitch.ratio,
            )
        if not map_segments:
            raise ValueError("Control map produced no usable segments")

        checkpoint_context = resolve_checkpoint_context(
            input_path=input_path,
            args=args,
            base_stretch=base_stretch,
            pitch_ratio=pitch.ratio,
        )
        if checkpoint_context is not None:
            checkpoint_id, checkpoint_dir = checkpoint_context
            checkpoint_state_path = checkpoint_dir / "state.json"
            render_fingerprint = _render_fingerprint(
                input_path=input_path,
                args=args,
                base_stretch=base_stretch,
                pitch_ratio=pitch.ratio,
                audio=audio,
            )

        output_path = _preflight_file_writes(
            input_path=input_path,
            args=args,
            checkpoint_dir=checkpoint_dir,
            checkpoint_id=checkpoint_id,
            segment_count=len(map_segments),
        )

        chunk_list: list[np.ndarray] = []
        for seg_idx, seg in enumerate(map_segments):
            start = int(round(seg.start_sec * sr))
            end = int(round(seg.end_sec * sr))
            if end <= start:
                continue
            progress_fraction = 0.12 + 0.70 * (seg_idx / max(1, len(map_segments)))
            progress_next_fraction = 0.12 + 0.70 * ((seg_idx + 1) / max(1, len(map_segments)))
            segment_detail = f"segment {seg_idx + 1}/{len(map_segments)}"
            progress.set(progress_fraction, segment_detail)
            checkpoint_chunk_path = (
                None if checkpoint_dir is None else checkpoint_dir / f"segment_{seg_idx:05d}.npy"
            )
            reused = False
            if (
                checkpoint_chunk_path is not None
                and bool(getattr(args, "resume", False))
                and checkpoint_chunk_path.exists()
            ):
                chunk = load_checkpoint_chunk(
                    checkpoint_chunk_path,
                    expected_meta={
                        "segment_index": int(seg_idx),
                        "input_start": int(start),
                        "input_end": int(end),
                        "sample_rate": int(sr),
                        "stretch": float(seg.stretch),
                        "pitch_ratio": float(seg.pitch_ratio),
                        "render_fingerprint": render_fingerprint,
                    },
                )
                reused = True
            else:
                piece = audio[start:end, :]
                segment_args = args
                segment_config = config
                if seg.overrides:
                    segment_args = voc_core.clone_args_namespace(args)
                    for key, value in seg.overrides.items():
                        setattr(segment_args, key, value)
                    if str(getattr(segment_args, "transient_mode", "off")) == "reset":
                        segment_args.transient_preserve = True
                    segment_config = voc_core.build_vocoder_config_from_args(segment_args)

                block = voc_core.process_audio_block(
                    piece,
                    sr,
                    segment_args,
                    segment_config,
                    stretch=seg.stretch,
                    pitch_ratio=seg.pitch_ratio,
                    progress_callback_factory=(
                        lambda _start,
                        _end,
                        _detail,
                        start_fraction=progress_fraction,
                        end_fraction=progress_next_fraction,
                        detail=segment_detail: make_progress_callback(
                            start_fraction,
                            end_fraction,
                            detail,
                        )
                    ),
                )
                chunk = block.audio
                stage_count = max(stage_count, int(block.stage_count))
                if checkpoint_chunk_path is not None:
                    save_checkpoint_chunk(
                        checkpoint_chunk_path,
                        chunk,
                        metadata={
                            "segment_index": int(seg_idx),
                            "input_start": int(start),
                            "input_end": int(end),
                            "sample_rate": int(sr),
                            "stretch": float(seg.stretch),
                            "pitch_ratio": float(seg.pitch_ratio),
                            "render_fingerprint": render_fingerprint,
                        },
                    )
            chunk_list.append(chunk)

            if checkpoint_state_path is not None:
                state = {
                    "input_path": str(input_path),
                    "sample_rate": int(sr),
                    "segments_total": len(map_segments),
                    "segments_completed": seg_idx + 1,
                    "last_segment_reused": reused,
                    "profile": str(getattr(args, "_active_quality_profile", "neutral")),
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                checkpoint_state_path.write_text(
                    json.dumps(state, indent=2) + "\n", encoding="utf-8"
                )
        progress.set(0.88, "assemble")
        crossfade_ms = float(args.pitch_map_crossfade_ms)
        if use_auto_segments or use_dynamic_controls:
            crossfade_ms = 0.0
        out_audio = concat_audio_chunks(
            chunk_list,
            sr=sr,
            crossfade_ms=crossfade_ms,
        )
        if map_segments:
            durations = np.array(
                [max(1e-9, seg.end_sec - seg.start_sec) for seg in map_segments],
                dtype=np.float64,
            )
            stretch_values = np.array([seg.stretch for seg in map_segments], dtype=np.float64)
            pitch_values = np.array([seg.pitch_ratio for seg in map_segments], dtype=np.float64)
            total_weight = float(np.sum(durations))
            if total_weight > 0.0:
                base_stretch = float(np.sum(stretch_values * durations) / total_weight)
                pitch = voc_core.PitchConfig(
                    ratio=float(np.sum(pitch_values * durations) / total_weight)
                )
                internal_stretch = base_stretch * pitch.ratio
    else:
        block = voc_core.process_audio_block(
            audio,
            sr,
            args,
            config,
            stretch=base_stretch,
            pitch_ratio=pitch.ratio,
            progress_callback_factory=make_progress_callback,
        )
        out_audio = block.audio
        internal_stretch = block.internal_stretch
        sync_plan = block.sync_plan
        stage_count = int(block.stage_count)

    if args.target_duration is not None:
        exact_len = max(1, int(round(args.target_duration * sr)))
        out_audio = voc_core.force_length_multi(out_audio, exact_len)

    out_sr = sr
    if args.target_sample_rate is not None and args.target_sample_rate != sr:
        new_len = max(1, int(round(out_audio.shape[0] * args.target_sample_rate / sr)))
        out_audio = voc_core.resample_multi(out_audio, new_len, args.resample_mode)
        out_sr = args.target_sample_rate

    out_audio = voc_core.apply_mastering_chain(out_audio, out_sr, args)
    out_audio, resolved_subtype = prepare_output_audio(
        out_audio,
        int(out_sr),
        args,
        explicit_subtype=getattr(args, "subtype", None),
    )

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
        progress.set(0.96, "write")
        _write_audio_output(output_path, out_audio, out_sr, args, subtype=resolved_subtype)
        sidecar = write_metadata_sidecar(
            output_path=output_path,
            input_path=(None if str(input_path) == "-" else input_path),
            audio=out_audio,
            sample_rate=int(out_sr),
            subtype=resolved_subtype,
            args=args,
            extra={
                "quality_profile": str(getattr(args, "_active_quality_profile", "neutral")),
                "stages": int(stage_count),
                "control_map_segments": len(map_segments),
                "dynamic_controls": [
                    {
                        "parameter": ref.parameter,
                        "path": str(ref.path),
                        "value_kind": ref.value_kind,
                        "interp": ref.interpolation,
                        "order": int(ref.order),
                    }
                    for ref in dict(getattr(args, "_dynamic_control_refs", {}) or {}).values()
                ],
                "checkpoint_id": checkpoint_id,
                "transform": str(config.transform),
                "window": str(config.window),
                "phase_engine": str(config.phase_engine),
                "transient_mode": str(args.transient_mode),
                "stereo_mode": str(args.stereo_mode),
                "coherence_strength": float(args.coherence_strength),
            },
        )
        if sidecar is not None:
            voc_core.log_message(args, f"[info] metadata sidecar -> {sidecar}", min_level="verbose")

    if voc_core.console_level(args) >= voc_core.VERBOSITY_TO_LEVEL["verbose"]:
        rt = voc_core.runtime_config()
        msg = (
            f"[info] {input_path.name}: channels={audio.shape[1]}, sr={sr}, "
            f"stretch={base_stretch:.6f}, pitch_ratio={pitch.ratio:.6f}, "
            f"internal_stretch={internal_stretch:.6f}, "
            f"phase_locking={config.phase_locking}, phase_engine={config.phase_engine}, "
            f"transient_mode={args.transient_mode}, "
            f"onset_credit={'on' if config.onset_time_credit else 'off'}, "
            f"stereo_mode={args.stereo_mode}, coherence={float(args.coherence_strength):.2f}, "
            f"pitch_mode={args.pitch_mode}, "
            f"fourier_sync={'on' if args.fourier_sync else 'off'}, "
            f"device={rt.active_device}, control_mode="
            f"{'dynamic' if use_dynamic_controls else ('map' if use_control_map else ('auto' if use_auto_segments else 'off'))}, "
            f"stretch_mode={args.stretch_mode}, stages={stage_count}"
        )
        if pitch.source_f0_hz is not None:
            msg += f", detected_f0={pitch.source_f0_hz:.3f}Hz"
        if sync_plan is not None and sync_plan.f0_track_hz.size:
            msg += (
                f", sync_f0_med={float(np.median(sync_plan.f0_track_hz)):.3f}Hz"
                f", sync_fft_med={int(np.median(sync_plan.frame_lengths))}"
            )
        if map_segments:
            msg += f", map_segments={len(map_segments)}"
        if checkpoint_id is not None:
            msg += f", checkpoint_id={checkpoint_id}"
        if resolved_subtype is not None:
            msg += f", subtype={resolved_subtype}"
        voc_core.log_message(args, msg, min_level="verbose")

    if checkpoint_state_path is not None:
        state = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sample_rate": int(out_sr),
            "segments_total": len(map_segments),
            "complete": True,
            "profile": str(getattr(args, "_active_quality_profile", "neutral")),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        checkpoint_state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    progress.finish("done")

    return voc_core.JobResult(
        input_path=input_path,
        output_path=output_path,
        in_sr=sr,
        out_sr=out_sr,
        in_samples=audio.shape[0],
        out_samples=out_audio.shape[0],
        channels=audio.shape[1],
        stretch=base_stretch,
        pitch_ratio=pitch.ratio,
        stage_count=stage_count,
        control_map_segments=len(map_segments),
        quality_profile=str(getattr(args, "_active_quality_profile", "neutral")),
        checkpoint_id=checkpoint_id,
    )


__all__ = [
    "_read_audio_input",
    "_stream_format_name",
    "_write_audio_output",
    "build_uniform_control_segments",
    "compute_output_path",
    "concat_audio_chunks",
    "load_checkpoint_chunk",
    "process_file",
    "resolve_checkpoint_context",
    "save_checkpoint_chunk",
    "write_manifest",
]
