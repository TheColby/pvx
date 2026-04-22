"""CLI entrypoint for the phase-vocoder tool."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pvx.core.presets import PRESET_CHOICES
from pvx.core.voc import (
    DynamicControlRef,
    JobResult,
    apply_quality_profile_overrides,
    build_parser,
    build_vocoder_config_from_args,
    configure_runtime_from_args,
    looks_like_control_signal_reference,
    parse_int_cli_value,
    ensure_runtime_dependencies,
    estimate_content_features,
    parse_numeric_expression,
    read_audio_input,
    process_file,
    resolve_base_stretch,
    resolve_transform_auto,
    runtime_config,
    suggest_quality_profile,
    validate_args,
    write_manifest,
)
from pvx.core.voc_console import (
    VERBOSITY_TO_LEVEL,
    apply_named_preset,
    clone_args_namespace,
    collect_cli_flags,
    console_level,
    log_error,
    log_message,
    print_cli_examples,
)

__all__ = ["build_parser", "expand_inputs", "main", "run_guided_mode"]


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _prompt_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    value = _prompt_text(prompt, default).strip().lower()
    if value not in choices:
        valid = ", ".join(choices)
        raise ValueError(f"Expected one of: {valid}")
    return value


def run_guided_mode(args: argparse.Namespace) -> argparse.Namespace:
    if not sys.stdin.isatty():
        raise ValueError("--guided requires an interactive terminal (TTY stdin)")

    print("pvxvoc guided mode")
    print("Press Enter to accept defaults.\n")

    out = clone_args_namespace(args)
    out.inputs = list(getattr(args, "inputs", []) or [])

    if not out.inputs:
        out.inputs = [_prompt_text("Input WAV/FLAC path", "input.wav")]

    if out.output is None and not out.stdout:
        output_text = _prompt_text("Output path", "output_pv.wav")
        if output_text:
            out.output = Path(output_text)

    mode = _prompt_choice("Operation (stretch/pitch/both)", ("stretch", "pitch", "both"), "stretch")
    if mode in {"stretch", "both"}:
        stretch_raw = _prompt_text("Stretch factor (>0)", f"{float(out.time_stretch):.3f}")
        out.time_stretch = float(
            parse_numeric_expression(stretch_raw, context="guided stretch factor")
        )

    if (
        mode in {"pitch", "both"}
        and out.pitch_shift_cents is None
        and out.pitch_shift_ratio is None
        and out.target_f0 is None
    ):
        semi_raw = _prompt_text("Pitch shift semitones", "0")
        out.pitch_shift_semitones = float(
            parse_numeric_expression(semi_raw, context="guided semitones")
        )

    out.preset = _prompt_choice(
        "Preset (none/default/vocal/vocal_studio/drums_safe/ambient/extreme/extreme_ambient/stereo_coherent)",
        PRESET_CHOICES,
        str(getattr(out, "preset", "none") or "none"),
    )
    out.device = _prompt_choice("Device (auto/cpu/cuda)", ("auto", "cpu", "cuda"), str(out.device))

    if _prompt_choice("Write to stdout instead of file? (no/yes)", ("no", "yes"), "no") == "yes":
        out.stdout = True
        out.output = None

    return out


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if pattern == "-":
            paths.append(Path("-"))
            continue
        if any(ch in pattern for ch in "*?["):
            matches = [Path(match) for match in glob.glob(pattern, recursive=True)]
        else:
            matches = [Path(pattern)]
        for match in matches:
            if match.is_file():
                paths.append(match)

    unique: list[Path] = []
    seen: set[Path] = set()
    saw_stdin = False
    for path in paths:
        if str(path) == "-":
            if not saw_stdin:
                unique.append(path)
                saw_stdin = True
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _build_explain_plan(
    args: argparse.Namespace,
    input_paths: list[Path],
    config: Any,
    profile_changes: list[str],
    auto_features: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "active_profile": str(args._active_quality_profile),
        "profile_overrides_applied": sorted(set(profile_changes)),
        "auto_profile_features": auto_features,
        "inputs": [str(path) for path in input_paths],
        "config": {
            "n_fft": config.n_fft,
            "win_length": config.win_length,
            "hop_size": config.hop_size,
            "window": config.window,
            "transform": config.transform,
            "phase_locking": config.phase_locking,
            "phase_engine": config.phase_engine,
            "transient_mode": str(args.transient_mode),
            "transient_sensitivity": float(args.transient_sensitivity),
            "transient_protect_ms": float(args.transient_protect_ms),
            "transient_crossfade_ms": float(args.transient_crossfade_ms),
            "stereo_mode": str(args.stereo_mode),
            "ref_channel": int(args.ref_channel),
            "coherence_strength": float(args.coherence_strength),
            "multires_fusion": bool(args.multires_fusion),
            "multires_ffts": list(getattr(args, "_multires_ffts", [])),
            "multires_weights": list(getattr(args, "_multires_weights", [])),
        },
        "runtime": {
            "device_requested": str(args.device),
            "device_active": runtime_config().active_device,
            "cuda_device": int(args.cuda_device),
        },
        "io": {
            "output_dir": None if args.output_dir is None else str(args.output_dir),
            "stdout": bool(args.stdout),
            "manifest_json": None if args.manifest_json is None else str(args.manifest_json),
            "checkpoint_dir": None if args.checkpoint_dir is None else str(args.checkpoint_dir),
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
            "output_policy": {
                "subtype": None if args.subtype is None else str(args.subtype),
                "bit_depth": str(args.bit_depth),
                "dither": str(args.dither),
                "dither_seed": args.dither_seed,
                "true_peak_max_dbtp": args.true_peak_max_dbtp,
                "metadata_policy": str(args.metadata_policy),
            },
        },
    }


def _manifest_entries(
    args: argparse.Namespace,
    config: Any,
    results: list[JobResult],
    failures: list[tuple[Path, str]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for result in results:
        entries.append(
            {
                "status": "ok",
                "input_path": str(result.input_path),
                "output_path": str(result.output_path),
                "in_sr": int(result.in_sr),
                "out_sr": int(result.out_sr),
                "in_samples": int(result.in_samples),
                "out_samples": int(result.out_samples),
                "channels": int(result.channels),
                "stretch": float(result.stretch),
                "pitch_ratio": float(result.pitch_ratio),
                "stage_count": int(result.stage_count),
                "control_map_segments": int(result.control_map_segments),
                "quality_profile": str(result.quality_profile),
                "checkpoint_id": result.checkpoint_id,
                "transform": str(config.transform),
                "window": str(config.window),
                "phase_engine": str(config.phase_engine),
                "transient_mode": str(args.transient_mode),
                "stereo_mode": str(args.stereo_mode),
                "coherence_strength": float(args.coherence_strength),
                "device": runtime_config().active_device,
                "output_policy": {
                    "subtype": None if args.subtype is None else str(args.subtype),
                    "bit_depth": str(args.bit_depth),
                    "dither": str(args.dither),
                    "dither_seed": args.dither_seed,
                    "true_peak_max_dbtp": args.true_peak_max_dbtp,
                    "metadata_policy": str(args.metadata_policy),
                },
            }
        )
    for path, error_message in failures:
        entries.append(
            {
                "status": "error",
                "input_path": str(path),
                "error": error_message,
                "quality_profile": str(args._active_quality_profile),
            }
        )
    return entries


def _preflight_manifest_target(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    manifest_path = getattr(args, "manifest_json", None)
    if manifest_path is None or bool(getattr(args, "dry_run", False)):
        return
    path = Path(manifest_path)
    append_mode = bool(getattr(args, "manifest_append", False))
    if path.exists() and not append_mode and not bool(getattr(args, "overwrite", False)):
        parser.error(
            f"Manifest exists: {path}. Use --overwrite/--force to replace or append to it."
        )
    if append_mode and path.exists():
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            parser.error(f"Cannot append to invalid manifest: {path} ({exc})")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    cli_flags = collect_cli_flags(argv_list)
    args = parser.parse_args(argv_list)
    args._cli_flags = cli_flags

    if args.example is not None:
        try:
            print_cli_examples(args.example)
        except ValueError as exc:
            parser.error(str(exc))
        return 0

    if args.guided:
        try:
            args = run_guided_mode(args)
        except ValueError as exc:
            parser.error(str(exc))

    if ("--transient-mode" not in cli_flags) and bool(getattr(args, "transient_preserve", False)):
        args.transient_mode = "reset"

    validate_args(args, parser)
    if not args.inputs:
        parser.error(
            "No input files were provided.\n"
            "Hint: run `pvx voc --example basic` for a copy-paste starter command."
        )

    ensure_runtime_dependencies()

    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        parser.error(
            "No readable input files matched the provided paths/patterns.\n"
            "Hint: check the path/glob, or run `pvx voc --guided`."
        )
    stdin_count = sum(1 for path in input_paths if str(path) == "-")
    if stdin_count > 1:
        parser.error("Input '-' (stdin) may only be specified once")
    if stdin_count and len(input_paths) != 1:
        parser.error("Input '-' (stdin) cannot be combined with additional input files")
    if args.stdout and len(input_paths) != 1:
        parser.error("--stdout requires exactly one resolved input")
    if args.stdout and args.output_dir is not None:
        parser.error("--output-dir cannot be used with --stdout")
    if args.output is not None and len(input_paths) != 1:
        parser.error("--output requires exactly one resolved input")
    control_map_stdin = (
        bool(args.pitch_map_stdin)
        or bool(getattr(args, "control_stdin", False))
        or str(args.pitch_map) == "-"
    )
    if control_map_stdin and len(input_paths) != 1:
        parser.error("Control-map stdin mode requires exactly one input file")
    if control_map_stdin and stdin_count:
        parser.error("stdin cannot be used for both audio input and control-map CSV")

    preset_changes = apply_named_preset(
        args,
        preset=str(args.preset),
        provided_flags=cli_flags,
    )

    if args.auto_profile and str(input_paths[0]) == "-":
        parser.error("--auto-profile is not supported when audio input is stdin ('-')")

    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    if args.pitch_map is not None and str(args.pitch_map) != "-":
        args.pitch_map = args.pitch_map.resolve()
    if getattr(args, "_dynamic_control_refs", None):
        resolved_refs: dict[str, DynamicControlRef] = {}
        for key, ref in dict(args._dynamic_control_refs).items():
            resolved_refs[key] = DynamicControlRef(
                parameter=ref.parameter,
                path=ref.path.resolve(),
                value_kind=ref.value_kind,
                interpolation=ref.interpolation,
                order=ref.order,
            )
        args._dynamic_control_refs = resolved_refs
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.resolve()
    if args.manifest_json is not None:
        args.manifest_json = args.manifest_json.resolve()

    _preflight_manifest_target(args, parser)

    auto_features: dict[str, float] | None = None
    active_profile = str(args.quality_profile)
    if args.auto_profile:
        profile_audio, profile_sr = read_audio_input(input_paths[0])
        if profile_audio.size == 0:
            parser.error("Cannot auto-profile an empty input")
        prefetched = dict(getattr(args, "_prefetched_audio_cache", {}) or {})
        prefetched[str(input_paths[0])] = (profile_audio, int(profile_sr))
        args._prefetched_audio_cache = prefetched
        stretch_estimate = resolve_base_stretch(args, profile_audio.shape[0], profile_sr)
        auto_features = estimate_content_features(
            profile_audio,
            profile_sr,
            channel_mode=str(args.analysis_channel),
            lookahead_seconds=float(args.auto_profile_lookahead_seconds),
        )
        active_profile = suggest_quality_profile(
            stretch_ratio=stretch_estimate, features=auto_features
        )

    args._active_quality_profile = active_profile
    profile_changes = apply_quality_profile_overrides(
        args,
        profile=active_profile,
        provided_flags=cli_flags,
    )
    profile_changes = list(preset_changes) + profile_changes

    if args.auto_transform:
        n_fft_for_auto = 2048
        try:
            if not looks_like_control_signal_reference(getattr(args, "n_fft", 2048)):
                n_fft_for_auto = parse_int_cli_value(
                    getattr(args, "n_fft", 2048), context="--n-fft"
                )
        except ValueError:
            n_fft_for_auto = 2048
        resolved_transform = resolve_transform_auto(
            requested_transform=str(args.transform),
            profile=active_profile,
            n_fft=int(n_fft_for_auto),
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

    validate_args(args, parser)
    configure_runtime_from_args(args, parser)

    if console_level(args) >= VERBOSITY_TO_LEVEL["verbose"]:
        info = (
            f"[info] profile={args._active_quality_profile}, "
            f"auto_profile={'on' if args.auto_profile else 'off'}, "
            f"auto_transform={'on' if args.auto_transform else 'off'}, "
            f"transform={args.transform}"
        )
        if profile_changes:
            info += f", overrides={','.join(sorted(set(profile_changes)))}"
        log_message(args, info, min_level="verbose")

    config = build_vocoder_config_from_args(args)

    if args.explain_plan:
        print(
            json.dumps(
                _build_explain_plan(args, input_paths, config, profile_changes, auto_features),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    results: list[JobResult] = []
    failures: list[tuple[Path, str]] = []

    for idx, path in enumerate(input_paths):
        try:
            result = process_file(path, args, config, file_index=idx, file_total=len(input_paths))
            results.append(result)
        except (
            OSError,
            ValueError,
            RuntimeError,
            SystemExit,
        ) as exc:  # pragma: no cover - per-file failures collected for batch reporting
            failures.append((path, str(exc)))

    for result in results:
        in_dur = result.in_samples / result.in_sr
        out_dur = result.out_samples / result.out_sr
        log_message(
            args,
            f"[ok] {result.input_path} -> {result.output_path} | "
            f"ch={result.channels}, sr={result.in_sr}->{result.out_sr}, "
            f"dur={in_dur:.3f}s->{out_dur:.3f}s, "
            f"stretch={result.stretch:.6f}, pitch_ratio={result.pitch_ratio:.6f}, "
            f"profile={result.quality_profile}, stages={result.stage_count}",
            min_level="normal",
        )

    for path, error_message in failures:
        log_error(args, f"[error] {path}: {error_message}")

    if args.manifest_json is not None:
        try:
            write_manifest(
                args.manifest_json,
                _manifest_entries(args, config, results, failures),
                append=bool(args.manifest_append),
            )
        except ValueError as exc:
            log_error(args, f"[error] {exc}")
            return 1

    log_message(
        args,
        f"[done] pvxvoc processed={len(input_paths)} failed={len(failures)}",
        min_level="normal",
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
