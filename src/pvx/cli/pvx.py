#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unified top-level CLI for the pvx command suite."""

from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib
import io
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pvx.core.streaming import run_stateful_stream


@dataclass(frozen=True)
class ToolSpec:
    name: str
    entrypoint: str
    summary: str
    aliases: tuple[str, ...] = ()


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="voc",
        entrypoint="pvx.core.voc:main",
        summary="General-purpose phase-vocoder time/pitch processing",
        aliases=("pvxvoc", "vocoder", "timepitch"),
    ),
    ToolSpec(
        name="freeze",
        entrypoint="pvx.cli.pvxfreeze:main",
        summary="Freeze a spectral frame into a sustained texture",
        aliases=("pvxfreeze",),
    ),
    ToolSpec(
        name="harmonize",
        entrypoint="pvx.cli.pvxharmonize:main",
        summary="Generate harmony voices from one source",
        aliases=("pvxharmonize", "harm"),
    ),
    ToolSpec(
        name="conform",
        entrypoint="pvx.cli.pvxconform:main",
        summary="Apply CSV segment map for time/pitch conformity",
        aliases=("pvxconform",),
    ),
    ToolSpec(
        name="morph",
        entrypoint="pvx.cli.pvxmorph:main",
        summary="Morph two sources in the STFT domain",
        aliases=("pvxmorph",),
    ),
    ToolSpec(
        name="warp",
        entrypoint="pvx.cli.pvxwarp:main",
        summary="Apply variable stretch map from CSV",
        aliases=("pvxwarp",),
    ),
    ToolSpec(
        name="formant",
        entrypoint="pvx.cli.pvxformant:main",
        summary="Formant shift/preserve processing",
        aliases=("pvxformant",),
    ),
    ToolSpec(
        name="transient",
        entrypoint="pvx.cli.pvxtransient:main",
        summary="Transient-aware time/pitch processing",
        aliases=("pvxtransient",),
    ),
    ToolSpec(
        name="unison",
        entrypoint="pvx.cli.pvxunison:main",
        summary="Unison thickening and width enhancement",
        aliases=("pvxunison",),
    ),
    ToolSpec(
        name="denoise",
        entrypoint="pvx.cli.pvxdenoise:main",
        summary="Spectral denoise",
        aliases=("pvxdenoise",),
    ),
    ToolSpec(
        name="deverb",
        entrypoint="pvx.cli.pvxdeverb:main",
        summary="Dereverb spectral tail reduction",
        aliases=("pvxdeverb", "dereverb"),
    ),
    ToolSpec(
        name="retune",
        entrypoint="pvx.cli.pvxretune:main",
        summary="Monophonic pitch retune to scale/root",
        aliases=("pvxretune",),
    ),
    ToolSpec(
        name="layer",
        entrypoint="pvx.cli.pvxlayer:main",
        summary="Split/process harmonic and percussive layers",
        aliases=("pvxlayer",),
    ),
    ToolSpec(
        name="pitch-track",
        entrypoint="pvx.cli.hps_pitch_track:main",
        summary="Track f0 and emit control-map CSV",
        aliases=("hps-pitch-track", "hps", "track"),
    ),
)


EXAMPLE_COMMANDS: dict[str, tuple[str, str]] = {
    "basic": ("Basic stretch", "pvx voc input.wav --stretch 1.20 --output output.wav"),
    "speech": (
        "Slow speech for review",
        "pvx voc speech.wav --preset vocal_studio --stretch 1.30 --output speech_slow.wav",
    ),
    "vocal": (
        "Vocal pitch/formant correction",
        "pvx voc vocal.wav --preset vocal_studio --pitch -2 --output vocal_fixed.wav",
    ),
    "retune": (
        "Scale retune",
        "pvx retune vocal.wav --root C --scale major --strength 0.85 --output vocal_retuned.wav",
    ),
    "freeze": (
        "Freeze a spectral moment",
        "pvx freeze hit.wav --freeze-time 0.2 --duration 10 --output hit_freeze.wav",
    ),
    "ambient": (
        "Extreme ambient stretch",
        "pvx voc one_shot.wav --preset extreme_ambient --target-duration 600 --output one_shot_ambient.wav",
    ),
    "drums": (
        "Transient-safe drums",
        "pvx voc drums.wav --preset drums_safe --stretch 1.25 --output drums_safe.wav",
    ),
    "morph": (
        "Source morph",
        "pvx morph source_a.wav source_b.wav --alpha controls/alpha_curve.csv --interp linear --output morph_traj.wav",
    ),
    "map": (
        "Time/pitch map conform",
        "pvx conform source.wav --map map_conform.csv --output source_conformed.wav",
    ),
    "microtonal": (
        "Microtonal pitch ratio",
        "pvx voc input.wav --stretch 1.0 --ratio 3/2 --output input_perfect_fifth.wav",
    ),
    "pipe": (
        "Short one-line pipe",
        "pvx voc input.wav --stretch 1.2 --stdout | pvx deverb - --strength 0.3 --output output.wav",
    ),
    "pipeline": (
        "Pitch-follow pipeline",
        "pvx pitch-track guide.wav --emit pitch_to_stretch --output - | pvx voc target.wav --control-stdin --output followed.wav",
    ),
    "follow": (
        "Single-command sidechain follow",
        "pvx follow guide.wav target.wav --output followed.wav --emit pitch_to_stretch --pitch-conf-min 0.75",
    ),
    "follow-feature": (
        "Feature-driven follow (MFCC + MPEG-7 spectral flux)",
        "pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(mfcc_01,0.002,1.0) --route pitch_ratio=clip(pitch_ratio,0.5,2.0) --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) --route stretch=clip(stretch,0.85,1.6) --output followed_feature.wav",
    ),
    "follow-formant": (
        "Feature-driven follow (formant and onset)",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(formant_f1_hz,0.0016,0.2) --route pitch_ratio=clip(pitch_ratio,0.7,1.5) --route stretch=affine(onset_norm,-0.35,1.2) --route stretch=clip(stretch,0.8,1.3) --output followed_formant_onset.wav",
    ),
    "follow-noise-aware": (
        "Feature-driven follow (noise-aware hiss/hum control)",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route stretch=affine(hiss_ratio,-0.6,1.2) --route stretch=clip(stretch,0.8,1.2) --route pitch_ratio=affine(hum_60_ratio,-0.4,1.15) --route pitch_ratio=clip(pitch_ratio,0.9,1.2) --output followed_noise_aware.wav",
    ),
    "chain": (
        "Managed multi-stage chain",
        'pvx chain input.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output output_chain.wav',
    ),
    "stream": (
        "Chunked stream wrapper over pvx voc",
        "pvx stream input.wav --output output_stream.wav --chunk-seconds 0.2 --time-stretch 2.0 --preset extreme_ambient",
    ),
}


FOLLOW_EXAMPLE_COMMANDS: dict[str, tuple[str, str]] = {
    "basic": (
        "Pitch-to-stretch sidechain",
        "pvx follow guide.wav target.wav --emit pitch_to_stretch --pitch-conf-min 0.75 --output followed.wav",
    ),
    "pitch": (
        "Pitch-map follow with fixed stretch",
        "pvx follow guide.wav target.wav --emit pitch_map --stretch 1.0 --output followed_pitch.wav",
    ),
    "mfcc_flux": (
        "MFCC + MPEG-7 flux dual control",
        "pvx follow guide.wav target.wav --feature-set all --mfcc-count 13 --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(mfcc_01,0.002,1.0) --route pitch_ratio=clip(pitch_ratio,0.5,2.0) --route stretch=affine(mpeg7_spectral_flux,0.05,1.0) --route stretch=clip(stretch,0.85,1.6) --output followed_mfcc_flux.wav",
    ),
    "formant_onset": (
        "Formant + onset dual control",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route pitch_ratio=affine(formant_f1_hz,0.0016,0.2) --route pitch_ratio=clip(pitch_ratio,0.7,1.5) --route stretch=affine(onset_norm,-0.35,1.2) --route stretch=clip(stretch,0.8,1.3) --output followed_formant_onset.wav",
    ),
    "noise_aware": (
        "Noise-aware hiss/hum routing",
        "pvx follow guide.wav target.wav --feature-set all --emit pitch_map --stretch 1.0 --route stretch=affine(hiss_ratio,-0.6,1.2) --route stretch=clip(stretch,0.8,1.2) --route pitch_ratio=affine(hum_60_ratio,-0.4,1.15) --route pitch_ratio=clip(pitch_ratio,0.9,1.2) --output followed_noise_aware.wav",
    ),
}

FOLLOW_EXAMPLE_CHOICES: tuple[str, ...] = (
    "all",
    *tuple(FOLLOW_EXAMPLE_COMMANDS.keys()),
)

_AUDIO_EXTENSIONS: set[str] = {
    ".wav",
    ".flac",
    ".aiff",
    ".aif",
    ".ogg",
    ".oga",
    ".caf",
    ".mp3",
    ".m4a",
    ".aac",
    ".wma",
}

_CHAIN_TOOL_ALLOWLIST: set[str] = {
    "voc",
    "freeze",
    "harmonize",
    "conform",
    "warp",
    "formant",
    "transient",
    "unison",
    "denoise",
    "deverb",
    "retune",
    "layer",
}
_CHAIN_STAGE_FORBIDDEN_FLAGS: set[str] = {
    "-o",
    "--out",
    "--output",
    "--output-dir",
    "--stdout",
}


def _tool_index() -> dict[str, ToolSpec]:
    out: dict[str, ToolSpec] = {}
    for spec in TOOL_SPECS:
        out[spec.name] = spec
        for alias in spec.aliases:
            out[alias] = spec
    return out


TOOL_INDEX = _tool_index()


def _load_entrypoint(entrypoint: str) -> Callable[[list[str] | None], int]:
    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    return fn


def _looks_like_audio_input(token: str) -> bool:
    if token == "-":
        return True
    if any(ch in token for ch in "*?["):
        return True
    path = Path(token)
    if path.suffix.lower() in _AUDIO_EXTENSIONS:
        return True
    return path.exists()


def _tool_names_csv() -> str:
    return ", ".join(spec.name for spec in TOOL_SPECS)


def print_tools() -> None:
    print("pvx command list")
    print("")
    print("Primary subcommands:")
    for spec in TOOL_SPECS:
        aliases = ""
        if spec.aliases:
            aliases = f" [aliases: {', '.join(spec.aliases)}]"
        print(f"  {spec.name:<12} {spec.summary}{aliases}")
    print("")
    print("Helper commands:")
    print("  list         Show this command table")
    print("  examples     Show copy-paste examples (use `pvx examples <name>`)")
    print("  guided       Interactive command builder")
    print("  follow       Track one file and control another in one command")
    print("  chain        Run a managed multi-stage one-line tool chain")
    print("  stream       Chunked stream wrapper around `pvx voc`")
    print("  help <tool>  Show subcommand help")
    print("")
    print(
        "Backward compatibility: existing wrappers remain supported (pvxvoc.py, pvxfreeze.py, ...)."
    )


def print_examples(which: str = "all") -> None:
    key = str(which).strip().lower()
    if key == "all":
        print("pvx example commands")
        print("")
        for name, (title, command) in EXAMPLE_COMMANDS.items():
            print(f"[{name}] {title}")
            print(command)
            print("")
        return
    if key not in EXAMPLE_COMMANDS:
        raise ValueError(
            f"Unknown example '{which}'. Use one of: {', '.join(sorted(EXAMPLE_COMMANDS))}, all"
        )
    title, command = EXAMPLE_COMMANDS[key]
    print(f"[{key}] {title}")
    print(command)


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else default


def _prompt_choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    value = _prompt_text(prompt, default).strip().lower()
    if value not in choices:
        raise ValueError(f"Expected one of: {', '.join(choices)}")
    return value


def _print_command_preview(command: str, forwarded_args: list[str]) -> None:
    cmd = " ".join(
        [shlex.quote("pvx"), shlex.quote(command)]
        + [shlex.quote(a) for a in forwarded_args]
    )
    print("")
    print("Generated command:")
    print(cmd)
    print("")


def print_follow_examples(which: str = "basic") -> None:
    key = str(which).strip().lower()
    if key == "all":
        print("pvx follow example commands")
        print("")
        for name, (title, command) in FOLLOW_EXAMPLE_COMMANDS.items():
            print(f"[{name}] {title}")
            print(command)
            print("")
        return
    if key not in FOLLOW_EXAMPLE_COMMANDS:
        raise ValueError(
            f"Unknown follow example '{which}'. Use one of: {', '.join(FOLLOW_EXAMPLE_CHOICES)}"
        )
    title, command = FOLLOW_EXAMPLE_COMMANDS[key]
    print(f"[{key}] {title}")
    print(command)


def _extract_follow_example_request(args: list[str]) -> str | None:
    tokens = [str(token).strip() for token in list(args or [])]
    for idx, token in enumerate(tokens):
        if token == "--example":
            if idx + 1 < len(tokens):
                candidate = tokens[idx + 1]
                if candidate and not candidate.startswith("-"):
                    return candidate
            return "basic"
        if token.startswith("--example="):
            candidate = token.split("=", 1)[1].strip()
            return candidate or "basic"
    return None


def run_guided_mode() -> int:
    if not sys.stdin.isatty():
        raise ValueError("`pvx guided` requires an interactive terminal (TTY stdin)")

    print("pvx guided mode")
    print("Press Enter to accept defaults.\n")

    mode = _prompt_choice(
        "Workflow (voc/freeze/harmonize/retune/morph)",
        ("voc", "freeze", "harmonize", "retune", "morph"),
        "voc",
    )

    if mode == "voc":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output.wav")
        stretch = _prompt_text("Stretch factor", "1.20")
        semitones = _prompt_text("Pitch shift semitones", "0")
        preset = _prompt_text("Preset", "default")
        forwarded = [
            input_path,
            "--stretch",
            stretch,
            "--pitch",
            semitones,
            "--preset",
            preset,
            "--output",
            output_path,
        ]
    elif mode == "freeze":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output_freeze.wav")
        freeze_time = _prompt_text("Freeze time (seconds)", "0.25")
        duration = _prompt_text("Output duration (seconds)", "8.0")
        forwarded = [
            input_path,
            "--freeze-time",
            freeze_time,
            "--duration",
            duration,
            "--output",
            output_path,
        ]
    elif mode == "harmonize":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output_harm.wav")
        intervals = _prompt_text("Intervals (semitones CSV)", "0,4,7")
        forwarded = [
            input_path,
            "--intervals",
            intervals,
            "--output",
            output_path,
        ]
    elif mode == "retune":
        input_path = _prompt_text("Input path", "input.wav")
        output_path = _prompt_text("Output path", "output_retune.wav")
        root = _prompt_text("Root note", "C")
        scale = _prompt_text("Scale", "major")
        strength = _prompt_text("Correction strength", "0.85")
        forwarded = [
            input_path,
            "--root",
            root,
            "--scale",
            scale,
            "--strength",
            strength,
            "--output",
            output_path,
        ]
    else:
        input_a = _prompt_text("Input A path", "a.wav")
        input_b = _prompt_text("Input B path", "b.wav")
        output_path = _prompt_text("Output path", "morph.wav")
        alpha = _prompt_text("Morph alpha (0..1 or CSV/JSON control file)", "0.50")
        forwarded = [
            input_a,
            input_b,
            "--alpha",
            alpha,
            "--output",
            output_path,
        ]

    _print_command_preview(mode, forwarded)
    run_now = _prompt_choice("Run now? (yes/no)", ("yes", "no"), "yes")
    if run_now == "no":
        print("Command preview only; no processing executed.")
        return 0
    return dispatch_tool(mode, forwarded)


def _split_pipeline_stages(pipeline: str) -> list[str]:
    return [stage.strip() for stage in str(pipeline).split("|") if stage.strip()]


def _token_flag(token: str) -> str:
    return token.split("=", 1)[0]


def _run_stage_command(stage_name: str, stage_args: list[str]) -> int:
    try:
        return int(dispatch_tool(stage_name, stage_args))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return int(code)


def _run_stage_capture_stdout(
    stage_name: str, stage_args: list[str]
) -> tuple[int, str]:
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            code = int(dispatch_tool(stage_name, stage_args))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        print(f"[error] {stage_name}: {exc}", file=sys.stderr)
        code = 1
    return int(code), capture.getvalue()


class _BytesStdin:
    def __init__(self, payload: bytes) -> None:
        self.buffer = io.BytesIO(payload)

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def _patched_stdin_bytes(payload: bytes):
    original_stdin = sys.stdin
    sys.stdin = _BytesStdin(payload)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdin = original_stdin


def run_follow_mode(forwarded_args: list[str]) -> int:
    example_request = _extract_follow_example_request(forwarded_args)
    if example_request is not None:
        try:
            print_follow_examples(example_request)
            return 0
        except ValueError as exc:
            print(f"pvx follow: error: {exc}", file=sys.stderr)
            return 2

    parser = argparse.ArgumentParser(
        prog="pvx follow",
        description=(
            "Single-command sidechain helper: track guide pitch/f0 and apply the resulting control map "
            "to a target via `pvx voc --control-stdin`."
        ),
    )
    parser.add_argument("guide", help="Guide/input A used for pitch tracking")
    parser.add_argument("target", help="Target/input B to be processed by pvx voc")
    parser.add_argument(
        "--output", "--out", dest="output", required=True, help="Output audio path"
    )
    parser.add_argument(
        "--emit",
        choices=["pitch_map", "stretch_map", "pitch_to_stretch"],
        default="pitch_to_stretch",
        help="Control map emit mode for the guide track (default: pitch_to_stretch)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pyin", "acf"],
        default="auto",
        help="Pitch tracker backend",
    )
    parser.add_argument(
        "--fmin", type=float, default=50.0, help="Minimum tracked f0 in Hz"
    )
    parser.add_argument(
        "--fmax", type=float, default=1200.0, help="Maximum tracked f0 in Hz"
    )
    parser.add_argument(
        "--frame-length", type=int, default=2048, help="Tracker frame length in samples"
    )
    parser.add_argument(
        "--hop-size", type=int, default=256, help="Tracker hop size in samples"
    )
    parser.add_argument(
        "--ratio-reference",
        choices=["median", "mean", "first", "hz"],
        default="median",
        help="Reference mode for pitch_ratio derivation in tracking",
    )
    parser.add_argument(
        "--reference-hz",
        type=float,
        default=None,
        help="Reference Hz when --ratio-reference hz",
    )
    parser.add_argument(
        "--ratio-min", type=float, default=0.25, help="Minimum pitch_ratio clamp"
    )
    parser.add_argument(
        "--ratio-max", type=float, default=4.0, help="Maximum pitch_ratio clamp"
    )
    parser.add_argument(
        "--smooth-frames", type=int, default=5, help="Smoothing window in frames"
    )
    parser.add_argument(
        "--confidence-floor", type=float, default=0.0, help="Minimum tracker confidence"
    )
    parser.add_argument(
        "--feature-set",
        choices=["none", "basic", "advanced", "all"],
        default="all",
        help="Feature columns emitted by pitch tracker (default: all)",
    )
    parser.add_argument(
        "--mfcc-count",
        type=int,
        default=13,
        help="MFCC column count emitted by pitch tracker (default: 13)",
    )
    parser.add_argument(
        "--stretch-from",
        choices=["pitch_ratio", "inv_pitch_ratio", "f0_hz"],
        default="pitch_ratio",
        help="Source for deriving stretch in stretch-oriented emit modes",
    )
    parser.add_argument(
        "--stretch-scale",
        type=float,
        default=1.0,
        help="Scale factor for derived stretch track",
    )
    parser.add_argument(
        "--stretch-min",
        type=float,
        default=0.25,
        help="Lower clamp for derived stretch",
    )
    parser.add_argument(
        "--stretch-max", type=float, default=4.0, help="Upper clamp for derived stretch"
    )
    parser.add_argument(
        "--stretch",
        type=float,
        default=1.0,
        help="Constant stretch value when --emit pitch_map",
    )
    parser.add_argument(
        "--pitch-conf-min",
        type=float,
        default=0.75,
        help="Minimum accepted map confidence for pvx voc (default: 0.75)",
    )
    parser.add_argument(
        "--pitch-lowconf-mode",
        choices=["hold", "unity", "interp"],
        default="hold",
        help="Low-confidence handling mode in pvx voc (default: hold)",
    )
    parser.add_argument(
        "--pitch-map-smooth-ms",
        type=float,
        default=0.0,
        help="Additional map smoothing in pvx voc (milliseconds)",
    )
    parser.add_argument(
        "--pitch-map-crossfade-ms",
        type=float,
        default=20.0,
        help="Map segment crossfade in pvx voc (milliseconds, default: 20)",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="EXPR",
        help=(
            "Optional pvx voc control route expression. Repeat to chain. "
            "Example: --route stretch=pitch_ratio --route pitch_ratio=const(1.0)"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce helper logs and hide progress bars"
    )
    parser.add_argument("--silent", action="store_true", help="Suppress helper logs")
    parser.add_argument(
        "--example",
        nargs="?",
        const="basic",
        default=None,
        choices=list(FOLLOW_EXAMPLE_CHOICES),
        metavar="NAME",
        help=(
            "Print follow example command(s) and exit. "
            "Use `--example` for basic or `--example all` for the full set."
        ),
    )
    args, passthrough = parser.parse_known_args(forwarded_args)
    if args.example is not None:
        print_follow_examples(str(args.example))
        return 0

    passthrough_flags = {
        _token_flag(token) for token in passthrough if token.startswith("-")
    }
    forbidden_passthrough = {
        "--output",
        "--out",
        "--stdout",
        "--pitch-map",
        "--pitch-map-stdin",
        "--control-stdin",
    }
    bad_flags = sorted(passthrough_flags & forbidden_passthrough)
    if bad_flags:
        parser.error(
            f"Do not pass {bad_flags} via passthrough in `pvx follow`; "
            "follow mode manages control-map and output routing."
        )
    if int(args.mfcc_count) < 0 or int(args.mfcc_count) > 40:
        parser.error("--mfcc-count must be in [0, 40]")

    track_args: list[str] = [
        str(args.guide),
        "--output",
        "-",
        "--emit",
        str(args.emit),
        "--backend",
        str(args.backend),
        "--fmin",
        f"{float(args.fmin):.12g}",
        "--fmax",
        f"{float(args.fmax):.12g}",
        "--frame-length",
        str(int(args.frame_length)),
        "--hop-size",
        str(int(args.hop_size)),
        "--ratio-reference",
        str(args.ratio_reference),
        "--ratio-min",
        f"{float(args.ratio_min):.12g}",
        "--ratio-max",
        f"{float(args.ratio_max):.12g}",
        "--smooth-frames",
        str(int(args.smooth_frames)),
        "--confidence-floor",
        f"{float(args.confidence_floor):.12g}",
        "--feature-set",
        str(args.feature_set),
        "--mfcc-count",
        str(int(args.mfcc_count)),
        "--stretch-from",
        str(args.stretch_from),
        "--stretch-scale",
        f"{float(args.stretch_scale):.12g}",
        "--stretch-min",
        f"{float(args.stretch_min):.12g}",
        "--stretch-max",
        f"{float(args.stretch_max):.12g}",
        "--stretch",
        f"{float(args.stretch):.12g}",
    ]
    if args.reference_hz is not None:
        track_args.extend(["--reference-hz", f"{float(args.reference_hz):.12g}"])
    if args.quiet:
        track_args.append("--quiet")
    if args.silent:
        track_args.append("--silent")

    code, control_csv = _run_stage_capture_stdout("pitch-track", track_args)
    if code != 0:
        return int(code)
    if not control_csv.strip():
        print(
            "[error] follow: pitch tracker emitted an empty control map",
            file=sys.stderr,
        )
        return 1

    voc_args: list[str] = [
        str(args.target),
        "--control-stdin",
        "--pitch-conf-min",
        f"{float(args.pitch_conf_min):.12g}",
        "--pitch-lowconf-mode",
        str(args.pitch_lowconf_mode),
        "--pitch-map-smooth-ms",
        f"{float(args.pitch_map_smooth_ms):.12g}",
        "--pitch-map-crossfade-ms",
        f"{float(args.pitch_map_crossfade_ms):.12g}",
        "--output",
        str(args.output),
    ]
    for route in list(args.route or []):
        voc_args.extend(["--route", str(route)])
    if args.overwrite:
        voc_args.append("--overwrite")
    if args.quiet:
        voc_args.append("--quiet")
    if args.silent:
        voc_args.append("--silent")
    voc_args.extend(passthrough)

    payload = control_csv.encode("utf-8")
    with _patched_stdin_bytes(payload):
        return _run_stage_command("voc", voc_args)


def run_chain_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx chain",
        description=(
            "Managed one-line chain runner for serial pvx audio tools. "
            "Each stage receives the previous stage output as input."
        ),
    )
    parser.add_argument("input", help="Initial input audio path or '-' for stdin")
    parser.add_argument(
        "--pipeline",
        required=True,
        help=(
            "Pipeline string with stages separated by '|'. "
            'Example: "voc --stretch 1.2 | formant --mode preserve"'
        ),
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        required=True,
        help="Final output path (or '-')",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Optional directory for intermediate stage files",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate stage files after successful completion",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Print a copy-paste chain example and exit",
    )
    args = parser.parse_args(forwarded_args)

    if args.example:
        print(EXAMPLE_COMMANDS["chain"][1])
        return 0

    raw_stages = _split_pipeline_stages(args.pipeline)
    if not raw_stages:
        parser.error("--pipeline produced no stages")

    stages: list[tuple[str, list[str]]] = []
    for stage_idx, stage_text in enumerate(raw_stages, start=1):
        try:
            tokens = shlex.split(stage_text)
        except ValueError as exc:
            parser.error(f"Invalid stage {stage_idx} syntax: {exc}")
        if not tokens:
            parser.error(f"Stage {stage_idx} is empty")

        stage_cmd = tokens[0].strip().lower()
        if stage_cmd not in TOOL_INDEX:
            parser.error(
                f"Unknown chain stage command '{tokens[0]}' in stage {stage_idx}"
            )
        stage_tool = TOOL_INDEX[stage_cmd].name
        if stage_tool not in _CHAIN_TOOL_ALLOWLIST:
            parser.error(
                f"Chain stage '{stage_tool}' is not supported in managed chain mode. "
                f"Supported: {', '.join(sorted(_CHAIN_TOOL_ALLOWLIST))}"
            )

        stage_flags = {
            _token_flag(token) for token in tokens[1:] if token.startswith("-")
        }
        bad_flags = sorted(stage_flags & _CHAIN_STAGE_FORBIDDEN_FLAGS)
        if bad_flags:
            parser.error(
                f"Stage {stage_idx} ({stage_tool}) contains output-routing flags {bad_flags}. "
                "Managed chain mode controls stage outputs automatically."
            )
        stages.append((stage_tool, tokens[1:]))

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir is None:
        if args.keep_intermediate:
            work_dir = Path(tempfile.mkdtemp(prefix="pvx-chain-"))
        else:
            temp_ctx = tempfile.TemporaryDirectory(prefix="pvx-chain-")
            work_dir = Path(temp_ctx.name)
    else:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

    current_input = str(args.input)
    for stage_idx, (stage_tool, stage_args) in enumerate(stages, start=1):
        is_last = stage_idx == len(stages)
        if is_last:
            stage_out = Path(str(args.output))
        else:
            stage_out = work_dir / f"stage_{stage_idx:02d}_{stage_tool}.wav"

        command_args = [
            current_input,
            *stage_args,
            "--output",
            str(stage_out),
            "--overwrite",
            "--quiet",
        ]
        print(f"[chain] stage {stage_idx}/{len(stages)}: {stage_tool}")
        code = _run_stage_command(stage_tool, command_args)
        if code != 0:
            print(
                f"[chain] stage {stage_idx} failed with exit code {code}",
                file=sys.stderr,
            )
            return int(code)

        current_input = str(stage_out)

    if temp_ctx is not None:
        temp_ctx.cleanup()
    elif args.keep_intermediate:
        print(f"[chain] intermediates kept in {work_dir}")

    print(f"[chain] done -> {args.output}")
    return 0


def run_stream_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx stream",
        description=(
            "Chunked streaming wrapper over `pvx voc` for long renders and pipe-friendly one-liners."
        ),
    )
    parser.add_argument("input", help="Input audio path or '-' for stdin")
    parser.add_argument(
        "--output", "--out", dest="output", required=True, help="Output path (or '-')"
    )
    parser.add_argument(
        "--mode",
        choices=["stateful", "wrapper"],
        default="stateful",
        help="Stream engine: stateful chunk processor (default) or wrapper compatibility mode",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.25,
        help="Chunk/segment duration for `--auto-segment-seconds` (default: 0.25)",
    )
    parser.add_argument(
        "--crossfade-ms",
        type=float,
        default=0.0,
        help="Crossfade used for segment assembly in milliseconds (default: 0.0)",
    )
    parser.add_argument(
        "--context-ms",
        type=float,
        default=None,
        help="Optional stateful context window in milliseconds (default: auto from window/hop)",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Print a copy-paste stream example and exit",
    )
    args, passthrough = parser.parse_known_args(forwarded_args)

    if args.example:
        print(EXAMPLE_COMMANDS["stream"][1])
        return 0

    if args.chunk_seconds <= 0.0:
        parser.error("--chunk-seconds must be > 0")
    if args.crossfade_ms < 0.0:
        parser.error("--crossfade-ms must be >= 0")
    if args.context_ms is not None and float(args.context_ms) < 0.0:
        parser.error("--context-ms must be >= 0")

    passthrough_flags = {
        _token_flag(token) for token in passthrough if token.startswith("-")
    }
    if passthrough_flags & {"--output", "--out", "--stdout"}:
        parser.error(
            "Do not pass --output/--stdout in passthrough args; use `pvx stream --output ...`"
        )

    if args.mode == "stateful":
        return run_stateful_stream(
            input_token=str(args.input),
            output_token=str(args.output),
            passthrough=list(passthrough),
            chunk_seconds=float(args.chunk_seconds),
            context_ms=None if args.context_ms is None else float(args.context_ms),
            crossfade_ms=float(args.crossfade_ms),
        )

    voc_args: list[str] = [str(args.input)]
    if "--auto-segment-seconds" not in passthrough_flags:
        voc_args.extend(["--auto-segment-seconds", f"{float(args.chunk_seconds):.6g}"])
    if "--pitch-map-crossfade-ms" not in passthrough_flags:
        voc_args.extend(["--pitch-map-crossfade-ms", f"{float(args.crossfade_ms):.6g}"])

    if str(args.output) == "-":
        voc_args.append("--stdout")
    else:
        voc_args.extend(["--output", str(args.output)])

    voc_args.extend(passthrough)
    return _run_stage_command("voc", voc_args)


def dispatch_tool(command: str, forwarded_args: list[str]) -> int:
    spec = TOOL_INDEX.get(command)
    if spec is None:
        raise ValueError(f"Unknown tool command: {command}")
    main_fn = _load_entrypoint(spec.entrypoint)
    return int(main_fn(forwarded_args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pvx",
        description=(
            "Unified CLI for pvx (audio quality first, speed second).\n"
            "Use subcommands to access all existing pvx tools from one entrypoint."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Quick start:\n"
            "  pvx voc input.wav --stretch 1.2 --output output.wav\n"
            "  pvx input.wav --stretch 1.2 --output output.wav   # defaults to `voc`\n"
            "  pvx follow guide.wav target.wav --output followed.wav --emit pitch_to_stretch\n"
            '  pvx chain input.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output out.wav\n'
            "  pvx stream input.wav --output out.wav --chunk-seconds 0.2 --time-stretch 2.0\n"
            "  pvx list\n"
            "  pvx examples basic\n"
            "  pvx help voc\n"
            "\n"
            f"Available tool commands: {_tool_names_csv()}"
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Subcommand name, helper command, or input path (defaults to `voc` when an input path is provided)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded directly to the selected subcommand",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    command_raw = args.command
    forwarded = list(args.args or [])

    if command_raw is None:
        parser.print_help()
        print("")
        print("Tip: run `pvx list` for tool descriptions.")
        return 0

    command = str(command_raw).strip().lower()
    helper_commands = {
        "list",
        "ls",
        "tools",
        "examples",
        "example",
        "guided",
        "guide",
        "follow",
        "chain",
        "stream",
        "help",
    }

    if command in {"list", "ls", "tools"}:
        print_tools()
        return 0
    if command in {"examples", "example"}:
        which = forwarded[0] if forwarded else "all"
        try:
            print_examples(which)
        except ValueError as exc:
            parser.error(str(exc))
        return 0
    if command in {"guided", "guide"}:
        try:
            return run_guided_mode()
        except ValueError as exc:
            parser.error(str(exc))
    if command == "follow":
        try:
            return run_follow_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "chain":
        try:
            return run_chain_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "stream":
        try:
            return run_stream_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "help":
        if not forwarded:
            parser.print_help()
            return 0
        target = str(forwarded[0]).strip().lower()
        if target in helper_commands:
            if target in {"examples", "example"}:
                print_examples("all")
                return 0
            if target in {"list", "ls", "tools"}:
                print_tools()
                return 0
            if target in {"guided", "guide"}:
                print("Run `pvx guided` from an interactive terminal.")
                return 0
            if target == "follow":
                print(
                    "Run `pvx follow --help` for one-command sidechain control mapping."
                )
                return 0
            if target == "chain":
                print("Run `pvx chain --help` for managed one-line tool chaining.")
                return 0
            if target == "stream":
                print("Run `pvx stream --help` for chunked streaming wrapper options.")
                return 0
            parser.print_help()
            return 0
        if target in TOOL_INDEX:
            return dispatch_tool(target, ["--help"])
        parser.error(f"Unknown help target '{forwarded[0]}'. Use `pvx list`.")

    if command in TOOL_INDEX:
        return dispatch_tool(command, forwarded)

    # Beginner shortcut: if first token looks like an input path or glob, treat as `pvx voc ...`.
    if _looks_like_audio_input(command_raw):
        return dispatch_tool("voc", [command_raw] + forwarded)

    candidates = sorted(
        set(
            [spec.name for spec in TOOL_SPECS]
            + [alias for spec in TOOL_SPECS for alias in spec.aliases]
            + list(helper_commands)
        )
    )
    suggestions = difflib.get_close_matches(command, candidates, n=3, cutoff=0.45)
    detail = ""
    if suggestions:
        detail = f" Did you mean: {', '.join(suggestions)}?"
    parser.error(
        f"Unknown command '{command_raw}'.{detail} Run `pvx list` to inspect commands."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
