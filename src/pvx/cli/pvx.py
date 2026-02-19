#!/usr/bin/env python3
"""Unified top-level CLI for the pvx command suite."""

from __future__ import annotations

import argparse
import difflib
import importlib
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
    "vocal": (
        "Vocal pitch/formant correction",
        "pvx voc vocal.wav --preset vocal_studio --pitch -2 --output vocal_fixed.wav",
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
        "pvx morph source_a.wav source_b.wav --alpha 0.4 --output morph.wav",
    ),
    "pipeline": (
        "Pitch-follow pipeline",
        "pvx pitch-track guide.wav | pvx voc target.wav --pitch-follow-stdin --output followed.wav",
    ),
    "chain": (
        "Managed multi-stage chain",
        "pvx chain input.wav --pipeline \"voc --stretch 1.2 | formant --mode preserve\" --output output_chain.wav",
    ),
    "stream": (
        "Chunked stream wrapper over pvx voc",
        "pvx stream input.wav --output output_stream.wav --chunk-seconds 0.2 --time-stretch 2.0 --preset extreme_ambient",
    ),
}

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
    print("  chain        Run a managed multi-stage one-line tool chain")
    print("  stream       Chunked stream wrapper around `pvx voc`")
    print("  help <tool>  Show subcommand help")
    print("")
    print("Backward compatibility: existing wrappers remain supported (pvxvoc.py, pvxfreeze.py, ...).")


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
    cmd = " ".join([shlex.quote("pvx"), shlex.quote(command)] + [shlex.quote(a) for a in forwarded_args])
    print("")
    print("Generated command:")
    print(cmd)
    print("")


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
        alpha = _prompt_text("Morph alpha (0..1)", "0.50")
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
            "Example: \"voc --stretch 1.2 | formant --mode preserve\""
        ),
    )
    parser.add_argument("--output", "--out", dest="output", required=True, help="Final output path (or '-')")
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
            parser.error(f"Unknown chain stage command '{tokens[0]}' in stage {stage_idx}")
        stage_tool = TOOL_INDEX[stage_cmd].name
        if stage_tool not in _CHAIN_TOOL_ALLOWLIST:
            parser.error(
                f"Chain stage '{stage_tool}' is not supported in managed chain mode. "
                f"Supported: {', '.join(sorted(_CHAIN_TOOL_ALLOWLIST))}"
            )

        stage_flags = {_token_flag(token) for token in tokens[1:] if token.startswith("-")}
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
            print(f"[chain] stage {stage_idx} failed with exit code {code}", file=sys.stderr)
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
    parser.add_argument("--output", "--out", dest="output", required=True, help="Output path (or '-')")
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

    passthrough_flags = {_token_flag(token) for token in passthrough if token.startswith("-")}
    if passthrough_flags & {"--output", "--out", "--stdout"}:
        parser.error("Do not pass --output/--stdout in passthrough args; use `pvx stream --output ...`")

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
            "  pvx chain input.wav --pipeline \"voc --stretch 1.2 | formant --mode preserve\" --output out.wav\n"
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
    helper_commands = {"list", "ls", "tools", "examples", "example", "guided", "guide", "chain", "stream", "help"}

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
    parser.error(f"Unknown command '{command_raw}'.{detail} Run `pvx list` to inspect commands.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
