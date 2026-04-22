#!/usr/bin/env python3

"""Pipeline-oriented helper commands for the unified pvx CLI."""

from __future__ import annotations

import argparse
import contextlib
import io
import shlex
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

from pvx.cli.catalog import (
    _CHAIN_STAGE_FORBIDDEN_FLAGS,
    _CHAIN_TOOL_ALLOWLIST,
    EXAMPLE_COMMANDS,
    FOLLOW_EXAMPLE_CHOICES,
    FOLLOW_EXAMPLE_COMMANDS,
    ToolSpec,
)
from pvx.core.streaming import run_stateful_stream


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


def _split_pipeline_stages(pipeline: str) -> list[str]:
    return [stage.strip() for stage in str(pipeline).split("|") if stage.strip()]


def _token_flag(token: str) -> str:
    return token.split("=", 1)[0]


def _extract_flag_value(args: list[str], flags: tuple[str, ...]) -> str | None:
    value: str | None = None
    i = 0
    while i < len(args):
        token = str(args[i])
        flag = _token_flag(token)
        if flag in flags:
            if "=" in token:
                value = token.split("=", 1)[1]
                i += 1
                continue
            if i + 1 < len(args):
                value = str(args[i + 1])
                i += 2
                continue
            value = ""
            i += 1
            continue
        i += 1
    return value


def _run_stage_command(
    stage_name: str,
    stage_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> int:
    try:
        return int(dispatch_tool(stage_name, stage_args))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return int(code)


def _run_stage_capture_stdout(
    stage_name: str,
    stage_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> tuple[int, str]:
    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            code = int(dispatch_tool(stage_name, stage_args))
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[error] {stage_name}: {exc}", file=sys.stderr)
        code = 1
    return int(code), capture.getvalue()


class _BytesStdin:
    def __init__(self, payload: bytes) -> None:
        self.buffer = io.BytesIO(payload)

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def _patched_stdin_bytes(payload: bytes) -> Iterator[None]:
    original_stdin = sys.stdin
    sys.stdin = _BytesStdin(payload)  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdin = original_stdin


def run_follow_mode(
    forwarded_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> int:
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
    parser.add_argument("--output", "--out", dest="output", required=True, help="Output audio path")
    parser.add_argument(
        "--emit",
        choices=["pitch_map", "stretch_map", "pitch_to_stretch"],
        default="pitch_to_stretch",
        help="Control map emit mode for the guide track (default: pitch_to_stretch)",
    )
    parser.add_argument(
        "--backend", choices=["auto", "pyin", "acf"], default="auto", help="Pitch tracker backend"
    )
    parser.add_argument("--fmin", type=float, default=50.0, help="Minimum tracked f0 in Hz")
    parser.add_argument("--fmax", type=float, default=1200.0, help="Maximum tracked f0 in Hz")
    parser.add_argument(
        "--frame-length", type=int, default=2048, help="Tracker frame length in samples"
    )
    parser.add_argument("--hop-size", type=int, default=256, help="Tracker hop size in samples")
    parser.add_argument(
        "--ratio-reference",
        choices=["median", "mean", "first", "hz"],
        default="median",
        help="Reference mode for pitch_ratio derivation in tracking",
    )
    parser.add_argument(
        "--reference-hz", type=float, default=None, help="Reference Hz when --ratio-reference hz"
    )
    parser.add_argument("--ratio-min", type=float, default=0.25, help="Minimum pitch_ratio clamp")
    parser.add_argument("--ratio-max", type=float, default=4.0, help="Maximum pitch_ratio clamp")
    parser.add_argument("--smooth-frames", type=int, default=5, help="Smoothing window in frames")
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
        "--stretch-scale", type=float, default=1.0, help="Scale factor for derived stretch track"
    )
    parser.add_argument(
        "--stretch-min", type=float, default=0.25, help="Lower clamp for derived stretch"
    )
    parser.add_argument(
        "--stretch-max", type=float, default=4.0, help="Upper clamp for derived stretch"
    )
    parser.add_argument(
        "--stretch", type=float, default=1.0, help="Constant stretch value when --emit pitch_map"
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
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

    passthrough_flags = {_token_flag(token) for token in passthrough if token.startswith("-")}
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

    code, control_csv = _run_stage_capture_stdout(
        "pitch-track",
        track_args,
        dispatch_tool=dispatch_tool,
    )
    if code != 0:
        return int(code)
    if not control_csv.strip():
        print("[error] follow: pitch tracker emitted an empty control map", file=sys.stderr)
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
        return _run_stage_command("voc", voc_args, dispatch_tool=dispatch_tool)


def run_chain_mode(
    forwarded_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
    tool_index: Mapping[str, ToolSpec],
) -> int:
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
        "--output", "--out", dest="output", required=True, help="Final output path (or '-')"
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
        if stage_cmd not in tool_index:
            parser.error(f"Unknown chain stage command '{tokens[0]}' in stage {stage_idx}")
        stage_tool = tool_index[stage_cmd].name
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

    stretch_product = 1.0
    stretch_terms = 0
    for stage_tool, stage_args in stages:
        if stage_tool not in {"voc", "transient"}:
            continue
        stretch_text = _extract_flag_value(
            stage_args, ("--stretch", "--time-stretch", "--time-stretch-factor")
        )
        if stretch_text in {None, ""}:
            continue
        try:
            stretch_val = float(str(stretch_text))
        except ValueError:
            continue
        if stretch_val <= 0.0:
            continue
        stretch_product *= stretch_val
        stretch_terms += 1
    if stretch_terms >= 2 and abs(stretch_product - 1.0) <= 0.06:
        print(
            (
                "[chain] note: cumulative stretch across pipeline is near unity "
                f"({stretch_product:.6f}x). Perceptual change may be subtle."
            ),
            file=sys.stderr,
        )

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
        code = _run_stage_command(stage_tool, command_args, dispatch_tool=dispatch_tool)
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


def run_stream_mode(
    forwarded_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> int:
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

    passthrough_flags = {_token_flag(token) for token in passthrough if token.startswith("-")}
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
    return _run_stage_command("voc", voc_args, dispatch_tool=dispatch_tool)


__all__ = [
    "print_follow_examples",
    "run_chain_mode",
    "run_follow_mode",
    "run_stream_mode",
]
