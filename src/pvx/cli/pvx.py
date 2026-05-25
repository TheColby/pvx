#!/usr/bin/env python3

"""Unified top-level CLI for the pvx command suite."""

from __future__ import annotations

import argparse
import difflib
import importlib
import random
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from pvx.cli.catalog import (
    _AUDIO_EXTENSIONS,
    EXAMPLE_COMMANDS,
    TOOL_SPECS,
    build_tool_index,
)
from pvx.cli.pvx_augment import (
    run_augment_manifest_mode as _run_augment_manifest_mode,
)
from pvx.cli.pvx_augment import (
    run_augment_mode as _run_augment_mode,
)
from pvx.cli.pvx_augment import (
    run_batch_gpu_mode as _run_batch_gpu_mode,
)
from pvx.cli.pvx_helpers import (
    _consume_lucky_options as helpers_consume_lucky_options,
)
from pvx.cli.pvx_helpers import (
    _infer_output_format as helpers_infer_output_format,
)
from pvx.cli.pvx_helpers import (
    _parse_size_bytes as helpers_parse_size_bytes,
)
from pvx.cli.pvx_helpers import (
    _run_lucky_helper_mode as helpers_run_lucky_helper_mode,
)
from pvx.cli.pvx_helpers import (
    _run_lucky_tool_mode as helpers_run_lucky_tool_mode,
)
from pvx.cli.pvx_helpers import (
    run_doctor_mode as helpers_run_doctor_mode,
)
from pvx.cli.pvx_helpers import (
    run_guided_mode as helpers_run_guided_mode,
)
from pvx.cli.pvx_helpers import (
    run_quickstart_mode as helpers_run_quickstart_mode,
)
from pvx.cli.pvx_helpers import (
    run_safe_mode as helpers_run_safe_mode,
)
from pvx.cli.pvx_helpers import (
    run_smoke_mode as helpers_run_smoke_mode,
)
from pvx.cli.pvx_helpers import (
    run_stretch_budget_mode as helpers_run_stretch_budget_mode,
)
from pvx.cli.pvx_helpers import (
    run_transforms_mode as helpers_run_transforms_mode,
)
from pvx.cli.pvx_pipeline import (
    print_follow_examples as _print_follow_examples,
)
from pvx.cli.pvx_pipeline import (
    run_chain_mode as _run_chain_mode,
)
from pvx.cli.pvx_pipeline import (
    run_follow_mode as _run_follow_mode,
)
from pvx.cli.pvx_pipeline import (
    run_stream_mode as _run_stream_mode,
)

TOOL_INDEX = build_tool_index(TOOL_SPECS)


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
    print("  quickstart   Print a minimal launch sequence")
    print("  doctor       Run environment diagnostics and suggested fixes")
    print("  transforms   Show transform choices and recommendations")
    print("  safe         Run `pvx voc` with conservative quality-first defaults")
    print("  smoke        Fast synthetic end-to-end smoke render")
    print("  augment      Deterministic AI dataset augmentation with manifests")
    print("  augment-manifest  Validate/merge augmentation manifest files")
    print("  guided       Interactive command builder")
    print("  follow       Track one file and control another in one command")
    print("  chain        Run a managed multi-stage one-line tool chain")
    print("  stream       Chunked stream wrapper around `pvx voc`")
    print("  stretch-budget  Estimate max safe stretch from file size/budget assumptions")
    print("  help <tool>  Show subcommand help")
    print("")
    print("Global randomizer:")
    print(
        "  --lucky N [--lucky-seed S]  Run selected workflow N times with randomized DSP settings"
    )
    print("")
    print(
        "Use installed commands (`pvx`, `pvxvoc`, `pvxfreeze`, ...) or `python -m pvx...` module entry points."
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
        [shlex.quote("pvx"), shlex.quote(command)] + [shlex.quote(a) for a in forwarded_args]
    )
    print("")
    print("Generated command:")
    print(cmd)
    print("")


def print_follow_examples(which: str = "basic") -> None:
    _print_follow_examples(which)


def run_doctor_mode(forwarded_args: list[str]) -> int:
    return helpers_run_doctor_mode(forwarded_args)


def run_quickstart_mode(forwarded_args: list[str]) -> int:
    return helpers_run_quickstart_mode(forwarded_args)


def run_safe_mode(forwarded_args: list[str]) -> int:
    return helpers_run_safe_mode(forwarded_args, dispatch_tool=dispatch_tool)


def run_transforms_mode(forwarded_args: list[str]) -> int:
    return helpers_run_transforms_mode(forwarded_args)


def run_smoke_mode(forwarded_args: list[str]) -> int:
    return helpers_run_smoke_mode(forwarded_args, dispatch_tool=dispatch_tool)


def _parse_split_ratios(text: str) -> tuple[float, float, float]:
    from pvx.cli.pvx_augment import _parse_split_ratios as augment_parse_split_ratios

    return augment_parse_split_ratios(text)


def _stable_seed_from_text(base_seed: int, text: str) -> int:
    from pvx.cli.pvx_augment import _stable_seed_from_text as augment_stable_seed_from_text

    return augment_stable_seed_from_text(base_seed, text)


def _augment_group_key(path: Path, grouping: str, separator: str) -> str:
    from pvx.cli.pvx_augment import _augment_group_key as augment_group_key

    return augment_group_key(path, grouping, separator)


def run_batch_gpu_mode(forwarded_args: list[str]) -> int:
    return _run_batch_gpu_mode(forwarded_args)


def run_augment_mode(forwarded_args: list[str]) -> int:
    return _run_augment_mode(forwarded_args, dispatch_tool=dispatch_tool)


def run_augment_manifest_mode(forwarded_args: list[str]) -> int:
    return _run_augment_manifest_mode(forwarded_args)


def run_guided_mode() -> int:
    return helpers_run_guided_mode(
        dispatch_tool=dispatch_tool,
        prompt_text=_prompt_text,
        prompt_choice=_prompt_choice,
        print_command_preview=_print_command_preview,
    )


def _token_flag(token: str) -> str:
    from pvx.cli.pvx_helpers import _token_flag as helpers_token_flag

    return helpers_token_flag(token)


def _extract_flag_value(args: list[str], flags: tuple[str, ...]) -> str | None:
    from pvx.cli.pvx_helpers import _extract_flag_value as helpers_extract_flag_value

    return helpers_extract_flag_value(args, flags)


def _strip_flags(args: list[str], flag_has_value: dict[str, bool]) -> list[str]:
    from pvx.cli.pvx_helpers import _strip_flags as helpers_strip_flags

    return helpers_strip_flags(args, flag_has_value)


def _replace_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    from pvx.cli.pvx_helpers import _replace_flag_value as helpers_replace_flag_value

    return helpers_replace_flag_value(args, flag, value)


def _consume_lucky_options(args: list[str]) -> tuple[list[str], int | None, int | None]:
    return helpers_consume_lucky_options(args)


def _lucky_output_variant(base: Path, idx: int) -> Path:
    from pvx.cli.pvx_helpers import _lucky_output_variant as helpers_lucky_output_variant

    return helpers_lucky_output_variant(base, idx)


def _lucky_mastering_overrides(rng: random.Random) -> list[str]:
    from pvx.cli.pvx_helpers import _lucky_mastering_overrides as helpers_lucky_mastering_overrides

    return helpers_lucky_mastering_overrides(rng)


def _lucky_tool_overrides(tool: str, rng: random.Random) -> list[str]:
    from pvx.cli.pvx_helpers import _lucky_tool_overrides as helpers_lucky_tool_overrides

    return helpers_lucky_tool_overrides(tool, rng)


def _run_lucky_tool_mode(
    tool: str, forwarded_args: list[str], lucky_count: int, lucky_seed: int | None
) -> int:
    return helpers_run_lucky_tool_mode(
        tool,
        forwarded_args,
        lucky_count,
        lucky_seed,
        dispatch_tool=dispatch_tool,
    )


def _run_lucky_helper_mode(
    helper: str,
    forwarded_args: list[str],
    lucky_count: int,
    lucky_seed: int | None,
) -> int:
    return helpers_run_lucky_helper_mode(
        helper,
        forwarded_args,
        lucky_count,
        lucky_seed,
        run_chain_mode=run_chain_mode,
        run_follow_mode=run_follow_mode,
        run_stream_mode=run_stream_mode,
    )


def run_follow_mode(forwarded_args: list[str]) -> int:
    return _run_follow_mode(forwarded_args, dispatch_tool=dispatch_tool)


def run_chain_mode(forwarded_args: list[str]) -> int:
    return _run_chain_mode(forwarded_args, dispatch_tool=dispatch_tool, tool_index=TOOL_INDEX)


def run_stream_mode(forwarded_args: list[str]) -> int:
    return _run_stream_mode(forwarded_args, dispatch_tool=dispatch_tool)


_SIZE_UNITS_BYTES: dict[str, float] = {
    "b": 1.0,
    "kb": 1_000.0,
    "mb": 1_000_000.0,
    "gb": 1_000_000_000.0,
    "tb": 1_000_000_000_000.0,
    "kib": 1024.0,
    "mib": 1024.0 * 1024.0,
    "gib": 1024.0 * 1024.0 * 1024.0,
    "tib": 1024.0 * 1024.0 * 1024.0 * 1024.0,
}

_SUBTYPE_BYTES_PER_SAMPLE: dict[str, int] = {
    "PCM_S8": 1,
    "PCM_U8": 1,
    "PCM_16": 2,
    "PCM_24": 3,
    "PCM_32": 4,
    "FLOAT": 4,
    "DOUBLE": 8,
}


def _parse_size_bytes(text: str) -> float:
    return helpers_parse_size_bytes(text)


def _format_bytes_human(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    x = float(value)
    idx = 0
    while idx < len(units) - 1 and abs(x) >= 1000.0:
        x /= 1000.0
        idx += 1
    return f"{x:.3f} {units[idx]}"


def _infer_output_format(input_path: Path, requested: str) -> str:
    return helpers_infer_output_format(input_path, requested)


def _bytes_per_sample_from_subtype(subtype: str) -> int | None:
    key = str(subtype).strip().upper()
    if not key:
        return None
    return _SUBTYPE_BYTES_PER_SAMPLE.get(key)


def run_stretch_budget_mode(forwarded_args: list[str]) -> int:
    return helpers_run_stretch_budget_mode(forwarded_args)


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
            "  pvx stretch-budget input.wav --disk-budget 20GB --bit-depth 16 --requested-stretch 1000000\n"
            "  pvx doctor\n"
            "  pvx quickstart input.wav --output output.wav\n"
            "  pvx safe input.wav --material mix --output output.wav\n"
            "  pvx transforms\n"
            "  pvx smoke --output smoke_out.wav\n"
            "  pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337\n"
            "  pvx augment-manifest validate aug_out/augment_manifest.jsonl\n"
            "  pvx voc input.wav --output-dir out --lucky 8\n"
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
    forwarded_raw = list(args.args or [])

    if command_raw is None:
        parser.print_help()
        print("")
        print("Tip: run `pvx list` for tool descriptions.")
        return 0

    try:
        forwarded, lucky_count, lucky_seed = _consume_lucky_options(forwarded_raw)
    except ValueError as exc:
        parser.error(str(exc))

    command = str(command_raw).strip().lower()
    helper_commands = {
        "list",
        "ls",
        "tools",
        "examples",
        "example",
        "quickstart",
        "doctor",
        "transforms",
        "safe",
        "smoke",
        "augment",
        "augment-manifest",
        "batch-gpu",
        "guided",
        "guide",
        "follow",
        "chain",
        "stream",
        "stretch-budget",
        "stretchbudget",
        "budget",
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
    if command == "quickstart":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx quickstart`")
        return run_quickstart_mode(forwarded)
    if command == "doctor":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx doctor`")
        return run_doctor_mode(forwarded)
    if command == "transforms":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx transforms`")
        return run_transforms_mode(forwarded)
    if command == "safe":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx safe`")
        return run_safe_mode(forwarded)
    if command == "smoke":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx smoke`")
        return run_smoke_mode(forwarded)
    if command == "augment":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx augment`")
        return run_augment_mode(forwarded)
    if command == "augment-manifest":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx augment-manifest`")
        return run_augment_manifest_mode(forwarded)
    if command == "batch-gpu":
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx batch-gpu`")
        return run_batch_gpu_mode(forwarded)
    if command in {"guided", "guide"}:
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx guided`")
        try:
            return run_guided_mode()
        except ValueError as exc:
            parser.error(str(exc))
    if command == "follow":
        if lucky_count is not None:
            try:
                return _run_lucky_helper_mode("follow", forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        try:
            return run_follow_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "chain":
        if lucky_count is not None:
            try:
                return _run_lucky_helper_mode("chain", forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        try:
            return run_chain_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command == "stream":
        if lucky_count is not None:
            try:
                return _run_lucky_helper_mode("stream", forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        try:
            return run_stream_mode(forwarded)
        except ValueError as exc:
            parser.error(str(exc))
    if command in {"stretch-budget", "stretchbudget", "budget"}:
        if lucky_count is not None:
            parser.error("--lucky is not supported with `pvx stretch-budget`")
        try:
            return run_stretch_budget_mode(forwarded)
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
            if target == "quickstart":
                print("Run `pvx quickstart` for a minimal launch/demo sequence.")
                return 0
            if target == "doctor":
                print("Run `pvx doctor` for environment diagnostics and actionable fixes.")
                return 0
            if target == "transforms":
                print("Run `pvx transforms` for transform choices and backend availability.")
                return 0
            if target == "safe":
                print("Run `pvx safe --help` for conservative quality-first voc defaults.")
                return 0
            if target == "smoke":
                print("Run `pvx smoke --help` for a fast synthetic end-to-end verification render.")
                return 0
            if target == "augment":
                print(
                    "Run `pvx augment --help` for deterministic AI dataset augmentation workflows."
                )
                return 0
            if target == "augment-manifest":
                print("Run `pvx augment-manifest --help` to validate/merge augmentation manifests.")
                return 0
            if target == "batch-gpu":
                print("Run `pvx batch-gpu --help` for batched GPU augmentation across many files.")
                return 0
            if target in {"list", "ls", "tools"}:
                print_tools()
                return 0
            if target in {"guided", "guide"}:
                print("Run `pvx guided` from an interactive terminal.")
                return 0
            if target == "follow":
                print("Run `pvx follow --help` for one-command sidechain control mapping.")
                return 0
            if target == "chain":
                print("Run `pvx chain --help` for managed one-line tool chaining.")
                return 0
            if target == "stream":
                print("Run `pvx stream --help` for chunked streaming wrapper options.")
                return 0
            if target in {"stretch-budget", "stretchbudget", "budget"}:
                print("Run `pvx stretch-budget --help` for file-based stretch-budget estimates.")
                return 0
            parser.print_help()
            return 0
        if target in TOOL_INDEX:
            return dispatch_tool(target, ["--help"])
        parser.error(f"Unknown help target '{forwarded[0]}'. Use `pvx list`.")

    if command in TOOL_INDEX:
        spec = TOOL_INDEX[command]
        if lucky_count is not None:
            try:
                return _run_lucky_tool_mode(spec.name, forwarded, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        return dispatch_tool(command, forwarded)

    # Beginner shortcut: if first token looks like an input path or glob, treat as `pvx voc ...`.
    if _looks_like_audio_input(command_raw):
        shortcut_args = [command_raw, *forwarded]
        if lucky_count is not None:
            try:
                return _run_lucky_tool_mode("voc", shortcut_args, lucky_count, lucky_seed)
            except ValueError as exc:
                parser.error(str(exc))
        return dispatch_tool("voc", shortcut_args)

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
