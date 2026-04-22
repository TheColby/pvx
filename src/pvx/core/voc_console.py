"""Console, preset, and example helpers for the pvx voc CLI."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable

from pvx.core.presets import PRESET_CHOICES, PRESET_OVERRIDES

EXAMPLE_CHOICES: tuple[str, ...] = (
    "all",
    "basic",
    "vocal",
    "ambient",
    "extreme",
    "drums_safe",
    "stereo_coherent",
    "hybrid",
    "benchmark",
    "gpu",
    "pipeline",
    "csv",
)

EXAMPLE_COMMANDS: dict[str, tuple[str, str]] = {
    "basic": (
        "Basic time stretch",
        "pvx voc input.wav --stretch 1.20 --output output.wav",
    ),
    "vocal": (
        "Vocal-friendly preset with formant preservation",
        "pvx voc vocal.wav --preset vocal --pitch -2 --output vocal_tuned.wav",
    ),
    "ambient": (
        "Extreme ambient stretch",
        "pvx voc texture.wav --preset ambient --target-duration 600 --output texture_ambient.wav",
    ),
    "extreme": (
        "Extreme long-form stretch with checkpoints",
        "pvx voc source.wav --preset extreme --auto-segment-seconds 0.5 --checkpoint-dir checkpoints --output source_extreme.wav",
    ),
    "drums_safe": (
        "Transient-safe drum stretch with WSOLA regions",
        "pvx voc drums.wav --preset drums_safe --time-stretch 1.35 --output drums_safe.wav",
    ),
    "stereo_coherent": (
        "Stereo-coherent stretch with mid/side coupling",
        "pvx voc mix_stereo.wav --preset stereo_coherent --time-stretch 1.2 --output mix_coherent.wav",
    ),
    "hybrid": (
        "Hybrid transient mode (PV steady-state + WSOLA transients)",
        "pvx voc speech.wav --transient-mode hybrid --transient-sensitivity 0.6 --time-stretch 1.25 --output speech_hybrid.wav",
    ),
    "benchmark": (
        "Benchmark pvx vs Rubber Band vs librosa (tiny suite)",
        "python3 benchmarks/run_bench.py --quick --out-dir benchmarks/out",
    ),
    "gpu": (
        "CUDA render",
        "pvx voc input.wav --device cuda --stretch 1.1 --output out_gpu.wav",
    ),
    "pipeline": (
        "Tracker sidechain pipeline (pitch -> stretch, no awk)",
        "pvx pitch-track A.wav --emit pitch_to_stretch --output - | pvx voc B.wav --control-stdin --pitch-conf-min 0.75 --output B_follow.wav",
    ),
    "csv": (
        "Segment map workflow",
        "pvx voc input.wav --pitch-map map_conform.csv --output input_mapped.wav",
    ),
}

VERBOSITY_LEVELS = ("silent", "quiet", "normal", "verbose", "debug")
VERBOSITY_TO_LEVEL = {name: idx for idx, name in enumerate(VERBOSITY_LEVELS)}


class ProgressBar:
    def __init__(self, label: str, enabled: bool, width: int = 32) -> None:
        self.label = label
        self.enabled = enabled
        self.width = max(10, width)
        self._last_fraction = -1.0
        self._last_ts = 0.0
        self._finished = False
        if self.enabled:
            self.set(0.0, "start")

    def set(self, fraction: float, detail: str = "") -> None:
        if not self.enabled or self._finished:
            return

        now = time.time()
        frac = min(1.0, max(0.0, fraction))
        should_render = (
            frac >= 1.0
            or self._last_fraction < 0.0
            or (frac - self._last_fraction) >= 0.005
            or (now - self._last_ts) >= 0.15
        )
        if not should_render:
            return

        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {detail}" if detail else ""
        sys.stderr.write(f"\r[{bar}] {frac * 100:6.2f}% {self.label}{suffix}")
        sys.stderr.flush()
        self._last_fraction = frac
        self._last_ts = now
        if frac >= 1.0:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._finished = True

    def finish(self, detail: str = "done") -> None:
        self.set(1.0, detail)


def add_console_args(
    parser: argparse.ArgumentParser,
    *,
    include_no_progress_alias: bool = False,
) -> None:
    parser.add_argument(
        "--verbosity",
        choices=list(VERBOSITY_LEVELS),
        default="normal",
        help="Console verbosity level",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeat for extra detail)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output and hide status bars")
    parser.add_argument("--silent", action="store_true", help="Suppress all console output")
    if include_no_progress_alias:
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help=argparse.SUPPRESS,
        )


def console_level(args: argparse.Namespace) -> int:
    cached = getattr(args, "_console_level_cache", None)
    if cached is not None:
        return int(cached)

    base_level = VERBOSITY_TO_LEVEL.get(
        str(getattr(args, "verbosity", "normal")), VERBOSITY_TO_LEVEL["normal"]
    )
    verbose_count = int(getattr(args, "verbose", 0) or 0)
    level = min(VERBOSITY_TO_LEVEL["debug"], base_level + verbose_count)
    if bool(getattr(args, "no_progress", False)):
        level = min(level, VERBOSITY_TO_LEVEL["quiet"])
    if bool(getattr(args, "quiet", False)):
        level = min(level, VERBOSITY_TO_LEVEL["quiet"])
    if bool(getattr(args, "silent", False)):
        level = VERBOSITY_TO_LEVEL["silent"]
    args._console_level_cache = level
    return level


def is_quiet(args: argparse.Namespace) -> bool:
    return console_level(args) <= VERBOSITY_TO_LEVEL["quiet"]


def is_silent(args: argparse.Namespace) -> bool:
    return console_level(args) == VERBOSITY_TO_LEVEL["silent"]


def log_message(
    args: argparse.Namespace, message: str, *, min_level: str = "normal", error: bool = False
) -> None:
    if console_level(args) < VERBOSITY_TO_LEVEL[min_level]:
        return
    stream_to_stdout = bool(getattr(args, "stdout", False))
    print(message, file=sys.stderr if error or stream_to_stdout else sys.stdout)


def log_error(args: argparse.Namespace, message: str) -> None:
    if is_silent(args):
        return
    print(message, file=sys.stderr)


def clone_args_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def collect_cli_flags(argv: Iterable[str]) -> set[str]:
    flags: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        flags.add(flag)
    return flags


def print_cli_examples(which: str) -> None:
    key = str(which).strip().lower()
    if key not in EXAMPLE_CHOICES:
        raise ValueError(f"Unknown example preset: {which}")

    print("pvx voc example commands\n")
    if key == "all":
        for name in EXAMPLE_CHOICES:
            if name == "all":
                continue
            title, command = EXAMPLE_COMMANDS[name]
            print(f"[{name}] {title}")
            print(command)
            print()
        return

    title, command = EXAMPLE_COMMANDS[key]
    print(f"[{key}] {title}")
    print(command)


def apply_named_preset(
    args: argparse.Namespace,
    *,
    preset: str,
    provided_flags: set[str],
) -> list[str]:
    key = str(preset or "none").strip().lower()
    if key not in PRESET_CHOICES:
        raise ValueError(f"Unknown preset: {preset}")

    overrides = PRESET_OVERRIDES.get(key, {})
    changes: list[str] = []
    for field, value in overrides.items():
        cli_flag = f"--{field.replace('_', '-')}"
        if cli_flag in provided_flags:
            continue
        if not hasattr(args, field):
            continue
        setattr(args, field, value)
        changes.append(field)
    return changes
