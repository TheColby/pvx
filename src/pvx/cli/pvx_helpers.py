#!/usr/bin/env python3

"""Helper modes and utility flows for the unified pvx CLI."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from pvx.cli.catalog import _LUCKY_PRESETS, _LUCKY_SUPPORTED_TOOLS, _LUCKY_WINDOWS


def run_doctor_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx doctor",
        description="Environment and launch-readiness diagnostics for pvx.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code if warnings are found",
    )
    args = parser.parse_args(forwarded_args)

    cwd = Path.cwd().resolve()
    venv_active = bool(getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    python_exe = str(Path(sys.executable).resolve())
    pvx_on_path = shutil.which("pvx")
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    venv_bin = str((cwd / ".venv" / "bin").resolve())

    try:
        importlib.import_module("scipy")
        scipy_ok = True
    except (ImportError, ModuleNotFoundError):
        scipy_ok = False
    try:
        importlib.import_module("cupy")
        cupy_ok = True
    except (ImportError, ModuleNotFoundError):
        cupy_ok = False

    warnings: list[str] = []
    if not venv_active:
        warnings.append("Python virtual environment is not active.")
    if pvx_on_path is None:
        warnings.append("`pvx` executable is not on PATH.")
    if (cwd / ".venv").exists() and venv_bin not in path_entries:
        warnings.append("Project virtualenv bin directory is not on PATH.")
    if not scipy_ok:
        warnings.append("SciPy not installed: czt/dct/dst/hartley transforms may be unavailable.")

    report = {
        "python_executable": python_exe,
        "python_version": sys.version.split()[0],
        "cwd": str(cwd),
        "venv_active": venv_active,
        "pvx_on_path": pvx_on_path,
        "venv_bin_expected": venv_bin,
        "venv_bin_on_path": venv_bin in path_entries,
        "scipy_installed": scipy_ok,
        "cupy_installed": cupy_ok,
        "warnings": list(warnings),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("pvx doctor")
        print(f"- python: {report['python_executable']} (v{report['python_version']})")
        print(f"- working directory: {report['cwd']}")
        print(f"- virtual environment active: {'yes' if venv_active else 'no'}")
        print(f"- pvx on PATH: {pvx_on_path if pvx_on_path is not None else 'no'}")
        print(f"- scipy installed: {'yes' if scipy_ok else 'no'}")
        print(f"- cupy installed: {'yes' if cupy_ok else 'no'}")
        if warnings:
            print("")
            print("Warnings:")
            for item in warnings:
                print(f"- {item}")
            print("")
            print("Suggested fixes:")
            print("- Activate virtual environment: source .venv/bin/activate")
            print("- Add pvx to PATH (zsh):")
            print(
                '  printf \'export PATH="%s/.venv/bin:$PATH"\\n\' "$(pwd)" >> ~/.zshrc && source ~/.zshrc'
            )
            print("- Install optional transforms/GPU dependencies:")
            print("  python3 -m pip install scipy")
            print("  python3 -m pip install cupy-cuda12x  # optional, NVIDIA CUDA only")
        else:
            print("")
            print("No launch-blocking warnings found.")

    if args.strict and warnings:
        return 1
    return 0


def run_quickstart_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx quickstart",
        description="Print a minimal copy-paste launch sequence for announcement demos.",
    )
    parser.add_argument(
        "input", nargs="?", default="input.wav", help="Input audio path (default: input.wav)"
    )
    parser.add_argument(
        "--output", default="output.wav", help="Output audio path (default: output.wav)"
    )
    parser.add_argument(
        "--material",
        choices=["mix", "speech", "vocal", "drums", "ambient"],
        default="mix",
        help="Material profile for `pvx safe` command generation",
    )
    args = parser.parse_args(forwarded_args)

    print("pvx quickstart")
    print("")
    print("1) Diagnose environment")
    print("pvx doctor")
    print("")
    print("2) Run quality-safe first render")
    print(
        f"pvx safe {args.input!s} --material {args.material} "
        f"--output {args.output!s}"
    )
    print("")
    print("3) Inspect transform options")
    print("pvx transforms")
    print("")
    print("4) Print curated examples")
    print("pvx examples basic")
    print("")
    print("5) Run a synthetic smoke test")
    print("pvx smoke --output smoke_out.wav")
    return 0


def _token_flag(token: str) -> str:
    return token.split("=", 1)[0]


def run_safe_mode(
    forwarded_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx safe",
        description="Run `pvx voc` with conservative, quality-first defaults for first-pass renders.",
    )
    parser.add_argument("input", help="Input audio path")
    parser.add_argument("--output", "--out", dest="output", required=True, help="Output audio path")
    parser.add_argument(
        "--material",
        choices=["mix", "speech", "vocal", "drums", "ambient"],
        default="mix",
        help="Material profile (default: mix)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs")
    parser.add_argument("--silent", action="store_true", help="Suppress logs")
    args, passthrough = parser.parse_known_args(forwarded_args)

    passthrough_flags = {_token_flag(token) for token in passthrough if token.startswith("-")}
    forbidden_passthrough = {"--output", "--out", "-o", "--stdout"}
    bad_flags = sorted(passthrough_flags & forbidden_passthrough)
    if bad_flags:
        parser.error(
            f"Do not pass {bad_flags} via passthrough in `pvx safe`; safe mode manages output routing."
        )

    preset_by_material = {
        "mix": "stereo_coherent",
        "speech": "vocal_studio",
        "vocal": "vocal_studio",
        "drums": "drums_safe",
        "ambient": "extreme_ambient",
    }

    voc_args: list[str] = [
        str(args.input),
        "--preset",
        preset_by_material[str(args.material)],
    ]

    if "--phase-locking" not in passthrough_flags:
        voc_args.extend(["--phase-locking", "identity"])
    if "--transient-mode" not in passthrough_flags:
        voc_args.extend(["--transient-mode", "hybrid"])
    if "--transient-sensitivity" not in passthrough_flags:
        voc_args.extend(["--transient-sensitivity", "0.60"])
    if "--stereo-mode" not in passthrough_flags:
        voc_args.extend(["--stereo-mode", "mid_side_lock"])
    if "--coherence-strength" not in passthrough_flags:
        voc_args.extend(["--coherence-strength", "0.85"])

    voc_args.extend(["--output", str(args.output)])
    if args.overwrite:
        voc_args.append("--overwrite")
    if args.quiet:
        voc_args.append("--quiet")
    if args.silent:
        voc_args.append("--silent")
    voc_args.extend(passthrough)
    return dispatch_tool("voc", voc_args)


def run_transforms_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx transforms",
        description="Show available per-frame transform backends and practical recommendations.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(forwarded_args)

    try:
        importlib.import_module("scipy.fft")
        scipy_fft_ok = True
    except (ImportError, ModuleNotFoundError):
        scipy_fft_ok = False
    try:
        from scipy.signal import czt as _scipy_czt  # noqa: F401

        scipy_czt_ok = True
    except (ImportError, ModuleNotFoundError):
        scipy_czt_ok = False

    transforms = [
        {
            "name": "fft",
            "available": True,
            "recommended_for": "default production use",
            "notes": "Best overall speed/quality baseline",
        },
        {
            "name": "dft",
            "available": True,
            "recommended_for": "reference and non-power-of-two research",
            "notes": "Slowest but direct reference path",
        },
        {
            "name": "czt",
            "available": scipy_czt_ok,
            "recommended_for": "zoomed/custom spectral focus",
            "notes": "Requires scipy.signal.czt",
        },
        {
            "name": "dct",
            "available": scipy_fft_ok,
            "recommended_for": "real-transform experiments",
            "notes": "Requires scipy.fft",
        },
        {
            "name": "dst",
            "available": scipy_fft_ok,
            "recommended_for": "real-transform experiments",
            "notes": "Requires scipy.fft",
        },
        {
            "name": "hartley",
            "available": scipy_fft_ok,
            "recommended_for": "real-transform experiments",
            "notes": "Requires scipy.fft",
        },
    ]

    if args.json:
        print(json.dumps({"transforms": transforms}, indent=2, sort_keys=True))
        return 0

    print("pvx transform guide")
    print("")
    for item in transforms:
        state = "yes" if bool(item["available"]) else "no"
        print(f"- {item['name']}: available={state}")
        print(f"  use: {item['recommended_for']}")
        print(f"  note: {item['notes']}")
    print("")
    print(
        "Rule of thumb: start with --transform fft, then A/B against alternatives only if you need a specific behavior."
    )
    return 0


def _build_smoke_signal(sample_rate: int, duration: float) -> np.ndarray:
    frames = max(1024, int(round(float(sample_rate) * float(duration))))
    t = np.arange(frames, dtype=np.float64) / float(sample_rate)
    tone = 0.18 * np.sin(2.0 * np.pi * 220.0 * t) + 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
    fade = min(256, max(8, frames // 20))
    ramp = np.linspace(0.0, 1.0, num=fade, endpoint=True, dtype=np.float64)
    tone[:fade] *= ramp
    tone[-fade:] *= ramp[::-1]
    return np.stack([tone, tone], axis=1)


def run_smoke_mode(
    forwarded_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx smoke",
        description="Run a fast synthetic end-to-end smoke render for launch confidence.",
    )
    parser.add_argument("--output", default="smoke_out.wav", help="Output path for smoke render")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.30,
        help="Synthetic input duration seconds (default: 0.30)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Synthetic input sample rate (default: 24000)",
    )
    parser.add_argument(
        "--stretch", type=float, default=1.25, help="Smoke render stretch factor (default: 1.25)"
    )
    parser.add_argument(
        "--pitch", type=float, default=0.0, help="Smoke render pitch semitones (default: 0.0)"
    )
    args = parser.parse_args(forwarded_args)

    if float(args.duration) <= 0.0:
        parser.error("--duration must be > 0")
    if int(args.sample_rate) <= 1000:
        parser.error("--sample-rate must be > 1000")
    if float(args.stretch) <= 0.0:
        parser.error("--stretch must be > 0")

    with tempfile.TemporaryDirectory(prefix="pvx-smoke-") as tmp:
        tmp_dir = Path(tmp)
        in_path = tmp_dir / "smoke_in.wav"
        out_path = Path(args.output).expanduser().resolve()
        signal = _build_smoke_signal(int(args.sample_rate), float(args.duration))
        sf.write(str(in_path), signal, int(args.sample_rate))
        code = dispatch_tool(
            "voc",
            [
                str(in_path),
                "--stretch",
                f"{float(args.stretch):.8g}",
                "--pitch",
                f"{float(args.pitch):.8g}",
                "--preset",
                "stereo_coherent",
                "--phase-locking",
                "identity",
                "--transient-mode",
                "hybrid",
                "--output",
                str(out_path),
                "--overwrite",
                "--silent",
            ],
        )
        if int(code) != 0:
            print(f"[smoke] failed: voc exited with code {code}", file=sys.stderr)
            return int(code)
        if not out_path.exists():
            print("[smoke] failed: output file was not created", file=sys.stderr)
            return 1
        info = sf.info(str(out_path))
        print(
            f"[smoke] ok -> {out_path} | frames={int(info.frames)} sr={int(info.samplerate)} channels={int(info.channels)}"
        )
    return 0


def run_guided_mode(
    *,
    dispatch_tool: Callable[[str, list[str]], int],
    prompt_text: Callable[[str, str], str],
    prompt_choice: Callable[[str, tuple[str, ...], str], str],
    print_command_preview: Callable[[str, list[str]], None],
) -> int:
    if not sys.stdin.isatty():
        raise ValueError("`pvx guided` requires an interactive terminal (TTY stdin)")

    print("pvx guided mode")
    print("Press Enter to accept defaults.\n")

    mode = prompt_choice(
        "Workflow (voc/freeze/harmonize/retune/morph)",
        ("voc", "freeze", "harmonize", "retune", "morph"),
        "voc",
    )

    if mode == "voc":
        input_path = prompt_text("Input path", "input.wav")
        output_path = prompt_text("Output path", "output.wav")
        stretch = prompt_text("Stretch factor", "1.20")
        semitones = prompt_text("Pitch shift semitones", "0")
        preset = prompt_text("Preset", "default")
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
        input_path = prompt_text("Input path", "input.wav")
        output_path = prompt_text("Output path", "output_freeze.wav")
        freeze_time = prompt_text("Freeze time (seconds)", "0.25")
        duration = prompt_text("Output duration (seconds)", "8.0")
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
        input_path = prompt_text("Input path", "input.wav")
        output_path = prompt_text("Output path", "output_harm.wav")
        intervals = prompt_text("Intervals (semitones CSV)", "0,4,7")
        forwarded = [
            input_path,
            "--intervals",
            intervals,
            "--output",
            output_path,
        ]
    elif mode == "retune":
        input_path = prompt_text("Input path", "input.wav")
        output_path = prompt_text("Output path", "output_retune.wav")
        root = prompt_text("Root note", "C")
        scale = prompt_text("Scale", "major")
        strength = prompt_text("Correction strength", "0.85")
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
        input_a = prompt_text("Input A path", "a.wav")
        input_b = prompt_text("Input B path", "b.wav")
        output_path = prompt_text("Output path", "morph.wav")
        alpha = prompt_text("Morph alpha (0..1 or CSV/JSON control file)", "0.50")
        forwarded = [
            input_a,
            input_b,
            "--alpha",
            alpha,
            "--output",
            output_path,
        ]

    print_command_preview(mode, forwarded)
    run_now = prompt_choice("Run now? (yes/no)", ("yes", "no"), "yes")
    if run_now == "no":
        print("Command preview only; no processing executed.")
        return 0
    return dispatch_tool(mode, forwarded)


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


def _strip_flags(args: list[str], flag_has_value: dict[str, bool]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        token = str(args[i])
        flag = _token_flag(token)
        has_value = flag_has_value.get(flag)
        if has_value is None:
            out.append(token)
            i += 1
            continue
        if "=" in token:
            i += 1
            continue
        if has_value:
            i += 2
            continue
        i += 1
    return out


def _replace_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    replaced = False
    i = 0
    while i < len(args):
        token = str(args[i])
        tok_flag = _token_flag(token)
        if tok_flag == flag:
            if not replaced:
                out.extend([flag, value])
                replaced = True
            if "=" in token:
                i += 1
            else:
                i += 2
            continue
        out.append(token)
        i += 1
    if not replaced:
        out.extend([flag, value])
    return out


def _consume_lucky_options(args: list[str]) -> tuple[list[str], int | None, int | None]:
    clean: list[str] = []
    lucky_count: int | None = None
    lucky_seed: int | None = None
    i = 0
    while i < len(args):
        token = str(args[i])
        if token == "--lucky":
            if i + 1 >= len(args):
                raise ValueError("--lucky requires an integer value")
            lucky_count = int(str(args[i + 1]))
            i += 2
            continue
        if token.startswith("--lucky="):
            lucky_count = int(token.split("=", 1)[1])
            i += 1
            continue
        if token == "--lucky-seed":
            if i + 1 >= len(args):
                raise ValueError("--lucky-seed requires an integer value")
            lucky_seed = int(str(args[i + 1]))
            i += 2
            continue
        if token.startswith("--lucky-seed="):
            lucky_seed = int(token.split("=", 1)[1])
            i += 1
            continue
        clean.append(token)
        i += 1

    if lucky_count is not None and lucky_count <= 0:
        raise ValueError("--lucky must be a positive integer")
    return clean, lucky_count, lucky_seed


def _lucky_output_variant(base: Path, idx: int) -> Path:
    stem = base.stem if base.stem else "output"
    suffix = base.suffix if base.suffix else ".wav"
    return base.with_name(f"{stem}_lucky_{idx:03d}{suffix}")


def _lucky_mastering_overrides(rng: random.Random) -> list[str]:
    out: list[str] = []
    if rng.random() < 0.85:
        out.extend(["--target-lufs", f"{rng.uniform(-18.0, -11.0):.3f}"])
    if rng.random() < 0.70:
        out.extend(
            [
                "--compressor-threshold-db",
                f"{rng.uniform(-30.0, -12.0):.3f}",
                "--compressor-ratio",
                f"{rng.uniform(1.5, 4.8):.3f}",
                "--compressor-attack-ms",
                f"{rng.uniform(2.0, 40.0):.3f}",
                "--compressor-release-ms",
                f"{rng.uniform(60.0, 280.0):.3f}",
                "--compressor-makeup-db",
                f"{rng.uniform(0.0, 6.0):.3f}",
            ]
        )
    if rng.random() < 0.90:
        out.extend(["--limiter-threshold", f"{rng.uniform(0.88, 0.995):.4f}"])
    if rng.random() < 0.65:
        out.extend(
            [
                "--soft-clip-level",
                f"{rng.uniform(0.90, 0.995):.4f}",
                "--soft-clip-type",
                rng.choice(["tanh", "arctan", "cubic"]),
                "--soft-clip-drive",
                f"{rng.uniform(0.8, 2.4):.4f}",
            ]
        )
    if rng.random() < 0.30:
        out.extend(["--hard-clip-level", f"{rng.uniform(0.95, 0.999):.4f}"])
    return out


def _lucky_tool_overrides(tool: str, rng: random.Random) -> list[str]:
    if tool == "voc":
        window = rng.choice(_LUCKY_WINDOWS)
        out = [
            "--preset",
            rng.choice(_LUCKY_PRESETS),
            "--stretch",
            f"{rng.uniform(0.4, 3.4):.4f}",
            "--pitch",
            f"{rng.uniform(-12.0, 12.0):.4f}",
            "--window",
            window,
        ]
        if window == "kaiser":
            out.extend(["--kaiser-beta", f"{rng.uniform(7.0, 22.0):.4f}"])
        return out
    if tool == "freeze":
        out = [
            "--freeze-time",
            f"{rng.uniform(0.02, 0.92):.4f}",
            "--duration",
            f"{rng.uniform(5.0, 90.0):.4f}",
            "--phase-mode",
            rng.choice(["instantaneous", "bin", "hold"]),
        ]
        if rng.random() < 0.55:
            out.append("--random-phase")
        return out
    if tool == "harmonize":
        return [
            "--intervals",
            rng.choice(["0,7,12", "0,4,7,11", "0,3,7,10", "0,7,14,19"]),
            "--force-stereo",
        ]
    if tool in {"conform", "warp"}:
        return ["--crossfade-ms", f"{rng.uniform(2.0, 35.0):.4f}"]
    if tool == "formant":
        return [
            "--mode",
            rng.choice(["shift", "preserve"]),
            "--formant-shift-ratio",
            f"{rng.uniform(0.72, 1.42):.4f}",
            "--pitch-shift-semitones",
            f"{rng.uniform(-5.0, 5.0):.4f}",
        ]
    if tool == "transient":
        return [
            "--time-stretch",
            f"{rng.uniform(0.55, 2.4):.4f}",
            "--pitch-shift-semitones",
            f"{rng.uniform(-8.0, 8.0):.4f}",
            "--transient-threshold",
            f"{rng.uniform(1.1, 2.4):.4f}",
        ]
    if tool == "unison":
        return [
            "--voices",
            str(rng.randint(3, 9)),
            "--detune-cents",
            f"{rng.uniform(4.0, 32.0):.4f}",
            "--width",
            f"{rng.uniform(0.2, 1.0):.4f}",
            "--dry-mix",
            f"{rng.uniform(0.05, 0.45):.4f}",
        ]
    if tool == "denoise":
        return [
            "--reduction-db",
            f"{rng.uniform(4.0, 18.0):.4f}",
            "--floor",
            f"{rng.uniform(0.05, 0.25):.4f}",
            "--smooth",
            str(rng.randint(3, 12)),
        ]
    if tool == "deverb":
        return [
            "--strength",
            f"{rng.uniform(0.15, 0.75):.4f}",
            "--decay",
            f"{rng.uniform(0.75, 0.97):.4f}",
            "--floor",
            f"{rng.uniform(0.05, 0.30):.4f}",
        ]
    if tool == "retune":
        return [
            "--root",
            rng.choice(["C", "D", "E", "F", "G", "A", "B"]),
            "--scale",
            rng.choice(["major", "minor", "pentatonic", "chromatic"]),
            "--strength",
            f"{rng.uniform(0.45, 1.0):.4f}",
        ]
    if tool == "layer":
        return [
            "--harmonic-stretch",
            f"{rng.uniform(0.6, 2.2):.4f}",
            "--percussive-stretch",
            f"{rng.uniform(0.7, 1.8):.4f}",
            "--harmonic-pitch-semitones",
            f"{rng.uniform(-6.0, 6.0):.4f}",
            "--percussive-pitch-semitones",
            f"{rng.uniform(-2.0, 2.0):.4f}",
            "--harmonic-gain",
            f"{rng.uniform(0.6, 1.4):.4f}",
            "--percussive-gain",
            f"{rng.uniform(0.6, 1.4):.4f}",
        ]
    if tool in {"filter", "tvfilter", "noisefilter", "bandamp", "spec-compander"}:
        return [
            "--response-mix",
            f"{rng.uniform(0.3, 1.0):.4f}",
            "--dry-mix",
            f"{rng.uniform(0.0, 0.35):.4f}",
            "--response-gain-db",
            f"{rng.uniform(-6.0, 8.0):.4f}",
            "--noise-floor",
            f"{rng.uniform(0.6, 2.0):.4f}",
            "--band-gain-db",
            f"{rng.uniform(2.0, 14.0):.4f}",
            "--peak-count",
            str(rng.randint(4, 16)),
            "--comp-ratio",
            f"{rng.uniform(1.1, 3.6):.4f}",
            "--expand-ratio",
            f"{rng.uniform(1.0, 2.5):.4f}",
        ]
    if tool in {"ring", "ringfilter", "ringtvfilter"}:
        return [
            "--frequency-hz",
            f"{rng.uniform(12.0, 1800.0):.4f}",
            "--depth",
            f"{rng.uniform(0.2, 1.0):.4f}",
            "--mix",
            f"{rng.uniform(0.25, 1.0):.4f}",
            "--feedback",
            f"{rng.uniform(0.0, 0.45):.4f}",
            "--resonance-hz",
            f"{rng.uniform(120.0, 4800.0):.4f}",
            "--resonance-q",
            f"{rng.uniform(1.0, 18.0):.4f}",
            "--resonance-mix",
            f"{rng.uniform(0.15, 0.95):.4f}",
        ]
    if tool == "chordmapper":
        return [
            "--root-hz",
            f"{rng.uniform(80.0, 440.0):.4f}",
            "--chord",
            rng.choice(["major", "minor", "sus4"]),
            "--strength",
            f"{rng.uniform(0.3, 1.0):.4f}",
            "--boost-db",
            f"{rng.uniform(2.0, 12.0):.4f}",
            "--attenuation",
            f"{rng.uniform(0.15, 0.85):.4f}",
        ]
    if tool == "inharmonator":
        return [
            "--inharmonic-f0-hz",
            f"{rng.uniform(60.0, 440.0):.4f}",
            "--inharmonicity",
            f"{rng.uniform(1e-6, 6e-4):.8f}",
            "--inharmonic-mix",
            f"{rng.uniform(0.25, 1.0):.4f}",
            "--dry-mix",
            f"{rng.uniform(0.0, 0.35):.4f}",
        ]
    if tool == "morph":
        return [
            "--alpha",
            f"{rng.uniform(0.15, 0.92):.4f}",
            "--blend-mode",
            rng.choice(["linear", "geometric", "carrier_a_envelope_b", "carrier_a_mask_b"]),
            "--phase-mix",
            f"{rng.uniform(0.0, 1.0):.4f}",
            "--mask-exponent",
            f"{rng.uniform(0.7, 2.2):.4f}",
            "--envelope-lifter",
            str(rng.randint(16, 72)),
        ]
    return []


def _run_lucky_tool_mode(
    tool: str,
    forwarded_args: list[str],
    lucky_count: int,
    lucky_seed: int | None,
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> int:
    if tool not in _LUCKY_SUPPORTED_TOOLS:
        raise ValueError(
            f"`--lucky` is not supported for `{tool}`. "
            f"Supported tools: {', '.join(sorted(_LUCKY_SUPPORTED_TOOLS))}"
        )
    if lucky_count <= 0:
        raise ValueError("--lucky must be > 0")
    seed = (
        int(lucky_seed) if lucky_seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    )
    rng = random.Random(seed)
    print(f"[lucky] seed={seed} tool={tool} runs={lucky_count}")

    if tool == "morph":
        output_value = _extract_flag_value(forwarded_args, ("--output", "--out", "-o"))
        output_base = Path(
            output_value if output_value not in {None, "", "-"} else "morph_lucky.wav"
        )
        if output_base.suffix == "":
            output_base = output_base.with_suffix(".wav")
        output_dir = output_base.parent if str(output_base.parent) else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        base = _strip_flags(
            forwarded_args,
            {
                "--output": True,
                "--out": True,
                "-o": True,
                "--stdout": False,
            },
        )
        for run_idx in range(1, lucky_count + 1):
            out_path = _lucky_output_variant(output_dir / output_base.name, run_idx)
            run_args = list(base)
            run_args.extend(_lucky_tool_overrides(tool, rng))
            run_args.extend(_lucky_mastering_overrides(rng))
            run_args.extend(["--output", str(out_path), "--overwrite"])
            print(f"[lucky] {tool} run {run_idx}/{lucky_count} -> {out_path}")
            code = dispatch_tool(tool, run_args)
            if code != 0:
                return int(code)
        return 0

    output_dir_value = _extract_flag_value(forwarded_args, ("--output-dir", "-o"))
    if output_dir_value in {None, ""}:
        explicit_out = _extract_flag_value(forwarded_args, ("--output", "--out"))
        if explicit_out not in {None, "", "-"}:
            output_dir = Path(str(explicit_out)).parent
        else:
            output_dir = Path("lucky_out")
    else:
        output_dir = Path(str(output_dir_value))
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _strip_flags(
        forwarded_args,
        {
            "--output": True,
            "--out": True,
            "--output-dir": True,
            "-o": True,
            "--stdout": False,
            "--suffix": True,
        },
    )
    for run_idx in range(1, lucky_count + 1):
        run_args = list(base)
        run_args.extend(_lucky_tool_overrides(tool, rng))
        run_args.extend(_lucky_mastering_overrides(rng))
        run_args.extend(
            [
                "--output-dir",
                str(output_dir),
                "--suffix",
                f"_lucky_{run_idx:03d}",
                "--overwrite",
            ]
        )
        print(f"[lucky] {tool} run {run_idx}/{lucky_count} -> {output_dir}/*_lucky_{run_idx:03d}.*")
        code = dispatch_tool(tool, run_args)
        if code != 0:
            return int(code)
    return 0


def _run_lucky_helper_mode(
    helper: str,
    forwarded_args: list[str],
    lucky_count: int,
    lucky_seed: int | None,
    *,
    run_chain_mode: Callable[[list[str]], int],
    run_follow_mode: Callable[[list[str]], int],
    run_stream_mode: Callable[[list[str]], int],
) -> int:
    seed = (
        int(lucky_seed) if lucky_seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    )
    rng = random.Random(seed)
    print(f"[lucky] seed={seed} helper={helper} runs={lucky_count}")

    output_value = _extract_flag_value(forwarded_args, ("--output", "--out"))
    output_base = Path(
        output_value if output_value not in {None, "", "-"} else f"{helper}_lucky.wav"
    )
    if output_base.suffix == "":
        output_base = output_base.with_suffix(".wav")
    output_dir = output_base.parent if str(output_base.parent) else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _strip_flags(
        forwarded_args,
        {
            "--output": True,
            "--out": True,
        },
    )

    for run_idx in range(1, lucky_count + 1):
        out_path = _lucky_output_variant(output_dir / output_base.name, run_idx)
        run_args = list(base)
        if helper == "chain":
            current_pipeline = _extract_flag_value(run_args, ("--pipeline",))
            if current_pipeline not in {None, ""}:
                random_stage = (
                    "voc "
                    f"--stretch {rng.uniform(0.55, 2.4):.4f} "
                    f"--pitch {rng.uniform(-7.0, 7.0):.4f} "
                    f"--preset {rng.choice(_LUCKY_PRESETS)}"
                )
                run_args = _replace_flag_value(
                    run_args, "--pipeline", f"{current_pipeline} | {random_stage}"
                )
            run_args.extend(["--output", str(out_path)])
            print(f"[lucky] chain run {run_idx}/{lucky_count} -> {out_path}")
            code = run_chain_mode(run_args)
        elif helper == "follow":
            run_args.extend(
                [
                    "--stretch",
                    f"{rng.uniform(0.7, 1.5):.4f}",
                    "--pitch-conf-min",
                    f"{rng.uniform(0.45, 0.9):.4f}",
                    "--pitch-map-smooth-ms",
                    f"{rng.uniform(0.0, 40.0):.4f}",
                    "--pitch-map-crossfade-ms",
                    f"{rng.uniform(8.0, 40.0):.4f}",
                    "--output",
                    str(out_path),
                    "--overwrite",
                ]
            )
            print(f"[lucky] follow run {run_idx}/{lucky_count} -> {out_path}")
            code = run_follow_mode(run_args)
        elif helper == "stream":
            run_args.extend(
                [
                    "--output",
                    str(out_path),
                    "--chunk-seconds",
                    f"{rng.uniform(0.06, 0.35):.4f}",
                    "--time-stretch",
                    f"{rng.uniform(0.6, 3.0):.4f}",
                    "--pitch",
                    f"{rng.uniform(-9.0, 9.0):.4f}",
                    "--preset",
                    rng.choice(_LUCKY_PRESETS),
                ]
            )
            print(f"[lucky] stream run {run_idx}/{lucky_count} -> {out_path}")
            code = run_stream_mode(run_args)
        else:
            raise ValueError(f"Unsupported helper for --lucky: {helper}")
        if code != 0:
            return int(code)
    return 0


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
    raw = str(text).strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]*)", raw)
    if match is None:
        raise ValueError(f"Invalid size '{text}'. Use forms like 500MB, 20GB, 2.5TB, or 10GiB.")
    value = float(match.group(1))
    unit = str(match.group(2) or "b")
    if unit not in _SIZE_UNITS_BYTES:
        choices = ", ".join(sorted(_SIZE_UNITS_BYTES))
        raise ValueError(f"Unsupported size unit '{unit}'. Supported units: {choices}")
    out = value * _SIZE_UNITS_BYTES[unit]
    if out <= 0.0:
        raise ValueError("Size must be > 0")
    return out


def _format_bytes_human(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    x = float(value)
    idx = 0
    while idx < len(units) - 1 and abs(x) >= 1000.0:
        x /= 1000.0
        idx += 1
    return f"{x:.3f} {units[idx]}"


def _infer_output_format(input_path: Path, requested: str) -> str:
    token = str(requested).strip().lower()
    if token == "auto":
        token = input_path.suffix.lower().lstrip(".")
    token = token.lstrip(".")
    if token == "aif":
        token = "aiff"
    if token == "oga":
        token = "ogg"
    allowed = {"wav", "flac", "aiff", "ogg", "caf"}
    if token not in allowed:
        raise ValueError(
            f"Unsupported output format '{requested}'. Choose one of: {', '.join(sorted(allowed))}, auto"
        )
    return token


def _bytes_per_sample_from_subtype(subtype: str) -> int | None:
    key = str(subtype).strip().upper()
    if not key:
        return None
    return _SUBTYPE_BYTES_PER_SAMPLE.get(key)


def run_stretch_budget_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx stretch-budget",
        description=(
            "Estimate maximum safe time-stretch for an input file under a disk budget.\n"
            "This is a storage-budget estimate, not a quality guarantee."
        ),
    )
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument(
        "--disk-budget",
        type=str,
        default=None,
        help="Total budget size (e.g., 500MB, 20GB, 2TiB). If omitted, use free space at --budget-path.",
    )
    parser.add_argument(
        "--budget-path",
        type=Path,
        default=Path("."),
        help="Path used to query free space when --disk-budget is omitted (default: current directory).",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.90,
        help="Usable fraction of budget in (0,1]; default: 0.90 (10%% headroom).",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="auto",
        help="Output format assumption: auto, wav, flac, aiff, ogg, caf (default: auto from input extension).",
    )
    parser.add_argument(
        "--bit-depth",
        choices=["inherit", "16", "24", "32f"],
        default="inherit",
        help="Bit-depth assumption when --subtype is not set (default: inherit from input subtype).",
    )
    parser.add_argument(
        "--subtype",
        type=str,
        default=None,
        help="Explicit libsndfile subtype assumption (e.g., PCM_16, PCM_24, FLOAT).",
    )
    parser.add_argument(
        "--requested-stretch",
        type=float,
        default=None,
        help="Optional stretch ratio to evaluate against the computed budget.",
    )
    parser.add_argument(
        "--fail-if-exceeds",
        action="store_true",
        help="Return non-zero when --requested-stretch does not fit the usable budget.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    args = parser.parse_args(forwarded_args)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input not found: {input_path}")
    if args.safety_margin <= 0.0 or args.safety_margin > 1.0:
        parser.error("--safety-margin must be in (0, 1]")
    if args.requested_stretch is not None and float(args.requested_stretch) <= 0.0:
        parser.error("--requested-stretch must be > 0")

    if args.disk_budget is not None:
        try:
            budget_bytes = _parse_size_bytes(str(args.disk_budget))
        except ValueError as exc:
            parser.error(str(exc))
        budget_source = f"explicit:{str(args.disk_budget).strip()}"
    else:
        budget_root = Path(args.budget_path).expanduser().resolve()
        try:
            budget_bytes = float(shutil.disk_usage(str(budget_root)).free)
        except OSError as exc:
            parser.error(f"Failed to query free disk at {budget_root}: {exc}")
        budget_source = f"free-space:{budget_root}"

    if budget_bytes <= 0.0:
        parser.error("Resolved budget must be > 0 bytes")

    try:
        info = sf.info(str(input_path))
    except (OSError, RuntimeError) as exc:
        parser.error(f"Failed to read input metadata: {exc}")
    frames = int(getattr(info, "frames", 0) or 0)
    channels = int(getattr(info, "channels", 0) or 0)
    sample_rate = int(getattr(info, "samplerate", 0) or 0)
    if frames <= 0:
        parser.error("Input has no audio frames")
    if channels <= 0:
        parser.error("Input has invalid channel count")
    if sample_rate <= 0:
        parser.error("Input has invalid sample rate")

    try:
        output_format = _infer_output_format(input_path, str(args.output_format))
    except ValueError as exc:
        parser.error(str(exc))

    bytes_per_sample: int | None = None
    bytes_source = ""
    subtype_assumed = ""
    if args.subtype is not None:
        bytes_per_sample = _bytes_per_sample_from_subtype(str(args.subtype))
        if bytes_per_sample is None:
            supported = ", ".join(sorted(_SUBTYPE_BYTES_PER_SAMPLE))
            parser.error(
                f"Unsupported --subtype '{args.subtype}' for estimator. Supported: {supported}"
            )
        subtype_assumed = str(args.subtype).strip().upper()
        bytes_source = "explicit --subtype"
    elif str(args.bit_depth) != "inherit":
        if str(args.bit_depth) == "16":
            bytes_per_sample = 2
            subtype_assumed = "PCM_16"
        elif str(args.bit_depth) == "24":
            bytes_per_sample = 3
            subtype_assumed = "PCM_24"
        else:
            bytes_per_sample = 4
            subtype_assumed = "FLOAT"
        bytes_source = "explicit --bit-depth"
    else:
        inferred_subtype = str(getattr(info, "subtype", "") or "")
        inferred_bps = _bytes_per_sample_from_subtype(inferred_subtype)
        if inferred_bps is None:
            bytes_per_sample = 4
            subtype_assumed = "FLOAT"
            bytes_source = f"fallback from input subtype '{inferred_subtype or 'unknown'}'"
        else:
            bytes_per_sample = inferred_bps
            subtype_assumed = inferred_subtype.strip().upper()
            bytes_source = "inherited input subtype"

    base_bytes = float(frames) * float(channels) * float(bytes_per_sample)
    usable_budget_bytes = float(budget_bytes) * float(args.safety_margin)
    max_safe_stretch = usable_budget_bytes / max(base_bytes, 1.0)
    input_duration_sec = float(frames) / float(sample_rate)
    max_duration_sec = input_duration_sec * max_safe_stretch
    max_frames = max(1, int(round(float(frames) * max_safe_stretch)))
    conservative_for_compressed = output_format in {"flac", "ogg"}

    requested_bytes: float | None = None
    requested_ok: bool | None = None
    requested_duration_sec: float | None = None
    if args.requested_stretch is not None:
        requested = float(args.requested_stretch)
        requested_bytes = base_bytes * requested
        requested_ok = bool(requested_bytes <= usable_budget_bytes)
        requested_duration_sec = input_duration_sec * requested

    payload = {
        "input_path": str(input_path),
        "input_frames": int(frames),
        "input_channels": int(channels),
        "input_sample_rate": int(sample_rate),
        "input_duration_sec": float(input_duration_sec),
        "input_subtype": str(getattr(info, "subtype", "") or ""),
        "output_format_assumed": str(output_format),
        "subtype_assumed": str(subtype_assumed),
        "bytes_per_sample_assumed": int(bytes_per_sample),
        "bytes_assumption_source": str(bytes_source),
        "estimate_mode": "conservative_pcm_equivalent"
        if conservative_for_compressed
        else "pcm_equivalent",
        "budget_source": str(budget_source),
        "budget_bytes": float(budget_bytes),
        "safety_margin": float(args.safety_margin),
        "usable_budget_bytes": float(usable_budget_bytes),
        "estimated_bytes_at_1x": float(base_bytes),
        "max_safe_stretch": float(max_safe_stretch),
        "max_safe_duration_sec": float(max_duration_sec),
        "max_safe_output_frames": int(max_frames),
        "requested_stretch": None
        if args.requested_stretch is None
        else float(args.requested_stretch),
        "requested_estimated_bytes": None if requested_bytes is None else float(requested_bytes),
        "requested_duration_sec": None
        if requested_duration_sec is None
        else float(requested_duration_sec),
        "requested_fits_budget": requested_ok,
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("pvx stretch budget estimate")
        print(f"- input: {payload['input_path']}")
        print(
            f"- input shape: {payload['input_channels']} ch, {payload['input_sample_rate']} Hz, "
            f"{payload['input_frames']} frames ({payload['input_duration_sec']:.3f} s)"
        )
        print(
            f"- output assumption: format={payload['output_format_assumed']}, "
            f"subtype={payload['subtype_assumed']}, bytes/sample={payload['bytes_per_sample_assumed']} "
            f"({payload['bytes_assumption_source']})"
        )
        print(
            f"- budget: {_format_bytes_human(payload['budget_bytes'])} "
            f"(usable {_format_bytes_human(payload['usable_budget_bytes'])} at safety-margin={payload['safety_margin']:.3f})"
        )
        print(f"- estimated size at 1.0x: {_format_bytes_human(payload['estimated_bytes_at_1x'])}")
        print(f"- max safe stretch: {payload['max_safe_stretch']:.6f}x")
        print(
            f"- max safe output: {_format_bytes_human(payload['usable_budget_bytes'])}, "
            f"{payload['max_safe_output_frames']} frames, {payload['max_safe_duration_sec']:.3f} s"
        )
        if conservative_for_compressed:
            print("- note: compressed format selected; estimate is conservative PCM-equivalent.")
        if args.requested_stretch is not None:
            fits_text = "yes" if bool(payload["requested_fits_budget"]) else "no"
            print(
                f"- requested stretch: {payload['requested_stretch']:.6f}x "
                f"(est. {_format_bytes_human(float(payload['requested_estimated_bytes'] or 0.0))}, "
                f"duration {float(payload['requested_duration_sec'] or 0.0):.3f} s) -> fits budget: {fits_text}"
            )

    if bool(args.fail_if_exceeds) and args.requested_stretch is not None and not bool(requested_ok):
        print("[stretch-budget] requested stretch exceeds usable budget", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "_consume_lucky_options",
    "_infer_output_format",
    "_parse_size_bytes",
    "_run_lucky_helper_mode",
    "_run_lucky_tool_mode",
    "run_doctor_mode",
    "run_guided_mode",
    "run_quickstart_mode",
    "run_safe_mode",
    "run_smoke_mode",
    "run_stretch_budget_mode",
    "run_transforms_mode",
]
