#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Generate advanced docs artifacts (coverage, limitations, benchmarks, citations, cookbook, architecture)."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

BENCH_DIR = DOCS_DIR / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from pvx.core.attribution import ATTRIBUTION_DOC_PATH, COPYRIGHT_NOTICE  # noqa: E402
from pvx.algorithms.registry import ALGORITHM_REGISTRY  # noqa: E402
from pvx.core import voc as voc_core  # noqa: E402
from pvx.core.audio_metrics import summarize_audio_metrics  # noqa: E402
from benchmarks.metrics import (  # noqa: E402
    envelope_correlation,
    log_spectral_distance,
    modulation_spectrum_distance,
    musical_noise_index,
    phasiness_index,
    signal_to_noise_ratio_db,
    si_sdr_db,
    spectral_convergence,
    stereo_coherence_drift,
    transient_smear_score,
)
import scripts_generate_html_docs as html_docs  # noqa: E402


def git_commit_meta() -> tuple[str, str]:
    return "[COMMIT_HASH]", "[COMMIT_DATE]"

COMMIT_HASH, COMMIT_DATE = git_commit_meta()


def generated_stamp_lines() -> list[str]:
    return [
        f"_Generated from commit `{COMMIT_HASH}` (commit date: {COMMIT_DATE})._",
        "",
    ]


def logo_lines() -> list[str]:
    return [
        "<p align=\"center\"><img src=\"../assets/pvx_logo.png\" alt=\"pvx logo\" width=\"192\" /></p>",
        "",
    ]


def attribution_section_lines() -> list[str]:
    return [
        "## Attribution",
        "",
        f"{COPYRIGHT_NOTICE} See [`{ATTRIBUTION_DOC_PATH}`](../{ATTRIBUTION_DOC_PATH}).",
        "",
    ]

def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _simple_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values: list[Any] = []
        for item in node.elts:
            value = _simple_literal(item)
            if value is None:
                return None
            values.append(value)
        return values
    return None


def _tool_name_for_path(path: Path) -> str:
    if path.name == "voc.py":
        return "pvxvoc"
    if path.parent.name == "cli":
        return "hps-pitch-track" if path.stem == "hps_pitch_track" else path.stem
    return path.name


def _iter_cli_sources() -> list[Path]:
    sources = [ROOT / "src" / "pvx" / "core" / "voc.py"]
    sources.extend(sorted((ROOT / "src" / "pvx" / "cli").glob("*.py")))
    return [p for p in sources if p.exists() and p.name != "__init__.py"]


def collect_cli_flags() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in _iter_cli_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        tool_name = _tool_name_for_path(path)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_argument":
                continue

            flags: list[str] = []
            for arg in node.args:
                lit = _string_literal(arg)
                if lit is not None:
                    flags.append(lit)
            long_flags = [f for f in flags if f.startswith("--")]
            if not long_flags:
                continue

            help_text = ""
            required = False
            default_value: Any = None
            choices: list[str] = []
            action = ""

            for kw in node.keywords:
                if kw.arg == "help":
                    lit = _string_literal(kw.value)
                    if lit is not None:
                        help_text = lit
                elif kw.arg == "required":
                    lit = _simple_literal(kw.value)
                    if isinstance(lit, bool):
                        required = lit
                elif kw.arg == "default":
                    default_value = _simple_literal(kw.value)
                elif kw.arg == "choices":
                    lit = _simple_literal(kw.value)
                    if isinstance(lit, list):
                        choices = [str(item) for item in lit]
                elif kw.arg == "action":
                    lit = _string_literal(kw.value)
                    if lit is not None:
                        action = lit

            for long_flag in long_flags:
                rows.append(
                    {
                        "tool": tool_name,
                        "source": str(path.relative_to(ROOT)),
                        "flag": long_flag,
                        "help": help_text,
                        "required": required,
                        "default": default_value,
                        "choices": choices,
                        "action": action,
                    }
                )

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["tool"]), str(row["flag"]))
        if key not in dedup:
            dedup[key] = row
    return sorted(dedup.values(), key=lambda item: (str(item["tool"]), str(item["flag"])))


def generate_cli_flags_reference() -> None:
    rows = collect_cli_flags()
    unique_flags = sorted({str(row["flag"]) for row in rows})

    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Command-Line Interface (CLI) Flags Reference")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("This file enumerates long-form CLI flags discovered from argparse declarations in canonical pvx CLI sources.")
    lines.append("")
    lines.append(f"Total tool+flag entries: **{len(rows)}**")
    lines.append(f"Total unique long flags: **{len(unique_flags)}**")
    lines.append("")
    lines.append("## Unique Long Flags")
    lines.append("")
    lines.append(", ".join(f"`{flag}`" for flag in unique_flags))
    lines.append("")

    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tool.setdefault(str(row["tool"]), []).append(row)

    for tool in sorted(by_tool):
        lines.append(f"## `{tool}`")
        lines.append("")
        lines.append("| Flag | Required | Default | Choices | Action | Description | Source |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in by_tool[tool]:
            default_text = "" if row["default"] is None else str(row["default"])
            choices_text = ", ".join(str(c) for c in row["choices"]) if row["choices"] else ""
            action_text = str(row["action"] or "")
            desc = str(row["help"] or "")
            source = str(row["source"])
            lines.append(
                f"| `{row['flag']}` | {row['required']} | `{default_text}` | `{choices_text}` | `{action_text}` | {desc} | `{source}` |"
            )
        lines.append("")

    lines.extend(attribution_section_lines())
    (DOCS_DIR / "CLI_FLAGS_REFERENCE.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    write_json(
        DOCS_DIR / "cli_flags_reference.json",
        {
            "commit": COMMIT_HASH,
            "commit_date": COMMIT_DATE,
            "entries": rows,
            "unique_flags": unique_flags,
        },
    )


GROUP_LIMITS: dict[str, dict[str, str]] = {
    "time_scale_and_pitch_core": {
        "assumption": "Frames are locally quasi-stationary and harmonic evolution is reasonably smooth.",
        "failure": "High-ratio stretch can introduce phasiness and blurred transients.",
        "avoid": "Avoid for extreme percussive-only material when attack realism is critical.",
    },
    "pitch_detection_and_tracking": {
        "assumption": "F0 evidence is strong in the selected analysis band and frame size.",
        "failure": "Octave errors and voicing flips under heavy noise/polyphony.",
        "avoid": "Avoid as the sole control signal for dense polyphonic mixtures.",
    },
    "retune_and_intonation": {
        "assumption": "Detected notes map cleanly to intended tonal center/scale.",
        "failure": "Over-correction can flatten expressive vibrato or slides.",
        "avoid": "Avoid aggressive correction when preserving natural micro-intonation is required.",
    },
    "spectral_time_frequency_transforms": {
        "assumption": "Transform parameterization matches target time-frequency structure.",
        "failure": "Incorrect parameterization can smear events or over-fragment spectra.",
        "avoid": "Avoid default settings for highly nonstationary signals without tuning.",
    },
    "separation_and_decomposition": {
        "assumption": "Sources have partially separable spectral or statistical structure.",
        "failure": "Component bleeding and musical noise under overlap or model mismatch.",
        "avoid": "Avoid expecting perfect stems from strongly correlated or co-modulated sources.",
    },
    "denoise_and_restoration": {
        "assumption": "Noise/artifacts are distinguishable from desired signal statistics.",
        "failure": "Over-reduction can remove detail and create modulation artifacts.",
        "avoid": "Avoid high reduction settings on sparse acoustic sources without auditioning.",
    },
    "dereverb_and_room_correction": {
        "assumption": "Late reverberation is separable from direct content under chosen model.",
        "failure": "Speech/music clarity can drop if early reflections are over-suppressed.",
        "avoid": "Avoid strong dereverb when room character is part of artistic intent.",
    },
    "dynamics_and_loudness": {
        "assumption": "Program dynamics fit compressor/limiter time constants and thresholds.",
        "failure": "Pumping, breathing, or overs if thresholds and release are mis-set.",
        "avoid": "Avoid applying multiple strong dynamics stages without gain staging checks.",
    },
    "creative_spectral_effects": {
        "assumption": "Spectral manipulations are desired even with timbral coloration.",
        "failure": "Can introduce intentional but strong coloration or temporal artifacts.",
        "avoid": "Avoid for transparent restoration/mastering paths.",
    },
    "granular_and_modulation": {
        "assumption": "Grain and modulation rates are musically matched to source texture.",
        "failure": "Incoherent grain scheduling can produce choppiness or blur.",
        "avoid": "Avoid dense granular settings on speech intelligibility-critical content.",
    },
    "analysis_qa_and_automation": {
        "assumption": "Feature extraction settings align with domain (speech vs music etc.).",
        "failure": "False positives/negatives under domain shift.",
        "avoid": "Avoid treating single metrics as absolute quality verdicts.",
    },
    "spatial_and_multichannel": {
        "assumption": "Channel geometry/order and timing metadata are correct.",
        "failure": "Spatial collapse, combing, or localization bias from misalignment.",
        "avoid": "Avoid blind spatial processing when channel order/calibration is unknown.",
    },
}

KEYWORD_LIMITS: list[tuple[str, str, str, str]] = [
    (
        "neural",
        "Model priors assume training-like signal statistics.",
        "Generalization gaps can produce unstable artifacts.",
        "Avoid fully unattended use on out-of-domain material.",
    ),
    (
        "retune",
        "Pitch trajectory estimates should be continuous enough for retuning.",
        "Fast F0 jumps can cause audible stepping.",
        "Avoid high-strength retune on breath/noise segments.",
    ),
    (
        "phase_vocoder",
        "Phase continuity assumptions hold best for moderate stretch ratios.",
        "Extreme settings increase phasiness/transient blur risk.",
        "Avoid very large stretch+pitch shifts without transient controls.",
    ),
    (
        "denoise",
        "Noise model should be representative of observed noise floor.",
        "Mismatched noise estimate leaves residue or damages detail.",
        "Avoid static settings on rapidly varying nonstationary noise.",
    ),
    (
        "dereverb",
        "Reverberation tail is assumed more diffuse than direct content.",
        "Over-suppression can thin tonal body and ambience.",
        "Avoid for intentionally wet effects unless mix preservation is planned.",
    ),
]


def _unique_join(parts: list[str]) -> str:
    out: list[str] = []
    for part in parts:
        text = part.strip()
        if text and text not in out:
            out.append(text)
    return " ".join(out)


def generate_algorithm_limitations() -> None:
    grouped: dict[str, list[dict[str, str]]] = {}

    for algorithm_id, meta in sorted(ALGORITHM_REGISTRY.items()):
        folder, slug = algorithm_id.split(".", 1)
        name = str(meta.get("name", slug))
        seed = GROUP_LIMITS.get(
            folder,
            {
                "assumption": "Algorithm assumptions depend on source-domain and parameter tuning.",
                "failure": "Model mismatch can reduce quality or stability.",
                "avoid": "Avoid default settings for mission-critical rendering without validation.",
            },
        )

        assumptions = [seed["assumption"]]
        failures = [seed["failure"]]
        avoids = [seed["avoid"]]

        text = f"{algorithm_id} {name}".lower()
        for keyword, a_text, f_text, w_text in KEYWORD_LIMITS:
            if keyword in text:
                assumptions.append(a_text)
                failures.append(f_text)
                avoids.append(w_text)

        if "mono" in text or "monophonic" in text:
            assumptions.append("Monophonic pitch contours are expected to dominate frame-wise estimates.")
            failures.append("Polyphonic overlap can trigger octave and target-selection instability.")
            avoids.append("Avoid as a single-stage processor for dense harmony stacks.")

        grouped.setdefault(folder, []).append(
            {
                "algorithm_id": algorithm_id,
                "name": name,
                "assumptions": _unique_join(assumptions),
                "failure_modes": _unique_join(failures),
                "when_not_to_use": _unique_join(avoids),
                "module": str(meta.get("module", "")),
                "theme": str(meta.get("theme", folder)),
            }
        )

    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Algorithm Limitations and Applicability")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("This document summarizes assumptions, likely failure modes, and practical exclusion cases for each algorithm group and algorithm module.")
    lines.append("")

    lines.append("## Group-Level Summary")
    lines.append("")
    lines.append("| Group | Assumptions | Failure Modes | When Not To Use |")
    lines.append("| --- | --- | --- | --- |")
    for group in sorted(grouped):
        seed = GROUP_LIMITS.get(group, {})
        lines.append(
            "| `{group}` | {a} | {f} | {w} |".format(
                group=group,
                a=seed.get("assumption", "Algorithm-specific."),
                f=seed.get("failure", "Algorithm-specific."),
                w=seed.get("avoid", "Algorithm-specific."),
            )
        )
    lines.append("")

    for group in sorted(grouped):
        lines.append(f"## `{group}`")
        lines.append("")
        lines.append("| Algorithm ID | Assumptions | Failure Modes | When Not To Use |")
        lines.append("| --- | --- | --- | --- |")
        for row in grouped[group]:
            lines.append(
                "| `{algorithm_id}` | {assumptions} | {failure_modes} | {when_not_to_use} |".format(
                    algorithm_id=row["algorithm_id"],
                    assumptions=row["assumptions"],
                    failure_modes=row["failure_modes"],
                    when_not_to_use=row["when_not_to_use"],
                )
            )
        lines.append("")

    lines.extend(attribution_section_lines())
    (DOCS_DIR / "ALGORITHM_LIMITATIONS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    write_json(
        DOCS_DIR / "algorithm_limitations.json",
        {
            "commit": COMMIT_HASH,
            "commit_date": COMMIT_DATE,
            "groups": grouped,
            "group_limits": GROUP_LIMITS,
        },
    )


def generate_cookbook() -> None:
    recipes: list[dict[str, str]] = [
        {
            "category": "Phase-vocoder core",
            "title": "Moderate vocal stretch with formant preservation",
            "command": "pvx voc vocal.wav --time-stretch 1.15 --pitch-mode formant-preserving --output-dir out --suffix _pv",
            "why": "Retains speech-like vowel envelope while stretching timing.",
        },
        {
            "category": "Phase-vocoder core",
            "title": "Independent cents retune",
            "command": "pvx voc lead.wav --pitch-shift-cents -23 --time-stretch 1.0 --output-dir out --suffix _cents",
            "why": "Applies precise microtonal offset without tempo change.",
        },
        {
            "category": "Phase-vocoder core",
            "title": "Extreme stretch with multistage strategy",
            "command": "pvx voc ambience.wav --target-duration 600 --ambient-preset --n-fft 16384 --win-length 16384 --hop-size 2048 --window kaiser --kaiser-beta 18 --output-dir out --suffix _ambient600x",
            "why": "PaulStretch-style ambient profile for very large ratios using stochastic phase and onset time-credit controls.",
        },
        {
            "category": "Phase-vocoder core",
            "title": "Ultra-smooth speech stretch (600x)",
            "command": "pvx voc speech.wav --target-duration 600 --stretch-mode standard --phase-engine propagate --phase-locking identity --n-fft 8192 --win-length 8192 --hop-size 256 --window hann --normalize peak --peak-dbfs -1 --compressor-threshold-db -30 --compressor-ratio 2.0 --compressor-attack-ms 25 --compressor-release-ms 250 --compressor-makeup-db 4 --limiter-threshold 0.95 --output-dir out --suffix _speech600x",
            "why": "Prefers continuity and intelligibility over texture animation; avoids choppy stochastic artifacts on speech sources.",
        },
        {
            "category": "Phase-vocoder core",
            "title": "Auto-profile plan preview",
            "command": "pvx voc input.wav --auto-profile --auto-transform --explain-plan",
            "why": "Prints the resolved profile/config plan before long renders.",
        },
        {
            "category": "Phase-vocoder core",
            "title": "Multi-resolution fusion stretch",
            "command": "pvx voc input.wav --multires-fusion --multires-ffts 1024,2048,4096 --multires-weights 0.2,0.35,0.45 --time-stretch 1.4 --output-dir out --suffix _multires",
            "why": "Blends several FFT scales to reduce single-resolution bias on complex program material.",
        },
        {
            "category": "Phase-vocoder core",
            "title": "Checkpointed long render with manifest",
            "command": "pvx voc long.wav --time-stretch 12 --auto-segment-seconds 0.5 --checkpoint-dir checkpoints --manifest-json reports/run_manifest.json --output-dir out --suffix _long",
            "why": "Caches segment renders for resume workflows and writes run metadata for reproducibility.",
        },
        {
            "category": "Transform selection",
            "title": "Default production backend (FFT + transient protection)",
            "command": "pvx voc mix.wav --transform fft --time-stretch 1.07 --transient-preserve --phase-locking identity --output-dir out --suffix _fft",
            "why": "Use when you need the fastest and most stable general-purpose phase-vocoder path.",
        },
        {
            "category": "Transform selection",
            "title": "Reference Fourier baseline using explicit DFT mode",
            "command": "pvx voc tone_sweep.wav --transform dft --time-stretch 1.00 --pitch-shift-semitones 0 --output-dir out --suffix _dft_ref",
            "why": "Useful for parity checks and controlled transform-comparison experiments.",
        },
        {
            "category": "Transform selection",
            "title": "Prime-size frame experiment with CZT backend",
            "command": "pvx voc archival_take.wav --transform czt --n-fft 1531 --win-length 1531 --hop-size 382 --time-stretch 1.03 --output-dir out --suffix _czt",
            "why": "Alternative numerical path for awkward/prime frame sizes when validating edge cases.",
        },
        {
            "category": "Transform selection",
            "title": "DCT timbral compaction for smooth harmonic material",
            "command": "pvx voc strings.wav --transform dct --pitch-shift-cents -18 --soft-clip-level 0.95 --output-dir out --suffix _dct",
            "why": "Real-basis coefficients can emphasize envelope-like structure for creative reshaping.",
        },
        {
            "category": "Transform selection",
            "title": "DST odd-symmetry color pass",
            "command": "pvx voc snare_loop.wav --transform dst --time-stretch 0.92 --phase-locking off --output-dir out --suffix _dst",
            "why": "Provides an alternate real-basis artifact profile useful for creative percussive processing.",
        },
        {
            "category": "Transform selection",
            "title": "Hartley real-basis exploratory render",
            "command": "pvx voc synth_pad.wav --transform hartley --time-stretch 1.30 --pitch-shift-semitones 3 --output-dir out --suffix _hartley",
            "why": "Compares Hartley-domain behavior against complex FFT phase-vocoder output.",
        },
        {
            "category": "Transform selection",
            "title": "A/B sweep of transform backends from shell loop",
            "command": "for t in fft dft czt dct dst hartley; do pvx voc voice.wav --transform \"$t\" --time-stretch 1.1 --output-dir out --suffix \"_$t\"; done",
            "why": "Fast listening workflow for selecting the least-artifact transform on your source.",
        },
        {
            "category": "Microtonal",
            "title": "Custom cents map retune",
            "command": "pvx retune vox.wav --root 60 --scale-cents 0,90,204,294,408,498,612,702,816,906,1020,1110 --strength 0.8 --output-dir out",
            "why": "Maps incoming notes to a custom 12-degree microtonal scale.",
        },
        {
            "category": "Microtonal",
            "title": "Conform CSV with per-segment ratios",
            "command": "pvx conform solo.wav map_conform.csv --pitch-mode ratio --output-dir out --suffix _conform",
            "why": "Applies timeline-specific time and pitch trajectories from CSV.",
        },
        {
            "category": "Pipelines",
            "title": "Time-stretch -> denoise -> dereverb in one pipe",
            "command": "pvx voc input.wav --time-stretch 1.25 --stdout | pvx denoise - --reduction-db 10 --stdout | pvx deverb - --strength 0.45 --output-dir out --suffix _clean",
            "why": "Single-pass CLI chain for serial DSP in Unix pipes.",
        },
        {
            "category": "Pipelines",
            "title": "Morph -> formant -> unison",
            "command": "pvx morph a.wav b.wav -o - | pvx formant - --mode preserve --stdout | pvx unison - --voices 5 --detune-cents 8 --output-dir out --suffix _morph_stack",
            "why": "Builds a richer timbre chain with no intermediate files.",
        },
        {
            "category": "Pipelines",
            "title": "Pitch-follow sidechain map (A controls B)",
            "command": "pvx pitch-track A.wav | pvx voc B.wav --pitch-follow-stdin --pitch-conf-min 0.75 --pitch-lowconf-mode hold --time-stretch-factor 1.0 --output output.wav",
            "why": "Tracks F0 contour from source A and applies it as a dynamic pitch-ratio control map on source B.",
        },
        {
            "category": "Mastering",
            "title": "Integrated loudness targeting with limiter",
            "command": "pvx voc mix.wav --time-stretch 1.0 --target-lufs -14 --compressor-threshold-db -20 --compressor-ratio 3 --limiter-threshold 0.98 --output-dir out --suffix _master",
            "why": "Combines dynamics and loudness controls in shared mastering chain.",
        },
        {
            "category": "Mastering",
            "title": "Soft clip and hard safety ceiling",
            "command": "pvx harmonize bus.wav --intervals 0,7,12 --mix 0.35 --soft-clip-level 0.92 --soft-clip-type tanh --hard-clip-level 0.99 --output-dir out",
            "why": "Adds saturation while enforcing a strict final peak ceiling.",
        },
        {
            "category": "Batch",
            "title": "Batch stretch over folder",
            "command": "pvx voc stems/*.wav --time-stretch 1.08 --output-dir out/stems --overwrite",
            "why": "Applies consistent transform to many files with one command.",
        },
        {
            "category": "Batch",
            "title": "Dry-run output validation",
            "command": "pvx denoise takes/*.wav --reduction-db 8 --dry-run --output-dir out/preview",
            "why": "Checks filename resolution and collisions without writing audio.",
        },
        {
            "category": "Automation",
            "title": "A/B report generation",
            "command": "python3 scripts/scripts_ab_compare.py --input mix.wav --a-args \"--time-stretch 1.1 --transform fft\" --b-args \"--time-stretch 1.1 --transform dct\" --out-dir reports/ab --name fft_vs_dct",
            "why": "Creates JSON/Markdown objective reports for fast algorithm and parameter comparisons.",
        },
        {
            "category": "Automation",
            "title": "Benchmark matrix sweep",
            "command": "python3 scripts/scripts_benchmark_matrix.py --input mix.wav --transforms fft,dft,dct --windows hann,kaiser --n-ffts 1024,2048 --devices cpu --out-dir reports/bench",
            "why": "Produces reproducible CSV/JSON runtime matrices across transform/window/FFT combinations.",
        },
        {
            "category": "Automation",
            "title": "Quality regression check",
            "command": "python3 scripts/scripts_quality_regression.py --input mix.wav --output out/reg.wav --render-args \"--time-stretch 1.2 --transform fft\" --baseline-json reports/baseline.json --report-json reports/regression.json",
            "why": "Compares current renders against baseline objective metrics with configurable tolerances.",
        },
        {
            "category": "Spatial",
            "title": "VBAP adaptive panning via algorithm dispatcher",
            "command": "python3 -m pvx.algorithms.spatial_and_multichannel.imaging_and_panning.vbap_adaptive_panning input.wav --output-channels 6 --azimuth-deg 35 --width 0.8 --output out/vbap.wav",
            "why": "Demonstrates algorithm-level spatial module invocation.",
        },
        {
            "category": "Analysis/QA",
            "title": "Quality metrics on processed speech",
            "command": "python3 -m pvx.algorithms.analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics clean.wav noisy.wav --output out/qa.json",
            "why": "Collects objective quality indicators for regression tracking.",
        },
    ]

    by_cat: dict[str, list[dict[str, str]]] = {}
    for recipe in recipes:
        by_cat.setdefault(recipe["category"], []).append(recipe)

    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Pipeline Cookbook")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("Curated one-line workflows for practical chaining, mastering, microtonal processing, and batch operation.")
    lines.append("")

    for category in sorted(by_cat):
        lines.append(f"## {category}")
        lines.append("")
        for recipe in by_cat[category]:
            lines.append(f"### {recipe['title']}")
            lines.append("")
            lines.append("```bash")
            lines.append(recipe["command"])
            lines.append("```")
            lines.append("")
            lines.append(f"Why: {recipe['why']}")
            lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Use `--stdout`/`-` to chain tools without intermediate files.")
    lines.append("- Add `--quiet` for script-driven runs; use default verbosity for live progress bars.")
    lines.append("- For production mastering, validate true peaks and loudness after all nonlinear stages.")
    lines.append("")

    lines.extend(attribution_section_lines())
    (DOCS_DIR / "PIPELINE_COOKBOOK.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    write_json(
        DOCS_DIR / "pipeline_cookbook.json",
        {
            "commit": COMMIT_HASH,
            "commit_date": COMMIT_DATE,
            "recipes": recipes,
        },
    )


def generate_architecture_doc() -> None:
    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Architecture")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("System architecture for runtime processing, algorithm dispatch, and documentation pipelines.")
    lines.append("")

    lines.append("## 1. Runtime and CLI Flow")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    lines.append("  A[User CLI Command] --> B[src/pvx/cli or pvxvoc parser]")
    lines.append("  B --> C[Runtime Selection: auto/cpu/cuda]")
    lines.append("  C --> D[Shared IO + Mastering Chain]")
    lines.append("  D --> E[Core DSP in src/pvx/core/voc.py]")
    lines.append("  E --> F[Output Writer / stdout stream]")
    lines.append("```")
    lines.append("")

    lines.append("## 2. Algorithm Registry and Dispatch")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    lines.append("  R[src/pvx/algorithms/registry.py] --> B[src/pvx/algorithms/base.py]")
    lines.append("  B --> M1[time_scale_and_pitch_core/*]")
    lines.append("  B --> M2[retune_and_intonation/*]")
    lines.append("  B --> M3[dynamics_and_loudness/*]")
    lines.append("  B --> M4[spatial_and_multichannel/*]")
    lines.append("```")
    lines.append("")

    lines.append("## 3. Documentation Build Graph")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    lines.append("  G1[scripts/scripts_generate_python_docs.py] --> D[docs/*]")
    lines.append("  G2[scripts/scripts_generate_theory_docs.py] --> D")
    lines.append("  G3[scripts/scripts_generate_docs_extras.py] --> D")
    lines.append("  G4[scripts/scripts_generate_html_docs.py] --> H[docs/html/*]")
    lines.append("  D --> H")
    lines.append("```")
    lines.append("")

    lines.append("## 4. CI + Pages")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    lines.append("  PR[Push / PR] --> CI[Doc and test workflow]")
    lines.append("  CI --> S[Generation + drift checks]")
    lines.append("  S --> T[Unit tests + docs coverage tests]")
    lines.append("  T --> P[GitHub Pages deploy workflow]")
    lines.append("  P --> SITE[Published docs/html site]")
    lines.append("```")
    lines.append("")

    lines.extend(attribution_section_lines())
    (DOCS_DIR / "ARCHITECTURE.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ensure_2d(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    if arr.size == 0:
        return np.zeros((0, 1), dtype=np.float64)
    return np.asarray(arr.reshape(arr.shape[0], -1), dtype=np.float64)


def _align_audio_pair(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = _ensure_2d(reference)
    cand = _ensure_2d(candidate)
    n = min(ref.shape[0], cand.shape[0])
    ch = min(ref.shape[1], cand.shape[1])
    if n <= 0 or ch <= 0:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0, 1), dtype=np.float64)
    return ref[:n, :ch], cand[:n, :ch]


def _finite_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(np.asarray(finite, dtype=np.float64))) if finite else float("nan")


def _finite_max(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.max(np.asarray(finite, dtype=np.float64))) if finite else float("nan")


def _make_benchmark_cases(sample_rate: int = 48000, seconds: float = 4.0) -> list[dict[str, Any]]:
    n = int(round(float(sample_rate) * float(seconds)))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    rng = np.random.default_rng(0)

    tonal = (
        0.42 * np.sin(2.0 * math.pi * 110.0 * t)
        + 0.18 * np.sin(2.0 * math.pi * 223.0 * t + 0.3)
        + 0.12 * np.sin(2.0 * math.pi * 451.0 * t + 0.7)
        + 0.08 * np.sin(2.0 * math.pi * 1780.0 * t)
    )
    tonal *= np.linspace(0.4, 1.0, n, dtype=np.float64)
    tonal = np.clip(tonal, -0.98, 0.98)

    speech = np.zeros(n, dtype=np.float64)
    f0 = 125.0
    for harmonic in range(1, 40):
        fk = harmonic * f0
        if fk >= sample_rate * 0.5:
            break
        env = np.exp(-0.5 * ((fk - 700.0) / 250.0) ** 2) + 0.75 * np.exp(-0.5 * ((fk - 1850.0) / 330.0) ** 2)
        speech += (env / max(1.0, float(harmonic))) * np.sin(2.0 * math.pi * fk * t)
    speech *= 0.32 / max(1e-9, float(np.max(np.abs(speech))))
    for pos in (0.22, 0.58, 0.91, 1.33, 2.10, 2.88, 3.40):
        idx = int(round(pos * sample_rate))
        if 0 <= idx < speech.size:
            speech[idx] += 0.6
    speech = np.clip(speech, -0.98, 0.98)

    drums = np.zeros(n, dtype=np.float64)
    hit_positions = np.linspace(0.15, max(0.15, seconds - 0.15), 12)
    for i, pos in enumerate(hit_positions):
        h = int(round(float(pos) * sample_rate))
        if h >= drums.size:
            continue
        length = min(int(round(0.06 * sample_rate)), drums.size - h)
        if length <= 4:
            continue
        decay = np.exp(-np.linspace(0.0, 8.0, length))
        tone = np.sin(2.0 * math.pi * (95.0 + 14.0 * (i % 4)) * np.arange(length) / sample_rate)
        noise = rng.standard_normal(length)
        burst = (0.65 * tone + 0.25 * noise) * decay
        drums[h : h + length] += burst
    drums = np.clip(drums, -0.98, 0.98)
    drums_stereo = np.stack([drums, np.roll(drums, 21)], axis=1)
    drums_stereo[:, 1] *= 0.95

    chirp_left = 0.26 * np.sin(2.0 * math.pi * (80.0 * t + 220.0 * t * t))
    chirp_right = 0.25 * np.sin(2.0 * math.pi * (82.0 * t + 216.0 * t * t) + 0.42)
    texture = rng.standard_normal(n)
    texture = np.convolve(texture, np.ones(11, dtype=np.float64) / 11.0, mode="same")
    ambience = np.stack([chirp_left + 0.10 * texture, chirp_right + 0.10 * np.roll(texture, 33)], axis=1)
    ambience = np.clip(ambience, -0.98, 0.98)

    cases: list[dict[str, Any]] = [
        {
            "name": "tonal_ramp_mono",
            "description": "Deterministic harmonic blend with slow ramp.",
            "audio": tonal[:, None],
        },
        {
            "name": "speech_like_mono",
            "description": "Speech-like harmonic formants with sparse glottal-like impulses.",
            "audio": speech[:, None],
        },
        {
            "name": "transient_drums_stereo",
            "description": "Percussive burst train for onset retention and stereo timing drift.",
            "audio": drums_stereo,
        },
        {
            "name": "chirp_texture_stereo",
            "description": "Wideband chirp-plus-texture to stress spectral and phase stability.",
            "audio": ambience,
        },
    ]

    for case in cases:
        audio = _ensure_2d(np.asarray(case["audio"], dtype=np.float64))
        case["audio"] = np.clip(audio, -0.999, 0.999)
        case["channels"] = int(audio.shape[1])
        case["duration_seconds"] = float(audio.shape[0] / max(1, sample_rate))
    return cases


def _quality_score_0_100(metrics: dict[str, Any]) -> float:
    def _higher_is_better(key: str, floor: float, span: float) -> float | None:
        value = float(metrics.get(key, math.nan))
        if not np.isfinite(value):
            return None
        return float(np.clip((value - floor) / max(1e-9, span), 0.0, 1.0))

    def _lower_is_better(key: str, scale: float) -> float | None:
        value = float(metrics.get(key, math.nan))
        if not np.isfinite(value):
            return None
        return float(1.0 - np.clip(value / max(1e-9, scale), 0.0, 1.0))

    parts: list[float] = []
    for score in (
        _higher_is_better("snr_vs_input_db", 20.0, 70.0),
        _higher_is_better("si_sdr_db", 20.0, 70.0),
        _higher_is_better("envelope_correlation", 0.85, 0.15),
        _lower_is_better("log_spectral_distance_db", 6.0),
        _lower_is_better("modulation_spectrum_distance", 0.30),
        _lower_is_better("transient_smear_score", 0.08),
        _lower_is_better("spectral_convergence", 0.20),
        _lower_is_better("phasiness_index", 0.20),
        _lower_is_better("musical_noise_index", 0.30),
    ):
        if score is not None:
            parts.append(score)

    coherence = _lower_is_better("stereo_coherence_drift", 0.20)
    if coherence is not None:
        parts.append(coherence)

    if not parts:
        return float("nan")
    return float(100.0 * np.mean(np.asarray(parts, dtype=np.float64)))


def _compute_roundtrip_metrics(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> dict[str, float]:
    ref, cand = _align_audio_pair(reference, candidate)
    ref_mono = np.mean(ref, axis=1) if ref.size else np.zeros(0, dtype=np.float64)
    cand_mono = np.mean(cand, axis=1) if cand.size else np.zeros(0, dtype=np.float64)
    ref_summary = summarize_audio_metrics(ref, int(sample_rate))
    cand_summary = summarize_audio_metrics(cand, int(sample_rate))

    coherence = float(stereo_coherence_drift(ref, cand)) if ref.shape[1] >= 2 and cand.shape[1] >= 2 else float("nan")

    return {
        "snr_vs_input_db": float(signal_to_noise_ratio_db(ref_mono, cand_mono)),
        "si_sdr_db": float(si_sdr_db(ref_mono, cand_mono)),
        "log_spectral_distance_db": float(log_spectral_distance(ref_mono, cand_mono)),
        "modulation_spectrum_distance": float(modulation_spectrum_distance(ref_mono, cand_mono)),
        "transient_smear_score": float(transient_smear_score(ref_mono, cand_mono)),
        "envelope_correlation": float(envelope_correlation(ref_mono, cand_mono)),
        "spectral_convergence": float(spectral_convergence(ref_mono, cand_mono)),
        "stereo_coherence_drift": coherence,
        "phasiness_index": float(phasiness_index(ref_mono, cand_mono)),
        "musical_noise_index": float(musical_noise_index(ref_mono, cand_mono)),
        "rms_delta_db": float(cand_summary.rms_dbfs - ref_summary.rms_dbfs),
        "peak_delta_db": float(cand_summary.peak_dbfs - ref_summary.peak_dbfs),
        "crest_delta_db": float(cand_summary.crest_db - ref_summary.crest_db),
        "centroid_delta_hz": float(cand_summary.spectral_centroid_hz - ref_summary.spectral_centroid_hz),
        "bandwidth95_delta_hz": float(cand_summary.bandwidth_95_hz - ref_summary.bandwidth_95_hz),
        "zcr_delta": float(cand_summary.zcr - ref_summary.zcr),
        "clip_delta_pct": float(cand_summary.clip_percent - ref_summary.clip_percent),
    }


def _roundtrip_stft_istft(audio: np.ndarray, config: voc_core.VocoderConfig) -> np.ndarray:
    arr = _ensure_2d(audio)
    out = np.zeros_like(arr, dtype=np.float64)
    for ch in range(arr.shape[1]):
        spectrum = voc_core.stft(arr[:, ch], config)
        reconstructed = np.asarray(
            voc_core.istft(spectrum, config, expected_length=int(arr.shape[0])),
            dtype=np.float64,
        ).reshape(-1)
        if reconstructed.size < arr.shape[0]:
            reconstructed = np.pad(reconstructed, (0, arr.shape[0] - reconstructed.size))
        out[:, ch] = reconstructed[: arr.shape[0]]
    return out


def _benchmark_backend_case(
    label: str,
    device: str,
    case: dict[str, Any],
    config: voc_core.VocoderConfig,
    reference: np.ndarray | None,
    sample_rate: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    signal = _ensure_2d(np.asarray(case["audio"], dtype=np.float64))
    payload: dict[str, Any] = {
        "backend": label,
        "signal": str(case["name"]),
        "description": str(case["description"]),
        "channels": int(signal.shape[1]),
        "duration_seconds": float(signal.shape[0] / max(1, sample_rate)),
        "device_request": device,
        "status": "ok",
        "reason": "",
        "notes": "",
    }

    try:
        voc_core.configure_runtime(device=device, cuda_device=0, verbose=False)
    except Exception as exc:
        payload.update({"status": "unavailable", "reason": str(exc), "notes": str(exc)})
        return payload, None

    tracemalloc.start()
    start = time.perf_counter()
    try:
        reconstructed = _roundtrip_stft_istft(signal, config)
    except Exception as exc:
        tracemalloc.stop()
        payload.update({"status": "failed", "reason": str(exc), "notes": str(exc)})
        return payload, None

    elapsed_s = float(time.perf_counter() - start)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_ms = 1000.0 * elapsed_s
    x_real_time = payload["duration_seconds"] / max(1e-9, elapsed_s)
    metrics = _compute_roundtrip_metrics(signal, reconstructed, sample_rate=sample_rate)
    payload.update(metrics)
    payload.update(
        {
            "elapsed_ms": float(elapsed_ms),
            "x_real_time": float(x_real_time),
            "peak_host_memory_mb": float(peak_bytes) / (1024.0 * 1024.0),
            "runtime_active_device": voc_core.runtime_config().active_device,
            "runtime_fallback_reason": voc_core.runtime_config().fallback_reason,
        }
    )

    if reference is not None:
        ref_cmp, cand_cmp = _align_audio_pair(reference, reconstructed)
        ref_mono = np.mean(ref_cmp, axis=1) if ref_cmp.size else np.zeros(0, dtype=np.float64)
        cand_mono = np.mean(cand_cmp, axis=1) if cand_cmp.size else np.zeros(0, dtype=np.float64)
        payload.update(
            {
                "snr_vs_cpu_db": float(signal_to_noise_ratio_db(ref_mono, cand_mono)),
                "spectral_distance_vs_cpu_db": float(log_spectral_distance(ref_mono, cand_mono)),
                "modulation_vs_cpu": float(modulation_spectrum_distance(ref_mono, cand_mono)),
                "max_abs_error_vs_cpu": float(np.max(np.abs(ref_cmp - cand_cmp))) if ref_cmp.size else float("nan"),
            }
        )
        if ref_cmp.shape[1] >= 2 and cand_cmp.shape[1] >= 2:
            payload["stereo_coherence_drift_vs_cpu"] = float(stereo_coherence_drift(ref_cmp, cand_cmp))
        else:
            payload["stereo_coherence_drift_vs_cpu"] = float("nan")

    if voc_core.runtime_config().active_device == "cuda" and getattr(voc_core, "cp", None) is not None:
        try:
            mem_pool = voc_core.cp.get_default_memory_pool()
            payload["gpu_pool_used_mb"] = float(mem_pool.used_bytes()) / (1024.0 * 1024.0)
            mem_pool.free_all_blocks()
        except Exception:
            pass

    payload["quality_score"] = _quality_score_0_100(payload)
    return payload, reconstructed


def _summarize_backend_rows(backend: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    cases_total = int(len(rows))
    ok_rows = [row for row in rows if str(row.get("status", "")) == "ok"]
    cases_ok = int(len(ok_rows))

    if cases_total == 0:
        status = "unavailable"
    elif cases_ok == cases_total:
        status = "ok"
    elif cases_ok > 0:
        status = "partial"
    elif any(str(row.get("status", "")) == "unavailable" for row in rows):
        status = "unavailable"
    else:
        status = "failed"

    reasons = sorted({str(row.get("reason", "")).strip() for row in rows if str(row.get("reason", "")).strip()})
    note = "; ".join(reasons[:3])

    def _mean_metric(key: str) -> float:
        vals: list[float] = []
        for row in ok_rows:
            try:
                vals.append(float(row.get(key, math.nan)))
            except Exception:
                continue
        return _finite_mean(vals)

    def _max_metric(key: str) -> float:
        vals: list[float] = []
        for row in ok_rows:
            try:
                vals.append(float(row.get(key, math.nan)))
            except Exception:
                continue
        return _finite_max(vals)

    return {
        "backend": backend,
        "status": status,
        "cases_total": cases_total,
        "cases_ok": cases_ok,
        "reason": note,
        "notes": note,
        "elapsed_ms_mean": _mean_metric("elapsed_ms"),
        "peak_host_memory_mb_max": _max_metric("peak_host_memory_mb"),
        "x_real_time_mean": _mean_metric("x_real_time"),
        "snr_vs_input_db_mean": _mean_metric("snr_vs_input_db"),
        "si_sdr_db_mean": _mean_metric("si_sdr_db"),
        "log_spectral_distance_db_mean": _mean_metric("log_spectral_distance_db"),
        "modulation_spectrum_distance_mean": _mean_metric("modulation_spectrum_distance"),
        "transient_smear_score_mean": _mean_metric("transient_smear_score"),
        "envelope_correlation_mean": _mean_metric("envelope_correlation"),
        "spectral_convergence_mean": _mean_metric("spectral_convergence"),
        "stereo_coherence_drift_mean": _mean_metric("stereo_coherence_drift"),
        "phasiness_index_mean": _mean_metric("phasiness_index"),
        "musical_noise_index_mean": _mean_metric("musical_noise_index"),
        "quality_score_mean": _mean_metric("quality_score"),
        "snr_vs_cpu_db_mean": _mean_metric("snr_vs_cpu_db"),
        "spectral_distance_vs_cpu_db_mean": _mean_metric("spectral_distance_vs_cpu_db"),
        # Backward-compatible aliases used in existing HTML render logic.
        "elapsed_ms": _mean_metric("elapsed_ms"),
        "peak_host_memory_mb": _max_metric("peak_host_memory_mb"),
        "snr_vs_input_db": _mean_metric("snr_vs_input_db"),
        "spectral_distance_vs_input_db": _mean_metric("log_spectral_distance_db"),
        "snr_vs_cpu_db": _mean_metric("snr_vs_cpu_db"),
        "spectral_distance_vs_cpu_db": _mean_metric("spectral_distance_vs_cpu_db"),
    }


def generate_benchmarks(run_benchmarks: bool) -> None:
    bench_json_path = BENCH_DIR / "latest.json"

    if run_benchmarks or not bench_json_path.exists():
        sample_rate = 48000
        duration_seconds = 4.0
        cases = _make_benchmark_cases(sample_rate=sample_rate, seconds=duration_seconds)
        config = voc_core.VocoderConfig(
            n_fft=2048,
            win_length=2048,
            hop_size=512,
            window="hann",
            center=True,
            phase_locking="none",
            transient_preserve=False,
            transient_threshold=1.5,
            kaiser_beta=14.0,
        )

        is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
        backend_specs: list[tuple[str, str, bool, str]] = [
            ("cpu", "cpu", True, ""),
            ("cuda", "cuda", True, ""),
            (
                "apple_silicon_native_cpu",
                "cpu",
                bool(is_apple_silicon),
                "Host platform is not Apple Silicon (Darwin arm64).",
            ),
        ]

        case_runs: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        cpu_reference_by_case: dict[str, np.ndarray] = {}

        for backend, device, enabled, disabled_reason in backend_specs:
            backend_rows: list[dict[str, Any]] = []
            for case in cases:
                if not enabled:
                    backend_rows.append(
                        {
                            "backend": backend,
                            "signal": str(case["name"]),
                            "description": str(case["description"]),
                            "channels": int(case["channels"]),
                            "duration_seconds": float(case["duration_seconds"]),
                            "device_request": device,
                            "status": "unavailable",
                            "reason": str(disabled_reason),
                            "notes": str(disabled_reason),
                        }
                    )
                    continue

                reference = cpu_reference_by_case.get(str(case["name"])) if backend != "cpu" else None
                row, reconstructed = _benchmark_backend_case(
                    backend,
                    device,
                    case,
                    config,
                    reference,
                    sample_rate=sample_rate,
                )
                backend_rows.append(row)
                if backend == "cpu" and row.get("status") == "ok" and reconstructed is not None:
                    cpu_reference_by_case[str(case["name"])] = np.asarray(reconstructed, dtype=np.float64)

            case_runs.extend(backend_rows)
            runs.append(_summarize_backend_rows(backend, backend_rows))

        payload = {
            "commit": COMMIT_HASH,
            "commit_date": COMMIT_DATE,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "benchmark_spec": {
                "sample_rate": sample_rate,
                "duration_seconds": duration_seconds,
                "signal_suite_count": len(cases),
                "signal_recipe": "deterministic synthetic suite: tonal, speech-like, transient stereo, chirp/texture stereo",
                "signals": [
                    {
                        "name": str(case["name"]),
                        "description": str(case["description"]),
                        "channels": int(case["channels"]),
                        "duration_seconds": float(case["duration_seconds"]),
                    }
                    for case in cases
                ],
                "config": {
                    "n_fft": config.n_fft,
                    "win_length": config.win_length,
                    "hop_size": config.hop_size,
                    "window": config.window,
                    "center": config.center,
                    "phase_locking": config.phase_locking,
                    "transient_preserve": config.transient_preserve,
                    "transient_threshold": config.transient_threshold,
                    "kaiser_beta": config.kaiser_beta,
                },
            },
            "host": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "runs": runs,
            "case_runs": case_runs,
        }
        write_json(bench_json_path, payload)
    else:
        payload = json.loads(bench_json_path.read_text(encoding="utf-8"))

    def _fmt(value: Any, precision: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            f = float(value)
        except Exception:
            text = str(value)
            if not text or text.lower() == "none":
                return "n/a"
            return text
        if not np.isfinite(f):
            return "n/a"
        return f"{f:.{precision}f}"

    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Benchmarks")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("Reproducible benchmark summary for core short-time Fourier transform/inverse short-time Fourier transform (STFT/ISTFT) path across central processing unit/Compute Unified Device Architecture/Apple-Silicon-native contexts.")
    lines.append("")
    lines.append("## Quick Setup (Install + PATH)")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 -m venv .venv")
    lines.append("source .venv/bin/activate")
    lines.append("python3 -m pip install -e .")
    lines.append("pvx --help")
    lines.append("```")
    lines.append("")
    lines.append("If `pvx` is not found, add the virtualenv binaries to your shell path (`zsh`):")
    lines.append("")
    lines.append("```bash")
    lines.append("printf 'export PATH=\"%s/.venv/bin:$PATH\"\\n' \"$(pwd)\" >> ~/.zshrc")
    lines.append("source ~/.zshrc")
    lines.append("pvx --help")
    lines.append("```")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/scripts_generate_docs_extras.py --run-benchmarks")
    lines.append("```")
    lines.append("")

    spec = payload.get("benchmark_spec", {})
    lines.append("## Benchmark Spec")
    lines.append("")
    lines.append(f"- Sample rate: `{spec.get('sample_rate', 'n/a')}` Hz")
    lines.append(f"- Duration: `{spec.get('duration_seconds', 'n/a')}` s")
    lines.append(f"- Signal suite size: `{spec.get('signal_suite_count', 'n/a')}`")
    lines.append(f"- Signal recipe: {spec.get('signal_recipe', 'n/a')}")
    cfg = spec.get("config", {})
    lines.append(
        "- STFT config: `n_fft={n_fft}`, `win_length={win_length}`, `hop_size={hop_size}`, `window={window}`, `center={center}`".format(
            n_fft=cfg.get("n_fft", "n/a"),
            win_length=cfg.get("win_length", "n/a"),
            hop_size=cfg.get("hop_size", "n/a"),
            window=cfg.get("window", "n/a"),
            center=cfg.get("center", "n/a"),
        )
    )
    lines.append("")
    signal_rows = spec.get("signals", [])
    if isinstance(signal_rows, list) and signal_rows:
        lines.append("| Signal | Channels | Duration (s) | Description |")
        lines.append("| --- | ---: | ---: | --- |")
        for row in signal_rows:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| `{name}` | {channels} | {duration} | {desc} |".format(
                    name=str(row.get("name", "n/a")),
                    channels=int(row.get("channels", 0) or 0),
                    duration=_fmt(row.get("duration_seconds"), 3),
                    desc=str(row.get("description", "")),
                )
            )
        lines.append("")

    host = payload.get("host", {})
    lines.append("## Host")
    lines.append("")
    lines.append(f"- Platform: `{host.get('platform', 'n/a')}`")
    lines.append(f"- Machine: `{host.get('machine', 'n/a')}`")
    lines.append(f"- Python: `{host.get('python', 'n/a')}`")
    lines.append("")

    lines.append("## Backend Summary")
    lines.append("")
    lines.append(
        "| Backend | Status | Cases (ok/total) | Mean xRT | Mean elapsed (ms) | Peak host memory (MB) | Mean quality score (/100) | Mean SNR in (dB) | Mean SI-SDR (dB) | Mean LSD (dB) | Mean ModSpec | Mean Smear | Mean EnvCorr | Mean Coherence Drift | Mean Phasiness | Mean Musical Noise | Mean SNR vs CPU (dB) | Mean LSD vs CPU (dB) | Notes |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        notes = str(run.get("notes", "") or run.get("reason", ""))
        note_parts: list[str] = []
        if run.get("reason"):
            note_parts.append(str(run["reason"]))
        if run.get("runtime_fallback_reason"):
            note_parts.append(f"fallback={run['runtime_fallback_reason']}")
        if "gpu_pool_used_mb" in run:
            note_parts.append(f"gpu_pool_used_mb={run['gpu_pool_used_mb']}")
        if note_parts:
            notes = "; ".join(note_parts)
        lines.append(
            "| {backend} | {status} | {cases_ok}/{cases_total} | {xrt} | {elapsed} | {peak} | {quality} | {snr_in} | {sisdr} | {lsd} | {mod} | {smear} | {env} | {coh} | {phasey} | {mnoise} | {snr_cpu} | {lsd_cpu} | {notes} |".format(
                backend=str(run.get("backend", "n/a")),
                status=str(run.get("status", "n/a")),
                cases_ok=int(run.get("cases_ok", 0) or 0),
                cases_total=int(run.get("cases_total", 0) or 0),
                xrt=_fmt(run.get("x_real_time_mean"), 2),
                elapsed=_fmt(run.get("elapsed_ms_mean"), 2),
                peak=_fmt(run.get("peak_host_memory_mb_max"), 2),
                quality=_fmt(run.get("quality_score_mean"), 2),
                snr_in=_fmt(run.get("snr_vs_input_db_mean"), 3),
                sisdr=_fmt(run.get("si_sdr_db_mean"), 3),
                lsd=_fmt(run.get("log_spectral_distance_db_mean"), 4),
                mod=_fmt(run.get("modulation_spectrum_distance_mean"), 4),
                smear=_fmt(run.get("transient_smear_score_mean"), 4),
                env=_fmt(run.get("envelope_correlation_mean"), 4),
                coh=_fmt(run.get("stereo_coherence_drift_mean"), 4),
                phasey=_fmt(run.get("phasiness_index_mean"), 4),
                mnoise=_fmt(run.get("musical_noise_index_mean"), 4),
                snr_cpu=_fmt(run.get("snr_vs_cpu_db_mean"), 3),
                lsd_cpu=_fmt(run.get("spectral_distance_vs_cpu_db_mean"), 4),
                notes=notes,
            )
        )
    lines.append("")

    case_runs = payload.get("case_runs", [])
    if isinstance(case_runs, list) and case_runs:
        lines.append("## Per-Signal Results")
        lines.append("")
        lines.append(
            "| Backend | Signal | Channels | Duration (s) | Status | xRT | Elapsed (ms) | Quality (/100) | SNR in (dB) | SI-SDR (dB) | LSD (dB) | ModSpec | Smear | EnvCorr | Coherence Drift | Phasiness | Musical Noise | SNR vs CPU (dB) | LSD vs CPU (dB) | Notes |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        for row in sorted(
            [item for item in case_runs if isinstance(item, dict)],
            key=lambda item: (str(item.get("backend", "")), str(item.get("signal", ""))),
        ):
            note = str(row.get("notes", "") or row.get("reason", ""))
            lines.append(
                "| {backend} | `{signal}` | {channels} | {duration} | {status} | {xrt} | {elapsed} | {quality} | {snr} | {sisdr} | {lsd} | {mod} | {smear} | {env} | {coh} | {phasey} | {mnoise} | {snr_cpu} | {lsd_cpu} | {note} |".format(
                    backend=str(row.get("backend", "n/a")),
                    signal=str(row.get("signal", "n/a")),
                    channels=int(row.get("channels", 0) or 0),
                    duration=_fmt(row.get("duration_seconds"), 3),
                    status=str(row.get("status", "n/a")),
                    xrt=_fmt(row.get("x_real_time"), 2),
                    elapsed=_fmt(row.get("elapsed_ms"), 2),
                    quality=_fmt(row.get("quality_score"), 2),
                    snr=_fmt(row.get("snr_vs_input_db"), 3),
                    sisdr=_fmt(row.get("si_sdr_db"), 3),
                    lsd=_fmt(row.get("log_spectral_distance_db"), 4),
                    mod=_fmt(row.get("modulation_spectrum_distance"), 4),
                    smear=_fmt(row.get("transient_smear_score"), 4),
                    env=_fmt(row.get("envelope_correlation"), 4),
                    coh=_fmt(row.get("stereo_coherence_drift"), 4),
                    phasey=_fmt(row.get("phasiness_index"), 4),
                    mnoise=_fmt(row.get("musical_noise_index"), 4),
                    snr_cpu=_fmt(row.get("snr_vs_cpu_db"), 3),
                    lsd_cpu=_fmt(row.get("spectral_distance_vs_cpu_db"), 4),
                    note=note,
                )
            )
        lines.append("")

    lines.append("Interpretation notes:")
    lines.append("- `xRT` (times real-time): higher is faster.")
    lines.append("- `Quality (/100)` is a composite from core artifact metrics; use it as a quick ranking aid, not as a single acceptance criterion.")
    lines.append("- Lower is better for LSD, ModSpec, Smear, Coherence Drift, Phasiness, and Musical Noise.")
    lines.append("- Higher is better for SNR, SI-SDR, and Envelope Correlation.")
    lines.append("")

    lines.append("Raw machine-readable benchmark output: `docs/benchmarks/latest.json`.")
    lines.append("")

    lines.extend(attribution_section_lines())
    (DOCS_DIR / "BENCHMARKS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _classify_reference_url(url: str) -> str:
    lower = url.lower()
    if "doi.org/" in lower:
        return "doi"
    if "arxiv.org" in lower:
        return "arxiv"
    if "scholar.google.com" in lower:
        return "scholar"
    if any(host in lower for host in ("ieeexplore", "acm.org", "springer", "sciencedirect", "wiley", "jstor", "itu.int", "tech.ebu.ch")):
        return "publisher_or_standard"
    return "web"


def _extract_doi(url: str) -> str:
    match = re.search(r"doi\.org/(10\.[^\s/]+/.+)$", url, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _bib_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace('"', "\\\"")
    )


def _bib_key(paper: dict[str, str], index: int) -> str:
    authors = paper.get("authors", "unknown")
    first_author = authors.split(";")[0].strip().split()
    surname = first_author[-1].lower() if first_author else "ref"
    surname = re.sub(r"[^a-z0-9]+", "", surname) or "ref"
    year = re.sub(r"[^0-9]", "", paper.get("year", "")) or "0000"
    title_word = re.sub(r"[^a-z0-9]+", "", paper.get("title", "").lower().split(" ")[0]) or "item"
    return f"{surname}{year}{title_word}{index:03d}"


def generate_citation_docs() -> None:
    papers = list(html_docs.PAPERS)
    counts: dict[str, int] = {}
    unresolved_scholar: list[dict[str, str]] = []

    for paper in papers:
        kind = _classify_reference_url(str(paper.get("url", "")))
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "scholar":
            unresolved_scholar.append(
                {
                    "title": str(paper.get("title", "")),
                    "authors": str(paper.get("authors", "")),
                    "year": str(paper.get("year", "")),
                    "url": str(paper.get("url", "")),
                }
            )

    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Citation Quality Report")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("This report classifies bibliography URLs by citation quality and highlights entries still using search-index links.")
    lines.append("")
    lines.append(f"Total references analyzed: **{len(papers)}**")
    lines.append("")
    lines.append("## Link-Type Summary")
    lines.append("")
    lines.append("| Link type | Count |")
    lines.append("| --- | ---: |")
    for key in sorted(counts):
        lines.append(f"| `{key}` | {counts[key]} |")
    lines.append("")

    lines.append("## Entries Still Using Scholar Links")
    lines.append("")
    lines.append("These are prime targets for future DOI/publisher URL upgrades.")
    lines.append("")
    lines.append("| Year | Authors | Title | URL |")
    lines.append("| --- | --- | --- | --- |")
    for entry in sorted(unresolved_scholar, key=lambda item: (item["year"], item["title"]), reverse=True):
        lines.append(
            "| {year} | {authors} | {title} | {url} |".format(
                year=entry["year"],
                authors=entry["authors"],
                title=entry["title"],
                url=entry["url"],
            )
        )
    lines.append("")

    lines.extend(attribution_section_lines())
    (DOCS_DIR / "CITATION_QUALITY.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    bib_lines: list[str] = []
    bib_lines.append(f"% Auto-generated by scripts/scripts_generate_docs_extras.py from commit {COMMIT_HASH} ({COMMIT_DATE})")
    for index, paper in enumerate(sorted(papers, key=lambda item: (item.get("year", ""), item.get("title", ""), item.get("authors", "")))):
        key = _bib_key(paper, index)
        authors = " and ".join(a.strip() for a in str(paper.get("authors", "")).split(";") if a.strip())
        doi = _extract_doi(str(paper.get("url", "")))
        bib_lines.append(f"@misc{{{key},")
        bib_lines.append(f"  title = \"{_bib_escape(str(paper.get('title', '')))}\",")
        bib_lines.append(f"  author = \"{_bib_escape(authors)}\",")
        bib_lines.append(f"  year = \"{_bib_escape(str(paper.get('year', '')))}\",")
        bib_lines.append(f"  howpublished = \"{_bib_escape(str(paper.get('venue', '')))}\",")
        bib_lines.append(f"  url = \"{_bib_escape(str(paper.get('url', '')))}\",")
        if doi:
            bib_lines.append(f"  doi = \"{_bib_escape(doi)}\",")
        bib_lines.append("}")
        bib_lines.append("")

    (DOCS_DIR / "references.bib").write_text("\n".join(bib_lines).rstrip() + "\n", encoding="utf-8")
    write_json(
        DOCS_DIR / "citation_quality.json",
        {
            "commit": COMMIT_HASH,
            "commit_date": COMMIT_DATE,
            "counts": counts,
            "unresolved_scholar": unresolved_scholar,
        },
    )


def generate_docs_contract() -> None:
    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Documentation Contribution Contract")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("Any code change that affects behavior, parameters, algorithms, windows, outputs, or references must update generated documentation in the same PR.")
    lines.append("")
    lines.append("## Required in Every Relevant PR")
    lines.append("")
    lines.append("1. Regenerate docs:")
    lines.append("```bash")
    lines.append("python3 scripts/scripts_generate_python_docs.py")
    lines.append("python3 scripts/scripts_generate_theory_docs.py")
    lines.append("python3 scripts/scripts_generate_docs_extras.py")
    lines.append("python3 scripts/scripts_generate_html_docs.py")
    lines.append("```")
    lines.append("2. Ensure no doc drift remains (ignoring commit-stamp lines): `git diff --exit-code -I '^_Generated from commit' -I '^% Auto-generated by scripts/scripts_generate_docs_extras.py from commit' -I 'Generated by <code>scripts/scripts_generate_html_docs.py</code> from commit'` after generation.")
    lines.append("3. Keep README links and algorithm inventories consistent with generated docs.")
    lines.append("4. Expand each acronym on first use (for example, command-line interface (CLI), digital signal processing (DSP), short-time Fourier transform (STFT)).")
    lines.append("5. If CLI flags changed, verify [docs/CLI_FLAGS_REFERENCE.md](CLI_FLAGS_REFERENCE.md) and `tests/test_docs_coverage.py` pass.")
    lines.append("6. If bibliography changed, regenerate `docs/references.bib` and [docs/CITATION_QUALITY.md](CITATION_QUALITY.md).")
    lines.append("")

    lines.append("## PR Checklist")
    lines.append("")
    lines.append("- [ ] Code changes reflect intended behavior.")
    lines.append("- [ ] Documentation regenerated.")
    lines.append("- [ ] Acronyms are expanded on first use in updated docs.")
    lines.append("- [ ] Tests pass (including docs coverage checks).")
    lines.append("- [ ] New parameters or windows are mathematically documented.")
    lines.append("- [ ] New references include DOI/publisher URLs when available.")
    lines.append("")

    lines.extend(attribution_section_lines())
    (DOCS_DIR / "DOCS_CONTRACT.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate advanced pvx documentation artifacts")
    parser.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run and refresh benchmark measurements before rendering benchmark docs",
    )
    args = parser.parse_args()

    generate_cli_flags_reference()
    generate_algorithm_limitations()
    generate_cookbook()
    generate_architecture_doc()
    generate_benchmarks(run_benchmarks=args.run_benchmarks)
    generate_citation_docs()
    generate_docs_contract()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
