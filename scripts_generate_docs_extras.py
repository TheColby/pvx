#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

BENCH_DIR = DOCS_DIR / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from pvx.algorithms.registry import ALGORITHM_REGISTRY  # noqa: E402
from pvx.core import voc as voc_core  # noqa: E402
import scripts_generate_html_docs as html_docs  # noqa: E402


def git_commit_meta() -> tuple[str, str]:
    commit = "unknown"
    commit_date = "unknown"
    try:
        commit_proc = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_proc.returncode == 0 and commit_proc.stdout.strip():
            commit = commit_proc.stdout.strip()
        date_proc = subprocess.run(
            ["git", "-C", str(ROOT), "show", "-s", "--format=%cI", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if date_proc.returncode == 0 and date_proc.stdout.strip():
            commit_date = date_proc.stdout.strip()
    except Exception:
        pass
    return commit, commit_date


COMMIT_HASH, COMMIT_DATE = git_commit_meta()


def generated_stamp_lines() -> list[str]:
    return [
        f"_Generated from commit `{COMMIT_HASH}` (commit date: {COMMIT_DATE})._",
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
        return "pvxvoc.py"
    if path.parent.name == "cli":
        return f"{path.stem}.py"
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
    lines.append("# pvx CLI Flags Reference")
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
        "beamform",
        "Array spacing and steering assumptions should match real geometry.",
        "Steering mismatch causes target suppression or sidelobe leakage.",
        "Avoid with uncertain microphone spacing/ordering.",
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
            "command": "python3 pvxvoc.py vocal.wav --time-stretch 1.15 --pitch-mode formant-preserving --output-dir out --suffix _pv",
            "why": "Retains speech-like vowel envelope while stretching timing.",
        },
        {
            "category": "Phase-vocoder core",
            "title": "Independent cents retune",
            "command": "python3 pvxvoc.py lead.wav --pitch-shift-cents -23 --time-stretch 1.0 --output-dir out --suffix _cents",
            "why": "Applies precise microtonal offset without tempo change.",
        },
        {
            "category": "Transform selection",
            "title": "Default production backend (FFT + transient protection)",
            "command": "python3 pvxvoc.py mix.wav --transform fft --time-stretch 1.07 --transient-preserve --phase-locking identity --output-dir out --suffix _fft",
            "why": "Use when you need the fastest and most stable general-purpose phase-vocoder path.",
        },
        {
            "category": "Transform selection",
            "title": "Reference Fourier baseline using explicit DFT mode",
            "command": "python3 pvxvoc.py tone_sweep.wav --transform dft --time-stretch 1.00 --pitch-shift-semitones 0 --output-dir out --suffix _dft_ref",
            "why": "Useful for parity checks and controlled transform-comparison experiments.",
        },
        {
            "category": "Transform selection",
            "title": "Prime-size frame experiment with CZT backend",
            "command": "python3 pvxvoc.py archival_take.wav --transform czt --n-fft 1531 --win-length 1531 --hop-size 382 --time-stretch 1.03 --output-dir out --suffix _czt",
            "why": "Alternative numerical path for awkward/prime frame sizes when validating edge cases.",
        },
        {
            "category": "Transform selection",
            "title": "DCT timbral compaction for smooth harmonic material",
            "command": "python3 pvxvoc.py strings.wav --transform dct --pitch-shift-cents -18 --soft-clip-level 0.95 --output-dir out --suffix _dct",
            "why": "Real-basis coefficients can emphasize envelope-like structure for creative reshaping.",
        },
        {
            "category": "Transform selection",
            "title": "DST odd-symmetry color pass",
            "command": "python3 pvxvoc.py snare_loop.wav --transform dst --time-stretch 0.92 --phase-locking off --output-dir out --suffix _dst",
            "why": "Provides an alternate real-basis artifact profile useful for creative percussive processing.",
        },
        {
            "category": "Transform selection",
            "title": "Hartley real-basis exploratory render",
            "command": "python3 pvxvoc.py synth_pad.wav --transform hartley --time-stretch 1.30 --pitch-shift-semitones 3 --output-dir out --suffix _hartley",
            "why": "Compares Hartley-domain behavior against complex FFT phase-vocoder output.",
        },
        {
            "category": "Transform selection",
            "title": "A/B sweep of transform backends from shell loop",
            "command": "for t in fft dft czt dct dst hartley; do python3 pvxvoc.py voice.wav --transform \"$t\" --time-stretch 1.1 --output-dir out --suffix \"_$t\"; done",
            "why": "Fast listening workflow for selecting the least-artifact transform on your source.",
        },
        {
            "category": "Microtonal",
            "title": "Custom cents map retune",
            "command": "python3 pvxretune.py vox.wav --root 60 --scale-cents 0,90,204,294,408,498,612,702,816,906,1020,1110 --strength 0.8 --output-dir out",
            "why": "Maps incoming notes to a custom 12-degree microtonal scale.",
        },
        {
            "category": "Microtonal",
            "title": "Conform CSV with per-segment ratios",
            "command": "python3 pvxconform.py solo.wav map_conform.csv --pitch-mode ratio --output-dir out --suffix _conform",
            "why": "Applies timeline-specific time and pitch trajectories from CSV.",
        },
        {
            "category": "Pipelines",
            "title": "Time-stretch -> denoise -> dereverb in one pipe",
            "command": "python3 pvxvoc.py input.wav --time-stretch 1.25 --stdout | python3 pvxdenoise.py - --reduction-db 10 --stdout | python3 pvxdeverb.py - --strength 0.45 --output-dir out --suffix _clean",
            "why": "Single-pass CLI chain for serial DSP in Unix pipes.",
        },
        {
            "category": "Pipelines",
            "title": "Morph -> formant -> unison",
            "command": "python3 pvxmorph.py a.wav b.wav -o - | python3 pvxformant.py - --mode preserve --stdout | python3 pvxunison.py - --voices 5 --detune-cents 8 --output-dir out --suffix _morph_stack",
            "why": "Builds a richer timbre chain with no intermediate files.",
        },
        {
            "category": "Mastering",
            "title": "Integrated loudness targeting with limiter",
            "command": "python3 pvxvoc.py mix.wav --time-stretch 1.0 --target-lufs -14 --compressor-threshold-db -20 --compressor-ratio 3 --limiter-threshold 0.98 --output-dir out --suffix _master",
            "why": "Combines dynamics and loudness controls in shared mastering chain.",
        },
        {
            "category": "Mastering",
            "title": "Soft clip and hard safety ceiling",
            "command": "python3 pvxharmonize.py bus.wav --intervals 0,7,12 --mix 0.35 --soft-clip-level 0.92 --soft-clip-type tanh --hard-clip-level 0.99 --output-dir out",
            "why": "Adds saturation while enforcing a strict final peak ceiling.",
        },
        {
            "category": "Batch",
            "title": "Batch stretch over folder",
            "command": "python3 pvxvoc.py stems/*.wav --time-stretch 1.08 --output-dir out/stems --overwrite",
            "why": "Applies consistent transform to many files with one command.",
        },
        {
            "category": "Batch",
            "title": "Dry-run output validation",
            "command": "python3 pvxdenoise.py takes/*.wav --reduction-db 8 --dry-run --output-dir out/preview",
            "why": "Checks filename resolution and collisions without writing audio.",
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
    lines.append("  G1[scripts_generate_python_docs.py] --> D[docs/*]")
    lines.append("  G2[scripts_generate_theory_docs.py] --> D")
    lines.append("  G3[scripts_generate_docs_extras.py] --> D")
    lines.append("  G4[scripts_generate_html_docs.py] --> H[docs/html/*]")
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

    (DOCS_DIR / "ARCHITECTURE.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _spectral_distance_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    eps = 1e-9
    fft_len = 4096
    ref_mag = np.abs(np.fft.rfft(reference, n=fft_len))
    cand_mag = np.abs(np.fft.rfft(candidate, n=fft_len))
    ref_db = 20.0 * np.log10(np.maximum(ref_mag, eps))
    cand_db = 20.0 * np.log10(np.maximum(cand_mag, eps))
    return float(np.sqrt(np.mean((ref_db - cand_db) ** 2)))


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    eps = 1e-12
    err = reference - candidate
    num = float(np.sum(reference * reference))
    den = float(np.sum(err * err))
    return 10.0 * math.log10(max(num, eps) / max(den, eps))


def _make_signal(sample_rate: int = 48000, seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(sample_rate * seconds), dtype=np.float64) / float(sample_rate)
    signal = (
        0.42 * np.sin(2.0 * math.pi * 110.0 * t)
        + 0.18 * np.sin(2.0 * math.pi * 223.0 * t + 0.3)
        + 0.12 * np.sin(2.0 * math.pi * 451.0 * t + 0.7)
        + 0.08 * np.sin(2.0 * math.pi * 1780.0 * t)
    )
    ramp = np.linspace(0.4, 1.0, signal.size, dtype=np.float64)
    signal = signal * ramp
    return np.clip(signal, -0.98, 0.98)


def _benchmark_backend(
    label: str,
    device: str,
    signal: np.ndarray,
    config: voc_core.VocoderConfig,
    reference: np.ndarray | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "backend": label,
        "device_request": device,
        "status": "ok",
        "reason": "",
    }

    try:
        voc_core.configure_runtime(device=device, cuda_device=0, verbose=False)
    except Exception as exc:
        payload.update({"status": "unavailable", "reason": str(exc)})
        return payload

    tracemalloc.start()
    start = time.perf_counter()
    try:
        spectrum = voc_core.stft(signal, config)
        reconstructed = voc_core.istft(spectrum, config, expected_length=signal.size)
    except Exception as exc:
        tracemalloc.stop()
        payload.update({"status": "failed", "reason": str(exc)})
        return payload

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    signal_f = np.asarray(signal, dtype=np.float64)

    payload.update(
        {
            "elapsed_ms": round(elapsed_ms, 3),
            "peak_host_memory_mb": round(float(peak_bytes) / (1024.0 * 1024.0), 3),
            "snr_vs_input_db": round(_snr_db(signal_f, reconstructed), 4),
            "spectral_distance_vs_input_db": round(_spectral_distance_db(signal_f, reconstructed), 4),
            "runtime_active_device": voc_core.runtime_config().active_device,
            "runtime_fallback_reason": voc_core.runtime_config().fallback_reason,
        }
    )

    if reference is not None:
        payload.update(
            {
                "snr_vs_cpu_db": round(_snr_db(np.asarray(reference, dtype=np.float64), reconstructed), 4),
                "spectral_distance_vs_cpu_db": round(_spectral_distance_db(np.asarray(reference, dtype=np.float64), reconstructed), 4),
                "max_abs_error_vs_cpu": round(float(np.max(np.abs(np.asarray(reference, dtype=np.float64) - reconstructed))), 10),
            }
        )

    if voc_core.runtime_config().active_device == "cuda" and getattr(voc_core, "cp", None) is not None:
        try:
            mem_pool = voc_core.cp.get_default_memory_pool()
            payload["gpu_pool_used_mb"] = round(float(mem_pool.used_bytes()) / (1024.0 * 1024.0), 3)
            mem_pool.free_all_blocks()
        except Exception:
            pass

    return payload


def generate_benchmarks(run_benchmarks: bool) -> None:
    bench_json_path = BENCH_DIR / "latest.json"

    if run_benchmarks or not bench_json_path.exists():
        sample_rate = 48000
        signal = _make_signal(sample_rate=sample_rate, seconds=4.0)
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

        cpu_result = _benchmark_backend("cpu", "cpu", signal, config, reference=None)
        cpu_reference = None
        if cpu_result.get("status") == "ok":
            voc_core.configure_runtime(device="cpu", cuda_device=0, verbose=False)
            cpu_reference = voc_core.istft(voc_core.stft(signal, config), config, expected_length=signal.size)

        cuda_result = _benchmark_backend("cuda", "cuda", signal, config, reference=np.asarray(cpu_reference) if cpu_reference is not None else None)

        is_apple_silicon = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
        if is_apple_silicon:
            apple_result = _benchmark_backend(
                "apple_silicon_native_cpu",
                "cpu",
                signal,
                config,
                reference=np.asarray(cpu_reference) if cpu_reference is not None else None,
            )
        else:
            apple_result = {
                "backend": "apple_silicon_native_cpu",
                "device_request": "cpu",
                "status": "unavailable",
                "reason": "Host platform is not Apple Silicon (Darwin arm64).",
            }

        payload = {
            "commit": COMMIT_HASH,
            "commit_date": COMMIT_DATE,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "benchmark_spec": {
                "sample_rate": sample_rate,
                "duration_seconds": 4.0,
                "signal_recipe": "sum of 4 deterministic sinusoids with linear amplitude ramp",
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
            "runs": [cpu_result, cuda_result, apple_result],
        }
        write_json(bench_json_path, payload)
    else:
        payload = json.loads(bench_json_path.read_text(encoding="utf-8"))

    lines: list[str] = []
    lines.append("# pvx Benchmarks")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("Reproducible benchmark summary for core STFT+ISTFT path across CPU/CUDA/Apple-Silicon-native contexts.")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts_generate_docs_extras.py --run-benchmarks")
    lines.append("```")
    lines.append("")

    spec = payload.get("benchmark_spec", {})
    lines.append("## Benchmark Spec")
    lines.append("")
    lines.append(f"- Sample rate: `{spec.get('sample_rate', 'n/a')}` Hz")
    lines.append(f"- Duration: `{spec.get('duration_seconds', 'n/a')}` s")
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

    host = payload.get("host", {})
    lines.append("## Host")
    lines.append("")
    lines.append(f"- Platform: `{host.get('platform', 'n/a')}`")
    lines.append(f"- Machine: `{host.get('machine', 'n/a')}`")
    lines.append(f"- Python: `{host.get('python', 'n/a')}`")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Backend | Status | Elapsed (ms) | Peak host memory (MB) | SNR vs input (dB) | Spectral distance vs input (dB) | SNR vs CPU (dB) | Spectral distance vs CPU (dB) | Notes |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for run in payload.get("runs", []):
        note_parts: list[str] = []
        if run.get("reason"):
            note_parts.append(str(run["reason"]))
        if run.get("runtime_fallback_reason"):
            note_parts.append(f"fallback={run['runtime_fallback_reason']}")
        if "gpu_pool_used_mb" in run:
            note_parts.append(f"gpu_pool_used_mb={run['gpu_pool_used_mb']}")

        lines.append(
            "| {backend} | {status} | {elapsed} | {peak} | {snr_in} | {lsd_in} | {snr_cpu} | {lsd_cpu} | {notes} |".format(
                backend=run.get("backend", "n/a"),
                status=run.get("status", "n/a"),
                elapsed=run.get("elapsed_ms", "n/a"),
                peak=run.get("peak_host_memory_mb", "n/a"),
                snr_in=run.get("snr_vs_input_db", "n/a"),
                lsd_in=run.get("spectral_distance_vs_input_db", "n/a"),
                snr_cpu=run.get("snr_vs_cpu_db", "n/a"),
                lsd_cpu=run.get("spectral_distance_vs_cpu_db", "n/a"),
                notes="; ".join(note_parts) if note_parts else "",
            )
        )
    lines.append("")

    lines.append("Raw machine-readable benchmark output: `docs/benchmarks/latest.json`.")
    lines.append("")

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

    (DOCS_DIR / "CITATION_QUALITY.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    bib_lines: list[str] = []
    bib_lines.append(f"% Auto-generated by scripts_generate_docs_extras.py from commit {COMMIT_HASH} ({COMMIT_DATE})")
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
    lines.append("# pvx Documentation Contribution Contract")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("Any code change that affects behavior, parameters, algorithms, windows, outputs, or references must update generated documentation in the same PR.")
    lines.append("")
    lines.append("## Required in Every Relevant PR")
    lines.append("")
    lines.append("1. Regenerate docs:")
    lines.append("```bash")
    lines.append("python3 scripts_generate_python_docs.py")
    lines.append("python3 scripts_generate_theory_docs.py")
    lines.append("python3 scripts_generate_docs_extras.py")
    lines.append("python3 scripts_generate_html_docs.py")
    lines.append("```")
    lines.append("2. Ensure no doc drift remains (ignoring commit-stamp lines): `git diff --exit-code -I '^_Generated from commit' -I '^% Auto-generated by scripts_generate_docs_extras.py from commit' -I 'Generated by <code>scripts_generate_html_docs.py</code> from commit'` after generation.")
    lines.append("3. Keep README links and algorithm inventories consistent with generated docs.")
    lines.append("4. If CLI flags changed, verify `docs/CLI_FLAGS_REFERENCE.md` and `tests/test_docs_coverage.py` pass.")
    lines.append("5. If bibliography changed, regenerate `docs/references.bib` and `docs/CITATION_QUALITY.md`.")
    lines.append("")

    lines.append("## PR Checklist")
    lines.append("")
    lines.append("- [ ] Code changes reflect intended behavior.")
    lines.append("- [ ] Documentation regenerated.")
    lines.append("- [ ] Tests pass (including docs coverage checks).")
    lines.append("- [ ] New parameters or windows are mathematically documented.")
    lines.append("- [ ] New references include DOI/publisher URLs when available.")
    lines.append("")

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
