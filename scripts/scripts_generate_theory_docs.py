#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Generate GitHub-renderable theory docs (math foundations + window reference)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
WINDOW_ASSETS_DIR = ASSETS_DIR / "windows"
INTERPOLATION_ASSETS_DIR = ASSETS_DIR / "interpolation"
FUNCTION_ASSETS_DIR = ASSETS_DIR / "functions"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
WINDOW_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
INTERPOLATION_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
FUNCTION_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))
from pvx.core.attribution import ATTRIBUTION_DOC_PATH, COPYRIGHT_NOTICE  # noqa: E402
from pvx.core import voc as voc_core  # noqa: E402


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

def window_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for name in voc_core.WINDOW_CHOICES:
        if name in voc_core._COSINE_SERIES_WINDOWS:
            coeffs = ", ".join(f"{c:g}" for c in voc_core._COSINE_SERIES_WINDOWS[name])
            family = "Cosine series"
            params = f"coeffs=({coeffs})"
            formula = "W1"
            note = "Cosine-sum taper with configurable sidelobe/main-lobe tradeoff."
        elif name == "sine":
            family = "Sinusoidal"
            params = "none"
            formula = "W2"
            note = "Raised-sine taper with smooth edges and moderate leakage suppression."
        elif name == "cosine":
            family = "Sinusoidal"
            params = "none"
            formula = "W2"
            note = "Implemented identically to sine in pvx."
        elif name == "bartlett":
            family = "Triangular"
            params = "none"
            formula = "W3"
            note = "Linear taper to zero at both ends."
        elif name == "boxcar":
            family = "Rectangular"
            params = "none"
            formula = "W0"
            note = "No tapering; strongest leakage, narrowest main lobe."
        elif name == "rect":
            family = "Rectangular"
            params = "none"
            formula = "W0"
            note = "Alias of boxcar."
        elif name == "triangular":
            family = "Triangular"
            params = "none"
            formula = "W4"
            note = "Triangular variant with $(N+1)/2$ denominator."
        elif name == "bartlett_hann":
            family = "Hybrid taper"
            params = "fixed coefficients"
            formula = "W5"
            note = "Blend of linear slope and cosine curvature."
        elif name in voc_core._TUKEY_WINDOWS:
            family = "Tukey"
            params = f"alpha={voc_core._TUKEY_WINDOWS[name]:g}"
            formula = "W6"
            note = "Flat middle with cosine tapers at both sides."
        elif name == "parzen":
            family = "Polynomial"
            params = "piecewise cubic"
            formula = "W7"
            note = "Smooth cubic taper with strong sidelobe attenuation."
        elif name == "lanczos":
            family = "Sinc"
            params = "none"
            formula = "W8"
            note = "Sinc-shaped taper often used in interpolation-related contexts."
        elif name == "welch":
            family = "Quadratic"
            params = "none"
            formula = "W9"
            note = "Parabolic taper that emphasizes center samples."
        elif name in voc_core._GAUSSIAN_WINDOWS:
            family = "Gaussian"
            params = f"sigma_ratio={voc_core._GAUSSIAN_WINDOWS[name]:g}"
            formula = "W10"
            note = "Bell curve; smaller sigma gives stronger edge attenuation."
        elif name in voc_core._GENERAL_GAUSSIAN_WINDOWS:
            power, sigma_ratio = voc_core._GENERAL_GAUSSIAN_WINDOWS[name]
            family = "Generalized Gaussian"
            params = f"power={power:g}, sigma_ratio={sigma_ratio:g}"
            formula = "W11"
            note = "Power parameter adjusts shoulder steepness."
        elif name in voc_core._EXPONENTIAL_WINDOWS:
            family = "Exponential"
            params = f"tau_ratio={voc_core._EXPONENTIAL_WINDOWS[name]:g}"
            formula = "W12"
            note = "Symmetric exponential decay from center."
        elif name in voc_core._CAUCHY_WINDOWS:
            family = "Cauchy / Lorentzian"
            params = f"gamma_ratio={voc_core._CAUCHY_WINDOWS[name]:g}"
            formula = "W13"
            note = "Heavy-tailed taper with slower decay than Gaussian."
        elif name in voc_core._COSINE_POWER_WINDOWS:
            family = "Cosine power"
            params = f"power={voc_core._COSINE_POWER_WINDOWS[name]:g}"
            formula = "W14"
            note = "Sine taper raised to power $p$."
        elif name in voc_core._HANN_POISSON_WINDOWS:
            family = "Hann-Poisson"
            params = f"alpha={voc_core._HANN_POISSON_WINDOWS[name]:g}"
            formula = "W15"
            note = "Hann multiplied by exponential envelope."
        elif name in voc_core._GENERAL_HAMMING_WINDOWS:
            family = "General Hamming"
            params = f"alpha={voc_core._GENERAL_HAMMING_WINDOWS[name]:.2f}"
            formula = "W16"
            note = "Hamming family with tunable cosine weight."
        elif name == "bohman":
            family = "Bohman"
            params = "none"
            formula = "W17"
            note = "Continuous-slope cosine/sine blend."
        elif name == "kaiser":
            family = "Kaiser-Bessel"
            params = "beta from --kaiser-beta"
            formula = "W18"
            note = "Adjustable Bessel-based tradeoff between sidelobes and width."
        else:
            family = "Other"
            params = "none"
            formula = "W0"
            note = "Window supported by pvx."
        pros, cons, usage = window_tradeoffs(name, family)
        entries.append(
            {
                "name": name,
                "family": family,
                "params": params,
                "formula": formula,
                "note": note,
                "pros": pros,
                "cons": cons,
                "usage": usage,
            }
        )
    return entries


def window_tradeoffs(name: str, family: str) -> tuple[str, str, str]:
    if name in {"hann", "hamming"}:
        return (
            "Balanced leakage suppression and frequency resolution.",
            "Not optimal for amplitude metering or extreme sidelobe rejection.",
            "Default choice for most pvx time-stretch and pitch-shift workflows.",
        )
    if name in {"boxcar", "rect"}:
        return (
            "Narrowest main lobe and maximal bin sharpness.",
            "Highest sidelobes and strongest leakage/phasiness on non-bin-centered content.",
            "Use only for controlled test tones or when leakage is acceptable.",
        )
    if name in {"blackman", "blackmanharris", "blackman_nuttall", "nuttall", "exact_blackman"}:
        return (
            "Strong sidelobe suppression for cleaner spectral separation.",
            "Wider main lobe than Hann/Hamming.",
            "Use for dense harmonic material when leakage artifacts dominate.",
        )
    if name == "flattop":
        return (
            "Very accurate amplitude estimation in FFT bins.",
            "Very wide main lobe and reduced frequency discrimination.",
            "Use for measurement-grade magnitude tracking, not fine pitch separation.",
        )
    if name == "kaiser":
        return (
            "Continuously tunable width/sidelobe tradeoff via beta.",
            "Needs beta tuning; poor beta choices can over-blur or under-suppress sidelobes.",
            "Use when you need explicit control over leakage versus resolution.",
        )
    if family == "Tukey":
        return (
            "Interpolates between rectangular and Hann behavior.",
            "Behavior changes strongly with alpha and can be inconsistent across presets.",
            "Use when you need a controllable flat center with soft edges.",
        )
    if family == "Gaussian":
        return (
            "Smooth bell taper with low ringing and predictable decay.",
            "Frequency resolution changes noticeably with sigma.",
            "Use sigma presets to tune transient locality versus spectral leakage.",
        )
    if family == "Generalized Gaussian":
        return (
            "Additional shape control beyond standard Gaussian.",
            "Extra parameterization increases tuning complexity.",
            "Use for research/tuning tasks where shoulder steepness matters.",
        )
    if family == "Exponential":
        return (
            "Simple edge decay with fast computation.",
            "Can produce less-uniform center weighting than cosine families.",
            "Use for experiments emphasizing center-heavy weighting.",
        )
    if family == "Cauchy / Lorentzian":
        return (
            "Heavy tails preserve more peripheral samples than Gaussian.",
            "Tail energy can increase leakage relative to steeper tapers.",
            "Use when you want softer attenuation of far-window samples.",
        )
    if family == "Cosine power":
        return (
            "Simple power parameter controls edge steepness.",
            "Higher powers can overly narrow effective support.",
            "Use to smoothly increase edge attenuation versus basic sine.",
        )
    if family == "Hann-Poisson":
        return (
            "Combines smooth Hann center with exponential edge suppression.",
            "Can over-attenuate edges for high alpha values.",
            "Use when you need stronger edge decay than Hann without full flattop cost.",
        )
    if family == "General Hamming":
        return (
            "Tunable Hamming-style cosine weighting.",
            "Less standardized than classic Hann/Hamming choices.",
            "Use to sweep sidelobe-vs-width behavior near Hamming family defaults.",
        )
    if family == "Triangular":
        return (
            "Cheap linear taper with intuitive behavior.",
            "Higher sidelobes than stronger cosine-sum windows.",
            "Use for low-cost processing or quick exploratory runs.",
        )
    if name == "bartlett_hann":
        return (
            "Moderate leakage suppression with lightweight computation.",
            "Less common and less interpretable than standard Hann/Hamming.",
            "Use as a middle-ground taper when Bartlett feels too sharp.",
        )
    if name == "parzen":
        return (
            "Smooth high-order taper with good sidelobe control.",
            "Broader main lobe than lightweight windows.",
            "Use for spectral denoising/analysis where sidelobe cleanup is critical.",
        )
    if name == "lanczos":
        return (
            "Sinc-derived shape with useful compromise behavior.",
            "Can exhibit oscillatory spectral behavior versus cosine-sum defaults.",
            "Use for interpolation-adjacent analysis experiments.",
        )
    if name == "welch":
        return (
            "Center-emphasizing parabola with simple form.",
            "Not as strong at sidelobe suppression as Blackman-family windows.",
            "Use when center weighting is desired with minimal complexity.",
        )
    if name in {"sine", "cosine"}:
        return (
            "Smooth endpoint behavior and straightforward implementation.",
            "Less configurable than Kaiser/Tukey families.",
            "Use for stable, low-complexity alternatives to Hann.",
        )
    if name == "bohman":
        return (
            "Continuous-slope taper with good qualitative leakage control.",
            "Less common in audio tooling and harder to tune by intuition.",
            "Use for exploratory spectral work needing smooth derivatives.",
        )
    return (
        "Supported by pvx and compatible with the shared phase-vocoder path.",
        "Tradeoffs depend on the specific shape parameters.",
        "Start with Hann/Hamming, then compare this window if artifacts persist.",
    )


def window_samples(name: str, length: int = 2048, kaiser_beta: float = 14.0) -> np.ndarray:
    samples = voc_core.make_window(name, length, length, kaiser_beta=kaiser_beta, xp=np)
    return np.asarray(samples, dtype=np.float64)


def _first_local_minimum(values: np.ndarray) -> int:
    if values.size < 4:
        return max(1, values.size - 1)
    diff = np.diff(values)
    for idx in range(1, diff.size):
        if diff[idx - 1] < 0.0 and diff[idx] >= 0.0:
            return idx
    below = np.flatnonzero(values < 1e-4)
    if below.size:
        return int(max(1, below[0]))
    return max(1, values.size // 8)


def compute_window_metrics(window: np.ndarray) -> dict[str, float]:
    n = int(window.size)
    eps = 1e-15
    sum_w = float(np.sum(window))
    coherent_gain = sum_w / max(float(n), eps)
    enbw_bins = (n * float(np.sum(window * window))) / max(sum_w * sum_w, eps)

    idx = np.arange(n, dtype=np.float64)
    half_bin_phasor = np.exp(-1j * math.pi * idx / max(float(n), eps))
    half_bin_resp = abs(np.sum(window * half_bin_phasor))
    dc_resp = abs(sum_w)
    scalloping_loss_db = 20.0 * math.log10(max(half_bin_resp / max(dc_resp, eps), eps))

    fft_len = 1
    while fft_len < n * 64:
        fft_len *= 2
    spectrum = np.abs(np.fft.rfft(window, n=fft_len))
    spectrum /= max(float(np.max(spectrum)), eps)

    first_min = _first_local_minimum(spectrum)
    main_lobe_width_bins = 2.0 * first_min * float(n) / float(fft_len)

    side = spectrum[first_min + 1 :] if (first_min + 1) < spectrum.size else np.array([eps])
    peak_sidelobe_db = 20.0 * math.log10(max(float(np.max(side)), eps))

    return {
        "coherent_gain": round(coherent_gain, 12),
        "enbw_bins": round(enbw_bins, 12),
        "scalloping_loss_db": round(scalloping_loss_db, 12),
        "main_lobe_width_bins": round(main_lobe_width_bins, 12),
        "peak_sidelobe_db": round(peak_sidelobe_db, 12),
    }


def _polyline_points(xs: np.ndarray, ys: np.ndarray, width: int, height: int, margin: int, y_min: float, y_max: float) -> str:
    plot_w = max(1, width - 2 * margin)
    plot_h = max(1, height - 2 * margin)
    y_span = max(1e-9, y_max - y_min)
    points: list[str] = []
    for x_val, y_val in zip(xs, ys):
        px = margin + plot_w * float(x_val)
        py = margin + plot_h * (1.0 - ((float(y_val) - y_min) / y_span))
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _downsample_series(ys: np.ndarray, max_points: int = 1024) -> np.ndarray:
    if ys.size <= max_points:
        return ys
    idx = np.linspace(0, ys.size - 1, max_points, dtype=np.int64)
    return ys[idx]


def write_line_svg(path: Path, ys: np.ndarray, *, title: str, y_min: float, y_max: float, x_label: str, y_label: str) -> None:
    width = 840
    height = 280
    margin = 36
    y_plot = _downsample_series(np.asarray(ys, dtype=np.float64), max_points=1024)
    xs = np.linspace(0.0, 1.0, y_plot.size, dtype=np.float64)
    points = _polyline_points(xs, y_plot, width, height, margin, y_min, y_max)

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#ffffff\" />
  <rect x=\"{margin}\" y=\"{margin}\" width=\"{width - 2 * margin}\" height=\"{height - 2 * margin}\" fill=\"#f7fbff\" stroke=\"#c8d9e6\" />
  <polyline fill=\"none\" stroke=\"#005f73\" stroke-width=\"2\" points=\"{points}\" />
  <text x=\"{width / 2:.1f}\" y=\"22\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"15\" fill=\"#102a43\">{title}</text>
  <text x=\"{width / 2:.1f}\" y=\"{height - 8}\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#334e68\">{x_label}</text>
  <text x=\"14\" y=\"{height / 2:.1f}\" transform=\"rotate(-90 14,{height / 2:.1f})\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#334e68\">{y_label}</text>
  <text x=\"{margin}\" y=\"{height - margin + 18}\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">0</text>
  <text x=\"{width - margin}\" y=\"{height - margin + 18}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">1</text>
  <text x=\"{margin - 6}\" y=\"{margin + 4}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{y_max:.1f}</text>
  <text x=\"{margin - 6}\" y=\"{height - margin + 4}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{y_min:.1f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _svg_plot_points(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    *,
    width: int,
    height: int,
    margin: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    x_span = max(1e-12, x_max - x_min)
    y_span = max(1e-12, y_max - y_min)
    plot_w = max(1, width - 2 * margin)
    plot_h = max(1, height - 2 * margin)
    points: list[str] = []
    for xv, yv in zip(x_vals, y_vals):
        px = margin + plot_w * ((float(xv) - x_min) / x_span)
        py = margin + plot_h * (1.0 - ((float(yv) - y_min) / y_span))
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def write_multiline_svg(
    path: Path,
    x_vals: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    width = 980
    height = 360
    margin = 42
    x = np.asarray(x_vals, dtype=np.float64)
    if x.size < 2:
        raise ValueError("x_vals must contain at least 2 samples")
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    y_arrays = [np.asarray(yv, dtype=np.float64) for _, yv, _ in series]
    if not y_arrays:
        raise ValueError("series must not be empty")
    y_all = np.concatenate(y_arrays)
    y_lo = float(np.min(y_all)) if y_min is None else float(y_min)
    y_hi = float(np.max(y_all)) if y_max is None else float(y_max)
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    plot_lines: list[str] = []
    legend_items: list[str] = []
    for idx, (name, y_vals, color) in enumerate(series):
        y_arr = np.asarray(y_vals, dtype=np.float64)
        pts = _svg_plot_points(
            x,
            y_arr,
            width=width,
            height=height,
            margin=margin,
            x_min=x_min,
            x_max=x_max,
            y_min=y_lo,
            y_max=y_hi,
        )
        plot_lines.append(
            f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" points=\"{pts}\" />"
        )
        legend_x = margin + 14
        legend_y = margin + 14 + idx * 16
        legend_items.append(
            f"<line x1=\"{legend_x:.1f}\" y1=\"{legend_y:.1f}\" x2=\"{legend_x + 22:.1f}\" y2=\"{legend_y:.1f}\" "
            f"stroke=\"{color}\" stroke-width=\"2\" />"
            f"<text x=\"{legend_x + 28:.1f}\" y=\"{legend_y + 4:.1f}\" font-family=\"Arial, Helvetica, sans-serif\" "
            f"font-size=\"11\" fill=\"#243b53\">{name}</text>"
        )

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#ffffff\" />
  <rect x=\"{margin}\" y=\"{margin}\" width=\"{width - 2 * margin}\" height=\"{height - 2 * margin}\" fill=\"#f7fbff\" stroke=\"#c8d9e6\" />
  {' '.join(plot_lines)}
  {' '.join(legend_items)}
  <text x=\"{width / 2:.1f}\" y=\"24\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"16\" fill=\"#102a43\">{title}</text>
  <text x=\"{width / 2:.1f}\" y=\"{height - 8}\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#334e68\">{x_label}</text>
  <text x=\"14\" y=\"{height / 2:.1f}\" transform=\"rotate(-90 14,{height / 2:.1f})\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#334e68\">{y_label}</text>
  <text x=\"{margin}\" y=\"{height - margin + 18}\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{x_min:.2f}</text>
  <text x=\"{width - margin}\" y=\"{height - margin + 18}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{x_max:.2f}</text>
  <text x=\"{margin - 6}\" y=\"{margin + 4}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{y_hi:.2f}</text>
  <text x=\"{margin - 6}\" y=\"{height - margin + 4}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{y_lo:.2f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _compressor_curve_db(x_db: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    return np.where(x_db <= threshold_db, x_db, threshold_db + (x_db - threshold_db) / ratio)


def _expander_curve_db(x_db: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
    return np.where(x_db >= threshold_db, x_db, threshold_db + (x_db - threshold_db) * ratio)


def _limiter_curve_db(x_db: np.ndarray, ceiling_db: float) -> np.ndarray:
    return np.minimum(x_db, ceiling_db)


def _softclip_cubic(x: np.ndarray) -> np.ndarray:
    y = x - (x**3) / 3.0
    y = np.where(np.abs(x) >= 1.0, np.sign(x) * (2.0 / 3.0), y)
    return y / (2.0 / 3.0)


def generate_function_assets() -> dict[str, object]:
    palette = ["#005f73", "#0a9396", "#ee9b00", "#ca6702", "#ae2012", "#9b2226", "#3a0ca3", "#4361ee"]
    entries: list[dict[str, str]] = []

    semitones = np.linspace(-24.0, 24.0, 801, dtype=np.float64)
    ratio_semitones = np.power(2.0, semitones / 12.0)
    semitone_path = FUNCTION_ASSETS_DIR / "pitch_ratio_vs_semitones.svg"
    write_multiline_svg(
        semitone_path,
        semitones,
        [("ratio", ratio_semitones, palette[0])],
        title="Pitch ratio vs semitone shift",
        x_label="semitones",
        y_label="ratio",
    )
    entries.append(
        {
            "name": "Pitch ratio from semitones",
            "path": "assets/functions/pitch_ratio_vs_semitones.svg",
            "usage": "`r = 2^(semitones/12)` conversion used by pitch-shift flags.",
        }
    )

    cents = np.linspace(-1200.0, 1200.0, 801, dtype=np.float64)
    ratio_cents = np.power(2.0, cents / 1200.0)
    cents_path = FUNCTION_ASSETS_DIR / "pitch_ratio_vs_cents.svg"
    write_multiline_svg(
        cents_path,
        cents,
        [("ratio", ratio_cents, palette[1])],
        title="Pitch ratio vs cent shift",
        x_label="cents",
        y_label="ratio",
    )
    entries.append(
        {
            "name": "Pitch ratio from cents",
            "path": "assets/functions/pitch_ratio_vs_cents.svg",
            "usage": "`r = 2^(cents/1200)` conversion for microtonal cent controls.",
        }
    )

    x_db = np.linspace(-80.0, 0.0, 1201, dtype=np.float64)
    comp = _compressor_curve_db(x_db, threshold_db=-24.0, ratio=4.0)
    expd = _expander_curve_db(x_db, threshold_db=-45.0, ratio=2.0)
    lim = _limiter_curve_db(x_db, ceiling_db=-1.0)
    dyn_path = FUNCTION_ASSETS_DIR / "dynamics_transfer_curves.svg"
    write_multiline_svg(
        dyn_path,
        x_db,
        [
            ("identity", x_db, "#94a3b8"),
            ("compressor 4:1 @ -24 dB", comp, palette[2]),
            ("expander 2:1 @ -45 dB", expd, palette[3]),
            ("limiter ceiling -1 dB", lim, palette[4]),
        ],
        title="Dynamics transfer curves (dB in -> dB out)",
        x_label="input level (dBFS)",
        y_label="output level (dBFS)",
    )
    entries.append(
        {
            "name": "Dynamics transfer functions",
            "path": "assets/functions/dynamics_transfer_curves.svg",
            "usage": "Compressor/expander/limiter conceptual transfer curves used in mastering stages.",
        }
    )

    x_lin = np.linspace(-1.2, 1.2, 1201, dtype=np.float64)
    drive = 2.5
    soft_tanh = np.tanh(drive * x_lin) / np.tanh(drive)
    soft_arctan = np.arctan(drive * x_lin) / np.arctan(drive)
    soft_cubic = _softclip_cubic(np.clip(x_lin, -1.5, 1.5))
    softclip_path = FUNCTION_ASSETS_DIR / "softclip_transfer_functions.svg"
    write_multiline_svg(
        softclip_path,
        x_lin,
        [
            ("identity", x_lin, "#94a3b8"),
            ("tanh", soft_tanh, palette[0]),
            ("arctan", soft_arctan, palette[5]),
            ("cubic", soft_cubic, palette[6]),
        ],
        title="Soft-clip transfer functions (linear amplitude domain)",
        x_label="input amplitude",
        y_label="output amplitude",
        y_min=-1.2,
        y_max=1.2,
    )
    entries.append(
        {
            "name": "Soft-clip transfer functions",
            "path": "assets/functions/softclip_transfer_functions.svg",
            "usage": "Shows `tanh`, `arctan`, and `cubic` soft clipping behavior.",
        }
    )

    alpha = np.linspace(0.0, 1.0, 801, dtype=np.float64)
    mag_a = 0.20
    mag_b = 1.00
    linear_blend = (1.0 - alpha) * mag_a + alpha * mag_b
    geometric_blend = np.exp((1.0 - alpha) * np.log(mag_a + 1e-12) + alpha * np.log(mag_b + 1e-12))
    product_target = np.sqrt(max(mag_a * mag_b, 0.0))
    product_blend = (1.0 - alpha) * mag_a + alpha * product_target
    max_mag_blend = (1.0 - alpha) * linear_blend + alpha * max(mag_a, mag_b)
    min_mag_blend = (1.0 - alpha) * linear_blend + alpha * min(mag_a, mag_b)
    blend_path = FUNCTION_ASSETS_DIR / "morph_blend_magnitude_curves.svg"
    write_multiline_svg(
        blend_path,
        alpha,
        [
            ("linear", linear_blend, palette[0]),
            ("geometric", geometric_blend, palette[1]),
            ("product", product_blend, palette[2]),
            ("max_mag", max_mag_blend, palette[3]),
            ("min_mag", min_mag_blend, palette[4]),
        ],
        title="Morph magnitude curves vs alpha (example magnitudes)",
        x_label="alpha (0=A, 1=B)",
        y_label="output magnitude",
    )
    entries.append(
        {
            "name": "Morph blend magnitude curves",
            "path": "assets/functions/morph_blend_magnitude_curves.svg",
            "usage": "Compares `linear`, `geometric`, `product`, `max_mag`, and `min_mag` blend behaviors.",
        }
    )

    x_mask = np.linspace(0.0, 2.0, 801, dtype=np.float64)
    mask_curve_p08 = np.power(np.maximum(x_mask, 0.0), 0.8)
    mask_curve_p10 = np.power(np.maximum(x_mask, 0.0), 1.0)
    mask_curve_p14 = np.power(np.maximum(x_mask, 0.0), 1.4)
    mask_path = FUNCTION_ASSETS_DIR / "mask_exponent_curves.svg"
    write_multiline_svg(
        mask_path,
        x_mask,
        [
            ("p=0.8", mask_curve_p08, palette[5]),
            ("p=1.0", mask_curve_p10, palette[0]),
            ("p=1.4", mask_curve_p14, palette[6]),
        ],
        title="Carrier/modulator mask exponent curves",
        x_label="normalized modulator magnitude",
        y_label="mask gain",
    )
    entries.append(
        {
            "name": "Mask exponent response",
            "path": "assets/functions/mask_exponent_curves.svg",
            "usage": "Shows how `--mask-exponent` shapes cross-synthesis mask gain.",
        }
    )

    phase_a = 0.0
    phase_b = 1.8
    phase_mix = alpha
    z = (1.0 - phase_mix) * np.exp(1j * phase_a) + phase_mix * np.exp(1j * phase_b)
    phase_angle = np.angle(z)
    phase_path = FUNCTION_ASSETS_DIR / "phase_mix_angle_curve.svg"
    write_multiline_svg(
        phase_path,
        phase_mix,
        [("output phase angle", phase_angle, palette[7])],
        title="Phase mix curve (example phase pair)",
        x_label="phase mix (0=A phase, 1=B phase)",
        y_label="output angle (radians)",
    )
    entries.append(
        {
            "name": "Phase mix to output angle",
            "path": "assets/functions/phase_mix_angle_curve.svg",
            "usage": "Example of complex-vector phase blending used by morph phase mix.",
        }
    )

    payload: dict[str, object] = {
        "commit": COMMIT_HASH,
        "commit_date": COMMIT_DATE,
        "plots": entries,
    }
    (DOCS_DIR / "function_gallery.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _natural_cubic_spline_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    n = int(x.size)
    if n < 3:
        return np.interp(xq, x, y)

    h = np.diff(x)
    if np.any(h <= 0.0):
        return np.interp(xq, x, y)

    a = h[:-1].copy()
    b = 2.0 * (h[:-1] + h[1:])
    c = h[1:].copy()
    d = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

    c_prime = np.zeros_like(c)
    d_prime = np.zeros_like(d)
    if b.size:
        c_prime[0] = c[0] / b[0] if c.size else 0.0
        d_prime[0] = d[0] / b[0]
        for i in range(1, d.size):
            denom = b[i] - a[i - 1] * c_prime[i - 1]
            if abs(denom) < 1e-12:
                return np.interp(xq, x, y)
            c_prime[i] = c[i] / denom if i < c.size else 0.0
            d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

    second = np.zeros(n, dtype=np.float64)
    if d.size:
        second[-2] = d_prime[-1]
        for i in range(d.size - 2, -1, -1):
            second[i + 1] = d_prime[i] - c_prime[i] * second[i + 2]

    idx = np.searchsorted(x, xq, side="right") - 1
    idx = np.clip(idx, 0, n - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    m0 = second[idx]
    m1 = second[idx + 1]
    hseg = x1 - x0
    hseg = np.where(hseg <= 0.0, 1.0, hseg)

    t0 = x1 - xq
    t1 = xq - x0
    yq = (
        (m0 * t0**3 + m1 * t1**3) / (6.0 * hseg)
        + (y0 - (m0 * hseg**2) / 6.0) * (t0 / hseg)
        + (y1 - (m1 * hseg**2) / 6.0) * (t1 / hseg)
    )
    return yq


def _sample_interpolation_curve(
    mode: str,
    x_control: np.ndarray,
    y_control: np.ndarray,
    x_query: np.ndarray,
    *,
    order: int = 3,
) -> np.ndarray:
    def _smoothstep(u: np.ndarray) -> np.ndarray:
        return (u * u) * (3.0 - 2.0 * u)

    def _smootherstep(u: np.ndarray) -> np.ndarray:
        return (u * u * u) * (u * (u * 6.0 - 15.0) + 10.0)

    def _exp_ease(u: np.ndarray, strength: float) -> np.ndarray:
        k = max(0.0, float(strength))
        if k <= 1e-9:
            return u
        den = np.expm1(k)
        if abs(den) <= 1e-12:
            return u
        return np.expm1(k * u) / den

    def _piecewise_ease(mode_name: str) -> np.ndarray:
        idx = np.searchsorted(x_control, x_query, side="right") - 1
        idx = np.clip(idx, 0, x_control.size - 2)
        x0 = x_control[idx]
        x1 = x_control[idx + 1]
        u = (x_query - x0) / np.maximum(1e-12, x1 - x0)
        u = np.clip(u, 0.0, 1.0)
        if mode_name == "exponential":
            eased = _exp_ease(u, max(1, int(order)))
        elif mode_name == "s_curve":
            eased = _smoothstep(u)
        else:
            eased = _smootherstep(u)
        return y_control[idx] + (y_control[idx + 1] - y_control[idx]) * eased

    mode_key = str(mode).strip().lower()
    if mode_key == "none":
        idx = np.searchsorted(x_control, x_query, side="right") - 1
        idx = np.clip(idx, 0, x_control.size - 1)
        return y_control[idx]
    if mode_key == "nearest":
        idx_r = np.searchsorted(x_control, x_query, side="left")
        idx_l = np.clip(idx_r - 1, 0, x_control.size - 1)
        idx_r = np.clip(idx_r, 0, x_control.size - 1)
        dl = np.abs(x_query - x_control[idx_l])
        dr = np.abs(x_control[idx_r] - x_query)
        use_left = dl <= dr
        idx = np.where(use_left, idx_l, idx_r)
        return y_control[idx]
    if mode_key == "linear":
        return np.interp(x_query, x_control, y_control)
    if mode_key == "cubic":
        return _natural_cubic_spline_eval(x_control, y_control, x_query)
    if mode_key in {"exponential", "s_curve", "smootherstep"}:
        return _piecewise_ease(mode_key)
    if mode_key == "polynomial":
        degree = min(max(1, int(order)), x_control.size - 1)
        coeffs = np.polyfit(x_control, y_control, deg=degree)
        return np.polyval(coeffs, x_query)
    raise ValueError(f"Unsupported interpolation mode: {mode}")


def _render_interpolation_svg(
    path: Path,
    x_dense: np.ndarray,
    y_dense: np.ndarray,
    x_control: np.ndarray,
    y_control: np.ndarray,
    *,
    title: str,
    y_min: float,
    y_max: float,
) -> None:
    width = 840
    height = 320
    margin = 36
    plot_w = max(1, width - 2 * margin)
    plot_h = max(1, height - 2 * margin)
    y_span = max(1e-9, y_max - y_min)

    def to_xy(xv: float, yv: float) -> tuple[float, float]:
        px = margin + plot_w * float(xv)
        py = margin + plot_h * (1.0 - ((float(yv) - y_min) / y_span))
        return px, py

    dense_points: list[str] = []
    for xv, yv in zip(x_dense, y_dense):
        px, py = to_xy(float(xv), float(yv))
        dense_points.append(f"{px:.2f},{py:.2f}")

    ctrl_points: list[str] = []
    circles: list[str] = []
    for idx, (xv, yv) in enumerate(zip(x_control, y_control)):
        px, py = to_xy(float(xv), float(yv))
        ctrl_points.append(f"{px:.2f},{py:.2f}")
        circles.append(
            f"<circle cx=\"{px:.2f}\" cy=\"{py:.2f}\" r=\"3.2\" fill=\"#b42318\" />"
            f"<text x=\"{px + 4:.2f}\" y=\"{py - 6:.2f}\" font-family=\"Arial, Helvetica, sans-serif\" "
            f"font-size=\"10\" fill=\"#7a271a\">p{idx}</text>"
        )

    legend_w = 250
    legend_h = 74
    legend_x = width - margin - legend_w
    legend_y = margin + 8

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#ffffff\" />
  <rect x=\"{margin}\" y=\"{margin}\" width=\"{width - 2 * margin}\" height=\"{height - 2 * margin}\" fill=\"#f7fbff\" stroke=\"#c8d9e6\" />
  <polyline fill=\"none\" stroke=\"#8f1d2c\" stroke-width=\"1.5\" stroke-dasharray=\"5 3\" points=\"{' '.join(ctrl_points)}\" />
  <polyline fill=\"none\" stroke=\"#005f73\" stroke-width=\"2\" points=\"{' '.join(dense_points)}\" />
  {''.join(circles)}
  <rect x=\"{legend_x}\" y=\"{legend_y}\" width=\"{legend_w}\" height=\"{legend_h}\" rx=\"6\" ry=\"6\" fill=\"#ffffff\" stroke=\"#c8d9e6\" />
  <line x1=\"{legend_x + 10}\" y1=\"{legend_y + 18}\" x2=\"{legend_x + 58}\" y2=\"{legend_y + 18}\" stroke=\"#005f73\" stroke-width=\"2\" />
  <text x=\"{legend_x + 66}\" y=\"{legend_y + 22}\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"10\" fill=\"#243b53\">interpolated curve u(t)</text>
  <line x1=\"{legend_x + 10}\" y1=\"{legend_y + 38}\" x2=\"{legend_x + 58}\" y2=\"{legend_y + 38}\" stroke=\"#8f1d2c\" stroke-width=\"1.5\" stroke-dasharray=\"5 3\" />
  <text x=\"{legend_x + 66}\" y=\"{legend_y + 42}\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"10\" fill=\"#243b53\">piecewise control polyline</text>
  <circle cx=\"{legend_x + 34}\" cy=\"{legend_y + 56}\" r=\"3.2\" fill=\"#b42318\" />
  <text x=\"{legend_x + 66}\" y=\"{legend_y + 60}\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"10\" fill=\"#243b53\">control points p_i</text>
  <text x=\"{width / 2:.1f}\" y=\"22\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"15\" fill=\"#102a43\">{title}</text>
  <text x=\"{width / 2:.1f}\" y=\"{height - 8}\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#334e68\">normalized control time</text>
  <text x=\"14\" y=\"{height / 2:.1f}\" transform=\"rotate(-90 14,{height / 2:.1f})\" text-anchor=\"middle\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"12\" fill=\"#334e68\">control value</text>
  <text x=\"{margin}\" y=\"{height - margin + 18}\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">0</text>
  <text x=\"{width - margin}\" y=\"{height - margin + 18}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">1</text>
  <text x=\"{margin - 6}\" y=\"{margin + 4}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{y_max:.2f}</text>
  <text x=\"{margin - 6}\" y=\"{height - margin + 4}\" text-anchor=\"end\" font-family=\"Arial, Helvetica, sans-serif\" font-size=\"11\" fill=\"#486581\">{y_min:.2f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def generate_interpolation_assets() -> dict[str, object]:
    x_control = np.asarray([0.0, 0.16, 0.39, 0.63, 0.82, 1.0], dtype=np.float64)
    y_control = np.asarray([1.00, 1.48, 0.82, 1.56, 1.18, 1.34], dtype=np.float64)
    x_dense = np.linspace(0.0, 1.0, 1200, dtype=np.float64)

    specs: list[tuple[str, str, int | None]] = [
        ("none", "none (stairstep)", None),
        ("nearest", "nearest", None),
        ("linear", "linear", None),
        ("cubic", "cubic", None),
        ("exponential", "exponential", None),
        ("s_curve", "S-curve (smoothstep)", None),
        ("smootherstep", "S-curve (smootherstep)", None),
        ("polynomial", "polynomial order 1", 1),
        ("polynomial", "polynomial order 2", 2),
        ("polynomial", "polynomial order 3", 3),
        ("polynomial", "polynomial order 5", 5),
    ]

    curves: list[tuple[str, np.ndarray, str, int | None]] = []
    for mode, label, order in specs:
        sampled = _sample_interpolation_curve(mode, x_control, y_control, x_dense, order=order or 3)
        curves.append((mode, sampled, label, order))

    y_stack = np.concatenate([y_control, *(curve for _, curve, _, _ in curves)])
    y_min = float(np.min(y_stack))
    y_max = float(np.max(y_stack))
    pad = max(0.05, 0.08 * (y_max - y_min))
    y_min -= pad
    y_max += pad

    plot_entries: list[dict[str, object]] = []
    for mode, sampled, label, order in curves:
        file_name = (
            f"interp_{mode}_order_{int(order)}.svg"
            if order is not None
            else f"interp_{mode}.svg"
        )
        out_path = INTERPOLATION_ASSETS_DIR / file_name
        _render_interpolation_svg(
            out_path,
            x_dense,
            sampled,
            x_control,
            y_control,
            title=f"{label} interpolation",
            y_min=y_min,
            y_max=y_max,
        )
        plot_entries.append(
            {
                "mode": mode,
                "label": label,
                "order": order,
                "path": f"assets/interpolation/{file_name}",
            }
        )

    payload: dict[str, object] = {
        "commit": COMMIT_HASH,
        "commit_date": COMMIT_DATE,
        "control_points": [
            {"time_sec": float(tx), "value": float(vx)}
            for tx, vx in zip(x_control, y_control)
        ],
        "plots": plot_entries,
    }
    (DOCS_DIR / "interpolation_gallery.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def generate_window_assets_and_metrics(entries: list[dict[str, str]]) -> dict[str, dict[str, float | str]]:
    metrics_by_name: dict[str, dict[str, float | str]] = {}
    for entry in entries:
        name = entry["name"]
        samples = window_samples(name)
        metrics = compute_window_metrics(samples)

        time_plot_path = WINDOW_ASSETS_DIR / f"{name}_time.svg"
        freq_plot_path = WINDOW_ASSETS_DIR / f"{name}_freq.svg"

        write_line_svg(
            time_plot_path,
            samples,
            title=f"{name} window (time domain)",
            y_min=0.0,
            y_max=max(1.0, float(np.max(samples))),
            x_label="normalized sample index",
            y_label="amplitude",
        )

        fft_len = 1
        while fft_len < samples.size * 64:
            fft_len *= 2
        spectrum = np.abs(np.fft.rfft(samples, n=fft_len))
        spectrum /= max(1e-15, float(np.max(spectrum)))
        spectrum_db = 20.0 * np.log10(np.maximum(spectrum, 1e-8))
        spectrum_db = np.clip(spectrum_db, -160.0, 5.0)

        write_line_svg(
            freq_plot_path,
            spectrum_db,
            title=f"{name} window (magnitude spectrum)",
            y_min=-160.0,
            y_max=0.0,
            x_label="normalized frequency",
            y_label="magnitude (dB)",
        )

        metrics_by_name[name] = {
            **metrics,
            "time_plot": f"assets/windows/{name}_time.svg",
            "freq_plot": f"assets/windows/{name}_freq.svg",
        }

    metrics_payload = {
        "commit": COMMIT_HASH,
        "commit_date": COMMIT_DATE,
        "windows": metrics_by_name,
    }
    (DOCS_DIR / "window_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics_by_name


def write_math_foundations(interpolation_gallery: dict[str, object], function_gallery: dict[str, object]) -> None:
    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Mathematical Foundations")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append("This document explains the core signal-processing equations used by pvx, with plain-English interpretation.")
    lines.append("All equations are written in GitHub-renderable LaTeX and are intended to render directly in normal GitHub Markdown view.")
    lines.append("")
    lines.append("## Notation")
    lines.append("")
    lines.append("- Analysis hop: $H_a$")
    lines.append("- Synthesis hop: $H_s$")
    lines.append("- FFT size: $N$")
    lines.append("- Frame index: $t$")
    lines.append("- Frequency-bin index: $k$")
    lines.append("- Frame phase: $\\phi_t[k]$")
    lines.append("- Bin-center angular frequency: $\\omega_k=2\\pi k/N$")
    lines.append("- Selected transform backend: $m \\in \\{\\text{fft},\\text{dft},\\text{czt},\\text{dct},\\text{dst},\\text{hartley}\\}$")
    lines.append("")
    lines.append("## 1. STFT Analysis and Synthesis")
    lines.append("")
    lines.append("Analysis transform:")
    lines.append("")
    lines.append("$$")
    lines.append("X_t[k]=\\sum_{n=0}^{N-1} x[n+tH_a]w[n]e^{-j2\\pi kn/N}")
    lines.append("$$")
    lines.append("")
    lines.append("Plain English: each frame `t` is windowed by `w[n]` and transformed into complex frequency bins `k`.")
    lines.append("")
    lines.append("Weighted overlap-add synthesis:")
    lines.append("")
    lines.append("$$")
    lines.append("\\hat{x}[n]=\\frac{\\sum_t \\hat{x}_t[n-tH_s]w[n-tH_s]}{\\sum_t w^2[n-tH_s]+\\varepsilon}")
    lines.append("$$")
    lines.append("")
    lines.append("Plain English: reconstructed frames are overlap-added, then normalized by accumulated window energy.")
    lines.append("")
    lines.append("## 2. Transform Backend Families and Tradeoffs")
    lines.append("")
    lines.append("pvx allows selecting the per-frame transform with `--transform` in STFT/ISTFT paths.")
    lines.append("The same overlap-add framework is used, but per-bin meaning and phase behavior differ by transform family.")
    lines.append("")
    lines.append("Complex Fourier family (`fft`, `dft`):")
    lines.append("")
    lines.append("$$")
    lines.append("X_t[k]=\\sum_{n=0}^{N-1} x_t[n]e^{-j2\\pi kn/N},\\qquad")
    lines.append("x_t[n]=\\frac{1}{N}\\sum_{k=0}^{N-1}X_t[k]e^{j2\\pi kn/N}")
    lines.append("$$")
    lines.append("")
    lines.append("Chirp-Z (`czt`, Bluestein-style contour):")
    lines.append("")
    lines.append("$$")
    lines.append("X_t[k]=\\sum_{n=0}^{N-1}x_t[n]A^{-n}W^{nk}")
    lines.append("$$")
    lines.append("")
    lines.append("With pvx defaults $A=1$ and $W=e^{-j2\\pi/N}$, CZT matches DFT samples but uses a different numerical path.")
    lines.append("")
    lines.append("DCT-II / IDCT-II (`dct`):")
    lines.append("")
    lines.append("$$")
    lines.append("C_t[k]=\\alpha_k\\sum_{n=0}^{N-1}x_t[n]\\cos\\left(\\frac{\\pi}{N}(n+\\tfrac{1}{2})k\\right)")
    lines.append("$$")
    lines.append("")
    lines.append("DST-II / IDST-II (`dst`):")
    lines.append("")
    lines.append("$$")
    lines.append("S_t[k]=\\beta_k\\sum_{n=0}^{N-1}x_t[n]\\sin\\left(\\frac{\\pi}{N}(n+\\tfrac{1}{2})(k+1)\\right)")
    lines.append("$$")
    lines.append("")
    lines.append("Discrete Hartley (`hartley`):")
    lines.append("")
    lines.append("$$")
    lines.append("H_t[k]=\\sum_{n=0}^{N-1}x_t[n]\\,\\mathrm{cas}\\left(\\frac{2\\pi kn}{N}\\right),\\quad")
    lines.append("\\mathrm{cas}(\\theta)=\\cos(\\theta)+\\sin(\\theta)")
    lines.append("$$")
    lines.append("")
    lines.append("| Transform | Strengths | Tradeoffs | Recommended use cases |")
    lines.append("| --- | --- | --- | --- |")
    lines.append("| `fft` | Fastest common path, stable, native one-sided complex bins, best CUDA coverage. | Needs careful window/hop tuning for extreme nonstationarity. | Default for most time-stretch, pitch-shift, and batch processing. |")
    lines.append("| `dft` | Canonical Fourier definition, deterministic parity baseline versus FFT implementations. | Usually slower than `fft`, little practical quality advantage in normal runs. | Cross-checking, verification, or research comparisons. |")
    lines.append("| `czt` | Useful alternative numerical path for awkward frame sizes; can be robust for prime/non-power sizes. | Requires SciPy; typically CPU-only and slower than FFT. | Edge-case frame-size experiments and diagnostic reruns. |")
    lines.append("| `dct` | Real-valued compact basis with strong energy compaction for smooth envelopes. | Loses explicit complex phase; less transparent for strict phase-coherent resynthesis. | Spectral shaping, denoise-like creative processing, coefficient-domain experiments. |")
    lines.append("| `dst` | Real-valued odd-symmetry basis; emphasizes different boundary behavior than DCT. | Same phase limitations as DCT; content-dependent coloration. | Creative timbre variants and odd-symmetry analysis experiments. |")
    lines.append("| `hartley` | Real transform using `cas`, often helpful for CPU-friendly exploratory analysis. | Different bin semantics vs complex STFT; can alter artifact profile. | Alternative real-basis phase-vocoder experiments and pedagogical comparisons. |")
    lines.append("")
    lines.append("Plain English: choose `fft` first, then move to other transforms when you specifically need different numerical behavior, basis structure, or artifact character.")
    lines.append("")
    lines.append("### Sample Transform Use Cases")
    lines.append("")
    lines.append("```bash")
    lines.append("pvx voc dialog.wav --transform fft --time-stretch 1.08 --transient-preserve --output-dir out")
    lines.append("pvx voc test_tones.wav --transform dft --time-stretch 1.00 --output-dir out")
    lines.append("pvx voc archival_take.wav --transform czt --n-fft 1536 --win-length 1536 --hop-size 384 --output-dir out")
    lines.append("pvx voc strings.wav --transform dct --pitch-shift-cents -17 --soft-clip-level 0.94 --output-dir out")
    lines.append("pvx voc percussion.wav --transform dst --time-stretch 0.92 --output-dir out")
    lines.append("pvx voc synth.wav --transform hartley --phase-locking off --time-stretch 1.30 --output-dir out")
    lines.append("```")
    lines.append("")
    lines.append("## 3. Phase-Vocoder Frequency and Phase Update")
    lines.append("")
    lines.append("$$")
    lines.append("\\Delta\\phi_t[k]=\\mathrm{princarg}\\left(\\phi_t[k]-\\phi_{t-1}[k]-\\omega_kH_a\\right)")
    lines.append("$$")
    lines.append("$$")
    lines.append("\\hat{\\omega}_t[k]=\\omega_k+\\frac{\\Delta\\phi_t[k]}{H_a}")
    lines.append("$$")
    lines.append("$$")
    lines.append("\\hat{\\phi}_t[k]=\\hat{\\phi}_{t-1}[k]+\\hat{\\omega}_t[k]H_s")
    lines.append("$$")
    lines.append("")
    lines.append("Plain English: pvx estimates true per-bin frequency from wrapped phase deviation, then re-accumulates phase at the synthesis hop.")
    lines.append("")
    lines.append("## 4. Time Stretch and Pitch Ratio")
    lines.append("")
    lines.append("Pitch shift ratio from semitones/cents:")
    lines.append("")
    lines.append("$$")
    lines.append("r_{\\text{pitch}}=2^{\\Delta s/12}=2^{\\Delta c/1200}")
    lines.append("$$")
    lines.append("")
    lines.append("Plain English: semitone and cent controls map to the same multiplicative ratio.")
    lines.append("")
    lines.append("## 5. Dynamic Control Interpolation (CSV/JSON)")
    lines.append("")
    lines.append("When a numeric flag receives a CSV/JSON control track, pvx samples a control function $u(t)$ over render time.")
    lines.append("")
    lines.append("Polynomial mode fits a global polynomial of degree $d$:")
    lines.append("")
    lines.append("$$")
    lines.append("u(t)=\\sum_{i=0}^{d} a_i t^i,\\qquad d=\\min(\\texttt{order}, M-1)")
    lines.append("$$")
    lines.append("")
    lines.append("where $M$ is the number of control points. In command-line interface (CLI) usage, `--order` accepts any integer $\\ge 1$.")
    lines.append("The effective degree is automatically capped to avoid over-specification when there are too few points.")
    lines.append("")
    lines.append("Nearest/linear/cubic/exponential/S-curve are local interpolation modes; `none` is sample-and-hold (stairstep).")
    lines.append("`exponential` uses piecewise exponential easing and `s_curve`/`smootherstep` use Hermite S-shaped easing between adjacent control points.")
    lines.append("Legend used in each plot: blue solid line = interpolated control curve $u(t)$, red dashed line = piecewise connection of control points, red circles labeled $p_i$ = original control points.")
    lines.append("")
    lines.append("| Interpolation mode | CLI form | Example plot |")
    lines.append("| --- | --- | --- |")
    plots = interpolation_gallery.get("plots", []) if isinstance(interpolation_gallery, dict) else []
    for plot in plots:
        if not isinstance(plot, dict):
            continue
        label = str(plot.get("label", plot.get("mode", "unknown")))
        mode = str(plot.get("mode", "unknown"))
        order = plot.get("order")
        if order is None:
            cli = f"`--interp {mode}`"
        else:
            cli = f"`--interp {mode} --order {int(order)}`"
        path = str(plot.get("path", "")).strip()
        if path:
            plot_cell = f"![{label}]({path})"
        else:
            plot_cell = "n/a"
        lines.append(f"| {label} | {cli} | {plot_cell} |")
    lines.append("")
    lines.append("## 6. Microtonal Retune Mapping")
    lines.append("")
    lines.append("For a detected frequency $f$, pvx scale-aware retuning chooses the nearest permitted scale target $f_{\\text{scale}}$ and applies:")
    lines.append("")
    lines.append("$$")
    lines.append("\\Delta s = 12\\log_2\\left(\\frac{f_{\\text{scale}}}{f}\\right)")
    lines.append("$$")
    lines.append("")
    lines.append("Plain English: each frame is shifted only as much as needed to land on the selected microtonal scale degree.")
    lines.append("")
    lines.append("## 7. Loudness and Dynamics")
    lines.append("")
    lines.append("LUFS target gain:")
    lines.append("")
    lines.append("$$")
    lines.append("g_{\\text{LUFS}}=10^{(L_{\\text{target}}-L_{\\text{in}})/20},\\qquad y[n]=g_{\\text{LUFS}}x[n]")
    lines.append("$$")
    lines.append("")
    lines.append("Compressor gain law (conceptual):")
    lines.append("")
    lines.append("$$")
    lines.append("g(x)=\\begin{cases}")
    lines.append("1, & |x|\\le T \\\\")
    lines.append("\\left(\\frac{T+(|x|-T)/R}{|x|}\\right), & |x|>T")
    lines.append("\\end{cases}")
    lines.append("$$")
    lines.append("")
    lines.append("Plain English: level is reduced above threshold $T$ by ratio $R$, then makeup/loudness stages may be applied.")
    lines.append("")
    lines.append("## 8. Spatial Coherence and Channel Alignment")
    lines.append("")
    lines.append("Inter-channel phase difference:")
    lines.append("")
    lines.append("$$")
    lines.append("\\Delta\\phi_{ij}(k,t)=\\phi_i(k,t)-\\phi_j(k,t)")
    lines.append("$$")
    lines.append("")
    lines.append("Phase-drift objective:")
    lines.append("")
    lines.append("$$")
    lines.append("J=\\sum_{k,t}\\left|\\Delta\\phi^{\\text{out}}_{ij}(k,t)-\\Delta\\phi^{\\text{in}}_{ij}(k,t)\\right|")
    lines.append("$$")
    lines.append("")
    lines.append("Channel delay estimate by phase-weighted cross-correlation:")
    lines.append("")
    lines.append("$$")
    lines.append("\\tau^*=\\arg\\max_\\tau\\,\\mathcal{F}^{-1}\\left\\{\\frac{X_i(\\omega)X_j^*(\\omega)}{|X_i(\\omega)X_j^*(\\omega)|+\\varepsilon}\\right\\}(\\tau)")
    lines.append("$$")
    lines.append("")
    lines.append("Plain English: pvx spatial modules prioritize channel coherence, stable image cues, and robust delay alignment for multichannel processing chains.")
    lines.append("")
    lines.append("## 9. Function Graph Gallery")
    lines.append("")
    lines.append("The plots below visualize core transfer functions and parameter curves used across pvx processing modules.")
    lines.append("")
    lines.append("| Function family | Plot | Why this matters |")
    lines.append("| --- | --- | --- |")
    function_plots = function_gallery.get("plots", []) if isinstance(function_gallery, dict) else []
    for plot in function_plots:
        if not isinstance(plot, dict):
            continue
        name = str(plot.get("name", "Function"))
        path = str(plot.get("path", "")).strip()
        usage = str(plot.get("usage", ""))
        image_cell = f"![{name}]({path})" if path else "n/a"
        lines.append(f"| {name} | {image_cell} | {usage} |")
    lines.append("")
    lines.extend(attribution_section_lines())
    (DOCS_DIR / "MATHEMATICAL_FOUNDATIONS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_window_reference() -> None:
    entries = window_entries()
    metrics_by_name = generate_window_assets_and_metrics(entries)

    lines: list[str] = []
    lines.extend(logo_lines())
    lines.append("# pvx Window Reference")
    lines.append("")
    lines.extend(generated_stamp_lines())
    lines.append(f"pvx currently supports **{len(entries)}** analysis windows. This file defines each one mathematically and explains it in plain English.")
    lines.append("")
    lines.append("## Notation")
    lines.append("")
    lines.append("- Window length: $N$")
    lines.append("- Sample index: $n \\in \\{0,\\dots,N-1\\}$")
    lines.append("- Center index: $m=(N-1)/2$")
    lines.append("- Normalized center coordinate: $x_n=(n-m)/m$")
    lines.append("")
    lines.append("## Formula Key")
    lines.append("")
    lines.append("**(W0) Rectangular**")
    lines.append("")
    lines.append("$$w[n]=1$$")
    lines.append("")
    lines.append("**(W1) Cosine series**")
    lines.append("")
    lines.append("$$w[n]=\\sum_{k=0}^{K} a_k\\cos\\left(\\frac{2\\pi k n}{N-1}\\right)$$")
    lines.append("")
    lines.append("**(W2) Sine/Cosine**")
    lines.append("")
    lines.append("$$w[n]=\\sin\\left(\\frac{\\pi n}{N-1}\\right)$$")
    lines.append("")
    lines.append("**(W3) Bartlett**")
    lines.append("")
    lines.append("$$w[n]=1-\\left|\\frac{n-m}{m}\\right|$$")
    lines.append("")
    lines.append("**(W4) Triangular**")
    lines.append("")
    lines.append("$$w[n]=\\max\\left(1-\\left|\\frac{n-m}{(N+1)/2}\\right|,0\\right)$$")
    lines.append("")
    lines.append("**(W5) Bartlett-Hann**")
    lines.append("")
    lines.append("$$x=\\frac{n}{N-1}-\\frac{1}{2},\\quad w[n]=0.62-0.48|x|+0.38\\cos(2\\pi x)$$")
    lines.append("")
    lines.append("**(W6) Tukey**")
    lines.append("")
    lines.append("$$")
    lines.append("x=\\frac{n}{N-1},\\quad")
    lines.append("w[n]=\\begin{cases}")
    lines.append("\\frac{1}{2}\\left(1+\\cos\\left(\\pi\\left(\\frac{2x}{\\alpha}-1\\right)\\right)\\right), & 0\\le x<\\frac{\\alpha}{2} \\\\")
    lines.append("1, & \\frac{\\alpha}{2}\\le x<1-\\frac{\\alpha}{2} \\\\")
    lines.append("\\frac{1}{2}\\left(1+\\cos\\left(\\pi\\left(\\frac{2x}{\\alpha}-\\frac{2}{\\alpha}+1\\right)\\right)\\right), & 1-\\frac{\\alpha}{2}\\le x\\le 1")
    lines.append("\\end{cases}")
    lines.append("$$")
    lines.append("")
    lines.append("Special cases in pvx: $\\alpha\\le 0$ gives rectangular behavior, and $\\alpha\\ge 1$ collapses to Hann.")
    lines.append("")
    lines.append("**(W7) Parzen**")
    lines.append("")
    lines.append("$$")
    lines.append("u=\\left|\\frac{2n}{N-1}-1\\right|,\\quad")
    lines.append("w[n]=\\begin{cases}")
    lines.append("1-6u^2+6u^3, & 0\\le u\\le \\frac{1}{2} \\\\")
    lines.append("2(1-u)^3, & \\frac{1}{2}<u\\le 1 \\\\")
    lines.append("0, & u>1")
    lines.append("\\end{cases}")
    lines.append("$$")
    lines.append("")
    lines.append("**(W8) Lanczos**")
    lines.append("")
    lines.append("$$w[n]=\\mathrm{sinc}\\left(\\frac{2n}{N-1}-1\\right)$$")
    lines.append("")
    lines.append("**(W9) Welch**")
    lines.append("")
    lines.append("$$w[n]=\\max\\left(1-x_n^2,0\\right)$$")
    lines.append("")
    lines.append("**(W10) Gaussian**")
    lines.append("")
    lines.append("$$w[n]=\\exp\\left(-\\frac{1}{2}\\left(\\frac{n-m}{\\sigma}\\right)^2\\right),\\quad \\sigma=r_\\sigma m$$")
    lines.append("")
    lines.append("**(W11) General Gaussian**")
    lines.append("")
    lines.append("$$w[n]=\\exp\\left(-\\frac{1}{2}\\left|\\frac{n-m}{\\sigma}\\right|^{2p}\\right)$$")
    lines.append("")
    lines.append("**(W12) Exponential**")
    lines.append("")
    lines.append("$$w[n]=\\exp\\left(-\\frac{|n-m|}{\\tau}\\right),\\quad \\tau=r_\\tau m$$")
    lines.append("")
    lines.append("**(W13) Cauchy**")
    lines.append("")
    lines.append("$$w[n]=\\frac{1}{1+\\left(\\frac{n-m}{\\gamma}\\right)^2},\\quad \\gamma=r_\\gamma m$$")
    lines.append("")
    lines.append("**(W14) Cosine power**")
    lines.append("")
    lines.append("$$w[n]=\\sin\\left(\\frac{\\pi n}{N-1}\\right)^p$$")
    lines.append("")
    lines.append("**(W15) Hann-Poisson**")
    lines.append("")
    lines.append("$$w[n]=w_{\\text{Hann}}[n]\\exp\\left(-\\alpha\\frac{|n-m|}{m}\\right)$$")
    lines.append("")
    lines.append("**(W16) General Hamming**")
    lines.append("")
    lines.append("$$w[n]=\\alpha-(1-\\alpha)\\cos\\left(\\frac{2\\pi n}{N-1}\\right)$$")
    lines.append("")
    lines.append("**(W17) Bohman**")
    lines.append("")
    lines.append("$$x=\\left|\\frac{2n}{N-1}-1\\right|,\\quad w[n]=(1-x)\\cos(\\pi x)+\\frac{\\sin(\\pi x)}{\\pi}$$")
    lines.append("")
    lines.append("**(W18) Kaiser-Bessel**")
    lines.append("")
    lines.append("$$w[n]=\\frac{I_0\\left(\\beta\\sqrt{1-r_n^2}\\right)}{I_0(\\beta)},\\quad r_n=\\frac{n-m}{m}$$")
    lines.append("")
    lines.append("Each supported pvx window maps to one of the formula families above with the per-window constants shown below.")
    lines.append("")
    lines.append("## Quantitative Metrics")
    lines.append("")
    lines.append("- Coherent gain: $CG=\\frac{1}{N}\\sum_{n=0}^{N-1}w[n]$")
    lines.append("- Equivalent noise bandwidth (bins): $ENBW=\\frac{N\\sum_n w[n]^2}{(\\sum_n w[n])^2}$")
    lines.append("- Scalloping loss (dB): response ratio at a half-bin sinusoid offset")
    lines.append("- Main-lobe width (bins): measured from the first post-DC local minimum in zero-padded FFT magnitude")
    lines.append("- Peak sidelobe (dB): maximum sidelobe level outside the main lobe")
    lines.append("")
    lines.append("## Complete Window Catalog")
    lines.append("")
    lines.append("| Window | Family | Parameters | Formula | Coherent gain | ENBW (bins) | Scalloping loss (dB) | Main-lobe width (bins) | Peak sidelobe (dB) | Plots | Pros | Cons | Usage advice |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for entry in entries:
        name = entry["name"]
        m = metrics_by_name[name]
        lines.append(
            "| `{name}` | {family} | `{params}` | {formula} | {cg:.6f} | {enbw:.6f} | {scallop:.3f} | {mlw:.3f} | {psl:.3f} | [time]({time_plot}) / [freq]({freq_plot}) | {pros} | {cons} | {usage} |".format(
                name=name,
                family=entry["family"],
                params=entry["params"],
                formula=entry["formula"],
                cg=float(m["coherent_gain"]),
                enbw=float(m["enbw_bins"]),
                scallop=float(m["scalloping_loss_db"]),
                mlw=float(m["main_lobe_width_bins"]),
                psl=float(m["peak_sidelobe_db"]),
                time_plot=str(m["time_plot"]),
                freq_plot=str(m["freq_plot"]),
                pros=entry["pros"],
                cons=entry["cons"],
                usage=entry["usage"],
            )
        )

    lines.append("")
    lines.append("## Complete Plot Gallery (All Windows)")
    lines.append("")
    lines.append("| Window | Time-domain shape | Magnitude spectrum |")
    lines.append("| --- | --- | --- |")
    for entry in entries:
        name = entry["name"]
        lines.append(
            f"| `{name}` | ![{name} time](assets/windows/{name}_time.svg) | ![{name} freq](assets/windows/{name}_freq.svg) |"
        )

    lines.append("")
    lines.append("## Practical Selection Guidance")
    lines.append("")
    lines.append("- Use `hann` or `hamming` for balanced everyday phase-vocoder work.")
    lines.append("- Use `kaiser` when you need explicit sidelobe control with `--kaiser-beta`.")
    lines.append("- Use `flattop` for amplitude-accuracy-focused spectral measurement.")
    lines.append("- Use `tukey_*` when you want a controllable flat center region.")
    lines.append("- Use Gaussian/Cauchy/Exponential families to experiment with edge-decay shape and time-locality.")
    lines.append("")
    lines.extend(attribution_section_lines())
    (DOCS_DIR / "WINDOW_REFERENCE.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    interpolation_gallery = generate_interpolation_assets()
    function_gallery = generate_function_assets()
    write_math_foundations(interpolation_gallery, function_gallery)
    write_window_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
