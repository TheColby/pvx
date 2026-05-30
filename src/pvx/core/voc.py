#!/usr/bin/env python3

"""Multi-channel phase vocoder CLI for time and pitch manipulation."""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import sys
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

try:
    import numpy as np
except (ImportError, ModuleNotFoundError):  # pragma: no cover - dependency guard
    np = None

try:
    import soundfile as sf
except (ImportError, ModuleNotFoundError):  # pragma: no cover - dependency guard
    sf = None

try:
    from scipy.signal import resample as scipy_resample
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    scipy_resample = None

try:
    from scipy import fft as scipy_fft
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    scipy_fft = None

try:
    from scipy.signal import czt as scipy_czt
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    scipy_czt = None

try:
    from scipy.interpolate import CubicSpline as scipy_cubic_spline  # noqa: N813
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    scipy_cubic_spline = None

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    cp = None

try:
    from cupyx.scipy.signal import resample as cupyx_resample
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    cupyx_resample = None

from pvx.core import voc_console as _voc_console
from pvx.core import voc_profiles as _voc_profiles
from pvx.core.control_bus import (
    ControlRoute,
    apply_control_routes_csv,
)
from pvx.core.stereo import lr_to_ms, ms_to_lr, validate_ref_channel
from pvx.core.transients import detect_transient_regions, map_mask_to_output, smooth_binary_mask
from pvx.core.voc_console import (
    VERBOSITY_TO_LEVEL,
    console_level,
)
from pvx.core.wsola import wsola_time_stretch

ProgressBar = _voc_console.ProgressBar
clone_args_namespace = _voc_console.clone_args_namespace
is_quiet = _voc_console.is_quiet
log_message = _voc_console.log_message
QUALITY_PROFILE_CHOICES = _voc_profiles.QUALITY_PROFILE_CHOICES
apply_quality_profile_overrides = _voc_profiles.apply_quality_profile_overrides
estimate_content_features = _voc_profiles.estimate_content_features
suggest_quality_profile = _voc_profiles.suggest_quality_profile

WINDOW_CHOICES = (
    "hann",
    "hamming",
    "blackman",
    "blackmanharris",
    "nuttall",
    "flattop",
    "blackman_nuttall",
    "exact_blackman",
    "sine",
    "bartlett",
    "boxcar",
    "triangular",
    "bartlett_hann",
    "tukey",
    "tukey_0p1",
    "tukey_0p25",
    "tukey_0p75",
    "tukey_0p9",
    "parzen",
    "lanczos",
    "welch",
    "gaussian_0p25",
    "gaussian_0p35",
    "gaussian_0p45",
    "gaussian_0p55",
    "gaussian_0p65",
    "general_gaussian_1p5_0p35",
    "general_gaussian_2p0_0p35",
    "general_gaussian_3p0_0p35",
    "general_gaussian_4p0_0p35",
    "exponential_0p25",
    "exponential_0p5",
    "exponential_1p0",
    "cauchy_0p5",
    "cauchy_1p0",
    "cauchy_2p0",
    "cosine_power_2",
    "cosine_power_3",
    "cosine_power_4",
    "hann_poisson_0p5",
    "hann_poisson_1p0",
    "hann_poisson_2p0",
    "general_hamming_0p50",
    "general_hamming_0p60",
    "general_hamming_0p70",
    "general_hamming_0p80",
    "bohman",
    "cosine",
    "kaiser",
    "rect",
)

WindowType = str
TransformMode = Literal["fft", "dft", "czt", "dct", "dst", "hartley"]
ResampleMode = Literal["auto", "fft", "linear"]
PhaseLockMode = Literal["off", "identity"]
PhaseEngineMode = Literal["propagate", "hybrid", "random"]
DeviceMode = Literal["auto", "cpu", "cuda"]
LowConfidenceMode = Literal["hold", "unity", "interp"]
TransientMode = Literal["off", "reset", "hybrid", "wsola"]
StereoMode = Literal["independent", "mid_side_lock", "ref_channel_lock"]
ProgressCallback = Callable[[int, int], None]
ControlInterpolationMode = Literal[
    "none",
    "linear",
    "nearest",
    "cubic",
    "polynomial",
    "exponential",
    "s_curve",
    "smootherstep",
]

TRANSFORM_CHOICES: tuple[TransformMode, ...] = ("fft", "dft", "czt", "dct", "dst", "hartley")
PHASE_ENGINE_CHOICES: tuple[PhaseEngineMode, ...] = ("propagate", "hybrid", "random")
CONTROL_INTERP_CHOICES: tuple[ControlInterpolationMode, ...] = (
    "none",
    "linear",
    "nearest",
    "cubic",
    "polynomial",
    "exponential",
    "s_curve",
    "smootherstep",
)

_DYNAMIC_NUMERIC_ARG_SPECS: tuple[tuple[str, str, Literal["float", "int"], float], ...] = (
    ("time_stretch", "time_stretch", "float", 1.0),
    ("extreme_stretch_threshold", "extreme_stretch_threshold", "float", 2.0),
    ("max_stage_stretch", "max_stage_stretch", "float", 1.8),
    ("n_fft", "n_fft", "int", 2048.0),
    ("win_length", "win_length", "int", 2048.0),
    ("hop_size", "hop_size", "int", 512.0),
    ("kaiser_beta", "kaiser_beta", "float", 14.0),
    ("fourier_sync_min_fft", "fourier_sync_min_fft", "int", 256.0),
    ("fourier_sync_max_fft", "fourier_sync_max_fft", "int", 8192.0),
    ("fourier_sync_smooth", "fourier_sync_smooth", "int", 5.0),
    ("ambient_phase_mix", "ambient_phase_mix", "float", 0.5),
    ("transient_threshold", "transient_threshold", "float", 2.0),
    ("transient_sensitivity", "transient_sensitivity", "float", 0.5),
    ("transient_protect_ms", "transient_protect_ms", "float", 30.0),
    ("transient_crossfade_ms", "transient_crossfade_ms", "float", 10.0),
    ("coherence_strength", "coherence_strength", "float", 0.0),
    ("onset_credit_pull", "onset_credit_pull", "float", 0.5),
    ("onset_credit_max", "onset_credit_max", "float", 8.0),
    ("formant_lifter", "formant_lifter", "int", 32.0),
    ("formant_strength", "formant_strength", "float", 1.0),
    ("formant_max_gain_db", "formant_max_gain_db", "float", 12.0),
)


@dataclass(frozen=True)
class VocoderConfig:
    n_fft: int
    win_length: int
    hop_size: int
    window: WindowType
    center: bool
    phase_locking: PhaseLockMode
    transient_preserve: bool
    transient_threshold: float
    kaiser_beta: float = 14.0
    transform: TransformMode = "fft"
    phase_engine: PhaseEngineMode = "propagate"
    ambient_phase_mix: float = 0.5
    phase_random_seed: int | None = None
    onset_time_credit: bool = False
    onset_credit_pull: float = 0.5
    onset_credit_max: float = 8.0
    onset_realign: bool = True


@dataclass(frozen=True)
class PitchConfig:
    ratio: float
    source_f0_hz: float | None = None


@dataclass(frozen=True)
class ControlSegment:
    start_sec: float
    end_sec: float
    stretch: float
    pitch_ratio: float
    confidence: float | None = None
    overrides: dict[str, Any] | None = None


@dataclass(frozen=True)
class DynamicControlRef:
    parameter: str
    path: Path
    value_kind: Literal["float", "int", "pitch_ratio", "pitch_semitones", "pitch_cents"]
    interpolation: ControlInterpolationMode
    order: int


@dataclass(frozen=True)
class DynamicControlSignal:
    parameter: str
    interpolation: ControlInterpolationMode
    order: int
    times_sec: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class JobResult:
    input_path: Path
    output_path: Path
    in_sr: int
    out_sr: int
    in_samples: int
    out_samples: int
    channels: int
    stretch: float
    pitch_ratio: float
    stage_count: int = 1
    control_map_segments: int = 0
    quality_profile: str = "neutral"
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class FourierSyncPlan:
    frame_lengths: np.ndarray
    f0_track_hz: np.ndarray
    reference_n_fft: int


@dataclass(frozen=True)
class AudioBlockResult:
    audio: np.ndarray
    internal_stretch: float
    sync_plan: FourierSyncPlan | None
    stage_count: int = 1


@dataclass(frozen=True)
class RuntimeConfig:
    requested_device: DeviceMode
    active_device: Literal["cpu", "cuda"]
    cuda_device: int
    fallback_reason: str | None = None


_RUNTIME_CONFIG = RuntimeConfig(
    requested_device="auto",
    active_device="cpu",
    cuda_device=0,
    fallback_reason=None,
)


_COSINE_SERIES_WINDOWS: dict[str, tuple[float, ...]] = {
    "hann": (0.5, -0.5),
    "hamming": (0.54, -0.46),
    "blackman": (0.42, -0.5, 0.08),
    "blackmanharris": (0.35875, -0.48829, 0.14128, -0.01168),
    "nuttall": (0.355768, -0.487396, 0.144232, -0.012604),
    "flattop": (1.0, -1.93, 1.29, -0.388, 0.0322),
    "blackman_nuttall": (0.3635819, -0.4891775, 0.1365995, -0.0106411),
    "exact_blackman": (0.4265907136715391, -0.4965606190885641, 0.07684866723989682),
}

_TUKEY_WINDOWS: dict[str, float] = {
    "tukey": 0.5,
    "tukey_0p1": 0.1,
    "tukey_0p25": 0.25,
    "tukey_0p75": 0.75,
    "tukey_0p9": 0.9,
}

_GAUSSIAN_WINDOWS: dict[str, float] = {
    "gaussian_0p25": 0.25,
    "gaussian_0p35": 0.35,
    "gaussian_0p45": 0.45,
    "gaussian_0p55": 0.55,
    "gaussian_0p65": 0.65,
}

_GENERAL_GAUSSIAN_WINDOWS: dict[str, tuple[float, float]] = {
    "general_gaussian_1p5_0p35": (1.5, 0.35),
    "general_gaussian_2p0_0p35": (2.0, 0.35),
    "general_gaussian_3p0_0p35": (3.0, 0.35),
    "general_gaussian_4p0_0p35": (4.0, 0.35),
}

_EXPONENTIAL_WINDOWS: dict[str, float] = {
    "exponential_0p25": 0.25,
    "exponential_0p5": 0.5,
    "exponential_1p0": 1.0,
}

_CAUCHY_WINDOWS: dict[str, float] = {
    "cauchy_0p5": 0.5,
    "cauchy_1p0": 1.0,
    "cauchy_2p0": 2.0,
}

_COSINE_POWER_WINDOWS: dict[str, float] = {
    "cosine_power_2": 2.0,
    "cosine_power_3": 3.0,
    "cosine_power_4": 4.0,
}

_HANN_POISSON_WINDOWS: dict[str, float] = {
    "hann_poisson_0p5": 0.5,
    "hann_poisson_1p0": 1.0,
    "hann_poisson_2p0": 2.0,
}

_GENERAL_HAMMING_WINDOWS: dict[str, float] = {
    "general_hamming_0p50": 0.50,
    "general_hamming_0p60": 0.60,
    "general_hamming_0p70": 0.70,
    "general_hamming_0p80": 0.80,
}


def db_to_amplitude(db: float) -> float:
    return 10.0 ** (db / 20.0)


def cents_to_ratio(cents: float) -> float:
    return 2.0 ** (cents / 1200.0)


_RATIO_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_RATIO_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


def _eval_numeric_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise ValueError("Boolean literals are not allowed")
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(f"Unsupported literal: {value!r}")

    if isinstance(node, ast.BinOp):
        lhs = _eval_numeric_expr(node.left)
        rhs = _eval_numeric_expr(node.right)
        if isinstance(node.op, ast.Add):
            return lhs + rhs
        if isinstance(node.op, ast.Sub):
            return lhs - rhs
        if isinstance(node.op, ast.Mult):
            return lhs * rhs
        if isinstance(node.op, ast.Div):
            return lhs / rhs
        if isinstance(node.op, ast.Pow):
            return lhs**rhs
        raise ValueError(f"Unsupported operator: {type(node.op).__name__}")

    if isinstance(node, ast.UnaryOp):
        value = _eval_numeric_expr(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.Name):
        if node.id in _RATIO_CONSTANTS:
            return _RATIO_CONSTANTS[node.id]
        raise ValueError(f"Unknown symbol: {node.id!r}")

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple math function names are supported")
        fn_name = node.func.id
        fn = _RATIO_FUNCTIONS.get(fn_name)
        if fn is None:
            raise ValueError(f"Unsupported function: {fn_name!r}")
        args = [_eval_numeric_expr(arg) for arg in node.args]
        return float(fn(*args))

    raise ValueError(f"Unsupported expression token: {type(node).__name__}")


def parse_numeric_expression(value: str, *, context: str = "value") -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context} cannot be empty")

    normalized = text.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{context} is not a valid numeric expression: {value!r}") from exc

    try:
        out = float(_eval_numeric_expr(tree.body))
    except ZeroDivisionError as exc:
        raise ValueError(f"{context} contains division by zero: {value!r}") from exc
    except OverflowError as exc:
        raise ValueError(f"{context} overflowed while evaluating: {value!r}") from exc
    except TypeError as exc:
        raise ValueError(f"{context} is invalid: {value!r} ({exc})") from exc
    except ValueError as exc:
        raise ValueError(f"{context} is invalid: {value!r} ({exc})") from exc

    if not math.isfinite(out):
        raise ValueError(f"{context} must be finite: {value!r}")
    return out


def parse_pitch_ratio_value(value: str | float | int, *, context: str = "pitch ratio") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")

    if isinstance(value, (int, float)):
        ratio = float(value)
    else:
        ratio = parse_numeric_expression(str(value), context=context)

    if not math.isfinite(ratio):
        raise ValueError(f"{context} must be finite")
    if ratio <= 0.0:
        raise ValueError(f"{context} must be > 0")
    return ratio


def _is_power_of_two(value: int) -> bool:
    n = int(value)
    return n > 0 and (n & (n - 1)) == 0


def parse_numeric_list(value: str, *, context: str) -> list[float]:
    tokens = [token.strip() for token in str(value).split(",")]
    if not tokens or all(not token for token in tokens):
        raise ValueError(f"{context} must contain at least one numeric value")
    out: list[float] = []
    for idx, token in enumerate(tokens, start=1):
        if not token:
            raise ValueError(f"{context}: entry {idx} is empty")
        out.append(parse_numeric_expression(token, context=f"{context} entry {idx}"))
    return out


def parse_int_list(value: str, *, context: str) -> list[int]:
    values = parse_numeric_list(value, context=context)
    out: list[int] = []
    for idx, number in enumerate(values, start=1):
        rounded = int(round(number))
        if abs(number - rounded) > 1e-9:
            raise ValueError(f"{context}: entry {idx} must be an integer")
        out.append(rounded)
    return out


def _looks_like_control_signal_reference(value: Any) -> bool:
    if isinstance(value, Path):
        suffix = value.suffix.lower()
        return suffix in {".csv", ".json"}
    if not isinstance(value, str):
        return False
    text = str(value).strip()
    if not text:
        return False
    candidate = Path(text)
    suffix = candidate.suffix.lower()
    if suffix in {".csv", ".json"}:
        return True
    return candidate.exists() and candidate.is_file() and suffix in {".csv", ".json"}


def looks_like_control_signal_reference(value: Any) -> bool:
    return _looks_like_control_signal_reference(value)


def _parse_scalar_cli_value(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        out = parse_numeric_expression(str(value), context=context)
    if not math.isfinite(out):
        raise ValueError(f"{context} must be finite")
    return float(out)


def _parse_int_cli_value(value: Any, *, context: str) -> int:
    out = _parse_scalar_cli_value(value, context=context)
    rounded = int(round(out))
    if abs(out - rounded) > 1e-9:
        raise ValueError(f"{context} must be an integer")
    return int(rounded)


def parse_int_cli_value(value: Any, *, context: str) -> int:
    return _parse_int_cli_value(value, context=context)


def _parse_control_signal_value(
    value: Any,
    *,
    value_kind: Literal["float", "int", "pitch_ratio", "pitch_semitones", "pitch_cents"],
    context: str,
) -> float:
    if value_kind == "float":
        return float(_parse_scalar_cli_value(value, context=context))
    if value_kind == "int":
        return float(_parse_int_cli_value(value, context=context))
    if value_kind == "pitch_ratio":
        return float(parse_pitch_ratio_value(value, context=context))
    if value_kind == "pitch_semitones":
        semi = _parse_scalar_cli_value(value, context=context)
        return float(2.0 ** (semi / 12.0))
    if value_kind == "pitch_cents":
        cents = _parse_scalar_cli_value(value, context=context)
        return float(cents_to_ratio(cents))
    raise ValueError(f"Unsupported control value kind: {value_kind}")


def _coerce_control_interp(value: Any, *, context: str) -> ControlInterpolationMode:
    text = str(value).strip().lower()
    if text not in CONTROL_INTERP_CHOICES:
        raise ValueError(f"{context} must be one of: {', '.join(CONTROL_INTERP_CHOICES)}")
    return text  # type: ignore[return-value]


def _control_value_column_candidates(parameter: str) -> tuple[str, ...]:
    base = parameter.strip().lower()
    if base == "time_stretch":
        return (
            "value",
            "stretch",
            "time_stretch",
            "time-stretch",
            "time_stretch_factor",
            "time-stretch-factor",
        )
    if base == "pitch_ratio":
        return ("value", "pitch_ratio", "ratio")
    return (
        "value",
        base,
        base.replace("_", "-"),
    )


def _deduplicate_points(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return times, values
    out_t: list[float] = []
    out_v: list[float] = []
    for t, v in zip(times, values):
        if out_t and abs(float(t) - out_t[-1]) <= 1e-12:
            out_v[-1] = float(v)
            continue
        out_t.append(float(t))
        out_v.append(float(v))
    return np.asarray(out_t, dtype=np.float64), np.asarray(out_v, dtype=np.float64)


def _normalize_control_points(
    times: np.ndarray,
    values: np.ndarray,
    *,
    total_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        raise ValueError("Control signal has no points")
    order = np.argsort(times)
    times = np.asarray(times[order], dtype=np.float64)
    values = np.asarray(values[order], dtype=np.float64)
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if times.size == 0:
        raise ValueError("Control signal has no finite points")
    times = np.clip(times, 0.0, max(0.0, float(total_seconds)))
    times, values = _deduplicate_points(times, values)
    if times.size == 0:
        raise ValueError("Control signal has no usable points")
    if times[0] > 0.0:
        times = np.insert(times, 0, 0.0)
        values = np.insert(values, 0, values[0])
    if total_seconds > 0.0 and times[-1] < total_seconds:
        times = np.append(times, total_seconds)
        values = np.append(values, values[-1])
    if total_seconds > 0.0 and times.size == 1:
        times = np.append(times, total_seconds)
        values = np.append(values, values[0])
    return times, values


def _parse_csv_control_points(
    payload: str,
    *,
    parameter: str,
    value_kind: Literal["float", "int", "pitch_ratio", "pitch_semitones", "pitch_cents"],
    source_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    reader = csv.DictReader(io.StringIO(payload))
    fields = [str(name).strip().lower() for name in (reader.fieldnames or []) if name is not None]
    if not fields:
        raise ValueError(f"{source_label}: CSV is empty or missing header")

    has_point_time = any(name in fields for name in ("time_sec", "time", "t"))
    has_segment_time = "start_sec" in fields and "end_sec" in fields
    candidates = _control_value_column_candidates(parameter)

    value_column = next((name for name in candidates if name in fields), None)
    if value_column is None:
        excluded = {"time_sec", "time", "t", "start_sec", "end_sec"}
        extras = [name for name in fields if name not in excluded]
        if len(extras) == 1:
            value_column = extras[0]
    if value_column is None:
        raise ValueError(
            f"{source_label}: could not infer value column; expected one of {list(candidates)}"
        )

    times: list[float] = []
    values: list[float] = []
    row_count = 0
    for row_idx, row in enumerate(reader, start=2):
        row_count += 1
        norm: dict[str, str] = {}
        for key, raw in row.items():
            if key is None:
                continue
            norm[str(key).strip().lower()] = "" if raw is None else str(raw).strip()

        value_text = norm.get(value_column, "")
        if not value_text:
            continue
        parsed_value = _parse_control_signal_value(
            value_text,
            value_kind=value_kind,
            context=f"{source_label} row {row_idx} {value_column}",
        )

        if has_segment_time and norm.get("start_sec", "") and norm.get("end_sec", ""):
            start = _parse_scalar_cli_value(
                norm["start_sec"], context=f"{source_label} row {row_idx} start_sec"
            )
            end = _parse_scalar_cli_value(
                norm["end_sec"], context=f"{source_label} row {row_idx} end_sec"
            )
            if end <= start:
                continue
            times.extend([float(start), float(end)])
            values.extend([float(parsed_value), float(parsed_value)])
            continue

        if has_point_time:
            time_text = norm.get("time_sec", "") or norm.get("time", "") or norm.get("t", "")
            if not time_text:
                continue
            t = _parse_scalar_cli_value(time_text, context=f"{source_label} row {row_idx} time")
            times.append(float(t))
            values.append(float(parsed_value))
            continue

        raise ValueError(
            f"{source_label}: CSV must include time_sec/time (point mode) "
            "or start_sec/end_sec (segment mode)"
        )

    if row_count == 0 or not times:
        raise ValueError(f"{source_label}: CSV contains no usable control rows")
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _parse_json_control_points(
    payload: Any,
    *,
    parameter: str,
    value_kind: Literal["float", "int", "pitch_ratio", "pitch_semitones", "pitch_cents"],
    source_label: str,
) -> tuple[np.ndarray, np.ndarray, ControlInterpolationMode | None, int | None]:
    interpolation_override: ControlInterpolationMode | None = None
    order_override: int | None = None

    root = payload
    if isinstance(root, dict) and isinstance(root.get("parameters"), dict):
        params = root["parameters"]
        candidate = params.get(parameter)
        if candidate is None and parameter == "time_stretch":
            candidate = params.get("stretch")
        if candidate is None:
            candidate = params.get("value")
        if candidate is not None:
            root = candidate

    if isinstance(root, dict):
        interp_raw = root.get("interp", root.get("interpolation"))
        if interp_raw is not None:
            interpolation_override = _coerce_control_interp(
                interp_raw,
                context=f"{source_label} interpolation",
            )
        if root.get("order") is not None:
            order_override = _parse_int_cli_value(
                root.get("order"),
                context=f"{source_label} order",
            )

    if isinstance(root, dict):
        if isinstance(root.get("points"), list):
            points = root["points"]
        elif isinstance(root.get("control"), list):
            points = root["control"]
        elif isinstance(root.get("segments"), list):
            points = root["segments"]
        else:
            points = None
    elif isinstance(root, list):
        points = root
    else:
        points = None

    times: list[float] = []
    values: list[float] = []

    if isinstance(points, list):
        for idx, entry in enumerate(points, start=1):
            if isinstance(entry, dict) and {"start_sec", "end_sec"}.issubset(set(entry.keys())):
                start = _parse_scalar_cli_value(
                    entry.get("start_sec"),
                    context=f"{source_label} segment {idx} start_sec",
                )
                end = _parse_scalar_cli_value(
                    entry.get("end_sec"),
                    context=f"{source_label} segment {idx} end_sec",
                )
                if end <= start:
                    continue
                value_raw = entry.get("value", entry.get(parameter, entry.get("stretch")))
                if value_raw is None:
                    raise ValueError(f"{source_label} segment {idx} missing value")
                value = _parse_control_signal_value(
                    value_raw,
                    value_kind=value_kind,
                    context=f"{source_label} segment {idx} value",
                )
                times.extend([float(start), float(end)])
                values.extend([float(value), float(value)])
                continue

            if isinstance(entry, dict):
                time_raw = entry.get("time_sec", entry.get("time", entry.get("t")))
                value_raw = entry.get("value", entry.get(parameter, entry.get("stretch")))
                if time_raw is None or value_raw is None:
                    raise ValueError(f"{source_label} point {idx} must include time and value")
                t = _parse_scalar_cli_value(time_raw, context=f"{source_label} point {idx} time")
                value = _parse_control_signal_value(
                    value_raw,
                    value_kind=value_kind,
                    context=f"{source_label} point {idx} value",
                )
                times.append(float(t))
                values.append(float(value))
                continue

            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                t = _parse_scalar_cli_value(entry[0], context=f"{source_label} point {idx} time")
                value = _parse_control_signal_value(
                    entry[1],
                    value_kind=value_kind,
                    context=f"{source_label} point {idx} value",
                )
                times.append(float(t))
                values.append(float(value))
                continue

            raise ValueError(f"{source_label}: unsupported control point format at index {idx}")
    elif isinstance(root, dict):
        numeric_keys = []
        for key, val in root.items():
            key_text = str(key).strip()
            if key_text in {"interp", "interpolation", "order", "parameters"}:
                continue
            try:
                t = _parse_scalar_cli_value(key_text, context=f"{source_label} time key")
            except ValueError:
                continue
            value = _parse_control_signal_value(
                val,
                value_kind=value_kind,
                context=f"{source_label} value at {key_text}",
            )
            numeric_keys.append((float(t), float(value)))
        if numeric_keys:
            numeric_keys.sort(key=lambda item: item[0])
            for t, v in numeric_keys:
                times.append(t)
                values.append(v)

    if not times:
        raise ValueError(f"{source_label}: JSON contains no usable control points")
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        interpolation_override,
        order_override,
    )


def load_dynamic_control_signal(
    ref: DynamicControlRef,
    *,
    total_seconds: float,
) -> DynamicControlSignal:
    if not ref.path.exists():
        raise ValueError(f"Control signal file not found: {ref.path}")
    payload = ref.path.read_text(encoding="utf-8")
    suffix = ref.path.suffix.lower()
    interp = ref.interpolation
    order = ref.order
    if suffix == ".json":
        root = json.loads(payload)
        times, values, interp_override, order_override = _parse_json_control_points(
            root,
            parameter=ref.parameter,
            value_kind=ref.value_kind,
            source_label=str(ref.path),
        )
        if interp_override is not None:
            interp = interp_override
        if order_override is not None:
            order = int(order_override)
    else:
        times, values = _parse_csv_control_points(
            payload,
            parameter=ref.parameter,
            value_kind=ref.value_kind,
            source_label=str(ref.path),
        )
    times, values = _normalize_control_points(times, values, total_seconds=total_seconds)
    return DynamicControlSignal(
        parameter=ref.parameter,
        interpolation=interp,
        order=max(1, int(order)),
        times_sec=times,
        values=values,
    )


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return (u * u) * (3.0 - 2.0 * u)


def _smootherstep(u: np.ndarray) -> np.ndarray:
    return (u * u * u) * (u * (u * 6.0 - 15.0) + 10.0)


def _exp_ease(u: np.ndarray, *, strength: float) -> np.ndarray:
    k = max(0.0, float(strength))
    if k <= 1e-9:
        return u
    den = np.expm1(k)
    if abs(den) <= 1e-12:
        return u
    return np.expm1(k * u) / den


def _piecewise_ease_sample(
    query_sec: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    idx = np.searchsorted(x, query_sec, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    u = (query_sec - x0) / np.maximum(1e-12, x1 - x0)
    u = np.clip(u, 0.0, 1.0)
    if mode == "exponential":
        eased = _exp_ease(u, strength=4.0)
    elif mode == "s_curve":
        eased = _smoothstep(u)
    else:  # smootherstep
        eased = _smootherstep(u)
    return y[idx] + (y[idx + 1] - y[idx]) * eased


def _sample_dynamic_signal(signal: DynamicControlSignal, query_sec: np.ndarray) -> np.ndarray:
    if signal.times_sec.size == 0:
        raise ValueError("Dynamic control signal has no points")
    if signal.times_sec.size == 1:
        return np.full(query_sec.shape, float(signal.values[0]), dtype=np.float64)

    x = signal.times_sec
    y = signal.values
    mode = signal.interpolation

    if mode == "none":
        idx = np.searchsorted(x, query_sec, side="right") - 1
        idx = np.clip(idx, 0, x.size - 1)
        return y[idx]

    if mode == "nearest":
        idx = np.searchsorted(x, query_sec, side="left")
        idx = np.clip(idx, 0, x.size - 1)
        prev_idx = np.clip(idx - 1, 0, x.size - 1)
        choose_prev = np.abs(query_sec - x[prev_idx]) <= np.abs(query_sec - x[idx])
        chosen = np.where(choose_prev, prev_idx, idx)
        return y[chosen]

    if mode == "linear":
        # Linear interpolation is the default because it is stable and cheap in long renders.
        return np.interp(query_sec, x, y)

    if mode == "cubic":
        if scipy_cubic_spline is not None and x.size >= 4:
            # Natural spline keeps endpoints well-behaved for sparse automation curves.
            spline = scipy_cubic_spline(x, y, bc_type="natural", extrapolate=True)
            return np.asarray(spline(query_sec), dtype=np.float64)
        return np.interp(query_sec, x, y)

    if mode in {"exponential", "s_curve", "smootherstep"}:
        return _piecewise_ease_sample(
            query_sec,
            x,
            y,
            mode=mode,
        )

    if mode == "polynomial":
        degree = min(max(1, int(signal.order)), x.size - 1)
        try:
            coeffs = np.polyfit(x, y, deg=degree)
            return np.polyval(coeffs, query_sec)
        except (np.linalg.LinAlgError, ValueError):
            return np.interp(query_sec, x, y)

    return np.interp(query_sec, x, y)


def resolve_transform_auto(
    *,
    requested_transform: str,
    profile: str,
    n_fft: int,
    provided_flags: set[str],
) -> str:
    if "--transform" in provided_flags:
        return requested_transform
    if requested_transform != "fft":
        return requested_transform
    if profile in {"ambient", "extreme"} and not _is_power_of_two(n_fft) and scipy_czt is not None:
        return "czt"
    if profile == "percussion":
        return "dst" if scipy_fft is not None else "fft"
    if profile == "speech":
        return "fft"
    return requested_transform


def _has_cupy() -> bool:
    return cp is not None


def _is_cupy_array(value: Any) -> bool:
    return _has_cupy() and isinstance(value, cp.ndarray)


def _array_module(value: Any):
    if _is_cupy_array(value):
        return cp
    return np


def _to_numpy(value: Any):
    if _is_cupy_array(value):
        return cp.asnumpy(value)
    return value


def _to_runtime_array(value: Any):
    if _RUNTIME_CONFIG.active_device != "cuda":
        return value
    if _is_cupy_array(value):
        return value
    return cp.asarray(value)


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _as_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _i0(value, *, xp=np):
    if hasattr(xp, "i0"):
        return xp.i0(value)
    value_np = np.asarray(_to_numpy(value), dtype=np.float64)
    i0_np = np.i0(value_np)
    if xp is np:
        return i0_np
    return xp.asarray(i0_np)


def normalize_transform_name(value: str | None) -> TransformMode:
    name = str(value or "fft").strip().lower()
    if name == "rfft":
        name = "fft"
    if name not in TRANSFORM_CHOICES:
        raise ValueError(
            f"Unsupported transform '{value}'. Choose from: {', '.join(TRANSFORM_CHOICES)}"
        )
    return name  # type: ignore[return-value]


def transform_bin_count(n_fft: int, transform: TransformMode) -> int:
    if transform in {"dct", "dst", "hartley"}:
        return int(n_fft)
    return int(n_fft // 2 + 1)


def _analysis_angular_velocity(
    n_bins: int, n_fft: int, hop_size: int, transform: TransformMode, *, xp=np
):
    idx = xp.arange(n_bins, dtype=xp.float64)
    if transform in {"dct", "dst", "hartley"}:
        # Real-valued transforms are mirrored around Nyquist; fold indices to signed bins.
        pivot = n_fft // 2
        idx = xp.where(idx <= pivot, idx, idx - n_fft)
    return 2.0 * np.pi * hop_size * idx / float(n_fft)


def _transform_requires_scipy(transform: TransformMode) -> bool:
    return transform in {"czt", "dct", "dst"}


def ensure_transform_backend_available(transform: TransformMode) -> None:
    if transform in {"dct", "dst"} and scipy_fft is None:
        raise RuntimeError(
            "Transform backend requires SciPy FFT routines. Install scipy to use --transform "
            f"{transform}."
        )
    if transform == "czt" and scipy_czt is None:
        raise RuntimeError(
            "Transform backend requires scipy.signal.czt. Install scipy to use --transform czt."
        )


def validate_transform_available(
    transform: str,
    parser: argparse.ArgumentParser | None = None,
) -> TransformMode:
    try:
        name = normalize_transform_name(transform)
        ensure_transform_backend_available(name)
        return name
    except (ValueError, RuntimeError) as exc:
        if parser is not None:
            parser.error(str(exc))
        raise


def _resize_or_pad_1d(values, size: int, *, xp=np):
    if values.size == size:
        return values
    if values.size > size:
        return values[:size]
    out = xp.zeros(size, dtype=values.dtype)
    out[: values.size] = values
    return out


def _onesided_to_full_spectrum(spectrum, n_fft: int, *, xp=np):
    bins = int(n_fft // 2 + 1)
    spec = _resize_or_pad_1d(spectrum, bins, xp=xp).astype(xp.complex128, copy=False)
    full = xp.zeros(n_fft, dtype=xp.complex128)
    full[:bins] = spec
    if n_fft > 1:
        if n_fft % 2 == 0:
            mirror_src = spec[1:-1]
        else:
            mirror_src = spec[1:]
        if mirror_src.size:
            # Rebuild negative frequencies via Hermitian symmetry for real output.
            full[bins:] = xp.conj(mirror_src[::-1])
    return full


def _forward_transform_numpy(frame: np.ndarray, n_fft: int, transform: TransformMode) -> np.ndarray:
    if transform == "fft":
        return np.fft.rfft(frame, n=n_fft).astype(np.complex128, copy=False)
    if transform == "dft":
        full = np.fft.fft(frame, n=n_fft).astype(np.complex128, copy=False)
        return full[: n_fft // 2 + 1]
    if transform == "czt":
        ensure_transform_backend_available(transform)
        if scipy_czt is None:
            raise RuntimeError(
                "scipy.signal.czt is required for --transform czt but is not available"
            )
        full = scipy_czt(frame, m=n_fft)
        return np.asarray(full[: n_fft // 2 + 1], dtype=np.complex128)
    if transform == "dct":
        ensure_transform_backend_available(transform)
        if scipy_fft is None:
            raise RuntimeError("scipy.fft is required for --transform dct but is not available")
        coeff = scipy_fft.dct(frame, type=2, n=n_fft, norm="ortho")
        return np.asarray(coeff, dtype=np.complex128)
    if transform == "dst":
        ensure_transform_backend_available(transform)
        if scipy_fft is None:
            raise RuntimeError("scipy.fft is required for --transform dst but is not available")
        coeff = scipy_fft.dst(frame, type=2, n=n_fft, norm="ortho")
        return np.asarray(coeff, dtype=np.complex128)
    if transform == "hartley":
        full = np.fft.fft(frame, n=n_fft).astype(np.complex128, copy=False)
        return (full.real - full.imag).astype(np.complex128)
    raise ValueError(f"Unsupported transform: {transform}")


def _inverse_transform_numpy(
    spectrum: np.ndarray, n_fft: int, transform: TransformMode
) -> np.ndarray:
    if transform == "fft":
        spec = _resize_or_pad_1d(spectrum, n_fft // 2 + 1, xp=np)
        return np.fft.irfft(spec, n=n_fft).astype(np.float64, copy=False)
    if transform in {"dft", "czt"}:
        full = _onesided_to_full_spectrum(spectrum, n_fft, xp=np)
        return np.fft.ifft(full, n=n_fft).real.astype(np.float64, copy=False)
    if transform == "dct":
        ensure_transform_backend_available(transform)
        if scipy_fft is None:
            raise RuntimeError("scipy.fft is required for --transform dct but is not available")
        coeff = _resize_or_pad_1d(spectrum.real.astype(np.float64, copy=False), n_fft, xp=np)
        return scipy_fft.idct(coeff, type=2, n=n_fft, norm="ortho").astype(np.float64, copy=False)
    if transform == "dst":
        ensure_transform_backend_available(transform)
        if scipy_fft is None:
            raise RuntimeError("scipy.fft is required for --transform dst but is not available")
        coeff = _resize_or_pad_1d(spectrum.real.astype(np.float64, copy=False), n_fft, xp=np)
        return scipy_fft.idst(coeff, type=2, n=n_fft, norm="ortho").astype(np.float64, copy=False)
    if transform == "hartley":
        coeff = _resize_or_pad_1d(spectrum.real.astype(np.float64, copy=False), n_fft, xp=np)
        full = np.fft.fft(coeff, n=n_fft)
        return ((full.real - full.imag) / float(n_fft)).astype(np.float64, copy=False)
    raise ValueError(f"Unsupported transform: {transform}")


def _forward_transform(frame, n_fft: int, transform: TransformMode, *, xp=np):
    if xp is np:
        return _forward_transform_numpy(
            np.asarray(frame, dtype=np.float64),
            n_fft,
            transform,
        )
    if _transform_requires_scipy(transform):
        out = _forward_transform_numpy(
            np.asarray(_to_numpy(frame), dtype=np.float64),
            n_fft,
            transform,
        )
        return xp.asarray(out)
    if transform == "fft":
        return xp.fft.rfft(frame, n=n_fft).astype(xp.complex128, copy=False)
    if transform == "dft":
        full = xp.fft.fft(frame, n=n_fft).astype(xp.complex128, copy=False)
        return full[: n_fft // 2 + 1]
    if transform == "hartley":
        full = xp.fft.fft(frame, n=n_fft).astype(xp.complex128, copy=False)
        return (full.real - full.imag).astype(xp.complex128, copy=False)
    raise ValueError(f"Unsupported transform: {transform}")


def _inverse_transform(spectrum, n_fft: int, transform: TransformMode, *, xp=np):
    if xp is np:
        return _inverse_transform_numpy(
            np.asarray(spectrum, dtype=np.complex128),
            n_fft,
            transform,
        )
    if _transform_requires_scipy(transform):
        out = _inverse_transform_numpy(
            np.asarray(_to_numpy(spectrum), dtype=np.complex128),
            n_fft,
            transform,
        )
        return xp.asarray(out)
    if transform == "fft":
        spec = _resize_or_pad_1d(spectrum, n_fft // 2 + 1, xp=xp)
        return xp.fft.irfft(spec, n=n_fft).astype(xp.float64, copy=False)
    if transform in {"dft", "czt"}:
        full = _onesided_to_full_spectrum(spectrum, n_fft, xp=xp)
        return xp.fft.ifft(full, n=n_fft).real.astype(xp.float64, copy=False)
    if transform == "hartley":
        coeff = _resize_or_pad_1d(spectrum.real.astype(xp.float64, copy=False), n_fft, xp=xp)
        full = xp.fft.fft(coeff, n=n_fft)
        return ((full.real - full.imag) / float(n_fft)).astype(xp.float64, copy=False)
    raise ValueError(f"Unsupported transform: {transform}")


def add_runtime_args(parser: Any) -> None:
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device: auto (prefer CUDA), cpu, or cuda",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index used when --device is auto/cuda (default: 0)",
    )


def runtime_config() -> RuntimeConfig:
    return _RUNTIME_CONFIG


def configure_runtime(
    device: DeviceMode = "auto",
    cuda_device: int = 0,
    *,
    verbose: bool = False,
) -> RuntimeConfig:
    global _RUNTIME_CONFIG

    if cuda_device < 0:
        raise ValueError("--cuda-device must be >= 0")

    requested = device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device mode: {device}")

    if requested == "cpu":
        _RUNTIME_CONFIG = RuntimeConfig(
            requested_device="cpu",
            active_device="cpu",
            cuda_device=cuda_device,
            fallback_reason=None,
        )
        return _RUNTIME_CONFIG

    if not _has_cupy():
        reason = "CuPy is not installed"
        if requested == "cuda":
            raise RuntimeError(
                "CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package."
            )
        _RUNTIME_CONFIG = RuntimeConfig(
            requested_device="auto",
            active_device="cpu",
            cuda_device=cuda_device,
            fallback_reason=reason,
        )
        if verbose:
            print(f"[info] {reason}; using CPU backend", file=sys.stderr)
        return _RUNTIME_CONFIG

    try:
        cp.cuda.Device(cuda_device).use()
        _ = cp.cuda.runtime.getDevice()
    except (RuntimeError, OSError) as exc:
        reason = f"CUDA device init failed: {exc}"
        if requested == "cuda":
            raise RuntimeError(reason) from exc
        _RUNTIME_CONFIG = RuntimeConfig(
            requested_device="auto",
            active_device="cpu",
            cuda_device=cuda_device,
            fallback_reason=reason,
        )
        if verbose:
            print(f"[info] {reason}; using CPU backend", file=sys.stderr)
        return _RUNTIME_CONFIG

    requested_device: DeviceMode = "auto" if requested == "auto" else "cuda"
    _RUNTIME_CONFIG = RuntimeConfig(
        requested_device=requested_device,
        active_device="cuda",
        cuda_device=cuda_device,
        fallback_reason=None,
    )
    if verbose:
        print(f"[info] Using CUDA backend on device {cuda_device}", file=sys.stderr)
    return _RUNTIME_CONFIG


def configure_runtime_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> RuntimeConfig:
    try:
        return configure_runtime(
            device=getattr(args, "device", "auto"),
            cuda_device=getattr(args, "cuda_device", 0),
            verbose=console_level(args) >= VERBOSITY_TO_LEVEL["verbose"],
        )
    except (RuntimeError, ValueError) as exc:
        if parser is not None:
            parser.error(str(exc))
        raise


def ensure_runtime_dependencies() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if sf is None:
        missing.append("soundfile")
    if missing:
        print(
            "Missing required dependencies: " + ", ".join(missing) + ". "
            "Install them with: pip install numpy soundfile",
            file=sys.stderr,
        )
        raise SystemExit(2)


def principal_angle(phase):
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _cosine_series_window(coeffs: tuple[float, ...], length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    phase = (2.0 * xp.pi * n) / float(length - 1)
    out = xp.zeros(length, dtype=xp.float64)
    for idx, coeff in enumerate(coeffs):
        if idx == 0:
            out += coeff
        else:
            out += coeff * xp.cos(idx * phase)
    return out


def _bartlett_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    half = 0.5 * float(length - 1)
    return 1.0 - xp.abs((n - half) / half)


def _bohman_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    x = xp.abs((2.0 * n) / float(length - 1) - 1.0)
    return (1.0 - x) * xp.cos(xp.pi * x) + (xp.sin(xp.pi * x) / xp.pi)


def _cosine_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    return xp.sin((xp.pi * n) / float(length - 1))


def _sine_window(length: int, *, xp=np):
    return _cosine_window(length, xp=xp)


def _triangular_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    denom = 0.5 * float(length + 1)
    center = 0.5 * float(length - 1)
    return xp.clip(1.0 - xp.abs((n - center) / denom), 0.0, 1.0)


def _bartlett_hann_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    x = n / float(length - 1)
    y = x - 0.5
    return 0.62 - 0.48 * xp.abs(y) + 0.38 * xp.cos(2.0 * xp.pi * y)


def _tukey_window(length: int, alpha: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1 or alpha <= 0.0:
        return xp.ones(length, dtype=xp.float64)
    if alpha >= 1.0:
        return _cosine_series_window(_COSINE_SERIES_WINDOWS["hann"], length, xp=xp)
    n = xp.arange(length, dtype=xp.float64)
    x = n / float(length - 1)
    w = xp.ones(length, dtype=xp.float64)
    left = x < (alpha * 0.5)
    right = x >= (1.0 - alpha * 0.5)
    w[left] = 0.5 * (1.0 + xp.cos(xp.pi * ((2.0 * x[left] / alpha) - 1.0)))
    w[right] = 0.5 * (1.0 + xp.cos(xp.pi * ((2.0 * x[right] / alpha) - (2.0 / alpha) + 1.0)))
    return w


def _parzen_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    x = xp.abs((2.0 * n / float(length - 1)) - 1.0)
    w = xp.zeros(length, dtype=xp.float64)
    inner = x <= 0.5
    outer = (x > 0.5) & (x <= 1.0)
    w[inner] = 1.0 - 6.0 * x[inner] * x[inner] + 6.0 * x[inner] * x[inner] * x[inner]
    w[outer] = 2.0 * (1.0 - x[outer]) ** 3
    return w


def _lanczos_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    x = (2.0 * n / float(length - 1)) - 1.0
    return xp.sinc(x)


def _welch_window(length: int, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    center = 0.5 * float(length - 1)
    x = (n - center) / center
    return xp.clip(1.0 - x * x, 0.0, 1.0)


def _gaussian_window(length: int, sigma_ratio: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    center = 0.5 * float(length - 1)
    sigma = max(1e-9, sigma_ratio * center)
    n = xp.arange(length, dtype=xp.float64)
    z = (n - center) / sigma
    return xp.exp(-0.5 * z * z)


def _general_gaussian_window(length: int, power: float, sigma_ratio: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    center = 0.5 * float(length - 1)
    sigma = max(1e-9, sigma_ratio * center)
    n = xp.arange(length, dtype=xp.float64)
    z = xp.abs((n - center) / sigma)
    return xp.exp(-0.5 * xp.power(z, 2.0 * power))


def _exponential_window(length: int, tau_ratio: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    center = 0.5 * float(length - 1)
    tau = max(1e-9, tau_ratio * center)
    n = xp.arange(length, dtype=xp.float64)
    return xp.exp(-xp.abs(n - center) / tau)


def _cauchy_window(length: int, gamma_ratio: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    center = 0.5 * float(length - 1)
    gamma = max(1e-9, gamma_ratio * center)
    n = xp.arange(length, dtype=xp.float64)
    z = (n - center) / gamma
    return 1.0 / (1.0 + z * z)


def _cosine_power_window(length: int, power: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    n = xp.arange(length, dtype=xp.float64)
    base = xp.sin((xp.pi * n) / float(length - 1))
    return xp.power(base, power)


def _hann_poisson_window(length: int, alpha: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    hann = _cosine_series_window(_COSINE_SERIES_WINDOWS["hann"], length, xp=xp)
    center = 0.5 * float(length - 1)
    n = xp.arange(length, dtype=xp.float64)
    envelope = xp.exp(-alpha * xp.abs(n - center) / max(1e-9, center))
    return hann * envelope


def _general_hamming_window(length: int, alpha: float, *, xp=np):
    coeffs = (alpha, -(1.0 - alpha))
    return _cosine_series_window(coeffs, length, xp=xp)


def _kaiser_window(length: int, beta: float, *, xp=np):
    if length <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if length == 1:
        return xp.ones(1, dtype=xp.float64)
    alpha = 0.5 * float(length - 1)
    n = xp.arange(length, dtype=xp.float64)
    r = (n - alpha) / alpha
    arg = beta * xp.sqrt(xp.clip(1.0 - (r * r), 0.0, 1.0))
    denom = _i0(beta, xp=xp)
    return _i0(arg, xp=xp) / denom


def make_window(kind: WindowType, n_fft: int, win_length: int, *, kaiser_beta: float = 14.0, xp=np):
    if kind in _COSINE_SERIES_WINDOWS:
        base = _cosine_series_window(_COSINE_SERIES_WINDOWS[kind], win_length, xp=xp)
    elif kind == "sine":
        base = _sine_window(win_length, xp=xp)
    elif kind == "bartlett":
        base = _bartlett_window(win_length, xp=xp)
    elif kind == "boxcar":
        base = xp.ones(win_length, dtype=xp.float64)
    elif kind == "triangular":
        base = _triangular_window(win_length, xp=xp)
    elif kind == "bartlett_hann":
        base = _bartlett_hann_window(win_length, xp=xp)
    elif kind in _TUKEY_WINDOWS:
        base = _tukey_window(win_length, _TUKEY_WINDOWS[kind], xp=xp)
    elif kind == "parzen":
        base = _parzen_window(win_length, xp=xp)
    elif kind == "lanczos":
        base = _lanczos_window(win_length, xp=xp)
    elif kind == "welch":
        base = _welch_window(win_length, xp=xp)
    elif kind in _GAUSSIAN_WINDOWS:
        base = _gaussian_window(win_length, _GAUSSIAN_WINDOWS[kind], xp=xp)
    elif kind in _GENERAL_GAUSSIAN_WINDOWS:
        power, sigma_ratio = _GENERAL_GAUSSIAN_WINDOWS[kind]
        base = _general_gaussian_window(win_length, power, sigma_ratio, xp=xp)
    elif kind in _EXPONENTIAL_WINDOWS:
        base = _exponential_window(win_length, _EXPONENTIAL_WINDOWS[kind], xp=xp)
    elif kind in _CAUCHY_WINDOWS:
        base = _cauchy_window(win_length, _CAUCHY_WINDOWS[kind], xp=xp)
    elif kind in _COSINE_POWER_WINDOWS:
        base = _cosine_power_window(win_length, _COSINE_POWER_WINDOWS[kind], xp=xp)
    elif kind in _HANN_POISSON_WINDOWS:
        base = _hann_poisson_window(win_length, _HANN_POISSON_WINDOWS[kind], xp=xp)
    elif kind in _GENERAL_HAMMING_WINDOWS:
        base = _general_hamming_window(win_length, _GENERAL_HAMMING_WINDOWS[kind], xp=xp)
    elif kind == "bohman":
        base = _bohman_window(win_length, xp=xp)
    elif kind == "cosine":
        base = _cosine_window(win_length, xp=xp)
    elif kind == "kaiser":
        base = _kaiser_window(win_length, kaiser_beta, xp=xp)
    elif kind == "rect":
        base = xp.ones(win_length, dtype=xp.float64)
    else:  # pragma: no cover - parser blocks this
        raise ValueError(f"Unsupported window: {kind}")

    if win_length == n_fft:
        return base.astype(xp.float64, copy=False)

    window = xp.zeros(n_fft, dtype=xp.float64)
    # Center short windows inside the FFT frame to preserve existing hop semantics.
    offset = (n_fft - win_length) // 2
    window[offset : offset + win_length] = base
    return window


def pad_for_framing(signal, n_fft: int, hop: int, center: bool):
    xp = _array_module(signal)
    if center:
        signal = xp.pad(signal, (n_fft // 2, n_fft // 2), mode="constant")

    if signal.size < n_fft:
        signal = xp.pad(signal, (0, n_fft - signal.size), mode="constant")

    remainder = (signal.size - n_fft) % hop
    pad_end = (hop - remainder) % hop
    if pad_end:
        signal = xp.pad(signal, (0, pad_end), mode="constant")

    frame_count = 1 + (signal.size - n_fft) // hop
    return signal, frame_count


def stft(signal: np.ndarray, config: VocoderConfig):
    bridge_to_cuda = _RUNTIME_CONFIG.active_device == "cuda" and not _is_cupy_array(signal)
    work_signal = _to_runtime_array(signal) if bridge_to_cuda else signal
    xp = _array_module(work_signal)
    transform = normalize_transform_name(config.transform)

    work_signal, frame_count = pad_for_framing(
        work_signal, config.n_fft, config.hop_size, config.center
    )
    window = make_window(
        config.window,
        config.n_fft,
        config.win_length,
        kaiser_beta=config.kaiser_beta,
        xp=xp,
    )
    n_bins = transform_bin_count(config.n_fft, transform)
    spectrum = xp.empty((n_bins, frame_count), dtype=xp.complex128)

    for frame_idx in range(frame_count):
        start = frame_idx * config.hop_size
        frame = work_signal[start : start + config.n_fft]
        # Windowing is applied per frame before transform to control leakage.
        spectrum[:, frame_idx] = _forward_transform(frame * window, config.n_fft, transform, xp=xp)

    if bridge_to_cuda:
        return _to_numpy(spectrum)
    return spectrum


def istft(
    spectrum,
    config: VocoderConfig,
    expected_length: int | None = None,
):
    bridge_to_cuda = _RUNTIME_CONFIG.active_device == "cuda" and not _is_cupy_array(spectrum)
    work_spectrum = _to_runtime_array(spectrum) if bridge_to_cuda else spectrum
    xp = _array_module(work_spectrum)
    transform = normalize_transform_name(config.transform)

    n_frames = work_spectrum.shape[1]
    output_len = config.n_fft + config.hop_size * max(0, n_frames - 1)
    output = xp.zeros(output_len, dtype=xp.float64)
    weight = xp.zeros(output_len, dtype=xp.float64)
    window = make_window(
        config.window,
        config.n_fft,
        config.win_length,
        kaiser_beta=config.kaiser_beta,
        xp=xp,
    )

    for frame_idx in range(n_frames):
        start = frame_idx * config.hop_size
        frame = _inverse_transform(work_spectrum[:, frame_idx], config.n_fft, transform, xp=xp)
        output[start : start + config.n_fft] += frame * window
        weight[start : start + config.n_fft] += window * window

    nz = weight > 1e-12
    # Weighted overlap-add normalization removes window-power bias.
    output[nz] /= weight[nz]

    if config.center:
        trim = config.n_fft // 2
        if output.size > 2 * trim:
            output = output[trim:-trim]
        else:
            output = xp.zeros(0, dtype=xp.float64)

    if expected_length is not None:
        output = force_length(output, expected_length)

    if bridge_to_cuda:
        return _to_numpy(output)
    return output


def scaled_win_length(base_win: int, base_fft: int, frame_len: int) -> int:
    if base_fft <= 0:
        return max(2, frame_len)
    scaled = int(round(base_win * frame_len / base_fft))
    return max(2, min(frame_len, scaled))


def resize_spectrum_bins(spectrum, target_bins: int):
    xp = _array_module(spectrum)
    if target_bins <= 0:
        raise ValueError("target_bins must be > 0")
    if spectrum.size == target_bins:
        return spectrum.astype(xp.complex128, copy=True)
    if spectrum.size == 0:
        return xp.zeros(target_bins, dtype=xp.complex128)

    x_old = xp.linspace(0.0, 1.0, num=spectrum.size, endpoint=True)
    x_new = xp.linspace(0.0, 1.0, num=target_bins, endpoint=True)
    real_new = xp.interp(x_new, x_old, spectrum.real)
    imag_new = xp.interp(x_new, x_old, spectrum.imag)
    return (real_new + 1j * imag_new).astype(xp.complex128)


def smooth_series(values: np.ndarray, span: int) -> np.ndarray:
    if span <= 1 or values.size <= 2:
        return values
    kernel = np.ones(span, dtype=np.float64) / span
    pad = span // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def regularize_frame_lengths(frame_lengths: np.ndarray, max_step: int) -> np.ndarray:
    if frame_lengths.size <= 1:
        return frame_lengths

    out = frame_lengths.astype(np.int64, copy=True)
    step = max(1, int(max_step))

    for idx in range(1, out.size):
        lo = out[idx - 1] - step
        hi = out[idx - 1] + step
        out[idx] = int(np.clip(out[idx], lo, hi))

    for idx in range(out.size - 2, -1, -1):
        lo = out[idx + 1] - step
        hi = out[idx + 1] + step
        out[idx] = int(np.clip(out[idx], lo, hi))

    return out


def fill_nan_with_nearest(values: np.ndarray, fallback: float) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if out.size == 0:
        return out
    out[np.isnan(out)] = fallback
    for idx in range(1, out.size):
        if not np.isfinite(out[idx]):
            out[idx] = out[idx - 1]
    for idx in range(out.size - 2, -1, -1):
        if not np.isfinite(out[idx]):
            out[idx] = out[idx + 1]
    out[~np.isfinite(out)] = fallback
    return out


def lock_fft_length_to_f0(
    f0_hz: float,
    sample_rate: int,
    harmonic_bin: int,
    min_fft: int,
    max_fft: int,
) -> int:
    safe_f0 = max(float(f0_hz), 1e-6)
    locked = int(round(max(1, harmonic_bin) * sample_rate / safe_f0))
    return int(np.clip(max(16, locked), min_fft, max_fft))


def build_fourier_sync_plan(
    signal: np.ndarray,
    sample_rate: int,
    config: VocoderConfig,
    f0_min_hz: float,
    f0_max_hz: float,
    min_fft: int,
    max_fft: int,
    smooth_span: int,
    progress_callback: ProgressCallback | None = None,
) -> FourierSyncPlan:
    framed, frame_count = pad_for_framing(signal, config.n_fft, config.hop_size, config.center)
    if frame_count <= 0:
        return FourierSyncPlan(
            frame_lengths=np.array([config.n_fft], dtype=np.int64),
            f0_track_hz=np.array([(f0_min_hz + f0_max_hz) * 0.5], dtype=np.float64),
            reference_n_fft=max(config.n_fft, min_fft),
        )

    f0_track = np.full(frame_count, np.nan, dtype=np.float64)
    for frame_idx in range(frame_count):
        start = frame_idx * config.hop_size
        frame = framed[start : start + config.n_fft]
        if frame.size >= 4 and float(np.sqrt(np.mean(frame * frame))) >= 1e-6:
            with suppress(ValueError, ArithmeticError, RuntimeError):
                f0_track[frame_idx] = estimate_f0_autocorrelation(
                    frame, sample_rate, f0_min_hz, f0_max_hz
                )
        if progress_callback is not None:
            progress_callback(frame_idx + 1, frame_count)

    finite_f0 = f0_track[np.isfinite(f0_track)]
    # Median fallback keeps plan deterministic if a subset of frames fail F0 estimation.
    fallback_f0 = float(np.median(finite_f0)) if finite_f0.size else (f0_min_hz + f0_max_hz) * 0.5
    f0_track = fill_nan_with_nearest(f0_track, fallback=fallback_f0)
    f0_track = smooth_series(f0_track, smooth_span)
    f0_track = np.clip(f0_track, f0_min_hz, f0_max_hz)

    min_fft = max(16, min_fft)
    max_fft = max(min_fft, max_fft)
    reference_f0 = float(np.median(f0_track)) if f0_track.size else fallback_f0
    target_bin = max(1, int(round(reference_f0 * config.n_fft / sample_rate)))
    frame_lengths = np.array(
        [
            lock_fft_length_to_f0(
                f0_hz=f0,
                sample_rate=sample_rate,
                harmonic_bin=target_bin,
                min_fft=min_fft,
                max_fft=max_fft,
            )
            for f0 in f0_track
        ],
        dtype=np.int64,
    )
    frame_lengths = np.rint(smooth_series(frame_lengths.astype(np.float64), smooth_span)).astype(
        np.int64
    )
    frame_lengths = np.clip(frame_lengths, min_fft, max_fft)
    # Regularization prevents frame-length flicker that would cause audible zippering.
    frame_lengths = regularize_frame_lengths(
        frame_lengths,
        max_step=max(2, int(round(config.n_fft * 0.015))),
    )
    reference_n_fft = int(max(config.n_fft, int(np.max(frame_lengths))))
    return FourierSyncPlan(
        frame_lengths=frame_lengths, f0_track_hz=f0_track, reference_n_fft=reference_n_fft
    )


def compute_transient_flags(magnitude, threshold_scale: float):
    xp = _array_module(magnitude)
    if magnitude.shape[1] <= 1:
        return xp.zeros(magnitude.shape[1], dtype=bool)

    flux = xp.zeros(magnitude.shape[1], dtype=xp.float64)
    positive_delta = xp.maximum(0.0, xp.diff(magnitude, axis=1))
    flux[1:] = xp.sqrt(xp.sum(positive_delta * positive_delta, axis=0))

    baseline = _as_float(xp.median(flux[1:])) if flux.size > 1 else 0.0
    if baseline <= 1e-12:
        baseline = _as_float(xp.mean(flux[1:])) if flux.size > 1 else 0.0
    if baseline <= 1e-12:
        return xp.zeros_like(flux, dtype=bool)

    flags = flux >= (baseline * threshold_scale)
    flags[0] = False
    return flags


def build_output_time_steps(
    *,
    out_frames: int,
    frame_count: int,
    stretch: float,
    transient_flags: np.ndarray | None = None,
    onset_time_credit: bool = False,
    onset_credit_pull: float = 0.5,
    onset_credit_max: float = 8.0,
    onset_realign: bool = True,
) -> np.ndarray:
    if out_frames <= 0:
        return np.zeros(0, dtype=np.float64)
    if frame_count <= 1:
        return np.zeros(out_frames, dtype=np.float64)

    if not onset_time_credit:
        steps = np.arange(out_frames, dtype=np.float64) / float(stretch)
        return np.clip(steps, 0.0, frame_count - 1.000001)

    flags = None
    if transient_flags is not None:
        flags = np.asarray(transient_flags, dtype=bool)
        if flags.size != frame_count:
            src = np.linspace(0.0, 1.0, num=max(1, flags.size), endpoint=True)
            dst = np.linspace(0.0, 1.0, num=frame_count, endpoint=True)
            mapped = np.interp(dst, src, flags.astype(np.float64))
            flags = mapped >= 0.5

    base_advance = 1.0 / float(stretch)
    credit_pull = float(np.clip(onset_credit_pull, 0.0, 1.0))
    credit_cap = max(0.0, float(onset_credit_max))

    steps = np.zeros(out_frames, dtype=np.float64)
    read_pos = 0.0
    onset_credit = 0.0
    last_onset_idx = -1

    for out_idx in range(out_frames):
        read_pos = float(np.clip(read_pos, 0.0, frame_count - 1.000001))
        frame_idx = int(math.floor(read_pos))
        frac = read_pos - frame_idx
        onset_idx = min(frame_idx + (1 if frac >= 0.5 else 0), frame_count - 1)

        if flags is not None and bool(flags[onset_idx]) and onset_idx != last_onset_idx:
            if onset_realign and frac > 1e-12:
                # Realign to frame boundary on onset, then bank removed advance as time credit.
                onset_credit += 1.0 - frac
                read_pos = float(frame_idx)
            else:
                onset_credit += max(0.0, 1.0 - frac)
            onset_credit = min(onset_credit, credit_cap)
            last_onset_idx = onset_idx

        steps[out_idx] = read_pos
        advance = base_advance
        if onset_credit > 0.0 and credit_pull > 0.0:
            # Spend accumulated credit to locally slow read advance around transients.
            credit_get = min(onset_credit, credit_pull * advance)
            onset_credit -= credit_get
            advance = max(1e-9, advance - credit_get)

        read_pos += advance

    return np.clip(steps, 0.0, frame_count - 1.000001)


def create_phase_rng(*, xp=np, seed: int | None = None):
    if xp is np:
        return np.random.default_rng(seed if seed is None else int(seed))
    if xp is cp:
        return cp.random.RandomState(seed if seed is None else int(seed))
    return None


def draw_random_phase(n_bins: int, *, xp=np, rng=None):
    if n_bins <= 0:
        return xp.zeros(0, dtype=xp.float64)
    if xp is np:
        if rng is None:
            rng = np.random.default_rng()
        return np.asarray(rng.uniform(-np.pi, np.pi, size=n_bins), dtype=np.float64)
    if xp is cp:
        generator = rng if rng is not None else cp.random
        return generator.uniform(-np.pi, np.pi, size=n_bins).astype(cp.float64, copy=False)
    return xp.zeros(n_bins, dtype=xp.float64)


def apply_phase_engine(
    phase,
    *,
    engine: PhaseEngineMode,
    mix: float,
    rng=None,
):
    if engine == "propagate":
        return phase
    xp = _array_module(phase)
    random_phase = draw_random_phase(phase.size, xp=xp, rng=rng)
    if engine == "random":
        return random_phase
    blend = float(np.clip(mix, 0.0, 1.0))
    if blend <= 1e-9:
        return phase
    if blend >= 1.0:
        return random_phase
    # Blend on the unit circle (complex plane), then unwrap back to phase angle.
    return xp.angle((1.0 - blend) * xp.exp(1j * phase) + blend * xp.exp(1j * random_phase))


def find_spectral_peaks(magnitude: np.ndarray) -> np.ndarray:
    mag = _to_numpy(magnitude)
    if mag.size < 3:
        return np.array([int(np.argmax(mag))], dtype=np.int64)

    interior = (mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:])
    peak_bins = np.where(interior)[0] + 1
    if peak_bins.size == 0:
        peak_bins = np.array([int(np.argmax(mag))], dtype=np.int64)
    return peak_bins.astype(np.int64, copy=False)


def apply_identity_phase_locking(
    synth_phase,
    analysis_phase,
    magnitude,
):
    xp = _array_module(synth_phase)
    peaks_np = find_spectral_peaks(magnitude)
    peaks = xp.asarray(peaks_np, dtype=xp.int64)
    if peaks.size == 0:
        return synth_phase

    bins = xp.arange(synth_phase.size, dtype=xp.int64)[:, None]
    nearest_peak_idx = xp.argmin(xp.abs(bins - peaks[None, :]), axis=1)
    nearest_peaks = peaks[nearest_peak_idx]

    locked = synth_phase.copy()
    rel = principal_angle(analysis_phase - analysis_phase[nearest_peaks])
    locked[:] = synth_phase[nearest_peaks] + rel
    return locked


def phase_vocoder_time_stretch(
    signal: np.ndarray,
    stretch: float,
    config: VocoderConfig,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    if stretch <= 0:
        raise ValueError("Stretch factor must be > 0")
    if signal.size == 0:
        return signal

    bridge_to_cuda = _RUNTIME_CONFIG.active_device == "cuda" and not _is_cupy_array(signal)
    work_signal = _to_runtime_array(signal) if bridge_to_cuda else signal
    xp = _array_module(work_signal)
    transform = normalize_transform_name(config.transform)

    input_stft = stft(work_signal, config)
    n_bins, n_frames = input_stft.shape
    if n_frames < 2:
        target_len = max(1, int(round(work_signal.size * stretch)))
        out = force_length(work_signal.copy(), target_len)
        return _to_numpy(out) if bridge_to_cuda else out

    input_phase = xp.angle(input_stft)
    input_mag = xp.abs(input_stft)
    transient_flags = (
        compute_transient_flags(input_mag, config.transient_threshold)
        if config.transient_preserve
        else xp.zeros(n_frames, dtype=bool)
    )
    out_frames = max(2, int(round(n_frames * stretch)))
    time_steps_np = build_output_time_steps(
        out_frames=out_frames,
        frame_count=n_frames,
        stretch=stretch,
        transient_flags=_to_numpy(transient_flags),
        onset_time_credit=config.onset_time_credit,
        onset_credit_pull=config.onset_credit_pull,
        onset_credit_max=config.onset_credit_max,
        onset_realign=config.onset_realign,
    )
    time_steps = xp.asarray(time_steps_np, dtype=xp.float64)

    phase = input_phase[:, 0].copy()
    phase_rng = create_phase_rng(xp=xp, seed=config.phase_random_seed)
    omega = _analysis_angular_velocity(n_bins, config.n_fft, config.hop_size, transform, xp=xp)
    output_stft = xp.zeros((n_bins, out_frames), dtype=xp.complex128)
    if progress_callback is not None:
        progress_callback(0, out_frames)

    for out_idx in range(out_frames):
        t = _as_float(time_steps[out_idx])
        frame_idx = int(math.floor(t))
        frac = t - frame_idx

        left = input_stft[:, frame_idx]
        right = input_stft[:, min(frame_idx + 1, n_frames - 1)]
        left_phase = input_phase[:, frame_idx]
        right_phase = input_phase[:, min(frame_idx + 1, n_frames - 1)]

        mag = (1.0 - frac) * xp.abs(left) + frac * xp.abs(right)

        # Canonical phase-vocoder update: predicted advance + wrapped residual correction.
        delta = right_phase - left_phase - omega
        delta = principal_angle(delta)
        synth_phase = phase + omega + delta
        synth_phase = apply_phase_engine(
            synth_phase,
            engine=config.phase_engine,
            mix=config.ambient_phase_mix,
            rng=phase_rng,
        )

        if config.transient_preserve:
            transient_idx = min(frame_idx + (1 if frac >= 0.5 else 0), n_frames - 1)
            if _as_bool(transient_flags[transient_idx]):
                phase_blend = (1.0 - frac) * xp.exp(1j * left_phase) + frac * xp.exp(
                    1j * right_phase
                )
                synth_phase = xp.angle(phase_blend)

        if config.phase_locking == "identity" and config.phase_engine != "random":
            analysis_phase = xp.angle(
                (1.0 - frac) * xp.exp(1j * left_phase) + frac * xp.exp(1j * right_phase)
            )
            # Identity locking ties nearby bins to local peaks to reduce vertical phase scatter.
            synth_phase = apply_identity_phase_locking(synth_phase, analysis_phase, mag)

        phase = synth_phase
        output_stft[:, out_idx] = mag * xp.exp(1j * phase)
        if progress_callback is not None and (out_idx == out_frames - 1 or (out_idx % 8) == 0):
            progress_callback(out_idx + 1, out_frames)

    target_length = max(1, int(round(work_signal.size * stretch)))
    out = istft(output_stft, config, expected_length=target_length)
    return _to_numpy(out) if bridge_to_cuda else out


def phase_vocoder_time_stretch_fourier_sync(
    signal: np.ndarray,
    stretch: float,
    config: VocoderConfig,
    sync_plan: FourierSyncPlan,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    if stretch <= 0:
        raise ValueError("Stretch factor must be > 0")
    if signal.size == 0:
        return signal

    bridge_to_cuda = _RUNTIME_CONFIG.active_device == "cuda" and not _is_cupy_array(signal)
    work_signal = _to_runtime_array(signal) if bridge_to_cuda else signal
    xp = _array_module(work_signal)
    transform = normalize_transform_name(config.transform)

    framed, frame_count = pad_for_framing(work_signal, config.n_fft, config.hop_size, config.center)
    if frame_count < 2:
        target_len = max(1, int(round(work_signal.size * stretch)))
        out = force_length(work_signal.copy(), target_len)
        return _to_numpy(out) if bridge_to_cuda else out

    if sync_plan.frame_lengths.size != frame_count:
        src = np.linspace(0.0, 1.0, num=sync_plan.frame_lengths.size, endpoint=True)
        dst = np.linspace(0.0, 1.0, num=frame_count, endpoint=True)
        frame_lengths = np.interp(dst, src, sync_plan.frame_lengths.astype(np.float64))
        frame_lengths = np.clip(np.round(frame_lengths), 16, None).astype(np.int64)
    else:
        frame_lengths = sync_plan.frame_lengths.astype(np.int64, copy=False)

    out_frames = max(2, int(round(frame_count * stretch)))
    total_steps = frame_count + out_frames + out_frames

    ref_n_fft = int(max(sync_plan.reference_n_fft, config.n_fft, int(np.max(frame_lengths))))
    ref_bins = transform_bin_count(ref_n_fft, transform)
    input_stft = xp.empty((ref_bins, frame_count), dtype=xp.complex128)
    if progress_callback is not None:
        progress_callback(0, total_steps)

    for frame_idx in range(frame_count):
        n_fft_i = int(frame_lengths[frame_idx])
        start = frame_idx * config.hop_size
        frame = force_length(framed[start : start + n_fft_i], n_fft_i)
        win_length_i = scaled_win_length(config.win_length, config.n_fft, n_fft_i)
        window = make_window(
            config.window,
            n_fft_i,
            win_length_i,
            kaiser_beta=config.kaiser_beta,
            xp=xp,
        )
        spectrum = _forward_transform(frame * window, n_fft_i, transform, xp=xp)
        # Each analysis frame is resized into a common reference bin grid for interpolation.
        input_stft[:, frame_idx] = resize_spectrum_bins(spectrum, ref_bins)
        if progress_callback is not None and (frame_idx == frame_count - 1 or (frame_idx % 8) == 0):
            progress_callback(frame_idx + 1, total_steps)

    input_phase = xp.angle(input_stft)
    input_mag = xp.abs(input_stft)
    transient_flags = (
        compute_transient_flags(input_mag, config.transient_threshold)
        if config.transient_preserve
        else xp.zeros(frame_count, dtype=bool)
    )
    time_steps_np = build_output_time_steps(
        out_frames=out_frames,
        frame_count=frame_count,
        stretch=stretch,
        transient_flags=_to_numpy(transient_flags),
        onset_time_credit=config.onset_time_credit,
        onset_credit_pull=config.onset_credit_pull,
        onset_credit_max=config.onset_credit_max,
        onset_realign=config.onset_realign,
    )
    time_steps = xp.asarray(time_steps_np, dtype=xp.float64)

    phase = input_phase[:, 0].copy()
    phase_rng = create_phase_rng(xp=xp, seed=config.phase_random_seed)
    omega = _analysis_angular_velocity(ref_bins, ref_n_fft, config.hop_size, transform, xp=xp)
    output_stft = xp.zeros((ref_bins, out_frames), dtype=xp.complex128)
    output_lengths = np.zeros(out_frames, dtype=np.int64)
    for completed_steps, out_idx in enumerate(range(out_frames), start=1):
        t = _as_float(time_steps[out_idx])
        frame_idx = int(math.floor(t))
        frac = t - frame_idx
        right_idx = min(frame_idx + 1, frame_count - 1)

        left = input_stft[:, frame_idx]
        right = input_stft[:, right_idx]
        left_phase = input_phase[:, frame_idx]
        right_phase = input_phase[:, right_idx]

        mag = (1.0 - frac) * xp.abs(left) + frac * xp.abs(right)
        delta = principal_angle(right_phase - left_phase - omega)
        synth_phase = phase + omega + delta
        synth_phase = apply_phase_engine(
            synth_phase,
            engine=config.phase_engine,
            mix=config.ambient_phase_mix,
            rng=phase_rng,
        )

        if config.transient_preserve:
            transient_idx = min(frame_idx + (1 if frac >= 0.5 else 0), frame_count - 1)
            if _as_bool(transient_flags[transient_idx]):
                phase_blend = (1.0 - frac) * xp.exp(1j * left_phase) + frac * xp.exp(
                    1j * right_phase
                )
                synth_phase = xp.angle(phase_blend)

        if config.phase_locking == "identity" and config.phase_engine != "random":
            analysis_phase = xp.angle(
                (1.0 - frac) * xp.exp(1j * left_phase) + frac * xp.exp(1j * right_phase)
            )
            synth_phase = apply_identity_phase_locking(synth_phase, analysis_phase, mag)

        phase = synth_phase
        output_stft[:, out_idx] = mag * xp.exp(1j * phase)

        n_left = int(frame_lengths[frame_idx])
        n_right = int(frame_lengths[right_idx])
        # Output FFT size follows interpolated sync-plan lengths to keep harmonic bins aligned.
        output_lengths[out_idx] = max(16, int(round((1.0 - frac) * n_left + frac * n_right)))
        if progress_callback is not None and (out_idx == out_frames - 1 or (out_idx % 8) == 0):
            progress_callback(frame_count + completed_steps, total_steps)

    output_len = config.hop_size * max(0, out_frames - 1) + int(np.max(output_lengths))
    output = xp.zeros(output_len, dtype=xp.float64)
    weight = xp.zeros(output_len, dtype=xp.float64)

    for frame_idx in range(out_frames):
        n_fft_i = int(output_lengths[frame_idx])
        bins_i = transform_bin_count(n_fft_i, transform)
        spec_i = resize_spectrum_bins(output_stft[:, frame_idx], bins_i)
        frame = _inverse_transform(spec_i, n_fft_i, transform, xp=xp)
        win_length_i = scaled_win_length(config.win_length, config.n_fft, n_fft_i)
        window = make_window(
            config.window,
            n_fft_i,
            win_length_i,
            kaiser_beta=config.kaiser_beta,
            xp=xp,
        )

        start = frame_idx * config.hop_size
        output[start : start + n_fft_i] += frame * window
        weight[start : start + n_fft_i] += window * window
        if progress_callback is not None and (frame_idx == out_frames - 1 or (frame_idx % 8) == 0):
            progress_callback(frame_count + out_frames + frame_idx + 1, total_steps)

    nz = weight > 1e-12
    output[nz] /= weight[nz]

    if config.center:
        trim = config.n_fft // 2
        if output.size > 2 * trim:
            output = output[trim:-trim]
        else:
            output = xp.zeros(0, dtype=xp.float64)

    target_length = max(1, int(round(work_signal.size * stretch)))
    out = force_length(output, target_length)
    return _to_numpy(out) if bridge_to_cuda else out


def compute_multistage_stretches(stretch: float, max_stage_stretch: float) -> list[float]:
    if stretch <= 0.0:
        raise ValueError("stretch must be > 0")
    max_stage = max(1.01, float(max_stage_stretch))
    inv_max = 1.0 / max_stage

    if inv_max <= stretch <= max_stage:
        return [float(stretch)]

    if stretch > 1.0:
        stage_count = int(math.ceil(math.log(stretch) / math.log(max_stage)))
    else:
        stage_count = int(math.ceil(math.log(1.0 / stretch) / math.log(max_stage)))
    stage_count = max(2, stage_count)
    # Equal-ratio stages keep each pass in a numerically safer operating range.
    stage = float(stretch ** (1.0 / stage_count))
    return [stage] * stage_count


def phase_vocoder_time_stretch_multistage(
    signal: np.ndarray,
    stretch: float,
    config: VocoderConfig,
    *,
    max_stage_stretch: float = 1.8,
    use_fourier_sync: bool = False,
    sample_rate: int | None = None,
    f0_min_hz: float = 50.0,
    f0_max_hz: float = 1000.0,
    fourier_sync_min_fft: int = 256,
    fourier_sync_max_fft: int = 8192,
    fourier_sync_smooth: int = 5,
    progress_callback: ProgressCallback | None = None,
) -> np.ndarray:
    stages = compute_multistage_stretches(stretch, max_stage_stretch)
    if len(stages) == 1:
        if use_fourier_sync:
            if sample_rate is None:
                raise ValueError("sample_rate is required for multistage fourier-sync")
            plan = build_fourier_sync_plan(
                signal=signal,
                sample_rate=sample_rate,
                config=config,
                f0_min_hz=f0_min_hz,
                f0_max_hz=f0_max_hz,
                min_fft=fourier_sync_min_fft,
                max_fft=fourier_sync_max_fft,
                smooth_span=fourier_sync_smooth,
            )
            return phase_vocoder_time_stretch_fourier_sync(
                signal,
                stretch,
                config,
                plan,
                progress_callback=progress_callback,
            )
        return phase_vocoder_time_stretch(
            signal,
            stretch,
            config,
            progress_callback=progress_callback,
        )

    work = np.asarray(signal, dtype=np.float64)
    original_len = work.size
    stage_count = len(stages)
    for stage_idx, stage_stretch in enumerate(stages):
        stage_start = stage_idx / stage_count
        stage_span = 1.0 / stage_count

        stage_cb: ProgressCallback | None = None
        if progress_callback is not None:

            def _stage_cb(
                done: int,
                total: int,
                *,
                _stage_start: float = stage_start,
                _stage_span: float = stage_span,
            ) -> None:
                frac = _stage_start + _stage_span * (done / max(1, total))
                progress_callback(frac, 1.0)

            stage_cb = _stage_cb

        if use_fourier_sync:
            if sample_rate is None:
                raise ValueError("sample_rate is required for multistage fourier-sync")
            plan = build_fourier_sync_plan(
                signal=work,
                sample_rate=sample_rate,
                config=config,
                f0_min_hz=f0_min_hz,
                f0_max_hz=f0_max_hz,
                min_fft=fourier_sync_min_fft,
                max_fft=fourier_sync_max_fft,
                smooth_span=fourier_sync_smooth,
            )
            work = phase_vocoder_time_stretch_fourier_sync(
                work,
                stage_stretch,
                config,
                plan,
                progress_callback=stage_cb,
            )
        else:
            work = phase_vocoder_time_stretch(
                work,
                stage_stretch,
                config,
                progress_callback=stage_cb,
            )

    # Final hard length pass removes cumulative rounding drift across stages.
    target_len = max(1, int(round(original_len * stretch)))
    return np.asarray(force_length(work, target_len), dtype=np.float64)


def stretch_channel_with_strategy(
    signal: np.ndarray,
    stretch: float,
    config: VocoderConfig,
    *,
    use_multistage: bool,
    max_stage_stretch: float,
    use_fourier_sync: bool,
    sample_rate: int,
    f0_min_hz: float,
    f0_max_hz: float,
    fourier_sync_min_fft: int,
    fourier_sync_max_fft: int,
    fourier_sync_smooth: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, int]:
    if use_multistage:
        out = phase_vocoder_time_stretch_multistage(
            signal,
            stretch,
            config,
            max_stage_stretch=max_stage_stretch,
            use_fourier_sync=use_fourier_sync,
            sample_rate=sample_rate,
            f0_min_hz=f0_min_hz,
            f0_max_hz=f0_max_hz,
            fourier_sync_min_fft=fourier_sync_min_fft,
            fourier_sync_max_fft=fourier_sync_max_fft,
            fourier_sync_smooth=fourier_sync_smooth,
            progress_callback=progress_callback,
        )
        stage_count = len(compute_multistage_stretches(stretch, max_stage_stretch))
        return out, stage_count

    if use_fourier_sync:
        plan = build_fourier_sync_plan(
            signal=signal,
            sample_rate=sample_rate,
            config=config,
            f0_min_hz=f0_min_hz,
            f0_max_hz=f0_max_hz,
            min_fft=fourier_sync_min_fft,
            max_fft=fourier_sync_max_fft,
            smooth_span=fourier_sync_smooth,
        )
        out = phase_vocoder_time_stretch_fourier_sync(
            signal,
            stretch,
            config,
            plan,
            progress_callback=progress_callback,
        )
        return out, 1

    out = phase_vocoder_time_stretch(
        signal,
        stretch,
        config,
        progress_callback=progress_callback,
    )
    return out, 1


def phase_vocoder_time_stretch_multires_fusion(
    signal: np.ndarray,
    stretch: float,
    config: VocoderConfig,
    *,
    fft_sizes: list[int],
    weights: list[float],
    use_multistage: bool,
    max_stage_stretch: float,
    use_fourier_sync: bool,
    sample_rate: int,
    f0_min_hz: float,
    f0_max_hz: float,
    fourier_sync_min_fft: int,
    fourier_sync_max_fft: int,
    fourier_sync_smooth: int,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, int]:
    if not fft_sizes:
        raise ValueError("multires fusion requires at least one FFT size")
    if len(weights) != len(fft_sizes):
        raise ValueError("multires fusion weights must match fft size count")

    valid_weights = np.asarray(weights, dtype=np.float64)
    if np.any(valid_weights < 0.0):
        raise ValueError("multires fusion weights must be non-negative")
    total_weight = float(np.sum(valid_weights))
    if total_weight <= 0.0:
        raise ValueError("multires fusion weights must sum to > 0")
    valid_weights /= total_weight

    hop_ratio = config.hop_size / float(max(1, config.n_fft))
    win_ratio = config.win_length / float(max(1, config.n_fft))
    target_len = max(1, int(round(signal.size * stretch)))
    fused = np.zeros(target_len, dtype=np.float64)
    max_stages = 1

    for idx, (fft_size, weight) in enumerate(zip(fft_sizes, valid_weights, strict=True)):
        fft_i = max(16, int(fft_size))
        win_i = max(2, min(fft_i, int(round(win_ratio * fft_i))))
        hop_i = max(1, min(win_i, int(round(hop_ratio * fft_i))))
        config_i = VocoderConfig(
            n_fft=fft_i,
            win_length=win_i,
            hop_size=hop_i,
            window=config.window,
            center=config.center,
            phase_locking=config.phase_locking,
            transient_preserve=config.transient_preserve,
            transient_threshold=config.transient_threshold,
            kaiser_beta=config.kaiser_beta,
            transform=config.transform,
            phase_engine=config.phase_engine,
            ambient_phase_mix=config.ambient_phase_mix,
            phase_random_seed=config.phase_random_seed,
            onset_time_credit=config.onset_time_credit,
            onset_credit_pull=config.onset_credit_pull,
            onset_credit_max=config.onset_credit_max,
            onset_realign=config.onset_realign,
        )

        sub_cb: ProgressCallback | None = None
        if progress_callback is not None:
            sub_start = idx / len(fft_sizes)
            sub_end = (idx + 1) / len(fft_sizes)

            def _sub_cb(
                done: int,
                total: int,
                *,
                _start: float = sub_start,
                _span: float = sub_end - sub_start,
            ) -> None:
                frac = _start + _span * (done / max(1, total))
                progress_callback(frac, 1)

            sub_cb = _sub_cb

        stretched_i, stage_count_i = stretch_channel_with_strategy(
            signal=signal,
            stretch=stretch,
            config=config_i,
            use_multistage=use_multistage,
            max_stage_stretch=max_stage_stretch,
            use_fourier_sync=use_fourier_sync,
            sample_rate=sample_rate,
            f0_min_hz=f0_min_hz,
            f0_max_hz=f0_max_hz,
            fourier_sync_min_fft=fourier_sync_min_fft,
            fourier_sync_max_fft=fourier_sync_max_fft,
            fourier_sync_smooth=fourier_sync_smooth,
            progress_callback=sub_cb,
        )
        stretched_i = np.asarray(force_length(stretched_i, target_len), dtype=np.float64)
        # Weighted sum of aligned renders acts as a simple multi-resolution ensemble.
        fused += float(weight) * stretched_i
        max_stages = max(max_stages, int(stage_count_i))

    return fused, max_stages


def linear_resample_1d(signal, output_samples: int):
    xp = _array_module(signal)
    if signal.size == 0:
        return xp.zeros(output_samples, dtype=xp.float64)
    if output_samples <= 1:
        return (
            xp.array([signal[0]], dtype=xp.float64)
            if output_samples == 1
            else xp.zeros(0, dtype=xp.float64)
        )
    if signal.size == 1:
        return xp.full(output_samples, signal[0], dtype=xp.float64)

    x_old = xp.linspace(0.0, 1.0, num=signal.size, endpoint=True)
    x_new = xp.linspace(0.0, 1.0, num=output_samples, endpoint=True)
    return xp.interp(x_new, x_old, signal).astype(xp.float64)


def resample_1d(signal, output_samples: int, mode: ResampleMode):
    if output_samples < 0:
        raise ValueError("output_samples must be non-negative")
    if output_samples == signal.size:
        return signal.copy()

    bridge_to_cuda = _RUNTIME_CONFIG.active_device == "cuda" and not _is_cupy_array(signal)
    work_signal = _to_runtime_array(signal) if bridge_to_cuda else signal
    xp = _array_module(work_signal)

    use_fft = mode == "fft" or (mode == "auto" and scipy_resample is not None)
    if use_fft:
        if xp is np and scipy_resample is not None:
            out = scipy_resample(work_signal, output_samples).astype(np.float64)
            return _to_numpy(out) if bridge_to_cuda else out
        if xp is cp and cupyx_resample is not None:
            out = cupyx_resample(work_signal, output_samples).astype(cp.float64)
            return _to_numpy(out) if bridge_to_cuda else out
        if xp is cp and scipy_resample is not None:
            out = scipy_resample(_to_numpy(work_signal), output_samples).astype(np.float64)
            return out

    out = linear_resample_1d(work_signal, output_samples)
    return _to_numpy(out) if bridge_to_cuda else out


def force_length(signal, length: int):
    xp = _array_module(signal)
    if length < 0:
        raise ValueError("Target length must be non-negative")
    if signal.size == length:
        return signal
    if signal.size > length:
        return signal[:length]
    return xp.pad(signal, (0, length - signal.size), mode="constant")


def estimate_f0_autocorrelation(
    samples: np.ndarray,
    sample_rate: int,
    f0_min_hz: float,
    f0_max_hz: float,
) -> float:
    samples = _to_numpy(samples)
    if samples.size < 4:
        raise ValueError("Signal is too short to estimate F0")

    centered = samples.astype(np.float64, copy=False)
    centered = centered - np.mean(centered)
    if np.allclose(centered, 0.0):
        raise ValueError("Signal appears silent; cannot estimate F0")

    analysis_len = min(centered.size, sample_rate * 3)
    frame = centered[:analysis_len]
    frame = frame * np.hanning(frame.size)

    corr = np.correlate(frame, frame, mode="full")
    corr = corr[corr.size // 2 :]

    min_lag = max(1, int(sample_rate / f0_max_hz))
    max_lag = min(corr.size - 1, int(sample_rate / f0_min_hz))
    if max_lag <= min_lag:
        raise ValueError("Invalid F0 search bounds")

    segment = corr[min_lag : max_lag + 1]
    peak_rel = int(np.argmax(segment))
    peak_val = segment[peak_rel]
    if not np.isfinite(peak_val) or peak_val <= 0:
        raise ValueError("No valid periodic peak found for F0 estimation")

    lag = min_lag + peak_rel

    # Optional parabolic refinement around the autocorrelation peak.
    if 1 <= lag < corr.size - 1:
        y0, y1, y2 = corr[lag - 1], corr[lag], corr[lag + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            lag = lag + 0.5 * (y0 - y2) / denom

    if lag <= 0:
        raise ValueError("Estimated lag is not positive")

    return float(sample_rate / lag)


def normalize_audio(
    audio,
    mode: str,
    peak_dbfs: float,
    rms_dbfs: float,
):
    if mode == "none":
        return audio

    xp = _array_module(audio)
    out = audio.astype(xp.float64, copy=True)
    if mode == "peak":
        peak = _as_float(xp.max(xp.abs(out))) if out.size else 0.0
        if peak > 0.0:
            out *= db_to_amplitude(peak_dbfs) / peak
        return out

    if mode == "rms":
        rms = _as_float(xp.sqrt(xp.mean(out * out))) if out.size else 0.0
        if rms > 0.0:
            out *= db_to_amplitude(rms_dbfs) / rms
        return out

    raise ValueError(f"Unknown normalization mode: {mode}")


def _envelope_coeff(sample_rate: int, ms: float) -> float:
    tau = max(0.01, float(ms)) / 1000.0
    return float(np.exp(-1.0 / max(1.0, sample_rate * tau)))


def _envelope_follower(
    signal_1d: np.ndarray, sample_rate: int, attack_ms: float, release_ms: float
) -> np.ndarray:
    attack = _envelope_coeff(sample_rate, attack_ms)
    release = _envelope_coeff(sample_rate, release_ms)
    env = np.zeros(signal_1d.size, dtype=np.float64)
    state = 0.0
    for idx, sample in enumerate(np.abs(signal_1d)):
        coef = attack if sample > state else release
        state = coef * state + (1.0 - coef) * sample
        env[idx] = state
    return env


def _estimate_lufs_or_rms_db(audio: np.ndarray, sample_rate: int) -> float:
    mono = np.mean(np.asarray(audio, dtype=np.float64), axis=1)
    if mono.size == 0:
        return -120.0
    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sample_rate)
        value = float(meter.integrated_loudness(mono))
        if np.isfinite(value):
            return value
    except (ImportError, ModuleNotFoundError, ValueError, RuntimeError):
        pass
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
    return float(20.0 * np.log10(rms + 1e-12))


def _apply_compressor(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float,
) -> np.ndarray:
    threshold = db_to_amplitude(threshold_db)
    ratio = max(1.0, float(ratio))
    out = np.asarray(audio, dtype=np.float64).copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        env = _envelope_follower(x, sample_rate, attack_ms, release_ms)
        gain = np.ones_like(env)
        over = env > threshold
        gain[over] = (threshold + (env[over] - threshold) / ratio) / (env[over] + 1e-12)
        out[:, ch] = x * gain
    out *= db_to_amplitude(makeup_db)
    return out


def _apply_expander(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    threshold = db_to_amplitude(threshold_db)
    ratio = max(1.0, float(ratio))
    out = np.asarray(audio, dtype=np.float64).copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        env = _envelope_follower(x, sample_rate, attack_ms, release_ms)
        gain = np.ones_like(env)
        under = env < threshold
        gain[under] = np.power(np.maximum(env[under], 1e-12) / threshold, ratio - 1.0)
        out[:, ch] = x * gain
    return out


def _apply_compander(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    compress_ratio: float,
    expand_ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float,
) -> np.ndarray:
    threshold = db_to_amplitude(threshold_db)
    comp = max(1.0, float(compress_ratio))
    expand = max(1.0, float(expand_ratio))
    out = np.asarray(audio, dtype=np.float64).copy()
    for ch in range(out.shape[1]):
        x = out[:, ch]
        env = _envelope_follower(x, sample_rate, attack_ms, release_ms)
        gain = np.ones_like(env)
        over = env > threshold
        under = env < threshold
        gain[over] = (threshold + (env[over] - threshold) / comp) / (env[over] + 1e-12)
        gain[under] = np.power(np.maximum(env[under], 1e-12) / threshold, expand - 1.0)
        out[:, ch] = x * gain
    out *= db_to_amplitude(makeup_db)
    return out


def _apply_limiter(audio: np.ndarray, threshold: float) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float64).copy()
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak <= threshold:
        return out
    return out * (threshold / (peak + 1e-12))


def _apply_soft_clip(audio: np.ndarray, level: float, clip_type: str, drive: float) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float64).copy()
    lvl = max(1e-6, float(level))
    drv = max(1e-6, float(drive))
    x = (out / lvl) * drv
    kind = clip_type.lower()
    if kind == "tanh":
        y = np.tanh(x) / np.tanh(drv)
    elif kind == "arctan":
        denom = np.arctan((np.pi * 0.5) * drv)
        y = np.arctan((np.pi * 0.5) * x) / max(1e-12, denom)
    elif kind == "cubic":
        z = np.clip(x, -1.0, 1.0)
        y = 1.5 * z - 0.5 * z * z * z
    else:
        raise ValueError(f"Unsupported soft clip type: {clip_type}")
    return lvl * y


def add_mastering_args(parser: Any) -> None:
    parser.add_argument(
        "--normalize",
        choices=["none", "peak", "rms"],
        default="none",
        help="Output normalization mode",
    )
    parser.add_argument(
        "--peak-dbfs", type=float, default=-1.0, help="Target peak dBFS when --normalize peak"
    )
    parser.add_argument(
        "--rms-dbfs", type=float, default=-18.0, help="Target RMS dBFS when --normalize rms"
    )
    parser.add_argument(
        "--target-lufs", type=float, default=None, help="Integrated loudness target in LUFS"
    )
    parser.add_argument(
        "--compressor-threshold-db",
        type=float,
        default=None,
        help="Enable compressor above threshold dBFS",
    )
    parser.add_argument(
        "--compressor-ratio", type=float, default=4.0, help="Compressor ratio (>=1)"
    )
    parser.add_argument(
        "--compressor-attack-ms", type=float, default=10.0, help="Compressor attack time in ms"
    )
    parser.add_argument(
        "--compressor-release-ms", type=float, default=120.0, help="Compressor release time in ms"
    )
    parser.add_argument(
        "--compressor-makeup-db", type=float, default=0.0, help="Compressor makeup gain in dB"
    )
    parser.add_argument(
        "--expander-threshold-db",
        type=float,
        default=None,
        help="Enable downward expander below threshold dBFS",
    )
    parser.add_argument("--expander-ratio", type=float, default=2.0, help="Expander ratio (>=1)")
    parser.add_argument(
        "--expander-attack-ms", type=float, default=5.0, help="Expander attack time in ms"
    )
    parser.add_argument(
        "--expander-release-ms", type=float, default=120.0, help="Expander release time in ms"
    )
    parser.add_argument(
        "--compander-threshold-db",
        type=float,
        default=None,
        help="Enable compander threshold in dBFS",
    )
    parser.add_argument(
        "--compander-compress-ratio",
        type=float,
        default=3.0,
        help="Compander compression ratio (>=1)",
    )
    parser.add_argument(
        "--compander-expand-ratio", type=float, default=1.8, help="Compander expansion ratio (>=1)"
    )
    parser.add_argument(
        "--compander-attack-ms", type=float, default=8.0, help="Compander attack time in ms"
    )
    parser.add_argument(
        "--compander-release-ms", type=float, default=120.0, help="Compander release time in ms"
    )
    parser.add_argument(
        "--compander-makeup-db", type=float, default=0.0, help="Compander makeup gain in dB"
    )
    parser.add_argument(
        "--limiter-threshold",
        type=float,
        default=None,
        help="Peak limiter threshold in linear full-scale",
    )
    parser.add_argument(
        "--soft-clip-level",
        type=float,
        default=None,
        help="Soft clip output ceiling in linear full-scale",
    )
    parser.add_argument(
        "--soft-clip-type",
        choices=["tanh", "arctan", "cubic"],
        default="tanh",
        help="Soft clip transfer type",
    )
    parser.add_argument(
        "--soft-clip-drive", type=float, default=1.0, help="Soft clip drive amount (>0)"
    )
    parser.add_argument(
        "--hard-clip-level", type=float, default=None, help="Hard clip level in linear full-scale"
    )
    parser.add_argument(
        "--clip", action="store_true", help="Legacy alias: hard clip at +/-1.0 when set"
    )


def validate_mastering_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.target_lufs is not None and not np.isfinite(args.target_lufs):
        parser.error("--target-lufs must be finite")
    if args.compressor_ratio < 1.0:
        parser.error("--compressor-ratio must be >= 1")
    if args.expander_ratio < 1.0:
        parser.error("--expander-ratio must be >= 1")
    if args.compander_compress_ratio < 1.0:
        parser.error("--compander-compress-ratio must be >= 1")
    if args.compander_expand_ratio < 1.0:
        parser.error("--compander-expand-ratio must be >= 1")
    for field in (
        "compressor_attack_ms",
        "compressor_release_ms",
        "expander_attack_ms",
        "expander_release_ms",
        "compander_attack_ms",
        "compander_release_ms",
    ):
        if getattr(args, field) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be > 0")
    if args.limiter_threshold is not None and args.limiter_threshold <= 0.0:
        parser.error("--limiter-threshold must be > 0")
    if args.soft_clip_level is not None and args.soft_clip_level <= 0.0:
        parser.error("--soft-clip-level must be > 0")
    if args.soft_clip_drive <= 0.0:
        parser.error("--soft-clip-drive must be > 0")
    if args.hard_clip_level is not None and args.hard_clip_level <= 0.0:
        parser.error("--hard-clip-level must be > 0")


def apply_mastering_chain(
    audio: np.ndarray, sample_rate: int, args: argparse.Namespace
) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    out = out.copy()

    # Dynamics order is intentional: downward expander -> compressor -> compander.
    if args.expander_threshold_db is not None:
        out = _apply_expander(
            out,
            sample_rate,
            args.expander_threshold_db,
            args.expander_ratio,
            args.expander_attack_ms,
            args.expander_release_ms,
        )

    if args.compressor_threshold_db is not None:
        out = _apply_compressor(
            out,
            sample_rate,
            args.compressor_threshold_db,
            args.compressor_ratio,
            args.compressor_attack_ms,
            args.compressor_release_ms,
            args.compressor_makeup_db,
        )

    if args.compander_threshold_db is not None:
        out = _apply_compander(
            out,
            sample_rate,
            args.compander_threshold_db,
            args.compander_compress_ratio,
            args.compander_expand_ratio,
            args.compander_attack_ms,
            args.compander_release_ms,
            args.compander_makeup_db,
        )

    # Normalize before optional LUFS trim so downstream limiter/clip stages have predictable input.
    out = normalize_audio(out, args.normalize, args.peak_dbfs, args.rms_dbfs)

    if args.target_lufs is not None:
        current = _estimate_lufs_or_rms_db(out, sample_rate)
        if np.isfinite(current):
            out *= db_to_amplitude(args.target_lufs - current)

    if args.limiter_threshold is not None:
        out = _apply_limiter(out, float(args.limiter_threshold))

    if args.soft_clip_level is not None:
        out = _apply_soft_clip(out, args.soft_clip_level, args.soft_clip_type, args.soft_clip_drive)

    # Hard clip is intentionally last and only used as a final guard rail.
    hard_clip_level = args.hard_clip_level
    if hard_clip_level is None and bool(getattr(args, "clip", False)):
        hard_clip_level = 1.0
    if hard_clip_level is not None:
        out = np.clip(out, -abs(float(hard_clip_level)), abs(float(hard_clip_level)))

    return out


def cepstral_envelope(magnitude, lifter: int):
    xp = _array_module(magnitude)
    n_bins = magnitude.size
    n_fft = max(2, (n_bins - 1) * 2)
    log_mag = xp.log(xp.maximum(magnitude.astype(xp.float64, copy=False), 1e-12))
    cep = xp.fft.irfft(log_mag, n=n_fft)

    if lifter > 0 and lifter < n_fft // 2:
        # Liftering keeps low-quefrency envelope structure while suppressing fine harmonic detail.
        lifted = xp.zeros_like(cep)
        lifted[: lifter + 1] = cep[: lifter + 1]
        lifted[-lifter:] = cep[-lifter:]
        cep = lifted

    env_log = xp.fft.rfft(cep, n=n_fft).real
    return xp.exp(env_log)


def apply_formant_preservation(
    reference: np.ndarray,
    shifted: np.ndarray,
    config: VocoderConfig,
    lifter: int,
    strength: float,
    max_gain_db: float,
) -> np.ndarray:
    if reference.size == 0 or shifted.size == 0 or strength <= 0.0:
        return shifted

    bridge_to_cuda = _RUNTIME_CONFIG.active_device == "cuda" and (
        (not _is_cupy_array(reference)) or (not _is_cupy_array(shifted))
    )
    ref_work = _to_runtime_array(reference) if bridge_to_cuda else reference
    shifted_work = _to_runtime_array(shifted) if bridge_to_cuda else shifted
    xp = _array_module(ref_work)

    ref_spec = stft(ref_work, config)
    tgt_spec = stft(shifted_work, config)
    ref_mag = xp.abs(ref_spec)
    tgt_mag = xp.abs(tgt_spec)
    tgt_phase = xp.angle(tgt_spec)

    ref_frames = ref_mag.shape[1]
    tgt_frames = tgt_mag.shape[1]
    if ref_frames == 0 or tgt_frames == 0:
        return shifted

    ref_env = xp.empty_like(ref_mag)
    for idx in range(ref_frames):
        ref_env[:, idx] = cepstral_envelope(ref_mag[:, idx], lifter)

    gain_limit = db_to_amplitude(max_gain_db)
    min_gain = 1.0 / gain_limit
    max_gain = gain_limit

    corrected = xp.empty_like(tgt_spec)
    for idx in range(tgt_frames):
        ref_idx = (
            0 if tgt_frames == 1 else int(round(idx * (ref_frames - 1) / max(1, tgt_frames - 1)))
        )
        tgt_env = cepstral_envelope(tgt_mag[:, idx], lifter)
        gain = ref_env[:, ref_idx] / xp.maximum(tgt_env, 1e-12)
        gain = xp.clip(gain, min_gain, max_gain)
        if strength < 1.0:
            gain = xp.power(gain, strength)
        corrected[:, idx] = (tgt_mag[:, idx] * gain) * xp.exp(1j * tgt_phase[:, idx])

    out = istft(corrected, config, expected_length=shifted_work.size)
    return _to_numpy(out) if bridge_to_cuda else out


def choose_pitch_ratio(args: argparse.Namespace, signal: np.ndarray, sr: int) -> PitchConfig:
    if args.pitch_shift_ratio is not None:
        ratio = parse_pitch_ratio_value(args.pitch_shift_ratio, context="--pitch-shift-ratio")
        return PitchConfig(ratio=ratio)

    if args.pitch_shift_semitones is not None:
        return PitchConfig(ratio=2.0 ** (args.pitch_shift_semitones / 12.0))

    if args.pitch_shift_cents is not None:
        return PitchConfig(ratio=cents_to_ratio(args.pitch_shift_cents))

    if args.target_f0 is None:
        return PitchConfig(ratio=1.0)

    if args.analysis_channel == "first":
        f0_source = signal[:, 0]
    else:
        f0_source = np.mean(signal, axis=1)

    detected_f0 = estimate_f0_autocorrelation(f0_source, sr, args.f0_min, args.f0_max)
    ratio = args.target_f0 / detected_f0
    if ratio <= 0:
        raise ValueError("Computed pitch ratio from target F0 is not positive")
    return PitchConfig(ratio=ratio, source_f0_hz=detected_f0)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def parse_control_segments_csv(
    payload: str,
    *,
    default_stretch: float,
    default_pitch_ratio: float,
) -> list[ControlSegment]:
    reader = csv.DictReader(io.StringIO(payload))
    fields = [str(name).strip().lower() for name in (reader.fieldnames or [])]
    if not fields:
        raise ValueError("Control-map CSV is empty")
    required = {"start_sec", "end_sec"}
    if not required.issubset(fields):
        raise ValueError("Control-map CSV must include: start_sec,end_sec")

    stretch_keys = (
        "stretch",
        "time_stretch",
        "time-stretch",
        "time_stretch_factor",
        "time-stretch-factor",
    )
    ratio_keys = ("pitch_ratio",)
    cents_keys = ("pitch_cents",)
    semitone_keys = ("pitch_semitones",)
    confidence_keys = ("confidence", "conf", "pitch_confidence", "f0_confidence")

    segments: list[ControlSegment] = []
    for row_index, row in enumerate(reader, start=2):
        values: dict[str, str] = {}
        for key, raw in row.items():
            norm_key = str(key).strip().lower()
            values[norm_key] = "" if raw is None else str(raw).strip()

        start_raw = values.get("start_sec", "")
        end_raw = values.get("end_sec", "")
        if not start_raw or not end_raw:
            raise ValueError(f"Control-map row {row_index}: start_sec and end_sec are required")

        start_sec = float(start_raw)
        end_sec = float(end_raw)
        if end_sec <= start_sec:
            continue

        stretch_value: float | None = None
        for key in stretch_keys:
            stretch_value = _parse_optional_float(values.get(key))
            if stretch_value is not None:
                break
        stretch = default_stretch if stretch_value is None else float(stretch_value)
        if stretch <= 0.0:
            raise ValueError(f"Control-map row {row_index}: stretch must be > 0")

        ratio_text = ""
        cents_text = ""
        semitone_text = ""
        for key in ratio_keys:
            ratio_text = values.get(key, "").strip()
            if ratio_text:
                break
        for key in cents_keys:
            cents_text = values.get(key, "").strip()
            if cents_text:
                break
        for key in semitone_keys:
            semitone_text = values.get(key, "").strip()
            if semitone_text:
                break

        populated = int(bool(ratio_text)) + int(bool(cents_text)) + int(bool(semitone_text))
        if populated > 1:
            raise ValueError(
                f"Control-map row {row_index}: use only one of pitch_ratio, pitch_cents, or pitch_semitones"
            )
        if ratio_text:
            pitch_ratio = parse_pitch_ratio_value(
                ratio_text,
                context=f"control-map row {row_index} pitch_ratio",
            )
        elif cents_text:
            pitch_ratio = cents_to_ratio(float(cents_text))
        elif semitone_text:
            pitch_ratio = 2.0 ** (float(semitone_text) / 12.0)
        else:
            pitch_ratio = default_pitch_ratio
        if pitch_ratio <= 0.0:
            raise ValueError(f"Control-map row {row_index}: pitch ratio must be > 0")

        confidence_value: float | None = None
        for key in confidence_keys:
            confidence_value = _parse_optional_float(values.get(key))
            if confidence_value is not None:
                break
        if confidence_value is not None and not math.isfinite(confidence_value):
            raise ValueError(f"Control-map row {row_index}: confidence must be finite")

        segments.append(
            ControlSegment(
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                stretch=float(stretch),
                pitch_ratio=float(pitch_ratio),
                confidence=confidence_value,
            )
        )

    segments.sort(key=lambda seg: seg.start_sec)
    return segments


def apply_control_confidence_policy(
    segments: list[ControlSegment],
    *,
    conf_min: float,
    mode: LowConfidenceMode,
    fallback_ratio: float,
) -> list[ControlSegment]:
    if not segments:
        return []
    if conf_min <= 0.0:
        return segments

    ratios = np.array([seg.pitch_ratio for seg in segments], dtype=np.float64)
    conf = np.array(
        [1.0 if seg.confidence is None else float(seg.confidence) for seg in segments],
        dtype=np.float64,
    )
    valid = conf >= conf_min

    if mode == "hold":
        last = float(fallback_ratio)
        for idx in range(ratios.size):
            if valid[idx]:
                last = float(ratios[idx])
            else:
                ratios[idx] = last
    elif mode == "unity":
        ratios[~valid] = 1.0
    else:  # mode == "interp"
        ratios = ratios.astype(np.float64, copy=True)
        ratios[~valid] = np.nan
        good_idx = np.flatnonzero(np.isfinite(ratios))
        if good_idx.size:
            x = np.arange(ratios.size, dtype=np.float64)
            ratios = np.interp(x, good_idx.astype(np.float64), ratios[good_idx])
        else:
            ratios.fill(float(fallback_ratio))

    out: list[ControlSegment] = []
    for seg, ratio in zip(segments, ratios):
        out.append(
            ControlSegment(
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                stretch=seg.stretch,
                pitch_ratio=float(max(1e-8, ratio)),
                confidence=seg.confidence,
            )
        )
    return out


def smooth_control_ratios(
    segments: list[ControlSegment], *, smooth_ms: float
) -> list[ControlSegment]:
    if smooth_ms <= 0.0 or len(segments) < 3:
        return segments

    durations = np.array(
        [max(1e-9, seg.end_sec - seg.start_sec) for seg in segments], dtype=np.float64
    )
    median_duration = float(np.median(durations))
    if median_duration <= 0.0:
        return segments

    window = int(round((smooth_ms / 1000.0) / median_duration))
    if window <= 1:
        return segments
    if window % 2 == 0:
        window += 1
    window = min(window, max(3, len(segments) | 1))
    if window <= 1:
        return segments

    ratios = np.array([seg.pitch_ratio for seg in segments], dtype=np.float64)
    pad = window // 2
    padded = np.pad(ratios, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(padded, kernel, mode="valid")

    out: list[ControlSegment] = []
    for seg, ratio in zip(segments, smoothed):
        out.append(
            ControlSegment(
                start_sec=seg.start_sec,
                end_sec=seg.end_sec,
                stretch=seg.stretch,
                pitch_ratio=float(max(1e-8, ratio)),
                confidence=seg.confidence,
            )
        )
    return out


def expand_control_segments(
    segments: list[ControlSegment],
    *,
    total_seconds: float,
    default_stretch: float,
    default_pitch_ratio: float,
) -> list[ControlSegment]:
    total_seconds = max(0.0, float(total_seconds))
    if total_seconds <= 0.0:
        return []

    ordered = sorted(segments, key=lambda seg: seg.start_sec)
    merged: list[ControlSegment] = []
    cursor = 0.0

    for seg in ordered:
        start = min(max(seg.start_sec, 0.0), total_seconds)
        end = min(max(seg.end_sec, 0.0), total_seconds)
        if end <= start:
            continue
        if end <= cursor:
            continue
        if start < cursor:
            start = cursor
        if start > cursor:
            merged.append(
                ControlSegment(
                    start_sec=cursor,
                    end_sec=start,
                    stretch=default_stretch,
                    pitch_ratio=default_pitch_ratio,
                    confidence=1.0,
                )
            )
        merged.append(
            ControlSegment(
                start_sec=start,
                end_sec=end,
                stretch=seg.stretch,
                pitch_ratio=seg.pitch_ratio,
                confidence=seg.confidence,
            )
        )
        cursor = end

    if cursor < total_seconds:
        merged.append(
            ControlSegment(
                start_sec=cursor,
                end_sec=total_seconds,
                stretch=default_stretch,
                pitch_ratio=default_pitch_ratio,
                confidence=1.0,
            )
        )

    if not merged:
        merged.append(
            ControlSegment(
                start_sec=0.0,
                end_sec=total_seconds,
                stretch=default_stretch,
                pitch_ratio=default_pitch_ratio,
                confidence=1.0,
            )
        )
    return merged


def load_control_segments(
    args: argparse.Namespace,
    *,
    default_stretch: float,
    default_pitch_ratio: float,
) -> list[ControlSegment]:
    map_path = getattr(args, "pitch_map", None)
    use_stdin = bool(getattr(args, "pitch_map_stdin", False)) or str(map_path) == "-"
    payload: str
    if use_stdin:
        raw = sys.stdin.buffer.read()
        if not raw:
            raise ValueError("No control-map bytes received on stdin")
        payload = raw.decode("utf-8-sig")
    else:
        if map_path is None:
            raise ValueError("--pitch-map is required when not reading from stdin")
        payload = Path(map_path).read_text(encoding="utf-8")

    routes: list[ControlRoute] = list(getattr(args, "_control_routes", []) or [])
    if routes:
        payload = apply_control_routes_csv(
            payload,
            routes=routes,
            source_label="stdin control-map" if use_stdin else f"control-map {map_path}",
        )

    segments = parse_control_segments_csv(
        payload,
        default_stretch=default_stretch,
        default_pitch_ratio=default_pitch_ratio,
    )
    if not segments:
        return []

    mode = str(getattr(args, "pitch_lowconf_mode", "hold")).strip().lower()
    if mode not in {"hold", "unity", "interp"}:
        raise ValueError(f"Unsupported low-confidence mode: {mode}")

    segments = apply_control_confidence_policy(
        segments,
        conf_min=float(getattr(args, "pitch_conf_min", 0.0)),
        mode=mode,  # type: ignore[arg-type]
        fallback_ratio=default_pitch_ratio,
    )
    segments = smooth_control_ratios(
        segments,
        smooth_ms=float(getattr(args, "pitch_map_smooth_ms", 0.0)),
    )
    return segments


def _lock_channel_phase_to_reference(
    reference: np.ndarray,
    target: np.ndarray,
    config: VocoderConfig,
    *,
    strength: float,
) -> np.ndarray:
    """Constrain target channel phase trajectory to a reference channel."""
    blend = float(np.clip(strength, 0.0, 1.0))
    if blend <= 1e-9:
        return np.asarray(target, dtype=np.float64)

    if reference.size == 0 or target.size == 0:
        return np.asarray(target, dtype=np.float64)

    n = min(reference.size, target.size)
    ref = np.asarray(reference[:n], dtype=np.float64)
    tgt = np.asarray(target[:n], dtype=np.float64)

    cfg = replace(config, transient_preserve=False)
    ref_spec = stft(ref, cfg)
    tgt_spec = stft(tgt, cfg)
    bins = min(ref_spec.shape[0], tgt_spec.shape[0])
    frames = min(ref_spec.shape[1], tgt_spec.shape[1])
    if bins == 0 or frames == 0:
        return np.asarray(force_length(tgt, target.size), dtype=np.float64)

    ref_spec = ref_spec[:bins, :frames]
    tgt_spec = tgt_spec[:bins, :frames]

    ref_phase = np.angle(ref_spec)
    tgt_phase = np.angle(tgt_spec)
    tgt_mag = np.abs(tgt_spec)

    base_delta = principal_angle(tgt_phase[:, :1] - ref_phase[:, :1])
    current_delta = principal_angle(tgt_phase - ref_phase)
    desired_delta = (1.0 - blend) * current_delta + blend * base_delta
    locked_phase = ref_phase + desired_delta
    locked_spec = tgt_mag * np.exp(1j * locked_phase)

    locked = istft(locked_spec, cfg, expected_length=n)
    if target.size > n:
        locked = force_length(locked, target.size)
    return np.asarray(locked, dtype=np.float64)


def process_audio_block(
    audio: np.ndarray,
    sr: int,
    args: argparse.Namespace,
    config: VocoderConfig,
    *,
    stretch: float,
    pitch_ratio: float,
    progress_callback_factory: Callable[[float, float, str], ProgressCallback | None] | None = None,
) -> AudioBlockResult:
    # Internal stretch combines explicit time stretch with pitch-resample compensation.
    internal_stretch = float(stretch * pitch_ratio)
    if internal_stretch <= 0.0:
        raise ValueError("Computed internal stretch must be > 0")
    if abs(internal_stretch - 1.0) <= 1e-10 and abs(pitch_ratio - 1.0) <= 1e-10:
        return AudioBlockResult(
            audio=np.asarray(audio, dtype=np.float64).copy(),
            internal_stretch=internal_stretch,
            sync_plan=None,
            stage_count=1,
        )

    transient_mode = str(getattr(args, "transient_mode", "off")).strip().lower()
    if transient_mode == "off" and bool(getattr(args, "transient_preserve", False)):
        transient_mode = "reset"
    coherence_strength = float(np.clip(getattr(args, "coherence_strength", 0.0), 0.0, 1.0))
    stereo_mode = str(getattr(args, "stereo_mode", "independent")).strip().lower()

    ms_active = stereo_mode == "mid_side_lock" and audio.shape[1] == 2
    # Mid/side mode keeps stereo image operations in M/S space for better coherence control.
    working_audio = lr_to_ms(audio) if ms_active else audio

    stretch_mode = str(getattr(args, "stretch_mode", "auto")).strip().lower()
    auto_threshold = max(1.01, float(getattr(args, "extreme_stretch_threshold", 2.0)))
    extreme_forced = bool(getattr(args, "extreme_time_stretch", False))
    if stretch_mode == "multistage":
        use_multistage = True
    elif stretch_mode == "standard":
        use_multistage = extreme_forced
    else:
        use_multistage = (
            extreme_forced
            or internal_stretch > auto_threshold
            or internal_stretch < (1.0 / auto_threshold)
        )

    stage_count = 1
    if use_multistage:
        stage_count = len(compute_multistage_stretches(internal_stretch, args.max_stage_stretch))

    multires_enabled = bool(getattr(args, "multires_fusion", False))
    multires_ffts = list(getattr(args, "_multires_ffts", [config.n_fft]))
    multires_weights = list(getattr(args, "_multires_weights", [1.0 for _ in multires_ffts]))
    if multires_enabled and not multires_ffts:
        multires_enabled = False

    sync_plan: FourierSyncPlan | None = None
    use_global_sync_plan = args.fourier_sync and (not use_multistage) and (not multires_enabled)
    if use_global_sync_plan:
        sync_source = (
            working_audio[:, 0]
            if args.analysis_channel == "first"
            else np.mean(working_audio, axis=1)
        )
        callback = (
            None
            if progress_callback_factory is None
            else progress_callback_factory(0.10, 0.25, "f0 prescan")
        )
        sync_plan = build_fourier_sync_plan(
            signal=sync_source,
            sample_rate=sr,
            config=config,
            f0_min_hz=args.f0_min,
            f0_max_hz=args.f0_max,
            min_fft=args.fourier_sync_min_fft,
            max_fft=args.fourier_sync_max_fft,
            smooth_span=args.fourier_sync_smooth,
            progress_callback=callback,
        )

    def _run_core_stretch(
        source_ch: np.ndarray,
        run_config: VocoderConfig,
        callback: ProgressCallback | None,
    ) -> tuple[np.ndarray, int]:
        if multires_enabled:
            stretched_i, stage_count_i = phase_vocoder_time_stretch_multires_fusion(
                source_ch,
                internal_stretch,
                run_config,
                fft_sizes=multires_ffts,
                weights=multires_weights,
                use_multistage=use_multistage,
                max_stage_stretch=args.max_stage_stretch,
                use_fourier_sync=args.fourier_sync,
                sample_rate=sr,
                f0_min_hz=args.f0_min,
                f0_max_hz=args.f0_max,
                fourier_sync_min_fft=args.fourier_sync_min_fft,
                fourier_sync_max_fft=args.fourier_sync_max_fft,
                fourier_sync_smooth=args.fourier_sync_smooth,
                progress_callback=callback,
            )
            return np.asarray(stretched_i, dtype=np.float64), int(stage_count_i)

        if use_global_sync_plan:
            if sync_plan is None:  # pragma: no cover - defensive
                raise RuntimeError("Fourier-sync plan is unavailable")
            stretched_i = phase_vocoder_time_stretch_fourier_sync(
                source_ch,
                internal_stretch,
                run_config,
                sync_plan,
                progress_callback=callback,
            )
            return np.asarray(stretched_i, dtype=np.float64), 1

        # Standard branch picks single-pass or multistage strategy based on resolved flags.
        stretched_i, stage_count_i = stretch_channel_with_strategy(
            signal=source_ch,
            stretch=internal_stretch,
            config=run_config,
            use_multistage=use_multistage,
            max_stage_stretch=args.max_stage_stretch,
            use_fourier_sync=args.fourier_sync,
            sample_rate=sr,
            f0_min_hz=args.f0_min,
            f0_max_hz=args.f0_max,
            fourier_sync_min_fft=args.fourier_sync_min_fft,
            fourier_sync_max_fft=args.fourier_sync_max_fft,
            fourier_sync_smooth=args.fourier_sync_smooth,
            progress_callback=callback,
        )
        return np.asarray(stretched_i, dtype=np.float64), int(stage_count_i)

    channel_start = 0.25 if use_global_sync_plan else 0.10
    channel_end = 0.88
    processed_channels: list[np.ndarray] = []
    for ch in range(working_audio.shape[1]):
        sub_start = channel_start + (channel_end - channel_start) * (ch / working_audio.shape[1])
        sub_end = channel_start + (channel_end - channel_start) * (
            (ch + 1) / working_audio.shape[1]
        )
        detail = f"channel {ch + 1}/{working_audio.shape[1]}"
        callback = (
            None
            if progress_callback_factory is None
            else progress_callback_factory(sub_start, sub_end, detail)
        )
        source_ch = working_audio[:, ch]

        if transient_mode in {"hybrid", "wsola"}:
            _, transient_mask, _ = detect_transient_regions(
                source_ch,
                sr,
                n_fft=config.n_fft,
                hop_size=config.hop_size,
                sensitivity=float(args.transient_sensitivity),
                protect_ms=float(args.transient_protect_ms),
                crossfade_ms=float(args.transient_crossfade_ms),
                center=config.center,
            )
            frame_ms = 1000.0 * max(config.win_length, config.hop_size * 2) / float(max(1, sr))
            hop_ms = 1000.0 * max(1, config.hop_size) / float(max(1, sr))
            stretched_wsola = wsola_time_stretch(
                source_ch,
                internal_stretch,
                sr,
                frame_ms=max(12.0, min(90.0, frame_ms)),
                analysis_hop_ms=max(2.0, min(30.0, hop_ms)),
                search_ms=max(4.0, min(60.0, float(args.transient_protect_ms) * 0.8)),
            )
            target_len = max(1, int(round(source_ch.size * internal_stretch)))
            stretched_wsola = np.asarray(
                force_length(stretched_wsola, target_len), dtype=np.float64
            )

            if transient_mode == "wsola":
                stretched = stretched_wsola
                stage_count_i = 1
            else:
                # Hybrid mode: PV for steady-state + WSOLA for transient regions.
                steady_cfg = replace(config, transient_preserve=False)
                stretched_pv, stage_count_i = _run_core_stretch(source_ch, steady_cfg, callback)
                stretched_pv = np.asarray(force_length(stretched_pv, target_len), dtype=np.float64)
                mask_out = map_mask_to_output(transient_mask, internal_stretch, target_len)
                fade_samples = int(
                    round(sr * max(0.0, float(args.transient_crossfade_ms)) / 1000.0)
                )
                blend = smooth_binary_mask(mask_out, fade_samples)
                stretched = stretched_pv * (1.0 - blend) + stretched_wsola * blend
        else:
            run_cfg = replace(config, transient_preserve=(transient_mode == "reset"))
            stretched, stage_count_i = _run_core_stretch(source_ch, run_cfg, callback)

        stage_count = max(stage_count, int(stage_count_i))

        if abs(pitch_ratio - 1.0) > 1e-10:
            # Pitch shift is done by time-domain resampling after internal time stretch.
            pitch_len = max(1, int(round(stretched.size / pitch_ratio)))
            shifted = resample_1d(stretched, pitch_len, args.resample_mode)
        else:
            shifted = stretched

        if args.pitch_mode == "formant-preserving" and abs(pitch_ratio - 1.0) > 1e-10:
            shifted = apply_formant_preservation(
                source_ch,
                shifted,
                config,
                lifter=args.formant_lifter,
                strength=args.formant_strength,
                max_gain_db=args.formant_max_gain_db,
            )
        processed_channels.append(shifted)

    out_len = max(ch_data.size for ch_data in processed_channels)
    out_audio = np.zeros((out_len, len(processed_channels)), dtype=np.float64)
    for ch, ch_data in enumerate(processed_channels):
        out_audio[: ch_data.size, ch] = ch_data

    if ms_active:
        if coherence_strength > 1e-9:
            # Optional phase alignment on side channel reduces L/R wander after heavy processing.
            out_audio[:, 1] = _lock_channel_phase_to_reference(
                out_audio[:, 0],
                out_audio[:, 1],
                config,
                strength=coherence_strength,
            )
        out_audio = ms_to_lr(out_audio)
    elif stereo_mode == "ref_channel_lock" and out_audio.shape[1] > 1 and coherence_strength > 1e-9:
        ref_idx = validate_ref_channel(int(getattr(args, "ref_channel", 0)), out_audio.shape[1])
        reference = out_audio[:, ref_idx].copy()
        for ch in range(out_audio.shape[1]):
            if ch == ref_idx:
                continue
            out_audio[:, ch] = _lock_channel_phase_to_reference(
                reference,
                out_audio[:, ch],
                config,
                strength=coherence_strength,
            )

    return AudioBlockResult(
        audio=out_audio,
        internal_stretch=internal_stretch,
        sync_plan=sync_plan,
        stage_count=stage_count,
    )


def resolve_base_stretch(args: argparse.Namespace, in_samples: int, sr: int) -> float:
    if args.target_duration is not None:
        return args.target_duration * sr / max(in_samples, 1)
    value = args.time_stretch
    if _looks_like_control_signal_reference(value):
        return 1.0
    return float(_parse_scalar_cli_value(value, context="--time-stretch"))


def build_vocoder_config_from_args(args: argparse.Namespace) -> VocoderConfig:
    return VocoderConfig(
        n_fft=int(args.n_fft),
        win_length=int(args.win_length),
        hop_size=int(args.hop_size),
        window=str(args.window),
        kaiser_beta=float(args.kaiser_beta),
        transform=normalize_transform_name(str(args.transform)),
        center=not bool(args.no_center),
        phase_locking=str(args.phase_locking),  # type: ignore[arg-type]
        phase_engine=str(args.phase_engine),  # type: ignore[arg-type]
        ambient_phase_mix=float(args.ambient_phase_mix),
        phase_random_seed=args.phase_random_seed,
        transient_preserve=bool(args.transient_preserve),
        transient_threshold=float(args.transient_threshold),
        onset_time_credit=bool(args.onset_time_credit),
        onset_credit_pull=float(args.onset_credit_pull),
        onset_credit_max=float(args.onset_credit_max),
        onset_realign=not bool(args.no_onset_realign),
    )


def _finalize_dynamic_segment_values(
    *,
    args: argparse.Namespace,
    stretch: float,
    pitch_ratio: float,
    overrides: dict[str, float],
) -> tuple[float, float, dict[str, Any]]:
    out: dict[str, Any] = {}

    stretch = max(1e-8, float(stretch))
    pitch_ratio = max(1e-8, float(pitch_ratio))

    if any(key in overrides for key in ("n_fft", "win_length", "hop_size")):
        n_fft = int(round(overrides.get("n_fft", float(args.n_fft))))
        win_length = int(round(overrides.get("win_length", float(args.win_length))))
        hop_size = int(round(overrides.get("hop_size", float(args.hop_size))))
        n_fft = max(16, n_fft)
        win_length = max(1, min(win_length, n_fft))
        hop_size = max(1, min(hop_size, win_length))
        out["n_fft"] = n_fft
        out["win_length"] = win_length
        out["hop_size"] = hop_size

    if "kaiser_beta" in overrides:
        out["kaiser_beta"] = max(0.0, float(overrides["kaiser_beta"]))
    if "fourier_sync_min_fft" in overrides:
        out["fourier_sync_min_fft"] = max(16, int(round(float(overrides["fourier_sync_min_fft"]))))
    if "fourier_sync_max_fft" in overrides:
        max_fft = max(16, int(round(float(overrides["fourier_sync_max_fft"]))))
        min_fft = int(
            out.get("fourier_sync_min_fft", int(getattr(args, "fourier_sync_min_fft", 256)))
        )
        out["fourier_sync_max_fft"] = max(min_fft, max_fft)
    if "fourier_sync_smooth" in overrides:
        out["fourier_sync_smooth"] = max(1, int(round(float(overrides["fourier_sync_smooth"]))))
    if "ambient_phase_mix" in overrides:
        out["ambient_phase_mix"] = float(np.clip(overrides["ambient_phase_mix"], 0.0, 1.0))
    if "transient_threshold" in overrides:
        out["transient_threshold"] = max(1e-9, float(overrides["transient_threshold"]))
    if "transient_sensitivity" in overrides:
        out["transient_sensitivity"] = float(np.clip(overrides["transient_sensitivity"], 0.0, 1.0))
    if "transient_protect_ms" in overrides:
        out["transient_protect_ms"] = max(1e-9, float(overrides["transient_protect_ms"]))
    if "transient_crossfade_ms" in overrides:
        out["transient_crossfade_ms"] = max(0.0, float(overrides["transient_crossfade_ms"]))
    if "coherence_strength" in overrides:
        out["coherence_strength"] = float(np.clip(overrides["coherence_strength"], 0.0, 1.0))
    if "onset_credit_pull" in overrides:
        out["onset_credit_pull"] = float(np.clip(overrides["onset_credit_pull"], 0.0, 1.0))
    if "onset_credit_max" in overrides:
        out["onset_credit_max"] = max(0.0, float(overrides["onset_credit_max"]))
    if "extreme_stretch_threshold" in overrides:
        out["extreme_stretch_threshold"] = max(
            1.0000001, float(overrides["extreme_stretch_threshold"])
        )
    if "max_stage_stretch" in overrides:
        out["max_stage_stretch"] = max(1.0000001, float(overrides["max_stage_stretch"]))
    if "formant_lifter" in overrides:
        out["formant_lifter"] = max(0, int(round(float(overrides["formant_lifter"]))))
    if "formant_strength" in overrides:
        out["formant_strength"] = float(np.clip(overrides["formant_strength"], 0.0, 1.0))
    if "formant_max_gain_db" in overrides:
        out["formant_max_gain_db"] = max(1e-9, float(overrides["formant_max_gain_db"]))

    return stretch, pitch_ratio, out


def build_dynamic_control_segments(
    *,
    args: argparse.Namespace,
    sr: int,
    total_seconds: float,
    base_stretch: float,
    base_pitch_ratio: float,
) -> list[ControlSegment]:
    refs: dict[str, DynamicControlRef] = dict(getattr(args, "_dynamic_control_refs", {}) or {})
    if not refs:
        return []

    signals: list[DynamicControlSignal] = []
    for parameter, ref in refs.items():
        signal = load_dynamic_control_signal(ref, total_seconds=total_seconds)
        if signal.parameter != parameter:
            signal = DynamicControlSignal(
                parameter=parameter,
                interpolation=signal.interpolation,
                order=signal.order,
                times_sec=signal.times_sec,
                values=signal.values,
            )
        signals.append(signal)
    if not signals:
        return []

    boundaries: set[float] = {0.0, max(0.0, float(total_seconds))}
    for signal in signals:
        boundaries.update(float(t) for t in signal.times_sec)

    if any(signal.interpolation != "none" for signal in signals):
        hop_seconds = max(1.0 / float(sr), float(args.hop_size) / float(sr))
        cursor = hop_seconds
        while cursor < total_seconds:
            boundaries.add(float(cursor))
            cursor += hop_seconds

    edges = np.asarray(sorted(boundaries), dtype=np.float64)
    edges = np.clip(edges, 0.0, max(0.0, float(total_seconds)))
    edges, _dummy = _deduplicate_points(edges, np.zeros_like(edges))
    if edges.size < 2:
        edges = np.asarray([0.0, max(0.0, float(total_seconds))], dtype=np.float64)

    mids = 0.5 * (edges[:-1] + edges[1:])
    sampled: dict[str, np.ndarray] = {}
    for signal in signals:
        sampled[signal.parameter] = _sample_dynamic_signal(signal, mids)

    out: list[ControlSegment] = []
    for idx, (start_sec, end_sec) in enumerate(pairwise(edges)):
        if end_sec <= start_sec:
            continue
        stretch = float(base_stretch)
        pitch_ratio = float(base_pitch_ratio)
        overrides: dict[str, float] = {}
        for parameter, values in sampled.items():
            value = float(values[idx])
            if parameter == "time_stretch":
                stretch = value
            elif parameter == "pitch_ratio":
                pitch_ratio = value
            else:
                overrides[parameter] = value
        stretch, pitch_ratio, clean = _finalize_dynamic_segment_values(
            args=args,
            stretch=stretch,
            pitch_ratio=pitch_ratio,
            overrides=overrides,
        )
        out.append(
            ControlSegment(
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                stretch=float(stretch),
                pitch_ratio=float(pitch_ratio),
                confidence=1.0,
                overrides=(clean if clean else None),
            )
        )
    return out


def compute_output_path(
    input_path: Path,
    output_dir: Path | None,
    suffix: str,
    output_format: str | None,
) -> Path:
    from pvx.core.voc_jobs import compute_output_path as jobs_compute_output_path

    return jobs_compute_output_path(input_path, output_dir, suffix, output_format)


def _stream_format_name(output_format: str | None, output_path: Path | None = None) -> str:
    from pvx.core.voc_jobs import _stream_format_name as jobs_stream_format_name

    return jobs_stream_format_name(output_format, output_path=output_path)


def _read_audio_input(input_path: Path) -> tuple[np.ndarray, int]:
    from pvx.core.voc_jobs import _read_audio_input as jobs_read_audio_input

    return jobs_read_audio_input(input_path)


def read_audio_input(input_path: Path) -> tuple[np.ndarray, int]:
    return _read_audio_input(input_path)


def _write_audio_output(
    output_path: Path,
    audio: np.ndarray,
    sr: int,
    args: argparse.Namespace,
    *,
    subtype: str | None = None,
) -> None:
    from pvx.core.voc_jobs import _write_audio_output as jobs_write_audio_output

    jobs_write_audio_output(output_path, audio, sr, args, subtype=subtype)


def concat_audio_chunks(chunks: list[np.ndarray], *, sr: int, crossfade_ms: float) -> np.ndarray:
    from pvx.core.voc_jobs import concat_audio_chunks as jobs_concat_audio_chunks

    return jobs_concat_audio_chunks(chunks, sr=sr, crossfade_ms=crossfade_ms)


def build_uniform_control_segments(
    *,
    total_seconds: float,
    segment_seconds: float,
    stretch: float,
    pitch_ratio: float,
) -> list[ControlSegment]:
    from pvx.core.voc_jobs import (
        build_uniform_control_segments as jobs_build_uniform_control_segments,
    )

    return jobs_build_uniform_control_segments(
        total_seconds=total_seconds,
        segment_seconds=segment_seconds,
        stretch=stretch,
        pitch_ratio=pitch_ratio,
    )


def resolve_checkpoint_context(
    *,
    input_path: Path,
    args: argparse.Namespace,
    base_stretch: float,
    pitch_ratio: float,
) -> tuple[str, Path] | None:
    from pvx.core.voc_jobs import resolve_checkpoint_context as jobs_resolve_checkpoint_context

    return jobs_resolve_checkpoint_context(
        input_path=input_path,
        args=args,
        base_stretch=base_stretch,
        pitch_ratio=pitch_ratio,
    )


def load_checkpoint_chunk(path: Path) -> np.ndarray:
    from pvx.core.voc_jobs import load_checkpoint_chunk as jobs_load_checkpoint_chunk

    return jobs_load_checkpoint_chunk(path)


def save_checkpoint_chunk(path: Path, values: np.ndarray) -> None:
    from pvx.core.voc_jobs import save_checkpoint_chunk as jobs_save_checkpoint_chunk

    jobs_save_checkpoint_chunk(path, values)


def write_manifest(
    path: Path,
    entries: list[dict[str, Any]],
    *,
    append: bool,
) -> None:
    from pvx.core.voc_jobs import write_manifest as jobs_write_manifest

    jobs_write_manifest(path, entries, append=append)


def process_file(
    input_path: Path,
    args: argparse.Namespace,
    config: VocoderConfig,
    file_index: int = 0,
    file_total: int = 1,
) -> JobResult:
    from pvx.core.voc_jobs import process_file as jobs_process_file

    return jobs_process_file(
        input_path,
        args,
        config,
        file_index=file_index,
        file_total=file_total,
    )


def force_length_multi(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        return audio[:length, :]
    pad = np.zeros((length - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
    return np.vstack([audio, pad])


def resample_multi(audio: np.ndarray, output_samples: int, mode: ResampleMode) -> np.ndarray:
    out = np.zeros((output_samples, audio.shape[1]), dtype=np.float64)
    for ch in range(audio.shape[1]):
        out[:, ch] = resample_1d(audio[:, ch], output_samples, mode)
    return out


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from pvx.core.voc_parser import validate_args as parser_validate_args

    parser_validate_args(args, parser)


def build_parser() -> argparse.ArgumentParser:
    from pvx.core.voc_parser import build_parser as parser_build_parser

    return parser_build_parser()


def run_guided_mode(args: argparse.Namespace) -> argparse.Namespace:
    from pvx.voc_cli import run_guided_mode as cli_run_guided_mode

    return cli_run_guided_mode(args)


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    from pvx.voc_cli import expand_inputs as cli_expand_inputs

    return cli_expand_inputs(patterns)


def collect_cli_flags(argv: Iterable[str]) -> set[str]:
    from pvx.core.voc_console import collect_cli_flags as console_collect_cli_flags

    return console_collect_cli_flags(argv)


def apply_named_preset(
    args: argparse.Namespace,
    *,
    preset: str,
    provided_flags: set[str],
) -> list[str]:
    from pvx.core.voc_console import apply_named_preset as console_apply_named_preset

    return console_apply_named_preset(args, preset=preset, provided_flags=provided_flags)


def main(argv: list[str] | None = None) -> int:
    from pvx.voc_cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
