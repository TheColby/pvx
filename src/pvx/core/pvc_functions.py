#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""PVC-style function-stream utilities for control-rate map authoring.

Phase 6 coverage:
- envelope: generate deterministic control trajectories
- reshape: transform existing control trajectories
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pvx.core.pvc_ops import evaluate_scalar_control

InterpMode = Literal[
    "none",
    "stairstep",
    "nearest",
    "linear",
    "cubic",
    "polynomial",
    "exponential",
    "s_curve",
    "smootherstep",
]
EnvelopeMode = Literal["adsr", "ramp", "exp", "sine", "triangle", "square", "saw_up", "saw_down"]
ReshapeOperation = Literal[
    "scale",
    "offset",
    "clip",
    "pow",
    "normalize",
    "invert",
    "smooth",
    "time-scale",
    "time-shift",
    "resample",
]

INTERP_CHOICES: tuple[InterpMode, ...] = (
    "none",
    "stairstep",
    "nearest",
    "linear",
    "cubic",
    "polynomial",
    "exponential",
    "s_curve",
    "smootherstep",
)
ENVELOPE_MODES: tuple[EnvelopeMode, ...] = (
    "adsr",
    "ramp",
    "exp",
    "sine",
    "triangle",
    "square",
    "saw_up",
    "saw_down",
)
RESHAPE_OPERATIONS: tuple[ReshapeOperation, ...] = (
    "scale",
    "offset",
    "clip",
    "pow",
    "normalize",
    "invert",
    "smooth",
    "time-scale",
    "time-shift",
    "resample",
)


def _sanitize_times_values(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if t.size != v.size:
        raise ValueError("times and values length mismatch")
    if t.size == 0:
        raise ValueError("control stream has no points")
    finite = np.isfinite(t) & np.isfinite(v)
    t = t[finite]
    v = v[finite]
    if t.size == 0:
        raise ValueError("control stream has no finite points")
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    v = v[order]
    out_t: list[float] = []
    out_v: list[float] = []
    for tt, vv in zip(t.tolist(), v.tolist()):
        # Merge exact-time duplicates by keeping the latest value (authoring-tool friendly behavior).
        if out_t and abs(tt - out_t[-1]) <= 1e-12:
            out_v[-1] = vv
        else:
            out_t.append(float(tt))
            out_v.append(float(vv))
    return np.asarray(out_t, dtype=np.float64), np.asarray(out_v, dtype=np.float64)


def _auto_format_from_path(path: Path | None, *, default: str = "csv") -> str:
    if path is None:
        return default
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    return "csv"


def parse_control_points_payload(
    payload: str,
    *,
    key: str,
    source_label: str,
    fmt: str = "csv",
) -> tuple[np.ndarray, np.ndarray]:
    text = str(payload)
    fmt_norm = str(fmt).strip().lower()
    if fmt_norm not in {"csv", "json"}:
        raise ValueError(f"{source_label}: unsupported format '{fmt}'")

    points_t: list[float] = []
    points_v: list[float] = []
    key_norm = str(key).strip()
    if not key_norm:
        raise ValueError("key must not be empty")

    if fmt_norm == "csv":
        reader = csv.DictReader(io.StringIO(text))
        fields = list(reader.fieldnames or [])
        if "time_sec" not in fields:
            raise ValueError(f"{source_label}: CSV must contain 'time_sec' column")
        if key_norm not in fields:
            if key_norm == "value" and "value" in fields:
                pass
            else:
                raise ValueError(f"{source_label}: CSV missing key column '{key_norm}'")
        for idx, row in enumerate(reader, start=2):
            t_raw = row.get("time_sec")
            v_raw = row.get(key_norm)
            if v_raw is None and key_norm != "value":
                continue
            if v_raw is None:
                v_raw = row.get("value")
            if t_raw is None or v_raw is None:
                continue
            try:
                t_val = float(str(t_raw).strip())
                v_val = float(str(v_raw).strip())
            except Exception as exc:
                raise ValueError(f"{source_label}: invalid numeric value at CSV row {idx}") from exc
            points_t.append(t_val)
            points_v.append(v_val)
    else:
        # JSON accepts either an object with points/control or a raw list of point rows.
        root = json.loads(text)
        rows: list[dict[str, Any]] = []
        if isinstance(root, dict):
            if isinstance(root.get("points"), list):
                rows = [item for item in root["points"] if isinstance(item, dict)]
            elif isinstance(root.get("control"), list):
                rows = [item for item in root["control"] if isinstance(item, dict)]
        elif isinstance(root, list):
            rows = [item for item in root if isinstance(item, dict)]
        if not rows:
            raise ValueError(f"{source_label}: JSON contains no point rows")
        for idx, row in enumerate(rows, start=1):
            t_raw = row.get("time_sec", row.get("time"))
            v_raw = row.get(key_norm, row.get("value"))
            if t_raw is None or v_raw is None:
                continue
            try:
                t_val = float(t_raw)
                v_val = float(v_raw)
            except Exception as exc:
                raise ValueError(f"{source_label}: invalid numeric value at JSON point {idx}") from exc
            points_t.append(t_val)
            points_v.append(v_val)

    if not points_t:
        raise ValueError(f"{source_label}: no usable points")
    return _sanitize_times_values(np.asarray(points_t), np.asarray(points_v))


def load_control_points(
    path: Path,
    *,
    key: str,
    fmt: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"Control map not found: {p}")
    fmt_norm = str(fmt or _auto_format_from_path(p)).strip().lower()
    payload = p.read_text(encoding="utf-8")
    return parse_control_points_payload(payload, key=key, source_label=str(p), fmt=fmt_norm)


def dump_control_points_csv(
    times: np.ndarray,
    values: np.ndarray,
    *,
    key: str,
) -> str:
    t, v = _sanitize_times_values(times, values)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["time_sec", key])
    for tt, vv in zip(t.tolist(), v.tolist()):
        writer.writerow([f"{tt:.9f}", f"{vv:.9f}"])
    return buffer.getvalue()


def dump_control_points_json(
    times: np.ndarray,
    values: np.ndarray,
    *,
    key: str,
) -> str:
    t, v = _sanitize_times_values(times, values)
    payload = {
        "format": "pvx-control-v1",
        "key": str(key),
        "points": [{"time_sec": float(tt), str(key): float(vv)} for tt, vv in zip(t.tolist(), v.tolist())],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def generate_envelope_points(
    *,
    duration_sec: float,
    rate_hz: float,
    mode: EnvelopeMode = "adsr",
    start: float = 0.0,
    peak: float = 1.0,
    sustain: float = 0.7,
    end: float = 0.0,
    attack_sec: float = 0.1,
    decay_sec: float = 0.2,
    release_sec: float = 0.2,
    exp_curve: float = 4.0,
    sine_cycles: float = 1.0,
    sine_phase_rad: float = 0.0,
    duty_cycle: float = 0.5,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    duration = float(duration_sec)
    rate = float(rate_hz)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be > 0")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("rate_hz must be > 0")

    count = max(2, int(round(duration * rate)) + 1)
    times = np.linspace(0.0, duration, num=count, endpoint=True, dtype=np.float64)
    values = np.zeros_like(times)
    mode_norm = str(mode).lower()

    if mode_norm == "adsr":
        a = max(0.0, float(attack_sec))
        d = max(0.0, float(decay_sec))
        r = max(0.0, float(release_sec))
        total_env = a + d + r
        if total_env > duration and total_env > 0.0:
            # If ADSR times exceed clip duration, compress stages proportionally.
            scale = duration / total_env
            a *= scale
            d *= scale
            r *= scale
        s_dur = max(0.0, duration - (a + d + r))

        p0 = 0.0  # noqa: F841
        p1 = a
        p2 = p1 + d
        p3 = p2 + s_dur
        for idx, t in enumerate(times.tolist()):
            if t <= p1 and a > 1e-12:
                u = t / a
                values[idx] = float(start) + (float(peak) - float(start)) * u
            elif t <= p2 and d > 1e-12:
                u = (t - p1) / d
                values[idx] = float(peak) + (float(sustain) - float(peak)) * u
            elif t <= p3 or r <= 1e-12:
                values[idx] = float(sustain)
            else:
                u = min(1.0, max(0.0, (t - p3) / max(r, 1e-12)))
                values[idx] = float(sustain) + (float(end) - float(sustain)) * u
    elif mode_norm == "ramp":
        values = float(start) + (float(end) - float(start)) * (times / max(duration, 1e-12))
    elif mode_norm == "exp":
        c = max(1e-6, float(exp_curve))
        u = times / max(duration, 1e-12)
        shaped = (1.0 - np.exp(-c * u)) / (1.0 - np.exp(-c))
        values = float(start) + (float(end) - float(start)) * shaped
    elif mode_norm == "sine":
        u = times / max(duration, 1e-12)
        phase = (2.0 * np.pi * float(sine_cycles) * u) + float(sine_phase_rad)
        values = float(start) + float(peak) * np.sin(phase)
    elif mode_norm == "triangle":
        cycles = float(sine_cycles)
        phase_cycles = float(sine_phase_rad) / (2.0 * np.pi)
        u = np.mod((times / max(duration, 1e-12)) * cycles + phase_cycles, 1.0)
        tri = 1.0 - 4.0 * np.abs(u - 0.5)
        values = float(start) + float(peak) * tri
    elif mode_norm == "square":
        duty = float(np.clip(float(duty_cycle), 1e-6, 1.0 - 1e-6))
        cycles = float(sine_cycles)
        phase_cycles = float(sine_phase_rad) / (2.0 * np.pi)
        u = np.mod((times / max(duration, 1e-12)) * cycles + phase_cycles, 1.0)
        sq = np.where(u < duty, 1.0, -1.0)
        values = float(start) + float(peak) * sq
    elif mode_norm == "saw_up":
        cycles = float(sine_cycles)
        phase_cycles = float(sine_phase_rad) / (2.0 * np.pi)
        u = np.mod((times / max(duration, 1e-12)) * cycles + phase_cycles, 1.0)
        saw = (2.0 * u) - 1.0
        values = float(start) + float(peak) * saw
    elif mode_norm == "saw_down":
        cycles = float(sine_cycles)
        phase_cycles = float(sine_phase_rad) / (2.0 * np.pi)
        u = np.mod((times / max(duration, 1e-12)) * cycles + phase_cycles, 1.0)
        saw = 1.0 - (2.0 * u)
        values = float(start) + float(peak) * saw
    else:
        raise ValueError(f"Unsupported envelope mode: {mode}")

    if min_value is not None or max_value is not None:
        lo = -np.inf if min_value is None else float(min_value)
        hi = np.inf if max_value is None else float(max_value)
        values = np.clip(values, lo, hi)

    return _sanitize_times_values(times, values)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    src = np.asarray(values, dtype=np.float64).reshape(-1)
    if src.size == 0:
        return src
    width = max(1, int(window))
    if width <= 1:
        return src.copy()
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(src, kernel, mode="same")


def reshape_control_points(
    times: np.ndarray,
    values: np.ndarray,
    *,
    operation: ReshapeOperation,
    factor: float = 1.0,
    offset: float = 0.0,
    min_value: float | None = None,
    max_value: float | None = None,
    exponent: float = 1.0,
    window: int = 5,
    target_min: float = 0.0,
    target_max: float = 1.0,
    interp: InterpMode = "linear",
    order: int = 3,
    resample_rate_hz: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    t, v = _sanitize_times_values(times, values)
    op = str(operation).lower()
    out_t = t.copy()
    out_v = v.copy()

    if op == "scale":
        out_v = out_v * float(factor)
    elif op == "offset":
        out_v = out_v + float(offset)
    elif op == "clip":
        lo = -np.inf if min_value is None else float(min_value)
        hi = np.inf if max_value is None else float(max_value)
        out_v = np.clip(out_v, lo, hi)
    elif op == "pow":
        exp = float(exponent)
        out_v = np.sign(out_v) * np.power(np.abs(out_v), exp)
    elif op == "normalize":
        lo_src = float(np.min(out_v))
        hi_src = float(np.max(out_v))
        if abs(hi_src - lo_src) <= 1e-12:
            mid = 0.5 * (float(target_min) + float(target_max))
            out_v = np.full(out_v.shape, mid, dtype=np.float64)
        else:
            u = (out_v - lo_src) / (hi_src - lo_src)
            out_v = float(target_min) + u * (float(target_max) - float(target_min))
    elif op == "invert":
        denom = np.where(np.abs(out_v) <= 1e-12, np.nan, out_v)
        out_v = 1.0 / denom
        fill = 1.0 / 1e-12
        out_v = np.nan_to_num(out_v, nan=fill, posinf=fill, neginf=-fill)
    elif op == "smooth":
        out_v = _moving_average(out_v, int(window))
    elif op == "time-scale":
        out_t = out_t * float(factor)
    elif op == "time-shift":
        out_t = out_t + float(offset)
    elif op == "resample":
        rate = float(resample_rate_hz)
        if rate <= 0.0 or not math.isfinite(rate):
            raise ValueError("resample_rate_hz must be > 0")
        start = float(out_t[0])
        stop = float(out_t[-1])
        if stop <= start:
            return out_t, out_v
        count = max(2, int(round((stop - start) * rate)) + 1)
        new_t = np.linspace(start, stop, num=count, endpoint=True, dtype=np.float64)
        # Reuse shared interpolation logic so reshape semantics match runtime control evaluation.
        new_v = evaluate_scalar_control(
            new_t,
            out_t,
            out_v,
            mode=str(interp),  # type: ignore[arg-type]
            order=int(order),
            default_value=float(out_v[0]),
        )
        out_t = new_t
        out_v = np.asarray(new_v, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported reshape operation: {operation}")

    if min_value is not None or max_value is not None:
        if op not in {"clip"}:
            lo = -np.inf if min_value is None else float(min_value)
            hi = np.inf if max_value is None else float(max_value)
            out_v = np.clip(out_v, lo, hi)

    return _sanitize_times_values(out_t, out_v)
