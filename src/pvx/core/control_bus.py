#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Control-bus routing helpers for time-varying CSV maps."""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from typing import Iterable, Literal

RouteOp = Literal["source", "const", "inv", "pow", "mul", "add", "affine", "clip"]

_TARGET_NAMES: set[str] = {"stretch", "pitch_ratio"}
_SOURCE_NAMES: set[str] = {
    "stretch",
    "pitch_ratio",
    "confidence",
    "f0_hz",
    "rms",
    "rms_db",
    "zcr",
    "spectral_centroid_hz",
    "spectral_flatness",
    "spectral_flux",
    "onset_strength",
    "rolloff_hz",
    "voicing_prob",
    "pitch_stability",
    "harmonic_ratio",
    "rms_norm",
    "centroid_norm",
    "flux_norm",
    "onset_norm",
    "rolloff_norm",
}

_NAME_ALIASES: dict[str, str] = {
    "stretch": "stretch",
    "time_stretch": "stretch",
    "time-stretch": "stretch",
    "time_stretch_factor": "stretch",
    "time-stretch-factor": "stretch",
    "pitch_ratio": "pitch_ratio",
    "ratio": "pitch_ratio",
    "confidence": "confidence",
    "conf": "confidence",
    "pitch_confidence": "confidence",
    "f0_confidence": "confidence",
    "f0_hz": "f0_hz",
    "f0": "f0_hz",
    "hz": "f0_hz",
    "frequency_hz": "f0_hz",
    "rms": "rms",
    "rms_linear": "rms",
    "rms_db": "rms_db",
    "rms_dbfs": "rms_db",
    "energy_db": "rms_db",
    "zcr": "zcr",
    "zero_crossing_rate": "zcr",
    "centroid_hz": "spectral_centroid_hz",
    "spectral_centroid": "spectral_centroid_hz",
    "spectral_centroid_hz": "spectral_centroid_hz",
    "brightness_hz": "spectral_centroid_hz",
    "spectral_flatness": "spectral_flatness",
    "flatness": "spectral_flatness",
    "spectral_flux": "spectral_flux",
    "flux": "spectral_flux",
    "onset_strength": "onset_strength",
    "onset": "onset_strength",
    "rolloff_hz": "rolloff_hz",
    "spectral_rolloff_hz": "rolloff_hz",
    "voicing_prob": "voicing_prob",
    "voicing_probability": "voicing_prob",
    "pitch_stability": "pitch_stability",
    "harmonic_ratio": "harmonic_ratio",
    "harm_ratio": "harmonic_ratio",
    "rms_norm": "rms_norm",
    "energy_norm": "rms_norm",
    "centroid_norm": "centroid_norm",
    "brightness_norm": "centroid_norm",
    "flux_norm": "flux_norm",
    "onset_norm": "onset_norm",
    "rolloff_norm": "rolloff_norm",
}


@dataclass(frozen=True)
class ControlRoute:
    target: str
    op: RouteOp
    source: str | None
    value: float | None
    exponent: float | None
    params: tuple[float, ...] | None
    expression: str


def normalize_control_name(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    return _NAME_ALIASES.get(key, key)


def _parse_finite_float(value: str, *, context: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context} must not be empty")
    try:
        out = float(text)
    except Exception as exc:
        raise ValueError(f"{context} must be numeric: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{context} must be finite")
    return float(out)


_SOURCE_NAME_PATTERN = re.compile(r"^[a-z_][a-z0-9_]*$")


def _parse_signal_name(value: str, *, context: str, allowed: set[str] | None) -> str:
    name = normalize_control_name(value)
    if not name:
        raise ValueError(f"{context}: empty signal name")
    if allowed is None:
        if not _SOURCE_NAME_PATTERN.match(name):
            raise ValueError(
                f"{context}: unsupported signal '{value}'. "
                "Use lowercase alphanumeric/underscore names (example: spectral_flux)."
            )
        return name
    if name not in allowed:
        allowed_txt = ", ".join(sorted(allowed))
        raise ValueError(
            f"{context}: unsupported signal '{value}'. Allowed: {allowed_txt}"
        )
    return name


def parse_control_route(expression: str) -> ControlRoute:
    raw = str(expression).strip()
    if not raw:
        raise ValueError("Route expression must not be empty")
    if "=" not in raw:
        raise ValueError(f"Route expression must contain '=': {raw!r}")

    lhs, rhs = raw.split("=", 1)
    target = _parse_signal_name(
        lhs, context=f"route '{raw}' target", allowed=_TARGET_NAMES
    )
    rhs_text = rhs.strip()
    if not rhs_text:
        raise ValueError(f"Route expression has an empty right-hand side: {raw!r}")

    lower_rhs = rhs_text.lower()
    if lower_rhs.startswith("const(") and rhs_text.endswith(")"):
        value_text = rhs_text[6:-1]
        value = _parse_finite_float(value_text, context=f"route '{raw}' const()")
        return ControlRoute(
            target=target,
            op="const",
            source=None,
            value=value,
            exponent=None,
            params=None,
            expression=raw,
        )

    if lower_rhs.startswith("inv(") and rhs_text.endswith(")"):
        src_text = rhs_text[4:-1]
        source = _parse_signal_name(
            src_text, context=f"route '{raw}' inv()", allowed=None
        )
        return ControlRoute(
            target=target,
            op="inv",
            source=source,
            value=None,
            exponent=None,
            params=None,
            expression=raw,
        )

    if lower_rhs.startswith("pow(") and rhs_text.endswith(")"):
        inner = rhs_text[4:-1]
        parts = [chunk.strip() for chunk in inner.split(",", 1)]
        if len(parts) != 2:
            raise ValueError(
                f"route '{raw}' pow() must be in the form pow(source, exponent)"
            )
        source = _parse_signal_name(
            parts[0], context=f"route '{raw}' pow() source", allowed=None
        )
        exponent = _parse_finite_float(
            parts[1], context=f"route '{raw}' pow() exponent"
        )
        return ControlRoute(
            target=target,
            op="pow",
            source=source,
            value=None,
            exponent=exponent,
            params=None,
            expression=raw,
        )

    if lower_rhs.startswith("mul(") and rhs_text.endswith(")"):
        inner = rhs_text[4:-1]
        parts = [chunk.strip() for chunk in inner.split(",", 1)]
        if len(parts) != 2:
            raise ValueError(
                f"route '{raw}' mul() must be in the form mul(source, factor)"
            )
        source = _parse_signal_name(
            parts[0], context=f"route '{raw}' mul() source", allowed=None
        )
        factor = _parse_finite_float(parts[1], context=f"route '{raw}' mul() factor")
        return ControlRoute(
            target=target,
            op="mul",
            source=source,
            value=None,
            exponent=None,
            params=(factor,),
            expression=raw,
        )

    if lower_rhs.startswith("add(") and rhs_text.endswith(")"):
        inner = rhs_text[4:-1]
        parts = [chunk.strip() for chunk in inner.split(",", 1)]
        if len(parts) != 2:
            raise ValueError(
                f"route '{raw}' add() must be in the form add(source, offset)"
            )
        source = _parse_signal_name(
            parts[0], context=f"route '{raw}' add() source", allowed=None
        )
        offset = _parse_finite_float(parts[1], context=f"route '{raw}' add() offset")
        return ControlRoute(
            target=target,
            op="add",
            source=source,
            value=None,
            exponent=None,
            params=(offset,),
            expression=raw,
        )

    if lower_rhs.startswith("affine(") and rhs_text.endswith(")"):
        inner = rhs_text[7:-1]
        parts = [chunk.strip() for chunk in inner.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"route '{raw}' affine() must be affine(source, scale, bias)"
            )
        source = _parse_signal_name(
            parts[0], context=f"route '{raw}' affine() source", allowed=None
        )
        scale = _parse_finite_float(parts[1], context=f"route '{raw}' affine() scale")
        bias = _parse_finite_float(parts[2], context=f"route '{raw}' affine() bias")
        return ControlRoute(
            target=target,
            op="affine",
            source=source,
            value=None,
            exponent=None,
            params=(scale, bias),
            expression=raw,
        )

    if lower_rhs.startswith("clip(") and rhs_text.endswith(")"):
        inner = rhs_text[5:-1]
        parts = [chunk.strip() for chunk in inner.split(",")]
        if len(parts) != 3:
            raise ValueError(f"route '{raw}' clip() must be clip(source, lo, hi)")
        source = _parse_signal_name(
            parts[0], context=f"route '{raw}' clip() source", allowed=None
        )
        lo = _parse_finite_float(parts[1], context=f"route '{raw}' clip() lo")
        hi = _parse_finite_float(parts[2], context=f"route '{raw}' clip() hi")
        if hi < lo:
            raise ValueError(f"route '{raw}' clip() requires hi >= lo")
        return ControlRoute(
            target=target,
            op="clip",
            source=source,
            value=None,
            exponent=None,
            params=(lo, hi),
            expression=raw,
        )

    try:
        value = _parse_finite_float(rhs_text, context=f"route '{raw}' constant")
        return ControlRoute(
            target=target,
            op="const",
            source=None,
            value=value,
            exponent=None,
            params=None,
            expression=raw,
        )
    except ValueError:
        pass

    source = _parse_signal_name(rhs_text, context=f"route '{raw}' source", allowed=None)
    return ControlRoute(
        target=target,
        op="source",
        source=source,
        value=None,
        exponent=None,
        params=None,
        expression=raw,
    )


def parse_control_routes(expressions: Iterable[str]) -> list[ControlRoute]:
    out: list[ControlRoute] = []
    for raw in expressions:
        text = str(raw).strip()
        if not text:
            continue
        out.append(parse_control_route(text))
    return out


def _source_column_candidates(source: str) -> tuple[str, ...]:
    if source == "stretch":
        return (
            "stretch",
            "time_stretch",
            "time-stretch",
            "time_stretch_factor",
            "time-stretch-factor",
        )
    if source == "pitch_ratio":
        return ("pitch_ratio", "ratio")
    if source == "confidence":
        return ("confidence", "conf", "pitch_confidence", "f0_confidence")
    if source == "f0_hz":
        return ("f0_hz", "f0", "hz", "frequency_hz")
    if source == "rms":
        return ("rms", "rms_linear")
    if source == "rms_db":
        return ("rms_db", "rms_dbfs", "energy_db")
    if source == "zcr":
        return ("zcr", "zero_crossing_rate")
    if source == "spectral_centroid_hz":
        return (
            "spectral_centroid_hz",
            "spectral_centroid",
            "centroid_hz",
            "brightness_hz",
        )
    if source == "spectral_flatness":
        return ("spectral_flatness", "flatness")
    if source == "spectral_flux":
        return ("spectral_flux", "flux")
    if source == "onset_strength":
        return ("onset_strength", "onset")
    if source == "rolloff_hz":
        return ("rolloff_hz", "spectral_rolloff_hz")
    if source == "voicing_prob":
        return ("voicing_prob", "voicing_probability")
    if source == "pitch_stability":
        return ("pitch_stability",)
    if source == "harmonic_ratio":
        return ("harmonic_ratio", "harm_ratio")
    if source == "rms_norm":
        return ("rms_norm", "energy_norm")
    if source == "centroid_norm":
        return ("centroid_norm", "brightness_norm")
    if source == "flux_norm":
        return ("flux_norm",)
    if source == "onset_norm":
        return ("onset_norm",)
    if source == "rolloff_norm":
        return ("rolloff_norm",)
    return (source,)


def _parse_row_float(
    row: dict[str, str],
    *,
    candidates: tuple[str, ...],
    context: str,
) -> float:
    for key in candidates:
        text = row.get(key, "")
        if text is None:
            continue
        value_text = str(text).strip()
        if not value_text:
            continue
        try:
            value = float(value_text)
        except Exception as exc:
            raise ValueError(f"{context}: value for '{key}' must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{context}: value for '{key}' must be finite")
        return float(value)
    raise ValueError(
        f"{context}: missing source column (expected one of {list(candidates)})"
    )


def _read_source_value(row: dict[str, str], source: str, *, row_index: int) -> float:
    if source == "pitch_ratio":
        try:
            return _parse_row_float(
                row,
                candidates=("pitch_ratio", "ratio"),
                context=f"control-map row {row_index}",
            )
        except ValueError:
            try:
                cents = _parse_row_float(
                    row,
                    candidates=("pitch_cents",),
                    context=f"control-map row {row_index}",
                )
                return float(2.0 ** (cents / 1200.0))
            except ValueError:
                semitones = _parse_row_float(
                    row,
                    candidates=("pitch_semitones",),
                    context=f"control-map row {row_index}",
                )
                return float(2.0 ** (semitones / 12.0))

    return _parse_row_float(
        row,
        candidates=_source_column_candidates(source),
        context=f"control-map row {row_index}",
    )


def _eval_route(route: ControlRoute, row: dict[str, str], *, row_index: int) -> float:
    if route.op == "const":
        assert route.value is not None
        return float(route.value)

    assert route.source is not None
    src = _read_source_value(row, route.source, row_index=row_index)

    if route.op == "source":
        out = src
    elif route.op == "inv":
        if abs(src) <= 1e-12:
            raise ValueError(
                f"control-map row {row_index}: inv({route.source}) division by zero"
            )
        out = 1.0 / src
    elif route.op == "mul":
        params = tuple(route.params or ())
        assert len(params) == 1
        out = src * params[0]
    elif route.op == "add":
        params = tuple(route.params or ())
        assert len(params) == 1
        out = src + params[0]
    elif route.op == "affine":
        params = tuple(route.params or ())
        assert len(params) == 2
        out = src * params[0] + params[1]
    elif route.op == "clip":
        params = tuple(route.params or ())
        assert len(params) == 2
        out = min(max(src, params[0]), params[1])
    else:
        assert route.op == "pow"
        assert route.exponent is not None
        out = float(src**route.exponent)
    if not math.isfinite(out):
        raise ValueError(
            f"control-map row {row_index}: route '{route.expression}' produced a non-finite value"
        )
    return float(out)


def apply_control_routes_csv(
    payload: str,
    *,
    routes: list[ControlRoute],
    source_label: str,
) -> str:
    if not routes:
        return payload

    reader = csv.DictReader(io.StringIO(payload))
    fields_raw = list(reader.fieldnames or [])
    if not fields_raw:
        raise ValueError(f"{source_label}: CSV is empty")

    fields_norm = [
        normalize_control_name(name) for name in fields_raw if name is not None
    ]
    if "start_sec" not in fields_norm or "end_sec" not in fields_norm:
        raise ValueError(
            f"{source_label}: CSV must include start_sec and end_sec columns"
        )

    output_fields = list(fields_norm)
    for route in routes:
        if route.target not in output_fields:
            output_fields.append(route.target)

    rows_out: list[dict[str, str]] = []
    for row_index, row in enumerate(reader, start=2):
        norm: dict[str, str] = {}
        for key, raw in row.items():
            if key is None:
                continue
            norm[normalize_control_name(key)] = "" if raw is None else str(raw).strip()

        for route in routes:
            mapped_value = _eval_route(route, norm, row_index=row_index)
            norm[route.target] = f"{mapped_value:.9f}"
        rows_out.append(norm)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=output_fields, lineterminator="\n")
    writer.writeheader()
    for row in rows_out:
        writer.writerow({name: row.get(name, "") for name in output_fields})
    return buffer.getvalue()
