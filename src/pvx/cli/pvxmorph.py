#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Spectral morphing between two input files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

from pvx.core.audio_metrics import (
    render_audio_comparison_table,
    render_audio_metrics_table,
    summarize_audio_metrics,
)
from pvx.core.common import (
    add_console_args,
    add_output_policy_args,
    add_vocoder_args,
    build_examples_epilog,
    build_status_bar,
    build_vocoder_config,
    ensure_runtime,
    log_error,
    log_message,
    read_audio,
    validate_vocoder_args,
    write_output,
)
from pvx.core.voc import (
    add_mastering_args,
    apply_mastering_chain,
    cepstral_envelope,
    istft,
    resample_1d,
    stft,
)

BLEND_MODE_CHOICES: tuple[str, ...] = (
    "linear",
    "geometric",
    "magnitude_b_phase_a",
    "magnitude_a_phase_b",
    "carrier_a_envelope_b",
    "carrier_b_envelope_a",
    "carrier_a_mask_b",
    "carrier_b_mask_a",
    "product",
    "max_mag",
    "min_mag",
)
CONTROL_INTERP_CHOICES: tuple[str, ...] = (
    "none",
    "linear",
    "nearest",
    "cubic",
    "polynomial",
)

EPS = 1e-12


@dataclass(frozen=True)
class MorphControlSignal:
    times_sec: np.ndarray
    values: np.ndarray
    interpolation: str
    order: int


def _normalize_control_points(
    points: list[tuple[float, float]], *, context: str
) -> tuple[np.ndarray, np.ndarray]:
    if not points:
        raise ValueError(f"{context}: control file has no points")
    points_sorted = sorted((float(t), float(v)) for t, v in points)
    times: list[float] = []
    values: list[float] = []
    for t, v in points_sorted:
        if times and abs(t - times[-1]) <= 1e-12:
            values[-1] = v
        else:
            times.append(t)
            values.append(v)
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64)


def _parse_csv_control_points(
    path: Path, *, context: str
) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{context}: CSV is missing a header row")
        field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
        points: list[tuple[float, float]] = []
        if {"time_sec", "value"}.issubset(field_map):
            t_key = field_map["time_sec"]
            v_key = field_map["value"]
            for row in reader:
                points.append((float(row.get(t_key, 0.0)), float(row.get(v_key, 0.0))))
            return _normalize_control_points(points, context=context)
        if {"start_sec", "end_sec", "value"}.issubset(field_map):
            s_key = field_map["start_sec"]
            e_key = field_map["end_sec"]
            v_key = field_map["value"]
            for row in reader:
                start = float(row.get(s_key, 0.0))
                end = float(row.get(e_key, start))
                value = float(row.get(v_key, 0.0))
                if end < start:
                    start, end = end, start
                points.append((start, value))
                points.append((end, value))
            return _normalize_control_points(points, context=context)
    raise ValueError(
        f"{context}: CSV must contain either time_sec,value or start_sec,end_sec,value"
    )


def _parse_json_control_points(
    path: Path, *, context: str
) -> tuple[np.ndarray, np.ndarray, str | None, int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    interp_override: str | None = None
    order_override: int | None = None
    points: list[tuple[float, float]] = []
    if isinstance(payload, dict):
        raw_interp = payload.get("interp", payload.get("interpolation"))
        if raw_interp is not None:
            interp_override = str(raw_interp).strip().lower()
        raw_order = payload.get("order")
        if raw_order is not None:
            order_override = int(raw_order)
        if isinstance(payload.get("points"), list):
            for point in payload["points"]:
                if not isinstance(point, dict):
                    continue
                t = point.get("time_sec", point.get("time"))
                v = point.get("value")
                if t is None or v is None:
                    continue
                points.append((float(t), float(v)))
        elif isinstance(payload.get("segments"), list):
            for segment in payload["segments"]:
                if not isinstance(segment, dict):
                    continue
                start = segment.get("start_sec", segment.get("start"))
                end = segment.get("end_sec", segment.get("end"))
                value = segment.get("value")
                if start is None or end is None or value is None:
                    continue
                start_f = float(start)
                end_f = float(end)
                if end_f < start_f:
                    start_f, end_f = end_f, start_f
                val_f = float(value)
                points.append((start_f, val_f))
                points.append((end_f, val_f))
    elif isinstance(payload, list):
        for point in payload:
            if not isinstance(point, dict):
                continue
            t = point.get("time_sec", point.get("time"))
            v = point.get("value")
            if t is None or v is None:
                continue
            points.append((float(t), float(v)))
    else:
        raise ValueError(f"{context}: JSON must be an object or array")
    times, values = _normalize_control_points(points, context=context)
    return times, values, interp_override, order_override


def _load_control_signal(
    path: Path,
    *,
    interpolation: str,
    order: int,
    context: str,
) -> MorphControlSignal:
    suffix = path.suffix.lower()
    interp = str(interpolation).strip().lower()
    fit_order = int(order)
    if suffix == ".csv":
        times, values = _parse_csv_control_points(path, context=context)
    elif suffix == ".json":
        times, values, interp_override, order_override = _parse_json_control_points(
            path, context=context
        )
        if interp_override is not None:
            interp = interp_override
        if order_override is not None:
            fit_order = int(order_override)
    else:
        raise ValueError(f"{context}: unsupported control file type '{path.suffix}'")
    if interp not in CONTROL_INTERP_CHOICES:
        raise ValueError(
            f"{context}: interpolation must be one of {', '.join(CONTROL_INTERP_CHOICES)}"
        )
    if fit_order < 1:
        raise ValueError(f"{context}: polynomial order must be >= 1")
    return MorphControlSignal(
        times_sec=np.asarray(times, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        interpolation=interp,
        order=fit_order,
    )


def _parse_scalar_or_control(
    raw_value: str | None,
    *,
    interpolation: str,
    order: int,
    context: str,
    allow_none: bool = False,
) -> tuple[float | None, MorphControlSignal | None, str | None]:
    text = "" if raw_value is None else str(raw_value).strip()
    if text == "":
        if allow_none:
            return None, None, None
        raise ValueError(f"{context}: missing value")
    path = Path(text)
    if path.suffix.lower() in {".csv", ".json"}:
        if not path.exists():
            raise ValueError(f"{context}: control file not found: {path}")
        signal = _load_control_signal(
            path, interpolation=interpolation, order=order, context=context
        )
        return None, signal, str(path)
    try:
        return float(text), None, None
    except ValueError as exc:  # pragma: no cover - parser-level guard
        raise ValueError(
            f"{context}: expected numeric value or .csv/.json control file"
        ) from exc


def _sample_cubic_local(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    out = np.empty_like(query, dtype=np.float64)
    for idx, t in enumerate(query):
        right = int(np.searchsorted(x, t, side="left"))
        left = max(0, right - 2)
        right = min(int(x.size), left + 4)
        left = max(0, right - 4)
        xs = x[left:right]
        ys = y[left:right]
        if xs.size < 2:
            out[idx] = float(y[0])
            continue
        deg = min(3, int(xs.size) - 1)
        if deg <= 1:
            out[idx] = float(np.interp(float(t), x, y))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(xs, ys, deg=deg)
        out[idx] = float(np.polyval(coeffs, float(t)))
    return out


def _sample_control_signal(
    signal: MorphControlSignal, query_sec: np.ndarray
) -> np.ndarray:
    x = np.asarray(signal.times_sec, dtype=np.float64)
    y = np.asarray(signal.values, dtype=np.float64)
    q = np.asarray(query_sec, dtype=np.float64)
    if x.size <= 1:
        fill = float(y[0]) if y.size else 0.0
        return np.full(q.shape, fill, dtype=np.float64)
    mode = str(signal.interpolation).strip().lower()
    if mode == "none":
        idx = np.searchsorted(x, q, side="right") - 1
        idx = np.clip(idx, 0, x.size - 1)
        return y[idx]
    if mode == "nearest":
        idx_r = np.searchsorted(x, q, side="left")
        idx_l = np.clip(idx_r - 1, 0, x.size - 1)
        idx_r = np.clip(idx_r, 0, x.size - 1)
        dl = np.abs(q - x[idx_l])
        dr = np.abs(x[idx_r] - q)
        use_left = dl <= dr
        idx = np.where(use_left, idx_l, idx_r)
        return y[idx]
    if mode == "linear":
        return np.interp(q, x, y, left=float(y[0]), right=float(y[-1]))
    if mode == "cubic":
        return _sample_cubic_local(x, y, q)
    if mode == "polynomial":
        degree = min(max(1, int(signal.order)), int(x.size) - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coeffs = np.polyfit(x, y, deg=degree)
        return np.polyval(coeffs, q)
    raise ValueError(f"Unsupported interpolation mode: {signal.interpolation}")


def match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[1] == channels:
        return audio
    if audio.shape[1] > channels:
        return audio[:, :channels]
    reps = channels - audio.shape[1]
    extra = np.repeat(audio[:, -1:], reps, axis=1)
    return np.hstack([audio, extra])


def _phase_blend(
    phase_a: np.ndarray, phase_b: np.ndarray, mix: float | np.ndarray
) -> np.ndarray:
    if np.isscalar(mix):
        m = float(np.clip(float(mix), 0.0, 1.0))
    else:
        m = np.asarray(mix, dtype=np.float64)
        if m.ndim != 1:
            raise ValueError("phase mix curve must be one-dimensional")
        if m.size != phase_a.shape[1]:
            raise ValueError("phase mix curve length must match number of STFT frames")
        m = np.clip(m, 0.0, 1.0)[np.newaxis, :]
    z = (1.0 - m) * np.exp(1j * phase_a) + m * np.exp(1j * phase_b)
    out = np.angle(z)
    out[np.abs(z) <= 1e-8] = phase_a[np.abs(z) <= 1e-8]
    return out


def _resolve_phase_mix_curve(
    mode: str,
    alpha_curve: np.ndarray,
    *,
    user_mix: float | None,
    user_mix_signal: MorphControlSignal | None,
    frame_times: np.ndarray,
) -> np.ndarray:
    if user_mix_signal is not None:
        values = _sample_control_signal(user_mix_signal, frame_times)
        return np.clip(values, 0.0, 1.0)
    if user_mix is not None:
        return np.full(
            frame_times.shape, float(np.clip(user_mix, 0.0, 1.0)), dtype=np.float64
        )
    if mode in {"magnitude_b_phase_a", "carrier_a_envelope_b", "carrier_a_mask_b"}:
        return np.zeros(frame_times.shape, dtype=np.float64)
    if mode in {"magnitude_a_phase_b", "carrier_b_envelope_a", "carrier_b_mask_a"}:
        return np.ones(frame_times.shape, dtype=np.float64)
    return np.clip(np.asarray(alpha_curve, dtype=np.float64), 0.0, 1.0)


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _framewise_envelope(mag: np.ndarray, lifter: int) -> np.ndarray:
    env = np.zeros_like(mag, dtype=np.float64)
    for i in range(mag.shape[1]):
        env[:, i] = np.asarray(cepstral_envelope(mag[:, i], lifter), dtype=np.float64)
    return np.maximum(env, EPS)


def _mask_from_modulator(mag_mod: np.ndarray, exponent: float) -> np.ndarray:
    frame_mean = np.mean(mag_mod, axis=0, keepdims=True)
    base = mag_mod / np.maximum(frame_mean, EPS)
    return np.power(np.maximum(base, EPS), float(exponent))


def morph_pair(
    a: np.ndarray,
    b: np.ndarray,
    sr: int,
    config,
    alpha: float | None,
    alpha_signal: MorphControlSignal | None,
    *,
    blend_mode: str,
    phase_mix: float | None,
    phase_mix_signal: MorphControlSignal | None,
    mask_exponent: float,
    envelope_lifter: int,
    normalize_energy: bool,
) -> np.ndarray:
    max_len = max(a.shape[0], b.shape[0])
    channels = max(a.shape[1], b.shape[1])
    a2 = np.zeros((max_len, channels), dtype=np.float64)
    b2 = np.zeros((max_len, channels), dtype=np.float64)
    a_adj = match_channels(a, channels)
    b_adj = match_channels(b, channels)
    a2[: a_adj.shape[0], :] = a_adj
    b2[: b_adj.shape[0], :] = b_adj

    out = np.zeros_like(a2)
    mode = str(blend_mode).strip().lower()
    alpha_curve_cache: np.ndarray | None = None
    phase_curve_cache: np.ndarray | None = None
    alpha_scalar = 0.5 if alpha is None else float(alpha)
    for ch in range(channels):
        sa = stft(a2[:, ch], config)
        sb = stft(b2[:, ch], config)
        n_bins = max(sa.shape[0], sb.shape[0])
        n_frames = max(sa.shape[1], sb.shape[1])
        sa2 = np.zeros((n_bins, n_frames), dtype=np.complex128)
        sb2 = np.zeros((n_bins, n_frames), dtype=np.complex128)
        sa2[: sa.shape[0], : sa.shape[1]] = sa
        sb2[: sb.shape[0], : sb.shape[1]] = sb

        mag_a = np.abs(sa2)
        mag_b = np.abs(sb2)
        phase_a = np.angle(sa2)
        phase_b = np.angle(sb2)

        if alpha_curve_cache is None or alpha_curve_cache.size != n_frames:
            frame_times = (
                np.arange(n_frames, dtype=np.float64) * float(config.hop_size)
            ) / float(sr)
            if alpha_signal is not None:
                alpha_curve = _sample_control_signal(alpha_signal, frame_times)
            else:
                alpha_curve = np.full(frame_times.shape, alpha_scalar, dtype=np.float64)
            alpha_curve = np.clip(alpha_curve, 0.0, 1.0)
            phase_curve = _resolve_phase_mix_curve(
                mode,
                alpha_curve,
                user_mix=phase_mix,
                user_mix_signal=phase_mix_signal,
                frame_times=frame_times,
            )
            alpha_curve_cache = alpha_curve
            phase_curve_cache = phase_curve

        assert alpha_curve_cache is not None  # for type checkers
        assert phase_curve_cache is not None
        alpha_mix = alpha_curve_cache[np.newaxis, :]
        linear_mag = (1.0 - alpha_mix) * mag_a + alpha_mix * mag_b

        if mode == "linear":
            out_mag = linear_mag
        elif mode == "geometric":
            out_mag = np.exp(
                (1.0 - alpha_mix) * np.log(mag_a + EPS)
                + alpha_mix * np.log(mag_b + EPS)
            )
        elif mode == "magnitude_b_phase_a":
            out_mag = linear_mag
        elif mode == "magnitude_a_phase_b":
            out_mag = linear_mag
        elif mode == "carrier_a_envelope_b":
            env_a = _framewise_envelope(mag_a, envelope_lifter)
            env_b = _framewise_envelope(mag_b, envelope_lifter)
            target = (mag_a / env_a) * env_b
            out_mag = (1.0 - alpha_mix) * mag_a + alpha_mix * target
        elif mode == "carrier_b_envelope_a":
            env_a = _framewise_envelope(mag_a, envelope_lifter)
            env_b = _framewise_envelope(mag_b, envelope_lifter)
            target = (mag_b / env_b) * env_a
            out_mag = (1.0 - alpha_mix) * mag_b + alpha_mix * target
        elif mode == "carrier_a_mask_b":
            target = mag_a * _mask_from_modulator(mag_b, mask_exponent)
            out_mag = (1.0 - alpha_mix) * mag_a + alpha_mix * target
        elif mode == "carrier_b_mask_a":
            target = mag_b * _mask_from_modulator(mag_a, mask_exponent)
            out_mag = (1.0 - alpha_mix) * mag_b + alpha_mix * target
        elif mode == "product":
            target = np.sqrt(np.maximum(mag_a * mag_b, 0.0))
            out_mag = (1.0 - alpha_mix) * mag_a + alpha_mix * target
        elif mode == "max_mag":
            target = np.maximum(mag_a, mag_b)
            out_mag = (1.0 - alpha_mix) * linear_mag + alpha_mix * target
        elif mode == "min_mag":
            target = np.minimum(mag_a, mag_b)
            out_mag = (1.0 - alpha_mix) * linear_mag + alpha_mix * target
        else:
            raise ValueError(f"Unsupported blend mode: {blend_mode}")

        out_phase = _phase_blend(phase_a, phase_b, phase_curve_cache)
        sm = np.maximum(out_mag, 0.0) * np.exp(1j * out_phase)
        rendered = istft(sm, config, expected_length=max_len)

        if normalize_energy:
            alpha_mean = float(np.mean(alpha_curve_cache))
            target_rms = (1.0 - alpha_mean) * _safe_rms(
                a2[:, ch]
            ) + alpha_mean * _safe_rms(b2[:, ch])
            cur_rms = _safe_rms(rendered)
            if target_rms > 0.0 and cur_rms > EPS:
                rendered = rendered * (target_rms / cur_rms)
        out[:, ch] = rendered
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Morph two audio files in the STFT domain",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=build_examples_epilog(
            [
                "pvx morph source_a.wav source_b.wav --alpha 0.35 --output morph_35.wav",
                "pvx morph A.wav B.wav --alpha controls/alpha_curve.csv --interp linear --output morph_traj.wav",
                "pvx morph vocal.wav strings.wav --alpha 0.6 --n-fft 4096 --hop-size 512 --output vocal_strings.wav",
                "pvx morph voice.wav synth.wav --blend-mode carrier_a_envelope_b --alpha 0.8 --envelope-lifter 40 --output voice_synth_env.wav",
                "pvx morph drums.wav guitar.wav --blend-mode carrier_a_mask_b --alpha 0.7 --mask-exponent 1.2 --output drums_guitar_mask.wav",
                "pvx morph a.wav b.wav --alpha 0.5 --stdout | pvx freeze - --freeze-time 0.8 --duration 12 --output morph_freeze.wav",
            ],
            notes=[
                "--alpha 0.0 keeps input A, --alpha 1.0 keeps input B.",
                "--alpha also accepts CSV/JSON control files for true A->B trajectory morphing over time.",
                "Try --blend-mode carrier_a_envelope_b or carrier_a_mask_b for stronger cross-synthesis character.",
                "Only one morph input can be '-' (stdin).",
            ],
        ),
    )
    parser.add_argument("input_a", type=Path, help="Input A path or '-' for stdin")
    parser.add_argument("input_b", type=Path, help="Input B path or '-' for stdin")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output file path"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write processed audio to stdout stream (for piping); equivalent to -o -",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help="Output extension/format; for --stdout defaults to wav",
    )
    parser.add_argument(
        "--alpha",
        type=str,
        default="0.5",
        help=(
            "Morph amount 0..1 (0=A, 1=B). Accepts scalar or control file (.csv/.json) "
            "for time-varying A->B trajectory morphing."
        ),
    )
    parser.add_argument(
        "--blend-mode",
        choices=list(BLEND_MODE_CHOICES),
        default="linear",
        help=(
            "Cross-synthesis blend style. "
            "linear/geometric are symmetric blends; carrier_* modes transfer envelope/mask from modulator to carrier."
        ),
    )
    parser.add_argument(
        "--phase-mix",
        type=str,
        default=None,
        help=(
            "Phase blend in [0,1]. If omitted, mode-specific defaults apply "
            "(A-phase for *_phase_a/carrier_a_*, B-phase for *_phase_b/carrier_b_*, alpha for symmetric modes). "
            "Accepts scalar or control file (.csv/.json)."
        ),
    )
    parser.add_argument(
        "--interp",
        choices=list(CONTROL_INTERP_CHOICES),
        default="linear",
        help="Interpolation mode for --alpha/--phase-mix control files (default: linear).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=3,
        help=(
            "Polynomial order for --interp polynomial (default: 3). "
            "Accepts any integer >= 1; effective fit degree is min(order, control_points-1)."
        ),
    )
    parser.add_argument(
        "--mask-exponent",
        type=float,
        default=1.0,
        help="Exponent used by carrier_*_mask_* modes (default: 1.0).",
    )
    parser.add_argument(
        "--envelope-lifter",
        type=int,
        default=32,
        help="Cepstral lifter cutoff for carrier_*_envelope_* modes (default: 32).",
    )
    parser.add_argument(
        "--normalize-energy",
        action="store_true",
        help="Normalize each output channel RMS toward alpha-blended input RMS.",
    )
    add_mastering_args(parser)
    add_output_policy_args(parser)
    parser.add_argument("--overwrite", action="store_true")
    add_console_args(parser)
    add_vocoder_args(
        parser, default_n_fft=2048, default_win_length=2048, default_hop_size=512
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ensure_runtime(args, parser)
    validate_vocoder_args(args, parser)
    if int(args.order) < 1:
        parser.error("--order must be >= 1")
    try:
        alpha_scalar, alpha_signal, alpha_source = _parse_scalar_or_control(
            args.alpha,
            interpolation=str(args.interp),
            order=int(args.order),
            context="--alpha",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if alpha_scalar is not None and not (0.0 <= float(alpha_scalar) <= 1.0):
        parser.error("--alpha must be between 0 and 1")
    try:
        phase_mix_scalar, phase_mix_signal, phase_mix_source = _parse_scalar_or_control(
            args.phase_mix,
            interpolation=str(args.interp),
            order=int(args.order),
            context="--phase-mix",
            allow_none=True,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if phase_mix_scalar is not None and not (0.0 <= float(phase_mix_scalar) <= 1.0):
        parser.error("--phase-mix must be between 0 and 1")
    if args.mask_exponent <= 0.0:
        parser.error("--mask-exponent must be > 0")
    if args.envelope_lifter < 0:
        parser.error("--envelope-lifter must be >= 0")
    if str(args.input_a) == "-" and str(args.input_b) == "-":
        parser.error("At most one morph input may be '-' (stdin)")
    if args.output is not None and str(args.output) == "-":
        args.stdout = True
    if args.stdout and args.output is not None and str(args.output) != "-":
        parser.error(
            "Use either --stdout (or -o -) or an explicit output file path, not both"
        )

    config = build_vocoder_config(args, phase_locking="off", transient_preserve=False)
    status = build_status_bar(args, "pvxmorph", 1)
    if alpha_signal is not None:
        log_message(
            args,
            (
                f"[info] alpha trajectory control={alpha_source} "
                f"(interp={alpha_signal.interpolation}, order={alpha_signal.order}, points={alpha_signal.times_sec.size})"
            ),
            min_level="verbose",
        )
    if phase_mix_signal is not None:
        log_message(
            args,
            (
                f"[info] phase-mix trajectory control={phase_mix_source} "
                f"(interp={phase_mix_signal.interpolation}, order={phase_mix_signal.order}, points={phase_mix_signal.times_sec.size})"
            ),
            min_level="verbose",
        )
    try:
        a, sr_a = read_audio(args.input_a)
        b, sr_b = read_audio(args.input_b)
        if sr_b != sr_a:
            target = max(1, int(round(b.shape[0] * sr_a / sr_b)))
            rb = np.zeros((target, b.shape[1]), dtype=np.float64)
            for ch in range(b.shape[1]):
                rb[:, ch] = resample_1d(b[:, ch], target, "auto")
            b = rb

        out = morph_pair(
            a,
            b,
            sr_a,
            config,
            alpha_scalar,
            alpha_signal,
            blend_mode=args.blend_mode,
            phase_mix=phase_mix_scalar,
            phase_mix_signal=phase_mix_signal,
            mask_exponent=args.mask_exponent,
            envelope_lifter=int(args.envelope_lifter),
            normalize_energy=bool(args.normalize_energy),
        )
        out = apply_mastering_chain(out, sr_a, args)

        if args.stdout:
            out_path = Path("-")
        elif args.output is not None:
            out_path = args.output
        else:
            base = Path("stdin.wav") if str(args.input_a) == "-" else args.input_a
            out_path = base.with_name(f"{base.stem}_morph.wav")
        metrics_table = render_audio_metrics_table(
            [
                (f"inA:{args.input_a}", summarize_audio_metrics(a, int(sr_a))),
                (f"inB:{args.input_b}", summarize_audio_metrics(b, int(sr_a))),
                (f"out:{out_path}", summarize_audio_metrics(out, int(sr_a))),
            ],
            title="Audio Metrics",
            include_delta_from_first=True,
        )
        compare_a = render_audio_comparison_table(
            reference_label=f"inA:{args.input_a}",
            reference_audio=a,
            reference_sr=int(sr_a),
            candidate_label=f"out:{out_path}",
            candidate_audio=out,
            candidate_sr=int(sr_a),
            title="Audio Compare Metrics (out vs inA)",
        )
        compare_b = render_audio_comparison_table(
            reference_label=f"inB:{args.input_b}",
            reference_audio=b,
            reference_sr=int(sr_a),
            candidate_label=f"out:{out_path}",
            candidate_audio=out,
            candidate_sr=int(sr_a),
            title="Audio Compare Metrics (out vs inB)",
        )
        log_message(
            args, f"{metrics_table}\n{compare_a}\n{compare_b}", min_level="quiet"
        )
        write_output(
            out_path,
            out,
            sr_a,
            args,
            metadata_extra={
                "morph": {
                    "input_a": str(args.input_a),
                    "input_b": str(args.input_b),
                    "alpha": None if alpha_scalar is None else float(alpha_scalar),
                    "alpha_control": alpha_source,
                    "blend_mode": str(args.blend_mode),
                    "phase_mix": None
                    if phase_mix_scalar is None
                    else float(phase_mix_scalar),
                    "phase_mix_control": phase_mix_source,
                    "control_interp": str(args.interp),
                    "control_order": int(args.order),
                    "mask_exponent": float(args.mask_exponent),
                    "envelope_lifter": int(args.envelope_lifter),
                    "normalize_energy": bool(args.normalize_energy),
                }
            },
        )
        log_message(
            args,
            f"[ok] {args.input_a} + {args.input_b} -> {out_path}",
            min_level="verbose",
        )
        status.step(1, out_path.name)
        status.finish("done")
        log_message(args, "[done] pvxmorph processed=1 failed=0", min_level="normal")
        return 0
    except Exception as exc:
        log_error(args, f"[error] {args.input_a} + {args.input_b}: {exc}")
        status.step(1, "error")
        status.finish("errors=1")
        log_message(args, "[done] pvxmorph processed=1 failed=1", min_level="normal")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
