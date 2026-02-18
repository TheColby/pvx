"""Shared audio metric summaries and ASCII table rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AudioMetricSummary:
    sample_rate: int
    channels: int
    samples: int
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    crest_db: float
    dc_offset: float
    zcr: float
    clip_percent: float
    spectral_centroid_hz: float
    bandwidth_95_hz: float


def _to_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.mean(arr, axis=1)


def _to_2d(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    if arr.size == 0:
        return np.zeros((0, 1), dtype=np.float64)
    return np.asarray(arr.reshape(arr.shape[0], -1), dtype=np.float64)


def _resample_1d_linear(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0 or src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
        return x
    n_out = max(1, int(round(x.size * float(dst_sr) / float(src_sr))))
    x_old = np.linspace(0.0, 1.0, x.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, x).astype(np.float64)


def _resample_audio_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = _to_2d(audio)
    if arr.size == 0 or src_sr == dst_sr:
        return arr
    channels = arr.shape[1]
    cols = [_resample_1d_linear(arr[:, ch], src_sr, dst_sr) for ch in range(channels)]
    n = min(col.size for col in cols) if cols else 0
    if n <= 0:
        return np.zeros((0, max(1, channels)), dtype=np.float64)
    out = np.zeros((n, channels), dtype=np.float64)
    for ch, col in enumerate(cols):
        out[:, ch] = col[:n]
    return out


def _match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _principal_angle(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _stft_complex(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    signal = np.asarray(x, dtype=np.float64).reshape(-1)
    if signal.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.complex128)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size), mode="constant")
    rem = (signal.size - n_fft) % hop
    if rem:
        signal = np.pad(signal, (0, hop - rem), mode="constant")
    frames = 1 + (signal.size - n_fft) // hop
    win = np.hanning(n_fft).astype(np.float64)
    out = np.empty((n_fft // 2 + 1, frames), dtype=np.complex128)
    for idx in range(frames):
        start = idx * hop
        out[:, idx] = np.fft.rfft(signal[start : start + n_fft] * win, n=n_fft)
    return out


def _stft_mag_db(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    spec = _stft_complex(x, n_fft=n_fft, hop=hop)
    return 20.0 * np.log10(np.abs(spec) + 1e-9)


def _onset_envelope(x: np.ndarray, *, n_fft: int = 1024, hop_size: int = 256) -> np.ndarray:
    if x.size == 0:
        return np.zeros(0, dtype=np.float64)
    spec_db = _stft_mag_db(x, n_fft=n_fft, hop=hop_size)
    mag = np.power(10.0, spec_db / 20.0)
    if mag.shape[1] <= 1:
        return np.zeros(mag.shape[1], dtype=np.float64)
    delta = np.maximum(0.0, mag[:, 1:] - mag[:, :-1])
    flux = np.sqrt(np.sum(delta * delta, axis=0))
    flux = np.concatenate([np.zeros(1, dtype=np.float64), flux], axis=0)
    peak = float(np.max(flux)) if flux.size else 0.0
    if peak > 1e-12:
        flux = flux / peak
    return flux


def _dbfs(amplitude: float) -> float:
    return float(20.0 * np.log10(max(float(amplitude), 1e-12)))


def _spectral_centroid_and_bw95(signal: np.ndarray, sr: int) -> tuple[float, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0 or sr <= 0:
        return 0.0, 0.0
    n_fft = int(2 ** np.ceil(np.log2(max(256, min(8192, x.size)))))
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size), mode="constant")
    win = np.hanning(n_fft).astype(np.float64)
    spec = np.fft.rfft(x[:n_fft] * win, n=n_fft)
    mag = np.abs(spec).astype(np.float64)
    if not np.any(mag > 0.0):
        return 0.0, 0.0
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    energy = mag * mag
    total = float(np.sum(energy)) + 1e-18
    centroid = float(np.sum(freqs * energy) / total)
    cdf = np.cumsum(energy) / total
    idx95 = int(np.searchsorted(cdf, 0.95))
    idx95 = max(0, min(idx95, freqs.size - 1))
    bw95 = float(freqs[idx95])
    return centroid, bw95


def summarize_audio_metrics(audio: np.ndarray, sample_rate: int) -> AudioMetricSummary:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]

    samples = int(arr.shape[0])
    channels = int(arr.shape[1]) if arr.ndim == 2 else 1
    sr = int(sample_rate)
    duration_s = float(samples / max(1, sr))

    if samples == 0 or channels == 0:
        return AudioMetricSummary(
            sample_rate=sr,
            channels=max(1, channels),
            samples=samples,
            duration_s=duration_s,
            peak_dbfs=-240.0,
            rms_dbfs=-240.0,
            crest_db=0.0,
            dc_offset=0.0,
            zcr=0.0,
            clip_percent=0.0,
            spectral_centroid_hz=0.0,
            bandwidth_95_hz=0.0,
        )

    mono = _to_mono(arr)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
    crest = float(_dbfs(peak / max(rms, 1e-12))) if peak > 0.0 else 0.0
    dc_offset = float(np.mean(mono)) if mono.size else 0.0
    if mono.size > 1:
        signs = np.signbit(mono)
        zcr = float(np.mean(signs[:-1] != signs[1:]))
    else:
        zcr = 0.0
    clip_percent = float(100.0 * np.mean(np.abs(mono) >= 0.999)) if mono.size else 0.0
    centroid_hz, bw95_hz = _spectral_centroid_and_bw95(mono, sr)

    return AudioMetricSummary(
        sample_rate=sr,
        channels=channels,
        samples=samples,
        duration_s=duration_s,
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        crest_db=crest,
        dc_offset=dc_offset,
        zcr=zcr,
        clip_percent=clip_percent,
        spectral_centroid_hz=centroid_hz,
        bandwidth_95_hz=bw95_hz,
    )


def _format_float(value: float, precision: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def render_audio_metrics_table(
    rows: Iterable[tuple[str, AudioMetricSummary]],
    *,
    title: str = "Audio Metrics",
    include_delta_from_first: bool = True,
) -> str:
    entries = list(rows)
    if not entries:
        return f"{title}\n(no audio rows)"

    headers = [
        "label",
        "sr",
        "ch",
        "samples",
        "dur_s",
        "peak_dbfs",
        "rms_dbfs",
        "crest_db",
        "dc",
        "zcr",
        "clip_pct",
        "centroid_hz",
        "bw95_hz",
    ]

    body: list[list[str]] = []
    for label, m in entries:
        body.append(
            [
                str(label),
                str(int(m.sample_rate)),
                str(int(m.channels)),
                str(int(m.samples)),
                _format_float(m.duration_s, 3),
                _format_float(m.peak_dbfs, 2),
                _format_float(m.rms_dbfs, 2),
                _format_float(m.crest_db, 2),
                _format_float(m.dc_offset, 6),
                _format_float(m.zcr, 4),
                _format_float(m.clip_percent, 3),
                _format_float(m.spectral_centroid_hz, 1),
                _format_float(m.bandwidth_95_hz, 1),
            ]
        )

    if include_delta_from_first and len(entries) >= 2:
        ref = entries[0][1]
        last = entries[-1][1]
        body.append(
            [
                "delta(last-first)",
                "-",
                "-",
                str(int(last.samples - ref.samples)),
                _format_float(last.duration_s - ref.duration_s, 3),
                _format_float(last.peak_dbfs - ref.peak_dbfs, 2),
                _format_float(last.rms_dbfs - ref.rms_dbfs, 2),
                _format_float(last.crest_db - ref.crest_db, 2),
                _format_float(last.dc_offset - ref.dc_offset, 6),
                _format_float(last.zcr - ref.zcr, 4),
                _format_float(last.clip_percent - ref.clip_percent, 3),
                _format_float(last.spectral_centroid_hz - ref.spectral_centroid_hz, 1),
                _format_float(last.bandwidth_95_hz - ref.bandwidth_95_hz, 1),
            ]
        )

    return f"{title}\n" + _ascii_table(headers, body)


def summarize_audio_comparison_metrics(
    reference_audio: np.ndarray,
    reference_sr: int,
    candidate_audio: np.ndarray,
    candidate_sr: int,
) -> dict[str, float]:
    ref_2d = _to_2d(reference_audio)
    cand_2d = _to_2d(candidate_audio)
    ref_2d = _resample_audio_linear(ref_2d, int(reference_sr), int(reference_sr))
    cand_2d = _resample_audio_linear(cand_2d, int(candidate_sr), int(reference_sr))
    ref_2d, cand_2d = _match_length(ref_2d, cand_2d)

    if ref_2d.size == 0 or cand_2d.size == 0:
        return {key: float("nan") for key in (
            "snr_input", "snr_output",
            "si_sdr_input", "si_sdr_output",
            "lsd_input", "lsd_output",
            "modspec_input", "modspec_output",
            "spectral_convergence_input", "spectral_convergence_output",
            "envelope_corr_input", "envelope_corr_output",
            "transient_smear_input", "transient_smear_output",
            "rms_dbfs_input", "rms_dbfs_output",
            "crest_db_input", "crest_db_output",
            "bw95_hz_input", "bw95_hz_output",
            "zcr_input", "zcr_output",
            "dc_input", "dc_output",
            "clip_pct_input", "clip_pct_output",
            "lufs_input", "lufs_output",
            "true_peak_dbtp_input", "true_peak_dbtp_output",
            "stereo_phase_drift_input", "stereo_phase_drift_output",
            "ild_drift_db_input", "ild_drift_db_output",
            "itd_drift_ms_input", "itd_drift_ms_output",
        )}

    ref = _to_mono(ref_2d)
    cand = _to_mono(cand_2d)
    ref, cand = _match_length(ref, cand)
    eps = 1e-12

    def _snr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
        reference, degraded = _match_length(reference, degraded)
        if reference.size == 0:
            return float("nan")
        err = reference - degraded
        p_sig = float(np.mean(reference * reference)) + eps
        p_err = float(np.mean(err * err)) + eps
        return float(10.0 * np.log10(p_sig / p_err))

    def _si_sdr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
        reference, degraded = _match_length(reference, degraded)
        if reference.size == 0:
            return float("nan")
        denom = float(np.dot(reference, reference)) + eps
        alpha = float(np.dot(degraded, reference) / denom)
        target = alpha * reference
        noise = degraded - target
        return float(10.0 * np.log10((float(np.dot(target, target)) + eps) / (float(np.dot(noise, noise)) + eps)))

    snr_input = _snr_db(ref, ref)
    snr_output = _snr_db(ref, cand)
    si_sdr_input = _si_sdr_db(ref, ref)
    si_sdr_output = _si_sdr_db(ref, cand)

    n_fft = 2048
    hop = 512
    ref_db = _stft_mag_db(ref, n_fft=n_fft, hop=hop)
    cand_db = _stft_mag_db(cand, n_fft=n_fft, hop=hop)
    bins = min(ref_db.shape[0], cand_db.shape[0])
    frames = min(ref_db.shape[1], cand_db.shape[1])
    if bins == 0 or frames == 0:
        lsd_output = float("nan")
        modspec_output = float("nan")
        spectral_convergence_output = float("nan")
    else:
        diff = ref_db[:bins, :frames] - cand_db[:bins, :frames]
        lsd_output = float(np.sqrt(np.mean(diff * diff)))

        ref_mod = np.abs(np.fft.rfft(ref_db[:bins, :frames], axis=1))
        cand_mod = np.abs(np.fft.rfft(cand_db[:bins, :frames], axis=1))
        m_bins = min(ref_mod.shape[1], cand_mod.shape[1])
        if m_bins > 0:
            m_diff = ref_mod[:, :m_bins] - cand_mod[:, :m_bins]
            modspec_output = float(np.sqrt(np.mean(m_diff * m_diff)) / (np.mean(np.abs(ref_mod[:, :m_bins])) + eps))
        else:
            modspec_output = float("nan")

        ref_mag = np.power(10.0, ref_db[:bins, :frames] / 20.0)
        cand_mag = np.power(10.0, cand_db[:bins, :frames] / 20.0)
        spectral_convergence_output = float(np.linalg.norm(ref_mag - cand_mag) / (np.linalg.norm(ref_mag) + eps))

    env_ref = _onset_envelope(ref, n_fft=1024, hop_size=256)
    env_cand = _onset_envelope(cand, n_fft=1024, hop_size=256)
    e_n = min(env_ref.size, env_cand.size)
    if e_n >= 3:
        a = env_ref[:e_n] - float(np.mean(env_ref[:e_n]))
        b = env_cand[:e_n] - float(np.mean(env_cand[:e_n]))
        envelope_corr_output = float(np.dot(a, b) / (float(np.linalg.norm(a) * np.linalg.norm(b)) + eps))
        transient_smear_output = float(np.mean(np.abs(env_ref[:e_n] - env_cand[:e_n])))
    else:
        envelope_corr_output = float("nan")
        transient_smear_output = float("nan")

    summary_ref = summarize_audio_metrics(ref_2d, int(reference_sr))
    summary_cand = summarize_audio_metrics(cand_2d, int(reference_sr))

    def _integrated_lufs(signal: np.ndarray, sample_rate: int) -> float:
        try:
            import pyloudnorm as pyln  # type: ignore

            meter = pyln.Meter(sample_rate)
            return float(meter.integrated_loudness(np.asarray(signal, dtype=np.float64).reshape(-1)))
        except Exception:
            rms = float(np.sqrt(np.mean(signal * signal)) + eps)
            return float(-0.691 + 20.0 * np.log10(rms + eps))

    def _true_peak_dbtp(signal: np.ndarray, sample_rate: int) -> float:
        up = _resample_1d_linear(signal, sample_rate, sample_rate * 4)
        peak = float(np.max(np.abs(up))) if up.size else 0.0
        return float(20.0 * np.log10(max(peak, eps)))

    lufs_input = float(_integrated_lufs(ref, int(reference_sr)))
    lufs_output = float(_integrated_lufs(cand, int(reference_sr)))
    true_peak_input = float(_true_peak_dbtp(ref, int(reference_sr)))
    true_peak_output = float(_true_peak_dbtp(cand, int(reference_sr)))

    stereo_phase_drift_output = float("nan")
    ild_drift_db_output = float("nan")
    itd_drift_ms_output = float("nan")
    if ref_2d.shape[1] >= 2 and cand_2d.shape[1] >= 2:
        ref_lr = ref_2d[:, :2]
        cand_lr = cand_2d[:, :2]
        spec_ref_l = _stft_complex(ref_lr[:, 0], n_fft=1024, hop=256)
        spec_ref_r = _stft_complex(ref_lr[:, 1], n_fft=1024, hop=256)
        spec_cand_l = _stft_complex(cand_lr[:, 0], n_fft=1024, hop=256)
        spec_cand_r = _stft_complex(cand_lr[:, 1], n_fft=1024, hop=256)
        b2 = min(spec_ref_l.shape[0], spec_ref_r.shape[0], spec_cand_l.shape[0], spec_cand_r.shape[0])
        f2 = min(spec_ref_l.shape[1], spec_ref_r.shape[1], spec_cand_l.shape[1], spec_cand_r.shape[1])
        if b2 > 0 and f2 > 0:
            ipd_ref = _principal_angle(np.angle(spec_ref_l[:b2, :f2]) - np.angle(spec_ref_r[:b2, :f2]))
            ipd_cand = _principal_angle(np.angle(spec_cand_l[:b2, :f2]) - np.angle(spec_cand_r[:b2, :f2]))
            stereo_phase_drift_output = float(np.mean(np.abs(_principal_angle(ipd_cand - ipd_ref))))

        l_ref = float(np.sqrt(np.mean(ref_lr[:, 0] * ref_lr[:, 0])) + eps)
        r_ref = float(np.sqrt(np.mean(ref_lr[:, 1] * ref_lr[:, 1])) + eps)
        l_cand = float(np.sqrt(np.mean(cand_lr[:, 0] * cand_lr[:, 0])) + eps)
        r_cand = float(np.sqrt(np.mean(cand_lr[:, 1] * cand_lr[:, 1])) + eps)
        ild_ref = float(20.0 * np.log10(l_ref / r_ref))
        ild_cand = float(20.0 * np.log10(l_cand / r_cand))
        ild_drift_db_output = float(ild_cand - ild_ref)

        max_lag = max(1, int(round(0.002 * int(reference_sr))))

        def _lag(a: np.ndarray, b: np.ndarray) -> int:
            aa = np.asarray(a, dtype=np.float64).reshape(-1)
            bb = np.asarray(b, dtype=np.float64).reshape(-1)
            aa, bb = _match_length(aa, bb)
            n = max(1, aa.size)
            cc = np.correlate(aa, bb, mode="full")
            center = n - 1
            lo = max(0, center - max_lag)
            hi = min(cc.size, center + max_lag + 1)
            local = cc[lo:hi]
            return int(np.argmax(np.abs(local)) + lo - center)

        ref_lag = _lag(ref_lr[:, 0], ref_lr[:, 1])
        cand_lag = _lag(cand_lr[:, 0], cand_lr[:, 1])
        itd_drift_ms_output = float(1000.0 * (cand_lag - ref_lag) / max(1, int(reference_sr)))

    return {
        "snr_input": snr_input,
        "snr_output": snr_output,
        "si_sdr_input": si_sdr_input,
        "si_sdr_output": si_sdr_output,
        "lsd_input": 0.0 if np.isfinite(lsd_output) else float("nan"),
        "lsd_output": lsd_output,
        "modspec_input": 0.0 if np.isfinite(modspec_output) else float("nan"),
        "modspec_output": modspec_output,
        "spectral_convergence_input": 0.0 if np.isfinite(spectral_convergence_output) else float("nan"),
        "spectral_convergence_output": spectral_convergence_output,
        "envelope_corr_input": 1.0 if np.isfinite(envelope_corr_output) else float("nan"),
        "envelope_corr_output": envelope_corr_output,
        "transient_smear_input": 0.0 if np.isfinite(transient_smear_output) else float("nan"),
        "transient_smear_output": transient_smear_output,
        "rms_dbfs_input": float(summary_ref.rms_dbfs),
        "rms_dbfs_output": float(summary_cand.rms_dbfs),
        "crest_db_input": float(summary_ref.crest_db),
        "crest_db_output": float(summary_cand.crest_db),
        "bw95_hz_input": float(summary_ref.bandwidth_95_hz),
        "bw95_hz_output": float(summary_cand.bandwidth_95_hz),
        "zcr_input": float(summary_ref.zcr),
        "zcr_output": float(summary_cand.zcr),
        "dc_input": float(summary_ref.dc_offset),
        "dc_output": float(summary_cand.dc_offset),
        "clip_pct_input": float(summary_ref.clip_percent),
        "clip_pct_output": float(summary_cand.clip_percent),
        "lufs_input": lufs_input,
        "lufs_output": lufs_output,
        "true_peak_dbtp_input": true_peak_input,
        "true_peak_dbtp_output": true_peak_output,
        "stereo_phase_drift_input": 0.0 if np.isfinite(stereo_phase_drift_output) else float("nan"),
        "stereo_phase_drift_output": stereo_phase_drift_output,
        "ild_drift_db_input": 0.0 if np.isfinite(ild_drift_db_output) else float("nan"),
        "ild_drift_db_output": ild_drift_db_output,
        "itd_drift_ms_input": 0.0 if np.isfinite(itd_drift_ms_output) else float("nan"),
        "itd_drift_ms_output": itd_drift_ms_output,
    }


def render_audio_comparison_table(
    *,
    reference_label: str,
    reference_audio: np.ndarray,
    reference_sr: int,
    candidate_label: str,
    candidate_audio: np.ndarray,
    candidate_sr: int,
    title: str = "Audio Compare Metrics",
) -> str:
    cmp = summarize_audio_comparison_metrics(
        reference_audio=reference_audio,
        reference_sr=int(reference_sr),
        candidate_audio=candidate_audio,
        candidate_sr=int(candidate_sr),
    )
    headers = ["metric", f"input ({reference_label})", f"output ({candidate_label})", "delta(out-in)", "ideal"]

    def _delta(out_key: str, in_key: str) -> float:
        out_v = float(cmp.get(out_key, float("nan")))
        in_v = float(cmp.get(in_key, float("nan")))
        if not (np.isfinite(out_v) and np.isfinite(in_v)):
            return float("nan")
        return out_v - in_v

    rows = [
        ["SNR dB", _format_float(cmp["snr_input"], 3), _format_float(cmp["snr_output"], 3), _format_float(_delta("snr_output", "snr_input"), 3), "higher"],
        ["SI-SDR dB", _format_float(cmp["si_sdr_input"], 3), _format_float(cmp["si_sdr_output"], 3), _format_float(_delta("si_sdr_output", "si_sdr_input"), 3), "higher"],
        ["LSD", _format_float(cmp["lsd_input"], 4), _format_float(cmp["lsd_output"], 4), _format_float(_delta("lsd_output", "lsd_input"), 4), "lower"],
        ["ModSpec Dist", _format_float(cmp["modspec_input"], 4), _format_float(cmp["modspec_output"], 4), _format_float(_delta("modspec_output", "modspec_input"), 4), "lower"],
        ["Spectral Conv", _format_float(cmp["spectral_convergence_input"], 4), _format_float(cmp["spectral_convergence_output"], 4), _format_float(_delta("spectral_convergence_output", "spectral_convergence_input"), 4), "lower"],
        ["Envelope Corr", _format_float(cmp["envelope_corr_input"], 4), _format_float(cmp["envelope_corr_output"], 4), _format_float(_delta("envelope_corr_output", "envelope_corr_input"), 4), "higher"],
        ["Transient Smear", _format_float(cmp["transient_smear_input"], 4), _format_float(cmp["transient_smear_output"], 4), _format_float(_delta("transient_smear_output", "transient_smear_input"), 4), "lower"],
        ["RMS dBFS", _format_float(cmp["rms_dbfs_input"], 3), _format_float(cmp["rms_dbfs_output"], 3), _format_float(_delta("rms_dbfs_output", "rms_dbfs_input"), 3), "0 ideal"],
        ["Crest dB", _format_float(cmp["crest_db_input"], 3), _format_float(cmp["crest_db_output"], 3), _format_float(_delta("crest_db_output", "crest_db_input"), 3), "0 ideal"],
        ["BW95 Hz", _format_float(cmp["bw95_hz_input"], 2), _format_float(cmp["bw95_hz_output"], 2), _format_float(_delta("bw95_hz_output", "bw95_hz_input"), 2), "0 ideal"],
        ["ZCR", _format_float(cmp["zcr_input"], 6), _format_float(cmp["zcr_output"], 6), _format_float(_delta("zcr_output", "zcr_input"), 6), "0 ideal"],
        ["DC", _format_float(cmp["dc_input"], 8), _format_float(cmp["dc_output"], 8), _format_float(_delta("dc_output", "dc_input"), 8), "0 ideal"],
        ["Clip %", _format_float(cmp["clip_pct_input"], 3), _format_float(cmp["clip_pct_output"], 3), _format_float(_delta("clip_pct_output", "clip_pct_input"), 3), "0 ideal"],
        ["LUFS", _format_float(cmp["lufs_input"], 3), _format_float(cmp["lufs_output"], 3), _format_float(_delta("lufs_output", "lufs_input"), 3), "0 ideal"],
        ["TruePeak dBTP", _format_float(cmp["true_peak_dbtp_input"], 3), _format_float(cmp["true_peak_dbtp_output"], 3), _format_float(_delta("true_peak_dbtp_output", "true_peak_dbtp_input"), 3), "0 ideal"],
        ["Stereo Phase Drift", _format_float(cmp["stereo_phase_drift_input"], 4), _format_float(cmp["stereo_phase_drift_output"], 4), _format_float(_delta("stereo_phase_drift_output", "stereo_phase_drift_input"), 4), "lower"],
        ["ILD Drift dB", _format_float(cmp["ild_drift_db_input"], 4), _format_float(cmp["ild_drift_db_output"], 4), _format_float(_delta("ild_drift_db_output", "ild_drift_db_input"), 4), "0 ideal"],
        ["ITD Drift ms", _format_float(cmp["itd_drift_ms_input"], 4), _format_float(cmp["itd_drift_ms_output"], 4), _format_float(_delta("itd_drift_ms_output", "itd_drift_ms_input"), 4), "0 ideal"],
    ]
    return f"{title}\n" + _ascii_table(headers, rows)
