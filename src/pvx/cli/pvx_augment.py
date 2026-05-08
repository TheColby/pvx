"""Augmentation helpers and subcommands for the unified pvx CLI."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import hashlib
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from pvx.cli.catalog import _AUDIO_EXTENSIONS

__all__ = [
    "_augment_group_key",
    "_parse_split_ratios",
    "_stable_seed_from_text",
    "run_augment_manifest_mode",
    "run_augment_mode",
    "run_batch_gpu_mode",
]

AUGMENT_LIMITER_THRESHOLD = "0.98"


def _expand_augment_inputs(tokens: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for token in tokens:
        raw = str(token).strip()
        if not raw:
            continue
        if any(ch in raw for ch in "*?["):
            candidates = [Path(p) for p in sorted(glob.glob(raw, recursive=True))]
        else:
            candidates = [Path(raw)]
        for candidate in candidates:
            p = candidate.expanduser().resolve()
            if p.is_dir():
                for child in sorted(p.rglob("*")):
                    if child.is_file() and child.suffix.lower() in _AUDIO_EXTENSIONS:
                        key = str(child.resolve())
                        if key not in seen:
                            seen.add(key)
                            resolved.append(child.resolve())
                continue
            if p.is_file() and p.suffix.lower() in _AUDIO_EXTENSIONS:
                key = str(p)
                if key not in seen:
                    seen.add(key)
                    resolved.append(p)
    return resolved


def _parse_split_ratios(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError("--split must be three comma-separated ratios: train,val,test")
    train, val, test = (float(parts[0]), float(parts[1]), float(parts[2]))
    if train < 0.0 or val < 0.0 or test < 0.0:
        raise ValueError("--split ratios must be >= 0")
    total = train + val + test
    if total <= 0.0:
        raise ValueError("--split ratio sum must be > 0")
    return train / total, val / total, test / total


def _pick_split(rng: random.Random, ratios: tuple[float, float, float]) -> str:
    train, val, _ = ratios
    x = rng.random()
    if x < train:
        return "train"
    if x < train + val:
        return "val"
    return "test"


def _stable_seed_from_text(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int((int(base_seed) ^ value) & 0x7FFFFFFF)


def _augment_group_key(path: Path, grouping: str, separator: str) -> str:
    if str(grouping) == "none":
        return str(path.resolve())
    stem = str(path.stem)
    sep = str(separator)
    if sep:
        return stem.split(sep)[0]
    return stem


def _token_flag(token: str) -> str:
    return str(token).split("=", 1)[0]


def _flag_present(argv: list[str], names: tuple[str, ...]) -> bool:
    for token in argv:
        raw = str(token)
        flag = _token_flag(raw)
        if flag in names:
            return True
    return False


def _load_augment_policy(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Augmentation policy must be a JSON object.")
    return dict(payload)


def _load_label_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"Label metadata CSV not found: {p}")
    out: dict[str, dict[str, str]] = {}
    with p.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Label metadata CSV has no header: {p}")
        for row in reader:
            source = str(
                row.get("source_path")
                or row.get("path")
                or row.get("file")
                or row.get("filename")
                or ""
            ).strip()
            if not source:
                continue
            label = str(row.get("label") or "").strip()
            speaker = str(row.get("speaker") or row.get("speaker_id") or "").strip()
            meta = {"label": label, "speaker": speaker}
            source_path = Path(source).expanduser()
            if source_path.is_absolute():
                key_abs = str(source_path.resolve())
                out[key_abs] = meta
            else:
                out[source] = meta
            out[source_path.name] = meta
    return out


def _source_metadata(source: Path, metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    abs_key = str(source.resolve())
    name_key = source.name
    empty = {"label": "", "speaker": ""}
    if abs_key in metadata:
        return dict(metadata[abs_key])
    if name_key in metadata:
        return dict(metadata[name_key])
    return empty


def _assign_balanced_split_for_groups(
    group_keys: list[str],
    *,
    ratios: tuple[float, float, float],
    base_seed: int,
    split_mode: str,
    group_meta: dict[str, dict[str, str]],
) -> dict[str, str]:
    train, val, test = ratios
    if split_mode == "random":
        out: dict[str, str] = {}
        for group_key in group_keys:
            rng = random.Random(_stable_seed_from_text(base_seed, f"split::{group_key}"))
            out[group_key] = _pick_split(rng, (train, val, test))
        return out

    token_key = "label" if split_mode == "label_balanced" else "speaker"
    split_names = ("train", "val", "test")
    ratios_map = {"train": train, "val": val, "test": test}
    out: dict[str, str] = {}
    total_groups = max(1, len(group_keys))
    global_counts = {"train": 0, "val": 0, "test": 0}

    buckets: dict[str, list[str]] = {}
    fallback: list[str] = []
    for group_key in group_keys:
        token = str(group_meta.get(group_key, {}).get(token_key, "")).strip()
        if token:
            buckets.setdefault(token, []).append(group_key)
        else:
            fallback.append(group_key)

    def _assign_bucket(keys: list[str], seed_tag: str) -> None:
        ordered = sorted(keys)
        rng = random.Random(_stable_seed_from_text(base_seed, seed_tag))
        rng.shuffle(ordered)
        for group_key in ordered:
            best_split = split_names[0]
            best_need = -10.0
            for split_name in split_names:
                target = ratios_map[split_name] * float(total_groups)
                need = target - float(global_counts[split_name])
                if need > best_need:
                    best_need = need
                    best_split = split_name
            out[group_key] = best_split
            global_counts[best_split] += 1

    for token, keys in sorted(buckets.items(), key=lambda item: item[0]):
        _assign_bucket(keys, f"{split_mode}::{token}")
    if fallback:
        _assign_bucket(fallback, f"{split_mode}::fallback")
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _audio_audit_metrics(path: Path) -> dict[str, float]:
    audio, sr = sf.read(str(path), always_2d=True)
    arr = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
    peak_dbfs = float(20.0 * np.log10(max(peak, 1e-12)))
    rms_dbfs = float(20.0 * np.log10(max(rms, 1e-12)))
    clip_pct = float(100.0 * np.mean(np.abs(arr) >= 0.999999)) if arr.size else 0.0
    mono = arr.mean(axis=1) if arr.ndim == 2 and arr.shape[1] > 0 else arr.reshape(-1)
    if mono.size > 1:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(mono).astype(np.int8)))))
    else:
        zcr = 0.0
    return {
        "sample_rate_hz": float(sr),
        "channels": float(arr.shape[1] if arr.ndim == 2 else 1),
        "samples": float(arr.shape[0] if arr.ndim == 2 else arr.size),
        "duration_sec": float(arr.shape[0] / float(sr)) if arr.ndim == 2 and sr > 0 else 0.0,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "clip_pct": clip_pct,
        "zcr": zcr,
    }


def _manifest_required_errors(record: dict[str, object]) -> list[str]:
    required = ("source_path", "output_path", "intent", "seed", "split", "group_key", "status")
    errors: list[str] = []
    for key in required:
        value = record.get(key)
        if value in {None, ""}:
            errors.append(f"missing required field '{key}'")
    params = record.get("params")
    if params is not None and not isinstance(params, dict):
        errors.append("field 'params' must be an object when present")
    return errors


def _load_manifest_jsonl(path: Path) -> list[dict[str, object]]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{p}:{line_no}: manifest row must be a JSON object")
        rows.append(dict(payload))
    return rows


def _merge_manifest_rows(
    rows: list[dict[str, object]], *, dedupe_by: str = "output_path"
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row.get(dedupe_by, "")).strip()
        if not key:
            key = json.dumps(row, sort_keys=True)
        merged[key] = dict(row)
    return [merged[key] for key in sorted(merged.keys())]


def _sample_augment_params(
    intent: str,
    rng: random.Random,
    *,
    label_policy: str = "allow_alter",
    policy_overrides: dict[str, object] | None = None,
) -> dict[str, str]:
    key = str(intent).strip().lower()
    if key == "asr_robust":
        stretch = rng.uniform(0.92, 1.12)
        pitch = rng.uniform(-1.5, 1.5)
        formant = rng.uniform(0.55, 0.95)
        transient = rng.uniform(0.50, 0.75)
        preset = "vocal_studio"
        window = rng.choice(["hann", "hamming", "blackmanharris"])
    elif key == "mir_music":
        stretch = rng.uniform(0.85, 1.25)
        pitch = rng.uniform(-3.0, 3.0)
        formant = rng.uniform(0.30, 0.70)
        transient = rng.uniform(0.45, 0.70)
        preset = "stereo_coherent"
        window = rng.choice(["hann", "blackmanharris", "kaiser"])
    else:
        stretch = rng.uniform(0.70, 1.45)
        pitch = rng.uniform(-4.5, 4.5)
        formant = rng.uniform(0.20, 0.80)
        transient = rng.uniform(0.40, 0.85)
        preset = rng.choice(["vocal_studio", "drums_safe", "stereo_coherent"])
        window = rng.choice(["hann", "hamming", "blackmanharris", "kaiser", "tukey"])

    if str(label_policy).strip().lower() == "preserve":
        stretch = float(np.clip(stretch, 0.95, 1.05))
        pitch = float(np.clip(pitch, -1.0, 1.0))
        formant = float(np.clip(formant, 0.60, 0.95))

    transform = rng.choice(["fft", "dft"])
    target_lufs = rng.uniform(-24.0, -14.0)
    choices = (
        dict((policy_overrides or {}).get("choices", {}))
        if isinstance(policy_overrides, dict)
        else {}
    )
    bounds = (
        dict((policy_overrides or {}).get("bounds", {}))
        if isinstance(policy_overrides, dict)
        else {}
    )

    def _override_bounds(name: str, value: float) -> float:
        item = bounds.get(name)
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lo = float(item[0])
            hi = float(item[1])
            if lo > hi:
                lo, hi = hi, lo
            return float(np.clip(value, lo, hi))
        return value

    def _override_choice(name: str, default_value: str) -> str:
        item = choices.get(name)
        if isinstance(item, list) and item:
            return str(rng.choice([str(v) for v in item]))
        return default_value

    stretch = _override_bounds("stretch", float(stretch))
    pitch = _override_bounds("pitch", float(pitch))
    formant = _override_bounds("formant_strength", float(formant))
    transient = _override_bounds("transient_sensitivity", float(transient))
    target_lufs = _override_bounds("target_lufs", float(target_lufs))
    window = _override_choice("window", window)
    transform = _override_choice("transform", transform)
    preset = _override_choice("preset", preset)

    return {
        "stretch": f"{stretch:.6f}",
        "pitch": f"{pitch:.6f}",
        "formant_strength": f"{formant:.6f}",
        "transient_sensitivity": f"{transient:.6f}",
        "preset": preset,
        "window": window,
        "transform": transform,
        "target_lufs": f"{target_lufs:.6f}",
        "phase_locking": "identity",
        "transient_mode": "hybrid",
        "stereo_mode": "mid_side_lock",
        "coherence_strength": "0.85",
    }


def _render_job_pytorch(
    record: dict[str, object],
    params: dict[str, object],
    src: Path,
    out_path: Path,
    engine_name: str = "pytorch",
) -> dict[str, object]:
    """Render an augmentation job using a Python-native engine."""
    from pvx.augment.time_domain import PitchShift, TimeStretch

    audio, sr = sf.read(str(src), always_2d=False, dtype="float32")

    stretch = float(params.get("stretch", 1.0))
    pitch_semitones = float(params.get("pitch", 0.0))

    if abs(stretch - 1.0) > 1e-6:
        ts = TimeStretch(rate=(stretch, stretch), preserve_pitch=True, engine=engine_name, p=1.0)
        audio, sr = ts(audio, sr)

    if abs(pitch_semitones) > 1e-6:
        ps = PitchShift(semitones=(pitch_semitones, pitch_semitones), engine=engine_name, p=1.0)
        audio, sr = ps(audio, sr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sr)
    record["status"] = "rendered"
    record["engine_used"] = engine_name
    return record


def _augment_output_conflict(out_path: Path, *, overwrite: bool) -> str | None:
    if overwrite or not out_path.exists():
        return None
    return f"output exists:{out_path}"


def run_batch_gpu_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx batch-gpu",
        description=(
            "Batch GPU augmentation for many files at once.\n"
            "Maximizes GPU throughput by stacking files into (B, C, T) tensors\n"
            "and running them through a TorchPipeline simultaneously."
        ),
    )
    parser.add_argument(
        "inputs", nargs="+", help="Input audio files, directories, and/or glob patterns"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Output directory for augmented files"
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=None,
        help=("Optional YAML/JSON pipeline manifest (see pvx.augment.config.load_pipeline)"),
    )
    parser.add_argument(
        "--intent",
        choices=["asr_robust", "mir_music", "ssl_contrastive"],
        default="asr_robust",
        help=(
            "Built-in augmentation intent profile when --pipeline-config is not given (default: asr_robust)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of files processed simultaneously on GPU (default: 16)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto — CUDA > MPS > CPU)",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument(
        "--max-length-s",
        type=float,
        default=30.0,
        help="Maximum file duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--output-suffix", default="_aug", help="Suffix appended before extension (default: _aug)"
    )
    parser.add_argument(
        "--output-subtype",
        default="PCM_16",
        choices=["PCM_16", "PCM_24", "PCM_32", "FLOAT"],
        help="Output sample format (default: PCM_16)",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Deterministic base seed (default: 1337)"
    )
    parser.add_argument("--no-progress", action="store_true", help="Suppress progress messages")
    args = parser.parse_args(forwarded_args)

    file_paths: list[str] = []
    for token in args.inputs:
        p = Path(token)
        if p.is_dir():
            for ext in ("wav", "flac", "ogg", "mp3"):
                file_paths.extend(str(x) for x in sorted(p.rglob(f"*.{ext}")))
        elif any(ch in token for ch in "*?["):
            file_paths.extend(sorted(glob.glob(token, recursive=True)))
        elif p.is_file():
            file_paths.append(str(p))

    if not file_paths:
        parser.error("no input audio files found")

    try:
        from pvx.augment.gpu import (
            TorchAddNoise,
            TorchGainPerturber,
            TorchPipeline,
            TorchSpecAugment,
            batch_process_files,
        )
    except ImportError as exc:
        parser.error(
            f"PyTorch is required for `pvx batch-gpu`. Install with: pip install 'pvx[torch]'. ({exc})"
        )

    if args.pipeline_config is not None:
        from pvx.augment.config import load_torch_pipeline

        pipeline = load_torch_pipeline(args.pipeline_config)
    elif args.intent == "asr_robust":
        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-6.0, 6.0), p=0.8),
                TorchAddNoise(snr_db=(15.0, 35.0), noise_type="pink", p=0.6),
                TorchSpecAugment(freq_mask_param=20, time_mask_param=30, p=0.5),
            ]
        )
    elif args.intent == "mir_music":
        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-3.0, 3.0), p=0.7),
                TorchAddNoise(snr_db=(25.0, 45.0), noise_type="white", p=0.3),
            ]
        )
    else:
        pipeline = TorchPipeline(
            [
                TorchGainPerturber(gain_db=(-9.0, 9.0), p=1.0),
                TorchAddNoise(snr_db=(5.0, 30.0), noise_type="pink", p=0.9),
                TorchSpecAugment(freq_mask_param=27, time_mask_param=100, p=0.8),
            ]
        )

    results = batch_process_files(
        file_paths,
        pipeline,
        sr=args.sr,
        output_dir=str(args.output_dir),
        output_suffix=args.output_suffix,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        max_length_s=args.max_length_s,
        output_subtype=args.output_subtype,
        progress=not args.no_progress,
    )

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = len(results) - ok
    if failed:
        print(f"[batch-gpu] {failed} file(s) failed:")
        for r in results:
            if r["status"] != "ok":
                print(f"  {r['input']}: {r['status']}")
        return 1
    print(f"[batch-gpu] OK — {ok} file(s) processed")
    return 0


def run_augment_mode(
    forwarded_args: list[str],
    *,
    dispatch_tool: Callable[[str, list[str]], int],
) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx augment",
        description=(
            "Deterministic dataset augmentation for AI research/data augmentation.\n"
            "Generates variants with reproducible random seeds and writes manifests."
        ),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Optional JSON policy file with augmentation defaults and bounds/choices overrides",
    )
    parser.add_argument(
        "inputs", nargs="+", help="Input audio files, directories, and/or glob patterns"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Output directory for augmented files"
    )
    parser.add_argument(
        "--variants-per-input",
        type=int,
        default=3,
        help="Number of augmented outputs per input file (default: 3)",
    )
    parser.add_argument(
        "--intent",
        choices=["asr_robust", "mir_music", "ssl_contrastive"],
        default="asr_robust",
        help="Augmentation intent profile (default: asr_robust)",
    )
    parser.add_argument(
        "--label-policy",
        choices=["allow_alter", "preserve"],
        default="allow_alter",
        help="Label perturbation policy (default: allow_alter)",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Deterministic random seed (default: 1337)"
    )
    parser.add_argument(
        "--split", default="0.8,0.1,0.1", help="train,val,test split ratios (default: 0.8,0.1,0.1)"
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "label_balanced", "speaker_balanced"],
        default="random",
        help="Split assignment mode (default: random)",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional metadata CSV with path/label/speaker columns for balanced split modes",
    )
    parser.add_argument(
        "--grouping",
        choices=["none", "stem-prefix"],
        default="stem-prefix",
        help=(
            "Split-grouping strategy (default: stem-prefix). `stem-prefix` keeps variants from similarly named sources in one split."
        ),
    )
    parser.add_argument(
        "--group-separator",
        default="__",
        help="Separator used by stem-prefix grouping (default: '__')",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["off", "contrastive2"],
        default="off",
        help="Pair-view mode: off or two-view contrastive output (default: off)",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL manifest path (default: <output-dir>/augment_manifest.jsonl)",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Optional CSV manifest path (default: <output-dir>/augment_manifest.csv)",
    )
    parser.add_argument(
        "--output-format",
        choices=["wav", "flac", "aiff", "ogg", "caf"],
        default="wav",
        help="Output format container (default: wav)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel worker count for rendering (default: 1)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-rendered outputs found in existing manifest/output-dir",
    )
    parser.add_argument(
        "--append-manifest",
        action="store_true",
        help="Append/merge with existing manifest instead of replacing it",
    )
    parser.add_argument(
        "--strict-manifest",
        action="store_true",
        help="Return non-zero if manifest validation errors are found",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan manifest and filenames without rendering audio"
    )
    parser.add_argument(
        "--audit-metrics",
        action="store_true",
        default=True,
        help="Compute per-output audit metrics (default: on)",
    )
    parser.add_argument(
        "--no-audit-metrics",
        dest="audit_metrics",
        action="store_false",
        help="Disable audit metric computation",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto", help="Processing device"
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "pytorch", "torchaudio", "pvx-cli"],
        default="auto",
        help=(
            "DSP engine for time-stretch and pitch-shift transforms. 'auto' prefers torchaudio > pytorch > pvx-cli. 'torchaudio' uses torchaudio.functional.phase_vocoder. 'pytorch' uses native PyTorch phase vocoder. 'pvx-cli' always uses subprocess. (default: auto)"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce logs")
    parser.add_argument("--silent", action="store_true", help="Suppress logs")
    args = parser.parse_args(forwarded_args)

    tokens = list(forwarded_args or [])
    policy_cfg: dict[str, object] = {}
    if args.policy is not None:
        policy_path = Path(args.policy).expanduser().resolve()
        if not policy_path.exists():
            parser.error(f"Augmentation policy not found: {policy_path}")
        try:
            raw_policy = _load_augment_policy(policy_path)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"Failed to load --policy {policy_path}: {exc}")
        policy_cfg = dict(
            raw_policy.get("augment", raw_policy) if isinstance(raw_policy, dict) else {}
        )

        if not _flag_present(tokens, ("--intent",)):
            args.intent = str(policy_cfg.get("intent", args.intent))
        if not _flag_present(tokens, ("--variants-per-input",)):
            args.variants_per_input = int(
                policy_cfg.get("variants_per_input", args.variants_per_input)
            )
        if not _flag_present(tokens, ("--seed",)):
            args.seed = int(policy_cfg.get("seed", args.seed))
        if not _flag_present(tokens, ("--split",)):
            args.split = str(policy_cfg.get("split", args.split))
        if not _flag_present(tokens, ("--split-mode",)):
            args.split_mode = str(policy_cfg.get("split_mode", args.split_mode))
        if not _flag_present(tokens, ("--grouping",)):
            args.grouping = str(policy_cfg.get("grouping", args.grouping))
        if not _flag_present(tokens, ("--group-separator",)):
            args.group_separator = str(policy_cfg.get("group_separator", args.group_separator))
        if not _flag_present(tokens, ("--pair-mode",)):
            args.pair_mode = str(policy_cfg.get("pair_mode", args.pair_mode))
        if not _flag_present(tokens, ("--label-policy",)):
            args.label_policy = str(policy_cfg.get("label_policy", args.label_policy))
        if not _flag_present(tokens, ("--workers",)):
            args.workers = int(policy_cfg.get("workers", args.workers))
        if not _flag_present(tokens, ("--device",)):
            args.device = str(policy_cfg.get("device", args.device))
        if not _flag_present(tokens, ("--engine",)):
            args.engine = str(policy_cfg.get("engine", args.engine))
        if not _flag_present(tokens, ("--output-format",)):
            args.output_format = str(policy_cfg.get("output_format", args.output_format))

    if int(args.variants_per_input) <= 0:
        parser.error("--variants-per-input must be > 0")
    if int(args.workers) <= 0:
        parser.error("--workers must be > 0")

    try:
        split_ratios = _parse_split_ratios(str(args.split))
    except ValueError as exc:
        parser.error(str(exc))

    sources = _expand_augment_inputs(list(args.inputs))
    if not sources:
        parser.error("No input audio files matched provided inputs")

    try:
        label_metadata = _load_label_metadata(args.labels_csv)
    except ValueError as exc:
        parser.error(str(exc))

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl = (
        Path(args.manifest_jsonl).expanduser().resolve()
        if args.manifest_jsonl is not None
        else out_dir / "augment_manifest.jsonl"
    )
    manifest_csv = (
        Path(args.manifest_csv).expanduser().resolve()
        if args.manifest_csv is not None
        else out_dir / "augment_manifest.csv"
    )

    existing_rows: list[dict[str, object]] = []
    if (bool(args.resume) or bool(args.append_manifest)) and manifest_jsonl.exists():
        try:
            existing_rows = _load_manifest_jsonl(manifest_jsonl)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(f"Failed to parse existing manifest {manifest_jsonl}: {exc}")
    existing_by_output: dict[str, dict[str, object]] = {}
    for row in existing_rows:
        key = str(row.get("output_path", "")).strip()
        if key:
            existing_by_output[key] = dict(row)

    group_meta: dict[str, dict[str, str]] = {}
    for src in sources:
        group_key = _augment_group_key(src, str(args.grouping), str(args.group_separator))
        if group_key not in group_meta:
            group_meta[group_key] = _source_metadata(src, label_metadata)
        else:
            current = group_meta[group_key]
            incoming = _source_metadata(src, label_metadata)
            if not str(current.get("label", "")).strip() and str(incoming.get("label", "")).strip():
                current["label"] = str(incoming.get("label", ""))
            if (
                not str(current.get("speaker", "")).strip()
                and str(incoming.get("speaker", "")).strip()
            ):
                current["speaker"] = str(incoming.get("speaker", ""))

    group_keys = sorted(
        {_augment_group_key(src, str(args.grouping), str(args.group_separator)) for src in sources}
    )
    group_to_split = _assign_balanced_split_for_groups(
        group_keys,
        ratios=split_ratios,
        base_seed=int(args.seed),
        split_mode=str(args.split_mode),
        group_meta=group_meta,
    )

    base_seed = int(args.seed)
    source_hash_cache: dict[str, str] = {}
    jobs: list[dict[str, object]] = []
    total_jobs = 0
    for src_idx, src in enumerate(sources):
        group_key = _augment_group_key(src, str(args.grouping), str(args.group_separator))
        src_meta = _source_metadata(src, label_metadata)
        split_name = group_to_split.get(group_key, "train")
        for variant_idx in range(int(args.variants_per_input)):
            sample_seed = base_seed + (src_idx * 10_000) + variant_idx
            views = ("main",) if str(args.pair_mode) == "off" else ("a", "b")
            pair_id = (
                ""
                if str(args.pair_mode) == "off"
                else f"{src.stem}_{variant_idx + 1:03d}_{sample_seed}"
            )
            for view_idx, view_id in enumerate(views):
                view_seed = int(sample_seed + (view_idx * 1_000_000))
                rng = random.Random(view_seed)
                params = _sample_augment_params(
                    str(args.intent),
                    rng,
                    label_policy=str(args.label_policy),
                    policy_overrides=policy_cfg,
                )
                view_suffix = "" if view_id == "main" else f"_view{view_id}"
                out_name = f"{src.stem}__aug_{args.intent!s}_{variant_idx + 1:03d}_{sample_seed}{view_suffix}.{args.output_format!s}"
                out_path = out_dir / out_name
                output_key = str(out_path.resolve())
                if bool(args.resume):
                    old = existing_by_output.get(output_key)
                    if (
                        old is not None
                        and str(old.get("status", "")).startswith("rendered")
                        and out_path.exists()
                    ):
                        continue
                total_jobs += 1
                if str(src.resolve()) not in source_hash_cache:
                    source_hash_cache[str(src.resolve())] = _sha256_file(src)
                jobs.append(
                    {
                        "source_path": str(src.resolve()),
                        "output_path": output_key,
                        "intent": str(args.intent),
                        "seed": int(view_seed),
                        "base_seed": int(sample_seed),
                        "split": str(split_name),
                        "split_mode": str(args.split_mode),
                        "group_key": str(group_key),
                        "label": str(src_meta.get("label", "")),
                        "speaker": str(src_meta.get("speaker", "")),
                        "pair_id": str(pair_id),
                        "view_id": str(view_id),
                        "label_policy": str(args.label_policy),
                        "source_sha256": str(source_hash_cache[str(src.resolve())]),
                        "params": dict(params),
                        "engine": str(args.engine),
                        "status": "planned" if bool(args.dry_run) else "rendering",
                    }
                )

    records: list[dict[str, object]] = []
    failures = 0

    def _render_job(job: dict[str, object], *, idx: int) -> dict[str, object]:
        record = dict(job)
        src = Path(str(record["source_path"]))
        out_path = Path(str(record["output_path"]))
        if not bool(args.silent):
            print(f"[augment] {idx}/{total_jobs} {src.name} -> {out_path.name}")
        if bool(args.dry_run):
            record["status"] = "planned"
            return record

        conflict = _augment_output_conflict(out_path, overwrite=bool(args.overwrite))
        if conflict is not None:
            record["status"] = f"error:{conflict}"
            return record

        params = dict(record.get("params", {}))
        engine_choice = str(args.engine)
        resolved_engine: str | None = None
        if engine_choice in ("pytorch", "torchaudio"):
            resolved_engine = engine_choice
        elif engine_choice == "auto":
            try:
                import torchaudio as _ta  # noqa: F401

                resolved_engine = "torchaudio"
            except ImportError:
                try:
                    import torch as _torch  # noqa: F401

                    resolved_engine = "pytorch"
                except ImportError:
                    resolved_engine = None

        if resolved_engine is not None:
            try:
                record = _render_job_pytorch(
                    record, params, src, out_path, engine_name=resolved_engine
                )
                return record
            except Exception as exc:
                if engine_choice in ("pytorch", "torchaudio"):
                    record["status"] = f"error:{engine_choice}:{exc}"
                    return record
                if not bool(args.silent):
                    print(f"[augment] PyTorch engine failed, falling back to pvx-cli: {exc}")

        voc_args = [
            str(src),
            "--stretch",
            str(params.get("stretch", "1.0")),
            "--pitch",
            str(params.get("pitch", "0.0")),
            "--preset",
            str(params.get("preset", "default")),
            "--window",
            str(params.get("window", "hann")),
            "--transform",
            str(params.get("transform", "fft")),
            "--phase-locking",
            str(params.get("phase_locking", "identity")),
            "--transient-mode",
            str(params.get("transient_mode", "hybrid")),
            "--transient-sensitivity",
            str(params.get("transient_sensitivity", "0.6")),
            "--stereo-mode",
            str(params.get("stereo_mode", "mid_side_lock")),
            "--coherence-strength",
            str(params.get("coherence_strength", "0.85")),
            "--formant-strength",
            str(params.get("formant_strength", "0.7")),
            "--target-lufs",
            str(params.get("target_lufs", "-18.0")),
            "--limiter-threshold",
            AUGMENT_LIMITER_THRESHOLD,
            "--output",
            str(out_path),
            "--output-format",
            str(args.output_format),
            "--device",
            str(args.device),
        ]
        if bool(args.overwrite):
            voc_args.append("--overwrite")
        if bool(args.quiet) or bool(args.silent) or int(args.workers) > 1:
            voc_args.append("--silent")
        code = dispatch_tool("voc", voc_args)
        if int(code) != 0:
            record["status"] = f"error:{code}"
            return record
        record["status"] = "rendered"
        if out_path.exists():
            record["output_sha256"] = _sha256_file(out_path)
            if bool(args.audit_metrics):
                try:
                    record["audit"] = _audio_audit_metrics(out_path)
                except (OSError, ValueError, RuntimeError) as exc:
                    record["audit_error"] = str(exc)
        return record

    if int(args.workers) <= 1 or bool(args.dry_run) or len(jobs) <= 1:
        for idx, job in enumerate(jobs, start=1):
            row = _render_job(job, idx=idx)
            if str(row.get("status", "")).startswith("error:"):
                failures += 1
            records.append(row)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
            futures: list[concurrent.futures.Future[dict[str, object]]] = []
            for idx, job in enumerate(jobs, start=1):
                futures.append(pool.submit(_render_job, job, idx=idx))
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                if str(row.get("status", "")).startswith("error:"):
                    failures += 1
                records.append(row)
        records.sort(
            key=lambda item: (str(item.get("output_path", "")), str(item.get("view_id", "")))
        )

    append_mode = bool(args.append_manifest) or bool(args.resume)
    final_records = (
        _merge_manifest_rows(existing_rows + records, dedupe_by="output_path")
        if append_mode
        else list(records)
    )

    manifest_errors: list[str] = []
    for idx, record in enumerate(final_records, start=1):
        for err in _manifest_required_errors(record):
            manifest_errors.append(f"row {idx}: {err}")
    if manifest_errors and bool(args.strict_manifest):
        for err in manifest_errors[:20]:
            print(f"[augment] manifest error: {err}", file=sys.stderr)
        return 1

    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for record in final_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_path",
        "output_path",
        "intent",
        "seed",
        "base_seed",
        "split",
        "split_mode",
        "group_key",
        "label",
        "speaker",
        "pair_id",
        "view_id",
        "label_policy",
        "status",
        "source_sha256",
        "output_sha256",
        "stretch",
        "pitch",
        "preset",
        "window",
        "transform",
        "formant_strength",
        "transient_sensitivity",
        "target_lufs",
        "audit_duration_sec",
        "audit_peak_dbfs",
        "audit_rms_dbfs",
        "audit_clip_pct",
        "audit_zcr",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in final_records:
            params = dict(record.get("params", {}))
            audit = dict(record.get("audit", {}))
            writer.writerow(
                {
                    "source_path": str(record.get("source_path", "")),
                    "output_path": str(record.get("output_path", "")),
                    "intent": str(record.get("intent", "")),
                    "seed": str(record.get("seed", "")),
                    "base_seed": str(record.get("base_seed", "")),
                    "split": str(record.get("split", "")),
                    "split_mode": str(record.get("split_mode", "")),
                    "group_key": str(record.get("group_key", "")),
                    "label": str(record.get("label", "")),
                    "speaker": str(record.get("speaker", "")),
                    "pair_id": str(record.get("pair_id", "")),
                    "view_id": str(record.get("view_id", "")),
                    "label_policy": str(record.get("label_policy", "")),
                    "status": str(record.get("status", "")),
                    "source_sha256": str(record.get("source_sha256", "")),
                    "output_sha256": str(record.get("output_sha256", "")),
                    "stretch": str(params.get("stretch", "")),
                    "pitch": str(params.get("pitch", "")),
                    "preset": str(params.get("preset", "")),
                    "window": str(params.get("window", "")),
                    "transform": str(params.get("transform", "")),
                    "formant_strength": str(params.get("formant_strength", "")),
                    "transient_sensitivity": str(params.get("transient_sensitivity", "")),
                    "target_lufs": str(params.get("target_lufs", "")),
                    "audit_duration_sec": str(audit.get("duration_sec", "")),
                    "audit_peak_dbfs": str(audit.get("peak_dbfs", "")),
                    "audit_rms_dbfs": str(audit.get("rms_dbfs", "")),
                    "audit_clip_pct": str(audit.get("clip_pct", "")),
                    "audit_zcr": str(audit.get("zcr", "")),
                }
            )

    rendered = sum(
        1 for record in final_records if str(record.get("status", "")).startswith("rendered")
    )
    planned_count = sum(
        1 for record in final_records if str(record.get("status", "")).startswith("planned")
    )
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for record in final_records:
        split = str(record.get("split", "")).strip().lower()
        if split in split_counts:
            split_counts[split] += 1
    if not bool(args.silent):
        print(
            f"[augment] done inputs={len(sources)} variants={len(final_records)} failures={failures}"
        )
        print(f"[augment] manifest jsonl -> {manifest_jsonl}")
        print(f"[augment] manifest csv   -> {manifest_csv}")
        print(
            "[augment] split counts "
            f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
        )
        if manifest_errors:
            print(f"[augment] manifest validation warnings={len(manifest_errors)}")
        if bool(args.dry_run):
            print(f"[augment] dry-run planned variants={planned_count}")
        else:
            print(f"[augment] rendered variants={rendered}")

    return 1 if failures else 0


def run_augment_manifest_mode(forwarded_args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pvx augment-manifest",
        description="Validate, merge, and inspect augmentation manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a manifest JSONL")
    validate_parser.add_argument("manifest", type=Path, help="Manifest JSONL path")
    validate_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    validate_parser.add_argument(
        "--strict", action="store_true", help="Return non-zero on validation errors"
    )

    merge_parser = subparsers.add_parser("merge", help="Merge one or more manifest JSONL files")
    merge_parser.add_argument("manifests", nargs="+", type=Path, help="Manifest JSONL inputs")
    merge_parser.add_argument(
        "--output-jsonl", required=True, type=Path, help="Merged JSONL output path"
    )
    merge_parser.add_argument(
        "--output-csv", type=Path, default=None, help="Optional merged CSV output path"
    )
    merge_parser.add_argument(
        "--dedupe-by", default="output_path", help="Deduplication key (default: output_path)"
    )

    stats_parser = subparsers.add_parser(
        "stats", help="Print quick split/status stats for a manifest"
    )
    stats_parser.add_argument("manifest", type=Path, help="Manifest JSONL path")
    stats_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    args = parser.parse_args(forwarded_args)

    if args.command == "validate":
        rows = _load_manifest_jsonl(Path(args.manifest).expanduser().resolve())
        errors: list[str] = []
        for idx, row in enumerate(rows, start=1):
            for err in _manifest_required_errors(row):
                errors.append(f"row {idx}: {err}")
        payload = {
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "rows": len(rows),
            "errors": errors,
            "valid": len(errors) == 0,
        }
        if bool(args.json):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("pvx augment-manifest validate")
            print(f"- manifest: {payload['manifest']}")
            print(f"- rows: {payload['rows']}")
            print(f"- valid: {'yes' if payload['valid'] else 'no'}")
            if errors:
                print("- errors:")
                for err in errors[:20]:
                    print(f"  - {err}")
        if bool(args.strict) and errors:
            return 1
        return 0

    if args.command == "merge":
        all_rows: list[dict[str, object]] = []
        for item in list(args.manifests):
            all_rows.extend(_load_manifest_jsonl(Path(item).expanduser().resolve()))
        merged = _merge_manifest_rows(all_rows, dedupe_by=str(args.dedupe_by))
        out_jsonl = Path(args.output_jsonl).expanduser().resolve()
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        if args.output_csv is not None:
            out_csv = Path(args.output_csv).expanduser().resolve()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            headers = sorted({key for row in merged for key in row})
            with out_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                for row in merged:
                    writer.writerow({key: row.get(key, "") for key in headers})
        print(f"[augment-manifest] merged rows={len(merged)} -> {out_jsonl}")
        return 0

    if args.command == "stats":
        rows = _load_manifest_jsonl(Path(args.manifest).expanduser().resolve())
        split_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        intent_counts: dict[str, int] = {}
        for row in rows:
            split = str(row.get("split", "unknown")).strip().lower() or "unknown"
            status = str(row.get("status", "unknown")).strip().lower() or "unknown"
            intent = str(row.get("intent", "unknown")).strip().lower() or "unknown"
            split_counts[split] = split_counts.get(split, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        payload = {
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "rows": len(rows),
            "split_counts": split_counts,
            "status_counts": status_counts,
            "intent_counts": intent_counts,
        }
        if bool(args.json):
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("pvx augment-manifest stats")
            print(f"- manifest: {payload['manifest']}")
            print(f"- rows: {payload['rows']}")
            print(f"- split_counts: {json.dumps(split_counts, sort_keys=True)}")
            print(f"- status_counts: {json.dumps(status_counts, sort_keys=True)}")
            print(f"- intent_counts: {json.dumps(intent_counts, sort_keys=True)}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
