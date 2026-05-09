#!/usr/bin/env python3
"""Speech enhancement training data generation using pvx.

Generates paired (clean, noisy) audio for training speech enhancement,
dereverberation, and noise suppression models.

For each clean utterance the script produces N degraded versions with
different noise/reverb conditions.  The output JSONL manifest contains
the clean path, noisy path, and all augmentation parameters — enabling
fully reproducible dataset generation and ablation studies.

Usage
-----
    python examples/speech_enhancement_pairs.py \
        --clean-dir data/clean_speech \
        --output-dir data/enhancement_pairs \
        --pairs 5 \
        --workers 8 \
        --seed 1337
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_degradation_pipeline(condition: str, seed: int = 42):
    """Build a degradation pipeline for one of several noise conditions."""
    from pvx.augment import (
        AddNoise,
        CodecDegradation,
        GainPerturber,
        ImpulseNoise,
        Pipeline,
        RoomSimulator,
    )

    conditions = {
        "clean_noise": Pipeline([
            GainPerturber(gain_db=(-3, 3), p=1.0),
            AddNoise(snr_db=(15, 35), noise_type="white", p=1.0),
        ], seed=seed),

        "reverb_only": Pipeline([
            GainPerturber(gain_db=(-3, 3), p=1.0),
            RoomSimulator(rt60_range=(0.3, 2.0), wet_range=(0.4, 0.95), p=1.0),
        ], seed=seed),

        "reverb_noise": Pipeline([
            GainPerturber(gain_db=(-3, 3), p=1.0),
            RoomSimulator(rt60_range=(0.1, 1.5), wet_range=(0.2, 0.8), p=1.0),
            AddNoise(snr_db=(5, 25), noise_type="pink", p=1.0),
        ], seed=seed),

        "telephone": Pipeline([
            GainPerturber(gain_db=(-3, 3), p=1.0),
            CodecDegradation(codec="telephone", p=1.0),
            AddNoise(snr_db=(5, 20), noise_type="white", p=0.7),
        ], seed=seed),

        "voip": Pipeline([
            GainPerturber(gain_db=(-3, 3), p=1.0),
            CodecDegradation(codec="voip_narrow", p=1.0),
            AddNoise(snr_db=(10, 30), noise_type="pink", p=0.5),
            ImpulseNoise(rate=0.5, amplitude_range=(0.01, 0.05), p=0.2),
        ], seed=seed),
    }
    return conditions.get(condition, conditions["reverb_noise"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate speech enhancement training pairs")
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=5,
                        help="Number of noisy versions per clean file (default: 5)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args(argv)

    from pvx.augment.core import load_audio, save_audio

    exts = {".wav", ".flac", ".aiff"}
    clean_files = sorted(p for p in args.clean_dir.rglob("*") if p.suffix.lower() in exts)
    if not clean_files:
        print(f"No audio files in {args.clean_dir}", file=sys.stderr)
        return 1

    conditions = ["clean_noise", "reverb_only", "reverb_noise", "telephone", "voip"]
    n_conditions = len(conditions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_out = args.output_dir / "clean"
    noisy_out = args.output_dir / "noisy"
    clean_out.mkdir(exist_ok=True)
    noisy_out.mkdir(exist_ok=True)

    manifest = []
    total = len(clean_files) * args.pairs
    print(f"Generating {total} pairs from {len(clean_files)} clean files …")

    for src_idx, src_path in enumerate(clean_files):
        try:
            audio, sr = load_audio(src_path, target_sr=args.sample_rate, mono=True)
        except Exception as exc:
            print(f"  Skip {src_path.name}: {exc}", file=sys.stderr)
            continue

        # Write clean reference once
        clean_copy = clean_out / src_path.name
        save_audio(clean_copy, audio, sr)

        for pair_idx in range(args.pairs):
            condition = conditions[pair_idx % n_conditions]
            item_seed = args.seed + src_idx * 10_000 + pair_idx
            pipeline = build_degradation_pipeline(condition, seed=item_seed)
            noisy_audio, _ = pipeline(audio, sr, seed=item_seed)

            noisy_name = f"{src_path.stem}__cond{condition}__pair{pair_idx:03d}.wav"
            noisy_path = noisy_out / noisy_name
            save_audio(noisy_path, noisy_audio, sr)

            manifest.append({
                "clean": str(clean_copy),
                "noisy": str(noisy_path),
                "condition": condition,
                "pair_idx": pair_idx,
                "seed": item_seed,
                "sr": sr,
            })

        if (src_idx + 1) % 50 == 0:
            print(f"  {src_idx + 1}/{len(clean_files)} files …")

    manifest_path = args.output_dir / "pairs_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    print(f"\nDone. {len(manifest)} pairs written.")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
