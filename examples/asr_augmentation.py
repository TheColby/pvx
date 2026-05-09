#!/usr/bin/env python3
"""ASR data augmentation example using pvx.augment.

Demonstrates how to build a reproducible augmentation pipeline for
automatic speech recognition training, apply it to a directory of WAV
files, and write a JSONL manifest for downstream training scripts.

Usage
-----
    python examples/asr_augmentation.py \
        --input-dir data/train_clean \
        --output-dir data/train_aug \
        --variants 3 \
        --workers 8 \
        --seed 1337
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_asr_pipeline(seed: int = 42):
    """Build a production-quality ASR augmentation pipeline."""
    from pvx.augment import (
        AddNoise,
        CodecDegradation,
        GainPerturber,
        Identity,
        OneOf,
        Pipeline,
        RoomSimulator,
        SpecAugment,
    )

    return Pipeline(
        [
            # Loudness variation
            GainPerturber(gain_db=(-6.0, 6.0), p=0.9),

            # Room acoustics — covers office, meeting room, phone booth
            RoomSimulator(
                rt60_range=(0.05, 1.2),
                wet_range=(0.1, 0.7),
                pre_delay_ms=5.0,
                p=0.5,
            ),

            # Background noise — pink approximates many real environments
            AddNoise(snr_db=(5.0, 35.0), noise_type="pink", p=0.6),

            # Codec degradation — simulate telephone/VoIP channels
            OneOf(
                [
                    CodecDegradation(codec="voip_narrow"),
                    CodecDegradation(codec="telephone"),
                    CodecDegradation(codec="mp3_low"),
                    Identity(),
                ],
                weights=[0.2, 0.2, 0.1, 0.5],
                p=0.35,
            ),

            # SpecAugment — proven WER reduction on LibriSpeech
            SpecAugment(
                freq_mask_param=27,
                time_mask_param=100,
                num_freq_masks=2,
                num_time_masks=2,
                p=0.8,
            ),
        ],
        seed=seed,
    )


def augment_directory(
    input_dir: Path,
    output_dir: Path,
    variants: int,
    workers: int,
    seed: int,
    sample_rate: int,
) -> list[dict]:
    """Augment all WAV files in *input_dir* and return a manifest."""
    import concurrent.futures
    import hashlib

    from pvx.augment.core import load_audio, save_audio

    input_files = sorted(
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in (".wav", ".flac", ".aiff")
    )
    if not input_files:
        print(f"No audio files found in {input_dir}", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = build_asr_pipeline(seed=seed)
    manifest: list[dict] = []

    def _process(args):
        src_idx, src_path, var_idx = args
        item_seed = seed + src_idx * 10_000 + var_idx
        try:
            audio, sr = load_audio(src_path, target_sr=sample_rate, mono=True)
        except Exception as exc:
            return {"source": str(src_path), "status": f"load_error:{exc}"}

        audio_aug, sr_out = pipeline(audio, sr, seed=item_seed)

        stem = src_path.stem
        rel = src_path.relative_to(input_dir).parent
        out_subdir = output_dir / rel
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_path = out_subdir / f"{stem}__aug{var_idx:03d}_seed{item_seed}.wav"

        try:
            save_audio(out_path, audio_aug, sr_out)
        except Exception as exc:
            return {"source": str(src_path), "status": f"write_error:{exc}"}

        # SHA256 fingerprint
        h = hashlib.sha256(audio_aug.tobytes()).hexdigest()[:16]
        return {
            "source": str(src_path),
            "output": str(out_path),
            "variant": var_idx,
            "seed": item_seed,
            "sr": sr_out,
            "sha256_prefix": h,
            "status": "ok",
        }

    jobs = [
        (src_idx, src_path, var_idx)
        for src_idx, src_path in enumerate(input_files)
        for var_idx in range(variants)
    ]

    print(f"Processing {len(input_files)} files x {variants} variants = {len(jobs)} jobs "
          f"using {workers} worker(s) …")

    if workers <= 1:
        for job in jobs:
            manifest.append(_process(job))
            if len(manifest) % 100 == 0:
                print(f"  {len(manifest)}/{len(jobs)}")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(_process, jobs), 1):
                manifest.append(result)
                if i % 100 == 0:
                    print(f"  {i}/{len(jobs)}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASR augmentation example")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    manifest = augment_directory(
        args.input_dir,
        args.output_dir,
        args.variants,
        args.workers,
        args.seed,
        args.sample_rate,
    )

    manifest_path = args.manifest or (args.output_dir / "asr_aug_manifest.jsonl")
    with open(manifest_path, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    ok = sum(1 for r in manifest if r.get("status") == "ok")
    print(f"\nDone. {ok}/{len(manifest)} files written.")
    print(f"Manifest: {manifest_path}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
