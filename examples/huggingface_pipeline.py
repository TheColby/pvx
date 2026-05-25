#!/usr/bin/env python3
"""HuggingFace datasets augmentation example using pvx.

Demonstrates how to load a speech dataset from the HuggingFace Hub,
apply pvx augmentation, and save the result for training.

Requirements: pip install datasets transformers pvx

Usage
-----
    # Augment CommonVoice English (replace with any HF audio dataset)
    python examples/huggingface_pipeline.py \
        --dataset mozilla-foundation/common_voice_13_0 \
        --subset en \
        --split train \
        --output-dir data/cv_aug \
        --variants 3 \
        --workers 8

    # Augment a local audio folder
    python examples/huggingface_pipeline.py \
        --local-dir data/my_audio \
        --output-dir data/my_audio_aug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_augment_pipeline(intent: str = "asr_robust", seed: int = 42):
    if intent == "asr_robust":
        from pvx.augment import asr_pipeline
        return asr_pipeline(seed=seed)
    if intent == "mir_music":
        from pvx.augment import music_pipeline
        return music_pipeline(seed=seed)
    if intent == "ssl_contrastive":
        from pvx.augment import contrastive_pipeline
        return contrastive_pipeline(seed=seed)[0]

    from pvx.augment import speech_enhancement_pipeline
    return speech_enhancement_pipeline(seed=seed)


def main(argv: list[str] | None = None) -> int:
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("HuggingFace datasets not installed. Run: pip install datasets", file=sys.stderr)
        return 1

    from datasets import Audio, load_dataset

    parser = argparse.ArgumentParser(description="HuggingFace augmentation example")
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset name")
    parser.add_argument("--subset", default=None, help="Dataset subset/config name")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--local-dir", default=None, type=Path, help="Local directory of audio files")
    parser.add_argument("--audio-column", default="audio", help="Audio column name (default: audio)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--intent", choices=["asr_robust", "mir_music", "ssl_contrastive", "speech_enhancement"],
                        default="asr_robust")
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--push-to-hub", default=None, help="HuggingFace Hub repo to push to")
    args = parser.parse_args(argv)

    # Load dataset
    if args.local_dir:
        print(f"Loading local audio from {args.local_dir} …")
        ds = load_dataset("audiofolder", data_dir=str(args.local_dir), split=args.split)
    elif args.dataset:
        print(f"Loading {args.dataset} ({args.subset or 'default'}) split={args.split} …")
        load_kwargs = {}
        if args.subset:
            load_kwargs["name"] = args.subset
        ds = load_dataset(args.dataset, split=args.split, **load_kwargs)
    else:
        parser.error("Provide --dataset or --local-dir")
        return 1

    # Cast audio column to target sample rate
    if args.audio_column in ds.features:
        ds = ds.cast_column(args.audio_column, Audio(sampling_rate=args.sample_rate))

    print(f"Dataset: {len(ds)} rows")

    pipeline = build_augment_pipeline(args.intent, args.seed)

    from pvx.integrations.huggingface import make_augment_map_fn

    all_augmented = []
    for var_idx in range(args.variants):
        variant_seed = args.seed + var_idx * 100_000
        fn = make_augment_map_fn(
            pipeline,
            audio_column=args.audio_column,
            output_column=args.audio_column,  # overwrite in-place for this variant
            base_seed=variant_seed,
            return_metadata=True,
        )
        ds_variant = ds.map(
            fn,
            batched=False,
            with_indices=True,
            num_proc=args.workers,
            desc=f"Augmenting variant {var_idx + 1}/{args.variants}",
        )
        all_augmented.append(ds_variant)
        print(f"Variant {var_idx + 1} done: {len(ds_variant)} rows")

    # Concatenate all variants
    from datasets import concatenate_datasets
    ds_aug = concatenate_datasets([ds, *all_augmented])
    print(f"\nFinal dataset: {len(ds_aug)} rows ({len(ds)} original + {len(ds) * args.variants} augmented)")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds_aug.save_to_disk(str(args.output_dir))
    print(f"Saved to {args.output_dir}")

    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub} …")
        ds_aug.push_to_hub(args.push_to_hub)
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
