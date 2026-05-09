#!/usr/bin/env python3
"""PyTorch DataLoader example with pvx augmentation.

Shows how to use PvxAugmentDataset in a complete PyTorch training loop
with multi-worker data loading, variable-length collation, and
deterministic validation.

Requirements: pip install torch torchaudio

Usage
-----
    python examples/pytorch_dataset.py \
        --train-dir data/train \
        --val-dir data/val \
        --epochs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_train_pipeline():
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
            GainPerturber(gain_db=(-6, 6), p=0.8),
            RoomSimulator(rt60_range=(0.05, 1.0), wet_range=(0.1, 0.6), p=0.4),
            AddNoise(snr_db=(10, 35), noise_type="pink", p=0.5),
            OneOf([CodecDegradation(codec="telephone"), CodecDegradation(codec="voip_narrow"), Identity()],
                  weights=[0.2, 0.2, 0.6], p=0.3),
            SpecAugment(freq_mask_param=27, time_mask_param=100, num_freq_masks=2, num_time_masks=2, p=0.7),
        ],
        seed=42,
    )


def build_val_pipeline():
    """Validation: no augmentation (or very mild normalisation only)."""
    from pvx.augment import Normalizer, Pipeline

    return Pipeline([Normalizer(mode="rms", target_db=-20.0, p=1.0)], seed=0)


def main(argv: list[str] | None = None) -> int:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch not installed. Run: pip install torch", file=sys.stderr)
        return 1

    from pvx.integrations.pytorch import PvxAugmentDataset

    parser = argparse.ArgumentParser(description="PyTorch training loop example")
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args(argv)

    # Collect files
    exts = {".wav", ".flac", ".aiff"}
    train_files = sorted(p for p in Path(args.train_dir).rglob("*") if p.suffix.lower() in exts)
    val_files = sorted(p for p in Path(args.val_dir).rglob("*") if p.suffix.lower() in exts)

    if not train_files:
        print(f"No audio files found in {args.train_dir}", file=sys.stderr)
        return 1

    print(f"Train: {len(train_files)} files | Val: {len(val_files)} files")

    # Datasets
    train_dataset = PvxAugmentDataset(
        train_files,
        pipeline=build_train_pipeline(),
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        mono=True,
        seed_offset=0,
    )

    val_dataset = PvxAugmentDataset(
        val_files,
        pipeline=build_val_pipeline(),
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        mono=True,
        seed_offset=99999,
    )

    train_loader = DataLoader(
        train_dataset.as_torch_dataset(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=PvxAugmentDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset.as_torch_dataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=PvxAugmentDataset.collate_fn,
    )

    # Minimal model stub for demonstration
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool1d(128),
                torch.nn.Flatten(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nStarting training for {args.epochs} epoch(s)…")
    for epoch in range(args.epochs):
        model.train()
        total_batches = 0
        for batch in train_loader:
            audio = batch["audio"]       # (B, T)
            optimizer.zero_grad()
            out = model(audio.unsqueeze(1))
            loss = out.mean()            # dummy loss for demonstration
            loss.backward()
            optimizer.step()
            total_batches += 1
        print(f"Epoch {epoch + 1}/{args.epochs}: {total_batches} train batches")

        model.eval()
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"]
                _ = model(audio.unsqueeze(1))
                val_batches += 1
        print(f"  Val: {val_batches} batches")

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
