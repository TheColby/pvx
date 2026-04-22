#!/usr/bin/env python3

"""Benchmark: WER impact of pvx augmentation on a speech recognition task.

This script measures the effect of ``pvx.augment`` pipelines on ASR accuracy
by fine-tuning a small Wav2Vec2 model on LibriSpeech-clean-100 with and
without augmentation, then evaluating WER on the ``test-clean`` split.

Requirements
------------
    pip install "pvx[torch]" datasets transformers evaluate jiwer torchaudio

Usage
-----
    # Quick benchmark (~20 min on a single GPU)
    python benchmarks/run_augment_wer_bench.py --epochs 3 --max-train-samples 2000

    # Full benchmark (~2-4 hours on a single GPU)
    python benchmarks/run_augment_wer_bench.py --epochs 10

    # CPU-only (slow, for verification only)
    python benchmarks/run_augment_wer_bench.py --epochs 2 --max-train-samples 500 --device cpu

Results are saved as JSON to ``benchmarks/out_augment/wer_results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark pvx augmentation WER impact on LibriSpeech")
    p.add_argument(
        "--epochs", type=int, default=5, help="Training epochs per condition (default: 5)"
    )
    p.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    p.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    p.add_argument(
        "--max-train-samples", type=int, default=None, help="Cap training set size for quick runs"
    )
    p.add_argument(
        "--max-eval-samples", type=int, default=500, help="Cap eval set size (default: 500)"
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out-dir", type=str, default="benchmarks/out_augment")
    p.add_argument(
        "--model-name",
        type=str,
        default="facebook/wav2vec2-base",
        help="HuggingFace model ID (default: facebook/wav2vec2-base)",
    )
    return p.parse_args()


def get_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def build_pipelines():
    """Return a dict of {name: pipeline_or_None} for benchmark conditions."""
    from pvx.augment import (
        AddNoise,
        CodecDegradation,
        GainPerturber,
        Pipeline,
        RoomSimulator,
        SpecAugment,
    )

    conditions = {
        "baseline (no augmentation)": None,
        "gain only": Pipeline(
            [
                GainPerturber(gain_db=(-6, 6), p=0.9),
            ],
            seed=42,
        ),
        "noise + gain": Pipeline(
            [
                GainPerturber(gain_db=(-6, 6), p=0.9),
                AddNoise(snr_db=(10, 35), noise_type="pink", p=0.6),
            ],
            seed=42,
        ),
        "SpecAugment only": Pipeline(
            [
                SpecAugment(
                    freq_mask_param=27,
                    time_mask_param=100,
                    num_freq_masks=2,
                    num_time_masks=2,
                    p=0.8,
                ),
            ],
            seed=42,
        ),
        "full asr_pipeline": Pipeline(
            [
                GainPerturber(gain_db=(-3, 3), p=0.8),
                RoomSimulator(rt60_range=(0.05, 0.6), wet_range=(0.1, 0.5), p=0.4),
                AddNoise(snr_db=(15, 40), noise_type="pink", p=0.5),
                CodecDegradation(codec="random", p=0.2),
                SpecAugment(
                    freq_mask_param=20,
                    time_mask_param=30,
                    num_freq_masks=1,
                    num_time_masks=1,
                    p=0.5,
                ),
            ],
            seed=42,
        ),
    }
    return conditions


def prepare_dataset(max_train: int | None, max_eval: int):
    """Load LibriSpeech train-clean-100 and test-clean via HuggingFace."""
    from datasets import Audio, load_dataset

    print("Loading LibriSpeech train-clean-100 and test-clean...")
    ds_train = load_dataset("librispeech_asr", "clean", split="train.100", trust_remote_code=True)
    ds_eval = load_dataset("librispeech_asr", "clean", split="test", trust_remote_code=True)

    if max_train is not None:
        ds_train = ds_train.select(range(min(max_train, len(ds_train))))
    if max_eval is not None:
        ds_eval = ds_eval.select(range(min(max_eval, len(ds_eval))))

    # Ensure consistent 16kHz
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_eval = ds_eval.cast_column("audio", Audio(sampling_rate=16000))

    print(f"  Train samples: {len(ds_train)}, Eval samples: {len(ds_eval)}")
    return ds_train, ds_eval


def train_and_evaluate(
    condition_name: str,
    pipeline,
    ds_train,
    ds_eval,
    args,
    processor,
    device: str,
):
    """Train Wav2Vec2 with one augmentation condition and return WER."""
    import evaluate
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2ForCTC,
    )

    print(f"\n{'=' * 60}")
    print(f"Condition: {condition_name}")
    print(f"{'=' * 60}")

    # Preprocess function: apply augmentation then extract features
    def preprocess(examples, indices=None):
        audio_arrays = [a["array"] for a in examples["audio"]]
        sample_rates = [a["sampling_rate"] for a in examples["audio"]]

        processed_arrays = []
        for i, (audio, sr) in enumerate(zip(audio_arrays, sample_rates)):
            audio = np.array(audio, dtype=np.float32)
            if pipeline is not None:
                idx = indices[i] if indices is not None else i
                audio, sr = pipeline(audio, sr, seed=idx)
            processed_arrays.append(audio)

        inputs = processor(
            processed_arrays,
            sampling_rate=16000,
            return_tensors="np",
            padding=True,
        )

        with processor.as_target_processor():
            labels = processor(examples["text"]).input_ids

        inputs["labels"] = labels
        return inputs

    # Preprocess eval (no augmentation)
    def preprocess_eval(examples):
        audio_arrays = [np.array(a["array"], dtype=np.float32) for a in examples["audio"]]
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="np",
            padding=True,
        )
        with processor.as_target_processor():
            labels = processor(examples["text"]).input_ids
        inputs["labels"] = labels
        return inputs

    print("  Preprocessing train set...")
    train_encoded = ds_train.map(
        preprocess,
        batched=True,
        batch_size=32,
        with_indices=True,
        remove_columns=ds_train.column_names,
        desc="Augmenting + encoding",
    )

    print("  Preprocessing eval set...")
    eval_encoded = ds_eval.map(
        preprocess_eval,
        batched=True,
        batch_size=32,
        remove_columns=ds_eval.column_names,
        desc="Encoding eval",
    )

    # Load fresh model for each condition
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_extractor()

    # WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        # Replace -100 in labels
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    out_dir = Path(args.out_dir) / condition_name.replace(" ", "_").replace("(", "").replace(
        ")", ""
    )
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    t0 = time.time()
    metrics = trainer.evaluate()
    eval_time = time.time() - t0

    wer = metrics["eval_wer"]
    print(f"  WER: {wer:.4f} ({wer * 100:.2f}%)")
    print(f"  Train time: {train_time:.1f}s, Eval time: {eval_time:.1f}s")

    return {
        "condition": condition_name,
        "wer": round(wer, 6),
        "wer_pct": round(wer * 100, 2),
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
        "epochs": args.epochs,
        "train_samples": len(train_encoded),
        "eval_samples": len(eval_encoded),
    }


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Check dependencies
    try:
        import datasets
        import evaluate
        import jiwer
        import torch
        import transformers
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print(
            "Install with: pip install 'pvx[torch]' datasets transformers evaluate jiwer torchaudio"
        )
        sys.exit(1)

    from transformers import Wav2Vec2Processor

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load processor
    print(f"Loading processor from {args.model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    # Load data
    ds_train, ds_eval = prepare_dataset(args.max_train_samples, args.max_eval_samples)

    # Build augmentation conditions
    conditions = build_pipelines()

    # Run benchmark
    results = []
    for name, pipeline in conditions.items():
        result = train_and_evaluate(name, pipeline, ds_train, ds_eval, args, processor, device)
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    baseline_wer = results[0]["wer_pct"]
    for r in results:
        delta = r["wer_pct"] - baseline_wer
        sign = "+" if delta > 0 else ""
        marker = (
            " (baseline)"
            if r["condition"] == "baseline (no augmentation)"
            else f" ({sign}{delta:.2f}pp)"
        )
        print(f"  {r['condition']:35s} WER: {r['wer_pct']:6.2f}%{marker}")

    # Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wer_results.json"
    payload = {
        "benchmark": "pvx_augment_wer",
        "model": args.model_name,
        "device": device,
        "args": vars(args),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
