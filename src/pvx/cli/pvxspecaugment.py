#!/usr/bin/env python3

"""pvxspecaugment — apply SpecAugment frequency and time masking.

Implements the SpecAugment policy (Park et al., 2019) for training robust
speech and audio models.  Applies random frequency and time masks directly
to the STFT magnitude spectrum, then reconstructs the waveform.

References
----------
Park, D. S., et al. (2019). "SpecAugment: A Simple Data Augmentation
Method for Automatic Speech Recognition." Interspeech.
https://arxiv.org/abs/1904.08779

Examples
--------
# Standard SpecAugment LB policy:
pvxspecaugment speech.wav --freq-mask 30 --time-mask 40 --num-masks 2 --output out.wav

# LD policy (fewer, narrower masks):
pvxspecaugment speech.wav --freq-mask 27 --time-mask 70 --num-masks 1 --output out.wav

# LibriSpeech LB policy:
pvxspecaugment speech.wav --freq-mask 30 --time-mask 100 --num-freq-masks 2 --num-time-masks 2 --output out.wav
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pvxspecaugment",
        description="Apply SpecAugment frequency and time masking to audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input audio file")
    p.add_argument("--output", "-o", required=True, help="Output audio file")
    p.add_argument(
        "--freq-mask",
        type=int,
        default=27,
        help="Maximum frequency mask width F in bins (default: 27)",
    )
    p.add_argument(
        "--time-mask",
        type=int,
        default=100,
        help="Maximum time mask width T in frames (default: 100)",
    )
    p.add_argument(
        "--num-freq-masks",
        type=int,
        default=2,
        help="Number of frequency masks (default: 2)",
    )
    p.add_argument(
        "--num-time-masks",
        type=int,
        default=2,
        help="Number of time masks (default: 2)",
    )
    p.add_argument(
        "--fill",
        default="0",
        help="Fill value for masked regions: 0, mean, or a float (default: 0)",
    )
    p.add_argument("--n-fft", type=int, default=512, help="FFT size (default: 512)")
    p.add_argument(
        "--hop-length",
        type=int,
        default=128,
        help="STFT hop length in samples (default: 128)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument(
        "--bit-depth",
        choices=["16", "24", "32", "float"],
        default="24",
        help="Output bit depth (default: 24)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from pvx.augment.core import load_audio, save_audio
    from pvx.augment.spectral import SpecAugment

    fill_value: float | str
    try:
        fill_value = float(args.fill)
    except ValueError:
        fill_value = str(args.fill)

    audio, sr = load_audio(args.input)
    aug = SpecAugment(
        freq_mask_param=args.freq_mask,
        time_mask_param=args.time_mask,
        num_freq_masks=args.num_freq_masks,
        num_time_masks=args.num_time_masks,
        fill_value=fill_value,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    audio_out, sr_out = aug(audio, sr, seed=args.seed)

    subtype_map = {"16": "PCM_16", "24": "PCM_24", "32": "PCM_32", "float": "FLOAT"}
    save_audio(
        args.output,
        audio_out,
        sr_out,
        subtype=subtype_map.get(args.bit_depth, "PCM_24"),
    )
    print(f"[pvxspecaugment] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
