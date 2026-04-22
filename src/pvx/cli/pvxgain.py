#!/usr/bin/env python3

"""pvxgain — apply random or fixed gain perturbation to audio.

A simple utility for volume normalization and random loudness perturbation.
Useful as a pre-processing or augmentation step in any audio ML pipeline.

Examples
--------
# Random gain -6 to +6 dB:
pvxgain speech.wav --gain -6,6 --output speech_gain.wav

# Normalize to -1 dBFS peak:
pvxgain speech.wav --normalize peak --target -1 --output normalized.wav

# Normalize to -20 dBFS RMS:
pvxgain speech.wav --normalize rms --target -20 --output normalized.wav

# Batch random loudness for dataset:
for f in dataset/*.wav; do
    pvxgain "$f" --gain -6,6 --seed $RANDOM --output augmented/"$(basename $f)"
done
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pvxgain",
        description="Apply random or fixed gain perturbation or normalization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input audio file")
    p.add_argument("--output", "-o", required=True, help="Output audio file")
    p.add_argument(
        "--gain",
        default=None,
        help=(
            "Gain in dB. Single value (fixed) or 'min,max' (random range). "
            "Mutually exclusive with --normalize."
        ),
    )
    p.add_argument(
        "--normalize",
        choices=["peak", "rms"],
        default=None,
        help="Normalization mode. Mutually exclusive with --gain.",
    )
    p.add_argument(
        "--target",
        type=float,
        default=-1.0,
        help="Target level in dBFS for normalization (default: -1.0)",
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

    if args.gain and args.normalize:
        parser.error("--gain and --normalize are mutually exclusive")
    if not args.gain and not args.normalize:
        parser.error("Provide either --gain or --normalize")

    from pvx.augment.core import load_audio, save_audio
    from pvx.augment.time_domain import GainPerturber, Normalizer

    audio, sr = load_audio(args.input)

    if args.gain:
        parts = [float(x) for x in str(args.gain).split(",")]
        gain_range = (parts[0], parts[-1])
        aug = GainPerturber(gain_db=gain_range)
        audio_out, sr_out = aug(audio, sr, seed=args.seed)
    else:
        aug = Normalizer(mode=args.normalize, target_db=args.target)
        audio_out, sr_out = aug(audio, sr, seed=args.seed)

    subtype_map = {"16": "PCM_16", "24": "PCM_24", "32": "PCM_32", "float": "FLOAT"}
    save_audio(
        args.output,
        audio_out,
        sr_out,
        subtype=subtype_map.get(args.bit_depth, "PCM_24"),
    )
    print(f"[pvxgain] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
