#!/usr/bin/env python3

"""pvxnoise — add synthetic noise to audio at a controlled SNR.

Supports white, pink, brown, and band-limited noise injection with
precise signal-to-noise ratio control.  Useful for building robust ASR,
speaker verification, and speech enhancement datasets.

Examples
--------
# Add pink noise at 20 dB SNR:
pvxnoise speech.wav --snr 20 --noise-type pink --output speech_noisy.wav

# Random SNR between 5-30 dB, white noise:
pvxnoise speech.wav --snr 5,30 --noise-type white --output speech_noisy.wav

# Band-limited noise (telephone band):
pvxnoise speech.wav --snr 15 --noise-type bandlimited --band 300,3400 --output out.wav

# Background noise mixing from a directory:
pvxnoise speech.wav --background-dir noise_samples/ --snr 10,25 --output out.wav
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pvxnoise",
        description="Add synthetic or background noise to audio at a controlled SNR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input audio file")
    p.add_argument("--output", "-o", required=True, help="Output audio file")
    p.add_argument(
        "--snr",
        default="20",
        help=(
            "Signal-to-noise ratio in dB. "
            "Single value for fixed SNR or 'min,max' for uniform sampling. "
            "(default: 20)"
        ),
    )
    p.add_argument(
        "--noise-type",
        choices=["white", "pink", "brown", "gaussian", "bandlimited"],
        default="white",
        help="Noise type (default: white)",
    )
    p.add_argument(
        "--band",
        default="300,4000",
        help="Low,high frequency band in Hz for bandlimited noise (default: 300,4000)",
    )
    p.add_argument(
        "--background-dir",
        default=None,
        help="Directory of background audio files to mix instead of synthetic noise",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
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
    from pvx.augment.noise import AddNoise, BackgroundMixer

    # Parse SNR range
    snr_parts = [float(x) for x in str(args.snr).split(",")]
    snr_range = (snr_parts[0], snr_parts[-1])

    # Parse band
    band_parts = [float(x) for x in str(args.band).split(",")]
    band_hz = (band_parts[0], band_parts[-1])

    audio, sr = load_audio(args.input)

    if args.background_dir:
        aug = BackgroundMixer(args.background_dir, snr_db=snr_range)
    else:
        aug = AddNoise(snr_db=snr_range, noise_type=args.noise_type, band_hz=band_hz)

    audio_out, sr_out = aug(audio, sr, seed=args.seed)

    subtype_map = {"16": "PCM_16", "24": "PCM_24", "32": "PCM_32", "float": "FLOAT"}
    save_audio(
        args.output,
        audio_out,
        sr_out,
        subtype=subtype_map.get(args.bit_depth, "PCM_24"),
    )
    print(f"[pvxnoise] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
