#!/usr/bin/env python3

"""pvxcodec — simulate lossy codec and transmission degradation.

Applies bandwidth limiting, sample-rate decimation, and bit-depth
quantization to simulate the artifacts of real-world codec processing
(MP3, telephone, VoIP, AM radio) without requiring external codec
binaries.

Available presets
-----------------
  mp3_low       ~32 kbps MP3 (11 kHz bandwidth, 12-bit quantization)
  mp3_medium    ~128 kbps MP3 (16 kHz bandwidth, 14-bit quantization)
  telephone     POTS telephone band (300-3400 Hz, 8 kHz, 8-bit)
  voip_narrow   Narrow-band VoIP (50-4000 Hz, 8 kHz, 8-bit)
  voip_wide     Wide-band VoIP (50-7000 Hz, 16 kHz, 12-bit)
  am_radio      AM radio (200-5000 Hz, 8-bit)
  lo_fi         Extreme lo-fi (8 kHz bandwidth, 6-bit)
  random        Pick a random preset each run

Examples
--------
pvxcodec speech.wav --codec telephone --output speech_phone.wav
pvxcodec music.wav --codec mp3_low --output music_lossy.wav
pvxcodec speech.wav --codec random --seed 7 --output out.wav
pvxcodec music.wav --bits 8 --output music_crushed.wav
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pvxcodec",
        description="Simulate lossy codec artifacts (MP3, telephone, VoIP) without codec binaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input audio file")
    p.add_argument("--output", "-o", required=True, help="Output audio file")
    p.add_argument(
        "--codec",
        choices=[
            "mp3_low",
            "mp3_medium",
            "telephone",
            "voip_narrow",
            "voip_wide",
            "am_radio",
            "lo_fi",
            "random",
        ],
        default=None,
        help="Codec preset to simulate (default: none — use --bits for raw bit-crush)",
    )
    p.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Bit depth for raw bit-crushing (4-16). Applied after codec if both specified.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument(
        "--bit-depth",
        choices=["16", "24", "32", "float"],
        default="24",
        help="Output file bit depth (default: 24)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from pvx.augment import Pipeline
    from pvx.augment.codec import BitCrusher, CodecDegradation
    from pvx.augment.core import load_audio, save_audio

    audio, sr = load_audio(args.input)

    transforms = []
    if args.codec:
        transforms.append(CodecDegradation(codec=args.codec))
    if args.bits is not None:
        transforms.append(BitCrusher(bits=(args.bits, args.bits)))
    if not transforms:
        parser.error("Provide --codec and/or --bits")

    pipeline = Pipeline(transforms)
    audio_out, sr_out = pipeline(audio, sr, seed=args.seed)

    subtype_map = {"16": "PCM_16", "24": "PCM_24", "32": "PCM_32", "float": "FLOAT"}
    save_audio(
        args.output,
        audio_out,
        sr_out,
        subtype=subtype_map.get(args.bit_depth, "PCM_24"),
    )
    print(f"[pvxcodec] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
