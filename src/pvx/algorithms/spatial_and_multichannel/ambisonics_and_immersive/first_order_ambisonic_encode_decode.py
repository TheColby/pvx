#!/usr/bin/env python3
"""First-order ambisonic encode/decode.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.first_order_ambisonic_encode_decode
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    for _parent in Path(__file__).resolve().parents:
        if (_parent / "pvx").is_dir():
            sys.path.append(str(_parent))
            break

from pvx.algorithms.base import AlgorithmResult, run_algorithm

ALGORITHM_ID = 'spatial_and_multichannel.first_order_ambisonic_encode_decode'
ALGORITHM_NAME = 'First-order ambisonic encode/decode'
THEME = 'Spatial and Multichannel'


def process(audio: np.ndarray, sample_rate: int, **params: Any) -> AlgorithmResult:
    """Run First-order ambisonic encode/decode on the provided audio buffer.

    Parameters:
    - audio: np.ndarray, shape (samples,) or (samples, channels)
    - sample_rate: input sample rate in Hz
    - **params: algorithm-specific controls routed by ALGORITHM_ID

    Returns:
    - AlgorithmResult with processed audio, output sample rate, and metadata.
    """
    return run_algorithm(
        algorithm_id=ALGORITHM_ID,
        algorithm_name=ALGORITHM_NAME,
        theme=THEME,
        audio=audio,
        sample_rate=sample_rate,
        params=params,
    )


def module_help_text() -> str:
    return (
        f"Module: {ALGORITHM_NAME}\n"
        f"Theme: {THEME}\n"
        f"Algorithm ID: {ALGORITHM_ID}\n\n"
        "Purpose:\n"
        "  Executes a concrete DSP/analysis implementation selected by ALGORITHM_ID.\n"
        "  Parameter names and defaults are handled in pvx.algorithms.base dispatch logic.\n\n"
        "Programmatic usage:\n"
        "  from <module> import process\n"
        "  result = process(audio, sample_rate, **params)\n\n"
        "Data contract:\n"
        "  - Input audio can be mono or multichannel; mono is promoted to (N,1).\n"
        "  - Output is AlgorithmResult(audio, sample_rate, metadata).\n"
        "  - metadata includes algorithm_id, theme, status, params, notes, and extras.\n\n"
        "For detailed per-algorithm parameter reference, see docs/PVX_ALGORITHM_PARAMS.md.\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Verbose help for {ALGORITHM_NAME} ({ALGORITHM_ID})",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print module help text (default behavior)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print(module_help_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
