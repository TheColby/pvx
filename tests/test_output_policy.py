"""Unit tests for shared output policy helpers."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pvx.core.output_policy import (
    prepare_output_audio,
    resolve_output_subtype,
    true_peak_dbtp,
    validate_output_policy_args,
    write_metadata_sidecar,
)


def _make_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "subtype": None,
        "bit_depth": "inherit",
        "dither": "none",
        "dither_seed": None,
        "true_peak_max_dbtp": None,
        "metadata_policy": "none",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestOutputPolicy(unittest.TestCase):
    def test_resolve_output_subtype_from_bit_depth(self) -> None:
        args = _make_args(bit_depth="24")
        self.assertEqual(resolve_output_subtype(args), "PCM_24")

    def test_prepare_output_audio_enforces_true_peak(self) -> None:
        args = _make_args(true_peak_max_dbtp=-12.0)
        x = (0.95 * np.sin(2.0 * np.pi * np.arange(4096, dtype=np.float64) / 64.0))[:, None]
        y, subtype = prepare_output_audio(x, 24000, args)
        self.assertIsNone(subtype)
        self.assertLessEqual(true_peak_dbtp(y, 24000), -11.9)

    def test_prepare_output_audio_dither_seed_is_deterministic(self) -> None:
        args = _make_args(bit_depth="16", dither="tpdf", dither_seed=1234)
        x = np.zeros((512, 1), dtype=np.float64)
        y1, subtype1 = prepare_output_audio(x, 24000, args)
        y2, subtype2 = prepare_output_audio(x, 24000, args)
        self.assertEqual(subtype1, "PCM_16")
        self.assertEqual(subtype2, "PCM_16")
        self.assertTrue(np.array_equal(y1, y2))
        self.assertFalse(np.array_equal(y1, x))

    def test_validate_output_policy_args_rejects_invalid_combo(self) -> None:
        parser = argparse.ArgumentParser()
        args = _make_args(bit_depth="32f", dither="tpdf")
        with self.assertRaises(SystemExit):
            validate_output_policy_args(args, parser)

    def test_write_metadata_sidecar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "test.wav"
            args = _make_args(
                metadata_policy="sidecar",
                bit_depth="16",
                dither="tpdf",
                dither_seed=7,
                true_peak_max_dbtp=-1.0,
            )
            audio = np.zeros((128, 2), dtype=np.float64)
            sidecar = write_metadata_sidecar(
                output_path=output,
                input_path=Path("in.wav"),
                audio=audio,
                sample_rate=24000,
                subtype="PCM_16",
                args=args,
                extra={"case": "unit"},
            )
            self.assertIsNotNone(sidecar)
            assert sidecar is not None
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata_policy"], "sidecar")
            self.assertEqual(payload["output"]["subtype"], "PCM_16")
            self.assertEqual(payload["output_policy"]["bit_depth"], "16")
            self.assertEqual(payload["extra"]["case"], "unit")


if __name__ == "__main__":
    unittest.main()
