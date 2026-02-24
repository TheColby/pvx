# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
# ruff: noqa: E402

"""Tests for map_mask_to_output in pvx.core.transients."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Ensure src is at the front of sys.path to prioritize package over root scripts
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

from pvx.core.transients import map_mask_to_output


class TestMapMaskToOutput(unittest.TestCase):
    def test_stretch_increases_length(self) -> None:
        """Verify that stretch > 1.0 produces longer mask with correct interpolation."""
        mask = np.array([True, False, True, False])
        # Stretch = 2.0 -> Output length 8
        out = map_mask_to_output(mask, 2.0, 8)
        self.assertEqual(out.shape, (8,))
        self.assertEqual(out.dtype, bool)
        # Expected: [T, T, F, T, T, T, F, F]
        expected = np.array([True, True, False, True, True, True, False, False])
        np.testing.assert_array_equal(out, expected)

    def test_shrink_decreases_length(self) -> None:
        """Verify that stretch < 1.0 produces shorter mask."""
        mask = np.array([True, False, True, False])
        # Stretch = 0.5 -> Output length 2
        out = map_mask_to_output(mask, 0.5, 2)
        self.assertEqual(out.shape, (2,))
        # Expected: [T, T] (indices 0->0, 1->2)
        expected = np.array([True, True])
        np.testing.assert_array_equal(out, expected)

    def test_identity_stretch(self) -> None:
        """Verify that stretch=1.0 preserves the mask."""
        mask = np.array([True, False, True])
        out = map_mask_to_output(mask, 1.0, 3)
        np.testing.assert_array_equal(out, mask)

    def test_zero_output_length(self) -> None:
        """Verify it returns empty array when output samples is 0."""
        mask = np.array([True, False])
        out = map_mask_to_output(mask, 1.0, 0)
        self.assertEqual(out.size, 0)
        self.assertEqual(out.dtype, bool)

    def test_empty_input_mask(self) -> None:
        """Verify it returns False array when input mask is empty."""
        mask = np.array([], dtype=bool)
        out = map_mask_to_output(mask, 1.0, 5)
        self.assertEqual(out.shape, (5,))
        self.assertFalse(np.any(out))

    def test_zero_stretch_edge_case(self) -> None:
        """Verify behavior when stretch is 0.0."""
        # Should be clamped to 1e-9 internally.
        # Index 0 maps to 0. Subsequent indices map to very large values which are clamped to the last element.
        mask = np.array([True, False])
        out = map_mask_to_output(mask, 0.0, 3)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(out, expected)

    def test_input_flattening(self) -> None:
        """Verify that multi-dimensional input masks are treated as flat."""
        mask = np.array([[True, False], [True, False]]) # Shape (2, 2) -> Flat: [T, F, T, F]
        out = map_mask_to_output(mask, 1.0, 4)
        expected = np.array([True, False, True, False])
        np.testing.assert_array_equal(out, expected)

    def test_large_stretch(self) -> None:
        """Verify behavior with very large stretch factors."""
        mask = np.array([True, False])
        # Stretch = 10.0 -> Output length 20
        out = map_mask_to_output(mask, 10.0, 20)
        self.assertEqual(out.size, 20)
        self.assertEqual(out.dtype, bool)
        # Check first few are True (from mask[0])
        self.assertTrue(out[0])
        # Check last few are False (from mask[1])
        self.assertFalse(out[-1])

if __name__ == "__main__":
    unittest.main()
