# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
# ruff: noqa: E402

"""Tests for build_transient_mask logic."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure src is at the front of sys.path to import pvx package correctly
# and avoid importing pvx.py from root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

# Also remove current directory from sys.path if present, to prevent pvx.py import
# This is crucial when running tests from the root directory
if "" in sys.path:
    sys.path.remove("")
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

import numpy as np

from pvx.core.transients import build_transient_mask


class TestBuildTransientMask(unittest.TestCase):
    def test_empty_signal(self) -> None:
        mask = build_transient_mask(
            signal_samples=0,
            onset_samples=np.array([10], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=5,
            min_region_samples=5,
        )
        self.assertEqual(mask.size, 0)
        self.assertEqual(mask.dtype, bool)

    def test_no_onsets(self) -> None:
        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=5,
            min_region_samples=5,
        )
        self.assertEqual(mask.size, 100)
        self.assertFalse(np.any(mask))

    def test_basic_mask_construction(self) -> None:
        # protect_samples = 10 -> pre=3 (round(3.5)=4? No 0.35*10=3.5. round(3.5)=4 in Python 3? No, round(x.5) rounds to nearest even. 4.
        # Wait, round(3.5) is 4. round(2.5) is 2.
        # Let's check python rounding.
        # 0.35 * 10 = 3.5. round(3.5) -> 4.
        # 0.65 * 10 = 6.5. round(6.5) -> 6?
        # 4 + 6 = 10. Total 10.
        # So pre=4, post=6.
        # Onset at 50 -> [50-4, 50+6] -> [46, 56].
        # Indices 46..55 are True.

        # Let's verify round behavior in python.
        # I'll adjust expectations based on standard python behavior or check explicitly.
        # round(3.5) is 4.

        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([50], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=0,
            min_region_samples=0,
        )

        # Calculate expected pre/post
        pre = max(1, int(round(0.35 * 10))) # round(3.5)=4
        post = max(1, int(round(0.65 * 10))) # round(6.5)=6? No, nearest even. 6 is even. So 6.
        # total 10.

        start = 50 - pre # 46
        end = 50 + post # 56

        self.assertTrue(np.all(mask[start:end]))
        if start > 0:
            self.assertFalse(mask[start - 1])
        if end < 100:
            self.assertFalse(mask[end])
        self.assertEqual(np.sum(mask), end - start)

    def test_boundary_handling(self) -> None:
        pre = max(1, int(round(0.35 * 10))) # 4
        post = max(1, int(round(0.65 * 10))) # 6

        # onset at 0 -> [-4, 6] -> [0, 6]
        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([0], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=0,
            min_region_samples=0,
        )
        self.assertTrue(np.all(mask[0:post]))
        self.assertFalse(mask[post])

        # onset at 99 -> [95, 105] -> [95, 100]
        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([99], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=0,
            min_region_samples=0,
        )
        start = 99 - pre # 95
        self.assertTrue(np.all(mask[start:100]))
        self.assertFalse(mask[start - 1])

    def test_gap_merging(self) -> None:
        # protect_samples=10 -> pre=4, post=6.
        # Onset 1 at 20 -> [16, 26] (indices 16..25). Last index 25.
        # Onset 2 at 35 -> [31, 41] (indices 31..40). First index 31.
        # Distance between last index (25) and first index (31) is 31 - 25 = 6.
        # Indices in gap: 26, 27, 28, 29, 30. (5 indices).
        # merge logic: if (true_idx[idx] - true_idx[idx-1]) <= gap.
        # true_idx includes 25 and 31. 31 - 25 = 6.
        # So gap=6 should merge.

        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([20, 35], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=6,
            min_region_samples=0,
        )
        # Check gap region
        self.assertTrue(np.all(mask[26:31]))
        # Check full range is True from 16 to 41 (exclusive) -> 16..40
        self.assertTrue(np.all(mask[16:41]))

    def test_gap_not_merging(self) -> None:
        # Same setup. Distance is 6.
        # If merge_gap_samples=5, it should NOT merge.
        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([20, 35], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=5,
            min_region_samples=0,
        )
        self.assertFalse(np.any(mask[26:31]))
        self.assertTrue(np.all(mask[16:26]))
        self.assertTrue(np.all(mask[31:41]))

    def test_min_region_enforcement(self) -> None:
        # Create a short transient region.
        # protect_samples=4 -> pre=1 (round(1.4)=1), post=3 (round(2.6)=3). Total 4.
        # Onset at 50 -> [49, 53]. Length 4.
        # min_region_samples=5. Should remove it (flip to False).
        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([50], dtype=np.int64),
            protect_samples=4,
            merge_gap_samples=0,
            min_region_samples=5,
        )
        self.assertFalse(np.any(mask))

        # Create a short gap.
        # Onset 1 at 20 -> [16, 26) -> 16..25. (using protect=10, pre=4, post=6)
        # Onset 2 at 32 -> [28, 38) -> 28..37.
        # Gap is [26, 28). Indices 26, 27. Length 2.
        # min_region_samples=3. Should fill the gap (flip False to True).
        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([20, 32], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=0, # Don't merge by gap logic
            min_region_samples=3, # Merge by min region logic
        )
        self.assertTrue(np.all(mask[26:28]))
        self.assertTrue(np.all(mask[16:38]))

    def test_dense_onsets(self) -> None:
        # Overlapping regions should merge naturally.
        # protect=10 -> pre=4, post=6.
        # Onset 20 -> [16, 26)
        # Onset 25 -> [21, 31)
        # Union is [16, 31)
        mask = build_transient_mask(
            signal_samples=100,
            onset_samples=np.array([20, 25], dtype=np.int64),
            protect_samples=10,
            merge_gap_samples=0,
            min_region_samples=0,
        )
        self.assertTrue(np.all(mask[16:31]))
        self.assertEqual(np.sum(mask), 15)

if __name__ == "__main__":
    unittest.main()
