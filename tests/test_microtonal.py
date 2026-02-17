import tempfile
import unittest
from pathlib import Path

import numpy as np

from pvxcommon import cents_to_ratio, read_segment_csv
from pvxretune import nearest_scale_freq
from pvxvoc import choose_pitch_ratio


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class TestMicrotonalSupport(unittest.TestCase):
    def test_read_segment_csv_accepts_pitch_cents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "map.csv"
            write_text(
                csv_path,
                "start_sec,end_sec,stretch,pitch_cents\n"
                "0.0,1.0,1.0,50\n",
            )
            segments = read_segment_csv(csv_path, has_pitch=True)
            self.assertEqual(len(segments), 1)
            self.assertAlmostEqual(segments[0].pitch_ratio, cents_to_ratio(50.0), delta=1e-12)

    def test_read_segment_csv_accepts_pitch_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "map.csv"
            write_text(
                csv_path,
                "start_sec,end_sec,stretch,pitch_ratio\n"
                "0.0,1.0,1.0,1.03715\n",
            )
            segments = read_segment_csv(csv_path, has_pitch=True)
            self.assertEqual(len(segments), 1)
            self.assertAlmostEqual(segments[0].pitch_ratio, 1.03715, delta=1e-12)

    def test_read_segment_csv_rejects_multiple_pitch_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "map.csv"
            write_text(
                csv_path,
                "start_sec,end_sec,stretch,pitch_cents,pitch_ratio\n"
                "0.0,1.0,1.0,25,1.02\n",
            )
            with self.assertRaises(ValueError):
                read_segment_csv(csv_path, has_pitch=True)

    def test_nearest_scale_freq_supports_custom_cents_scale(self) -> None:
        source = 440.0 * cents_to_ratio(33.0)
        target = nearest_scale_freq(
            source,
            "A",
            "chromatic",
            custom_scale_cents=[0.0, 50.0, 100.0],
        )
        self.assertAlmostEqual(target, 440.0 * cents_to_ratio(50.0), delta=0.01)

    def test_choose_pitch_ratio_supports_cents(self) -> None:
        args = type(
            "Args",
            (),
            {
                "pitch_shift_ratio": None,
                "pitch_shift_semitones": None,
                "pitch_shift_cents": 75.0,
                "target_f0": None,
                "analysis_channel": "mix",
                "f0_min": 50.0,
                "f0_max": 1000.0,
            },
        )()
        signal = np.zeros((128, 1), dtype=np.float64)
        cfg = choose_pitch_ratio(args, signal, 24000)
        self.assertAlmostEqual(cfg.ratio, cents_to_ratio(75.0), delta=1e-12)


if __name__ == "__main__":
    unittest.main()
