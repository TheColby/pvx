import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pvx.core import streaming


class TestStreamingSecurity(unittest.TestCase):
    @patch("pvx.core.streaming.sf.read")
    @patch("pvx.core.streaming.sf.info")
    def test_read_audio_large_file_check(self, mock_info, mock_read):
        # Setup mock to simulate a large file
        # Mock MAX_AUDIO_BYTES to be small for testing
        with patch("pvx.core.streaming.MAX_AUDIO_BYTES", 1000):
            # 200 frames * 1 channel * 8 bytes = 1600 bytes > 1000 bytes
            mock_info.return_value = MagicMock(frames=200, channels=1)
            mock_read.return_value = (np.zeros((200, 1)), 44100)

            with self.assertRaises(ValueError) as cm:
                streaming._read_audio(Path("fake_large_file.wav"))
            self.assertIn("Decoded audio exceeds memory limit", str(cm.exception))

    @patch("pvx.core.streaming.sys.stdin")
    @patch("pvx.core.streaming.sf.read")
    @patch("pvx.core.streaming.sf.info")
    def test_read_audio_stdin_large_input(self, mock_info, mock_read, mock_stdin):
        mock_buffer = MagicMock()
        mock_stdin.buffer = mock_buffer

        # Mock MAX_INPUT_BYTES to be small
        with patch("pvx.core.streaming.MAX_INPUT_BYTES", 100):
            # read returns chunks. 50 bytes + 60 bytes = 110 bytes > 100 bytes
            mock_buffer.read.side_effect = [b"0" * 50, b"0" * 60, b""]

            with self.assertRaises(ValueError) as cm:
                streaming._read_audio(Path("-"))
            self.assertIn("exceeds size limit", str(cm.exception))

    @patch("pvx.core.streaming.sys.stdin")
    @patch("pvx.core.streaming.sf.read")
    @patch("pvx.core.streaming.sf.info")
    def test_read_audio_stdin_large_decoded(self, mock_info, mock_read, mock_stdin):
        # Input is small, but decoded is large
        mock_buffer = MagicMock()
        mock_stdin.buffer = mock_buffer

        with (
            patch("pvx.core.streaming.MAX_INPUT_BYTES", 1000),
            patch("pvx.core.streaming.MAX_AUDIO_BYTES", 100),
        ):
            mock_buffer.read.side_effect = [b"0" * 50, b""]  # 50 bytes input, OK

            # Decoded: 20 frames * 1 channel * 8 bytes = 160 bytes > 100 bytes
            mock_info.return_value = MagicMock(frames=20, channels=1)

            with self.assertRaises(ValueError) as cm:
                streaming._read_audio(Path("-"))
            self.assertIn("Decoded audio exceeds memory limit", str(cm.exception))

    @patch("pvx.core.streaming.sf.read")
    @patch("pvx.core.streaming.sf.info")
    def test_read_audio_valid(self, mock_info, mock_read):
        with patch("pvx.core.streaming.MAX_AUDIO_BYTES", 1000):
            # 10 frames * 1 channel * 8 bytes = 80 bytes < 1000 bytes
            mock_info.return_value = MagicMock(frames=10, channels=1)
            mock_read.return_value = (np.zeros((10, 1)), 44100)

            audio, sr = streaming._read_audio(Path("valid.wav"))
            self.assertEqual(sr, 44100)
            self.assertEqual(audio.shape, (10, 1))


if __name__ == "__main__":
    unittest.main()
