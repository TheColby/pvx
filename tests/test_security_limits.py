import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import numpy as np
from pathlib import Path

# Fix import path
import sys
sys.path.insert(0, "src")

from pvx.core import voc

class TestSecurityLimits(unittest.TestCase):
    def test_stdin_too_large(self):
        # We patch the constant in the module
        with patch('pvx.core.voc.MAX_STDIN_BYTES', 100):
            with patch('sys.stdin.buffer.read') as mock_read:
                # Return data larger than limit (101 bytes)
                mock_read.return_value = b'a' * 101

                # Mock SoundFile to avoid trying to parse garbage
                with patch('pvx.core.voc.sf.SoundFile') as mock_sf:
                    # We expect ValueError from the size check, NOT from sf
                    try:
                        voc._read_audio_input(Path("-"))
                    except ValueError as e:
                        if "safe size limit" in str(e):
                            return # Passed
                        raise e

                    self.fail("Did not raise ValueError for large stdin input")

    def test_file_too_large(self):
        with patch('pvx.core.voc.MAX_DECODED_SAMPLES', 100):
            with patch('pvx.core.voc.sf.info') as mock_info:
                # Simulate a file with 100 frames, 2 channels = 200 samples
                # Limit is 100.
                mock_info.return_value.frames = 100
                mock_info.return_value.channels = 2
                mock_info.return_value.samplerate = 44100

                with patch('pvx.core.voc.sf.read') as mock_read:
                    # We assume sf.read would be called if check passed
                    mock_read.return_value = (np.zeros((100, 2)), 44100)

                    try:
                        voc._read_audio_input(Path("fake.wav"))
                    except ValueError as e:
                        if "too large to load" in str(e):
                            return
                        raise e

                    self.fail("Did not raise ValueError for large file input")

    def test_stdin_decoded_too_large(self):
        # Test case where stdin bytes are small (compressed) but decoded size is huge
        with patch('pvx.core.voc.MAX_STDIN_BYTES', 1000):
            with patch('pvx.core.voc.MAX_DECODED_SAMPLES', 100):
                with patch('sys.stdin.buffer.read') as mock_read:
                    # Small input payload
                    mock_read.return_value = b'header'

                    # Mock SoundFile to report huge decoded size
                    with patch('pvx.core.voc.sf.SoundFile') as mock_sf_cls:
                        mock_sf = mock_sf_cls.return_value
                        mock_sf.__enter__.return_value = mock_sf
                        mock_sf.frames = 100
                        mock_sf.channels = 2
                        mock_sf.samplerate = 44100

                        try:
                            voc._read_audio_input(Path("-"))
                        except ValueError as e:
                            if "Decoded audio exceeds memory limit" in str(e):
                                return
                            raise e
                        self.fail("Did not raise ValueError for large decoded stdin input")

if __name__ == '__main__':
    unittest.main()
