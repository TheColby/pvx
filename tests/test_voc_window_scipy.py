import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

# Add src to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pvx.core import voc

class TestMakeWindowScipyDispatch(unittest.TestCase):
    def test_scipy_hann_window(self):
        # Create a mock for scipy.signal
        mock_signal = MagicMock()
        # Mock get_window to return a numpy array of ones
        mock_signal.get_window.return_value = np.ones(10)

        # Patch pvx.core.voc.scipy_signal with our mock
        # We need create=True because scipy_signal might not exist yet in voc module
        with patch('pvx.core.voc.scipy_signal', mock_signal, create=True):
            # Call make_window with 'hann'
            win = voc.make_window('hann', 10, 10)

            # Verify get_window was called with correct args
            mock_signal.get_window.assert_called()
            args, kwargs = mock_signal.get_window.call_args
            # window name/tuple
            self.assertEqual(args[0], 'hann')
            # length
            self.assertEqual(args[1], 10)
            # fftbins=False for symmetric window (matching manual implementation)
            self.assertEqual(kwargs.get('fftbins'), False)

            # Verify result
            self.assertTrue(np.all(win == 1.0))

    def test_scipy_tukey_window(self):
        mock_signal = MagicMock()
        mock_signal.get_window.return_value = np.ones(10)

        with patch('pvx.core.voc.scipy_signal', mock_signal, create=True):
            win = voc.make_window('tukey_0p25', 10, 10)

            mock_signal.get_window.assert_called()
            args, kwargs = mock_signal.get_window.call_args
            self.assertEqual(args[0], ('tukey', 0.25))
            self.assertEqual(args[1], 10)
            self.assertEqual(kwargs.get('fftbins'), False)

    def test_fallback_when_scipy_fails(self):
        mock_signal = MagicMock()
        mock_signal.get_window.side_effect = ValueError("Unknown window")

        with patch('pvx.core.voc.scipy_signal', mock_signal, create=True):
            # Should fall back to manual implementation
            win = voc.make_window('hann', 10, 10)

            # Manual hann window: 0.5 - 0.5*cos(...)
            # Verify it returns a valid window (not None)
            self.assertIsInstance(win, np.ndarray)
            self.assertEqual(len(win), 10)
            # Check basic property (hann starts/ends near 0)
            self.assertAlmostEqual(win[0], 0.0)
            self.assertAlmostEqual(win[-1], 0.0)

    def test_fallback_unsupported_window(self):
        mock_signal = MagicMock()

        with patch('pvx.core.voc.scipy_signal', mock_signal, create=True):
            # exact_blackman is not mapped to scipy
            win = voc.make_window('exact_blackman', 10, 10)

            # Should use manual implementation
            self.assertIsInstance(win, np.ndarray)
            self.assertEqual(len(win), 10)

if __name__ == '__main__':
    unittest.main()
