# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.

"""Unit tests for numeric expression evaluation."""

import unittest
import math
import sys
from pathlib import Path

# Ensure src is in path and takes precedence over root pvx.py
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pvx.core.eval import parse_numeric_expression  # noqa: E402

class TestEval(unittest.TestCase):
    def test_literals(self):
        self.assertEqual(parse_numeric_expression("1"), 1.0)
        self.assertEqual(parse_numeric_expression("2.5"), 2.5)
        self.assertEqual(parse_numeric_expression("-5"), -5.0)

    def test_basic_arithmetic(self):
        self.assertEqual(parse_numeric_expression("1 + 1"), 2.0)
        self.assertEqual(parse_numeric_expression("10 - 4"), 6.0)
        self.assertEqual(parse_numeric_expression("3 * 4"), 12.0)
        self.assertEqual(parse_numeric_expression("12 / 3"), 4.0)
        self.assertEqual(parse_numeric_expression("2 ** 3"), 8.0)
        self.assertEqual(parse_numeric_expression("2 ^ 3"), 8.0)  # caret replacement

    def test_operator_precedence(self):
        self.assertEqual(parse_numeric_expression("1 + 2 * 3"), 7.0)
        self.assertEqual(parse_numeric_expression("(1 + 2) * 3"), 9.0)

    def test_constants(self):
        self.assertAlmostEqual(parse_numeric_expression("pi"), math.pi)
        self.assertAlmostEqual(parse_numeric_expression("e"), math.e)
        self.assertAlmostEqual(parse_numeric_expression("tau"), math.tau)

    def test_functions(self):
        self.assertAlmostEqual(parse_numeric_expression("sqrt(4)"), 2.0)
        self.assertAlmostEqual(parse_numeric_expression("log10(100)"), 2.0)
        self.assertAlmostEqual(parse_numeric_expression("sin(pi/2)"), 1.0)
        self.assertAlmostEqual(parse_numeric_expression("cos(0)"), 1.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            parse_numeric_expression("")
        with self.assertRaises(ValueError):
            parse_numeric_expression("True")
        with self.assertRaises(ValueError):
            parse_numeric_expression("1 / 0")
        with self.assertRaises(ValueError):
            parse_numeric_expression("unknown_func(1)")
        with self.assertRaises(ValueError):
            parse_numeric_expression("unknown_var")
        with self.assertRaises(ValueError):
            parse_numeric_expression("'string'")

if __name__ == "__main__":
    unittest.main()
