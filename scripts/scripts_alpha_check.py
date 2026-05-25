#!/usr/bin/env python3

"""Run alpha-release validation checks without relying on `make`."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALPHA_PYTHON_PATHS: tuple[str, ...] = (
    "scripts/scripts_alpha_check.py",
    "src/pvx/cli/pvx.py",
    "src/pvx/cli/pvx_helpers.py",
    "src/pvx/cli/pvx_pipeline.py",
    "src/pvx/cli/pvx_augment.py",
    "src/pvx/cli/pvxvoc.py",
    "src/pvx/core/voc.py",
    "src/pvx/core/voc_parser.py",
    "src/pvx/core/voc_console.py",
    "src/pvx/core/voc_jobs.py",
    "src/pvx/core/streaming.py",
    "src/pvxalgorithms",
    "scripts/scripts_sync_homebrew_tap_formula.py",
    "tests/test_alpha_release_ready.py",
    "tests/test_augment_bench_gate.py",
    "tests/test_voc_parser.py",
    "tests/test_voc_console.py",
    "tests/test_pvxvoc_cli.py",
    "tests/test_augment_mode.py",
    "tests/test_cli_regression.py",
)

ALPHA_TEST_MODULES: tuple[str, ...] = (
    "tests.test_alpha_release_ready",
    "tests.test_augment_bench_gate",
    "tests.test_voc_parser",
    "tests.test_voc_console",
    "tests.test_pvxvoc_cli",
    "tests.test_augment_mode",
    "tests.test_cli_regression",
)

COMMANDS: tuple[list[str], ...] = (
    ["uv", "run", "ruff", "check", *ALPHA_PYTHON_PATHS],
    ["uv", "run", "python", "-m", "unittest", *ALPHA_TEST_MODULES, "-v"],
    ["uv", "run", "pytest", "-q", "tests/test_docs_coverage.py"],
)


def main() -> int:
    for cmd in COMMANDS:
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            return int(proc.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
