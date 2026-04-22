"""CLI entrypoint wrapper for the phase-vocoder tool."""

from __future__ import annotations

from pvx.voc_cli import build_parser, expand_inputs, main, run_guided_mode

__all__ = ["build_parser", "expand_inputs", "main", "run_guided_mode"]


if __name__ == "__main__":
    raise SystemExit(main())
