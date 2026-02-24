#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
"""Centralized attribution text shared across pvx code and documentation."""

from __future__ import annotations

COPYRIGHT_NOTICE = "Copyright (c) 2026 Colby Leider and contributors."
ATTRIBUTION_DOC_PATH = "ATTRIBUTION.md"


def python_header_reference() -> str:
    """Return the canonical one-line Python header reference."""
    return f"# {COPYRIGHT_NOTICE} See {ATTRIBUTION_DOC_PATH}."


def markdown_notice(relative_attribution_path: str) -> str:
    """Return a Markdown attribution notice line with a relative link."""
    return f"> {COPYRIGHT_NOTICE} See [{ATTRIBUTION_DOC_PATH}]({relative_attribution_path})."


def html_notice(relative_attribution_path: str) -> str:
    """Return an HTML attribution notice sentence with a relative link."""
    return (
        f"{COPYRIGHT_NOTICE} "
        f'See <a href="{relative_attribution_path}"><code>{ATTRIBUTION_DOC_PATH}</code></a>.'
    )
