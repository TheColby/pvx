#!/usr/bin/env python3
# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
"""Validate that runtime requirements.txt matches project runtime dependencies."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"


def normalize_name(spec: str) -> str:
    name = (
        re.split(r"[<>=!~;\[]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")
    )
    return name


def read_requirements() -> set[str]:
    names: set[str] = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r ") or line.startswith("--"):
            continue
        names.add(normalize_name(line))
    return names


def read_pyproject_runtime_dependencies() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    names = {normalize_name(str(dep)) for dep in deps}
    return names


def main() -> int:
    req = read_requirements()
    project = read_pyproject_runtime_dependencies()

    missing_in_requirements = sorted(project - req)
    extra_in_requirements = sorted(req - project)

    if not missing_in_requirements and not extra_in_requirements:
        print("[ok] runtime dependency sync: requirements.txt matches pyproject.toml")
        return 0

    if missing_in_requirements:
        print(
            "[error] missing in requirements.txt:", ", ".join(missing_in_requirements)
        )
    if extra_in_requirements:
        print("[error] extra in requirements.txt:", ", ".join(extra_in_requirements))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
