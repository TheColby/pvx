"""Documentation coverage checks for CLI flags."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _iter_cli_sources() -> list[Path]:
    sources = [ROOT / "src" / "pvx" / "core" / "voc.py"]
    sources.extend(sorted((ROOT / "src" / "pvx" / "cli").glob("*.py")))
    return [p for p in sources if p.exists() and p.name != "__init__.py"]


def _tool_name_for_path(path: Path) -> str:
    if path.name == "voc.py":
        return "pvxvoc.py"
    if path.parent.name == "cli":
        return f"{path.stem}.py"
    return path.name


def extract_flags_from_code() -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in _iter_cli_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tool = _tool_name_for_path(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            flags = [_string_literal(arg) for arg in node.args]
            for flag in flags:
                if flag and flag.startswith("--"):
                    pairs.add((tool, flag))
    return pairs


def load_doc_pairs() -> set[tuple[str, str]]:
    payload_path = ROOT / "docs" / "cli_flags_reference.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    rows = payload.get("entries", [])
    return {(str(row["tool"]), str(row["flag"])) for row in rows}


def test_cli_flag_docs_match_parser_definitions() -> None:
    code_pairs = extract_flags_from_code()
    doc_pairs = load_doc_pairs()

    missing_in_docs = sorted(code_pairs - doc_pairs)
    stale_in_docs = sorted(doc_pairs - code_pairs)

    assert not missing_in_docs, f"Flags present in parser code but missing from docs: {missing_in_docs}"
    assert not stale_in_docs, f"Flags documented but no longer in parser code: {stale_in_docs}"


def test_readme_long_flags_exist_in_parser_sources() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_flags = set(re.findall(r"`(--[a-z0-9][a-z0-9\-]*)`", readme))

    known_flags = {flag for _, flag in extract_flags_from_code()}
    allowed = {"--help"}
    unknown = sorted(flag for flag in (readme_flags - known_flags) if flag not in allowed)

    assert not unknown, f"README references long flags not found in parser sources: {unknown}"
