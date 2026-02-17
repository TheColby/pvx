#!/usr/bin/env python3
"""Generate comprehensive documentation for every Python file in the repository."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from textwrap import indent

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

PY_FILES = sorted(
    p for p in ROOT.rglob("*.py") if ".venv" not in p.parts and "__pycache__" not in p.parts
)

CLI_HELP_CANDIDATES = {
    ROOT / "main.py",
    ROOT / "pvxvoc.py",
    ROOT / "pvxfreeze.py",
    ROOT / "pvxharmonize.py",
    ROOT / "pvxconform.py",
    ROOT / "pvxmorph.py",
    ROOT / "pvxwarp.py",
    ROOT / "pvxformant.py",
    ROOT / "pvxtransient.py",
    ROOT / "pvxunison.py",
    ROOT / "pvxdenoise.py",
    ROOT / "pvxdeverb.py",
    ROOT / "pvxretune.py",
    ROOT / "pvxlayer.py",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_module(path: Path) -> dict:
    src = safe_read(path)
    tree = ast.parse(src)
    module_doc = ast.get_docstring(tree) or ""
    functions = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]
    classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]

    has_main = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in tree.body
    )

    uses_argparse = "argparse.ArgumentParser" in src
    has_process = "def process(" in src
    algorithm_id = None
    theme = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ALGORITHM_ID" and isinstance(node.value, ast.Constant):
                    algorithm_id = str(node.value.value)
                if isinstance(target, ast.Name) and target.id == "THEME" and isinstance(node.value, ast.Constant):
                    theme = str(node.value.value)

    return {
        "doc": module_doc.strip(),
        "functions": functions,
        "classes": classes,
        "has_main": has_main,
        "uses_argparse": uses_argparse,
        "has_process": has_process,
        "algorithm_id": algorithm_id,
        "theme": theme,
    }


def cli_help(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["python3", str(path), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=25,
            check=False,
        )
    except Exception as exc:  # pragma: no cover
        return f"[help unavailable: {exc}]"

    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    out = out.strip()
    if not out:
        return None
    if len(out) > 5000:
        out = out[:5000] + "\n... [truncated]"
    return out


def extract_algorithm_params(base_path: Path) -> dict[str, list[str]]:
    text = safe_read(base_path)
    mapping: dict[str, list[str]] = {}
    lines = text.splitlines()

    current_slug = None
    bucket: list[str] = []

    def commit() -> None:
        nonlocal current_slug, bucket
        if current_slug is None:
            return
        dedup = []
        for item in bucket:
            if item not in dedup:
                dedup.append(item)
        mapping[current_slug] = dedup

    for line in lines:
        line_s = line.strip()
        if line_s.startswith('if slug == "') or line_s.startswith('elif slug == "'):
            if current_slug is not None:
                commit()
            current_slug = line_s.split('"')[1]
            bucket = []
        if "params.get(" in line_s and current_slug is not None:
            try:
                name = line_s.split("params.get(", 1)[1].split(")", 1)[0]
                key = name.split(",", 1)[0].strip().strip('"').strip("'")
                if key:
                    bucket.append(key)
            except Exception:
                continue
    if current_slug is not None:
        commit()
    return mapping


def generate_algorithm_param_doc() -> None:
    params = extract_algorithm_params(ROOT / "pvxalgorithms" / "base.py")
    lines: list[str] = []
    lines.append("# PVX Algorithm Parameter Reference")
    lines.append("")
    lines.append("This file lists per-algorithm parameter keys consumed by `pvxalgorithms.base.run_algorithm()` dispatch.")
    lines.append("Use these keys as `**params` when calling module `process(audio, sample_rate, **params)`. ")
    lines.append("")
    for slug in sorted(params):
        lines.append(f"## `{slug}`")
        keys = params[slug]
        if not keys:
            lines.append("- No algorithm-specific keys (uses generic/default path).")
        else:
            for key in keys:
                lines.append(f"- `{key}`")
        lines.append("")

    (DOCS_DIR / "PVX_ALGORITHM_PARAMS.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_python_help_doc() -> None:
    lines: list[str] = []
    lines.append("# Python File Documentation and Help")
    lines.append("")
    lines.append("Comprehensive reference for every Python file in this repository.")
    lines.append("")
    lines.append(f"Total Python files documented: **{len(PY_FILES)}**")
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    for path in PY_FILES:
        anchor = rel(path).replace("/", "").replace(".", "").replace("_", "")
        lines.append(f"- [`{rel(path)}`](#{anchor})")
    lines.append("")

    for path in PY_FILES:
        info = parse_module(path)
        title = rel(path)
        lines.append(f"## `{title}`")
        lines.append("")

        summary = info["doc"].splitlines()[0] if info["doc"] else "No module docstring present."
        lines.append(f"**Purpose:** {summary}")
        lines.append("")

        if info["algorithm_id"]:
            lines.append(f"**Algorithm ID:** `{info['algorithm_id']}`")
            lines.append(f"**Theme:** `{info['theme']}`")
            lines.append("**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`")
            lines.append("**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.")
            lines.append("")

        lines.append(f"**Classes:** {', '.join('`'+c+'`' for c in info['classes']) if info['classes'] else 'None'}")
        lines.append(f"**Functions:** {', '.join('`'+f+'`' for f in info['functions']) if info['functions'] else 'None'}")
        lines.append("")

        help_cmds = []
        if path in CLI_HELP_CANDIDATES:
            help_cmds.append(f"`python3 {title} --help`")
        elif info["has_main"]:
            help_cmds.append(f"`python3 {title}`")
            if info["uses_argparse"]:
                help_cmds.append(f"`python3 {title} --help`")

        if help_cmds:
            lines.append("**Help commands:** " + ", ".join(help_cmds))
            lines.append("")

        if path in CLI_HELP_CANDIDATES:
            h = cli_help(path)
            if h:
                lines.append("### CLI Help Snapshot")
                lines.append("")
                lines.append("```text")
                lines.append(h)
                lines.append("```")
                lines.append("")

        if info["doc"]:
            lines.append("### Module Docstring")
            lines.append("")
            lines.append("```text")
            lines.append(info["doc"])
            lines.append("```")
            lines.append("")

    out = "\n".join(lines).rstrip() + "\n"
    (DOCS_DIR / "PYTHON_FILE_HELP.md").write_text(out, encoding="utf-8")


def main() -> int:
    generate_algorithm_param_doc()
    generate_python_help_doc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
