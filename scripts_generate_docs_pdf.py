#!/usr/bin/env python3
"""Generate one combined PDF from all HTML documentation pages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable


@dataclass(frozen=True)
class SourcePage:
    path: Path
    title: str
    main_html: str


VERBOSITY_LEVELS = ("silent", "quiet", "normal", "verbose", "debug")
_VERBOSITY_TO_LEVEL = {name: idx for idx, name in enumerate(VERBOSITY_LEVELS)}


class ProgressBar:
    def __init__(self, label: str, enabled: bool, width: int = 32) -> None:
        self.label = label
        self.enabled = enabled
        self.width = max(10, width)
        self._finished = False
        if self.enabled:
            self.set(0.0, "start")

    def set(self, fraction: float, detail: str = "") -> None:
        if not self.enabled or self._finished:
            return
        frac = max(0.0, min(1.0, fraction))
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {detail}" if detail else ""
        sys.stderr.write(f"\r[{bar}] {frac * 100:6.2f}% {self.label}{suffix}")
        sys.stderr.flush()
        if frac >= 1.0:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._finished = True


def add_console_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbosity",
        choices=list(VERBOSITY_LEVELS),
        default="normal",
        help="Console verbosity level",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeat for extra detail)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output and hide status bars")
    parser.add_argument("--silent", action="store_true", help="Suppress all console output")


def console_level(args: argparse.Namespace) -> int:
    base_level = _VERBOSITY_TO_LEVEL.get(str(getattr(args, "verbosity", "normal")), _VERBOSITY_TO_LEVEL["normal"])
    verbose_count = int(getattr(args, "verbose", 0) or 0)
    level = min(_VERBOSITY_TO_LEVEL["debug"], base_level + verbose_count)
    if bool(getattr(args, "quiet", False)):
        level = min(level, _VERBOSITY_TO_LEVEL["quiet"])
    if bool(getattr(args, "silent", False)):
        level = _VERBOSITY_TO_LEVEL["silent"]
    return level


def is_quiet(args: argparse.Namespace) -> bool:
    return console_level(args) <= _VERBOSITY_TO_LEVEL["quiet"]


def is_silent(args: argparse.Namespace) -> bool:
    return console_level(args) == _VERBOSITY_TO_LEVEL["silent"]


def log(args: argparse.Namespace, message: str, *, min_level: str = "normal", error: bool = False) -> None:
    if console_level(args) < _VERBOSITY_TO_LEVEL[min_level]:
        return
    print(message, file=sys.stderr if error else sys.stdout)


CORE_ORDER = [
    "index.html",
    "architecture.html",
    "math.html",
    "windows.html",
    "glossary.html",
    "papers.html",
    "limitations.html",
    "benchmarks.html",
    "cookbook.html",
    "cli_flags.html",
    "citations.html",
]


def html_sort_key(path: Path, docs_html_dir: Path) -> tuple[int, str]:
    rel = path.relative_to(docs_html_dir)
    if rel.parts and rel.parts[0] == "groups":
        return (200, str(rel))
    if rel.name in CORE_ORDER:
        return (CORE_ORDER.index(rel.name), rel.name)
    return (100, str(rel))


def collect_html_pages(docs_html_dir: Path, include_groups: bool) -> list[Path]:
    pages: list[Path] = []
    for path in docs_html_dir.glob("*.html"):
        if path.name == "style.css":
            continue
        pages.append(path)

    if include_groups:
        pages.extend(sorted((docs_html_dir / "groups").glob("*.html")))

    pages = [p for p in pages if p.is_file()]
    pages.sort(key=lambda p: html_sort_key(p, docs_html_dir))
    return pages


def extract_title(html_text: str, fallback: str) -> str:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        return re.sub(r"\s+", " ", title_match.group(1)).strip() or fallback
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if h1_match:
        value = re.sub(r"<[^>]+>", "", h1_match.group(1))
        return re.sub(r"\s+", " ", value).strip() or fallback
    return fallback


def extract_main_html(html_text: str) -> str:
    main_match = re.search(r"<main\b[^>]*>(.*?)</main>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if main_match:
        return main_match.group(1)
    body_match = re.search(r"<body\b[^>]*>(.*?)</body>", html_text, flags=re.IGNORECASE | re.DOTALL)
    if body_match:
        return body_match.group(1)
    return html_text


def parse_source_page(path: Path) -> SourcePage:
    text = path.read_text(encoding="utf-8")
    title = extract_title(text, fallback=path.stem)
    main_html = extract_main_html(text)
    return SourcePage(path=path, title=title, main_html=main_html)


def build_combined_html(pages: list[SourcePage], docs_html_dir: Path) -> str:
    style_path = docs_html_dir / "style.css"
    external_css = style_path.read_text(encoding="utf-8") if style_path.exists() else ""

    sections: list[str] = []
    toc_rows = "".join(
        f"<li><strong>{idx + 1:02d}.</strong> {escape(page.title)} <span class=\"src\">({escape(str(page.path.relative_to(docs_html_dir)))})</span></li>"
        for idx, page in enumerate(pages)
    )

    cover = (
        "<section class=\"doc-cover\">"
        "<h1>PVX Complete HTML Documentation</h1>"
        f"<p>Total HTML pages included: <strong>{len(pages)}</strong></p>"
        "<p class=\"small\">Generated by <code>scripts_generate_docs_pdf.py</code>.</p>"
        "<h2>Contents</h2>"
        f"<ol>{toc_rows}</ol>"
        "</section>"
    )
    sections.append(cover)

    for idx, page in enumerate(pages):
        sections.append(
            "<section class=\"doc-section\">"
            f"<header class=\"doc-header\"><h1>{escape(page.title)}</h1><p class=\"src\">Source: {escape(str(page.path.relative_to(docs_html_dir)))}</p></header>"
            f"<div class=\"doc-main\">{page.main_html}</div>"
            "</section>"
        )
        if idx != len(pages) - 1:
            sections.append("<div class=\"pdf-page-break\"></div>")

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>PVX Documentation (Combined PDF)</title>
  <style>
    {external_css}

    @page {{
      size: A4;
      margin: 12mm;
    }}
    body {{
      font-family: Arial, Helvetica, sans-serif;
      color: #122635;
      line-height: 1.4;
    }}
    .doc-cover {{
      page-break-after: always;
    }}
    .doc-cover ol {{
      columns: 2;
      column-gap: 24px;
    }}
    .doc-cover li {{
      margin-bottom: 6px;
    }}
    .doc-header h1 {{
      margin-bottom: 2px;
      font-size: 1.35rem;
    }}
    .src {{
      color: #4b5f6d;
      font-size: 0.9rem;
    }}
    .doc-main .site-header,
    .doc-main .site-footer,
    .doc-main nav {{
      display: none !important;
    }}
    .doc-main table {{
      page-break-inside: auto;
      white-space: normal !important;
    }}
    .doc-main tr,
    .doc-main td,
    .doc-main th,
    .doc-main img,
    .doc-main pre,
    .doc-main code {{
      page-break-inside: avoid;
    }}
    .doc-main pre,
    .doc-main code {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .pdf-page-break {{
      page-break-after: always;
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
  <script type=\"module\">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, securityLevel: 'loose' }});
  </script>
</head>
<body>
  {''.join(sections)}
</body>
</html>
"""


def discover_chromium_executable() -> str | None:
    candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
        "msedge",
    ]
    for cmd in candidates:
        path = shutil.which(cmd)
        if path:
            return path

    mac_candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ]
    for path in mac_candidates:
        if Path(path).exists():
            return path
    return None


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def render_pdf_with_chromium(html_path: Path, output_pdf: Path, executable: str) -> None:
    uri = html_path.resolve().as_uri()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    common = [
        executable,
        "--disable-gpu",
        "--no-sandbox",
        "--allow-file-access-from-files",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={str(output_pdf.resolve())}",
        uri,
    ]

    attempts = [
        [executable, "--headless=new", *common[1:]],
        [executable, "--headless", *common[1:]],
    ]

    last_err = ""
    for cmd in attempts:
        code, _, err = run_cmd(cmd)
        if code == 0 and output_pdf.exists() and output_pdf.stat().st_size > 0:
            return
        last_err = err

    raise RuntimeError(f"Chromium PDF export failed. stderr: {last_err.strip()}")


def render_pdf_with_wkhtmltopdf(html_path: Path, output_pdf: Path, executable: str) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        executable,
        "--enable-local-file-access",
        "--print-media-type",
        str(html_path),
        str(output_pdf),
    ]
    code, _, err = run_cmd(cmd)
    if code != 0 or not output_pdf.exists() or output_pdf.stat().st_size == 0:
        raise RuntimeError(f"wkhtmltopdf export failed. stderr: {err.strip()}")


def render_pdf_with_weasyprint(html_path: Path, output_pdf: Path) -> None:
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"WeasyPrint unavailable: {exc}") from exc

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(filename=str(html_path), base_url=str(html_path.parent)).write_pdf(str(output_pdf))
    if not output_pdf.exists() or output_pdf.stat().st_size == 0:
        raise RuntimeError("WeasyPrint produced no output PDF.")


def render_pdf_with_playwright(html_path: Path, output_pdf: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Playwright unavailable: {exc}") from exc

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    uri = html_path.resolve().as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(uri, wait_until="networkidle")
            page.wait_for_timeout(1200)
            page.pdf(
                path=str(output_pdf),
                format="A4",
                print_background=True,
                margin={"top": "12mm", "right": "12mm", "bottom": "12mm", "left": "12mm"},
            )
        finally:
            browser.close()

    if not output_pdf.exists() or output_pdf.stat().st_size == 0:
        raise RuntimeError("Playwright produced no output PDF.")


EngineFunc = Callable[[Path, Path], None]


def build_engine_registry() -> dict[str, EngineFunc]:
    engines: dict[str, EngineFunc] = {}

    chrome_exec = discover_chromium_executable()
    if chrome_exec:
        engines["chromium"] = lambda html_path, out_pdf: render_pdf_with_chromium(html_path, out_pdf, chrome_exec)

    wk_exec = shutil.which("wkhtmltopdf")
    if wk_exec:
        engines["wkhtmltopdf"] = lambda html_path, out_pdf: render_pdf_with_wkhtmltopdf(html_path, out_pdf, wk_exec)

    engines["weasyprint"] = render_pdf_with_weasyprint
    engines["playwright"] = render_pdf_with_playwright
    return engines


def auto_engine_order() -> list[str]:
    return ["chromium", "wkhtmltopdf", "weasyprint", "playwright"]


def render_pdf(html_path: Path, output_pdf: Path, engine: str, args: argparse.Namespace) -> str:
    registry = build_engine_registry()

    if engine != "auto":
        renderer = registry.get(engine)
        if renderer is None:
            available = ", ".join(sorted(registry)) or "none"
            raise RuntimeError(f"Requested engine '{engine}' unavailable. Detected engines: {available}")
        renderer(html_path, output_pdf)
        return engine

    last_exc: Exception | None = None
    for candidate in auto_engine_order():
        renderer = registry.get(candidate)
        if renderer is None:
            continue
        try:
            log(args, f"[info] trying engine: {candidate}", min_level="verbose")
            renderer(html_path, output_pdf)
            return candidate
        except Exception as exc:  # pragma: no cover - depends on host tools
            last_exc = exc
            log(args, f"[warn] engine {candidate} failed: {exc}", min_level="verbose", error=True)
            continue

    available = ", ".join(sorted(registry)) or "none"
    message = (
        "Unable to render PDF with auto engine selection. "
        f"Detected engines: {available}. "
        "Install one of: Chromium/Chrome, wkhtmltopdf, WeasyPrint, or Playwright+Chromium."
    )
    if last_exc is not None:
        message += f" Last error: {last_exc}"
    raise RuntimeError(message)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a single PDF from all HTML docs")
    parser.add_argument(
        "--docs-html-dir",
        type=Path,
        default=Path("docs/html"),
        help="Directory containing HTML documentation pages",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/pvx_documentation.pdf"),
        help="Output PDF file path",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "chromium", "wkhtmltopdf", "weasyprint", "playwright"],
        default="auto",
        help="PDF rendering engine",
    )
    parser.add_argument(
        "--no-groups",
        action="store_true",
        help="Exclude docs/html/groups/*.html pages from the combined PDF",
    )
    parser.add_argument(
        "--keep-combined-html",
        action="store_true",
        help="Keep intermediate combined HTML next to output PDF for inspection",
    )
    add_console_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    docs_html_dir = args.docs_html_dir.resolve()
    output_pdf = args.output.resolve()
    include_groups = not bool(args.no_groups)

    if not docs_html_dir.exists():
        log(args, f"[error] HTML docs directory not found: {docs_html_dir}", error=True)
        return 2

    html_paths = collect_html_pages(docs_html_dir, include_groups=include_groups)
    if not html_paths:
        log(args, f"[error] No HTML pages found in {docs_html_dir}", error=True)
        return 2

    progress = ProgressBar("build combined docs html", enabled=not is_quiet(args))

    pages: list[SourcePage] = []
    total = len(html_paths)
    for idx, path in enumerate(html_paths, start=1):
        pages.append(parse_source_page(path))
        progress.set(idx / float(total), f"{path.name}")

    combined_html = build_combined_html(pages, docs_html_dir)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    keep_combined_html = bool(args.keep_combined_html)
    combined_keep_path = output_pdf.with_suffix(".combined.html")

    try:
        with tempfile.TemporaryDirectory(prefix="pvx_docs_pdf_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            combined_path = tmp_path / "combined_docs.html"
            combined_path.write_text(combined_html, encoding="utf-8")

            progress_render = ProgressBar("render pdf", enabled=not is_quiet(args))
            progress_render.set(0.2, args.engine)
            used_engine = render_pdf(combined_path, output_pdf, engine=args.engine, args=args)
            progress_render.set(1.0, used_engine)

            if keep_combined_html:
                combined_keep_path.write_text(combined_html, encoding="utf-8")

    except Exception as exc:
        log(args, f"[error] PDF generation failed: {exc}", error=True)
        return 1

    if not output_pdf.exists() or output_pdf.stat().st_size == 0:
        log(args, f"[error] Output PDF was not created: {output_pdf}", error=True)
        return 1

    if not is_silent(args):
        log(args, f"[ok] wrote {output_pdf}")
        log(args, f"[ok] included {len(pages)} HTML pages from {docs_html_dir}")
        if keep_combined_html:
            log(args, f"[ok] kept combined HTML: {combined_keep_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
