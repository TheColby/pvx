"""Tests for scripts_generate_docs_pdf.py helpers."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import scripts_generate_docs_pdf as pdfgen


class TestDocsPdfHelpers(unittest.TestCase):
    def test_extract_main_html_prefers_main_block(self) -> None:
        html = """
<!doctype html>
<html><head><title>Sample</title></head>
<body>
  <div>outside</div>
  <main><p>inside-main</p></main>
</body></html>
"""
        main_html = pdfgen.extract_main_html(html)
        self.assertIn("inside-main", main_html)
        self.assertNotIn("outside", main_html)

    def test_collect_html_pages_respects_core_order_and_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = root / "groups"
            groups.mkdir(parents=True, exist_ok=True)

            for name in ("windows.html", "index.html", "papers.html", "math.html"):
                (root / name).write_text("<html></html>", encoding="utf-8")
            (groups / "zeta.html").write_text("<html></html>", encoding="utf-8")
            (groups / "alpha.html").write_text("<html></html>", encoding="utf-8")

            paths = pdfgen.collect_html_pages(root, include_groups=True)
            names = [p.name for p in paths]

            self.assertEqual(names[:4], ["index.html", "math.html", "windows.html", "papers.html"])
            self.assertEqual(names[-2:], ["alpha.html", "zeta.html"])


if __name__ == "__main__":
    unittest.main()
