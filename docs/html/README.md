# pvx HyperText Markup Language (HTML) Documentation

GitHub repository view generally displays `.html` files as source text rather than rendering them as web pages.

## Acronym Primer

- HyperText Markup Language (HTML)
- Portable Document Format (PDF)
- command-line interface (CLI)
- digital signal processing (DSP)
- short-time Fourier transform (STFT)

Use this folder in one of these ways:

- Open `docs/html/index.html` in a local browser.
- Publish `docs/html/` through GitHub Pages (or another static host).

For GitHub-native math rendering in Markdown, use:

- [`../MATHEMATICAL_FOUNDATIONS.md`](../MATHEMATICAL_FOUNDATIONS.md)
- [`../WINDOW_REFERENCE.md`](../WINDOW_REFERENCE.md)

Generated files in this folder:

- `index.html`: grouped algorithm docs landing page.
- `groups/*.html`: per-folder algorithm pages.
- `papers.html`: research bibliography.
- `glossary.html`: technical glossary.
- `math.html`: mathematical foundations.
- `windows.html`: complete window reference.
- `architecture.html`: rendered Mermaid architecture diagrams.
- `limitations.html`: algorithm assumptions/failure-mode guidance.
- `benchmarks.html`: reproducible benchmark report.
- `cookbook.html`: command cookbook (one-liners and pipelines).
- `cli_flags.html`: parser-derived CLI flag reference.
- `citations.html`: citation-quality report and upgrade targets.

Single-file PDF export (all HTML docs combined):

```bash
python3 scripts_generate_docs_pdf.py --output docs/pvx_documentation.pdf
```
