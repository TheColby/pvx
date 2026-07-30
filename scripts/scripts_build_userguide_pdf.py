#!/usr/bin/env python3

"""Build a book-length USERGUIDE.pdf for pvx from curated repository docs."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT / "tmp" / "pdfs"
OUTPUT_DIR = ROOT / "output" / "pdf"
PREAMBLE = ROOT / "docs" / "userguide_preamble.tex"
INLINE_CODE_FILTER = ROOT / "scripts" / "userguide_inline_code.lua"


@dataclass(frozen=True)
class ChapterSpec:
    source: Path
    title: str | None = None
    strip_quick_setup: bool = True


@dataclass(frozen=True)
class WindowSpec:
    name: str
    family: str
    parameters: str
    formula: str
    coherent_gain: str
    enbw: str
    scalloping: str
    main_lobe: str
    sidelobe: str
    strengths: str
    limitations: str
    advice: str


BOOK_PARTS: tuple[tuple[str, tuple[ChapterSpec, ...]], ...] = (
    (
        "Orientation and Core Workflow",
        (
            ChapterSpec(ROOT / "docs" / "GETTING_STARTED.md", strip_quick_setup=False),
            ChapterSpec(ROOT / "docs" / "QUALITY_GUIDE.md"),
            ChapterSpec(ROOT / "docs" / "SUPPORTED_SURFACE.md"),
            ChapterSpec(ROOT / "docs" / "FILE_TYPES.md"),
        ),
    ),
    (
        "Theory and Internal Model",
        (
            ChapterSpec(ROOT / "docs" / "MATHEMATICAL_FOUNDATIONS.md"),
            ChapterSpec(ROOT / "docs" / "ARCHITECTURE.md"),
            ChapterSpec(ROOT / "docs" / "WINDOW_REFERENCE.md"),
            ChapterSpec(ROOT / "docs" / "PVC_LESSONS.md"),
            ChapterSpec(ROOT / "docs" / "PVC_PARITY_MATRIX.md"),
        ),
    ),
    (
        "Workflow Cookbook",
        (
            ChapterSpec(ROOT / "docs" / "EXAMPLES.md"),
            ChapterSpec(ROOT / "docs" / "FEATURE_SIDECHAIN_EXAMPLES.md"),
            ChapterSpec(ROOT / "docs" / "FOLLOW_MIGRATION.md"),
            ChapterSpec(ROOT / "docs" / "CRAZY_100.md"),
        ),
    ),
    (
        "Python, ML, and Dataset Work",
        (
            ChapterSpec(ROOT / "docs" / "API_OVERVIEW.md"),
            ChapterSpec(ROOT / "docs" / "ML_INTEGRATION.md"),
            ChapterSpec(ROOT / "docs" / "AI_AUGMENTATION.md"),
            ChapterSpec(ROOT / "docs" / "AUGMENTATION_COOKBOOK.md"),
            ChapterSpec(ROOT / "docs" / "PIPELINE_COOKBOOK.md"),
        ),
    ),
    (
        "Operations and Delivery",
        (
            ChapterSpec(ROOT / "docs" / "BENCHMARKS.md"),
            ChapterSpec(ROOT / "docs" / "HOMEBREW.md"),
            ChapterSpec(ROOT / "docs" / "ALPHA_RELEASE.md"),
        ),
    ),
)

APPENDICES: tuple[ChapterSpec, ...] = (
    ChapterSpec(ROOT / "docs" / "CLI_FLAGS_REFERENCE.md", strip_quick_setup=False),
    ChapterSpec(ROOT / "docs" / "ALGORITHM_LIMITATIONS.md"),
    ChapterSpec(ROOT / "docs" / "DIAGRAMS.md"),
    ChapterSpec(ROOT / "docs" / "PHASINESS_IMPLEMENTATION_PLAN.md"),
    ChapterSpec(ROOT / "docs" / "CITATION_QUALITY.md"),
    ChapterSpec(ROOT / "docs" / "PVC_PHASE3_5_ARCHITECTURE.md"),
)

HTML_IMAGE_BLOCK_RE = re.compile(r"<p\s+align=\"center\">.*?</p>\s*", re.DOTALL | re.IGNORECASE)
IMG_TAG_RE = re.compile(r"<img\b[^>]*>\s*", re.IGNORECASE)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)(?:\{[^}]*\})?")
ATTRIBUTION_RE = re.compile(r"\n## Attribution\b.*$", re.DOTALL)
MERMAID_RE = re.compile(r"```mermaid.*?```", re.DOTALL)
LOCAL_LINK_RE = re.compile(r"\[([^\]]+)\]\(((?!https?://|mailto:)[^)]+)\)")
SIMPLE_NUMBERED_HEADING_RE = re.compile(r"^(#{1,6}\s+)\d+(?:\.\d+)*(?:[.)])?\s+(.*)$")
RANGE_HEADING_RE = re.compile(r"^(#{1,6}\s+)\d+\s*-\s*\d+:\s+(.*)$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
DISPLAY_MATH_LINE_RE = re.compile(r"^\s*\$\$\s*$")


def index_key(value: str) -> str:
    value = re.sub(r"`|\*\*|__|\*|_", " ", value)
    value = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", value)
    value = re.sub(r"[^A-Za-z0-9+./() -]+", " ", value)
    return re.sub(r"\s+", " ", value).strip(" .-")


def ensure_list_introductions(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for idx, line in enumerate(lines):
        out.append(line)
        if not line.lstrip().startswith("#"):
            continue
        cursor = idx + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines) and LIST_ITEM_RE.match(lines[cursor]):
            out.extend(
                (
                    "",
                    "This section begins with the following points, which establish the context for the details that follow.",
                )
            )
    return "\n".join(out).strip() + "\n"


def ensure_equation_where_clauses(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    in_math = False
    for idx, line in enumerate(lines):
        out.append(line)
        if not DISPLAY_MATH_LINE_RE.match(line):
            continue
        if not in_math:
            in_math = True
            continue
        in_math = False
        cursor = idx + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        next_text = lines[cursor].strip().lower() if cursor < len(lines) else ""
        if not next_text.startswith("where"):
            out.extend(
                (
                    "",
                    "where the symbols retain the meanings established in this section and subscripts identify samples, frames, channels, or spectral bins as applicable.",
                )
            )
    return "\n".join(out).strip() + "\n"


def add_index_entries(text: str) -> str:
    lines = text.splitlines()
    indexed: list[str] = []
    terms: set[str] = set()
    for line in lines:
        indexed.append(line)
        if line.lstrip().startswith("#"):
            heading = index_key(line.lstrip("#").strip())
            if heading:
                indexed.append(rf"\index{{{heading}}}")
        terms.update(re.findall(r"--[a-z0-9][a-z0-9-]*", line))
        terms.update(re.findall(r"\bpvx[a-z0-9-]*\b", line))
    if terms:
        indexed.append("")
        for term in sorted(terms, key=str.casefold):
            category = "CLI options" if term.startswith("--") else "commands"
            indexed.append(rf"\index{{{category}!{term}}}")
    return "\n".join(indexed).strip() + "\n"


WINDOW_FORMULAE: dict[str, tuple[str, str]] = {
    "W0": (
        r"w[n]=1",
        r"where \(w[n]\) is the window coefficient and \(n\) is any sample index from \(0\) through \(N-1\).",
    ),
    "W1": (
        r"w[n]=\sum_{k=0}^{K}a_k\cos\left(\frac{2\pi kn}{N-1}\right)",
        r"where \(a_k\) is cosine coefficient \(k\), \(K\) is the highest cosine order, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W2": (
        r"w[n]=\sin\left(\frac{\pi n}{N-1}\right)",
        r"where \(n\) is the sample index and \(N\) is the window length.",
    ),
    "W3": (
        r"w[n]=1-\left|\frac{n-m}{m}\right|",
        r"where \(m=(N-1)/2\) is the center sample, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W4": (
        r"w[n]=\max\left(1-\left|\frac{n-m}{(N+1)/2}\right|,0\right)",
        r"where \(m=(N-1)/2\) is the center sample, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W5": (
        r"x=\frac{n}{N-1}-\frac12,\qquad w[n]=0.62-0.48|x|+0.38\cos(2\pi x)",
        r"where \(x\) is the centered normalized position, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W6": (
        r"w[n]=\begin{cases}\frac12\left[1+\cos\left(\pi\left(\frac{2x}{\alpha}-1\right)\right)\right],&0\le x<\alpha/2\\1,&\alpha/2\le x<1-\alpha/2\\\frac12\left[1+\cos\left(\pi\left(\frac{2x}{\alpha}-\frac{2}{\alpha}+1\right)\right)\right],&1-\alpha/2\le x\le1\end{cases}",
        r"where \(x=n/(N-1)\) is normalized position, \(\alpha\) is the tapered fraction, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W7": (
        r"w[n]=\begin{cases}1-6u^2+6u^3,&0\le u\le1/2\\2(1-u)^3,&1/2<u\le1\\0,&u>1\end{cases}",
        r"where \(u=\left|2n/(N-1)-1\right|\) is absolute centered position, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W8": (
        r"w[n]=\operatorname{sinc}\left(\frac{2n}{N-1}-1\right)",
        r"where \(\operatorname{sinc}(x)=\sin(\pi x)/(\pi x)\), \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W9": (
        r"w[n]=\max(1-x_n^2,0)",
        r"where \(x_n=(n-m)/m\) is centered normalized position, \(m=(N-1)/2\), and \(N\) is the window length.",
    ),
    "W10": (
        r"w[n]=\exp\left[-\frac12\left(\frac{n-m}{\sigma}\right)^2\right],\qquad \sigma=r_\sigma m",
        r"where \(\sigma\) is spread, \(r_\sigma\) is the configured spread ratio, \(m=(N-1)/2\), and \(n\) is the sample index.",
    ),
    "W11": (
        r"w[n]=\exp\left[-\frac12\left|\frac{n-m}{\sigma}\right|^{2p}\right]",
        r"where \(p\) controls shoulder shape, \(\sigma\) controls spread, \(m=(N-1)/2\), and \(n\) is the sample index.",
    ),
    "W12": (
        r"w[n]=\exp\left(-\frac{|n-m|}{\tau}\right),\qquad \tau=r_\tau m",
        r"where \(\tau\) is the decay constant, \(r_\tau\) is its configured ratio, \(m=(N-1)/2\), and \(n\) is the sample index.",
    ),
    "W13": (
        r"w[n]=\frac{1}{1+\left(\frac{n-m}{\gamma}\right)^2},\qquad \gamma=r_\gamma m",
        r"where \(\gamma\) is the Lorentzian scale, \(r_\gamma\) is its configured ratio, \(m=(N-1)/2\), and \(n\) is the sample index.",
    ),
    "W14": (
        r"w[n]=\sin^p\left(\frac{\pi n}{N-1}\right)",
        r"where \(p\) is the cosine-power exponent, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W15": (
        r"w[n]=w_{\mathrm{Hann}}[n]\exp\left(-\alpha\frac{|n-m|}{m}\right)",
        r"where \(w_{\mathrm{Hann}}[n]\) is the Hann window, \(\alpha\) controls exponential decay, and \(m=(N-1)/2\).",
    ),
    "W16": (
        r"w[n]=\alpha-(1-\alpha)\cos\left(\frac{2\pi n}{N-1}\right)",
        r"where \(\alpha\) sets the constant-to-cosine balance, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W17": (
        r"w[n]=(1-x)\cos(\pi x)+\frac{\sin(\pi x)}{\pi}",
        r"where \(x=\left|2n/(N-1)-1\right|\) is absolute centered position, \(n\) is the sample index, and \(N\) is the window length.",
    ),
    "W18": (
        r"w[n]=\frac{I_0\left(\beta\sqrt{1-r_n^2}\right)}{I_0(\beta)},\qquad r_n=\frac{n-m}{m}",
        r"where \(I_0\) is the modified Bessel function, \(\beta\) controls taper strength, \(m=(N-1)/2\), and \(n\) is the sample index.",
    ),
}


def parse_window_specs() -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for line in (ROOT / "docs" / "WINDOW_REFERENCE.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 13:
            continue
        specs.append(
            WindowSpec(
                name=fields[0].strip("`"),
                family=fields[1],
                parameters=fields[2].strip("`"),
                formula=fields[3],
                coherent_gain=fields[4],
                enbw=fields[5],
                scalloping=fields[6],
                main_lobe=fields[7],
                sidelobe=fields[8],
                strengths=fields[10],
                limitations=fields[11],
                advice=fields[12],
            )
        )
    if len(specs) != 50:
        raise RuntimeError(f"expected 50 window specifications, found {len(specs)}")
    return specs


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def window_reference_chapter() -> str:
    specs = parse_window_specs()
    chunks: list[str] = [
        r"""# pvx Window Reference

This chapter is a mathematical and visual atlas of every analysis window supported by pvx. A window is not a decorative preprocessing choice. It determines which samples contribute most strongly to each frame, how a sinusoid spreads across bins, how much neighbouring partials interfere, and how sharply a transient can be located in time.

## Notation

The notation below is used throughout the atlas. Each symbol is restated in the where clause following every equation so an entry can be read independently.

The principal quantities are \(N\), the window length; \(n\), the sample index in \(0,\ldots,N-1\); \(m=(N-1)/2\), the center index; \(w[n]\), the window coefficient; and \(W(e^{j\omega})\), the discrete-time Fourier transform of the window.

## Formula Families

The 50 named windows map to 19 formula families. The constants and parameter values in each atlas entry identify the exact member used by pvx.
"""
    ]
    for code, (equation, where_clause) in WINDOW_FORMULAE.items():
        chunks.append(f"### {code}\n\n$$\n{equation}\n$$\n\n{where_clause}\n")
    chunks.append(
        r"""## Quantitative Metrics

The comparison uses coherent gain, equivalent noise bandwidth, scalloping loss, main-lobe width, and peak sidelobe level. Together these measurements describe amplitude calibration, noise integration, sensitivity to frequencies between FFT bins, frequency discrimination, and leakage rejection.

$$
CG=\frac{1}{N}\sum_{n=0}^{N-1}w[n]
$$

where \(CG\) is coherent gain, \(w[n]\) is the window coefficient, \(n\) is the sample index, and \(N\) is the window length.

$$
ENBW=N\frac{\sum_{n=0}^{N-1}w^2[n]}{\left(\sum_{n=0}^{N-1}w[n]\right)^2}
$$

where \(ENBW\) is equivalent noise bandwidth measured in FFT bins, \(w[n]\) is the window coefficient, \(n\) is the sample index, and \(N\) is the window length.

$$
SL=20\log_{10}\left|\frac{W(2\pi/(2N))}{W(0)}\right|
$$

where \(SL\) is half-bin scalloping loss in decibels, \(W(\omega)\) is the window spectrum, and \(N\) is the window length.

## Comparative Window Summary

The following landscape table is deliberately limited to the quantities most useful for first selection. The atlas entries that follow carry the parameters, complete formula, strengths, limitations, advice, and full-size graphs.

\begin{landscape}
\small
\setlength{\tabcolsep}{3pt}
\begin{longtable}{@{}p{0.15\linewidth}p{0.13\linewidth}p{0.07\linewidth}p{0.08\linewidth}p{0.09\linewidth}p{0.10\linewidth}p{0.30\linewidth}@{}}
\caption{Comparison of all pvx analysis windows.}\\
\toprule
Window & Family & CG & ENBW & Main lobe & Sidelobe & Recommended use \\
\midrule
\endfirsthead
\toprule
Window & Family & CG & ENBW & Main lobe & Sidelobe & Recommended use \\
\midrule
\endhead
"""
    )
    for spec in specs:
        chunks.append(
            rf"\texttt{{{latex_escape(spec.name)}}} & {latex_escape(spec.family)} & "
            rf"{spec.coherent_gain} & {spec.enbw} & {spec.main_lobe} & {spec.sidelobe} & "
            rf"{latex_escape(spec.advice)} \\"
        )
    chunks.append(
        r"""\bottomrule
\end{longtable}
\end{landscape}

## Complete Window Atlas

Each entry begins with the exact formula family and parameters, followed by calibrated metrics, a time-domain graph, a frequency-domain graph, and listening guidance. The repeated structure is intentional: it supports direct comparison without forcing the reader to shuttle between a catalog and a distant plate section.
"""
    )
    for spec in specs:
        equation, where_clause = WINDOW_FORMULAE[spec.formula]
        display_name = spec.name.replace("_", " ")
        time_path = f"docs/assets/windows/print/{spec.name}_time.svg.png"
        freq_path = f"docs/assets/windows/print/{spec.name}_freq.svg.png"
        chunks.append(
            rf"""
\clearpage
### {display_name}
\index{{windows!{display_name}}}

The `{spec.name}` window belongs to the {spec.family.lower()} family. Its configured parameters are `{spec.parameters}`. This entry presents the mathematical definition before the graphs so the visible shape can be connected to the coefficients that produce it.

$$
{equation}
$$

{where_clause}

\begin{{table}}[H]
\centering
\caption{{Measured properties of the {latex_escape(display_name)} window.}}
\begin{{tabular}}{{@{{}}ll@{{}}}}
\toprule
Metric & Value \\
\midrule
Coherent gain & {spec.coherent_gain} \\
Equivalent noise bandwidth & {spec.enbw} bins \\
Scalloping loss & {spec.scalloping} dB \\
Main-lobe width & {spec.main_lobe} bins \\
Peak sidelobe & {spec.sidelobe} dB \\
\bottomrule
\end{{tabular}}
\end{{table}}

\clearpage
\begin{{figure}}[p]
\centering
\includegraphics[width=0.94\textwidth,height=0.78\textheight,keepaspectratio]{{{time_path}}}
\caption{{Time-domain shape of the {latex_escape(display_name)} window.}}
\end{{figure}}
\clearpage

\begin{{figure}}[p]
\centering
\includegraphics[width=0.94\textwidth,height=0.78\textheight,keepaspectratio]{{{freq_path}}}
\caption{{Magnitude spectrum of the {latex_escape(display_name)} window.}}
\end{{figure}}
\clearpage

#### Interpretation and use

The principal strength of this window is straightforward: {spec.strengths} That advantage should be weighed against its main limitation: {spec.limitations}

In a phase-vocoder render, the time-domain shape controls how strongly samples near the frame edges participate in the analysis. The frequency-domain graph shows the corresponding compromise between main-lobe width and sidelobe rejection. A narrower main lobe separates nearby stable partials more readily, while lower sidelobes reduce leakage from strong components into weaker neighbours.

The practical recommendation is: {spec.advice} Start with a short representative passage, compare against Hann at matched FFT and hop sizes, and listen separately for attack definition, tonal focus, high-frequency haze, and stereo stability. A window should be selected for the material and the requested transformation, not for a single attractive metric.

The reported measurements describe the window itself. They do not replace overlap-add verification. The synthesis hop, normalization policy, phase mode, and transient strategy can change the audible result even when the analysis window is held constant.
"""
        )
    return "\n".join(chunks).strip() + "\n"


def graph_atlas_appendix() -> str:
    function_graphs = (
        ("pitch ratio from semitones", "pitch_ratio_vs_semitones"),
        ("pitch ratio from cents", "pitch_ratio_vs_cents"),
        ("dynamics transfer curves", "dynamics_transfer_curves"),
        ("soft-clip transfer functions", "softclip_transfer_functions"),
        ("morph blend magnitude curves", "morph_blend_magnitude_curves"),
        ("mask exponent response", "mask_exponent_curves"),
        ("phase-mix angle response", "phase_mix_angle_curve"),
    )
    interpolation_graphs = (
        ("sample and hold", "interp_none"),
        ("nearest neighbour", "interp_nearest"),
        ("linear", "interp_linear"),
        ("cubic", "interp_cubic"),
        ("exponential", "interp_exponential"),
        ("smoothstep S-curve", "interp_s_curve"),
        ("smootherstep", "interp_smootherstep"),
        ("polynomial order 1", "interp_polynomial_order_1"),
        ("polynomial order 2", "interp_polynomial_order_2"),
        ("polynomial order 3", "interp_polynomial_order_3"),
        ("polynomial order 5", "interp_polynomial_order_5"),
    )
    chunks = [
        r"""# Transfer-Function and Automation Graph Atlas

This appendix restores the project graphs at a scale suitable for print. Each graph is introduced by a short statement of the mathematical or control relationship it illustrates. The figures are original pvx project assets.

## Transfer functions

The transfer-function group covers pitch ratios, dynamics, soft clipping, spectral morphing, masking, and phase mixing. These graphs connect command-line values to the curves applied inside a render.

$$
r_s=2^{s/12},\qquad r_c=2^{c/1200}
$$

where \(r_s\) is the pitch ratio for \(s\) semitones and \(r_c\) is the pitch ratio for \(c\) cents.
"""
    ]
    for title, stem in function_graphs:
        path = f"docs/assets/functions/print/{stem}.svg.png"
        chunks.append(
            rf"""
\clearpage
### {title}
\index{{graphs!{title}}}

The following graph presents the {title} relationship at full print scale. Read the horizontal axis as the controlling value and the vertical axis as the resulting ratio, gain, magnitude, or angle identified by the labels.

\begin{{figure}}[p]
\centering
\includegraphics[width=0.95\textwidth,height=0.80\textheight,keepaspectratio]{{{path}}}
\caption{{{title.capitalize()}.}}
\end{{figure}}
\clearpage
"""
        )
    chunks.append(
        r"""## Automation and interpolation

The interpolation group shows how pvx connects control points across render time. An interpolation rule is not merely a visual convenience. Its slope and curvature determine how quickly an audible parameter moves and whether the motion has discontinuous derivatives.

$$
u_{\mathrm{linear}}(t)=(1-\lambda)u_i+\lambda u_{i+1}
$$

where \(u_i\) and \(u_{i+1}\) are adjacent control values, \(t\) lies between their timestamps, and \(\lambda\) is normalized segment position in \([0,1]\).

$$
u_{\mathrm{smooth}}(t)=(1-h(\lambda))u_i+h(\lambda)u_{i+1},\qquad h(\lambda)=3\lambda^2-2\lambda^3
$$

where \(h(\lambda)\) is the smoothstep easing function, \(\lambda\) is normalized segment position, and \(u_i,u_{i+1}\) are adjacent control values.
"""
    )
    for title, stem in interpolation_graphs:
        path = f"docs/assets/interpolation/print/{stem}.svg.png"
        chunks.append(
            rf"""
\Needspace{{0.45\textheight}}
### {title}
\index{{interpolation!{title}}}

The following graph shows {title} interpolation between a fixed set of control points. The comparison is most useful when the curve near each point and the continuity of its slope are examined together.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.88\textwidth,height=0.24\textheight,trim=0 1000 0 0,clip,keepaspectratio]{{{path}}}
\caption{{{title.capitalize()} interpolation.}}
\end{{figure}}
"""
        )
    return "\n".join(chunks).strip() + "\n"


def youtube_search_url(query: str) -> str:
    """Return a durable ASCII YouTube catalog-search URL."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "+", query).strip("+")
    return f"https://www.youtube.com/results?search_query={normalized}"


def phase_vocoder_listening_appendix() -> str:
    import json

    catalog_path = ROOT / "docs" / "phase_vocoder_listening_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if len(catalog) != 100:
        raise RuntimeError(f"expected 100 listening examples, found {len(catalog)}")

    chunks = [
        r"""# A Phase-Vocoder Listening Library: 100 Musical Works

\index{listening guide}
\index{YouTube catalog searches}

This appendix is a guided listening and processing curriculum. It combines a small set of historically grounded phase-vocoder precedents with a much larger laboratory of famous works chosen because their voices, attacks, resonances, textures, or large-scale timing make particular transform behaviors easy to hear. The label attached to each entry matters. **Documented** identifies direct phase-vocoder practice supported by the historical literature. **Historical context** identifies related spectral and computer-music practice without claiming that every sound in the named work was made by a phase vocoder. **Listening laboratory** means that the work is recommended as source material or as a perceptual reference; it is not a claim about the production of the original recording.

YouTube changes individual upload URLs, regional availability, editions, and rights status frequently. For that reason, every entry supplies catalog-search links rather than endorsing one upload. The first search favors complete performances or recordings. The second favors scores, analyses, composer interviews, and demonstrations. Readers should choose lawful sources, respect recording and composition rights, and use public-domain or properly licensed audio for exported experiments.

The historically grounded group is supported by Trevor Wishart's retrospective on twenty-five years of phase-vocoder practice, the Library of Congress genealogy of Roger Reynolds's *Transfigured Wind*, IRCAM's analysis of Jonathan Harvey's *Mortuos Plango, Vivos Voco*, and the broader phase-vocoder literature cited in Chapter 1. These sources can be consulted through \href{https://doi.org/10.13128/Music_Tec-13207}{Wishart's retrospective}, the \href{https://www.loc.gov/collections/roger-reynolds/articles-and-essays/the-genealogy-of-transfigured-wind/introduction/}{Library of Congress essay}, and the \href{https://ressources.ircam.fr/fr/analysis/}{IRCAM analysis collection}.

## How to use the library

Each entry proposes one focused experiment. Begin with a short, legally obtained excerpt and keep an unprocessed reference at matched loudness. Change one variable at a time, render enough duration to expose the behavior, and record the command, window, hop, phase mode, transient policy, and random seed. The listening notes are prompts rather than expected answers.
"""
    ]
    current_category = None
    evidence_labels = {
        "documented": "Documented phase-vocoder practice",
        "historical context": "Related historical and spectral context",
        "listening laboratory": "Comparative listening laboratory",
    }
    for number, row in enumerate(catalog, start=1):
        if len(row) != 7:
            raise RuntimeError(f"listening example {number} does not contain seven fields")
        category, composer, title, year, evidence, focus, experiment = row
        if category != current_category:
            current_category = category
            chunks.append(
                f"""
\\clearpage
## {category}

The works in this group emphasize {category.lower()}. Read each note before listening, then use the proposed experiment as a controlled comparison rather than a recipe that must produce one correct sound.
"""
            )
        performance_url = youtube_search_url(f"{composer} {title} complete performance recording")
        study_url = youtube_search_url(f"{composer} {title} score analysis interview")
        safe_composer = latex_escape(composer)
        safe_title = latex_escape(title)
        safe_year = latex_escape(year)
        safe_focus = latex_escape(focus)
        safe_experiment = latex_escape(experiment)
        safe_label = latex_escape(evidence_labels[evidence])
        chunks.append(
            rf"""
\Needspace{{0.28\textheight}}
### {number}. {composer}: {title}

\index{{listening guide!{index_key(composer)}!{index_key(title)}}}
\index{{composers!{index_key(composer)}}}

\textbf{{{safe_label}.}} \textit{{{safe_composer}: {safe_title}}} ({safe_year}). {safe_focus}

{safe_experiment}

Catalog searches provide two starting points: \href{{{performance_url}}}{{YouTube performance search}} and \href{{{study_url}}}{{YouTube score, analysis, and interview search}}.
"""
        )
    chunks.append(
        r"""
\clearpage
## Comparative listening worksheet

The catalog becomes more useful when impressions are recorded consistently. For each experiment, note the source and edition, time range, rights status, pvx command, window and hop, stretch or pitch trajectory, phase mode, transient policy, normalization method, and monitoring level. Then rate attack definition, tonal focus, formant credibility, ambience continuity, stereo stability, noise coloration, and overall musical usefulness. A failed transformation is still useful evidence when its settings and source are documented.

The most revealing comparisons use matched loudness and short randomized presentation. Keep the dry source, a conservative transform, and an intentionally extreme transform. Describe artifacts in operational terms such as doubled attack, pre-echo, diffuse partial, unstable center image, shifted vowel, or metallic tail. Those descriptions lead back to parameters more reliably than a single global quality score.
"""
    )
    return "\n".join(chunks).strip() + "\n"


def cli_reference_index_entries() -> str:
    import json

    data = json.loads((ROOT / "docs" / "cli_flags_reference.json").read_text(encoding="utf-8"))
    lines = [
        r"\chapter*{Index Registration Notes}",
        r"\addcontentsline{toc}{chapter}{Index Registration Notes}",
        "The following invisible index registrations connect the alphabetical index to the generated CLI inventory.",
    ]
    for entry in data["entries"]:
        tool = index_key(entry["tool"])
        flag = entry["flag"]
        source = index_key(Path(entry["source"]).name)
        if tool and flag:
            lines.append(rf"\index{{commands!{tool}!{flag}}}")
            lines.append(rf"\index{{CLI options!{flag}!{tool}}}")
        if source:
            lines.append(rf"\index{{source modules!{source}}}")
        for choice in entry.get("choices", []):
            choice_key = index_key(str(choice))
            if choice_key:
                lines.append(rf"\index{{option values!{choice_key}}}")
    return "\n".join(lines).strip() + "\n"


def strip_named_section(text: str, heading_matchers: tuple[str, ...]) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    lowered_matchers = tuple(token.lower() for token in heading_matchers)
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip().lower()
            if any(token in heading_text for token in lowered_matchers):
                level = len(stripped) - len(stripped.lstrip("#"))
                i += 1
                while i < len(lines):
                    candidate = lines[i].strip()
                    if candidate.startswith("#"):
                        next_level = len(candidate) - len(candidate.lstrip("#"))
                        if next_level <= level:
                            break
                    i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out).strip() + "\n"


def clean_markdown(spec: ChapterSpec) -> tuple[str, str]:
    raw = spec.source.read_text(encoding="utf-8")
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_IMAGE_BLOCK_RE.sub("", text)
    text = IMG_TAG_RE.sub("", text)
    # Most repository illustrations are SVG.  The book uses raster archival
    # images and native TikZ figures so it stays portable without shell escape.
    text = MARKDOWN_IMAGE_RE.sub("", text)
    text = ATTRIBUTION_RE.sub("", text)
    text = MERMAID_RE.sub(
        "> Diagram source omitted in the PDF build. Use the HTML docs for the rendered version.\n",
        text,
    )
    # Documentation chapters resolve image paths relative to docs/, while the
    # compiled LaTeX runs from the repository root.
    text = re.sub(r"(!\[[^\]]*\]\()assets/", r"\1docs/assets/", text)
    # The negative lookbehind intentionally excludes images.  Treating an image as
    # a local link leaves its leading exclamation mark behind in the printed PDF.
    text = re.sub(r"(?<!!)\[([^\]]+)\]\(((?!https?://|mailto:)[^)]+)\)", r"\1", text)
    text = re.sub(r"⚠\s*Alpha release", "Alpha release", text)
    text = text.replace("⚠", "Note:")
    text = text.replace("→", "->")
    text = text.replace("←", "<-")
    text = text.replace("↔", "<->")
    text = text.replace("—", ",")
    text = text.replace("–", "-")

    title = spec.title
    lines = text.splitlines()
    body_start = 0
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            if title is None:
                title = line[2:].strip()
            body_start = idx + 1
            break
    if title is None:
        title = spec.source.stem.replace("_", " ")

    body = "\n".join(lines[body_start:]).strip() + "\n"
    if spec.strip_quick_setup:
        body = strip_named_section(
            body,
            (
                "quick setup",
                "install + path",
                "running any command with uv",
                "launch-ready helper commands",
            ),
        )
    body = strip_named_section(body, ("attribution",))
    normalized_lines: list[str] = []
    for line in body.splitlines():
        match = SIMPLE_NUMBERED_HEADING_RE.match(line)
        if match:
            line = f"{match.group(1)}{match.group(2)}"
        else:
            match = RANGE_HEADING_RE.match(line)
            if match:
                line = f"{match.group(1)}{match.group(2)}"
        normalized_lines.append(line)
    body = "\n".join(normalized_lines).strip() + "\n"
    body = ensure_list_introductions(body)
    body = ensure_equation_where_clauses(body)
    body = add_index_entries(body)
    return title, body


def frontmatter(today: date) -> str:
    return f"""
\\frontmatter
\\begin{{titlepage}}
\\centering
\\vspace*{{1.6cm}}
\\includegraphics[width=0.34\\textwidth]{{assets/pvx_logo.png}}\\par
\\vspace{{1.3cm}}
{{\\Huge\\bfseries pvx\\par}}
\\vspace{{0.35cm}}
{{\\LARGE User Guide\\par}}
\\vspace{{0.9cm}}
{{\\large Colby Leider\\par}}
\\vspace{{0.45cm}}
{{\\large Time stretch, pitch shift, automation, augmentation, and reference workflows\\par}}
\\vspace{{0.4cm}}
{{\\normalsize Command-line practice, quality tuning, multistage rendering, and machine-learning pipelines\\par}}
\\vfill
{{\\large pvx repository handbook\\par}}
\\vspace{{0.3cm}}
{{\\large Compiled from project documentation and repository source\\par}}
\\vspace{{0.6cm}}
{{\\large {today.strftime("%B %d, %Y")}\\par}}
\\end{{titlepage}}

\\clearpage
\\chapter*{{About This Guide}}
\\addcontentsline{{toc}}{{chapter}}{{About This Guide}}
This book is a curated, book-length guide to \\texttt{{pvx}} built from the repository's user-facing
documentation, examples, and command reference. It is meant to be read in layers: a practical first
pass for musicians and engineers who want useful results quickly, a deeper pass for readers who want
to understand the phase-vocoder model, and a reference pass for people building repeatable workflows,
augmentation pipelines, or release processes around the tool.

\\vspace{{1em}}
The supported public surface for the current alpha remains intentionally narrow:
\\texttt{{pvx}}, \\texttt{{pvxvoc}}, \\texttt{{pvxfreeze}}, \\texttt{{pvxwarp}}, \\texttt{{pvxformant}},
\\texttt{{pvxfilter}}, \\texttt{{pvxretune}}, and \\texttt{{pvxanalysis}}.
Where the appendices document broader flags and exploratory surfaces, they should be read as inventory
rather than a stronger compatibility promise.

\\clearpage
\\chapter*{{Preface}}
\\addcontentsline{{toc}}{{chapter}}{{Preface}}

\\begin{{center}}\\fbox{{\\large [TBA]}}\\end{{center}}

This preface marks the beginning of a longer editorial layer for the pvx project. The technical
reference can tell a reader what a flag accepts; it cannot by itself answer the question that arises
in a session: what should I listen for, what should I change first, and when is an artifact useful
rather than merely accidental? This book exists to put those questions beside the commands.

Phase-vocoder work rewards both curiosity and restraint. A small time change can be nearly invisible.
An extreme stretch can make a cymbal bloom into weather. A voice can be moved through an impossible
register while still carrying a recognisable vowel. The same transform that yields a clean utility
render can become an instrument when it is pushed past its transparent operating range. pvx is built
for that continuum, from sensible production repair to deliberate spectral invention.

The guide therefore has two jobs. One is pragmatic: it should help someone reach a dependable result
without treating digital signal processing as a private language. The other is historical and musical:
it should make clear that these tools come from a long line of experiments in speech, tape, synthesis,
and computer music. The algorithms are not neutral buttons. They embody choices about continuity,
frequency resolution, timing, and what a listener is willing to hear as a coherent sound.

The present volume is intentionally a living document. Its public contract follows the alpha surface
described in the release material. Some pages record exploratory capabilities because understanding
the surrounding landscape is useful, but a printed description is not a promise that every experiment
will remain stable. The supported commands are the ones to carry into a session, a batch pipeline, or
a release workflow.

The chapters can be read in order, but they do not demand it. A producer can start with the guided
workflows and return to the mathematics only when a decision needs explanation. A researcher can begin
with the first chapter and then move directly to architecture, windows, and the API material. A person
building a dataset pipeline can treat the cookbook and flag reference as a working bench. The same
book should be useful at three speeds: a quick answer during a render, a careful study at a desk, and a
deeper rereading after one has heard the artifacts for oneself.

The remaining preface will eventually include a fuller account of the project's origin, contributors,
and the listening practices that informed its defaults. Until then, [TBA] is kept here deliberately:
it is an invitation to distinguish the finished technical apparatus from the still-forming human story.

\\clearpage
\\chapter*{{Introduction}}
\\addcontentsline{{toc}}{{chapter}}{{Introduction}}

\\begin{{center}}\\fbox{{\\large [TBA]}}\\end{{center}}

pvx is a command-line environment for changing how recorded sound occupies time, pitch, spectrum, and
space. Its core is the phase vocoder, but the experience of using it is not reducible to one algorithm.
The command line holds choices about analysis windows, phase propagation, transient preservation,
automation curves, channels, format, loudness, and output discipline. This introduction offers a map
of those choices before the later chapters name every option.

## The central promise

The ordinary recording presents pitch and duration as linked facts. Play a tape faster and the event
becomes shorter and higher. Play it slower and it becomes longer and lower. Digital analysis makes it
possible to loosen that link. Time can be stretched while pitch remains broadly stable; pitch can move
while duration remains broadly stable. The word *broadly* matters. Audio is not a set of independent
knobs. A transform must decide what continuity to preserve, and every decision leaves a trace.

pvx gives that trace a place in the workflow. Some material wants transparency: dialogue, a solo
instrument, a finished mix, or a restoration pass. Other material wants character: a frozen chord,
a voice that turns granular at the edges, or an attack that opens into a spectral cloud. Both are valid
uses. The practical skill is to name the goal before selecting the settings.

## A workflow begins with listening

Before choosing an FFT size or a phase-locking mode, listen to the source as if it were a score. Ask
where the important attacks are, whether the sound is dominated by pitched partials or noise, whether
the stereo image carries musical information, and how much change the listener is expected to accept.
A drum loop and a sustained organ chord may both be called audio, but they make almost opposite demands
on a transform. The loop asks for sharp onset identity; the chord asks for stable harmonic motion.

One useful first pass is to render only a short region. Keep the source untouched. Make a conservative
version and an ambitious version. Compare them at the level where the finished work will be heard.
Headphones reveal phase movement and high-frequency fizz. Speakers reveal whether a transient still
has weight. A mono fold-down reveals whether a stereo treatment has undermined the image. This is not
extra ceremony. It is how a parameter becomes audible rather than theoretical.

## The four levers

Most pvx work can be understood through four levers:

1. **Time.** The stretch ratio changes the output duration. It also changes the distance between
   reconstructed analysis frames, which is why large ratios make phase and transient decisions more
   audible.
2. **Pitch.** Semitone and cent controls map musical intervals to frequency ratios. Pitch moves are
   often paired with formant treatment when a voice needs to retain its apparent vocal-tract identity.
3. **Continuity.** Window length, hop size, phase locking, and transient modes determine how the
   analysis observes motion and how the resynthesis joins it together.
4. **Change over time.** Automation lets a parameter follow a curve. Instead of treating a file as a
   single static setting, a render can grow, bend, open, narrow, or shift across its duration.

The later reference material separates these controls because the command line needs precise names.
While learning, it is better to remember their relationship. Time and pitch describe the intended
transformation. Continuity describes the cost of making it sound whole. Automation describes how the
transformation moves through the piece.

## Transparent, expressive, and extreme rendering

There are three useful attitudes toward phase-vocoder output. Transparent rendering aims to leave the
listener unsure that anything happened except the requested change. It favours moderate ratios,
appropriate windows, protected transients, and careful comparison against the source. Expressive
rendering lets a little process become part of the timbre. It can work beautifully on pads, sampled
voices, guitars, and ambient material. Extreme rendering makes the artifacts themselves the subject:
the granular haze, the stretched breath, the unstable pitch edge, and the strange persistence of a
single moment.

None of these attitudes is inherently more advanced than another. The mistake is to treat an extreme
result as failed transparency, or a clean result as timid creativity. The command is only half the
gesture. The context in which it is heard completes it.

## The stable surface and the experimental edge

For the current alpha, this guide treats `pvx`, `pvxvoc`, `pvxfreeze`, `pvxwarp`, `pvxformant`,
`pvxfilter`, `pvxretune`, and `pvxanalysis` as the supported command surface. They are the commands
to learn first and the commands around which fixtures and release checks are organised. Other names
may appear in source, examples, or larger inventories. Those are useful for orientation, but they do
not widen the compatibility promise.

This distinction is a kindness to future work. It lets the project explore without pretending every
experiment is ready to be depended upon. It also gives readers a clear place to stand. Learn the
supported tools deeply. Use the exploratory material with curiosity, a disposable output directory,
and a willingness to verify the result.

## Reading and working with this book

Each major chapter has a different role. The first chapter establishes the vocabulary and history.
Getting Started gives a dependable first render. The Quality Guide teaches diagnostic listening and
parameter tuning. Mathematical Foundations explains why windows, hops, and phase updates behave as
they do. The cookbook chapters turn recurring musical and production tasks into named recipes. The
appendix is for the moment when a descriptive explanation is no longer enough and an exact flag is
needed.

The document is also designed to be used with a terminal open. Read a small section, run a small
command, and listen. Preserve the command line that produced a result you like. Keep input and output
names unambiguous. For longer work, use checkpoints, manifests, or explicit notes about the settings.
The best render is hard to value if it cannot be reproduced.

The introduction is deliberately fuller than a command synopsis, and it will grow with the project.
The [TBA] marker remains because this is still an alpha handbook. The technical route is present now;
the final account of its audience, practice, and evolution is still being written.

\\clearpage
\\chapter*{{Reading Paths}}
\\addcontentsline{{toc}}{{chapter}}{{Reading Paths}}

If you are new to `pvx`, read this book in this order:

1. *Getting Started with pvx*
2. *pvx Quality Guide*
3. *pvx Example Cookbook*
4. The appendices when you need exact flags or deployment details

If you are using `pvx` as a production or research tool, add:

1. *Mathematical Foundations*
2. *pvx Application Programming Interface (API) Overview*
3. *ML Integration*
4. *CLI Flags Reference*

\\clearpage
\\chapter*{{List of Symbols}}
\\addcontentsline{{toc}}{{chapter}}{{List of Symbols}}
\\begin{{tabular}}{{ll}}
$N$ & FFT size, in samples \\\\
$H_a$ & Analysis hop, in samples \\\\
$H_s$ & Synthesis hop, in samples \\\\
$t$ & Frame index \\\\
$k$ & Frequency-bin index \\\\
$X_t[k]$ & Complex spectrum for frame $t$, bin $k$ \\\\
$\\phi_t[k]$ & Observed phase for frame $t$, bin $k$ \\\\
$\\widehat{{\\phi}}_t[k]$ & Reconstructed output phase \\\\
\\end{{tabular}}

\\clearpage
\\tableofcontents
\\clearpage
\\listoffigures
\\clearpage
\\listoftables
\\clearpage
\\mainmatter
""".strip() + "\n"


def phase_vocoder_history_expansion() -> str:
    return (ROOT / "docs" / "userguide_phase_vocoder_history_expansion.md").read_text(
        encoding="utf-8"
    ).strip()


def phase_vocoder_chapter() -> str:
    chapter = r"""
# What Is a Phase Vocoder?

\index{phase vocoder|(}
\index{short-time Fourier transform|see{STFT}}
\index{time stretching}
\index{pitch shifting}

A phase vocoder is an analysis-and-resynthesis instrument for recorded sound. It listens to a signal through a sequence of short, overlapping windows, estimates how each frequency component is moving, and rebuilds the signal on a new time grid. That one change of grid is what makes duration and pitch independently controllable. The result may be transparent, deliberately unreal, or somewhere wonderfully in between.

The name can be misleading. A conventional vocoder divides speech into broad frequency bands and transfers their envelopes to a carrier. A phase vocoder also measures frequency content, but its central concern is the phase advance of thousands of narrow spectral bins from one frame to the next. This lets it infer local frequency and re-schedule time without throwing away continuity.

## A quick mental model

Imagine a recording as a roll of film. Each frame is not an image but a tiny spectrum: how much bass, midrange, brightness, noise, and harmonic detail is present at one moment. A phase vocoder overlaps those frames so the film has no visible joins. It can space the reconstructed frames farther apart to lengthen time, closer together to compress time, or use an additional resampling step to move pitch.

\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=7mm, box/.style={draw, rounded corners=2pt, minimum width=25mm, minimum height=10mm, align=center, fill=black!3}, arrow/.style={-{Stealth[length=2mm]}, thick}]
\node[box] (input) {Input\\waveform};
\node[box, right=of input] (window) {Overlapping\\windows};
\node[box, right=of window] (fft) {Complex\\spectrum};
\node[box, right=of fft] (phase) {Phase\\trajectory};
\node[box, right=of phase] (output) {Output\\waveform};
\draw[arrow] (input) -- (window);
\draw[arrow] (window) -- (fft);
\draw[arrow] (fft) -- (phase);
\draw[arrow] (phase) -- (output);
\end{tikzpicture}
\caption{The phase-vocoder loop. Each output frame is rebuilt from a spectrum whose phase has been advanced on a new time grid.}
\end{figure}

The practical cycle is simple:

1. Window a short slice of audio.
2. Transform it into magnitude and phase.
3. Compare phase with the preceding slice to estimate true frequency.
4. Advance a new output phase at the desired synthesis hop.
5. Invert the spectrum and overlap-add the result.

The word *phase* matters because magnitude alone is not enough to make a stable waveform. A sine wave can have the same magnitude at many moments while occupying different positions in its cycle. When thousands of partials are rebuilt without a coherent phase story, the ear hears blur, shimmer, or the familiar metallic wash called phasiness. \index{phasiness}

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=1.15cm,y=0.8cm, arrow/.style={-{Stealth[length=2mm]}, thick}]
\draw[->] (0,0) -- (8,0) node[right] {time};
\foreach \x in {0.5,1.5,...,7.5} {\draw[fill=black!12] (\x,-0.14) rectangle +(1.5,0.28);}
\node[anchor=east] at (0,0) {analysis frames};
\draw[->] (0,-1.2) -- (8,-1.2) node[right] {time};
\foreach \x in {0.5,2.0,3.5,5.0,6.5} {\draw[fill=black!12] (\x,-1.34) rectangle +(1.5,0.28);}
\node[anchor=east] at (0,-1.2) {stretched synthesis};
\draw[arrow] (3.7,-0.25) -- (3.7,-1.0) node[midway,right,align=left] {larger synthesis\\hop};
\end{tikzpicture}
\caption{Time stretching changes the distance between reconstructed frames while preserving their spectral content.}
\end{figure}

## The historical thread

The phase vocoder belongs to a longer attempt to separate the identity of sound from the speed at which it unfolds. Mechanical and tape methods came first. If tape runs faster, a recording becomes shorter and its pitch rises. If it runs slower, it becomes longer and its pitch falls. The basic artistic wish, especially in speech, film, and music production, was to change one of those dimensions while keeping the other stable.

### Speech analysis before digital audio

In the 1930s and 1940s, Bell Telephone Laboratories developed systems that analyzed speech into spectral control information and reconstructed it with a separate source. Homer Dudley's vocoder and Voder were built for speech research and demonstration, not for the later practice of digital time scaling. Still, they established the crucial conceptual split: a sound can be described by time-varying spectral structure and then recreated by another mechanism. \index{Dudley, Homer}\index{vocoder}

![A Voder demonstration at the 1939 New York World's Fair, reproduced from Homer Dudley's 1940 paper. Public-domain or no-known-restrictions scan via Internet Archive Book Images on Wikimedia Commons.](docs/assets/userguide/history/voder_worlds_fair_1940.jpg){ width=48% }

The historical vocoder used filter banks and envelope followers. It was not yet a short-time Fourier transform, but it trained a generation of engineers to think of speech as a continuously changing spectrum rather than a single indivisible waveform.

![Dudley's 1940 block diagram of the voice mechanism. Public-domain or no-known-restrictions scan via Internet Archive Book Images on Wikimedia Commons.](docs/assets/userguide/history/dudley_voice_mechanism_1940.jpg){ width=88% }

### Tape, television, and the wish to uncouple time from pitch

During the tape era, machines such as Anton Springer's rotating-head systems and the later Eltro information rate changer attacked the same problem by slicing and reassembling material mechanically. The techniques made duration and pitch more independent than ordinary varispeed, but editing points, tape-head geometry, and repeated segments imposed their own audible fingerprint. Film and studio work adopted these systems precisely because the problem was already urgent: dialogue had to fit picture, voices had to become strange, and musical phrases had to land where an edit demanded.

The digital phase vocoder inherits that ambition but changes the object being stitched. Instead of small tape segments, it joins carefully phase-propagated spectra. The audible problem remains recognizable: continuity is everything.

### Flanagan's digital formulation

James L. Flanagan introduced the phase-vocoder idea in the 1960s as part of speech analysis and synthesis research. His formulation described a signal in terms of slowly varying amplitudes and instantaneous phases for bands of frequency. By the mid-1970s, Mark Portnoff showed how the short-time Fourier transform and the fast Fourier transform made the method practical on digital hardware. \index{Flanagan, James L.}\index{Portnoff, Mark}

![A historical vocoder schematic from Dudley's 1940 paper. Public-domain or no-known-restrictions scan via Internet Archive Book Images on Wikimedia Commons.](docs/assets/userguide/history/dudley_vocoder_schematic_1940.jpg){ width=88% }

![Dudley's companion Voder schematic, showing the related synthesis-oriented system. Public-domain or no-known-restrictions scan via Internet Archive Book Images on Wikimedia Commons.](docs/assets/userguide/history/dudley_voder_schematic_1940.jpg){ width=88% }

The key move is to compare observed phase advance with the advance expected at the center of each FFT bin. Their difference estimates the signal's local frequency. That estimate is then accumulated on a new synthesis timeline.

$$
\Delta\phi_t[k]=\operatorname{princarg}\left(\phi_t[k]-\phi_{t-1}[k]-\frac{2\pi kH_a}{N}\right)
$$
$$
\widehat{\omega}_t[k]=\frac{2\pi k}{N}+\frac{\Delta\phi_t[k]}{H_a}
$$
$$
\widehat{\phi}_t[k]=\widehat{\phi}_{t-1}[k]+\widehat{\omega}_t[k]H_s
$$

Here $H_a$ is the analysis hop, $H_s$ is the synthesis hop, $N$ is the FFT size, and $k$ identifies a spectral bin. Increasing $H_s$ relative to $H_a$ spreads reconstructed frames farther apart, lengthening the signal. \index{analysis hop}\index{synthesis hop}\index{FFT size}

### From research technique to musical tool

In the 1980s and 1990s, the phase vocoder became an expressive music-processing technique as personal computers and dedicated DSP systems became capable of enough overlapping FFTs. The method made radical time stretching, pitch transposition, spectral freezing, and formant experiments possible in a way tape systems could not. The tradeoff was clear: a phase vocoder can preserve stable tones remarkably well, but transients and rapidly changing spectra require special care.

Modern systems therefore add phase locking, transient detection, multi-resolution analysis, identity phase handling, and hybrid resynthesis. pvx follows this tradition with controls that make the quality decisions explicit rather than mysterious. \index{phase locking}\index{transient preservation}\index{multi-resolution analysis}

### A chronology of ideas, not merely machines

The history of the phase vocoder is easiest to understand as a sequence of changing representations. Dudley's channel vocoder represented speech by slowly varying band energies and source decisions. Flanagan and Golden represented a signal by short-time amplitude and phase information. Portnoff reorganised the same ideas around the FFT so they could be computed efficiently. Dolson made the digital method legible to musicians and computer-music practitioners. Laroche and Dolson then revisited the phase assumptions that caused loss of presence and gave the field a practical account of vertical coherence.

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=1.55cm,y=1cm, every node/.style={align=center,font=\small}]
\draw[very thick,-{Stealth[length=2.5mm]}] (0,0) -- (7.2,0);
\draw (0.4,0.12) -- (0.4,-0.12); \node[above=3mm] at (0.4,0) {1939}; \node[below=4mm] at (0.4,0) {Dudley\\Voder};
\draw (1.8,0.12) -- (1.8,-0.12); \node[above=3mm] at (1.8,0) {1966}; \node[below=4mm] at (1.8,0) {Flanagan and Golden\\phase vocoder};
\draw (3.2,0.12) -- (3.2,-0.12); \node[above=3mm] at (3.2,0) {1976}; \node[below=4mm] at (3.2,0) {Portnoff\\FFT implementation};
\draw (4.6,0.12) -- (4.6,-0.12); \node[above=3mm] at (4.6,0) {1986}; \node[below=4mm] at (4.6,0) {Dolson\\tutorial};
\draw (6.0,0.12) -- (6.0,-0.12); \node[above=3mm] at (6.0,0) {1999}; \node[below=4mm] at (6.0,0) {Laroche and Dolson\\phase locking};
\draw (7.0,0.12) -- (7.0,-0.12); \node[above=3mm] at (7.0,0) {Now}; \node[below=4mm] at (7.0,0) {hybrid and\\multi-resolution};
\end{tikzpicture}
\caption{A selected chronology of representations and implementation ideas leading to modern phase-vocoder practice.}
\end{figure}

This chronology should not be read as a single straight line of replacement. Channel vocoders, sinusoidal models, overlap-add systems, time-domain methods, and phase-vocoder techniques continued to influence one another. The important shift is that each generation made a different quantity convenient to manipulate. Once short-time phase derivatives became available, duration could be changed by editing the resynthesis schedule rather than by mechanically changing playback speed.

### Flanagan and Golden: phase as transmitted information

James L. Flanagan and Robert M. Golden's 1966 paper, *Phase Vocoder*, described speech in terms of short-time amplitude and phase spectra and simulated a complete analysis-transmission-synthesis system on a digital computer. Their title joined an older word, vocoder, to the quantity that distinguished the new representation. The method did not merely transmit the output of a bank of envelope followers. It retained information about local phase change, which could be interpreted as instantaneous frequency within each analysis channel.

For a complex channel output \(X_k(t)=A_k(t)e^{j\phi_k(t)}\), the local angular frequency can be expressed as:

$$
\omega_k(t)=\frac{d\phi_k(t)}{dt}
$$

where \(\omega_k(t)\) is instantaneous angular frequency for channel \(k\), \(\phi_k(t)\) is its unwrapped phase, and \(t\) is continuous time.

The equation looks modest, but it changes the meaning of phase from a static angle into a trajectory. If that trajectory is sampled consistently, a sinusoid that falls between channel centres can be represented by the difference between expected and observed phase advance. That principle survives in modern STFT implementations.

Flanagan and Golden also discussed time-scale expansion and compression. The historical significance is not that every later implementation copied their system detail for detail. It is that the paper established a phase-aware analysis-resynthesis vocabulary in which timing could become an independent operation. The paper appeared in the *Bell System Technical Journal*, volume 45, number 9, pages 1493 through 1509, with DOI 10.1002/j.1538-7305.1966.tb01706.x.

### Portnoff: the FFT makes the method practical

Mark R. Portnoff's 1976 paper, *Implementation of the Digital Phase Vocoder Using the Fast Fourier Transform*, moved the method toward the computational form now familiar in software. The FFT performed the bulk of both analysis and synthesis. Windows, frame hops, phase estimates, and overlap-add reconstruction could be described as a repeatable block algorithm rather than as a large bank of individually implemented filters.

\begin{figure}[H]
\centering
\begin{tikzpicture}[node distance=6mm, every node/.style={draw,minimum height=9mm,minimum width=24mm,align=center,font=\small}, arrow/.style={-{Stealth[length=2mm]},thick}]
\node (frame) {frame and\\window};
\node[right=of frame] (fft) {FFT};
\node[right=of fft] (polar) {magnitude and\\phase};
\node[below=of polar] (modify) {phase\\modification};
\node[left=of modify] (ifft) {inverse FFT};
\node[left=of ifft] (ola) {overlap-add};
\draw[arrow] (frame) -- (fft);
\draw[arrow] (fft) -- (polar);
\draw[arrow] (polar) -- (modify);
\draw[arrow] (modify) -- (ifft);
\draw[arrow] (ifft) -- (ola);
\end{tikzpicture}
\caption{The FFT-oriented analysis and synthesis loop associated with the practical digital phase vocoder.}
\end{figure}

The computational saving matters because a phase vocoder requires many transforms. For a signal of length \(L\), FFT size \(N\), and analysis hop \(H_a\), an approximate transform count is:

$$
F\approx 2\left\lceil\frac{L-N}{H_a}\right\rceil+2
$$

where \(F\) is the combined number of forward and inverse transforms, \(L\) is the signal length in samples, \(N\) is the FFT size, and \(H_a\) is the analysis hop in samples.

The exact count depends on padding and boundary policy, but the relationship explains why hop size is both a quality parameter and a cost parameter. Smaller hops increase temporal sampling and overlap while increasing the number of transforms. Portnoff's FFT formulation made that tradeoff manageable on the minicomputers of the period and natural on later workstations.

Portnoff also placed window design close to implementation. A phase estimate is only as useful as the frame that produced it. Leakage, main-lobe width, and overlap behaviour determine how clearly a sinusoidal component appears and how smoothly frames recombine. The complete window atlas later in this book develops that part of the inheritance.

### Computer music and the widening of purpose

By the late 1970s and 1980s, the phase vocoder had escaped the narrow category of speech transmission. Computer-music systems treated analysis data as compositional material. Magnitudes could be frozen, interpolated, filtered, or transferred. Phase trajectories could be advanced at a new rate. A sound could be stretched until its internal partials became a landscape.

This change of audience mattered. A telecommunications engineer might measure intelligibility and bandwidth. A composer might value a transformation precisely because it revealed hidden modulation or made a transient dissolve into texture. The same artifact could be an error in one setting and an aesthetic resource in another. Phase-vocoder history is therefore also a history of listening criteria.

The separation of analysis from resynthesis encouraged archives of spectral frames and offline transformations. A program could analyze once and render many times. This pattern remains visible in pvx analysis artifacts, response profiles, checkpointing, and batch workflows. What changed is the speed and scale: operations that once required a research workstation can now be scripted across a corpus.

### Mark Dolson and the tutorial synthesis

Mark Dolson's 1986 article, *The Phase Vocoder: A Tutorial*, became a major bridge between specialist signal-processing literature and computer-music practice. Its importance lies partly in synthesis. The phase vocoder had accumulated several equivalent descriptions, implementation conventions, and practical uses. Dolson presented the method as a coherent system and related analysis parameters to transformations that musicians could recognize.

The tutorial emphasized that the analysis produces time-varying amplitude and frequency information. Time scaling can then be understood as resampling those trajectories or changing the relationship between analysis and synthesis time. Pitch modification can be built by combining time scaling with sample-rate conversion, or by altering spectral placement more directly.

Let the time-scale factor be \(\alpha\). A simple hop relationship is:

$$
H_s=\alpha H_a
$$

where \(H_s\) is the synthesis hop, \(H_a\) is the analysis hop, and \(\alpha\) is the requested output-duration ratio.

This compact equation is not a complete algorithm because phase must follow the new schedule. It is nevertheless the control relationship users encounter most directly. If \(\alpha>1\), output frames are spaced farther apart and the sound becomes longer. If \(0<\alpha<1\), they are spaced closer together and the sound becomes shorter.

Dolson also helped establish a vocabulary for practical applications: time scaling, pitch transposition, cross-synthesis, spectral modification, and the analysis of musical sounds. The tutorial appeared in *Computer Music Journal*, volume 10, number 4, pages 14 through 27. Its enduring role is pedagogical. It taught readers to see the phase vocoder as a general musical instrument whose operations emerge from one analysis-resynthesis model.

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=1.05cm,y=1.0cm]
\draw[->] (0,0) -- (8,0) node[right] {analysis time};
\draw[->] (0,0) -- (0,3.2) node[above] {parameter value};
\draw[thick] plot[smooth] coordinates {(0.2,0.6) (1.2,1.2) (2.2,1.0) (3.2,2.4) (4.2,2.0) (5.2,2.7) (6.2,1.8) (7.2,2.2)};
\foreach \x in {0.2,1.2,2.2,3.2,4.2,5.2,6.2,7.2} {\draw[densely dotted] (\x,0) -- (\x,3);}
\node[align=left,anchor=west] at (4.7,0.55) {analysis trajectories can be\\sampled on a new time grid};
\end{tikzpicture}
\caption{A conceptual parameter trajectory, the representation that makes time remapping and spectral editing intelligible.}
\end{figure}

### Horizontal coherence and vertical coherence

For many years the classic phase vocoder's central weakness was described with perceptual words: phasiness, reverberation, diffuseness, or loss of presence. These terms overlap but are not identical. They point to a sound whose partials no longer seem to belong to one compact source.

Two kinds of relationship help explain the problem. Horizontal coherence concerns a bin across successive frames. Its phase should evolve according to the component's instantaneous frequency. Vertical coherence concerns neighbouring bins within one frame. Bins belonging to the same spectral peak should retain a meaningful phase relationship.

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=1.0cm,y=0.75cm,font=\small]
\foreach \t in {0,...,6} {\draw[gray!35] (\t,0) -- (\t,5);}
\foreach \k in {0,...,5} {\draw[gray!35] (0,\k) -- (6,\k);}
\draw[very thick,-{Stealth[length=2mm]}] (0.3,3.2) -- (5.7,3.2);
\node[anchor=south] at (3,3.2) {horizontal phase continuity across frames};
\draw[very thick,-{Stealth[length=2mm]}] (3.2,0.3) -- (3.2,4.7);
\node[rotate=90,anchor=south] at (3.2,2.5) {vertical coherence across bins};
\node[below] at (3,-0.25) {frame index};
\node[rotate=90] at (-0.45,2.5) {frequency bin};
\end{tikzpicture}
\caption{Horizontal and vertical phase relationships in the STFT lattice.}
\end{figure}

Classic propagation can preserve horizontal continuity while allowing bins within a peak to drift independently. The resynthesised peak then behaves less like one sinusoidal component observed through a window and more like a collection of unrelated oscillators. That decorrelation is one source of the diffuse quality associated with phase-vocoder stretching.

### Jean Laroche and Mark Dolson: explaining phasiness

Jean Laroche and Mark Dolson's 1999 paper, *Improved Phase Vocoder Time-Scale Modification of Audio*, directly examined phasiness and proposed new phase-calculation techniques. The paper's historical importance is that it connected a familiar perceptual complaint to the loss of phase coherence across bins and then offered practical algorithms that improved sound quality.

The authors distinguished phase propagation at spectral peaks from the treatment of bins around those peaks. A peak can be regarded as the centre of a region of influence. Rather than allowing every bin to accumulate phase independently, surrounding bins can inherit a phase relationship from the peak.

For peak bin \(p\) and neighbouring bin \(k\), identity phase locking can be written conceptually as:

$$
\widehat{\phi}_t[k]=\widehat{\phi}_t[p]+\phi_t[k]-\phi_t[p]
$$

where \(\widehat{\phi}_t[k]\) is the synthesis phase of neighbour bin \(k\), \(\widehat{\phi}_t[p]\) is the propagated synthesis phase of peak bin \(p\), \(\phi_t[k]\) is the observed analysis phase of bin \(k\), and \(\phi_t[p]\) is the observed analysis phase of the peak.

The subtraction retains the analysis-frame phase offset between the neighbour and its peak. The addition places that offset around the peak's coherent synthesis phase. The bins therefore preserve more of the structure imposed by one underlying partial and its window transform.

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=0.7cm,y=1.1cm,font=\small]
\draw[->] (0,0) -- (12,0) node[right] {frequency bin};
\draw[->] (0,0) -- (0,4) node[above] {magnitude};
\draw[thick] plot[smooth] coordinates {(0,0.1)(1,0.25)(2,0.8)(3,2.8)(4,1.0)(5,0.3)(6,0.2)(7,0.6)(8,3.4)(9,1.1)(10,0.35)(11,0.1)};
\fill (3,2.8) circle (2pt) node[above] {peak \(p_1\)};
\fill (8,3.4) circle (2pt) node[above] {peak \(p_2\)};
\draw[decorate,decoration={brace,mirror,amplitude=5pt}] (1.2,-0.2) -- (5.4,-0.2) node[midway,below=6pt] {region of influence};
\draw[decorate,decoration={brace,mirror,amplitude=5pt}] (5.6,-0.2) -- (10.8,-0.2) node[midway,below=6pt] {region of influence};
\end{tikzpicture}
\caption{Identity phase locking assigns neighbouring bins to spectral peaks and preserves their observed relative phases.}
\end{figure}

Laroche and Dolson described two extensions and reported significantly improved results. Their work also showed that improved coherence could reduce computational cost in some formulations. The paper appeared in *IEEE Transactions on Speech and Audio Processing*, volume 7, number 3, pages 323 through 332, DOI 10.1109/89.759041.

The contribution should not be reduced to one option labelled phase locking. It changed the field's account of why the classic sound occurred. Once phasiness was described as a coherence problem, later systems could compare strategies for peak selection, region assignment, transient handling, and channel consistency.

### Pitch shifting, harmonising, and spectral translation

Laroche and Dolson also described frequency-domain techniques for pitch shifting, harmonising, chorusing, and non-standard frequency modification. Their 1999 workshop paper presented integer-bin and fractional-bin spectral translation approaches with phase adjustment across successive frames. This line of work is important for pvx because pitch processing is not merely a time stretch followed by anonymous resampling. It can be understood as controlled movement of spectral regions.

For a semitone displacement \(s\), the ideal frequency ratio is:

$$
r=2^{s/12}
$$

where \(r\) is the multiplicative frequency ratio and \(s\) is the displacement in equal-tempered semitones.

For a cent displacement \(c\), the corresponding relation is:

$$
r=2^{c/1200}
$$

where \(r\) is the multiplicative frequency ratio and \(c\) is the displacement in cents.

These equations state the musical target, not the full spectral operation. A practical algorithm must decide how magnitudes move between bins, how phases remain continuous, how components crossing one another are treated, and whether formant structure should follow the pitch shift.

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=0.9cm,y=0.7cm,font=\small]
\draw[->] (0,0) -- (10,0) node[right] {frequency};
\foreach \x/\h in {1/1.2,2/2.2,3/1.6,4/2.8,5/1.4} {\draw[very thick] (\x,0) -- (\x,\h);}
\foreach \x/\h in {4.2/1.2,5.4/2.2,6.6/1.6,7.8/2.8,9/1.4} {\draw[very thick,green!40!black] (\x,0) -- (\x,\h);}
\draw[-{Stealth[length=2mm]},thick] (3.2,3.2) -- (6.0,3.2) node[midway,above] {spectral translation};
\node at (3, -0.5) {source partials};
\node at (7, -0.5) {shifted partials};
\end{tikzpicture}
\caption{A conceptual spectral translation used for pitch shifting and harmonisation.}
\end{figure}

### Transients expose the limits of stationarity

The phase vocoder assumes that a short frame can be treated as a useful local spectral description. A transient challenges that assumption. Its energy changes rapidly, and the onset may occupy only a small fraction of the window. Stretching the surrounding spectral frames can spread that event through time and reduce its impact.

Transient-aware systems detect sudden changes, preserve or reset selected phases, shorten the effective window, or route the event through a time-domain method. Each strategy makes a different compromise. A reset restores alignment at an onset but can interrupt a stable sinusoidal trajectory. A longer window improves low-frequency discrimination but includes more samples from both sides of the attack.

One common spectral-flux measure is:

$$
SF_t=\sum_k\max\left(|X_t[k]|-|X_{t-1}[k]|,0\right)
$$

where \(SF_t\) is positive spectral flux at frame \(t\), \(X_t[k]\) is the complex STFT coefficient at bin \(k\), and the maximum retains only increases in magnitude.

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=1.0cm,y=0.9cm,font=\small]
\draw[->] (0,0) -- (8,0) node[right] {time};
\draw[->] (0,0) -- (0,3.4) node[above] {energy};
\draw[thick] plot[smooth] coordinates {(0,0.15)(1,0.2)(2,0.18)(3,0.25)(3.5,3.0)(3.8,1.3)(4.3,0.7)(5,0.4)(6,0.25)(7,0.2)};
\draw[densely dashed] (3.5,0) -- (3.5,3.1);
\node[anchor=west] at (3.7,2.7) {detected onset};
\draw[<->,thick] (2.3,-0.35) -- (4.7,-0.35) node[midway,below] {window support};
\end{tikzpicture}
\caption{A transient can occupy only a small part of a frame, making onset preservation a separate design problem.}
\end{figure}

This problem led to transient detection, phase resets, hybrid methods, and multi-resolution systems. It also explains why no single window is best for every source. A percussive signal asks the analysis to localise time; a sustained bass note asks it to discriminate nearby low frequencies.

### Stereo and multichannel coherence

The historical phase vocoder was often described for one channel. Modern production material is commonly stereo or multichannel, and independent processing can disturb interchannel time and level differences. A stable image requires the algorithm to consider relationships between channels as well as relationships between bins.

For channels \(i\) and \(j\), interchannel phase difference is:

$$
\Delta\phi_{ij}(k,t)=\phi_i(k,t)-\phi_j(k,t)
$$

where \(\Delta\phi_{ij}(k,t)\) is the phase difference at bin \(k\) and frame \(t\), while \(\phi_i\) and \(\phi_j\) are the channel phases.

Preserving that difference exactly is not always the only goal. Low-frequency image stability, diffuse ambience, decorrelated noise, and transient direction can require different treatment. pvx therefore exposes coherence choices rather than assuming that copying one channel's phase is universally correct.

### What the historical work leaves us

The historical line from Dudley through Flanagan, Golden, Portnoff, Dolson, Laroche, and later researchers leaves three durable lessons. First, representation controls imagination: once amplitude and phase trajectories are explicit, time and pitch become editable structures. Second, efficiency changes art: the FFT turned a laboratory model into a repeatable process, and modern hardware turns it into a live or corpus-scale operation. Third, perceptual quality depends on relationships. A bin is not heard alone; it belongs to a peak, a frame, a transient, a channel, and a musical event.

pvx inherits all three lessons. It treats the phase vocoder as a practical engine, a controllable source of artifacts, and a framework around which automation, analysis, checkpointing, and reproducibility can be built. The rest of the book moves from this history into implementation detail, but the listening problem remains the same: decide which relationships must survive the transformation.

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=1.0cm,y=1.0cm, arrow/.style={-{Stealth[length=2mm]}, thick}]
\draw[->] (0,0) -- (7,0) node[right] {render time};
\draw[->] (0,0) -- (0,3) node[above] {parameter value};
\draw[very thick] plot[smooth] coordinates {(0,0.25) (1,0.32) (2,0.65) (3,1.55) (4,2.45) (5,2.72) (6.5,2.78)};
\foreach \x/\y in {0/0.25,2/0.65,4/2.45,6.5/2.78} {\fill (\x,\y) circle (1.6pt);}
\end{tikzpicture}
\caption{Automation treats a parameter as a continuous curve across the render, not as a sequence of unrelated settings.}
\end{figure}

## What the transform sees

For every analysis frame, pvx creates a complex spectrum $X_t[k]$. Its magnitude describes the energy at bin $k$; its phase describes that component's cycle position. A standard short-time Fourier transform is:

$$
X_t[k]=\sum_{n=0}^{N-1}x[n+tH_a]w[n]e^{-j2\pi kn/N}
$$

The window $w[n]$ softens frame boundaries. Overlap between windows lets the reconstructed signal remain smooth. A narrow window follows quick attacks well but provides less precise pitch information. A longer window gives a clearer view of closely spaced frequencies but can smear a drum hit or consonant across time. There is no universal perfect choice; musical material decides. \index{window function}\index{STFT}

\begin{figure}[H]
\centering
\begin{tikzpicture}[x=1.0cm,y=1.0cm, arrow/.style={-{Stealth[length=2mm]}, thick}]
\draw[->] (0,0) -- (7,0) node[right] {frequency};
\draw[->] (0,0) -- (0,3) node[above] {relative level};
\draw[very thick] plot[smooth] coordinates {(0,2.7) (0.7,2.45) (1.4,1.75) (2.1,0.7) (2.5,0.15) (2.9,0.38) (3.3,0.12) (3.8,0.24) (4.3,0.08) (4.8,0.16) (5.5,0.05) (6.5,0.03)};
\draw[densely dashed] (2.5,0) -- (2.5,2.75);
\node[align=center,anchor=west] at (3.0,2.3) {main lobe and\\side lobes};
\end{tikzpicture}
\caption{A conceptual window response. Window choice balances temporal detail against frequency selectivity.}
\end{figure}

## A small comparison

\begin{table}[H]
\centering
\caption{Three ways to think about changing recorded sound.}
\begin{tabular}{p{0.22\textwidth}p{0.32\textwidth}p{0.32\textwidth}}
\toprule
Method & What it changes directly & Characteristic limitation \\
\midrule
Tape varispeed & Playback speed & Pitch and duration remain coupled. \\
Segmented tape systems & Short physical segments & Joins and repeats can be audible. \\
Phase vocoder & Spectral frames and phase trajectories & Transients and dense attacks need careful handling. \\
\bottomrule
\end{tabular}
\end{table}

## How pvx puts the idea to work

In pvx, the phase vocoder is the engine beneath time stretching, pitch shifting, freeze-like spectral holds, formant-aware workflows, and many automated transformations. The stable approach is usually modest: choose a sensible window and hop, preserve transients when the material needs it, use phase locking for harmonic sources, and audition a small section before committing to an extreme render.

```bash
pvx voc voice.wav --stretch 1.25 --transient-preserve --phase-locking identity --output voice_longer.wav
pvx voc piano.wav --pitch -3 --formant-preserve --output piano_down.wav
pvx freeze cymbal.wav --freeze-time 0.42 --duration 12 --output cymbal_cloud.wav
```

The commands look compact because the conceptual work is inside the analysis loop. The rest of this guide turns those choices into repeatable practice.

\index{phase vocoder|)}
""".strip() + "\n"
    marker = "### What the historical work leaves us"
    if marker not in chapter:
        raise RuntimeError("phase-vocoder history insertion marker is missing")
    return chapter.replace(
        marker,
        f"{phase_vocoder_history_expansion()}\n\n{marker}",
        1,
    )


def glossary_chapter() -> str:
    import json

    glossary_path = ROOT / "docs" / "userguide_glossary.json"
    entries = json.loads(glossary_path.read_text(encoding="utf-8"))
    if len(entries) < 120:
        raise RuntimeError(f"expected at least 120 glossary entries, found {len(entries)}")
    terms = [entry[0] for entry in entries]
    if len(terms) != len(set(terms)):
        raise RuntimeError("glossary terms must be unique")

    chunks = [
        r"""\backmatter
# Glossary

\index{glossary}
"""
    ]
    for term, definition in sorted(entries, key=lambda entry: entry[0].casefold()):
        chunks.append(
            rf"""
\Needspace{{4\baselineskip}}
\index{{{index_key(term)}}}
\noindent\textbf{{{latex_escape(term)}.}} {latex_escape(definition)}
"""
        )
    return "\n".join(chunks).strip() + "\n"


def parse_bibtex_records() -> dict[str, dict[str, str]]:
    source = (ROOT / "docs" / "references.bib").read_text(encoding="utf-8")
    records: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@\w+\{([^,]+),\s*(.*?)\n\}", source, re.DOTALL):
        key, body = match.groups()
        fields = {
            field.lower(): value.strip()
            for field, value in re.findall(r'(\w+)\s*=\s*"([^"]*)"', body, re.DOTALL)
        }
        records[key] = fields
    return records


def bibliography_sort_key(author: str) -> tuple[str, str]:
    first_author = author.split(" and ", 1)[0].split(";", 1)[0].strip()
    if "," in first_author:
        surname = first_author.split(",", 1)[0]
    else:
        surname = first_author.split()[-1]
    return surname.casefold(), author.casefold()


def bibliography_chapter() -> str:
    import json

    paper_keys = json.loads(
        (ROOT / "docs" / "userguide_bibliography_papers.json").read_text(encoding="utf-8")
    )
    books = json.loads(
        (ROOT / "docs" / "userguide_bibliography_books.json").read_text(encoding="utf-8")
    )
    if len(paper_keys) != len(set(paper_keys)):
        raise RuntimeError("bibliography paper keys must be unique")
    if len(paper_keys) + len(books) < 100:
        raise RuntimeError("the bibliography must contain at least 100 papers and books")

    records = parse_bibtex_records()
    missing = [key for key in paper_keys if key not in records]
    if missing:
        raise RuntimeError(f"missing bibliography records: {', '.join(missing)}")
    venue_overrides = {
        "marchand1998improving056": "Proceedings of the International Computer Music Conference",
        "garas1998timepitch058": "Proceedings of ProRISC/IEEE CSSP98",
        "bristowjohnson2002intraframe085": "Proceedings of the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics",
        "ellis2006modelbased105": "Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing",
        "evangelista2008modified121": "Proceedings of the IEEE International Symposium on Communications, Control and Signal Processing",
        "juillerat2017audio195": "Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing",
        "akaishi2023improving241": "Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing",
    }

    chunks = [
        r"""\clearpage

# Bibliography

\index{bibliography}

This bibliography gathers one hundred twenty-five books and papers that support the theory, engineering practice, listening methods, and historical discussion in this guide. The papers emphasize phase-vocoder development, short-time analysis and reconstruction, pitch and formant processing, transient handling, time-scale modification, source separation, and objective evaluation. The books supply broader foundations in digital signal processing, speech, psychoacoustics, computer music, and sound engineering.

\section*{Papers and articles}
\markright{Bibliography: Papers and articles}
\index{bibliography!papers and articles}

The one hundred entries in this section are alphabetized by the first named author. DOI links are preferred when the repository record supplies one; otherwise, a stable publisher or catalog record is used where available.
"""
    ]
    papers = [(key, records[key]) for key in paper_keys]
    papers.sort(key=lambda pair: bibliography_sort_key(pair[1].get("author", "")))
    for key, item in papers:
        author = latex_escape(item.get("author", "").replace(" and ", "; "))
        plain_title = item.get("title", "").rstrip()
        title = latex_escape(plain_title)
        title_mark = "" if plain_title.endswith((".", "?", "!")) else "."
        venue = latex_escape(venue_overrides.get(key, item.get("howpublished", "")))
        year = latex_escape(item.get("year", ""))
        doi = item.get("doi", "").rstrip(".")
        url = item.get("url", "")
        link = ""
        if doi:
            link = rf" \href{{https://doi.org/{doi}}}{{DOI}}."
        elif url and "scholar.google.com" not in url:
            link = rf" \href{{{url}}}{{catalog record}}."
        chunks.append(
            rf"""
\Needspace{{3\baselineskip}}
\noindent\hangindent=1.5em\hangafter=1
{author}. ``{title}{title_mark}'' \textit{{{venue}}} ({year}).{link}\par
"""
        )

    chunks.append(
        r"""
\clearpage
\section*{Books}
\markright{Bibliography: Books}
\index{bibliography!books}

The twenty-five books in this section provide durable background and extended treatments. Editions are stated when the catalog distinguishes them.
"""
    )
    books.sort(key=lambda item: bibliography_sort_key(item[0]))
    for author, title, publisher, year, edition in books:
        edition_text = f" {latex_escape(edition)}." if edition else ""
        chunks.append(
            rf"""
\Needspace{{3\baselineskip}}
\noindent\hangindent=1.5em\hangafter=1
{latex_escape(author)}. \textit{{{latex_escape(title)}}}. {latex_escape(publisher)}, {latex_escape(year)}.{edition_text}\par
"""
        )
    return "\n".join(chunks).strip() + "\n"


def backmatter() -> str:
    return glossary_chapter() + bibliography_chapter() + r"""
\chapter*{Open Media Credits}
\addcontentsline{toc}{chapter}{Open Media Credits}

The historical illustrations in Chapter 1 are reproduced from Homer Dudley's 1940 paper, *The Carrier Nature of Speech*, via Internet Archive Book Images and Wikimedia Commons. Wikimedia Commons records them as public-domain or no-known-copyright-restrictions scans. Source records: Voder demonstration, voice-mechanism block diagram, and vocoder schematic: \url{https://commons.wikimedia.org/wiki/Category:Vocoder}. The analytical plots and window diagrams are original pvx project assets and are included with the project documentation.

\clearpage
\printindex
""".strip() + "\n"


def build_book_markdown(today: date) -> str:
    chunks: list[str] = [frontmatter(today)]
    chunks.append(r"\part{Foundations}" + "\n")
    history = phase_vocoder_chapter()
    history = ensure_list_introductions(history)
    history = ensure_equation_where_clauses(history)
    history = add_index_entries(history)
    chunks.append(history)
    for part_title, chapters in BOOK_PARTS:
        chunks.append(f"\\part{{{part_title}}}\n")
        for spec in chapters:
            if spec.source.name == "WINDOW_REFERENCE.md":
                chunks.append(add_index_entries(window_reference_chapter()))
                continue
            title, body = clean_markdown(spec)
            chunks.append(f"# {title}\n\n{body}")
    chunks.append("\\appendix\n")
    for spec in APPENDICES:
        title, body = clean_markdown(spec)
        chunks.append(f"# {title}\n\n{body}")
    chunks.append(add_index_entries(phase_vocoder_listening_appendix()))
    chunks.append(add_index_entries(graph_atlas_appendix()))
    chunks.append(cli_reference_index_entries())
    chunks.append(backmatter())
    book = "\n".join(chunks).strip() + "\n"
    book = book.replace(" — ", ", ").replace("—", ",").replace("–", "-")
    book = book.replace(r"\(", "$").replace(r"\)", "$")
    return book


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build output/pdf/USERGUIDE.pdf for pvx.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "USERGUIDE.pdf",
        help="Output PDF path (default: output/pdf/USERGUIDE.pdf)",
    )
    parser.add_argument(
        "--source-out",
        type=Path,
        default=TMP_DIR / "USERGUIDE.generated.md",
        help="Where to write the generated combined markdown source.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_out = args.source_out.resolve()
    output_pdf = args.output.resolve()
    book_md = build_book_markdown(date.today())
    source_out.write_text(book_md, encoding="utf-8")

    tex_out = TMP_DIR / "USERGUIDE.tex"
    cmd = [
        "pandoc",
        str(source_out),
        "--from=markdown+raw_tex+raw_html",
        "--to=latex",
        "--top-level-division=chapter",
        "--include-in-header",
        str(PREAMBLE),
        "--lua-filter",
        str(INLINE_CODE_FILTER),
        "--resource-path",
        f"{ROOT}:{ROOT / 'docs'}:{ROOT / 'assets'}",
        "-V",
        "documentclass=book",
        "-V",
        "papersize=letter",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=10pt",
        "-V",
        "linestretch=1.05",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=black",
        "-V",
        "urlcolor=blue",
        "--listings",
        "--output", str(tex_out),
    ]
    run(cmd)
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={TMP_DIR}", str(tex_out)])
    run(["makeindex", "USERGUIDE.idx"], cwd=TMP_DIR)
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={TMP_DIR}", str(tex_out)])
    run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={TMP_DIR}", str(tex_out)])
    (TMP_DIR / "USERGUIDE.pdf").replace(output_pdf)
    root_pdf = ROOT / "USERGUIDE.pdf"
    if root_pdf != output_pdf:
        shutil.copy2(output_pdf, root_pdf)
    print(f"[ok] wrote {output_pdf}")
    print(f"[ok] copied {root_pdf}")
    print(f"[ok] source {source_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
