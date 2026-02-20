#!/usr/bin/env python3
"""Generate grouped HTML documentation for pvx algorithms and research references."""

from __future__ import annotations

import ast
from collections import Counter, OrderedDict
from html import escape
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote_plus
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DOCS_HTML_DIR = ROOT / "docs" / "html"
GROUPS_DIR = DOCS_HTML_DIR / "groups"
README_PATH = ROOT / "README.md"
GLOSSARY_PATH = ROOT / "docs" / "technical_glossary.json"
EXTRA_PAPERS_PATH = ROOT / "docs" / "papers_extra.json"
WINDOW_METRICS_PATH = ROOT / "docs" / "window_metrics.json"
CLI_FLAGS_PATH = ROOT / "docs" / "cli_flags_reference.json"
LIMITATIONS_PATH = ROOT / "docs" / "algorithm_limitations.json"
COOKBOOK_PATH = ROOT / "docs" / "pipeline_cookbook.json"
BENCHMARKS_PATH = ROOT / "docs" / "benchmarks" / "latest.json"
CITATION_QUALITY_PATH = ROOT / "docs" / "citation_quality.json"
README_ALGO_BEGIN = "<!-- BEGIN ALGORITHM CATALOG -->"
README_ALGO_END = "<!-- END ALGORITHM CATALOG -->"

sys.path.insert(0, str(SRC_DIR))
from pvx.algorithms.registry import ALGORITHM_REGISTRY  # noqa: E402
from pvx.core import voc as voc_core  # noqa: E402


def git_commit_meta() -> tuple[str, str]:
    commit = "unknown"
    commit_date = "unknown"
    try:
        commit_proc = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_proc.returncode == 0 and commit_proc.stdout.strip():
            commit = commit_proc.stdout.strip()
        date_proc = subprocess.run(
            ["git", "-C", str(ROOT), "show", "-s", "--format=%cI", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if date_proc.returncode == 0 and date_proc.stdout.strip():
            commit_date = date_proc.stdout.strip()
    except Exception:
        pass
    return commit, commit_date


COMMIT_HASH, COMMIT_DATE = git_commit_meta()


def scholar(title: str) -> str:
    return f"https://scholar.google.com/scholar?q={quote_plus(title)}"


OUT_OF_SCOPE_TOKENS: tuple[str, ...] = ()


def _contains_out_of_scope_text(*parts: str) -> bool:
    text = " ".join(parts).lower()
    return any(token in text for token in OUT_OF_SCOPE_TOKENS)


def _is_out_of_scope_paper(paper: dict[str, str]) -> bool:
    return _contains_out_of_scope_text(
        str(paper.get("category", "")),
        str(paper.get("title", "")),
        str(paper.get("authors", "")),
        str(paper.get("venue", "")),
    )


def _is_out_of_scope_glossary(entry: dict[str, str]) -> bool:
    return _contains_out_of_scope_text(
        str(entry.get("term", "")),
        str(entry.get("category", "")),
        str(entry.get("description", "")),
    )


PAPERS: list[dict[str, str]] = [
    {
        "category": "Phase Vocoder Foundations",
        "authors": "J. L. Flanagan; R. M. Golden",
        "year": "1966",
        "title": "Phase Vocoder",
        "venue": "Bell System Technical Journal",
        "url": "https://doi.org/10.1002/j.1538-7305.1966.tb01706.x",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "M. R. Portnoff",
        "year": "1976",
        "title": "Implementation of the Digital Phase Vocoder Using the Fast Fourier Transform",
        "venue": "IEEE Trans. ASSP",
        "url": scholar("Implementation of the Digital Phase Vocoder Using the Fast Fourier Transform"),
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "M. R. Portnoff",
        "year": "1980",
        "title": "Time-Scale Modification of Speech Based on Short-Time Fourier Analysis",
        "venue": "IEEE Trans. ASSP",
        "url": scholar("Time-Scale Modification of Speech Based on Short-Time Fourier Analysis"),
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "J. B. Allen; L. R. Rabiner",
        "year": "1977",
        "title": "A Unified Approach to Short-Time Fourier Analysis and Synthesis",
        "venue": "Proceedings of the IEEE",
        "url": "https://doi.org/10.1109/PROC.1977.10770",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "D. W. Griffin; J. S. Lim",
        "year": "1984",
        "title": "Signal Estimation from Modified Short-Time Fourier Transform",
        "venue": "IEEE Trans. ASSP",
        "url": "https://doi.org/10.1109/TASSP.1984.1164317",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "M. Dolson",
        "year": "1986",
        "title": "The Phase Vocoder: A Tutorial",
        "venue": "Computer Music Journal",
        "url": "https://www.jstor.org/stable/3680093",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "R. J. McAulay; T. F. Quatieri",
        "year": "1986",
        "title": "Speech Analysis/Synthesis Based on a Sinusoidal Representation",
        "venue": "IEEE Trans. ASSP",
        "url": scholar("Speech Analysis/Synthesis Based on a Sinusoidal Representation"),
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "X. Serra; J. O. Smith",
        "year": "1990",
        "title": "Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based on a Deterministic Plus Stochastic Decomposition",
        "venue": "Computer Music Journal",
        "url": "https://www.jstor.org/stable/3680788",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "S. M. Bernsee",
        "year": "1999",
        "title": "Pitch Shifting Using the Fourier Transform",
        "venue": "DAFX Workshop Note",
        "url": "https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "J. Laroche; M. Dolson",
        "year": "1999",
        "title": "Improved Phase Vocoder Time-Scale Modification of Audio",
        "venue": "IEEE Trans. Speech and Audio Processing",
        "url": "https://doi.org/10.1109/89.759041",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "J. Laroche; M. Dolson",
        "year": "1999",
        "title": "New Phase-Vocoder Techniques for Pitch-Shifting, Harmonizing and Other Exotic Effects",
        "venue": "WASPAA",
        "url": "https://doi.org/10.1109/ASPAA.1999.810857",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "A. Röbel",
        "year": "2003",
        "title": "A New Approach to Transient Processing in the Phase Vocoder",
        "venue": "DAFx",
        "url": "https://www.dafx.de/paper-archive/2003/pdfs/dafx81.pdf",
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "C. Duxbury; M. Davies; M. Sandler",
        "year": "2002",
        "title": "Improved Time-Scaling of Musical Audio Using Phase Locking at Transients",
        "venue": "AES Convention",
        "url": scholar("Improved Time-Scaling of Musical Audio Using Phase Locking at Transients"),
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "J. Bonada",
        "year": "2000",
        "title": "Automatic Technique in Frequency Domain for Near-Lossless Time-Scale Modification of Audio",
        "venue": "ICMC",
        "url": scholar("Automatic Technique in Frequency Domain for Near-Lossless Time-Scale Modification of Audio"),
    },
    {
        "category": "Phase Vocoder Foundations",
        "authors": "N. Roucos; A. Wilgus",
        "year": "1985",
        "title": "High Quality Time-Scale Modification for Speech",
        "venue": "ICASSP",
        "url": scholar("High Quality Time-Scale Modification for Speech"),
    },
    {
        "category": "Time-Scale and Pitch Methods",
        "authors": "E. Moulines; F. Charpentier",
        "year": "1990",
        "title": "Pitch-Synchronous Waveform Processing Techniques for Text-to-Speech Synthesis Using Diphones",
        "venue": "Speech Communication",
        "url": "https://doi.org/10.1016/0167-6393(90)90021-Z",
    },
    {
        "category": "Time-Scale and Pitch Methods",
        "authors": "W. Verhelst; M. Roelands",
        "year": "1993",
        "title": "An Overlap-Add Technique Based on Waveform Similarity (WSOLA) for High Quality Time-Scale Modification of Speech",
        "venue": "ICASSP",
        "url": scholar("WSOLA overlap-add technique waveform similarity"),
    },
    {
        "category": "Time-Scale and Pitch Methods",
        "authors": "A. Röbel; X. Rodet",
        "year": "2005",
        "title": "Efficient Spectral Envelope Estimation and Its Application to Pitch Shifting and Envelope Preservation",
        "venue": "DAFx",
        "url": "https://www.dafx.de/paper-archive/2005/P_189.pdf",
    },
    {
        "category": "Time-Scale and Pitch Methods",
        "authors": "J. Driedger; M. Müller",
        "year": "2016",
        "title": "A Review of Time-Scale Modification of Music Signals",
        "venue": "Applied Sciences",
        "url": "https://doi.org/10.3390/app6020057",
    },
    {
        "category": "Time-Scale and Pitch Methods",
        "authors": "S. Arfib; D. Keiler; U. Zölzer",
        "year": "2002",
        "title": "DAFX: Digital Audio Effects (chapter references on pitch/time processing)",
        "venue": "Wiley",
        "url": scholar("DAFX Digital Audio Effects pitch shifting chapter"),
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "A. M. Noll",
        "year": "1967",
        "title": "Cepstrum Pitch Determination",
        "venue": "JASA",
        "url": scholar("Cepstrum Pitch Determination"),
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "L. R. Rabiner; M. J. Cheng; A. E. Rosenberg; C. A. McGonegal",
        "year": "1976",
        "title": "A Comparative Performance Study of Several Pitch Detection Algorithms",
        "venue": "IEEE Trans. ASSP",
        "url": scholar("A Comparative Performance Study of Several Pitch Detection Algorithms"),
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "D. Talkin",
        "year": "1995",
        "title": "A Robust Algorithm for Pitch Tracking (RAPT)",
        "venue": "In Speech Coding and Synthesis",
        "url": scholar("A Robust Algorithm for Pitch Tracking RAPT"),
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "P. Boersma",
        "year": "1993",
        "title": "Accurate Short-Term Analysis of the Fundamental Frequency and the Harmonics-to-Noise Ratio of a Sampled Sound",
        "venue": "IFA Proceedings",
        "url": scholar("Accurate Short-Term Analysis of the Fundamental Frequency and the Harmonics-to-Noise Ratio"),
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "A. de Cheveigné; H. Kawahara",
        "year": "2002",
        "title": "YIN, a Fundamental Frequency Estimator for Speech and Music",
        "venue": "JASA",
        "url": "https://doi.org/10.1121/1.1458024",
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "A. Camacho; J. G. Harris",
        "year": "2008",
        "title": "A Sawtooth Waveform Inspired Pitch Estimator for Speech and Music (SWIPE)",
        "venue": "JASA",
        "url": "https://doi.org/10.1121/1.2951592",
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "M. Mauch; S. Dixon",
        "year": "2014",
        "title": "pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions",
        "venue": "ICASSP",
        "url": "https://doi.org/10.1109/ICASSP.2014.6853678",
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "J. W. Kim; J. Salamon; P. Li; J. P. Bello",
        "year": "2018",
        "title": "CREPE: A Convolutional Representation for Pitch Estimation",
        "venue": "ICASSP",
        "url": "https://arxiv.org/abs/1802.06182",
    },
    {
        "category": "Pitch Detection and Tracking",
        "authors": "R. Bittner et al.",
        "year": "2017",
        "title": "Deep Salience Representations for F0 Estimation in Polyphonic Music",
        "venue": "ISMIR",
        "url": "https://arxiv.org/abs/1706.02292",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "J. C. Brown",
        "year": "1991",
        "title": "Calculation of a Constant Q Spectral Transform",
        "venue": "JASA",
        "url": "https://doi.org/10.1121/1.400476",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "J. C. Brown; M. S. Puckette",
        "year": "1992",
        "title": "An Efficient Algorithm for the Calculation of a Constant Q Transform",
        "venue": "JASA",
        "url": "https://doi.org/10.1121/1.404385",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "C. Schörkhuber; A. Klapuri",
        "year": "2010",
        "title": "Constant-Q Transform Toolbox for Music Processing",
        "venue": "SMC",
        "url": scholar("Constant-Q Transform Toolbox for Music Processing"),
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "G. Velasco; N. Holighaus; M. Dörfler; T. Grill",
        "year": "2011",
        "title": "Constructing an Invertible Constant-Q Transform with Nonstationary Gabor Frames",
        "venue": "DAFx",
        "url": "https://grrrr.org/data/publications/2011_VelascoHolighausDorflerGrill_DAFx.pdf",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "F. Auger; P. Flandrin",
        "year": "1995",
        "title": "Improving the Readability of Time-Frequency and Time-Scale Representations by the Reassignment Method",
        "venue": "IEEE Trans. SP",
        "url": "https://doi.org/10.1109/78.382394",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "P. Flandrin; F. Auger; E. Chassande-Mottin",
        "year": "2002",
        "title": "Time-Frequency Reassignment: From Principles to Algorithms",
        "venue": "Applications in Time-Frequency Signal Processing",
        "url": scholar("Time-Frequency Reassignment: From Principles to Algorithms"),
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "I. Daubechies; J. Lu; H.-T. Wu",
        "year": "2011",
        "title": "Synchrosqueezed Wavelet Transforms",
        "venue": "Applied and Computational Harmonic Analysis",
        "url": "https://doi.org/10.1016/j.acha.2010.08.002",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "R. R. Coifman; M. V. Wickerhauser",
        "year": "1992",
        "title": "Entropy-Based Algorithms for Best Basis Selection",
        "venue": "IEEE Trans. IT",
        "url": "https://doi.org/10.1109/18.119732",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "S. Mallat",
        "year": "1989",
        "title": "A Theory for Multiresolution Signal Decomposition: The Wavelet Representation",
        "venue": "IEEE Trans. PAMI",
        "url": "https://doi.org/10.1109/34.192463",
    },
    {
        "category": "Time-Frequency and Transform Methods",
        "authors": "S. Mann; S. Haykin",
        "year": "1991",
        "title": "The Chirplet Transform: Physical Considerations",
        "venue": "IEEE Trans. SP",
        "url": scholar("The Chirplet Transform: Physical Considerations"),
    },
    {
        "category": "Separation and Decomposition",
        "authors": "D. D. Lee; H. S. Seung",
        "year": "1999",
        "title": "Learning the Parts of Objects by Non-negative Matrix Factorization",
        "venue": "Nature",
        "url": "https://doi.org/10.1038/44565",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "D. D. Lee; H. S. Seung",
        "year": "2001",
        "title": "Algorithms for Non-negative Matrix Factorization",
        "venue": "NIPS",
        "url": "https://doi.org/10.1162/089976601750541778",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "T. Virtanen",
        "year": "2007",
        "title": "Monaural Sound Source Separation by Nonnegative Matrix Factorization with Temporal Continuity and Sparseness Criteria",
        "venue": "IEEE Trans. Audio, Speech, and Language Processing",
        "url": scholar("Monaural sound source separation by nonnegative matrix factorization with temporal continuity and sparseness criteria"),
    },
    {
        "category": "Separation and Decomposition",
        "authors": "D. Fitzgerald",
        "year": "2010",
        "title": "Harmonic/Percussive Separation Using Median Filtering",
        "venue": "DAFx",
        "url": "https://www.dafx.de/paper-archive/2010/DAFx10/Rx35.pdf",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "E. J. Candès; X. Li; Y. Ma; J. Wright",
        "year": "2011",
        "title": "Robust Principal Component Analysis?",
        "venue": "JACM",
        "url": "https://doi.org/10.1145/1970392.1970395",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "P. Comon",
        "year": "1994",
        "title": "Independent Component Analysis, A New Concept?",
        "venue": "Signal Processing",
        "url": "https://doi.org/10.1016/0165-1684(94)90029-9",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "A. Hyvärinen",
        "year": "1999",
        "title": "Fast and Robust Fixed-Point Algorithms for Independent Component Analysis",
        "venue": "IEEE Trans. Neural Networks",
        "url": "https://doi.org/10.1109/72.761722",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "A. Hyvärinen; E. Oja",
        "year": "2000",
        "title": "Independent Component Analysis: Algorithms and Applications",
        "venue": "Neural Networks",
        "url": "https://doi.org/10.1016/S0893-6080(00)00026-5",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "A. Défossez et al.",
        "year": "2019",
        "title": "Music Source Separation in the Waveform Domain",
        "venue": "arXiv",
        "url": "https://arxiv.org/abs/1911.13254",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "F.-R. Stöter; S. Uhlich; A. Liutkus; Y. Mitsufuji",
        "year": "2019",
        "title": "Open-Unmix - A Reference Implementation for Music Source Separation",
        "venue": "Journal of Open Source Software",
        "url": "https://doi.org/10.21105/joss.01667",
    },
    {
        "category": "Separation and Decomposition",
        "authors": "A. Jansson et al.",
        "year": "2017",
        "title": "Singing Voice Separation with Deep U-Net Convolutional Networks",
        "venue": "ISMIR",
        "url": "https://arxiv.org/abs/1706.09088",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "N. Wiener",
        "year": "1949",
        "title": "Extrapolation, Interpolation, and Smoothing of Stationary Time Series",
        "venue": "Book",
        "url": scholar("Extrapolation Interpolation and Smoothing of Stationary Time Series Wiener"),
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "S. F. Boll",
        "year": "1979",
        "title": "Suppression of Acoustic Noise in Speech Using Spectral Subtraction",
        "venue": "IEEE Trans. ASSP",
        "url": "https://doi.org/10.1109/TASSP.1979.1163209",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "Y. Ephraim; D. Malah",
        "year": "1984",
        "title": "Speech Enhancement Using a Minimum Mean-Square Error Short-Time Spectral Amplitude Estimator",
        "venue": "IEEE Trans. ASSP",
        "url": "https://doi.org/10.1109/TASSP.1984.1164453",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "Y. Ephraim; D. Malah",
        "year": "1985",
        "title": "Speech Enhancement Using a Minimum Mean-Square Error Log-Spectral Amplitude Estimator",
        "venue": "IEEE Trans. ASSP",
        "url": "https://doi.org/10.1109/TASSP.1985.1164550",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "P. Scalart; J. V. Filho",
        "year": "1996",
        "title": "Speech Enhancement Based on a Priori Signal to Noise Estimation",
        "venue": "ICASSP",
        "url": scholar("Speech Enhancement Based on a Priori Signal to Noise Estimation"),
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "R. Martin",
        "year": "2001",
        "title": "Noise Power Spectral Density Estimation Based on Optimal Smoothing and Minimum Statistics",
        "venue": "IEEE Trans. Speech and Audio Processing",
        "url": "https://doi.org/10.1109/89.917870",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "J.-M. Valin",
        "year": "2018",
        "title": "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement (RNNoise)",
        "venue": "MMSP / arXiv",
        "url": "https://arxiv.org/abs/1709.08243",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "T. Nakatani; M. Miyoshi; K. Kinoshita",
        "year": "2010",
        "title": "Speech Dereverberation Based on Variance-Normalized Delayed Linear Prediction",
        "venue": "IEEE Trans. Audio, Speech, and Language Processing",
        "url": "https://doi.org/10.1109/TASL.2010.2052251",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "Y. Yoshioka; T. Nakatani",
        "year": "2012",
        "title": "Generalization of Multi-Channel Linear Prediction Methods for Blind MIMO Impulse Response Shortening",
        "venue": "IEEE TASLP",
        "url": scholar("Generalization of Multi-Channel Linear Prediction Methods for Blind MIMO Impulse Response Shortening"),
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "J. Capon",
        "year": "1969",
        "title": "High-Resolution Frequency-Wavenumber Spectrum Analysis",
        "venue": "Proceedings of the IEEE",
        "url": "https://doi.org/10.1109/PROC.1969.7278",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "O. L. Frost III",
        "year": "1972",
        "title": "An Algorithm for Linearly Constrained Adaptive Array Processing",
        "venue": "Proceedings of the IEEE",
        "url": "https://doi.org/10.1109/PROC.1972.8817",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "C. Knapp; G. Carter",
        "year": "1976",
        "title": "The Generalized Correlation Method for Estimation of Time Delay",
        "venue": "IEEE Trans. ASSP",
        "url": "https://doi.org/10.1109/TASSP.1976.1162830",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "S. Pulkki",
        "year": "1997",
        "title": "Virtual Sound Source Positioning Using Vector Base Amplitude Panning",
        "venue": "JAES",
        "url": scholar("Virtual Sound Source Positioning Using Vector Base Amplitude Panning"),
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "A. W. Rix; J. G. Beerends; M. P. Hollier; A. P. Hekstra",
        "year": "2001",
        "title": "Perceptual Evaluation of Speech Quality (PESQ)",
        "venue": "ICASSP",
        "url": "https://doi.org/10.1109/ICASSP.2001.941023",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "C. H. Taal; R. C. Hendriks; R. Heusdens; J. Jensen",
        "year": "2011",
        "title": "An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech (STOI)",
        "venue": "IEEE TASLP",
        "url": "https://doi.org/10.1109/TASL.2010.2089940",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "M. H. C. de Gesmundo et al.",
        "year": "2019",
        "title": "ViSQOL v3: An Open Source Production Ready Objective Speech and Audio Metric",
        "venue": "QoMEX",
        "url": scholar("ViSQOL v3 objective speech and audio metric"),
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "ITU-R",
        "year": "2015",
        "title": "BS.1770-4: Algorithms to Measure Audio Programme Loudness and True-Peak Audio Level",
        "venue": "Recommendation",
        "url": "https://www.itu.int/rec/R-REC-BS.1770",
    },
    {
        "category": "Denoising, Dereverberation, and Spatial Audio",
        "authors": "EBU",
        "year": "2023",
        "title": "EBU R128: Loudness Normalisation and Permitted Maximum Level of Audio Signals",
        "venue": "Recommendation",
        "url": "https://tech.ebu.ch/publications/r128",
    },
]


DEFAULT_GLOSSARY: list[dict[str, str]] = [
    {
        "term": "Phase vocoder",
        "category": "Phase and Time-Scale DSP",
        "description": "Frequency-domain method for time-stretching/pitch-shifting via STFT phase propagation.",
        "url": "https://en.wikipedia.org/wiki/Phase_vocoder",
    },
    {
        "term": "Short-time Fourier transform (STFT)",
        "category": "Phase and Time-Scale DSP",
        "description": "Windowed Fourier transform used for local time-frequency analysis.",
        "url": "https://en.wikipedia.org/wiki/Short-time_Fourier_transform",
    },
    {
        "term": "WSOLA",
        "category": "Phase and Time-Scale DSP",
        "description": "Time-domain overlap-add method that aligns waveform-similar grains.",
        "url": "https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling",
    },
    {
        "term": "PSOLA",
        "category": "Phase and Time-Scale DSP",
        "description": "Pitch-synchronous overlap-add family for speech/music pitch modification.",
        "url": "https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling",
    },
    {
        "term": "Constant-Q transform (CQT)",
        "category": "Time-Frequency Transforms",
        "description": "Log-frequency transform with constant Q-factor per bin.",
        "url": "https://en.wikipedia.org/wiki/Constant-Q_transform",
    },
    {
        "term": "Wavelet transform",
        "category": "Time-Frequency Transforms",
        "description": "Multi-resolution analysis with scalable wavelet atoms.",
        "url": "https://en.wikipedia.org/wiki/Wavelet",
    },
    {
        "term": "Synchrosqueezing",
        "category": "Time-Frequency Transforms",
        "description": "Reassignment-like method that sharpens time-frequency ridges.",
        "url": "https://en.wikipedia.org/wiki/Wavelet_transform#Synchrosqueezing_transform",
    },
    {
        "term": "YIN",
        "category": "Pitch Detection",
        "description": "Difference-function-based F0 estimator for monophonic signals.",
        "url": "https://librosa.org/doc/main/generated/librosa.yin.html",
    },
    {
        "term": "pYIN",
        "category": "Pitch Detection",
        "description": "Probabilistic extension of YIN with voicing estimation.",
        "url": "https://librosa.org/doc/main/generated/librosa.pyin.html",
    },
    {
        "term": "SWIPE",
        "category": "Pitch Detection",
        "description": "Sawtooth-inspired frequency-domain fundamental estimator.",
        "url": "https://doi.org/10.1121/1.2951592",
    },
    {
        "term": "NMF",
        "category": "Separation and Decomposition",
        "description": "Nonnegative matrix factorization for source decomposition.",
        "url": "https://en.wikipedia.org/wiki/Non-negative_matrix_factorization",
    },
    {
        "term": "ICA",
        "category": "Separation and Decomposition",
        "description": "Independent component analysis for blind source separation.",
        "url": "https://en.wikipedia.org/wiki/Independent_component_analysis",
    },
    {
        "term": "RPCA",
        "category": "Separation and Decomposition",
        "description": "Robust PCA decomposition into low-rank and sparse components.",
        "url": "https://en.wikipedia.org/wiki/Robust_principal_component_analysis",
    },
    {
        "term": "Wiener filter",
        "category": "Denoising and Dereverb",
        "description": "Minimum mean-square error linear estimator.",
        "url": "https://en.wikipedia.org/wiki/Wiener_filter",
    },
    {
        "term": "MMSE-STSA",
        "category": "Denoising and Dereverb",
        "description": "Minimum-mean-square-error short-time spectral amplitude estimator.",
        "url": scholar("MMSE short-time spectral amplitude estimator"),
    },
    {
        "term": "Log-MMSE",
        "category": "Denoising and Dereverb",
        "description": "Log-spectral MMSE speech enhancement estimator.",
        "url": scholar("log-MMSE speech enhancement"),
    },
    {
        "term": "Weighted prediction error (WPE)",
        "category": "Denoising and Dereverb",
        "description": "Long-term linear prediction framework for dereverberation.",
        "url": "https://nara-wpe.readthedocs.io/en/latest/",
    },
    {
        "term": "LUFS / ITU-R BS.1770",
        "category": "Dynamics and Loudness",
        "description": "Loudness unit framework and measurement recommendation used in mastering.",
        "url": "https://en.wikipedia.org/wiki/LKFS",
    },
    {
        "term": "EBU R128",
        "category": "Dynamics and Loudness",
        "description": "Broadcast loudness normalization recommendation aligned to BS.1770.",
        "url": "https://tech.ebu.ch/publications/r128",
    },
    {
        "term": "Multiband compression",
        "category": "Dynamics and Loudness",
        "description": "Band-split compression with independent dynamics per band.",
        "url": "https://en.wikipedia.org/wiki/Audio_compressor#Multiband_compression",
    },
    {
        "term": "Ring modulation",
        "category": "Creative Effects",
        "description": "Multiplicative modulation creating sidebands at sum/difference frequencies.",
        "url": "https://en.wikipedia.org/wiki/Ring_modulation",
    },
    {
        "term": "Binaural audio",
        "category": "Spatial Audio",
        "description": "Two-channel rendering that preserves direction cues for headphones.",
        "url": "https://en.wikipedia.org/wiki/Binaural_recording",
    },
    {
        "term": "HRTF",
        "category": "Spatial Audio",
        "description": "Head-related transfer function used in binaural rendering.",
        "url": "https://en.wikipedia.org/wiki/Head-related_transfer_function",
    },
    {
        "term": "Interaural time difference (ITD)",
        "category": "Spatial Audio",
        "description": "Localization cue based on left-right arrival-time differences.",
        "url": "https://en.wikipedia.org/wiki/Interaural_time_difference",
    },
    {
        "term": "Interaural level difference (ILD)",
        "category": "Spatial Audio",
        "description": "Localization cue based on left-right level differences.",
        "url": "https://en.wikipedia.org/wiki/Sound_localization",
    },
    {
        "term": "VBAP",
        "category": "Spatial Audio",
        "description": "Vector-base amplitude panning across loudspeaker triplets/pairs.",
        "url": scholar("VBAP vector base amplitude panning"),
    },
    {
        "term": "DBAP",
        "category": "Spatial Audio",
        "description": "Distance-based amplitude panning driven by source-speaker distance.",
        "url": scholar("distance-based amplitude panning DBAP"),
    },
]


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "item"


def dedupe_papers(papers: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, str]] = []
    for paper in papers:
        key = (
            paper.get("year", "").strip(),
            paper.get("title", "").strip().lower(),
            paper.get("authors", "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(paper)
    return output


PAPER_URL_UPGRADES: dict[str, str] = {
    "MMSE short-time spectral amplitude estimator": "https://doi.org/10.1109/TASSP.1984.1164453",
    "log-MMSE speech enhancement": "https://doi.org/10.1109/TASSP.1985.1164550",
}


def _upgrade_paper_url(title: str, url: str) -> str:
    if "scholar.google.com" not in url:
        return url
    title_l = title.lower()
    for key, replacement in PAPER_URL_UPGRADES.items():
        if key.lower() in title_l:
            return replacement
    return url


def upgrade_paper_urls(papers: list[dict[str, str]]) -> list[dict[str, str]]:
    upgraded: list[dict[str, str]] = []
    for paper in papers:
        upgraded.append(
            {
                **paper,
                "url": _upgrade_paper_url(str(paper.get("title", "")), str(paper.get("url", ""))),
            }
        )
    return upgraded


def load_extra_papers() -> list[dict[str, str]]:
    if not EXTRA_PAPERS_PATH.exists():
        return []
    try:
        payload = json.loads(EXTRA_PAPERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    records: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if not {"category", "authors", "year", "title", "venue", "url"}.issubset(item):
            continue
        record = {k: str(item[k]) for k in ("category", "authors", "year", "title", "venue", "url")}
        if _is_out_of_scope_paper(record):
            continue
        records.append(record)
    return records


def load_glossary() -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    if GLOSSARY_PATH.exists():
        try:
            loaded = json.loads(GLOSSARY_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                for item in loaded:
                    if not isinstance(item, dict):
                        continue
                    if not {"term", "category", "description", "url"}.issubset(item):
                        continue
                    payload.append({k: str(item[k]) for k in ("term", "category", "description", "url")})
        except Exception:
            payload = []
    if not payload:
        payload = list(DEFAULT_GLOSSARY)
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for entry in payload:
        term_key = entry["term"].strip().lower()
        if _is_out_of_scope_glossary(entry):
            continue
        if term_key in seen:
            continue
        seen.add(term_key)
        out.append(entry)
    return out


PAPERS = [
    paper
    for paper in dedupe_papers(upgrade_paper_urls(PAPERS + load_extra_papers()))
    if not _is_out_of_scope_paper(paper)
]
TECHNICAL_GLOSSARY = [entry for entry in load_glossary() if not _is_out_of_scope_glossary(entry)]
GLOSSARY_LOOKUP = {entry["term"].lower(): entry for entry in TECHNICAL_GLOSSARY}


def infer_glossary_terms(*texts: str, limit: int = 6) -> list[dict[str, str]]:
    haystack = " ".join(texts).lower()
    norm_haystack = re.sub(r"[^a-z0-9]+", " ", haystack)
    matches: list[dict[str, str]] = []
    for entry in TECHNICAL_GLOSSARY:
        term = entry["term"]
        token = term.lower()
        token_norm = re.sub(r"[^a-z0-9]+", " ", token).strip()
        if not token_norm:
            continue
        if re.search(rf"\b{re.escape(token_norm)}\b", norm_haystack):
            matches.append(entry)
            continue
        # Support acronym-style terms (e.g., LUFS, STFT) without
        # matching arbitrary substrings inside unrelated words.
        compact = re.sub(r"[^a-z0-9]+", "", token)
        if compact and len(compact) <= 8 and re.search(rf"\b{re.escape(compact)}\b", norm_haystack):
            matches.append(entry)
    dedup: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in matches:
        key = entry["term"].lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(entry)
        if len(dedup) >= limit:
            break
    return dedup


def glossary_links_html(entries: list[dict[str, str]], *, page_prefix: str = "") -> str:
    if not entries:
        return "None"
    return ", ".join(
        f"<a href=\"{page_prefix}glossary.html#{escape(slugify(entry['term']))}\">{escape(entry['term'])}</a>"
        for entry in entries
    )


def load_json(path: Path, default: dict | list):
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if isinstance(default, dict) and not isinstance(payload, dict):
        return default
    if isinstance(default, list) and not isinstance(payload, list):
        return default
    return payload


def classify_reference_url(url: str) -> str:
    lower = url.lower()
    if "doi.org/" in lower:
        return "doi"
    if "arxiv.org" in lower:
        return "arxiv"
    if "scholar.google.com" in lower:
        return "scholar"
    if any(host in lower for host in ("ieeexplore", "acm.org", "springer", "sciencedirect", "wiley", "jstor", "itu.int", "tech.ebu.ch")):
        return "publisher_or_standard"
    return "web"


def window_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for name in voc_core.WINDOW_CHOICES:
        if name in voc_core._COSINE_SERIES_WINDOWS:
            coeffs = ", ".join(f"{c:g}" for c in voc_core._COSINE_SERIES_WINDOWS[name])
            family = "Cosine series"
            params = f"coeffs=({coeffs})"
            formula = "W1"
            note = "Cosine-sum taper balancing leakage and resolution."
        elif name == "sine":
            family = "Sinusoidal"
            params = "none"
            formula = "W2"
            note = "Raised-sine taper; smooth edges with moderate main-lobe width."
        elif name == "cosine":
            family = "Sinusoidal"
            params = "none"
            formula = "W2"
            note = "Same implementation as sine window in pvx."
        elif name == "bartlett":
            family = "Triangular"
            params = "none"
            formula = "W3"
            note = "Linear triangular taper; simple and fast."
        elif name == "boxcar":
            family = "Rectangular"
            params = "none"
            formula = "W0"
            note = "No tapering; best frequency resolution but strongest leakage."
        elif name == "rect":
            family = "Rectangular"
            params = "none"
            formula = "W0"
            note = "Alias of boxcar in pvx."
        elif name == "triangular":
            family = "Triangular"
            params = "none"
            formula = "W4"
            note = "Triangular taper variant with denominator $(N+1)/2$."
        elif name == "bartlett_hann":
            family = "Hybrid taper"
            params = "fixed coefficients"
            formula = "W5"
            note = "Blend of Bartlett-like slope and cosine curvature."
        elif name in voc_core._TUKEY_WINDOWS:
            family = "Tukey"
            params = f"alpha={voc_core._TUKEY_WINDOWS[name]:g}"
            formula = "W6"
            note = "Cosine tapers at edges with flat center region."
        elif name == "parzen":
            family = "Polynomial"
            params = "piecewise cubic"
            formula = "W7"
            note = "Smooth piecewise cubic window with strong sidelobe suppression."
        elif name == "lanczos":
            family = "Sinc"
            params = "none"
            formula = "W8"
            note = "Sinc-based taper with useful compromise for spectral analysis."
        elif name == "welch":
            family = "Quadratic"
            params = "none"
            formula = "W9"
            note = "Parabolic taper emphasizing center samples."
        elif name in voc_core._GAUSSIAN_WINDOWS:
            family = "Gaussian"
            params = f"sigma_ratio={voc_core._GAUSSIAN_WINDOWS[name]:g}"
            formula = "W10"
            note = "Bell-shaped taper; smaller sigma gives stronger edge attenuation."
        elif name in voc_core._GENERAL_GAUSSIAN_WINDOWS:
            power, sigma_ratio = voc_core._GENERAL_GAUSSIAN_WINDOWS[name]
            family = "Generalized Gaussian"
            params = f"power={power:g}, sigma_ratio={sigma_ratio:g}"
            formula = "W11"
            note = "Adjustable shape exponent controls shoulder steepness."
        elif name in voc_core._EXPONENTIAL_WINDOWS:
            family = "Exponential"
            params = f"tau_ratio={voc_core._EXPONENTIAL_WINDOWS[name]:g}"
            formula = "W12"
            note = "Symmetric exponential decay away from center."
        elif name in voc_core._CAUCHY_WINDOWS:
            family = "Cauchy / Lorentzian"
            params = f"gamma_ratio={voc_core._CAUCHY_WINDOWS[name]:g}"
            formula = "W13"
            note = "Heavy-tailed taper with slower side decay than Gaussian."
        elif name in voc_core._COSINE_POWER_WINDOWS:
            family = "Cosine power"
            params = f"power={voc_core._COSINE_POWER_WINDOWS[name]:g}"
            formula = "W14"
            note = "Raises sine taper to power $p$; higher $p$ narrows effective support."
        elif name in voc_core._HANN_POISSON_WINDOWS:
            family = "Hann-Poisson"
            params = f"alpha={voc_core._HANN_POISSON_WINDOWS[name]:g}"
            formula = "W15"
            note = "Hann multiplied by exponential envelope for stronger edge decay."
        elif name in voc_core._GENERAL_HAMMING_WINDOWS:
            family = "General Hamming"
            params = f"alpha={voc_core._GENERAL_HAMMING_WINDOWS[name]:.2f}"
            formula = "W16"
            note = "Hamming family with tunable cosine weight."
        elif name == "bohman":
            family = "Bohman"
            params = "none"
            formula = "W17"
            note = "Continuous slope window with cosine plus sine correction."
        elif name == "kaiser":
            family = "Kaiser-Bessel"
            params = "beta from --kaiser-beta"
            formula = "W18"
            note = "Adjustable trade-off window using modified Bessel function."
        else:
            family = "Other"
            params = "none"
            formula = "W0"
            note = "Window supported by pvx."
        pros, cons, usage = window_tradeoffs(name, family)
        entries.append(
            {
                "name": name,
                "family": family,
                "params": params,
                "formula": formula,
                "note": note,
                "pros": pros,
                "cons": cons,
                "usage": usage,
            }
        )
    return entries


def window_tradeoffs(name: str, family: str) -> tuple[str, str, str]:
    if name in {"hann", "hamming"}:
        return (
            "Balanced leakage suppression and frequency resolution.",
            "Not optimal for amplitude metering or extreme sidelobe rejection.",
            "Default choice for most pvx time-stretch and pitch-shift workflows.",
        )
    if name in {"boxcar", "rect"}:
        return (
            "Narrowest main lobe and maximal bin sharpness.",
            "Highest sidelobes and strongest leakage/phasiness on non-bin-centered content.",
            "Use only for controlled test tones or when leakage is acceptable.",
        )
    if name in {"blackman", "blackmanharris", "blackman_nuttall", "nuttall", "exact_blackman"}:
        return (
            "Strong sidelobe suppression for cleaner spectral separation.",
            "Wider main lobe than Hann/Hamming.",
            "Use for dense harmonic material when leakage artifacts dominate.",
        )
    if name == "flattop":
        return (
            "Very accurate amplitude estimation in FFT bins.",
            "Very wide main lobe and reduced frequency discrimination.",
            "Use for measurement-grade magnitude tracking, not fine pitch separation.",
        )
    if name == "kaiser":
        return (
            "Continuously tunable width/sidelobe tradeoff via beta.",
            "Needs beta tuning; poor beta choices can over-blur or under-suppress sidelobes.",
            "Use when you need explicit control over leakage versus resolution.",
        )
    if family == "Tukey":
        return (
            "Interpolates between rectangular and Hann behavior.",
            "Behavior changes strongly with alpha and can be inconsistent across presets.",
            "Use when you need a controllable flat center with soft edges.",
        )
    if family == "Gaussian":
        return (
            "Smooth bell taper with low ringing and predictable decay.",
            "Frequency resolution changes noticeably with sigma.",
            "Use sigma presets to tune transient locality versus spectral leakage.",
        )
    if family == "Generalized Gaussian":
        return (
            "Additional shape control beyond standard Gaussian.",
            "Extra parameterization increases tuning complexity.",
            "Use for research/tuning tasks where shoulder steepness matters.",
        )
    if family == "Exponential":
        return (
            "Simple edge decay with fast computation.",
            "Can produce less-uniform center weighting than cosine families.",
            "Use for experiments emphasizing center-heavy weighting.",
        )
    if family == "Cauchy / Lorentzian":
        return (
            "Heavy tails preserve more peripheral samples than Gaussian.",
            "Tail energy can increase leakage relative to steeper tapers.",
            "Use when you want softer attenuation of far-window samples.",
        )
    if family == "Cosine power":
        return (
            "Simple power parameter controls edge steepness.",
            "Higher powers can overly narrow effective support.",
            "Use to smoothly increase edge attenuation versus basic sine.",
        )
    if family == "Hann-Poisson":
        return (
            "Combines smooth Hann center with exponential edge suppression.",
            "Can over-attenuate edges for high alpha values.",
            "Use when you need stronger edge decay than Hann without full flattop cost.",
        )
    if family == "General Hamming":
        return (
            "Tunable Hamming-style cosine weighting.",
            "Less standardized than classic Hann/Hamming choices.",
            "Use to sweep sidelobe-vs-width behavior near Hamming family defaults.",
        )
    if family == "Triangular":
        return (
            "Cheap linear taper with intuitive behavior.",
            "Higher sidelobes than stronger cosine-sum windows.",
            "Use for low-cost processing or quick exploratory runs.",
        )
    if name == "bartlett_hann":
        return (
            "Moderate leakage suppression with lightweight computation.",
            "Less common and less interpretable than standard Hann/Hamming.",
            "Use as a middle-ground taper when Bartlett feels too sharp.",
        )
    if name == "parzen":
        return (
            "Smooth high-order taper with good sidelobe control.",
            "Broader main lobe than lightweight windows.",
            "Use for spectral denoising/analysis where sidelobe cleanup is critical.",
        )
    if name == "lanczos":
        return (
            "Sinc-derived shape with useful compromise behavior.",
            "Can exhibit oscillatory spectral behavior versus cosine-sum defaults.",
            "Use for interpolation-adjacent analysis experiments.",
        )
    if name == "welch":
        return (
            "Center-emphasizing parabola with simple form.",
            "Not as strong at sidelobe suppression as Blackman-family windows.",
            "Use when center weighting is desired with minimal complexity.",
        )
    if name in {"sine", "cosine"}:
        return (
            "Smooth endpoint behavior and straightforward implementation.",
            "Less configurable than Kaiser/Tukey families.",
            "Use for stable, low-complexity alternatives to Hann.",
        )
    if name == "bohman":
        return (
            "Continuous-slope taper with good qualitative leakage control.",
            "Less common in audio tooling and harder to tune by intuition.",
            "Use for exploratory spectral work needing smooth derivatives.",
        )
    return (
        "Supported by pvx and compatible with the shared phase-vocoder path.",
        "Tradeoffs depend on the specific shape parameters.",
        "Start with Hann/Hamming, then compare this window if artifacts persist.",
    )


WINDOW_EQUATIONS_HTML = """
<div class="card">
  <h2>Window Formula Key</h2>
  <p class="small">
    Let $n=0,\\dots,N-1$, center index $m=(N-1)/2$, and normalized coordinate
    $x_n=(n-m)/m$.
  </p>
  <p><strong>(W0) Rectangular:</strong> $$w[n]=1$$</p>
  <p><strong>(W1) Cosine series:</strong> $$w[n]=\\sum_{k=0}^{K} a_k\\cos\\left(\\frac{2\\pi k n}{N-1}\\right)$$</p>
  <p><strong>(W2) Sine/Cosine:</strong> $$w[n]=\\sin\\left(\\frac{\\pi n}{N-1}\\right)$$</p>
  <p><strong>(W3) Bartlett:</strong> $$w[n]=1-\\left|\\frac{n-m}{m}\\right|$$</p>
  <p><strong>(W4) Triangular:</strong> $$w[n]=\\max\\left(1-\\left|\\frac{n-m}{(N+1)/2}\\right|,0\\right)$$</p>
  <p><strong>(W5) Bartlett-Hann:</strong> $$x=\\frac{n}{N-1}-\\frac{1}{2},\\quad w[n]=0.62-0.48|x|+0.38\\cos(2\\pi x)$$</p>
  <p><strong>(W6) Tukey:</strong></p>
  <p>$$x=\\frac{n}{N-1},\\quad w[n]=\\begin{cases}
  \\frac{1}{2}\\left(1+\\cos\\left(\\pi\\left(\\frac{2x}{\\alpha}-1\\right)\\right)\\right), & 0\\le x<\\frac{\\alpha}{2} \\\\
  1, & \\frac{\\alpha}{2}\\le x<1-\\frac{\\alpha}{2} \\\\
  \\frac{1}{2}\\left(1+\\cos\\left(\\pi\\left(\\frac{2x}{\\alpha}-\\frac{2}{\\alpha}+1\\right)\\right)\\right), & 1-\\frac{\\alpha}{2}\\le x\\le 1
  \\end{cases}$$</p>
  <p class="small">pvx special cases: $\\alpha\\le 0$ is rectangular and $\\alpha\\ge 1$ becomes Hann.</p>
  <p><strong>(W7) Parzen:</strong></p>
  <p>$$u=\\left|\\frac{2n}{N-1}-1\\right|,\\quad w[n]=\\begin{cases}
  1-6u^2+6u^3, & 0\\le u\\le \\frac{1}{2} \\\\
  2(1-u)^3, & \\frac{1}{2}<u\\le 1 \\\\
  0, & u>1
  \\end{cases}$$</p>
  <p><strong>(W8) Lanczos:</strong> $$w[n]=\\mathrm{sinc}\\left(\\frac{2n}{N-1}-1\\right)$$</p>
  <p><strong>(W9) Welch:</strong> $$w[n]=\\max\\left(1-x_n^2,0\\right)$$</p>
  <p><strong>(W10) Gaussian:</strong> $$w[n]=\\exp\\left(-\\frac{1}{2}\\left(\\frac{n-m}{\\sigma}\\right)^2\\right),\\ \\sigma=r_\\sigma m$$</p>
  <p><strong>(W11) General Gaussian:</strong> $$w[n]=\\exp\\left(-\\frac{1}{2}\\left|\\frac{n-m}{\\sigma}\\right|^{2p}\\right)$$</p>
  <p><strong>(W12) Exponential:</strong> $$w[n]=\\exp\\left(-\\frac{|n-m|}{\\tau}\\right),\\ \\tau=r_\\tau m$$</p>
  <p><strong>(W13) Cauchy:</strong> $$w[n]=\\frac{1}{1+\\left(\\frac{n-m}{\\gamma}\\right)^2},\\ \\gamma=r_\\gamma m$$</p>
  <p><strong>(W14) Cosine power:</strong> $$w[n]=\\sin\\left(\\frac{\\pi n}{N-1}\\right)^p$$</p>
  <p><strong>(W15) Hann-Poisson:</strong> $$w[n]=w_{\\text{Hann}}[n]\\exp\\left(-\\alpha\\frac{|n-m|}{m}\\right)$$</p>
  <p><strong>(W16) General Hamming:</strong> $$w[n]=\\alpha-(1-\\alpha)\\cos\\left(\\frac{2\\pi n}{N-1}\\right)$$</p>
  <p><strong>(W17) Bohman:</strong> $$x=\\left|\\frac{2n}{N-1}-1\\right|,\\ \\ w[n]=(1-x)\\cos(\\pi x)+\\frac{\\sin(\\pi x)}{\\pi}$$</p>
  <p><strong>(W18) Kaiser:</strong> $$w[n]=\\frac{I_0\\left(\\beta\\sqrt{1-r_n^2}\\right)}{I_0(\\beta)},\\quad r_n=\\frac{n-m}{m}$$</p>
  <p class="small">Each window name in the table maps to one formula family above plus fixed constants/shape parameters.</p>
</div>
""".strip()

def _split_top_level_once(text: str, delimiter: str = ",") -> tuple[str, str | None]:
    depth = 0
    quote: str | None = None
    escaped = False
    for idx, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            continue
        if ch == delimiter and depth == 0:
            return text[:idx], text[idx + 1 :]
    return text, None


def _extract_params_get_calls(line: str) -> list[tuple[str, str | None]]:
    calls: list[tuple[str, str | None]] = []
    marker = "params.get("
    search_pos = 0
    while True:
        start = line.find(marker, search_pos)
        if start < 0:
            break
        i = start + len(marker)
        depth = 1
        quote: str | None = None
        escaped = False
        while i < len(line):
            ch = line[i]
            if quote is not None:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1

        if i >= len(line):
            break
        inner = line[start + len(marker) : i].strip()
        left, right = _split_top_level_once(inner, ",")
        key = left.strip().strip('"').strip("'")
        default = right.strip() if right is not None else None
        if key:
            calls.append((key, default))
        search_pos = i + 1
    return calls


def extract_algorithm_param_specs(base_path: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    text = base_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    key_map: dict[str, list[str]] = {}
    default_map: dict[str, dict[str, str]] = {}

    current_slug: str | None = None
    keys_bucket: list[str] = []
    defaults_bucket: dict[str, str] = {}

    def commit() -> None:
        nonlocal current_slug, keys_bucket, defaults_bucket
        if current_slug is None:
            return
        dedup: list[str] = []
        for item in keys_bucket:
            if item not in dedup:
                dedup.append(item)
        key_map[current_slug] = dedup
        default_map[current_slug] = dict(defaults_bucket)

    for line in lines:
        line_s = line.strip()
        if line_s.startswith('if slug == "') or line_s.startswith('elif slug == "'):
            if current_slug is not None:
                commit()
            current_slug = line_s.split('"')[1]
            keys_bucket = []
            defaults_bucket = {}
            continue
        if current_slug is None or "params.get(" not in line_s:
            continue
        for key, default in _extract_params_get_calls(line_s):
            keys_bucket.append(key)
            if default is not None and key not in defaults_bucket:
                defaults_bucket[key] = default

    if current_slug is not None:
        commit()
    return key_map, default_map


def extract_algorithm_params(base_path: Path) -> dict[str, list[str]]:
    key_map, _ = extract_algorithm_param_specs(base_path)
    return key_map


def extract_module_cli_flags(module_path: Path) -> list[str]:
    try:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    flags: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flag = str(arg.value).strip()
                if flag and flag not in flags:
                    flags.append(flag)
    return flags


def collect_algorithm_module_flags(groups: OrderedDict[str, list[tuple[str, dict[str, str]]]]) -> dict[str, list[str]]:
    by_slug: dict[str, list[str]] = {}
    for items in groups.values():
        for algorithm_id, meta in items:
            _, slug = algorithm_id.split(".", 1)
            module_path = ROOT / "src" / Path(*str(meta["module"]).split(".")) .with_suffix(".py")
            by_slug[slug] = extract_module_cli_flags(module_path)
    return by_slug


def sample_value_from_default(key: str, default_expr: str | None) -> str:
    if default_expr is not None:
        return default_expr.strip()
    key_l = key.lower()
    if key_l == "scale_cents":
        return "[0.0, 200.0, 400.0, 500.0, 700.0, 900.0, 1100.0]"
    if key_l.endswith("_hz"):
        return "440.0"
    if "seed" in key_l:
        return "1307"
    if key_l.startswith(("is_", "has_", "use_", "enable_", "apply_")):
        return "True"
    if "channels" in key_l:
        return "2"
    if any(token in key_l for token in ("path", "file", "name")):
        return "\"example\""
    if any(token in key_l for token in ("ratio", "amount", "strength", "mix", "depth")):
        return "1.0"
    if any(token in key_l for token in ("threshold", "floor", "gain", "db", "ms")):
        return "0.0"
    return "<value>"


def format_sample_params(keys: list[str], defaults: dict[str, str], *, max_items: int = 6) -> str:
    if not keys:
        return "{}"
    parts: list[str] = []
    for key in keys[:max_items]:
        value = sample_value_from_default(key, defaults.get(key))
        parts.append(f"{key}={value}")
    if len(keys) > max_items:
        parts.append("...")
    return "{" + ", ".join(parts) + "}"


def compute_unique_cli_flags(
    params: dict[str, list[str]],
    module_flags_by_slug: dict[str, list[str]],
) -> dict[str, list[str]]:
    key_counts: Counter[str] = Counter()
    for keys in params.values():
        key_counts.update(keys)

    module_flag_counts: Counter[str] = Counter()
    for flags in module_flags_by_slug.values():
        module_flag_counts.update(flags)

    result: dict[str, list[str]] = {}
    for slug, keys in params.items():
        unique: list[str] = []
        for flag in module_flags_by_slug.get(slug, []):
            if module_flag_counts.get(flag, 0) == 1 and flag not in unique:
                unique.append(flag)
        for key in keys:
            if key_counts.get(key, 0) == 1:
                flag = f"--{key.replace('_', '-')}"
                if flag not in unique:
                    unique.append(flag)
        result[slug] = unique
    return result


def grouped_algorithms() -> OrderedDict[str, list[tuple[str, dict[str, str]]]]:
    groups: OrderedDict[str, list[tuple[str, dict[str, str]]]] = OrderedDict()
    for algorithm_id, meta in ALGORITHM_REGISTRY.items():
        folder, _ = algorithm_id.split(".", 1)
        groups.setdefault(folder, []).append((algorithm_id, meta))
    return groups


def html_page(
    title: str,
    content: str,
    *,
    css_path: str,
    breadcrumbs: str = "",
    include_mermaid: bool = False,
) -> str:
    mermaid_header = ""
    if include_mermaid:
        mermaid_header = (
            "  <script type=\"module\">\n"
            "    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';\n"
            "    mermaid.initialize({ startOnLoad: true, securityLevel: 'loose' });\n"
            "  </script>\n"
        )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(title)}</title>
  <link rel=\"stylesheet\" href=\"{css_path}\" />
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
{mermaid_header}</head>
<body>
  <header class=\"site-header\">
    <div class=\"wrap\">
      <h1>{escape(title)}</h1>
      {breadcrumbs}
    </div>
  </header>
  <main class=\"wrap\">{content}</main>
  <footer class=\"site-footer\">
    <div class=\"wrap\">
      Generated by <code>scripts_generate_html_docs.py</code> from commit <code>{escape(COMMIT_HASH)}</code>
      (commit date: {escape(COMMIT_DATE)}).
    </div>
  </footer>
</body>
</html>
"""


def write_style_css() -> None:
    css = """
:root {
  --bg: #f7f8fa;
  --card: #ffffff;
  --text: #13202b;
  --muted: #4d5d6b;
  --line: #d8e0e6;
  --accent: #005f73;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
.wrap {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 16px;
}
.site-header {
  background: linear-gradient(120deg, #0b3a53 0%, #005f73 60%, #0a9396 100%);
  color: #fff;
  padding: 22px 0;
}
.site-header h1 {
  margin: 0 0 8px;
  font-size: 1.8rem;
}
.site-footer {
  border-top: 1px solid var(--line);
  margin-top: 24px;
  padding: 12px 0 20px;
  color: var(--muted);
  font-size: 0.9rem;
}
nav a, a {
  color: var(--accent);
  text-decoration: none;
}
nav a:hover, a:hover {
  text-decoration: underline;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 14px;
  margin: 16px 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 12px 0 24px;
  background: var(--card);
  border: 1px solid var(--line);
}
th, td {
  border: 1px solid var(--line);
  padding: 8px 10px;
  vertical-align: top;
  text-align: left;
  font-size: 0.95rem;
}
th {
  background: #edf3f7;
  font-weight: 700;
}
code {
  background: #eef5f8;
  border: 1px solid #d8e6ec;
  border-radius: 4px;
  padding: 1px 5px;
}
.kicker {
  color: #d7eef2;
  font-size: 0.95rem;
}
.small {
  color: var(--muted);
  font-size: 0.9rem;
}
.pill {
  display: inline-block;
  border: 1px solid #b6cad3;
  background: #edf5f9;
  border-radius: 999px;
  padding: 2px 10px;
  margin: 2px 6px 2px 0;
  font-size: 0.82rem;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
pre.mermaid {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px;
  overflow-x: auto;
}
@media (max-width: 900px) {
  table { display: block; overflow-x: auto; white-space: nowrap; }
}
""".strip() + "\n"
    (DOCS_HTML_DIR / "style.css").write_text(css, encoding="utf-8")


def render_index(groups: OrderedDict[str, list[tuple[str, dict[str, str]]]], params: dict[str, list[str]]) -> None:
    total_algorithms = sum(len(items) for items in groups.values())
    total_param_keys = len({k for values in params.values() for k in values})
    unique_concepts = len(TECHNICAL_GLOSSARY)
    content_parts: list[str] = []

    content_parts.append(
        """
<div class=\"card\">
  <p>
    This HTML documentation is organized by algorithm folder/theme and generated from
    <code>src/pvx/algorithms/registry.py</code> and
    <code>src/pvx/algorithms/base.py</code> dispatch parameters.
  </p>
  <p>
    <strong>Totals:</strong>
  </p>
  <ul>
    <li><strong>{total_algorithms}</strong> algorithms</li>
    <li><strong>{group_count}</strong> folders/themes</li>
    <li><strong>{param_count}</strong> distinct algorithm parameter keys in dispatch</li>
    <li><strong>{glossary_count}</strong> linked technical glossary concepts</li>
    <li><strong>{paper_count}</strong> bibliography references</li>
  </ul>
</div>
""".format(
            total_algorithms=total_algorithms,
            group_count=len(groups),
            param_count=total_param_keys,
            glossary_count=unique_concepts,
            paper_count=len(PAPERS),
        )
    )

    content_parts.append(
        f"""
<div class=\"card\">
  <p>
    <a href=\"papers.html\"><strong>Research bibliography ({len(PAPERS)} references)</strong></a>
    - foundational and directly related literature for phase vocoder, time-scale/pitch,
    time-frequency transforms, separation, denoising, dereverberation, spatial audio, and mastering workflows.
  </p>
  <p>
    <a href=\"glossary.html\"><strong>Linked technical glossary ({unique_concepts} terms)</strong></a>
    - quick definitions and external references (Wikipedia, standards pages, docs, and papers).
  </p>
  <p>
    <a href=\"math.html\"><strong>Mathematical foundations</strong></a> and
    <a href=\"windows.html\"><strong>window reference (all {len(voc_core.WINDOW_CHOICES)} windows)</strong></a>.
  </p>
  <p>
    <a href=\"architecture.html\"><strong>Architecture diagrams</strong></a>,
    <a href=\"limitations.html\"><strong>algorithm limitations</strong></a>,
    <a href=\"benchmarks.html\"><strong>benchmark report</strong></a>,
    <a href=\"cookbook.html\"><strong>pipeline cookbook</strong></a>,
    <a href=\"cli_flags.html\"><strong>CLI flag index</strong></a>, and
    <a href=\"citations.html\"><strong>citation quality report</strong></a>.
  </p>
  <p class=\"small\">
    GitHub usually shows <code>.html</code> files as source. For browser-rendered math on GitHub itself,
    use the Markdown mirrors:
    <a href=\"../MATHEMATICAL_FOUNDATIONS.md\"><code>docs/MATHEMATICAL_FOUNDATIONS.md</code></a> and
    <a href=\"../WINDOW_REFERENCE.md\"><code>docs/WINDOW_REFERENCE.md</code></a>.
  </p>
</div>
"""
    )

    rows = []
    for folder, items in groups.items():
        theme = items[0][1]["theme"] if items else folder
        rows.append(
            "<tr>"
            f"<td><code>{escape(folder)}</code></td>"
            f"<td>{escape(theme)}</td>"
            f"<td>{len(items)}</td>"
            f"<td><a href=\"groups/{escape(folder)}.html\">Open</a></td>"
            "</tr>"
        )

    content_parts.append(
        """
<h2>Algorithm Groups</h2>
<table>
  <thead>
    <tr>
      <th>Folder</th>
      <th>Theme</th>
      <th>Algorithms</th>
      <th>Page</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".format(rows="\n".join(rows))
    )

    html = html_page(
        "pvx HTML Documentation",
        "\n".join(content_parts),
        css_path="style.css",
        breadcrumbs="<p class=\"kicker\">Index of grouped algorithm docs, linked glossary, and bibliography</p>",
    )
    (DOCS_HTML_DIR / "index.html").write_text(html, encoding="utf-8")


def module_path_from_meta(meta: dict[str, str]) -> str:
    return "src/" + meta["module"].replace(".", "/") + ".py"


def render_group_pages(groups: OrderedDict[str, list[tuple[str, dict[str, str]]]], params: dict[str, list[str]]) -> None:
    for folder, items in groups.items():
        theme = items[0][1]["theme"] if items else folder
        rows: list[str] = []
        subgroup_counts: OrderedDict[str, int] = OrderedDict()
        for algorithm_id, meta in items:
            _, slug = algorithm_id.split(".", 1)
            module_path = module_path_from_meta(meta)
            module_parts = meta["module"].split(".")
            subgroup = module_parts[-2] if len(module_parts) > 4 else "core"
            subgroup_counts[subgroup] = subgroup_counts.get(subgroup, 0) + 1
            keys = params.get(slug, [])
            key_text = ", ".join(f"<code>{escape(k)}</code>" for k in keys) if keys else "None (generic/default path)"
            concepts = infer_glossary_terms(algorithm_id, meta["name"], subgroup, theme, limit=6)
            rows.append(
                "<tr>"
                f"<td><code>{escape(algorithm_id)}</code></td>"
                f"<td>{escape(meta['name'])}</td>"
                f"<td><code>{escape(subgroup)}</code></td>"
                f"<td><code>{escape(module_path)}</code></td>"
                f"<td>{key_text}</td>"
                f"<td>{glossary_links_html(concepts, page_prefix='../')}</td>"
                "</tr>"
            )

        subgroup_pills = " ".join(
            f"<span class=\"pill\"><span class=\"mono\">{escape(name)}</span> ({count})</span>"
            for name, count in subgroup_counts.items()
        )

        content = (
            f"<div class=\"card\"><p><strong>Theme:</strong> {escape(theme)}</p>"
            f"<p><strong>Folder:</strong> <code>{escape(folder)}</code> | <strong>Algorithms:</strong> {len(items)}</p>"
            f"<p><strong>Subgroups:</strong> {subgroup_pills or '<span class="small">None</span>'}</p>"
            "</div>"
            "<table>"
            "<thead><tr><th>Algorithm ID</th><th>Name</th><th>Subgroup</th><th>Module Path</th><th>Parameter keys</th><th>Concept links</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

        breadcrumbs = (
            "<nav>"
            "<a href=\"../index.html\">Home</a> | "
            "<a href=\"../papers.html\">Research papers</a> | "
            "<a href=\"../glossary.html\">Technical glossary</a> | "
            "<a href=\"../math.html\">Math</a> | "
            "<a href=\"../windows.html\">Windows</a> | "
            "<a href=\"../architecture.html\">Architecture</a> | "
            "<a href=\"../limitations.html\">Limitations</a>"
            "</nav>"
        )
        html = html_page(
            f"pvx Group: {theme}",
            content,
            css_path="../style.css",
            breadcrumbs=breadcrumbs,
        )
        (GROUPS_DIR / f"{folder}.html").write_text(html, encoding="utf-8")


def render_papers_page() -> None:
    by_category: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for paper in PAPERS:
        by_category.setdefault(paper["category"], []).append(paper)

    sections: list[str] = []
    sections.append(
        """
<div class=\"card\">
  <p>
    This bibliography collects foundational and directly related literature that informed
    pvx's phase-vocoder-centric architecture and the broader DSP algorithm roadmap.
  </p>
  <p>
    Total references: <strong>{count}</strong> across <strong>{categories}</strong> categories.
  </p>
  <p class=\"small\">
    Links point to DOI pages, publisher archives, arXiv, standards documents, project docs,
    or Google Scholar queries where an official landing page can vary by publisher access.
  </p>
</div>
""".format(count=len(PAPERS), categories=len(by_category))
    )

    toc_items = "".join(
        f"<li><a href=\"#{escape(slugify(category))}\">{escape(category)} ({len(by_category[category])})</a></li>"
        for category in by_category
    )
    sections.append(f"<div class=\"card\"><h2>Categories</h2><ul>{toc_items}</ul></div>")

    for category, papers in by_category.items():
        anchor = slugify(category)
        rows = []
        for p in sorted(papers, key=lambda item: (item.get("year", ""), item.get("title", "")), reverse=True):
            link_type = classify_reference_url(p["url"])
            rows.append(
                "<tr>"
                f"<td>{escape(p['year'])}</td>"
                f"<td>{escape(p['authors'])}</td>"
                f"<td>{escape(p['title'])}</td>"
                f"<td>{escape(p['venue'])}</td>"
                f"<td><code>{escape(link_type)}</code></td>"
                f"<td><a href=\"{escape(p['url'])}\" target=\"_blank\" rel=\"noopener\">Link</a></td>"
                "</tr>"
            )
        sections.append(
            f"<h2 id=\"{escape(anchor)}\">{escape(category)}</h2>"
            "<table>"
            "<thead><tr><th>Year</th><th>Authors</th><th>Title</th><th>Venue</th><th>Link type</th><th>Link</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"glossary.html\">Technical glossary</a> | "
        "<a href=\"math.html\">Math</a> | "
        "<a href=\"windows.html\">Windows</a> | "
        "<a href=\"citations.html\">Citation quality</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Research Bibliography (Phase Vocoder and Related DSP)",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "papers.html").write_text(html, encoding="utf-8")


def render_glossary_page() -> None:
    by_category: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for entry in TECHNICAL_GLOSSARY:
        by_category.setdefault(entry["category"], []).append(entry)

    sections: list[str] = []
    sections.append(
        """
<div class=\"card\">
  <p>
    Linked glossary for core concepts used throughout pvx algorithms, CLIs, and research docs.
    Entries include concise definitions plus external references (Wikipedia, standards pages,
    project docs, and canonical papers).
  </p>
  <p>
    Total terms: <strong>{count}</strong> across <strong>{categories}</strong> categories.
  </p>
</div>
""".format(count=len(TECHNICAL_GLOSSARY), categories=len(by_category))
    )

    toc = "".join(
        f"<li><a href=\"#{escape(slugify(category))}\">{escape(category)} ({len(entries)})</a></li>"
        for category, entries in by_category.items()
    )
    sections.append(f"<div class=\"card\"><h2>Categories</h2><ul>{toc}</ul></div>")

    for category, entries in by_category.items():
        rows: list[str] = []
        for entry in sorted(entries, key=lambda item: item["term"].lower()):
            rows.append(
                "<tr>"
                f"<td id=\"{escape(slugify(entry['term']))}\">{escape(entry['term'])}</td>"
                f"<td>{escape(entry['description'])}</td>"
                f"<td><a href=\"{escape(entry['url'])}\" target=\"_blank\" rel=\"noopener\">Reference</a></td>"
                "</tr>"
            )
        sections.append(
            f"<h2 id=\"{escape(slugify(category))}\">{escape(category)}</h2>"
            "<table>"
            "<thead><tr><th>Term</th><th>Description</th><th>External link</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"papers.html\">Research papers</a> | "
        "<a href=\"math.html\">Math</a> | "
        "<a href=\"windows.html\">Windows</a> | "
        "<a href=\"architecture.html\">Architecture</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Technical Glossary",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "glossary.html").write_text(html, encoding="utf-8")


def render_math_page() -> None:
    sections: list[str] = []
    sections.append(
        """
<div class=\"card\">
  <p>
    Mathematical summary of the core signal-processing model used across pvx.
    Equations are rendered with MathJax when this page is opened in a normal browser.
  </p>
  <p class=\"small\">
    GitHub usually displays HTML source text. For GitHub-native math rendering, use
    <a href=\"../MATHEMATICAL_FOUNDATIONS.md\"><code>docs/MATHEMATICAL_FOUNDATIONS.md</code></a>.
  </p>
</div>
"""
    )
    sections.append(
        """
<div class=\"card\">
  <h2>STFT Analysis and Synthesis</h2>
  <p><strong>Analysis:</strong></p>
  <p>$$X_t[k]=\\sum_{n=0}^{N-1} x[n+tH_a]w[n]e^{-j2\\pi kn/N}$$</p>
  <p class=\"small\">Each frame $t$ is windowed by $w[n]$ and transformed to complex bins $k$.</p>
  <p><strong>Synthesis:</strong></p>
  <p>$$\\hat{x}[n]=\\frac{\\sum_t \\hat{x}_t[n-tH_s]w[n-tH_s]}{\\sum_t w^2[n-tH_s]+\\varepsilon}$$</p>
  <p class=\"small\">Overlap-add with energy normalization preserves level stability.</p>
</div>
"""
    )
    sections.append(
        """
<div class=\"card\">
  <h2>Transform Backend Selection (<code>--transform</code>)</h2>
  <p>
    pvx lets you choose the per-frame transform backend used in analysis/resynthesis:
    <code>fft</code>, <code>dft</code>, <code>czt</code>, <code>dct</code>, <code>dst</code>, and <code>hartley</code>.
  </p>
  <p><strong>Fourier family (<code>fft</code>, <code>dft</code>):</strong></p>
  <p>$$X_t[k]=\\sum_{n=0}^{N-1} x_t[n]e^{-j2\\pi kn/N},\\qquad x_t[n]=\\frac{1}{N}\\sum_{k=0}^{N-1}X_t[k]e^{j2\\pi kn/N}$$</p>
  <p><strong>Chirp-Z (<code>czt</code>):</strong> $$X_t[k]=\\sum_{n=0}^{N-1}x_t[n]A^{-n}W^{nk}$$</p>
  <p class=\"small\">With pvx defaults $A=1$ and $W=e^{-j2\\pi/N}$, CZT samples the DFT contour via a different numerical path.</p>
  <p><strong>DCT-II (<code>dct</code>):</strong> $$C_t[k]=\\alpha_k\\sum_{n=0}^{N-1}x_t[n]\\cos\\left(\\frac{\\pi}{N}(n+\\tfrac{1}{2})k\\right)$$</p>
  <p><strong>DST-II (<code>dst</code>):</strong> $$S_t[k]=\\beta_k\\sum_{n=0}^{N-1}x_t[n]\\sin\\left(\\frac{\\pi}{N}(n+\\tfrac{1}{2})(k+1)\\right)$$</p>
  <p><strong>Hartley (<code>hartley</code>):</strong> $$H_t[k]=\\sum_{n=0}^{N-1}x_t[n]\\,\\mathrm{cas}\\left(\\frac{2\\pi kn}{N}\\right),\\ \\mathrm{cas}(\\theta)=\\cos\\theta+\\sin\\theta$$</p>

  <table>
    <thead>
      <tr>
        <th>Transform</th>
        <th>Why use it</th>
        <th>Tradeoffs</th>
        <th>Example use case</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code>fft</code></td>
        <td>Fastest and most robust default; best runtime and CUDA support.</td>
        <td>Typical STFT leakage/time-resolution tradeoffs still apply.</td>
        <td>General production stretch/pitch workflows.</td>
      </tr>
      <tr>
        <td><code>dft</code></td>
        <td>Reference Fourier baseline for transform parity checks.</td>
        <td>Usually slower than <code>fft</code> with little audible benefit.</td>
        <td>Verification and algorithm A/B testing.</td>
      </tr>
      <tr>
        <td><code>czt</code></td>
        <td>Alternative numerical path for awkward or prime frame sizes.</td>
        <td>Requires SciPy; generally slower and CPU-oriented.</td>
        <td>Edge-case frame-size validation and diagnostics.</td>
      </tr>
      <tr>
        <td><code>dct</code></td>
        <td>Real-basis energy compaction, useful for envelope-focused shaping.</td>
        <td>No explicit complex phase bins; less transparent for strict phase coherence.</td>
        <td>Creative timbre shaping and coefficient-domain experiments.</td>
      </tr>
      <tr>
        <td><code>dst</code></td>
        <td>Odd-symmetry real-basis alternative with distinct coloration.</td>
        <td>Same phase limitations as DCT; artifacts are content-dependent.</td>
        <td>Experimental percussive/spectral texture variants.</td>
      </tr>
      <tr>
        <td><code>hartley</code></td>
        <td>Real transform with Fourier-related basis for exploratory comparisons.</td>
        <td>Different bin semantics from complex STFT can change artifact character.</td>
        <td>Pedagogical and creative real-domain phase-vocoder tests.</td>
      </tr>
    </tbody>
  </table>

  <h3>Sample use cases</h3>
  <pre><code>python3 pvxvoc.py dialog.wav --transform fft --time-stretch 1.08 --transient-preserve --output-dir out
python3 pvxvoc.py tone_sweep.wav --transform dft --time-stretch 1.00 --output-dir out
python3 pvxvoc.py archival_take.wav --transform czt --n-fft 1531 --win-length 1531 --hop-size 382 --output-dir out
python3 pvxvoc.py strings.wav --transform dct --pitch-shift-cents -17 --output-dir out
python3 pvxvoc.py percussion.wav --transform dst --time-stretch 0.92 --phase-locking off --output-dir out
python3 pvxvoc.py synth_pad.wav --transform hartley --time-stretch 1.30 --output-dir out</code></pre>
</div>
"""
    )
    sections.append(
        """
<div class=\"card\">
  <h2>Phase-Vocoder Propagation</h2>
  <p>$$\\Delta\\phi_t[k]=\\mathrm{princarg}\\Big(\\phi_t[k]-\\phi_{t-1}[k]-\\omega_kH_a\\Big)$$</p>
  <p>$$\\hat{\\omega}_t[k]=\\omega_k+\\frac{\\Delta\\phi_t[k]}{H_a}$$</p>
  <p>$$\\hat{\\phi}_t[k]=\\hat{\\phi}_{t-1}[k]+\\hat{\\omega}_t[k]H_s$$</p>
  <p class=\"small\">
    pvx uses these relationships to estimate true instantaneous frequency and
    re-accumulate phase under a new synthesis hop.
  </p>
</div>
"""
    )
    sections.append(
        """
<div class=\"card\">
  <h2>Pitch and Microtonal Mapping</h2>
  <p>$$r_{\\text{pitch}}=2^{\\Delta s/12}=2^{\\Delta c/1200}$$</p>
  <p class=\"small\">
    Semitone shift $\\Delta s$ and cents shift $\\Delta c$ are equivalent ratio controls.
    Scale-constrained retuning maps detected F0 to the nearest permitted scale target.
  </p>
</div>
"""
    )
    sections.append(
        """
<div class=\"card\">
  <h2>Dynamics and Loudness</h2>
  <p>$$g_{\\text{LUFS}}=10^{(L_{\\text{target}}-L_{\\text{in}})/20}$$</p>
  <p>$$y[n]=x[n]\\cdot g_{\\text{LUFS}}$$</p>
  <p class=\"small\">
    Target-loudness gain is applied in linear amplitude domain after dynamics stages,
    before limiting/clipping safety stages.
  </p>
</div>
"""
    )
    sections.append(
        """
<div class=\"card\">
  <h2>Spatial and Multichannel Highlights</h2>
  <p><strong>Inter-channel phase difference:</strong> $$\\Delta\\phi_{ij}(k,t)=\\phi_i(k,t)-\\phi_j(k,t)$$</p>
  <p><strong>Coherence-preserving objective:</strong> $$J=\\sum_{k,t}\\left|\\Delta\\phi^{out}_{ij}(k,t)-\\Delta\\phi^{in}_{ij}(k,t)\\right|$$</p>
  <p><strong>Delay estimate:</strong> $$\\tau^*=\\arg\\max_\\tau\\,\\mathcal{F}^{-1}\\left\\{\\frac{X_i(\\omega)X_j^*(\\omega)}{|X_i(\\omega)X_j^*(\\omega)|+\\varepsilon}\\right\\}(\\tau)$$</p>
  <p class=\"small\">
    pvx spatial modules focus on channel coherence, stable image cues, and
    phase-vocoder-consistent multichannel processing for robust chain composition.
  </p>
</div>
"""
    )
    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"papers.html\">Research papers</a> | "
        "<a href=\"glossary.html\">Technical glossary</a> | "
        "<a href=\"windows.html\">Window reference</a> | "
        "<a href=\"benchmarks.html\">Benchmarks</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Mathematical Foundations",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "math.html").write_text(html, encoding="utf-8")


def render_windows_page() -> None:
    metrics_payload = load_json(WINDOW_METRICS_PATH, {"windows": {}})
    metrics_map = metrics_payload.get("windows", {}) if isinstance(metrics_payload, dict) else {}

    rows: list[str] = []
    entries = window_entries()
    for entry in entries:
        metrics = metrics_map.get(entry["name"], {})
        coherent_gain = metrics.get("coherent_gain", "n/a")
        enbw_bins = metrics.get("enbw_bins", "n/a")
        scalloping_loss_db = metrics.get("scalloping_loss_db", "n/a")
        main_lobe_width_bins = metrics.get("main_lobe_width_bins", "n/a")
        peak_sidelobe_db = metrics.get("peak_sidelobe_db", "n/a")
        time_plot = metrics.get("time_plot")
        freq_plot = metrics.get("freq_plot")
        plot_html = "n/a"
        if time_plot and freq_plot:
            plot_html = (
                f"<a href=\"../{escape(str(time_plot))}\" target=\"_blank\" rel=\"noopener\">time</a> / "
                f"<a href=\"../{escape(str(freq_plot))}\" target=\"_blank\" rel=\"noopener\">freq</a>"
            )

        rows.append(
            "<tr>"
            f"<td><code>{escape(entry['name'])}</code></td>"
            f"<td>{escape(entry['family'])}</td>"
            f"<td><code>{escape(entry['params'])}</code></td>"
            f"<td>{escape(entry['formula'])}</td>"
            f"<td>{coherent_gain if isinstance(coherent_gain, str) else f'{float(coherent_gain):.6f}'}</td>"
            f"<td>{enbw_bins if isinstance(enbw_bins, str) else f'{float(enbw_bins):.6f}'}</td>"
            f"<td>{scalloping_loss_db if isinstance(scalloping_loss_db, str) else f'{float(scalloping_loss_db):.3f}'}</td>"
            f"<td>{main_lobe_width_bins if isinstance(main_lobe_width_bins, str) else f'{float(main_lobe_width_bins):.3f}'}</td>"
            f"<td>{peak_sidelobe_db if isinstance(peak_sidelobe_db, str) else f'{float(peak_sidelobe_db):.3f}'}</td>"
            f"<td>{plot_html}</td>"
            f"<td>{escape(entry['note'])}</td>"
            f"<td>{escape(entry['pros'])}</td>"
            f"<td>{escape(entry['cons'])}</td>"
            f"<td>{escape(entry['usage'])}</td>"
            "</tr>"
        )

    gallery_rows: list[str] = []
    for entry in entries:
        name = entry["name"]
        gallery_rows.append(
            "<tr>"
            f"<td><code>{escape(name)}</code></td>"
            f"<td><img src=\"../assets/windows/{escape(name)}_time.svg\" alt=\"{escape(name)} time-domain plot\" loading=\"lazy\" /></td>"
            f"<td><img src=\"../assets/windows/{escape(name)}_freq.svg\" alt=\"{escape(name)} magnitude spectrum\" loading=\"lazy\" /></td>"
            "</tr>"
        )

    sections: list[str] = []
    sections.append(
        f"""
<div class=\"card\">
  <p>
    Complete pvx analysis-window reference. This page covers all <strong>{len(voc_core.WINDOW_CHOICES)}</strong>
    supported windows from <code>pvx.core.voc.WINDOW_CHOICES</code>, with formula-family mapping and practical interpretation.
  </p>
  <p class=\"small\">
    For GitHub-native equation rendering, use
    <a href=\"../WINDOW_REFERENCE.md\"><code>docs/WINDOW_REFERENCE.md</code></a>.
  </p>
  <p class=\"small\">
    Quantitative metrics and per-window SVG plots are generated from
    <code>docs/window_metrics.json</code> and <code>docs/assets/windows/*</code>.
  </p>
</div>
"""
    )
    sections.append(WINDOW_EQUATIONS_HTML)
    sections.append(
        """
<table>
  <thead>
    <tr>
      <th>Window</th>
      <th>Family</th>
      <th>Parameters</th>
      <th>Formula key</th>
      <th>Coherent gain</th>
      <th>ENBW (bins)</th>
      <th>Scalloping loss (dB)</th>
      <th>Main-lobe width (bins)</th>
      <th>Peak sidelobe (dB)</th>
      <th>Plots</th>
      <th>Plain-English behavior</th>
      <th>Pros</th>
      <th>Cons</th>
      <th>Usage advice</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
""".format(rows="".join(rows))
    )
    sections.append(
        """
<div class="card">
  <h2>Complete Plot Gallery (All Windows)</h2>
  <p class="small">
    Time-domain and frequency-magnitude plots for every supported window.
  </p>
  <table>
    <thead>
      <tr>
        <th>Window</th>
        <th>Time-domain shape</th>
        <th>Magnitude spectrum</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</div>
""".format(rows="".join(gallery_rows))
    )
    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"math.html\">Math foundations</a> | "
        "<a href=\"glossary.html\">Technical glossary</a> | "
        "<a href=\"benchmarks.html\">Benchmarks</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Window Reference",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "windows.html").write_text(html, encoding="utf-8")


def render_architecture_page() -> None:
    content = """
<div class=\"card\">
  <p>
    Architecture overview for runtime processing, algorithm dispatch, documentation generation,
    and CI/Pages publication.
  </p>
</div>
<div class=\"card\">
  <h2>Runtime and CLI Flow</h2>
  <pre class=\"mermaid\">
flowchart LR
  A[User CLI Command] --> B[src/pvx/cli or pvxvoc parser]
  B --> C[Runtime Selection: auto/cpu/cuda]
  C --> D[Shared IO + Mastering Chain]
  D --> E[Core DSP in src/pvx/core/voc.py]
  E --> F[Output Writer / stdout stream]
  </pre>
</div>
<div class=\"card\">
  <h2>Algorithm Registry and Dispatch</h2>
  <pre class=\"mermaid\">
flowchart TD
  R[src/pvx/algorithms/registry.py] --> B[src/pvx/algorithms/base.py]
  B --> M1[time_scale_and_pitch_core/*]
  B --> M2[retune_and_intonation/*]
  B --> M3[dynamics_and_loudness/*]
  B --> M4[spatial_and_multichannel/*]
  </pre>
</div>
<div class=\"card\">
  <h2>Documentation Build Graph</h2>
  <pre class=\"mermaid\">
flowchart LR
  G1[scripts_generate_python_docs.py] --> D[docs/*]
  G2[scripts_generate_theory_docs.py] --> D
  G3[scripts_generate_docs_extras.py] --> D
  G4[scripts_generate_html_docs.py] --> H[docs/html/*]
  D --> H
  </pre>
</div>
<div class=\"card\">
  <h2>CI and Pages</h2>
  <pre class=\"mermaid\">
flowchart LR
  PR[Push / PR] --> CI[Doc and test workflow]
  CI --> S[Generation + drift checks]
  S --> T[Unit tests + docs coverage tests]
  T --> P[GitHub Pages deploy workflow]
  P --> SITE[Published docs site]
  </pre>
</div>
""".strip()
    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"math.html\">Math</a> | "
        "<a href=\"windows.html\">Windows</a> | "
        "<a href=\"limitations.html\">Limitations</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Architecture",
        content,
        css_path="style.css",
        breadcrumbs=breadcrumbs,
        include_mermaid=True,
    )
    (DOCS_HTML_DIR / "architecture.html").write_text(html, encoding="utf-8")


def render_cli_flags_page() -> None:
    payload = load_json(CLI_FLAGS_PATH, {"entries": [], "unique_flags": []})
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    by_tool: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool", "unknown"))
        by_tool.setdefault(tool, []).append(entry)

    sections: list[str] = []
    sections.append(
        "<div class=\"card\">"
        f"<p>Total unique long flags: <strong>{len(payload.get('unique_flags', [])) if isinstance(payload, dict) else 0}</strong></p>"
        "<p>Source index: <code>docs/CLI_FLAGS_REFERENCE.md</code> and <code>docs/cli_flags_reference.json</code>.</p>"
        "</div>"
    )

    for tool, rows in by_tool.items():
        table_rows = []
        for row in sorted(rows, key=lambda item: str(item.get("flag", ""))):
            default_text = "" if row.get("default") is None else str(row.get("default"))
            choices = row.get("choices")
            choices_text = ", ".join(str(c) for c in choices) if isinstance(choices, list) else ""
            table_rows.append(
                "<tr>"
                f"<td><code>{escape(str(row.get('flag', '')))}</code></td>"
                f"<td>{escape(str(row.get('required', False)))}</td>"
                f"<td><code>{escape(default_text)}</code></td>"
                f"<td><code>{escape(choices_text)}</code></td>"
                f"<td><code>{escape(str(row.get('action', '')))}</code></td>"
                f"<td>{escape(str(row.get('help', '')))}</td>"
                f"<td><code>{escape(str(row.get('source', '')))}</code></td>"
                "</tr>"
            )
        sections.append(
            f"<h2>{escape(tool)}</h2>"
            "<table>"
            "<thead><tr><th>Flag</th><th>Required</th><th>Default</th><th>Choices</th><th>Action</th><th>Description</th><th>Source</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
        )

    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"cookbook.html\">Cookbook</a> | "
        "<a href=\"limitations.html\">Limitations</a>"
        "</nav>"
    )
    html = html_page(
        "pvx CLI Flag Index",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "cli_flags.html").write_text(html, encoding="utf-8")


def render_limitations_page() -> None:
    payload = load_json(LIMITATIONS_PATH, {"groups": {}, "group_limits": {}})
    groups = payload.get("groups", {}) if isinstance(payload, dict) else {}
    group_limits = payload.get("group_limits", {}) if isinstance(payload, dict) else {}

    sections: list[str] = []
    summary_rows: list[str] = []
    for group in sorted(groups):
        seed = group_limits.get(group, {})
        summary_rows.append(
            "<tr>"
            f"<td><code>{escape(group)}</code></td>"
            f"<td>{escape(str(seed.get('assumption', 'n/a')))}</td>"
            f"<td>{escape(str(seed.get('failure', 'n/a')))}</td>"
            f"<td>{escape(str(seed.get('avoid', 'n/a')))}</td>"
            "</tr>"
        )
    sections.append(
        "<div class=\"card\"><p>Limitations guidance for all algorithm groups and algorithm IDs.</p></div>"
        "<h2>Group Summary</h2>"
        "<table>"
        "<thead><tr><th>Group</th><th>Assumptions</th><th>Failure modes</th><th>When not to use</th></tr></thead>"
        f"<tbody>{''.join(summary_rows)}</tbody>"
        "</table>"
    )

    for group in sorted(groups):
        rows = groups[group]
        table_rows: list[str] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                table_rows.append(
                    "<tr>"
                    f"<td><code>{escape(str(row.get('algorithm_id', '')))}</code></td>"
                    f"<td>{escape(str(row.get('assumptions', '')))}</td>"
                    f"<td>{escape(str(row.get('failure_modes', '')))}</td>"
                    f"<td>{escape(str(row.get('when_not_to_use', '')))}</td>"
                    "</tr>"
                )
        sections.append(
            f"<h2 id=\"{escape(group)}\"><code>{escape(group)}</code></h2>"
            "<table>"
            "<thead><tr><th>Algorithm ID</th><th>Assumptions</th><th>Failure modes</th><th>When not to use</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
        )

    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"architecture.html\">Architecture</a> | "
        "<a href=\"benchmarks.html\">Benchmarks</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Algorithm Limitations",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "limitations.html").write_text(html, encoding="utf-8")


def render_benchmarks_page() -> None:
    payload = load_json(BENCHMARKS_PATH, {"runs": [], "benchmark_spec": {}, "host": {}})
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    spec = payload.get("benchmark_spec", {}) if isinstance(payload, dict) else {}
    host = payload.get("host", {}) if isinstance(payload, dict) else {}

    rows: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        note_parts: list[str] = []
        if run.get("reason"):
            note_parts.append(str(run["reason"]))
        if run.get("runtime_fallback_reason"):
            note_parts.append(f"fallback={run['runtime_fallback_reason']}")
        if "gpu_pool_used_mb" in run:
            note_parts.append(f"gpu_pool_used_mb={run['gpu_pool_used_mb']}")
        rows.append(
            "<tr>"
            f"<td><code>{escape(str(run.get('backend', 'n/a')))}</code></td>"
            f"<td>{escape(str(run.get('status', 'n/a')))}</td>"
            f"<td>{escape(str(run.get('elapsed_ms', 'n/a')))}</td>"
            f"<td>{escape(str(run.get('peak_host_memory_mb', 'n/a')))}</td>"
            f"<td>{escape(str(run.get('snr_vs_input_db', 'n/a')))}</td>"
            f"<td>{escape(str(run.get('spectral_distance_vs_input_db', 'n/a')))}</td>"
            f"<td>{escape(str(run.get('snr_vs_cpu_db', 'n/a')))}</td>"
            f"<td>{escape(str(run.get('spectral_distance_vs_cpu_db', 'n/a')))}</td>"
            f"<td>{escape('; '.join(note_parts))}</td>"
            "</tr>"
        )

    content = (
        "<div class=\"card\">"
        "<p>Reproducible STFT/ISTFT benchmark summary.</p>"
        "<p><code>python3 scripts_generate_docs_extras.py --run-benchmarks</code></p>"
        f"<p class=\"small\">Spec: sample_rate={escape(str(spec.get('sample_rate', 'n/a')))} Hz, "
        f"duration={escape(str(spec.get('duration_seconds', 'n/a')))} s</p>"
        f"<p class=\"small\">Host: {escape(str(host.get('platform', 'n/a')))} | machine={escape(str(host.get('machine', 'n/a')))} | python={escape(str(host.get('python', 'n/a')))}</p>"
        "</div>"
        "<table>"
        "<thead><tr><th>Backend</th><th>Status</th><th>Elapsed (ms)</th><th>Peak host memory (MB)</th><th>SNR vs input (dB)</th><th>Spectral dist vs input (dB)</th><th>SNR vs CPU (dB)</th><th>Spectral dist vs CPU (dB)</th><th>Notes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"math.html\">Math</a> | "
        "<a href=\"windows.html\">Windows</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Benchmark Report",
        content,
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "benchmarks.html").write_text(html, encoding="utf-8")


def render_cookbook_page() -> None:
    payload = load_json(COOKBOOK_PATH, {"recipes": []})
    recipes = payload.get("recipes", []) if isinstance(payload, dict) else []
    by_cat: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        by_cat.setdefault(str(recipe.get("category", "General")), []).append(recipe)

    sections: list[str] = []
    sections.append(
        "<div class=\"card\">"
        "<p>Pipeline cookbook for practical one-liners, including Unix pipes and mastering chains.</p>"
        "<p>Markdown source: <code>docs/PIPELINE_COOKBOOK.md</code>.</p>"
        "</div>"
    )
    for category, rows in by_cat.items():
        table_rows = []
        for row in rows:
            table_rows.append(
                "<tr>"
                f"<td>{escape(str(row.get('title', '')))}</td>"
                f"<td><code>{escape(str(row.get('command', '')))}</code></td>"
                f"<td>{escape(str(row.get('why', '')))}</td>"
                "</tr>"
            )
        sections.append(
            f"<h2>{escape(category)}</h2>"
            "<table>"
            "<thead><tr><th>Workflow</th><th>Command</th><th>Why</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
        )

    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"cli_flags.html\">CLI flags</a> | "
        "<a href=\"benchmarks.html\">Benchmarks</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Pipeline Cookbook",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "cookbook.html").write_text(html, encoding="utf-8")


def render_citations_page() -> None:
    payload = load_json(CITATION_QUALITY_PATH, {"counts": {}, "unresolved_scholar": []})
    counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
    unresolved = payload.get("unresolved_scholar", []) if isinstance(payload, dict) else []

    summary_rows = "".join(
        "<tr>"
        f"<td><code>{escape(str(kind))}</code></td>"
        f"<td>{escape(str(counts[kind]))}</td>"
        "</tr>"
        for kind in sorted(counts)
    )
    unresolved_rows: list[str] = []
    for row in unresolved:
        if not isinstance(row, dict):
            continue
        unresolved_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('year', '')))}</td>"
            f"<td>{escape(str(row.get('authors', '')))}</td>"
            f"<td>{escape(str(row.get('title', '')))}</td>"
            f"<td><a href=\"{escape(str(row.get('url', '')))}\" target=\"_blank\" rel=\"noopener\">link</a></td>"
            "</tr>"
        )

    content = (
        "<div class=\"card\">"
        "<p>Citation quality report and BibTeX export.</p>"
        "<p><a href=\"../references.bib\"><code>docs/references.bib</code></a></p>"
        "</div>"
        "<h2>Link-Type Summary</h2>"
        "<table><thead><tr><th>Link type</th><th>Count</th></tr></thead>"
        f"<tbody>{summary_rows}</tbody></table>"
        "<h2>Scholar-Link Entries (upgrade candidates)</h2>"
        "<table><thead><tr><th>Year</th><th>Authors</th><th>Title</th><th>URL</th></tr></thead>"
        f"<tbody>{''.join(unresolved_rows)}</tbody></table>"
    )
    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a> | "
        "<a href=\"papers.html\">Research papers</a> | "
        "<a href=\"glossary.html\">Glossary</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Citation Quality",
        content,
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "citations.html").write_text(html, encoding="utf-8")


def write_docs_root_index() -> None:
    page = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"refresh\" content=\"0; url=html/index.html\" />
  <title>pvx docs</title>
</head>
<body>
  <p>Redirecting to <a href=\"html/index.html\">docs/html/index.html</a> ...</p>
</body>
</html>
"""
    (ROOT / "docs" / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    DOCS_HTML_DIR.mkdir(parents=True, exist_ok=True)
    GROUPS_DIR.mkdir(parents=True, exist_ok=True)
    for stale_group_page in GROUPS_DIR.glob("*.html"):
        stale_group_page.unlink()

    params = extract_algorithm_params(ROOT / "src" / "pvx" / "algorithms" / "base.py")
    groups = grouped_algorithms()

    write_style_css()
    render_index(groups, params)
    render_group_pages(groups, params)
    render_papers_page()
    render_glossary_page()
    render_math_page()
    render_windows_page()
    render_architecture_page()
    render_limitations_page()
    render_benchmarks_page()
    render_cookbook_page()
    render_cli_flags_page()
    render_citations_page()
    write_docs_root_index()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
