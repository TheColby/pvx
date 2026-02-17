#!/usr/bin/env python3
"""Generate grouped HTML documentation for pvx algorithms and research references."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import quote_plus
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DOCS_HTML_DIR = ROOT / "docs" / "html"
GROUPS_DIR = DOCS_HTML_DIR / "groups"

sys.path.insert(0, str(SRC_DIR))
from pvx.algorithms.registry import ALGORITHM_REGISTRY  # noqa: E402


def scholar(title: str) -> str:
    return f"https://scholar.google.com/scholar?q={quote_plus(title)}"


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
        "authors": "L. J. Griffiths; C. W. Jim",
        "year": "1982",
        "title": "An Alternative Approach to Linearly Constrained Adaptive Beamforming",
        "venue": "IEEE Trans. Antennas and Propagation",
        "url": "https://doi.org/10.1109/TAP.1982.1142739",
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


def extract_algorithm_params(base_path: Path) -> dict[str, list[str]]:
    text = base_path.read_text(encoding="utf-8")
    mapping: dict[str, list[str]] = {}
    lines = text.splitlines()

    current_slug: str | None = None
    bucket: list[str] = []

    def commit() -> None:
        nonlocal current_slug, bucket
        if current_slug is None:
            return
        dedup: list[str] = []
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


def grouped_algorithms() -> OrderedDict[str, list[tuple[str, dict[str, str]]]]:
    groups: OrderedDict[str, list[tuple[str, dict[str, str]]]] = OrderedDict()
    for algorithm_id, meta in ALGORITHM_REGISTRY.items():
        folder, _ = algorithm_id.split(".", 1)
        groups.setdefault(folder, []).append((algorithm_id, meta))
    return groups


def html_page(title: str, content: str, *, css_path: str, breadcrumbs: str = "") -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{escape(title)}</title>
  <link rel=\"stylesheet\" href=\"{css_path}\" />
</head>
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
      Generated by <code>scripts_generate_html_docs.py</code> on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.
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
@media (max-width: 900px) {
  table { display: block; overflow-x: auto; white-space: nowrap; }
}
""".strip() + "\n"
    (DOCS_HTML_DIR / "style.css").write_text(css, encoding="utf-8")


def render_index(groups: OrderedDict[str, list[tuple[str, dict[str, str]]]], params: dict[str, list[str]]) -> None:
    total_algorithms = sum(len(items) for items in groups.values())
    total_param_keys = len({k for values in params.values() for k in values})
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
  </ul>
</div>
""".format(total_algorithms=total_algorithms, group_count=len(groups), param_count=total_param_keys)
    )

    content_parts.append(
        """
<div class=\"card\">
  <p>
    <a href=\"papers.html\"><strong>Research bibliography (dozens of papers)</strong></a>
    - foundational and directly related literature for phase vocoder, time-scale/pitch,
    time-frequency transforms, separation, denoising, and spatial processing.
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
        breadcrumbs="<p class=\"kicker\">Index of grouped algorithm docs and bibliography</p>",
    )
    (DOCS_HTML_DIR / "index.html").write_text(html, encoding="utf-8")


def render_group_pages(groups: OrderedDict[str, list[tuple[str, dict[str, str]]]], params: dict[str, list[str]]) -> None:
    for folder, items in groups.items():
        theme = items[0][1]["theme"] if items else folder
        rows: list[str] = []
        for algorithm_id, meta in items:
            _, slug = algorithm_id.split(".", 1)
            module_path = f"src/pvx/algorithms/{folder}/{slug}.py"
            keys = params.get(slug, [])
            key_text = ", ".join(f"<code>{escape(k)}</code>" for k in keys) if keys else "None (generic/default path)"
            rows.append(
                "<tr>"
                f"<td><code>{escape(algorithm_id)}</code></td>"
                f"<td>{escape(meta['name'])}</td>"
                f"<td><code>{escape(module_path)}</code></td>"
                f"<td>{key_text}</td>"
                "</tr>"
            )

        content = (
            f"<div class=\"card\"><p><strong>Theme:</strong> {escape(theme)}</p>"
            f"<p><strong>Folder:</strong> <code>{escape(folder)}</code> | <strong>Algorithms:</strong> {len(items)}</p>"
            "</div>"
            "<table>"
            "<thead><tr><th>Algorithm ID</th><th>Name</th><th>Module Path</th><th>Parameter keys</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

        breadcrumbs = (
            "<nav>"
            "<a href=\"../index.html\">Home</a> | "
            "<a href=\"../papers.html\">Research papers</a>"
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
    the formulation of this repository's phase-vocoder-centric architecture and the broader
    DSP algorithm roadmap.
  </p>
  <p>
    Total references: <strong>{count}</strong>
  </p>
</div>
""".format(count=len(PAPERS))
    )

    toc_items = "".join(
        f"<li><a href=\"#{escape(category.lower().replace(' ', '-').replace(',', '').replace('/', '-'))}\">{escape(category)}</a></li>"
        for category in by_category
    )
    sections.append(f"<div class=\"card\"><h2>Categories</h2><ul>{toc_items}</ul></div>")

    for category, papers in by_category.items():
        anchor = category.lower().replace(" ", "-").replace(",", "").replace("/", "-")
        rows = []
        for p in papers:
            rows.append(
                "<tr>"
                f"<td>{escape(p['year'])}</td>"
                f"<td>{escape(p['authors'])}</td>"
                f"<td>{escape(p['title'])}</td>"
                f"<td>{escape(p['venue'])}</td>"
                f"<td><a href=\"{escape(p['url'])}\" target=\"_blank\" rel=\"noopener\">Link</a></td>"
                "</tr>"
            )
        sections.append(
            f"<h2 id=\"{escape(anchor)}\">{escape(category)}</h2>"
            "<table>"
            "<thead><tr><th>Year</th><th>Authors</th><th>Title</th><th>Venue</th><th>Link</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    breadcrumbs = (
        "<nav>"
        "<a href=\"index.html\">Home</a>"
        "</nav>"
    )
    html = html_page(
        "pvx Research Bibliography (Phase Vocoder and Related DSP)",
        "\n".join(sections),
        css_path="style.css",
        breadcrumbs=breadcrumbs,
    )
    (DOCS_HTML_DIR / "papers.html").write_text(html, encoding="utf-8")


def main() -> int:
    DOCS_HTML_DIR.mkdir(parents=True, exist_ok=True)
    GROUPS_DIR.mkdir(parents=True, exist_ok=True)

    params = extract_algorithm_params(ROOT / "src" / "pvx" / "algorithms" / "base.py")
    groups = grouped_algorithms()

    write_style_css()
    render_index(groups, params)
    render_group_pages(groups, params)
    render_papers_page()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
