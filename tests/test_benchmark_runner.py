"""Tests for benchmark runner profile selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.run_bench import (
    TaskSpec,
    _check_gate,
    _diagnose_metrics,
    _prepare_dataset,
    _pvx_bench_args,
)


class TestBenchmarkRunnerProfiles(unittest.TestCase):
    def _write_wav(self, path: Path, channels: int) -> None:
        sr = 24000
        t = np.arange(int(sr * 0.1)) / sr
        tone = 0.2 * np.sin(2 * np.pi * 220.0 * t)
        if channels == 1:
            audio = tone[:, None]
        else:
            audio = np.stack([tone, np.roll(tone, 3)], axis=1)
        sf.write(path, audio, sr)

    def test_tuned_profile_mono_stretch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mono.wav"
            self._write_wav(path, channels=1)
            args = _pvx_bench_args(path, TaskSpec("stretch", "stretch", 1.8), tuned=True)
            self.assertIn("--transient-mode", args)
            self.assertIn("off", args)
            self.assertIn("--n-fft", args)
            self.assertIn("1024", args)
            self.assertNotIn("--stereo-mode", args)

    def test_tuned_profile_stereo_pitch_adds_stereo_and_formant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            self._write_wav(path, channels=2)
            args = _pvx_bench_args(path, TaskSpec("pitch", "pitch", 4.0), tuned=True)
            self.assertIn("--stereo-mode", args)
            self.assertIn("mid_side_lock", args)
            self.assertIn("--coherence-strength", args)
            self.assertIn("--pitch-mode", args)
            self.assertIn("formant-preserving", args)

    def test_legacy_profile_keeps_hybrid_reference_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mono.wav"
            self._write_wav(path, channels=1)
            args = _pvx_bench_args(path, TaskSpec("stretch", "stretch", 1.8), tuned=False)
            self.assertIn("--transient-mode", args)
            self.assertIn("hybrid", args)
            self.assertIn("--stereo-mode", args)
            self.assertIn("ref_channel_lock", args)

    def test_check_gate_detects_signature_mismatch(self) -> None:
        current = {
            "methods": [
                {
                    "name": "pvx",
                    "aggregate": {
                        "log_spectral_distance": 1.30,
                        "modulation_spectrum_distance": 0.30,
                        "transient_smear_score": 0.06,
                        "stereo_coherence_drift": 0.02,
                    },
                    "rows": [],
                    "signatures": {"speech:stretch": "abcd"},
                }
            ],
            "determinism": {"mismatch_count": 0},
        }
        baseline = {
            "methods": [
                {
                    "name": "pvx",
                    "aggregate": {
                        "log_spectral_distance": 1.10,
                        "modulation_spectrum_distance": 0.20,
                        "transient_smear_score": 0.04,
                        "stereo_coherence_drift": 0.01,
                    },
                    "rows": [],
                    "signatures": {"speech:stretch": "wxyz"},
                }
            ]
        }
        failures = _check_gate(
            current,
            baseline,
            rule_overrides={
                "log_spectral_distance": ("max", 0.01),
                "modulation_spectrum_distance": ("max", 0.01),
                "transient_smear_score": ("max", 0.01),
                "stereo_coherence_drift": ("max", 0.01),
            },
            row_level=False,
            signature_gate=True,
        )
        self.assertTrue(any("Signature mismatch" in msg for msg in failures))
        self.assertTrue(any("log_spectral_distance" in msg for msg in failures))

    def test_check_gate_row_level_regression(self) -> None:
        current = {
            "methods": [
                {
                    "name": "pvx",
                    "aggregate": {"log_spectral_distance": 1.0},
                    "rows": [
                        {
                            "input": "a.wav",
                            "task": "stretch",
                            "log_spectral_distance": 2.0,
                        }
                    ],
                    "signatures": {},
                }
            ],
            "determinism": {"mismatch_count": 0},
        }
        baseline = {
            "methods": [
                {
                    "name": "pvx",
                    "aggregate": {"log_spectral_distance": 1.0},
                    "rows": [
                        {
                            "input": "a.wav",
                            "task": "stretch",
                            "log_spectral_distance": 1.0,
                        }
                    ],
                    "signatures": {},
                }
            ]
        }
        failures = _check_gate(
            current,
            baseline,
            rule_overrides={"log_spectral_distance": ("max", 0.01)},
            row_level=True,
            signature_gate=False,
        )
        self.assertTrue(any("row a.wav::stretch metric log_spectral_distance regressed" in msg for msg in failures))

    def test_diagnostics_emit_actionable_hints(self) -> None:
        diagnostics = _diagnose_metrics(
            {
                "transient_smear_score": 0.20,
                "onset_f1": 0.6,
                "phasiness_index": 0.3,
                "stereo_coherence_drift": 0.35,
                "perceptual_proxy_fraction": 1.0,
            }
        )
        joined = "\n".join(diagnostics)
        self.assertIn("transient", joined.lower())
        self.assertIn("phase", joined.lower())
        self.assertIn("stereo", joined.lower())
        self.assertIn("prox", joined.lower())

    def test_prepare_dataset_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            manifest = data_dir / "manifest.json"
            paths, payload, issues = _prepare_dataset(
                data_dir=data_dir,
                manifest_path=manifest,
                refresh_manifest=True,
                strict_corpus=True,
            )
            self.assertTrue(paths)
            self.assertTrue(manifest.exists())
            self.assertEqual(issues, [])
            parsed = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("entries", parsed)
            paths2, payload2, issues2 = _prepare_dataset(
                data_dir=data_dir,
                manifest_path=manifest,
                refresh_manifest=False,
                strict_corpus=True,
            )
            self.assertEqual(issues2, [])
            self.assertEqual(len(paths2), len(paths))
            self.assertEqual(len(payload2.get("entries", [])), len(payload.get("entries", [])))


if __name__ == "__main__":
    unittest.main()
