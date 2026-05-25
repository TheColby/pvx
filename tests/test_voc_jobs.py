"""Focused regression tests for extracted voc job helpers."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pvx.core import voc
from pvx.core.voc_jobs import load_checkpoint_chunk, process_file, save_checkpoint_chunk


class TestVocJobs(unittest.TestCase):
    def test_segmented_processing_passes_progress_factory_into_audio_blocks(self) -> None:
        args = argparse.Namespace(
            stdout=False,
            pitch_map=None,
            pitch_map_stdin=False,
            auto_segment_seconds=0.5,
            target_duration=None,
            target_sample_rate=None,
            resample_mode="linear",
            dry_run=True,
            output=None,
            overwrite=False,
            output_dir=None,
            suffix="_out",
            output_format="wav",
            pitch_map_crossfade_ms=0.0,
            checkpoint_dir=None,
            resume=False,
            transient_mode="off",
            transient_preserve=False,
            coherence_strength=0.0,
            stereo_mode="independent",
            stretch_mode="standard",
            extreme_stretch_threshold=2.0,
            max_stage_stretch=1.8,
            multires_fusion=False,
            _multires_ffts=[],
            _multires_weights=[],
            fourier_sync=False,
            analysis_channel="mix",
            f0_min=50.0,
            f0_max=1000.0,
            fourier_sync_min_fft=256,
            fourier_sync_max_fft=8192,
            fourier_sync_smooth=5,
            normalize="none",
            peak_dbfs=-1.0,
            rms_dbfs=-18.0,
            target_lufs=None,
            expander_threshold_db=None,
            expander_ratio=1.0,
            expander_attack_ms=10.0,
            expander_release_ms=100.0,
            compressor_threshold_db=None,
            compressor_ratio=1.0,
            compressor_attack_ms=10.0,
            compressor_release_ms=100.0,
            compressor_makeup_db=0.0,
            compander_threshold_db=None,
            compander_compress_ratio=1.0,
            compander_expand_ratio=1.0,
            compander_attack_ms=10.0,
            compander_release_ms=100.0,
            compander_makeup_db=0.0,
            limiter_threshold=None,
            soft_clip_level=None,
            soft_clip_type="tanh",
            soft_clip_drive=1.0,
            hard_clip_level=None,
            clip=False,
            subtype=None,
            bit_depth="inherit",
            dither="none",
            dither_seed=None,
            true_peak_max_dbtp=None,
            metadata_policy="none",
            _active_quality_profile="neutral",
            _dynamic_control_refs={},
            quiet=False,
            silent=False,
            no_progress=False,
            verbosity="normal",
            verbose=0,
        )
        config = voc.VocoderConfig(
            n_fft=2048,
            win_length=2048,
            hop_size=512,
            window="hann",
            center=True,
            phase_locking="identity",
            transient_preserve=False,
            transient_threshold=2.0,
        )
        callback_invocations: list[str] = []

        def fake_process_audio_block(
            audio: np.ndarray,
            sr: int,
            block_args: argparse.Namespace,
            block_config: voc.VocoderConfig,
            *,
            stretch: float,
            pitch_ratio: float,
            progress_callback_factory=None,
        ) -> voc.AudioBlockResult:
            self.assertIsNotNone(progress_callback_factory)
            callback = progress_callback_factory(0.10, 0.90, "inner")
            self.assertIsNotNone(callback)
            callback(1, 2)
            callback_invocations.append("called")
            return voc.AudioBlockResult(
                audio=np.asarray(audio, dtype=np.float64),
                internal_stretch=stretch * pitch_ratio,
                sync_plan=None,
                stage_count=1,
            )

        with tempfile.TemporaryDirectory(prefix="pvx-voc-jobs-") as tmp:
            output = Path(tmp) / "out.wav"
            args.output = output
            with (
                patch(
                    "pvx.core.voc_jobs._read_audio_input", return_value=(np.ones((1000, 1)), 1000)
                ),
                patch(
                    "pvx.core.voc_jobs.voc_core.choose_pitch_ratio",
                    return_value=voc.PitchConfig(ratio=1.0),
                ),
                patch("pvx.core.voc_jobs.voc_core.resolve_base_stretch", return_value=1.0),
                patch(
                    "pvx.core.voc_jobs.voc_core.process_audio_block",
                    side_effect=fake_process_audio_block,
                ),
                patch(
                    "pvx.core.voc_jobs.voc_core.apply_mastering_chain",
                    side_effect=lambda audio, *_: audio,
                ),
                patch(
                    "pvx.core.voc_jobs.prepare_output_audio",
                    side_effect=lambda audio, *_args, **_kwargs: (audio, None),
                ),
                patch("pvx.core.voc_jobs.render_audio_metrics_table", return_value=""),
                patch("pvx.core.voc_jobs.render_audio_comparison_table", return_value=""),
                patch("pvx.core.voc_jobs.summarize_audio_metrics", return_value={}),
            ):
                result = process_file(Path("input.wav"), args, config)

        self.assertGreaterEqual(len(callback_invocations), 2)
        self.assertEqual(result.control_map_segments, 2)

    def test_process_file_fails_before_processing_when_output_exists_without_overwrite(
        self,
    ) -> None:
        args = argparse.Namespace(
            stdout=False,
            pitch_map=None,
            pitch_map_stdin=False,
            auto_segment_seconds=0.0,
            target_duration=None,
            target_sample_rate=None,
            resample_mode="linear",
            dry_run=False,
            output=Path("/tmp/out.wav"),
            overwrite=False,
            output_dir=None,
            suffix="_out",
            output_format="wav",
            pitch_map_crossfade_ms=0.0,
            checkpoint_dir=None,
            resume=False,
            transient_mode="off",
            transient_preserve=False,
            coherence_strength=0.0,
            stereo_mode="independent",
            stretch_mode="standard",
            extreme_stretch_threshold=2.0,
            max_stage_stretch=1.8,
            multires_fusion=False,
            _multires_ffts=[],
            _multires_weights=[],
            fourier_sync=False,
            analysis_channel="mix",
            f0_min=50.0,
            f0_max=1000.0,
            fourier_sync_min_fft=256,
            fourier_sync_max_fft=8192,
            fourier_sync_smooth=5,
            normalize="none",
            peak_dbfs=-1.0,
            rms_dbfs=-18.0,
            target_lufs=None,
            expander_threshold_db=None,
            expander_ratio=1.0,
            expander_attack_ms=10.0,
            expander_release_ms=100.0,
            compressor_threshold_db=None,
            compressor_ratio=1.0,
            compressor_attack_ms=10.0,
            compressor_release_ms=100.0,
            compressor_makeup_db=0.0,
            compander_threshold_db=None,
            compander_compress_ratio=1.0,
            compander_expand_ratio=1.0,
            compander_attack_ms=10.0,
            compander_release_ms=100.0,
            compander_makeup_db=0.0,
            limiter_threshold=None,
            soft_clip_level=None,
            soft_clip_type="tanh",
            soft_clip_drive=1.0,
            hard_clip_level=None,
            clip=False,
            subtype=None,
            bit_depth="inherit",
            dither="none",
            dither_seed=None,
            true_peak_max_dbtp=None,
            metadata_policy="none",
            _active_quality_profile="neutral",
            _dynamic_control_refs={},
            quiet=False,
            silent=False,
            no_progress=False,
            verbosity="normal",
            verbose=0,
        )
        config = voc.VocoderConfig(
            n_fft=2048,
            win_length=2048,
            hop_size=512,
            window="hann",
            center=True,
            phase_locking="identity",
            transient_preserve=False,
            transient_threshold=2.0,
        )
        with tempfile.TemporaryDirectory(prefix="pvx-voc-jobs-") as tmp:
            output = Path(tmp) / "exists.wav"
            output.write_bytes(b"existing")
            args.output = output
            with (
                patch(
                    "pvx.core.voc_jobs._read_audio_input",
                    side_effect=AssertionError("should not read input"),
                ),
                patch(
                    "pvx.core.voc_jobs.voc_core.choose_pitch_ratio",
                    return_value=voc.PitchConfig(ratio=1.0),
                ),
                patch("pvx.core.voc_jobs.voc_core.resolve_base_stretch", return_value=1.0),
                patch(
                    "pvx.core.voc_jobs.voc_core.process_audio_block",
                    side_effect=AssertionError("should not process"),
                ),
                self.assertRaises(FileExistsError),
            ):
                process_file(Path("input.wav"), args, config)

    def test_process_file_uses_prefetched_audio_cache(self) -> None:
        args = argparse.Namespace(
            stdout=False,
            pitch_map=None,
            pitch_map_stdin=False,
            auto_segment_seconds=0.0,
            target_duration=None,
            target_sample_rate=None,
            resample_mode="linear",
            dry_run=True,
            output=None,
            overwrite=False,
            output_dir=None,
            suffix="_out",
            output_format="wav",
            pitch_map_crossfade_ms=0.0,
            checkpoint_dir=None,
            resume=False,
            transient_mode="off",
            transient_preserve=False,
            coherence_strength=0.0,
            stereo_mode="independent",
            stretch_mode="standard",
            extreme_stretch_threshold=2.0,
            max_stage_stretch=1.8,
            multires_fusion=False,
            _multires_ffts=[],
            _multires_weights=[],
            fourier_sync=False,
            analysis_channel="mix",
            f0_min=50.0,
            f0_max=1000.0,
            fourier_sync_min_fft=256,
            fourier_sync_max_fft=8192,
            fourier_sync_smooth=5,
            normalize="none",
            peak_dbfs=-1.0,
            rms_dbfs=-18.0,
            target_lufs=None,
            expander_threshold_db=None,
            expander_ratio=1.0,
            expander_attack_ms=10.0,
            expander_release_ms=100.0,
            compressor_threshold_db=None,
            compressor_ratio=1.0,
            compressor_attack_ms=10.0,
            compressor_release_ms=100.0,
            compressor_makeup_db=0.0,
            compander_threshold_db=None,
            compander_compress_ratio=1.0,
            compander_expand_ratio=1.0,
            compander_attack_ms=10.0,
            compander_release_ms=100.0,
            compander_makeup_db=0.0,
            limiter_threshold=None,
            soft_clip_level=None,
            soft_clip_type="tanh",
            soft_clip_drive=1.0,
            hard_clip_level=None,
            clip=False,
            subtype=None,
            bit_depth="inherit",
            dither="none",
            dither_seed=None,
            true_peak_max_dbtp=None,
            metadata_policy="none",
            _active_quality_profile="neutral",
            _dynamic_control_refs={},
            quiet=False,
            silent=False,
            no_progress=False,
            verbosity="normal",
            verbose=0,
            _prefetched_audio_cache={str(Path("input.wav")): (np.ones((1000, 1)), 1000)},
        )
        config = voc.VocoderConfig(
            n_fft=2048,
            win_length=2048,
            hop_size=512,
            window="hann",
            center=True,
            phase_locking="identity",
            transient_preserve=False,
            transient_threshold=2.0,
        )
        with (
            patch(
                "pvx.core.voc_jobs._read_audio_input",
                side_effect=AssertionError("should not read input"),
            ),
            patch(
                "pvx.core.voc_jobs.voc_core.choose_pitch_ratio",
                return_value=voc.PitchConfig(ratio=1.0),
            ),
            patch("pvx.core.voc_jobs.voc_core.resolve_base_stretch", return_value=1.0),
            patch(
                "pvx.core.voc_jobs.voc_core.process_audio_block",
                return_value=voc.AudioBlockResult(
                    audio=np.ones((1000, 1)),
                    internal_stretch=1.0,
                    sync_plan=None,
                    stage_count=1,
                ),
            ),
            patch(
                "pvx.core.voc_jobs.voc_core.apply_mastering_chain",
                side_effect=lambda audio, *_: audio,
            ),
            patch(
                "pvx.core.voc_jobs.prepare_output_audio",
                side_effect=lambda audio, *_args, **_kwargs: (audio, None),
            ),
            patch("pvx.core.voc_jobs.render_audio_metrics_table", return_value=""),
            patch("pvx.core.voc_jobs.render_audio_comparison_table", return_value=""),
            patch("pvx.core.voc_jobs.summarize_audio_metrics", return_value={}),
        ):
            result = process_file(Path("input.wav"), args, config)
        self.assertEqual(result.in_samples, 1000)
        self.assertEqual(args._prefetched_audio_cache, {})

    def test_load_checkpoint_chunk_rejects_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pvx-voc-jobs-") as tmp:
            chunk = Path(tmp) / "segment_00000.npy"
            save_checkpoint_chunk(
                chunk,
                np.ones((8, 2)),
                metadata={
                    "segment_index": 0,
                    "input_start": 0,
                    "input_end": 4,
                    "sample_rate": 1000,
                    "stretch": 1.0,
                    "pitch_ratio": 1.0,
                    "render_fingerprint": "abc",
                },
            )
            with self.assertRaises(ValueError):
                load_checkpoint_chunk(
                    chunk,
                    expected_meta={
                        "segment_index": 0,
                        "input_start": 1,
                        "input_end": 4,
                        "sample_rate": 1000,
                        "stretch": 1.0,
                        "pitch_ratio": 1.0,
                        "render_fingerprint": "xyz",
                    },
                )


if __name__ == "__main__":
    unittest.main()
