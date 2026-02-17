<p align="center"><br><br></p>

<p align="center">
  <img src="assets/pvx_logo.png" alt="pvx logo" width="360" />
</p>

<p align="center"><br><br></p>

# pvx

`pvx` is a multi-tool audio DSP toolkit built around phase-vocoder workflows.

The repository currently includes:

- `pvxvoc.py`: full-feature phase vocoder (time/pitch/F0/formant/fourier-sync).
- `pvxfreeze.py`: spectral freeze / sustained texture generation.
- `pvxharmonize.py`: multi-voice harmonizer.
- `pvxconform.py`: time+pitch conforming via CSV map.
- `pvxmorph.py`: spectral morph between two files.
- `pvxwarp.py`: variable time warp via CSV map.
- `pvxformant.py`: formant shift/preserve processing.
- `pvxtransient.py`: transient-first time/pitch processing.
- `pvxunison.py`: stereo detuned unison thickener.
- `pvxdenoise.py`: spectral subtraction denoiser.
- `pvxdeverb.py`: reverb-tail suppression.
- `pvxretune.py`: monophonic scale retune.
- `pvxlayer.py`: harmonic/percussive split and independent processing.
- `src/pvx/algorithms/`: canonical themed algorithm library with 95 DSP modules (time/pitch, transforms, separation, denoise, dereverb, dynamics, spatial, creative, granular, and analysis).
- `pvxalgorithms/`: compatibility shim package so legacy imports still work (`pvxalgorithms.*` -> `pvx.algorithms.*`).

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/pvx/core/` | Core DSP/runtime modules (`voc.py`, shared `common.py`). |
| `src/pvx/cli/` | Canonical CLI implementations for `pvx*` tools and `main.py` navigator. |
| `src/pvx/algorithms/` | Canonical algorithm package (`pvx.algorithms.*`). |
| `pvxvoc.py`, `pvxfreeze.py`, ... | Root compatibility wrappers forwarding to `src/pvx/...`. |
| `pvxalgorithms/` | Legacy package compatibility shim forwarding to `src/pvx/algorithms/`. |
| `docs/` | Generated documentation artifacts. |
| `tests/` | Unit and regression tests. |

## Install

### Option A: Requirements file

```bash
python3 -m pip install -r requirements.txt
```

### Option B: Editable package install

```bash
python3 -m pip install -e .
```

This exposes console commands from `pyproject.toml` (`pvxvoc`, `pvxfreeze`, etc.).

### Optional CUDA Acceleration

CUDA processing is available when CuPy is installed for your CUDA runtime:

```bash
python3 -m pip install cupy-cuda12x
```

Use `--device auto` (default), `--device cpu`, or `--device cuda` plus `--cuda-device <index>`.

## Quick Start

```bash
python3 pvxvoc.py test.wav --time-stretch 1.25 --pitch-shift-semitones 3
```

## Comprehensive Python Documentation and Help

`pvx` now ships a full, generated Python-file documentation set that covers every `.py` file in the repository.

Documentation artifacts:

- `docs/PYTHON_FILE_HELP.md`: exhaustive per-file reference and help index.
  - Current coverage: all `130` Python files in this repository.
  - Includes purpose, top-level symbols, help commands, and CLI help snapshots where applicable.
- `docs/PVX_ALGORITHM_PARAMS.md`: per-algorithm parameter keys consumed by `pvx.algorithms.base` dispatch.
- `ALGORITHM_INVENTORY.md`: themed inventory of all algorithm modules.

Help access patterns:

| File type | How to get help |
| --- | --- |
| Main pvx tool CLIs (`pvxvoc.py`, `pvxfreeze.py`, etc.) | `python3 <tool>.py --help` |
| Top-level navigator | `python3 main.py --help` |
| Algorithm modules (`src/pvx/algorithms/.../*.py`) | `python3 src/pvx/algorithms/<theme>/<module>.py --help` or `python3 -m pvx.algorithms.<theme>.<module> --help` |
| Test modules | `python3 tests/<file>.py` or `python3 -m unittest ...` |

Legacy algorithm module path `python3 -m pvxalgorithms.<theme>.<module> --help` is still supported via compatibility shims.

Regenerate docs after code changes:

```bash
python3 scripts_generate_python_docs.py
```

<!-- BEGIN ALGORITHM CATALOG -->
## Algorithm Catalog By Folder and Parameter Keys

This section enumerates every registered algorithm (95 total) grouped by folder.
`Parameter keys` are the exact `**params` keys consumed by `pvx.algorithms.base.run_algorithm()` dispatch.

| Folder | Theme | Algorithm count |
| --- | --- | ---: |
| `time_scale_and_pitch_core` | Time-Scale and Pitch Core | 7 |
| `pitch_detection_and_tracking` | Pitch Detection and Tracking | 8 |
| `retune_and_intonation` | Retune and Intonation | 8 |
| `spectral_time_frequency_transforms` | Spectral and Time-Frequency Transforms | 8 |
| `separation_and_decomposition` | Separation and Decomposition | 8 |
| `denoise_and_restoration` | Denoise and Restoration | 8 |
| `dereverb_and_room_correction` | Dereverb and Room Correction | 8 |
| `dynamics_and_loudness` | Dynamics and Loudness | 8 |
| `spatial_and_multichannel` | Spatial and Multichannel | 8 |
| `creative_spectral_effects` | Creative Spectral Effects | 8 |
| `granular_and_modulation` | Granular and Modulation | 8 |
| `analysis_qa_and_automation` | Analysis, QA, and Automation | 8 |

### `time_scale_and_pitch_core`

Theme: **Time-Scale and Pitch Core**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `time_scale_and_pitch_core.wsola_waveform_similarity_overlap_add` | WSOLA (Waveform Similarity Overlap-Add) | `src/pvx/algorithms/time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py` | `stretch`, `grain_size` |
| `time_scale_and_pitch_core.td_psola` | TD-PSOLA | `src/pvx/algorithms/time_scale_and_pitch_core/td_psola.py` | `semitones`, `stretch` |
| `time_scale_and_pitch_core.lp_psola` | LP-PSOLA | `src/pvx/algorithms/time_scale_and_pitch_core/lp_psola.py` | `semitones` |
| `time_scale_and_pitch_core.multi_resolution_phase_vocoder` | Multi-resolution phase vocoder | `src/pvx/algorithms/time_scale_and_pitch_core/multi_resolution_phase_vocoder.py` | `stretch` |
| `time_scale_and_pitch_core.harmonic_percussive_split_tsm` | Harmonic/percussive split TSM | `src/pvx/algorithms/time_scale_and_pitch_core/harmonic_percussive_split_tsm.py` | `harmonic_stretch`, `percussive_stretch` |
| `time_scale_and_pitch_core.beat_synchronous_time_warping` | Beat-synchronous time warping | `src/pvx/algorithms/time_scale_and_pitch_core/beat_synchronous_time_warping.py` | `stretch` |
| `time_scale_and_pitch_core.nonlinear_time_maps` | Nonlinear time maps (curves, anchors, spline timing) | `src/pvx/algorithms/time_scale_and_pitch_core/nonlinear_time_maps.py` | `curve`, `stretch`, `fmin` |

### `pitch_detection_and_tracking`

Theme: **Pitch Detection and Tracking**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `pitch_detection_and_tracking.yin` | YIN | `src/pvx/algorithms/pitch_detection_and_tracking/yin.py` | None (generic/default path) |
| `pitch_detection_and_tracking.pyin` | pYIN | `src/pvx/algorithms/pitch_detection_and_tracking/pyin.py` | None (generic/default path) |
| `pitch_detection_and_tracking.rapt` | RAPT | `src/pvx/algorithms/pitch_detection_and_tracking/rapt.py` | None (generic/default path) |
| `pitch_detection_and_tracking.swipe` | SWIPE | `src/pvx/algorithms/pitch_detection_and_tracking/swipe.py` | None (generic/default path) |
| `pitch_detection_and_tracking.harmonic_product_spectrum_hps` | Harmonic Product Spectrum (HPS) | `src/pvx/algorithms/pitch_detection_and_tracking/harmonic_product_spectrum_hps.py` | None (generic/default path) |
| `pitch_detection_and_tracking.subharmonic_summation` | Subharmonic summation | `src/pvx/algorithms/pitch_detection_and_tracking/subharmonic_summation.py` | None (generic/default path) |
| `pitch_detection_and_tracking.crepe_style_neural_f0` | CREPE-style neural F0 | `src/pvx/algorithms/pitch_detection_and_tracking/crepe_style_neural_f0.py` | None (generic/default path) |
| `pitch_detection_and_tracking.viterbi_smoothed_pitch_contour_tracking` | Viterbi-smoothed pitch contour tracking | `src/pvx/algorithms/pitch_detection_and_tracking/viterbi_smoothed_pitch_contour_tracking.py` | `root_midi`, `scale_cents`, `scale`, `fmin` |

### `retune_and_intonation`

Theme: **Retune and Intonation**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `retune_and_intonation.chord_aware_retuning` | Chord-aware retuning | `src/pvx/algorithms/retune_and_intonation/chord_aware_retuning.py` | `strength` |
| `retune_and_intonation.key_aware_retuning_with_confidence_weighting` | Key-aware retuning with confidence weighting | `src/pvx/algorithms/retune_and_intonation/key_aware_retuning_with_confidence_weighting.py` | None (generic/default path) |
| `retune_and_intonation.just_intonation_mapping_per_key_center` | Just intonation mapping per key center | `src/pvx/algorithms/retune_and_intonation/just_intonation_mapping_per_key_center.py` | None (generic/default path) |
| `retune_and_intonation.adaptive_intonation_context_sensitive_intervals` | Adaptive intonation (context-sensitive intervals) | `src/pvx/algorithms/retune_and_intonation/adaptive_intonation_context_sensitive_intervals.py` | None (generic/default path) |
| `retune_and_intonation.scala_mts_scale_import_and_quantization` | Scala/MTS scale import and quantization | `src/pvx/algorithms/retune_and_intonation/scala_mts_scale_import_and_quantization.py` | None (generic/default path) |
| `retune_and_intonation.time_varying_cents_maps` | Time-varying cents maps | `src/pvx/algorithms/retune_and_intonation/time_varying_cents_maps.py` | `cents_curve` |
| `retune_and_intonation.vibrato_preserving_correction` | Vibrato-preserving correction | `src/pvx/algorithms/retune_and_intonation/vibrato_preserving_correction.py` | `strength` |
| `retune_and_intonation.portamento_aware_retune_curves` | Portamento-aware retune curves | `src/pvx/algorithms/retune_and_intonation/portamento_aware_retune_curves.py` | `max_semitone_step`, `compression` |

### `spectral_time_frequency_transforms`

Theme: **Spectral and Time-Frequency Transforms**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `spectral_time_frequency_transforms.constant_q_transform_cqt_processing` | Constant-Q Transform (CQT) processing | `src/pvx/algorithms/spectral_time_frequency_transforms/constant_q_transform_cqt_processing.py` | None (generic/default path) |
| `spectral_time_frequency_transforms.variable_q_transform_vqt` | Variable-Q Transform (VQT) | `src/pvx/algorithms/spectral_time_frequency_transforms/variable_q_transform_vqt.py` | None (generic/default path) |
| `spectral_time_frequency_transforms.nsgt_based_processing` | NSGT-based processing | `src/pvx/algorithms/spectral_time_frequency_transforms/nsgt_based_processing.py` | None (generic/default path) |
| `spectral_time_frequency_transforms.reassigned_spectrogram_methods` | Reassigned spectrogram methods | `src/pvx/algorithms/spectral_time_frequency_transforms/reassigned_spectrogram_methods.py` | None (generic/default path) |
| `spectral_time_frequency_transforms.synchrosqueezed_stft` | Synchrosqueezed STFT | `src/pvx/algorithms/spectral_time_frequency_transforms/synchrosqueezed_stft.py` | None (generic/default path) |
| `spectral_time_frequency_transforms.chirplet_transform_analysis` | Chirplet transform analysis | `src/pvx/algorithms/spectral_time_frequency_transforms/chirplet_transform_analysis.py` | None (generic/default path) |
| `spectral_time_frequency_transforms.wavelet_packet_processing` | Wavelet packet processing | `src/pvx/algorithms/spectral_time_frequency_transforms/wavelet_packet_processing.py` | None (generic/default path) |
| `spectral_time_frequency_transforms.multi_window_stft_fusion` | Multi-window STFT fusion | `src/pvx/algorithms/spectral_time_frequency_transforms/multi_window_stft_fusion.py` | None (generic/default path) |

### `separation_and_decomposition`

Theme: **Separation and Decomposition**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `separation_and_decomposition.rpca_hpss` | RPCA HPSS | `src/pvx/algorithms/separation_and_decomposition/rpca_hpss.py` | None (generic/default path) |
| `separation_and_decomposition.nmf_decomposition` | NMF decomposition | `src/pvx/algorithms/separation_and_decomposition/nmf_decomposition.py` | `components` |
| `separation_and_decomposition.ica_bss_for_multichannel_stems` | ICA/BSS for multichannel stems | `src/pvx/algorithms/separation_and_decomposition/ica_bss_for_multichannel_stems.py` | None (generic/default path) |
| `separation_and_decomposition.sinusoidal_residual_transient_decomposition` | Sinusoidal+residual+transient decomposition | `src/pvx/algorithms/separation_and_decomposition/sinusoidal_residual_transient_decomposition.py` | None (generic/default path) |
| `separation_and_decomposition.demucs_style_stem_separation_backend` | Demucs-style stem separation backend | `src/pvx/algorithms/separation_and_decomposition/demucs_style_stem_separation_backend.py` | None (generic/default path) |
| `separation_and_decomposition.u_net_vocal_accompaniment_split` | U-Net vocal/accompaniment split | `src/pvx/algorithms/separation_and_decomposition/u_net_vocal_accompaniment_split.py` | None (generic/default path) |
| `separation_and_decomposition.tensor_decomposition_cp_tucker` | Tensor decomposition (CP/Tucker) | `src/pvx/algorithms/separation_and_decomposition/tensor_decomposition_cp_tucker.py` | `rank` |
| `separation_and_decomposition.probabilistic_latent_component_separation` | Probabilistic latent component separation | `src/pvx/algorithms/separation_and_decomposition/probabilistic_latent_component_separation.py` | None (generic/default path) |

### `denoise_and_restoration`

Theme: **Denoise and Restoration**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `denoise_and_restoration.wiener_denoising` | Wiener denoising | `src/pvx/algorithms/denoise_and_restoration/wiener_denoising.py` | None (generic/default path) |
| `denoise_and_restoration.mmse_stsa` | MMSE-STSA | `src/pvx/algorithms/denoise_and_restoration/mmse_stsa.py` | None (generic/default path) |
| `denoise_and_restoration.log_mmse` | Log-MMSE | `src/pvx/algorithms/denoise_and_restoration/log_mmse.py` | None (generic/default path) |
| `denoise_and_restoration.minimum_statistics_noise_tracking` | Minimum-statistics noise tracking | `src/pvx/algorithms/denoise_and_restoration/minimum_statistics_noise_tracking.py` | None (generic/default path) |
| `denoise_and_restoration.rnnoise_style_denoiser` | RNNoise-style denoiser | `src/pvx/algorithms/denoise_and_restoration/rnnoise_style_denoiser.py` | None (generic/default path) |
| `denoise_and_restoration.diffusion_based_speech_audio_denoise` | Diffusion-based speech/audio denoise | `src/pvx/algorithms/denoise_and_restoration/diffusion_based_speech_audio_denoise.py` | None (generic/default path) |
| `denoise_and_restoration.declip_via_sparse_reconstruction` | Declip via sparse reconstruction | `src/pvx/algorithms/denoise_and_restoration/declip_via_sparse_reconstruction.py` | `clip_threshold` |
| `denoise_and_restoration.declick_decrackle_median_wavelet_interpolation` | Declick/decrackle (median/wavelet + interpolation) | `src/pvx/algorithms/denoise_and_restoration/declick_decrackle_median_wavelet_interpolation.py` | `spike_threshold` |

### `dereverb_and_room_correction`

Theme: **Dereverb and Room Correction**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `dereverb_and_room_correction.wpe_dereverberation` | WPE dereverberation | `src/pvx/algorithms/dereverb_and_room_correction/wpe_dereverberation.py` | `taps` |
| `dereverb_and_room_correction.spectral_decay_subtraction` | Spectral decay subtraction | `src/pvx/algorithms/dereverb_and_room_correction/spectral_decay_subtraction.py` | None (generic/default path) |
| `dereverb_and_room_correction.late_reverb_suppression_via_coherence` | Late reverb suppression via coherence | `src/pvx/algorithms/dereverb_and_room_correction/late_reverb_suppression_via_coherence.py` | None (generic/default path) |
| `dereverb_and_room_correction.room_impulse_inverse_filtering` | Room impulse inverse filtering | `src/pvx/algorithms/dereverb_and_room_correction/room_impulse_inverse_filtering.py` | None (generic/default path) |
| `dereverb_and_room_correction.multi_band_adaptive_deverb` | Multi-band adaptive deverb | `src/pvx/algorithms/dereverb_and_room_correction/multi_band_adaptive_deverb.py` | None (generic/default path) |
| `dereverb_and_room_correction.drr_guided_dereverb` | DRR-guided dereverb | `src/pvx/algorithms/dereverb_and_room_correction/drr_guided_dereverb.py` | None (generic/default path) |
| `dereverb_and_room_correction.blind_deconvolution_dereverb` | Blind deconvolution dereverb | `src/pvx/algorithms/dereverb_and_room_correction/blind_deconvolution_dereverb.py` | None (generic/default path) |
| `dereverb_and_room_correction.neural_dereverb_module` | Neural dereverb module | `src/pvx/algorithms/dereverb_and_room_correction/neural_dereverb_module.py` | None (generic/default path) |

### `dynamics_and_loudness`

Theme: **Dynamics and Loudness**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `dynamics_and_loudness.ebu_r128_normalization` | EBU R128 normalization | `src/pvx/algorithms/dynamics_and_loudness/ebu_r128_normalization.py` | `target_lufs` |
| `dynamics_and_loudness.itu_bs_1770_loudness_measurement_gating` | ITU BS.1770 loudness measurement/gating | `src/pvx/algorithms/dynamics_and_loudness/itu_bs_1770_loudness_measurement_gating.py` | `gate_lufs` |
| `dynamics_and_loudness.multi_band_compression` | Multi-band compression | `src/pvx/algorithms/dynamics_and_loudness/multi_band_compression.py` | None (generic/default path) |
| `dynamics_and_loudness.upward_compression` | Upward compression | `src/pvx/algorithms/dynamics_and_loudness/upward_compression.py` | `threshold_db` |
| `dynamics_and_loudness.transient_shaping` | Transient shaping | `src/pvx/algorithms/dynamics_and_loudness/transient_shaping.py` | `attack_boost` |
| `dynamics_and_loudness.spectral_dynamics_bin_wise_compressor_expander` | Spectral dynamics (bin-wise compressor/expander) | `src/pvx/algorithms/dynamics_and_loudness/spectral_dynamics_bin_wise_compressor_expander.py` | `threshold_db` |
| `dynamics_and_loudness.true_peak_limiting` | True-peak limiting | `src/pvx/algorithms/dynamics_and_loudness/true_peak_limiting.py` | `threshold` |
| `dynamics_and_loudness.lufs_target_mastering_chain` | LUFS-target mastering chain | `src/pvx/algorithms/dynamics_and_loudness/lufs_target_mastering_chain.py` | `target_lufs` |

### `spatial_and_multichannel`

Theme: **Spatial and Multichannel**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `spatial_and_multichannel.mid_side_adaptive_widening` | Mid/Side adaptive widening | `src/pvx/algorithms/spatial_and_multichannel/mid_side_adaptive_widening.py` | `width` |
| `spatial_and_multichannel.binaural_hrtf_rendering` | Binaural HRTF rendering | `src/pvx/algorithms/spatial_and_multichannel/binaural_hrtf_rendering.py` | `azimuth_deg` |
| `spatial_and_multichannel.ambisonics_encode_decode` | Ambisonics encode/decode | `src/pvx/algorithms/spatial_and_multichannel/ambisonics_encode_decode.py` | `angle_deg` |
| `spatial_and_multichannel.mvdr_beamforming` | MVDR beamforming | `src/pvx/algorithms/spatial_and_multichannel/mvdr_beamforming.py` | None (generic/default path) |
| `spatial_and_multichannel.gsc_beamforming` | GSC beamforming | `src/pvx/algorithms/spatial_and_multichannel/gsc_beamforming.py` | None (generic/default path) |
| `spatial_and_multichannel.gcc_phat_localization` | GCC-PHAT localization | `src/pvx/algorithms/spatial_and_multichannel/gcc_phat_localization.py` | None (generic/default path) |
| `spatial_and_multichannel.stereo_decorrelation_for_width` | Stereo decorrelation for width | `src/pvx/algorithms/spatial_and_multichannel/stereo_decorrelation_for_width.py` | `delay_samples` |
| `spatial_and_multichannel.upmix_downmix_with_phase_coherent_routing` | Upmix/downmix with phase-coherent routing | `src/pvx/algorithms/spatial_and_multichannel/upmix_downmix_with_phase_coherent_routing.py` | `mode` |

### `creative_spectral_effects`

Theme: **Creative Spectral Effects**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `creative_spectral_effects.cross_synthesis_vocoder` | Cross-synthesis vocoder | `src/pvx/algorithms/creative_spectral_effects/cross_synthesis_vocoder.py` | None (generic/default path) |
| `creative_spectral_effects.spectral_convolution_effects` | Spectral convolution effects | `src/pvx/algorithms/creative_spectral_effects/spectral_convolution_effects.py` | `kernel_size` |
| `creative_spectral_effects.spectral_freeze_banks` | Spectral freeze banks | `src/pvx/algorithms/creative_spectral_effects/spectral_freeze_banks.py` | `frame_ratio` |
| `creative_spectral_effects.spectral_blur_smear` | Spectral blur/smear | `src/pvx/algorithms/creative_spectral_effects/spectral_blur_smear.py` | None (generic/default path) |
| `creative_spectral_effects.phase_randomization_textures` | Phase randomization textures | `src/pvx/algorithms/creative_spectral_effects/phase_randomization_textures.py` | `strength` |
| `creative_spectral_effects.formant_painting_warping` | Formant painting/warping | `src/pvx/algorithms/creative_spectral_effects/formant_painting_warping.py` | `ratio` |
| `creative_spectral_effects.resonator_filterbank_morphing` | Resonator/filterbank morphing | `src/pvx/algorithms/creative_spectral_effects/resonator_filterbank_morphing.py` | None (generic/default path) |
| `creative_spectral_effects.spectral_contrast_exaggeration` | Spectral contrast exaggeration | `src/pvx/algorithms/creative_spectral_effects/spectral_contrast_exaggeration.py` | `amount` |

### `granular_and_modulation`

Theme: **Granular and Modulation**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `granular_and_modulation.granular_time_stretch_engine` | Granular time-stretch engine | `src/pvx/algorithms/granular_and_modulation/granular_time_stretch_engine.py` | `stretch` |
| `granular_and_modulation.grain_cloud_pitch_textures` | Grain-cloud pitch textures | `src/pvx/algorithms/granular_and_modulation/grain_cloud_pitch_textures.py` | `seed`, `grain`, `count`, `stretch` |
| `granular_and_modulation.freeze_grain_morphing` | Freeze-grain morphing | `src/pvx/algorithms/granular_and_modulation/freeze_grain_morphing.py` | `grain`, `start` |
| `granular_and_modulation.am_fm_ring_modulation_blocks` | AM/FM/ring modulation blocks | `src/pvx/algorithms/granular_and_modulation/am_fm_ring_modulation_blocks.py` | `freq_hz` |
| `granular_and_modulation.spectral_tremolo` | Spectral tremolo | `src/pvx/algorithms/granular_and_modulation/spectral_tremolo.py` | `lfo_hz` |
| `granular_and_modulation.formant_lfo_modulation` | Formant LFO modulation | `src/pvx/algorithms/granular_and_modulation/formant_lfo_modulation.py` | `lfo_hz` |
| `granular_and_modulation.rhythmic_gate_stutter_quantizer` | Rhythmic gate/stutter quantizer | `src/pvx/algorithms/granular_and_modulation/rhythmic_gate_stutter_quantizer.py` | `rate_hz` |
| `granular_and_modulation.envelope_followed_modulation_routing` | Envelope-followed modulation routing | `src/pvx/algorithms/granular_and_modulation/envelope_followed_modulation_routing.py` | `depth` |

### `analysis_qa_and_automation`

Theme: **Analysis, QA, and Automation**

| Algorithm ID | Algorithm name | Module Path | Parameter keys |
| --- | --- | --- | --- |
| `analysis_qa_and_automation.onset_beat_downbeat_tracking` | Onset/beat/downbeat tracking | `src/pvx/algorithms/analysis_qa_and_automation/onset_beat_downbeat_tracking.py` | None (generic/default path) |
| `analysis_qa_and_automation.key_chord_detection` | Key/chord detection | `src/pvx/algorithms/analysis_qa_and_automation/key_chord_detection.py` | None (generic/default path) |
| `analysis_qa_and_automation.structure_segmentation_verse_chorus_sections` | Structure segmentation (verse/chorus/sections) | `src/pvx/algorithms/analysis_qa_and_automation/structure_segmentation_verse_chorus_sections.py` | None (generic/default path) |
| `analysis_qa_and_automation.silence_speech_music_classifiers` | Silence/speech/music classifiers | `src/pvx/algorithms/analysis_qa_and_automation/silence_speech_music_classifiers.py` | None (generic/default path) |
| `analysis_qa_and_automation.clip_hum_buzz_artifact_detection` | Clip/hum/buzz artifact detection | `src/pvx/algorithms/analysis_qa_and_automation/clip_hum_buzz_artifact_detection.py` | None (generic/default path) |
| `analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics` | PESQ/STOI/VISQOL quality metrics | `src/pvx/algorithms/analysis_qa_and_automation/pesq_stoi_visqol_quality_metrics.py` | None (generic/default path) |
| `analysis_qa_and_automation.auto_parameter_tuning_bayesian_optimization` | Auto-parameter tuning (Bayesian optimization) | `src/pvx/algorithms/analysis_qa_and_automation/auto_parameter_tuning_bayesian_optimization.py` | `target_centroid` |
| `analysis_qa_and_automation.batch_preset_recommendation_based_on_source_features` | Batch preset recommendation based on source features | `src/pvx/algorithms/analysis_qa_and_automation/batch_preset_recommendation_based_on_source_features.py` | None (generic/default path) |
<!-- END ALGORITHM CATALOG -->


## Shared Conventions (Most Tools)

Most `pvx*` scripts are intentionally aligned around one CLI contract for batch audio workflows.  
`pvxmorph.py` is the main exception (two explicit inputs and one explicit output path).

### I/O and Output Conventions

| Area | Flags | Behavior | Notes |
| --- | --- | --- | --- |
| Input selection | `inputs` (positional) | Accepts one or more paths/globs | If no paths resolve, the parser exits with an error. |
| Output location | `-o`, `--output-dir` | Writes beside source file when omitted | Applies to tools using shared I/O args. |
| Output naming | `--suffix`, `--output-format` | Appends tool suffix and optional extension override | Examples: `_harm`, `_denoise`, `_retune`. |
| Overwrite policy | `--overwrite` | Allows replacing an existing output file | Without it, existing output is an error. |
| Dry-run mode | `--dry-run` | Validates and resolves outputs without writing audio | Useful for large batch verification. |
| File encoding | `--subtype` | Sets libsndfile subtype (`PCM_16`, `PCM_24`, `FLOAT`, etc.) | Leave unset for default output subtype. |

### Console Output and Verbosity

Default behavior is a live status bar plus final summary lines.  
Status bars are shown unless quiet/silent mode is active.

| Control | Effective level | What you see |
| --- | --- | --- |
| No verbosity flags | `normal` | Status bar, final `[done]` summary, and errors (if any). |
| `-v` or `--verbosity verbose` | `verbose` | Normal output plus per-file `[ok]` details. |
| `-vv` or `--verbosity debug` | `debug` | Maximum detail level (currently most visible in `pvxvoc`). |
| `--quiet` or `--verbosity quiet` | `quiet` | No status bar and reduced non-error output. |
| `--silent` or `--verbosity silent` | `silent` | Suppresses all console output. |

Precedence notes:

- Repeating `-v` increases detail (`-vv`, `-vvv`), capped at `debug`.
- `--quiet`/`--silent` override higher verbosity requests.
- `pvxvoc.py` still accepts legacy `--no-progress` as a hidden alias mapped to quiet-style output.

### Signal Level and Clipping Controls

| Flag | Values | Purpose |
| --- | --- | --- |
| `--normalize` | `none`, `peak`, `rms` | Applies output normalization strategy. |
| `--peak-dbfs` | float dBFS | Target peak when `--normalize peak`. |
| `--rms-dbfs` | float dBFS | Target RMS when `--normalize rms`. |
| `--clip` | boolean | Hard-limits output samples to `[-1.0, 1.0]`. |

### Microtonal Pitch Controls

| Tool | Microtonal inputs | Notes |
| --- | --- | --- |
| `pvxvoc.py` | `--pitch-shift-cents`, fractional `--pitch-shift-semitones`, `--pitch-shift-ratio` | One pitch selector at a time via mutual exclusivity. |
| `pvxtransient.py` | `--pitch-shift-cents`, fractional `--pitch-shift-semitones`, `--pitch-shift-ratio` | Cents are additive to semitones unless ratio is used. |
| `pvxformant.py` | `--pitch-shift-cents`, fractional `--pitch-shift-semitones` | Cents are additive to semitones. |
| `pvxharmonize.py` | fractional `--intervals`, `--intervals-cents` | Cents offsets are per voice and added to intervals. |
| `pvxlayer.py` | `--harmonic-pitch-cents`, `--percussive-pitch-cents` plus semitone controls | Harmonic/percussive paths are controlled independently. |
| `pvxretune.py` | `--scale-cents` | Custom octave degree map in cents relative to `--root`. |
| `pvxconform.py` | CSV `pitch_cents` / `pitch_ratio` / `pitch_semitones` | Each segment row provides exactly one pitch field. |

### STFT and Windowing Controls

| Area | Flags | Purpose |
| --- | --- | --- |
| FFT framing | `--n-fft`, `--win-length`, `--hop-size` | Controls frequency/time resolution and overlap. |
| Window shape | `--window` | Selects one of 50 available analysis windows. |
| Kaiser tuning | `--kaiser-beta` | Beta parameter when `--window kaiser` is selected. |
| Frame centering | `--no-center` | Disables centered framing. |

Window families include classic, cosine-sum, tapered, gaussian/generalized gaussian, exponential, cauchy, cosine-power, hann-poisson, and general-hamming variants.

All supported windows: `hann`, `hamming`, `blackman`, `blackmanharris`, `nuttall`, `flattop`, `blackman_nuttall`, `exact_blackman`, `sine`, `bartlett`, `boxcar`, `triangular`, `bartlett_hann`, `tukey`, `tukey_0p1`, `tukey_0p25`, `tukey_0p75`, `tukey_0p9`, `parzen`, `lanczos`, `welch`, `gaussian_0p25`, `gaussian_0p35`, `gaussian_0p45`, `gaussian_0p55`, `gaussian_0p65`, `general_gaussian_1p5_0p35`, `general_gaussian_2p0_0p35`, `general_gaussian_3p0_0p35`, `general_gaussian_4p0_0p35`, `exponential_0p25`, `exponential_0p5`, `exponential_1p0`, `cauchy_0p5`, `cauchy_1p0`, `cauchy_2p0`, `cosine_power_2`, `cosine_power_3`, `cosine_power_4`, `hann_poisson_0p5`, `hann_poisson_1p0`, `hann_poisson_2p0`, `general_hamming_0p50`, `general_hamming_0p60`, `general_hamming_0p70`, `general_hamming_0p80`, `bohman`, `cosine`, `kaiser`, `rect`.

Use `--help` on any `pvx*` tool to view the exact current window list.

### Runtime and Acceleration Controls

| Flag | Values | Behavior |
| --- | --- | --- |
| `--device` | `auto`, `cpu`, `cuda` | Chooses runtime backend for FFT/vocoder operations. |
| `--cuda-device` | non-negative integer | Selects CUDA GPU index when CUDA backend is active. |

### Exit Behavior

| Scenario | Exit code | Behavior |
| --- | --- | --- |
| All files processed successfully | `0` | Normal completion. |
| One or more files fail in batch mode | `1` | Tool keeps processing remaining files, then reports failures. |
| Invalid arguments/configuration | non-zero | Parser/runtime validation exits before processing. |

## Tool Reference

### 1. `pvxvoc.py`

Primary phase-vocoder engine for batch time/pitch processing.

Key capabilities:

- Multi-file + multi-channel processing.
- Time stretch by ratio (`--time-stretch`) or absolute length (`--target-duration`).
- Pitch by semitones, cents, ratio, or target F0 (`--target-f0`).
- Formant-preserving pitch mode (`--pitch-mode formant-preserving`).
- Transient-preserve + phase locking.
- Fourier-sync mode (`--fourier-sync`) with fundamental frame locking.
- Live status bar by default (`--quiet` or `--silent` suppresses it; legacy `--no-progress` is accepted as an alias).

Example:

```bash
python3 pvxvoc.py stems/*.wav \
  --time-stretch 1.2 \
  --pitch-shift-semitones -2 \
  --phase-locking identity \
  --transient-preserve \
  --normalize peak --peak-dbfs -1
```

Fourier-sync example:

```bash
python3 pvxvoc.py vocal.wav \
  --fourier-sync --n-fft 1500 --win-length 1500 --hop-size 375 \
  --f0-min 70 --f0-max 500 --time-stretch 1.25
```

### 2. `pvxfreeze.py`

Freezes one spectral snapshot and synthesizes a sustained output.

Key options:

- `--freeze-time`: time (seconds) where freeze frame is captured.
- `--duration`: length of rendered freeze.
- `--random-phase`: subtle animated phase drift.

Example:

```bash
python3 pvxfreeze.py pad.wav --freeze-time 0.42 --duration 8.0 --random-phase
```

### 3. `pvxharmonize.py`

Creates harmony voices by pitch-shifting and summing.

Key options:

- `--intervals`: semitone list per voice (fractional values allowed).
- `--intervals-cents`: optional per-voice cents offsets added to intervals.
- `--gains`: per-voice gains.
- `--pans`: per-voice pan in `[-1, 1]`.
- `--force-stereo`: force stereo render path.

Example:

```bash
python3 pvxharmonize.py lead.wav \
  --intervals 0,3,7 \
  --gains 1.0,0.8,0.7 \
  --pans -0.4,0,0.4 \
  --force-stereo
```

### 4. `pvxconform.py`

Conforms timing and pitch by segments from CSV.

Required CSV columns:

- `start_sec`
- `end_sec`
- `stretch`
- one pitch field per row: `pitch_semitones` or `pitch_cents` or `pitch_ratio`

Key options:

- `--map`: CSV path.
- `--crossfade-ms`: segment boundary smoothing.

Example CSV (`map_conform.csv`):

```csv
start_sec,end_sec,stretch,pitch_cents
0.0,1.0,1.0,0
1.0,2.4,1.2,35
2.4,3.1,0.9,-28
```

Example command:

```bash
python3 pvxconform.py vocal.wav --map map_conform.csv --crossfade-ms 10
```

### 5. `pvxmorph.py`

Morphs two inputs in STFT magnitude/phase space.

Key options:

- `input_a`, `input_b`: two source files.
- `--alpha`: morph blend (`0` = A, `1` = B).
- `-o/--output`: explicit output path.

Example:

```bash
python3 pvxmorph.py source_a.wav source_b.wav --alpha 0.35 -o hybrid.wav
```

### 6. `pvxwarp.py`

Time-only variable stretch from CSV map.

Required CSV columns:

- `start_sec`
- `end_sec`
- `stretch`

Key options:

- `--map`: CSV path.
- `--crossfade-ms`: segment joins.

Example CSV (`map_warp.csv`):

```csv
start_sec,end_sec,stretch
0.0,0.8,1.0
0.8,2.0,1.3
2.0,3.0,0.85
```

Example command:

```bash
python3 pvxwarp.py drums.wav --map map_warp.csv
```

### 7. `pvxformant.py`

Formant-domain processing with optional pitch shift stage.

Key options:

- `--pitch-shift-semitones`: optional pitch shift before formant stage.
- `--pitch-shift-cents`: microtonal cents offset added to semitone shift.
- `--formant-shift-ratio`: formant move factor.
- `--mode {shift,preserve}`:
  - `shift`: explicitly shift formants.
  - `preserve`: compensate formants when pitch shifting.
- `--formant-lifter`, `--formant-max-gain-db`: envelope extraction and gain limits.

Example:

```bash
python3 pvxformant.py voice.wav \
  --pitch-shift-semitones 4 \
  --mode preserve \
  --formant-lifter 32
```

### 8. `pvxtransient.py`

Transient-emphasis variant of phase-vocoder processing.

Key options:

- `--time-stretch` or `--target-duration`.
- `--pitch-shift-semitones`, `--pitch-shift-cents`, or `--pitch-shift-ratio`.
- `--transient-threshold`: transient detector sensitivity.

Example:

```bash
python3 pvxtransient.py percussion.wav --time-stretch 1.35 --transient-threshold 1.4
```

### 9. `pvxunison.py`

Stereo thickening via multiple detuned voices.

Key options:

- `--voices`: number of unison voices.
- `--detune-cents`: overall detune span.
- `--width`: stereo spread control.
- `--dry-mix`: blend original signal.

Example:

```bash
python3 pvxunison.py synth.wav --voices 7 --detune-cents 18 --width 1.2 --dry-mix 0.15
```

### 10. `pvxdenoise.py`

Spectral subtraction denoiser with optional external noise reference.

Key options:

- `--noise-seconds`: profile from beginning of target file.
- `--noise-file`: optional separate noise capture.
- `--reduction-db`: subtraction strength.
- `--floor`: residual floor factor.
- `--smooth`: temporal smoothing span.

Example:

```bash
python3 pvxdenoise.py noisy.wav --noise-file noise_print.wav --reduction-db 14 --smooth 7
```

### 11. `pvxdeverb.py`

Reduces spectral tail smear (de-reverb style).

Key options:

- `--strength`: tail suppression amount.
- `--decay`: tail memory decay.
- `--floor`: per-bin floor safeguard.

Example:

```bash
python3 pvxdeverb.py room_take.wav --strength 0.5 --decay 0.9 --floor 0.12
```

### 12. `pvxretune.py`

Monophonic retuning toward a scale.

Key options:

- `--root`: tonic note (`C`, `C#`, ... `B`).
- `--scale`: `chromatic`, `major`, `minor`, `pentatonic`.
- `--scale-cents`: optional custom microtonal degree list (cents in one octave relative to `--root`).
- `--strength`: correction amount.
- `--chunk-ms`, `--overlap-ms`: analysis block design.
- `--f0-min`, `--f0-max`: F0 detection bounds.

Example:

```bash
python3 pvxretune.py vocal_mono.wav --root A --scale major --strength 0.9
```

### 13. `pvxlayer.py`

Splits audio into harmonic/percussive components, then applies independent pitch/time processing and recombines.

Key options:

- Harmonic path: `--harmonic-stretch`, `--harmonic-pitch-semitones`, `--harmonic-pitch-cents`, `--harmonic-gain`.
- Percussive path: `--percussive-stretch`, `--percussive-pitch-semitones`, `--percussive-pitch-cents`, `--percussive-gain`.
- HPSS controls: `--harmonic-kernel`, `--percussive-kernel`.

Example:

```bash
python3 pvxlayer.py loop.wav \
  --harmonic-stretch 1.2 --harmonic-pitch-semitones 3 --harmonic-gain 0.9 \
  --percussive-stretch 1.0 --percussive-pitch-semitones 0 --percussive-gain 1.1
```

## Build Executables (All Major Desktop OSes)

`pvx` uses [PyInstaller](https://pyinstaller.org/) for executable builds.

Important notes:

- Build on each target OS directly.
- Cross-OS builds are not officially supported (Windows binaries should be built on Windows, etc.).
- For broad compatibility, build on the oldest platform you intend to support.

### Common Build Preparation

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pyinstaller
```

### Build One Tool

Example (`pvxvoc`):

```bash
python3 -m PyInstaller --onefile --name pvxvoc --collect-all soundfile pvxvoc.py
```

### Build All Tools (macOS/Linux shell)

```bash
for tool in pvxvoc pvxfreeze pvxharmonize pvxconform pvxmorph pvxwarp pvxformant pvxtransient pvxunison pvxdenoise pvxdeverb pvxretune pvxlayer; do
  python3 -m PyInstaller --onefile --name "$tool" --collect-all soundfile "$tool.py"
done
```

### Build All Tools (Windows PowerShell)

```powershell
$tools = @("pvxvoc","pvxfreeze","pvxharmonize","pvxconform","pvxmorph","pvxwarp","pvxformant","pvxtransient","pvxunison","pvxdenoise","pvxdeverb","pvxretune","pvxlayer")
foreach ($tool in $tools) {
  python -m PyInstaller --onefile --name $tool --collect-all soundfile "$tool.py"
}
```

### Platform-Specific Notes

#### Windows

- Use PowerShell or Command Prompt.
- Output binaries are `.exe` files in `dist\`.
- Install the Microsoft VC++ runtime if running on clean machines.

Run example:

```powershell
.\dist\pvxvoc.exe --help
```

#### macOS (Apple Silicon + Intel)

- Build separately on Apple Silicon and Intel for native binaries.
- Gatekeeper may quarantine unsigned binaries distributed outside your machine.
- For distribution, code-sign and notarize executables.

Run example:

```bash
./dist/pvxvoc --help
```

#### Linux

- Build on a baseline distro for best `glibc` compatibility.
- Prefer same distro family/version as deployment targets.
- Output is an ELF binary in `dist/`.

Run example:

```bash
./dist/pvxvoc --help
```

### Onefile vs Onedir

- `--onefile`: single executable, easier distribution, slower startup.
- `--onedir`: folder output, faster startup, easier debugging.

Onedir example:

```bash
python3 -m PyInstaller --onedir --name pvxvoc --collect-all soundfile pvxvoc.py
```

## Validation / Test

Run unit + regression tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Run CLI help checks for all tools:

```bash
for tool in pvxvoc pvxfreeze pvxharmonize pvxconform pvxmorph pvxwarp pvxformant pvxtransient pvxunison pvxdenoise pvxdeverb pvxretune pvxlayer; do
  python3 "$tool.py" --help > /dev/null
done
```
