# PVX Algorithm Parameter Reference

This file lists per-algorithm parameter keys consumed by `pvxalgorithms.base.run_algorithm()` dispatch.
Use these keys as `**params` when calling module `process(audio, sample_rate, **params)`. 

## `adaptive_intonation_context_sensitive_intervals`
- No algorithm-specific keys (uses generic/default path).

## `am_fm_ring_modulation_blocks`
- `freq_hz`

## `ambisonics_encode_decode`
- `angle_deg`

## `auto_parameter_tuning_bayesian_optimization`
- `target_centroid`

## `batch_preset_recommendation_based_on_source_features`
- No algorithm-specific keys (uses generic/default path).

## `beat_synchronous_time_warping`
- `stretch`

## `binaural_hrtf_rendering`
- `azimuth_deg`

## `blind_deconvolution_dereverb`
- No algorithm-specific keys (uses generic/default path).

## `chirplet_transform_analysis`
- No algorithm-specific keys (uses generic/default path).

## `chord_aware_retuning`
- `strength`

## `clip_hum_buzz_artifact_detection`
- No algorithm-specific keys (uses generic/default path).

## `crepe_style_neural_f0`
- No algorithm-specific keys (uses generic/default path).

## `cross_synthesis_vocoder`
- No algorithm-specific keys (uses generic/default path).

## `declick_decrackle_median_wavelet_interpolation`
- `spike_threshold`

## `declip_via_sparse_reconstruction`
- `clip_threshold`

## `demucs_style_stem_separation_backend`
- No algorithm-specific keys (uses generic/default path).

## `diffusion_based_speech_audio_denoise`
- No algorithm-specific keys (uses generic/default path).

## `drr_guided_dereverb`
- No algorithm-specific keys (uses generic/default path).

## `ebu_r128_normalization`
- `target_lufs`

## `envelope_followed_modulation_routing`
- `depth`

## `formant_lfo_modulation`
- `lfo_hz`

## `formant_painting_warping`
- `ratio`

## `freeze_grain_morphing`
- `grain`
- `start`

## `gcc_phat_localization`
- No algorithm-specific keys (uses generic/default path).

## `grain_cloud_pitch_textures`
- `seed`
- `grain`
- `count`
- `stretch`

## `granular_time_stretch_engine`
- `stretch`

## `gsc_beamforming`
- No algorithm-specific keys (uses generic/default path).

## `harmonic_percussive_split_tsm`
- `harmonic_stretch`
- `percussive_stretch`

## `harmonic_product_spectrum_hps`
- No algorithm-specific keys (uses generic/default path).

## `ica_bss_for_multichannel_stems`
- No algorithm-specific keys (uses generic/default path).

## `itu_bs_1770_loudness_measurement_gating`
- `gate_lufs`

## `just_intonation_mapping_per_key_center`
- No algorithm-specific keys (uses generic/default path).

## `key_aware_retuning_with_confidence_weighting`
- No algorithm-specific keys (uses generic/default path).

## `key_chord_detection`
- No algorithm-specific keys (uses generic/default path).

## `late_reverb_suppression_via_coherence`
- No algorithm-specific keys (uses generic/default path).

## `log_mmse`
- No algorithm-specific keys (uses generic/default path).

## `lp_psola`
- `semitones`

## `lufs_target_mastering_chain`
- `target_lufs`

## `mid_side_adaptive_widening`
- `width`

## `minimum_statistics_noise_tracking`
- No algorithm-specific keys (uses generic/default path).

## `mmse_stsa`
- No algorithm-specific keys (uses generic/default path).

## `multi_band_adaptive_deverb`
- No algorithm-specific keys (uses generic/default path).

## `multi_band_compression`
- No algorithm-specific keys (uses generic/default path).

## `multi_resolution_phase_vocoder`
- `stretch`

## `multi_window_stft_fusion`
- No algorithm-specific keys (uses generic/default path).

## `mvdr_beamforming`
- No algorithm-specific keys (uses generic/default path).

## `neural_dereverb_module`
- No algorithm-specific keys (uses generic/default path).

## `nmf_decomposition`
- `components`

## `nonlinear_time_maps`
- `curve`
- `stretch`
- `fmin`

## `onset_beat_downbeat_tracking`
- No algorithm-specific keys (uses generic/default path).

## `pesq_stoi_visqol_quality_metrics`
- No algorithm-specific keys (uses generic/default path).

## `phase_randomization_textures`
- `strength`

## `portamento_aware_retune_curves`
- `max_semitone_step`
- `compression`

## `probabilistic_latent_component_separation`
- No algorithm-specific keys (uses generic/default path).

## `pyin`
- No algorithm-specific keys (uses generic/default path).

## `rapt`
- No algorithm-specific keys (uses generic/default path).

## `reassigned_spectrogram_methods`
- No algorithm-specific keys (uses generic/default path).

## `resonator_filterbank_morphing`
- No algorithm-specific keys (uses generic/default path).

## `rhythmic_gate_stutter_quantizer`
- `rate_hz`

## `rnnoise_style_denoiser`
- No algorithm-specific keys (uses generic/default path).

## `room_impulse_inverse_filtering`
- No algorithm-specific keys (uses generic/default path).

## `rpca_hpss`
- No algorithm-specific keys (uses generic/default path).

## `scala_mts_scale_import_and_quantization`
- No algorithm-specific keys (uses generic/default path).

## `silence_speech_music_classifiers`
- No algorithm-specific keys (uses generic/default path).

## `sinusoidal_residual_transient_decomposition`
- No algorithm-specific keys (uses generic/default path).

## `spectral_blur_smear`
- No algorithm-specific keys (uses generic/default path).

## `spectral_contrast_exaggeration`
- `amount`

## `spectral_convolution_effects`
- `kernel_size`

## `spectral_decay_subtraction`
- No algorithm-specific keys (uses generic/default path).

## `spectral_dynamics_bin_wise_compressor_expander`
- `threshold_db`

## `spectral_freeze_banks`
- `frame_ratio`

## `spectral_tremolo`
- `lfo_hz`

## `stereo_decorrelation_for_width`
- `delay_samples`

## `structure_segmentation_verse_chorus_sections`
- No algorithm-specific keys (uses generic/default path).

## `subharmonic_summation`
- No algorithm-specific keys (uses generic/default path).

## `swipe`
- No algorithm-specific keys (uses generic/default path).

## `synchrosqueezed_stft`
- No algorithm-specific keys (uses generic/default path).

## `td_psola`
- `semitones`
- `stretch`

## `tensor_decomposition_cp_tucker`
- `rank`

## `time_varying_cents_maps`
- `cents_curve`

## `transient_shaping`
- `attack_boost`

## `true_peak_limiting`
- `threshold`

## `u_net_vocal_accompaniment_split`
- No algorithm-specific keys (uses generic/default path).

## `upmix_downmix_with_phase_coherent_routing`
- `mode`

## `upward_compression`
- `threshold_db`

## `vibrato_preserving_correction`
- `strength`

## `viterbi_smoothed_pitch_contour_tracking`
- `root_midi`
- `scale_cents`
- `scale`
- `fmin`

## `wavelet_packet_processing`
- No algorithm-specific keys (uses generic/default path).

## `wiener_denoising`
- No algorithm-specific keys (uses generic/default path).

## `wpe_dereverberation`
- `taps`

## `wsola_waveform_similarity_overlap_add`
- `stretch`
- `grain_size`

## `yin`
- No algorithm-specific keys (uses generic/default path).
