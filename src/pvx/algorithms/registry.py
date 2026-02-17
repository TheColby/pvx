"""Registry for generated pvx algorithm scaffolds."""

from __future__ import annotations

ALGORITHM_REGISTRY = {
    'time_scale_and_pitch_core.wsola_waveform_similarity_overlap_add': {
        'name': 'WSOLA (Waveform Similarity Overlap-Add)',
        'theme': 'Time-Scale and Pitch Core',
        'module': 'pvx.algorithms.time_scale_and_pitch_core.wsola_waveform_similarity_overlap_add',
    },
    'time_scale_and_pitch_core.td_psola': {
        'name': 'TD-PSOLA',
        'theme': 'Time-Scale and Pitch Core',
        'module': 'pvx.algorithms.time_scale_and_pitch_core.td_psola',
    },
    'time_scale_and_pitch_core.lp_psola': {
        'name': 'LP-PSOLA',
        'theme': 'Time-Scale and Pitch Core',
        'module': 'pvx.algorithms.time_scale_and_pitch_core.lp_psola',
    },
    'time_scale_and_pitch_core.multi_resolution_phase_vocoder': {
        'name': 'Multi-resolution phase vocoder',
        'theme': 'Time-Scale and Pitch Core',
        'module': 'pvx.algorithms.time_scale_and_pitch_core.multi_resolution_phase_vocoder',
    },
    'time_scale_and_pitch_core.harmonic_percussive_split_tsm': {
        'name': 'Harmonic/percussive split TSM',
        'theme': 'Time-Scale and Pitch Core',
        'module': 'pvx.algorithms.time_scale_and_pitch_core.harmonic_percussive_split_tsm',
    },
    'time_scale_and_pitch_core.beat_synchronous_time_warping': {
        'name': 'Beat-synchronous time warping',
        'theme': 'Time-Scale and Pitch Core',
        'module': 'pvx.algorithms.time_scale_and_pitch_core.beat_synchronous_time_warping',
    },
    'time_scale_and_pitch_core.nonlinear_time_maps': {
        'name': 'Nonlinear time maps (curves, anchors, spline timing)',
        'theme': 'Time-Scale and Pitch Core',
        'module': 'pvx.algorithms.time_scale_and_pitch_core.nonlinear_time_maps',
    },
    'pitch_detection_and_tracking.yin': {
        'name': 'YIN',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.yin',
    },
    'pitch_detection_and_tracking.pyin': {
        'name': 'pYIN',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.pyin',
    },
    'pitch_detection_and_tracking.rapt': {
        'name': 'RAPT',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.rapt',
    },
    'pitch_detection_and_tracking.swipe': {
        'name': 'SWIPE',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.swipe',
    },
    'pitch_detection_and_tracking.harmonic_product_spectrum_hps': {
        'name': 'Harmonic Product Spectrum (HPS)',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.harmonic_product_spectrum_hps',
    },
    'pitch_detection_and_tracking.subharmonic_summation': {
        'name': 'Subharmonic summation',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.subharmonic_summation',
    },
    'pitch_detection_and_tracking.crepe_style_neural_f0': {
        'name': 'CREPE-style neural F0',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.crepe_style_neural_f0',
    },
    'pitch_detection_and_tracking.viterbi_smoothed_pitch_contour_tracking': {
        'name': 'Viterbi-smoothed pitch contour tracking',
        'theme': 'Pitch Detection and Tracking',
        'module': 'pvx.algorithms.pitch_detection_and_tracking.viterbi_smoothed_pitch_contour_tracking',
    },
    'retune_and_intonation.chord_aware_retuning': {
        'name': 'Chord-aware retuning',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.chord_aware_retuning',
    },
    'retune_and_intonation.key_aware_retuning_with_confidence_weighting': {
        'name': 'Key-aware retuning with confidence weighting',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.key_aware_retuning_with_confidence_weighting',
    },
    'retune_and_intonation.just_intonation_mapping_per_key_center': {
        'name': 'Just intonation mapping per key center',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.just_intonation_mapping_per_key_center',
    },
    'retune_and_intonation.adaptive_intonation_context_sensitive_intervals': {
        'name': 'Adaptive intonation (context-sensitive intervals)',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.adaptive_intonation_context_sensitive_intervals',
    },
    'retune_and_intonation.scala_mts_scale_import_and_quantization': {
        'name': 'Scala/MTS scale import and quantization',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.scala_mts_scale_import_and_quantization',
    },
    'retune_and_intonation.time_varying_cents_maps': {
        'name': 'Time-varying cents maps',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.time_varying_cents_maps',
    },
    'retune_and_intonation.vibrato_preserving_correction': {
        'name': 'Vibrato-preserving correction',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.vibrato_preserving_correction',
    },
    'retune_and_intonation.portamento_aware_retune_curves': {
        'name': 'Portamento-aware retune curves',
        'theme': 'Retune and Intonation',
        'module': 'pvx.algorithms.retune_and_intonation.portamento_aware_retune_curves',
    },
    'spectral_time_frequency_transforms.constant_q_transform_cqt_processing': {
        'name': 'Constant-Q Transform (CQT) processing',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.constant_q_transform_cqt_processing',
    },
    'spectral_time_frequency_transforms.variable_q_transform_vqt': {
        'name': 'Variable-Q Transform (VQT)',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.variable_q_transform_vqt',
    },
    'spectral_time_frequency_transforms.nsgt_based_processing': {
        'name': 'NSGT-based processing',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.nsgt_based_processing',
    },
    'spectral_time_frequency_transforms.reassigned_spectrogram_methods': {
        'name': 'Reassigned spectrogram methods',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.reassigned_spectrogram_methods',
    },
    'spectral_time_frequency_transforms.synchrosqueezed_stft': {
        'name': 'Synchrosqueezed STFT',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.synchrosqueezed_stft',
    },
    'spectral_time_frequency_transforms.chirplet_transform_analysis': {
        'name': 'Chirplet transform analysis',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.chirplet_transform_analysis',
    },
    'spectral_time_frequency_transforms.wavelet_packet_processing': {
        'name': 'Wavelet packet processing',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.wavelet_packet_processing',
    },
    'spectral_time_frequency_transforms.multi_window_stft_fusion': {
        'name': 'Multi-window STFT fusion',
        'theme': 'Spectral and Time-Frequency Transforms',
        'module': 'pvx.algorithms.spectral_time_frequency_transforms.multi_window_stft_fusion',
    },
    'separation_and_decomposition.rpca_hpss': {
        'name': 'RPCA HPSS',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.rpca_hpss',
    },
    'separation_and_decomposition.nmf_decomposition': {
        'name': 'NMF decomposition',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.nmf_decomposition',
    },
    'separation_and_decomposition.ica_bss_for_multichannel_stems': {
        'name': 'ICA/BSS for multichannel stems',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.ica_bss_for_multichannel_stems',
    },
    'separation_and_decomposition.sinusoidal_residual_transient_decomposition': {
        'name': 'Sinusoidal+residual+transient decomposition',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.sinusoidal_residual_transient_decomposition',
    },
    'separation_and_decomposition.demucs_style_stem_separation_backend': {
        'name': 'Demucs-style stem separation backend',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.demucs_style_stem_separation_backend',
    },
    'separation_and_decomposition.u_net_vocal_accompaniment_split': {
        'name': 'U-Net vocal/accompaniment split',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.u_net_vocal_accompaniment_split',
    },
    'separation_and_decomposition.tensor_decomposition_cp_tucker': {
        'name': 'Tensor decomposition (CP/Tucker)',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.tensor_decomposition_cp_tucker',
    },
    'separation_and_decomposition.probabilistic_latent_component_separation': {
        'name': 'Probabilistic latent component separation',
        'theme': 'Separation and Decomposition',
        'module': 'pvx.algorithms.separation_and_decomposition.probabilistic_latent_component_separation',
    },
    'denoise_and_restoration.wiener_denoising': {
        'name': 'Wiener denoising',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.wiener_denoising',
    },
    'denoise_and_restoration.mmse_stsa': {
        'name': 'MMSE-STSA',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.mmse_stsa',
    },
    'denoise_and_restoration.log_mmse': {
        'name': 'Log-MMSE',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.log_mmse',
    },
    'denoise_and_restoration.minimum_statistics_noise_tracking': {
        'name': 'Minimum-statistics noise tracking',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.minimum_statistics_noise_tracking',
    },
    'denoise_and_restoration.rnnoise_style_denoiser': {
        'name': 'RNNoise-style denoiser',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.rnnoise_style_denoiser',
    },
    'denoise_and_restoration.diffusion_based_speech_audio_denoise': {
        'name': 'Diffusion-based speech/audio denoise',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.diffusion_based_speech_audio_denoise',
    },
    'denoise_and_restoration.declip_via_sparse_reconstruction': {
        'name': 'Declip via sparse reconstruction',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.declip_via_sparse_reconstruction',
    },
    'denoise_and_restoration.declick_decrackle_median_wavelet_interpolation': {
        'name': 'Declick/decrackle (median/wavelet + interpolation)',
        'theme': 'Denoise and Restoration',
        'module': 'pvx.algorithms.denoise_and_restoration.declick_decrackle_median_wavelet_interpolation',
    },
    'dereverb_and_room_correction.wpe_dereverberation': {
        'name': 'WPE dereverberation',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.wpe_dereverberation',
    },
    'dereverb_and_room_correction.spectral_decay_subtraction': {
        'name': 'Spectral decay subtraction',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.spectral_decay_subtraction',
    },
    'dereverb_and_room_correction.late_reverb_suppression_via_coherence': {
        'name': 'Late reverb suppression via coherence',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.late_reverb_suppression_via_coherence',
    },
    'dereverb_and_room_correction.room_impulse_inverse_filtering': {
        'name': 'Room impulse inverse filtering',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.room_impulse_inverse_filtering',
    },
    'dereverb_and_room_correction.multi_band_adaptive_deverb': {
        'name': 'Multi-band adaptive deverb',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.multi_band_adaptive_deverb',
    },
    'dereverb_and_room_correction.drr_guided_dereverb': {
        'name': 'DRR-guided dereverb',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.drr_guided_dereverb',
    },
    'dereverb_and_room_correction.blind_deconvolution_dereverb': {
        'name': 'Blind deconvolution dereverb',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.blind_deconvolution_dereverb',
    },
    'dereverb_and_room_correction.neural_dereverb_module': {
        'name': 'Neural dereverb module',
        'theme': 'Dereverb and Room Correction',
        'module': 'pvx.algorithms.dereverb_and_room_correction.neural_dereverb_module',
    },
    'dynamics_and_loudness.ebu_r128_normalization': {
        'name': 'EBU R128 normalization',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.ebu_r128_normalization',
    },
    'dynamics_and_loudness.itu_bs_1770_loudness_measurement_gating': {
        'name': 'ITU BS.1770 loudness measurement/gating',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.itu_bs_1770_loudness_measurement_gating',
    },
    'dynamics_and_loudness.multi_band_compression': {
        'name': 'Multi-band compression',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.multi_band_compression',
    },
    'dynamics_and_loudness.upward_compression': {
        'name': 'Upward compression',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.upward_compression',
    },
    'dynamics_and_loudness.transient_shaping': {
        'name': 'Transient shaping',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.transient_shaping',
    },
    'dynamics_and_loudness.spectral_dynamics_bin_wise_compressor_expander': {
        'name': 'Spectral dynamics (bin-wise compressor/expander)',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.spectral_dynamics_bin_wise_compressor_expander',
    },
    'dynamics_and_loudness.true_peak_limiting': {
        'name': 'True-peak limiting',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.true_peak_limiting',
    },
    'dynamics_and_loudness.lufs_target_mastering_chain': {
        'name': 'LUFS-target mastering chain',
        'theme': 'Dynamics and Loudness',
        'module': 'pvx.algorithms.dynamics_and_loudness.lufs_target_mastering_chain',
    },
    'spatial_and_multichannel.mid_side_adaptive_widening': {
        'name': 'Mid/Side adaptive widening',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.mid_side_adaptive_widening',
    },
    'spatial_and_multichannel.binaural_hrtf_rendering': {
        'name': 'Binaural HRTF rendering',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.binaural_hrtf_rendering',
    },
    'spatial_and_multichannel.ambisonics_encode_decode': {
        'name': 'Ambisonics encode/decode',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.ambisonics_encode_decode',
    },
    'spatial_and_multichannel.mvdr_beamforming': {
        'name': 'MVDR beamforming',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.mvdr_beamforming',
    },
    'spatial_and_multichannel.gsc_beamforming': {
        'name': 'GSC beamforming',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.gsc_beamforming',
    },
    'spatial_and_multichannel.gcc_phat_localization': {
        'name': 'GCC-PHAT localization',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.gcc_phat_localization',
    },
    'spatial_and_multichannel.stereo_decorrelation_for_width': {
        'name': 'Stereo decorrelation for width',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.stereo_decorrelation_for_width',
    },
    'spatial_and_multichannel.upmix_downmix_with_phase_coherent_routing': {
        'name': 'Upmix/downmix with phase-coherent routing',
        'theme': 'Spatial and Multichannel',
        'module': 'pvx.algorithms.spatial_and_multichannel.upmix_downmix_with_phase_coherent_routing',
    },
    'creative_spectral_effects.cross_synthesis_vocoder': {
        'name': 'Cross-synthesis vocoder',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.cross_synthesis_vocoder',
    },
    'creative_spectral_effects.spectral_convolution_effects': {
        'name': 'Spectral convolution effects',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.spectral_convolution_effects',
    },
    'creative_spectral_effects.spectral_freeze_banks': {
        'name': 'Spectral freeze banks',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.spectral_freeze_banks',
    },
    'creative_spectral_effects.spectral_blur_smear': {
        'name': 'Spectral blur/smear',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.spectral_blur_smear',
    },
    'creative_spectral_effects.phase_randomization_textures': {
        'name': 'Phase randomization textures',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.phase_randomization_textures',
    },
    'creative_spectral_effects.formant_painting_warping': {
        'name': 'Formant painting/warping',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.formant_painting_warping',
    },
    'creative_spectral_effects.resonator_filterbank_morphing': {
        'name': 'Resonator/filterbank morphing',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.resonator_filterbank_morphing',
    },
    'creative_spectral_effects.spectral_contrast_exaggeration': {
        'name': 'Spectral contrast exaggeration',
        'theme': 'Creative Spectral Effects',
        'module': 'pvx.algorithms.creative_spectral_effects.spectral_contrast_exaggeration',
    },
    'granular_and_modulation.granular_time_stretch_engine': {
        'name': 'Granular time-stretch engine',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.granular_time_stretch_engine',
    },
    'granular_and_modulation.grain_cloud_pitch_textures': {
        'name': 'Grain-cloud pitch textures',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.grain_cloud_pitch_textures',
    },
    'granular_and_modulation.freeze_grain_morphing': {
        'name': 'Freeze-grain morphing',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.freeze_grain_morphing',
    },
    'granular_and_modulation.am_fm_ring_modulation_blocks': {
        'name': 'AM/FM/ring modulation blocks',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.am_fm_ring_modulation_blocks',
    },
    'granular_and_modulation.spectral_tremolo': {
        'name': 'Spectral tremolo',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.spectral_tremolo',
    },
    'granular_and_modulation.formant_lfo_modulation': {
        'name': 'Formant LFO modulation',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.formant_lfo_modulation',
    },
    'granular_and_modulation.rhythmic_gate_stutter_quantizer': {
        'name': 'Rhythmic gate/stutter quantizer',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.rhythmic_gate_stutter_quantizer',
    },
    'granular_and_modulation.envelope_followed_modulation_routing': {
        'name': 'Envelope-followed modulation routing',
        'theme': 'Granular and Modulation',
        'module': 'pvx.algorithms.granular_and_modulation.envelope_followed_modulation_routing',
    },
    'analysis_qa_and_automation.onset_beat_downbeat_tracking': {
        'name': 'Onset/beat/downbeat tracking',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.onset_beat_downbeat_tracking',
    },
    'analysis_qa_and_automation.key_chord_detection': {
        'name': 'Key/chord detection',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.key_chord_detection',
    },
    'analysis_qa_and_automation.structure_segmentation_verse_chorus_sections': {
        'name': 'Structure segmentation (verse/chorus/sections)',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.structure_segmentation_verse_chorus_sections',
    },
    'analysis_qa_and_automation.silence_speech_music_classifiers': {
        'name': 'Silence/speech/music classifiers',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.silence_speech_music_classifiers',
    },
    'analysis_qa_and_automation.clip_hum_buzz_artifact_detection': {
        'name': 'Clip/hum/buzz artifact detection',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.clip_hum_buzz_artifact_detection',
    },
    'analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics': {
        'name': 'PESQ/STOI/VISQOL quality metrics',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics',
    },
    'analysis_qa_and_automation.auto_parameter_tuning_bayesian_optimization': {
        'name': 'Auto-parameter tuning (Bayesian optimization)',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.auto_parameter_tuning_bayesian_optimization',
    },
    'analysis_qa_and_automation.batch_preset_recommendation_based_on_source_features': {
        'name': 'Batch preset recommendation based on source features',
        'theme': 'Analysis, QA, and Automation',
        'module': 'pvx.algorithms.analysis_qa_and_automation.batch_preset_recommendation_based_on_source_features',
    },
}

ALGORITHM_COUNT = len(ALGORITHM_REGISTRY)

__all__ = ['ALGORITHM_COUNT', 'ALGORITHM_REGISTRY']
