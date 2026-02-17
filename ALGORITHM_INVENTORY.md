# Generated Algorithm Inventory

Total algorithms: 123

## Time-Scale and Pitch Core
- `time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py`: WSOLA (Waveform Similarity Overlap-Add)
- `time_scale_and_pitch_core/td_psola.py`: TD-PSOLA
- `time_scale_and_pitch_core/lp_psola.py`: LP-PSOLA
- `time_scale_and_pitch_core/multi_resolution_phase_vocoder.py`: Multi-resolution phase vocoder
- `time_scale_and_pitch_core/harmonic_percussive_split_tsm.py`: Harmonic/percussive split TSM
- `time_scale_and_pitch_core/beat_synchronous_time_warping.py`: Beat-synchronous time warping
- `time_scale_and_pitch_core/nonlinear_time_maps.py`: Nonlinear time maps (curves, anchors, spline timing)

## Pitch Detection and Tracking
- `pitch_detection_and_tracking/yin.py`: YIN
- `pitch_detection_and_tracking/pyin.py`: pYIN
- `pitch_detection_and_tracking/rapt.py`: RAPT
- `pitch_detection_and_tracking/swipe.py`: SWIPE
- `pitch_detection_and_tracking/harmonic_product_spectrum_hps.py`: Harmonic Product Spectrum (HPS)
- `pitch_detection_and_tracking/subharmonic_summation.py`: Subharmonic summation
- `pitch_detection_and_tracking/crepe_style_neural_f0.py`: CREPE-style neural F0
- `pitch_detection_and_tracking/viterbi_smoothed_pitch_contour_tracking.py`: Viterbi-smoothed pitch contour tracking

## Retune and Intonation
- `retune_and_intonation/chord_aware_retuning.py`: Chord-aware retuning
- `retune_and_intonation/key_aware_retuning_with_confidence_weighting.py`: Key-aware retuning with confidence weighting
- `retune_and_intonation/just_intonation_mapping_per_key_center.py`: Just intonation mapping per key center
- `retune_and_intonation/adaptive_intonation_context_sensitive_intervals.py`: Adaptive intonation (context-sensitive intervals)
- `retune_and_intonation/scala_mts_scale_import_and_quantization.py`: Scala/MTS scale import and quantization
- `retune_and_intonation/time_varying_cents_maps.py`: Time-varying cents maps
- `retune_and_intonation/vibrato_preserving_correction.py`: Vibrato-preserving correction
- `retune_and_intonation/portamento_aware_retune_curves.py`: Portamento-aware retune curves

## Spectral and Time-Frequency Transforms
- `spectral_time_frequency_transforms/constant_q_transform_cqt_processing.py`: Constant-Q Transform (CQT) processing
- `spectral_time_frequency_transforms/variable_q_transform_vqt.py`: Variable-Q Transform (VQT)
- `spectral_time_frequency_transforms/nsgt_based_processing.py`: NSGT-based processing
- `spectral_time_frequency_transforms/reassigned_spectrogram_methods.py`: Reassigned spectrogram methods
- `spectral_time_frequency_transforms/synchrosqueezed_stft.py`: Synchrosqueezed STFT
- `spectral_time_frequency_transforms/chirplet_transform_analysis.py`: Chirplet transform analysis
- `spectral_time_frequency_transforms/wavelet_packet_processing.py`: Wavelet packet processing
- `spectral_time_frequency_transforms/multi_window_stft_fusion.py`: Multi-window STFT fusion

## Separation and Decomposition
- `separation_and_decomposition/rpca_hpss.py`: RPCA HPSS
- `separation_and_decomposition/nmf_decomposition.py`: NMF decomposition
- `separation_and_decomposition/ica_bss_for_multichannel_stems.py`: ICA/BSS for multichannel stems
- `separation_and_decomposition/sinusoidal_residual_transient_decomposition.py`: Sinusoidal+residual+transient decomposition
- `separation_and_decomposition/demucs_style_stem_separation_backend.py`: Demucs-style stem separation backend
- `separation_and_decomposition/u_net_vocal_accompaniment_split.py`: U-Net vocal/accompaniment split
- `separation_and_decomposition/tensor_decomposition_cp_tucker.py`: Tensor decomposition (CP/Tucker)
- `separation_and_decomposition/probabilistic_latent_component_separation.py`: Probabilistic latent component separation

## Denoise and Restoration
- `denoise_and_restoration/wiener_denoising.py`: Wiener denoising
- `denoise_and_restoration/mmse_stsa.py`: MMSE-STSA
- `denoise_and_restoration/log_mmse.py`: Log-MMSE
- `denoise_and_restoration/minimum_statistics_noise_tracking.py`: Minimum-statistics noise tracking
- `denoise_and_restoration/rnnoise_style_denoiser.py`: RNNoise-style denoiser
- `denoise_and_restoration/diffusion_based_speech_audio_denoise.py`: Diffusion-based speech/audio denoise
- `denoise_and_restoration/declip_via_sparse_reconstruction.py`: Declip via sparse reconstruction
- `denoise_and_restoration/declick_decrackle_median_wavelet_interpolation.py`: Declick/decrackle (median/wavelet + interpolation)

## Dereverb and Room Correction
- `dereverb_and_room_correction/wpe_dereverberation.py`: WPE dereverberation
- `dereverb_and_room_correction/spectral_decay_subtraction.py`: Spectral decay subtraction
- `dereverb_and_room_correction/late_reverb_suppression_via_coherence.py`: Late reverb suppression via coherence
- `dereverb_and_room_correction/room_impulse_inverse_filtering.py`: Room impulse inverse filtering
- `dereverb_and_room_correction/multi_band_adaptive_deverb.py`: Multi-band adaptive deverb
- `dereverb_and_room_correction/drr_guided_dereverb.py`: DRR-guided dereverb
- `dereverb_and_room_correction/blind_deconvolution_dereverb.py`: Blind deconvolution dereverb
- `dereverb_and_room_correction/neural_dereverb_module.py`: Neural dereverb module

## Dynamics and Loudness
- `dynamics_and_loudness/ebu_r128_normalization.py`: EBU R128 normalization
- `dynamics_and_loudness/itu_bs_1770_loudness_measurement_gating.py`: ITU BS.1770 loudness measurement/gating
- `dynamics_and_loudness/multi_band_compression.py`: Multi-band compression
- `dynamics_and_loudness/upward_compression.py`: Upward compression
- `dynamics_and_loudness/transient_shaping.py`: Transient shaping
- `dynamics_and_loudness/spectral_dynamics_bin_wise_compressor_expander.py`: Spectral dynamics (bin-wise compressor/expander)
- `dynamics_and_loudness/true_peak_limiting.py`: True-peak limiting
- `dynamics_and_loudness/lufs_target_mastering_chain.py`: LUFS-target mastering chain

## Creative Spectral Effects
- `creative_spectral_effects/cross_synthesis_vocoder.py`: Cross-synthesis vocoder
- `creative_spectral_effects/spectral_convolution_effects.py`: Spectral convolution effects
- `creative_spectral_effects/spectral_freeze_banks.py`: Spectral freeze banks
- `creative_spectral_effects/spectral_blur_smear.py`: Spectral blur/smear
- `creative_spectral_effects/phase_randomization_textures.py`: Phase randomization textures
- `creative_spectral_effects/formant_painting_warping.py`: Formant painting/warping
- `creative_spectral_effects/resonator_filterbank_morphing.py`: Resonator/filterbank morphing
- `creative_spectral_effects/spectral_contrast_exaggeration.py`: Spectral contrast exaggeration

## Granular and Modulation
- `granular_and_modulation/granular_time_stretch_engine.py`: Granular time-stretch engine
- `granular_and_modulation/grain_cloud_pitch_textures.py`: Grain-cloud pitch textures
- `granular_and_modulation/freeze_grain_morphing.py`: Freeze-grain morphing
- `granular_and_modulation/am_fm_ring_modulation_blocks.py`: AM/FM/ring modulation blocks
- `granular_and_modulation/spectral_tremolo.py`: Spectral tremolo
- `granular_and_modulation/formant_lfo_modulation.py`: Formant LFO modulation
- `granular_and_modulation/rhythmic_gate_stutter_quantizer.py`: Rhythmic gate/stutter quantizer
- `granular_and_modulation/envelope_followed_modulation_routing.py`: Envelope-followed modulation routing

## Analysis, QA, and Automation
- `analysis_qa_and_automation/onset_beat_downbeat_tracking.py`: Onset/beat/downbeat tracking
- `analysis_qa_and_automation/key_chord_detection.py`: Key/chord detection
- `analysis_qa_and_automation/structure_segmentation_verse_chorus_sections.py`: Structure segmentation (verse/chorus/sections)
- `analysis_qa_and_automation/silence_speech_music_classifiers.py`: Silence/speech/music classifiers
- `analysis_qa_and_automation/clip_hum_buzz_artifact_detection.py`: Clip/hum/buzz artifact detection
- `analysis_qa_and_automation/pesq_stoi_visqol_quality_metrics.py`: PESQ/STOI/VISQOL quality metrics
- `analysis_qa_and_automation/auto_parameter_tuning_bayesian_optimization.py`: Auto-parameter tuning (Bayesian optimization)
- `analysis_qa_and_automation/batch_preset_recommendation_based_on_source_features.py`: Batch preset recommendation based on source features

## Spatial and Multichannel
- `spatial_and_multichannel/imaging_and_panning/vbap_adaptive_panning.py`: VBAP adaptive panning
- `spatial_and_multichannel/imaging_and_panning/dbap_distance_based_amplitude_panning.py`: DBAP (distance-based amplitude panning)
- `spatial_and_multichannel/imaging_and_panning/binaural_itd_ild_synthesis.py`: Binaural ITD/ILD synthesis
- `spatial_and_multichannel/imaging_and_panning/transaural_crosstalk_cancellation.py`: Transaural crosstalk cancellation
- `spatial_and_multichannel/imaging_and_panning/stereo_width_frequency_dependent_control.py`: Stereo width (frequency-dependent control)
- `spatial_and_multichannel/imaging_and_panning/phase_aligned_mid_side_field_rotation.py`: Phase-aligned mid/side field rotation
- `spatial_and_multichannel/beamforming_and_directionality/delay_and_sum_beamforming.py`: Delay-and-sum beamforming
- `spatial_and_multichannel/beamforming_and_directionality/mvdr_beamformer_wideband.py`: MVDR beamformer (wideband)
- `spatial_and_multichannel/beamforming_and_directionality/generalized_sidelobe_canceller.py`: Generalized sidelobe canceller
- `spatial_and_multichannel/beamforming_and_directionality/superdirective_beamformer.py`: Superdirective beamformer
- `spatial_and_multichannel/beamforming_and_directionality/diffuse_field_coherence_masking.py`: Diffuse-field coherence masking
- `spatial_and_multichannel/beamforming_and_directionality/direction_of_arrival_grid_tracking.py`: Direction-of-arrival grid tracking
- `spatial_and_multichannel/ambisonics_and_immersive/first_order_ambisonic_encode_decode.py`: First-order ambisonic encode/decode
- `spatial_and_multichannel/ambisonics_and_immersive/higher_order_ambisonic_rotation.py`: Higher-order ambisonic rotation
- `spatial_and_multichannel/ambisonics_and_immersive/ambisonic_binaural_rendering.py`: Ambisonic binaural rendering
- `spatial_and_multichannel/ambisonics_and_immersive/spherical_harmonic_diffuse_enhancement.py`: Spherical-harmonic diffuse enhancement
- `spatial_and_multichannel/ambisonics_and_immersive/hoa_order_truncation_and_upmix.py`: HOA order truncation and upmix
- `spatial_and_multichannel/ambisonics_and_immersive/spatial_room_impulse_convolution.py`: Spatial room impulse convolution
- `spatial_and_multichannel/phase_vocoder_spatial/pvx_interchannel_phase_locking.py`: pvx interchannel phase locking
- `spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_transient_preservation.py`: pvx spatial transient preservation
- `spatial_and_multichannel/phase_vocoder_spatial/pvx_interaural_coherence_shaping.py`: pvx interaural coherence shaping
- `spatial_and_multichannel/phase_vocoder_spatial/pvx_directional_spectral_warp.py`: pvx directional spectral warp
- `spatial_and_multichannel/phase_vocoder_spatial/pvx_multichannel_time_alignment.py`: pvx multichannel time alignment
- `spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_freeze_and_trajectory.py`: pvx spatial freeze and trajectory
- `spatial_and_multichannel/multichannel_restoration/multichannel_wiener_postfilter.py`: Multichannel Wiener postfilter
- `spatial_and_multichannel/multichannel_restoration/coherence_based_dereverb_multichannel.py`: Coherence-based dereverb (multichannel)
- `spatial_and_multichannel/multichannel_restoration/multichannel_noise_psd_tracking.py`: Multichannel noise PSD tracking
- `spatial_and_multichannel/multichannel_restoration/phase_consistent_multichannel_denoise.py`: Phase-consistent multichannel denoise
- `spatial_and_multichannel/multichannel_restoration/microphone_array_calibration_tones.py`: Microphone-array calibration tones
- `spatial_and_multichannel/multichannel_restoration/cross_channel_click_pop_repair.py`: Cross-channel click/pop repair
- `spatial_and_multichannel/creative_spatial_fx/rotating_speaker_doppler_field.py`: Rotating-speaker Doppler field
- `spatial_and_multichannel/creative_spatial_fx/binaural_motion_trajectory_designer.py`: Binaural motion trajectory designer
- `spatial_and_multichannel/creative_spatial_fx/stochastic_spatial_diffusion_cloud.py`: Stochastic spatial diffusion cloud
- `spatial_and_multichannel/creative_spatial_fx/decorrelated_reverb_upmix.py`: Decorrelated reverb upmix
- `spatial_and_multichannel/creative_spatial_fx/spectral_spatial_granulator.py`: Spectral spatial granulator
- `spatial_and_multichannel/creative_spatial_fx/spatial_freeze_resynthesis.py`: Spatial freeze resynthesis
