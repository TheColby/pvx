<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# Python File Documentation and Help

Comprehensive reference for every Python file in this repository.

Total Python files documented: **268**

## Contents

- [`benchmarks/__init__.py`](#benchmarksinitpy)
- [`benchmarks/metrics.py`](#benchmarksmetricspy)
- [`benchmarks/run_augment_bench.py`](#benchmarksrunaugmentbenchpy)
- [`benchmarks/run_augment_profile_suite.py`](#benchmarksrunaugmentprofilesuitepy)
- [`benchmarks/run_augment_wer_bench.py`](#benchmarksrunaugmentwerbenchpy)
- [`benchmarks/run_bench.py`](#benchmarksrunbenchpy)
- [`benchmarks/run_pvc_parity.py`](#benchmarksrunpvcparitypy)
- [`examples/asr_augmentation.py`](#examplesasraugmentationpy)
- [`examples/huggingface_pipeline.py`](#exampleshuggingfacepipelinepy)
- [`examples/pytorch_dataset.py`](#examplespytorchdatasetpy)
- [`examples/speech_enhancement_pairs.py`](#examplesspeechenhancementpairspy)
- [`scripts/scripts_ab_compare.py`](#scriptsscriptsabcomparepy)
- [`scripts/scripts_alpha_check.py`](#scriptsscriptsalphacheckpy)
- [`scripts/scripts_apply_attribution.py`](#scriptsscriptsapplyattributionpy)
- [`scripts/scripts_benchmark_matrix.py`](#scriptsscriptsbenchmarkmatrixpy)
- [`scripts/scripts_check_dependency_sync.py`](#scriptsscriptscheckdependencysyncpy)
- [`scripts/scripts_generate_docs_extras.py`](#scriptsscriptsgeneratedocsextraspy)
- [`scripts/scripts_generate_docs_pdf.py`](#scriptsscriptsgeneratedocspdfpy)
- [`scripts/scripts_generate_html_docs.py`](#scriptsscriptsgeneratehtmldocspy)
- [`scripts/scripts_generate_python_docs.py`](#scriptsscriptsgeneratepythondocspy)
- [`scripts/scripts_generate_theory_docs.py`](#scriptsscriptsgeneratetheorydocspy)
- [`scripts/scripts_install_man_pages.py`](#scriptsscriptsinstallmanpagespy)
- [`scripts/scripts_quality_regression.py`](#scriptsscriptsqualityregressionpy)
- [`src/pvx/__init__.py`](#srcpvxinitpy)
- [`src/pvx/algorithms/__init__.py`](#srcpvxalgorithmsinitpy)
- [`src/pvx/algorithms/analysis_qa_and_automation/__init__.py`](#srcpvxalgorithmsanalysisqaandautomationinitpy)
- [`src/pvx/algorithms/analysis_qa_and_automation/auto_parameter_tuning_bayesian_optimization.py`](#srcpvxalgorithmsanalysisqaandautomationautoparametertuningbayesianoptimizationpy)
- [`src/pvx/algorithms/analysis_qa_and_automation/batch_preset_recommendation_based_on_source_features.py`](#srcpvxalgorithmsanalysisqaandautomationbatchpresetrecommendationbasedonsourcefeaturespy)
- [`src/pvx/algorithms/analysis_qa_and_automation/clip_hum_buzz_artifact_detection.py`](#srcpvxalgorithmsanalysisqaandautomationcliphumbuzzartifactdetectionpy)
- [`src/pvx/algorithms/analysis_qa_and_automation/key_chord_detection.py`](#srcpvxalgorithmsanalysisqaandautomationkeychorddetectionpy)
- [`src/pvx/algorithms/analysis_qa_and_automation/onset_beat_downbeat_tracking.py`](#srcpvxalgorithmsanalysisqaandautomationonsetbeatdownbeattrackingpy)
- [`src/pvx/algorithms/analysis_qa_and_automation/pesq_stoi_visqol_quality_metrics.py`](#srcpvxalgorithmsanalysisqaandautomationpesqstoivisqolqualitymetricspy)
- [`src/pvx/algorithms/analysis_qa_and_automation/silence_speech_music_classifiers.py`](#srcpvxalgorithmsanalysisqaandautomationsilencespeechmusicclassifierspy)
- [`src/pvx/algorithms/analysis_qa_and_automation/structure_segmentation_verse_chorus_sections.py`](#srcpvxalgorithmsanalysisqaandautomationstructuresegmentationversechorussectionspy)
- [`src/pvx/algorithms/audio_dispatch.py`](#srcpvxalgorithmsaudiodispatchpy)
- [`src/pvx/algorithms/base.py`](#srcpvxalgorithmsbasepy)
- [`src/pvx/algorithms/creative_spectral_effects/__init__.py`](#srcpvxalgorithmscreativespectraleffectsinitpy)
- [`src/pvx/algorithms/creative_spectral_effects/cross_synthesis_vocoder.py`](#srcpvxalgorithmscreativespectraleffectscrosssynthesisvocoderpy)
- [`src/pvx/algorithms/creative_spectral_effects/formant_painting_warping.py`](#srcpvxalgorithmscreativespectraleffectsformantpaintingwarpingpy)
- [`src/pvx/algorithms/creative_spectral_effects/phase_randomization_textures.py`](#srcpvxalgorithmscreativespectraleffectsphaserandomizationtexturespy)
- [`src/pvx/algorithms/creative_spectral_effects/resonator_filterbank_morphing.py`](#srcpvxalgorithmscreativespectraleffectsresonatorfilterbankmorphingpy)
- [`src/pvx/algorithms/creative_spectral_effects/spectral_blur_smear.py`](#srcpvxalgorithmscreativespectraleffectsspectralblursmearpy)
- [`src/pvx/algorithms/creative_spectral_effects/spectral_contrast_exaggeration.py`](#srcpvxalgorithmscreativespectraleffectsspectralcontrastexaggerationpy)
- [`src/pvx/algorithms/creative_spectral_effects/spectral_convolution_effects.py`](#srcpvxalgorithmscreativespectraleffectsspectralconvolutioneffectspy)
- [`src/pvx/algorithms/creative_spectral_effects/spectral_freeze_banks.py`](#srcpvxalgorithmscreativespectraleffectsspectralfreezebankspy)
- [`src/pvx/algorithms/denoise_and_restoration/__init__.py`](#srcpvxalgorithmsdenoiseandrestorationinitpy)
- [`src/pvx/algorithms/denoise_and_restoration/declick_decrackle_median_wavelet_interpolation.py`](#srcpvxalgorithmsdenoiseandrestorationdeclickdecracklemedianwaveletinterpolationpy)
- [`src/pvx/algorithms/denoise_and_restoration/declip_via_sparse_reconstruction.py`](#srcpvxalgorithmsdenoiseandrestorationdeclipviasparsereconstructionpy)
- [`src/pvx/algorithms/denoise_and_restoration/diffusion_based_speech_audio_denoise.py`](#srcpvxalgorithmsdenoiseandrestorationdiffusionbasedspeechaudiodenoisepy)
- [`src/pvx/algorithms/denoise_and_restoration/log_mmse.py`](#srcpvxalgorithmsdenoiseandrestorationlogmmsepy)
- [`src/pvx/algorithms/denoise_and_restoration/minimum_statistics_noise_tracking.py`](#srcpvxalgorithmsdenoiseandrestorationminimumstatisticsnoisetrackingpy)
- [`src/pvx/algorithms/denoise_and_restoration/mmse_stsa.py`](#srcpvxalgorithmsdenoiseandrestorationmmsestsapy)
- [`src/pvx/algorithms/denoise_and_restoration/rnnoise_style_denoiser.py`](#srcpvxalgorithmsdenoiseandrestorationrnnoisestyledenoiserpy)
- [`src/pvx/algorithms/denoise_and_restoration/wiener_denoising.py`](#srcpvxalgorithmsdenoiseandrestorationwienerdenoisingpy)
- [`src/pvx/algorithms/dereverb_and_room_correction/__init__.py`](#srcpvxalgorithmsdereverbandroomcorrectioninitpy)
- [`src/pvx/algorithms/dereverb_and_room_correction/blind_deconvolution_dereverb.py`](#srcpvxalgorithmsdereverbandroomcorrectionblinddeconvolutiondereverbpy)
- [`src/pvx/algorithms/dereverb_and_room_correction/drr_guided_dereverb.py`](#srcpvxalgorithmsdereverbandroomcorrectiondrrguideddereverbpy)
- [`src/pvx/algorithms/dereverb_and_room_correction/late_reverb_suppression_via_coherence.py`](#srcpvxalgorithmsdereverbandroomcorrectionlatereverbsuppressionviacoherencepy)
- [`src/pvx/algorithms/dereverb_and_room_correction/multi_band_adaptive_deverb.py`](#srcpvxalgorithmsdereverbandroomcorrectionmultibandadaptivedeverbpy)
- [`src/pvx/algorithms/dereverb_and_room_correction/neural_dereverb_module.py`](#srcpvxalgorithmsdereverbandroomcorrectionneuraldereverbmodulepy)
- [`src/pvx/algorithms/dereverb_and_room_correction/room_impulse_inverse_filtering.py`](#srcpvxalgorithmsdereverbandroomcorrectionroomimpulseinversefilteringpy)
- [`src/pvx/algorithms/dereverb_and_room_correction/spectral_decay_subtraction.py`](#srcpvxalgorithmsdereverbandroomcorrectionspectraldecaysubtractionpy)
- [`src/pvx/algorithms/dereverb_and_room_correction/wpe_dereverberation.py`](#srcpvxalgorithmsdereverbandroomcorrectionwpedereverberationpy)
- [`src/pvx/algorithms/dynamics_and_loudness/__init__.py`](#srcpvxalgorithmsdynamicsandloudnessinitpy)
- [`src/pvx/algorithms/dynamics_and_loudness/ebu_r128_normalization.py`](#srcpvxalgorithmsdynamicsandloudnessebur128normalizationpy)
- [`src/pvx/algorithms/dynamics_and_loudness/itu_bs_1770_loudness_measurement_gating.py`](#srcpvxalgorithmsdynamicsandloudnessitubs1770loudnessmeasurementgatingpy)
- [`src/pvx/algorithms/dynamics_and_loudness/lufs_target_mastering_chain.py`](#srcpvxalgorithmsdynamicsandloudnesslufstargetmasteringchainpy)
- [`src/pvx/algorithms/dynamics_and_loudness/multi_band_compression.py`](#srcpvxalgorithmsdynamicsandloudnessmultibandcompressionpy)
- [`src/pvx/algorithms/dynamics_and_loudness/spectral_dynamics_bin_wise_compressor_expander.py`](#srcpvxalgorithmsdynamicsandloudnessspectraldynamicsbinwisecompressorexpanderpy)
- [`src/pvx/algorithms/dynamics_and_loudness/transient_shaping.py`](#srcpvxalgorithmsdynamicsandloudnesstransientshapingpy)
- [`src/pvx/algorithms/dynamics_and_loudness/true_peak_limiting.py`](#srcpvxalgorithmsdynamicsandloudnesstruepeaklimitingpy)
- [`src/pvx/algorithms/dynamics_and_loudness/upward_compression.py`](#srcpvxalgorithmsdynamicsandloudnessupwardcompressionpy)
- [`src/pvx/algorithms/effects_dispatch.py`](#srcpvxalgorithmseffectsdispatchpy)
- [`src/pvx/algorithms/granular_and_modulation/__init__.py`](#srcpvxalgorithmsgranularandmodulationinitpy)
- [`src/pvx/algorithms/granular_and_modulation/am_fm_ring_modulation_blocks.py`](#srcpvxalgorithmsgranularandmodulationamfmringmodulationblockspy)
- [`src/pvx/algorithms/granular_and_modulation/envelope_followed_modulation_routing.py`](#srcpvxalgorithmsgranularandmodulationenvelopefollowedmodulationroutingpy)
- [`src/pvx/algorithms/granular_and_modulation/formant_lfo_modulation.py`](#srcpvxalgorithmsgranularandmodulationformantlfomodulationpy)
- [`src/pvx/algorithms/granular_and_modulation/freeze_grain_morphing.py`](#srcpvxalgorithmsgranularandmodulationfreezegrainmorphingpy)
- [`src/pvx/algorithms/granular_and_modulation/grain_cloud_pitch_textures.py`](#srcpvxalgorithmsgranularandmodulationgraincloudpitchtexturespy)
- [`src/pvx/algorithms/granular_and_modulation/granular_time_stretch_engine.py`](#srcpvxalgorithmsgranularandmodulationgranulartimestretchenginepy)
- [`src/pvx/algorithms/granular_and_modulation/rhythmic_gate_stutter_quantizer.py`](#srcpvxalgorithmsgranularandmodulationrhythmicgatestutterquantizerpy)
- [`src/pvx/algorithms/granular_and_modulation/spectral_tremolo.py`](#srcpvxalgorithmsgranularandmodulationspectraltremolopy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/__init__.py`](#srcpvxalgorithmspitchdetectionandtrackinginitpy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/crepe_style_neural_f0.py`](#srcpvxalgorithmspitchdetectionandtrackingcrepestyleneuralf0py)
- [`src/pvx/algorithms/pitch_detection_and_tracking/harmonic_product_spectrum_hps.py`](#srcpvxalgorithmspitchdetectionandtrackingharmonicproductspectrumhpspy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/pyin.py`](#srcpvxalgorithmspitchdetectionandtrackingpyinpy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/rapt.py`](#srcpvxalgorithmspitchdetectionandtrackingraptpy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/subharmonic_summation.py`](#srcpvxalgorithmspitchdetectionandtrackingsubharmonicsummationpy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/swipe.py`](#srcpvxalgorithmspitchdetectionandtrackingswipepy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/viterbi_smoothed_pitch_contour_tracking.py`](#srcpvxalgorithmspitchdetectionandtrackingviterbismoothedpitchcontourtrackingpy)
- [`src/pvx/algorithms/pitch_detection_and_tracking/yin.py`](#srcpvxalgorithmspitchdetectionandtrackingyinpy)
- [`src/pvx/algorithms/pitch_dispatch.py`](#srcpvxalgorithmspitchdispatchpy)
- [`src/pvx/algorithms/registry.py`](#srcpvxalgorithmsregistrypy)
- [`src/pvx/algorithms/retune_and_intonation/__init__.py`](#srcpvxalgorithmsretuneandintonationinitpy)
- [`src/pvx/algorithms/retune_and_intonation/adaptive_intonation_context_sensitive_intervals.py`](#srcpvxalgorithmsretuneandintonationadaptiveintonationcontextsensitiveintervalspy)
- [`src/pvx/algorithms/retune_and_intonation/chord_aware_retuning.py`](#srcpvxalgorithmsretuneandintonationchordawareretuningpy)
- [`src/pvx/algorithms/retune_and_intonation/just_intonation_mapping_per_key_center.py`](#srcpvxalgorithmsretuneandintonationjustintonationmappingperkeycenterpy)
- [`src/pvx/algorithms/retune_and_intonation/key_aware_retuning_with_confidence_weighting.py`](#srcpvxalgorithmsretuneandintonationkeyawareretuningwithconfidenceweightingpy)
- [`src/pvx/algorithms/retune_and_intonation/portamento_aware_retune_curves.py`](#srcpvxalgorithmsretuneandintonationportamentoawareretunecurvespy)
- [`src/pvx/algorithms/retune_and_intonation/scala_mts_scale_import_and_quantization.py`](#srcpvxalgorithmsretuneandintonationscalamtsscaleimportandquantizationpy)
- [`src/pvx/algorithms/retune_and_intonation/time_varying_cents_maps.py`](#srcpvxalgorithmsretuneandintonationtimevaryingcentsmapspy)
- [`src/pvx/algorithms/retune_and_intonation/vibrato_preserving_correction.py`](#srcpvxalgorithmsretuneandintonationvibratopreservingcorrectionpy)
- [`src/pvx/algorithms/separation_and_decomposition/__init__.py`](#srcpvxalgorithmsseparationanddecompositioninitpy)
- [`src/pvx/algorithms/separation_and_decomposition/demucs_style_stem_separation_backend.py`](#srcpvxalgorithmsseparationanddecompositiondemucsstylestemseparationbackendpy)
- [`src/pvx/algorithms/separation_and_decomposition/ica_bss_for_multichannel_stems.py`](#srcpvxalgorithmsseparationanddecompositionicabssformultichannelstemspy)
- [`src/pvx/algorithms/separation_and_decomposition/nmf_decomposition.py`](#srcpvxalgorithmsseparationanddecompositionnmfdecompositionpy)
- [`src/pvx/algorithms/separation_and_decomposition/probabilistic_latent_component_separation.py`](#srcpvxalgorithmsseparationanddecompositionprobabilisticlatentcomponentseparationpy)
- [`src/pvx/algorithms/separation_and_decomposition/rpca_hpss.py`](#srcpvxalgorithmsseparationanddecompositionrpcahpsspy)
- [`src/pvx/algorithms/separation_and_decomposition/sinusoidal_residual_transient_decomposition.py`](#srcpvxalgorithmsseparationanddecompositionsinusoidalresidualtransientdecompositionpy)
- [`src/pvx/algorithms/separation_and_decomposition/tensor_decomposition_cp_tucker.py`](#srcpvxalgorithmsseparationanddecompositiontensordecompositioncptuckerpy)
- [`src/pvx/algorithms/separation_and_decomposition/u_net_vocal_accompaniment_split.py`](#srcpvxalgorithmsseparationanddecompositionunetvocalaccompanimentsplitpy)
- [`src/pvx/algorithms/spatial_and_multichannel/__init__.py`](#srcpvxalgorithmsspatialandmultichannelinitpy)
- [`src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/__init__.py`](#srcpvxalgorithmsspatialandmultichannelcreativespatialfxinitpy)
- [`src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/binaural_motion_trajectory_designer.py`](#srcpvxalgorithmsspatialandmultichannelcreativespatialfxbinauralmotiontrajectorydesignerpy)
- [`src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/decorrelated_reverb_upmix.py`](#srcpvxalgorithmsspatialandmultichannelcreativespatialfxdecorrelatedreverbupmixpy)
- [`src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/rotating_speaker_doppler_field.py`](#srcpvxalgorithmsspatialandmultichannelcreativespatialfxrotatingspeakerdopplerfieldpy)
- [`src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spatial_freeze_resynthesis.py`](#srcpvxalgorithmsspatialandmultichannelcreativespatialfxspatialfreezeresynthesispy)
- [`src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spectral_spatial_granulator.py`](#srcpvxalgorithmsspatialandmultichannelcreativespatialfxspectralspatialgranulatorpy)
- [`src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/stochastic_spatial_diffusion_cloud.py`](#srcpvxalgorithmsspatialandmultichannelcreativespatialfxstochasticspatialdiffusioncloudpy)
- [`src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/__init__.py`](#srcpvxalgorithmsspatialandmultichannelimagingandpanninginitpy)
- [`src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/binaural_itd_ild_synthesis.py`](#srcpvxalgorithmsspatialandmultichannelimagingandpanningbinauralitdildsynthesispy)
- [`src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/dbap_distance_based_amplitude_panning.py`](#srcpvxalgorithmsspatialandmultichannelimagingandpanningdbapdistancebasedamplitudepanningpy)
- [`src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/phase_aligned_mid_side_field_rotation.py`](#srcpvxalgorithmsspatialandmultichannelimagingandpanningphasealignedmidsidefieldrotationpy)
- [`src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/stereo_width_frequency_dependent_control.py`](#srcpvxalgorithmsspatialandmultichannelimagingandpanningstereowidthfrequencydependentcontrolpy)
- [`src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/transaural_crosstalk_cancellation.py`](#srcpvxalgorithmsspatialandmultichannelimagingandpanningtransauralcrosstalkcancellationpy)
- [`src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/vbap_adaptive_panning.py`](#srcpvxalgorithmsspatialandmultichannelimagingandpanningvbapadaptivepanningpy)
- [`src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/__init__.py`](#srcpvxalgorithmsspatialandmultichannelmultichannelrestorationinitpy)
- [`src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/coherence_based_dereverb_multichannel.py`](#srcpvxalgorithmsspatialandmultichannelmultichannelrestorationcoherencebaseddereverbmultichannelpy)
- [`src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/cross_channel_click_pop_repair.py`](#srcpvxalgorithmsspatialandmultichannelmultichannelrestorationcrosschannelclickpoprepairpy)
- [`src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/microphone_array_calibration_tones.py`](#srcpvxalgorithmsspatialandmultichannelmultichannelrestorationmicrophonearraycalibrationtonespy)
- [`src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_noise_psd_tracking.py`](#srcpvxalgorithmsspatialandmultichannelmultichannelrestorationmultichannelnoisepsdtrackingpy)
- [`src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_wiener_postfilter.py`](#srcpvxalgorithmsspatialandmultichannelmultichannelrestorationmultichannelwienerpostfilterpy)
- [`src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/phase_consistent_multichannel_denoise.py`](#srcpvxalgorithmsspatialandmultichannelmultichannelrestorationphaseconsistentmultichanneldenoisepy)
- [`src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/__init__.py`](#srcpvxalgorithmsspatialandmultichannelphasevocoderspatialinitpy)
- [`src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_directional_spectral_warp.py`](#srcpvxalgorithmsspatialandmultichannelphasevocoderspatialpvxdirectionalspectralwarppy)
- [`src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interaural_coherence_shaping.py`](#srcpvxalgorithmsspatialandmultichannelphasevocoderspatialpvxinterauralcoherenceshapingpy)
- [`src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interchannel_phase_locking.py`](#srcpvxalgorithmsspatialandmultichannelphasevocoderspatialpvxinterchannelphaselockingpy)
- [`src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_multichannel_time_alignment.py`](#srcpvxalgorithmsspatialandmultichannelphasevocoderspatialpvxmultichanneltimealignmentpy)
- [`src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_freeze_and_trajectory.py`](#srcpvxalgorithmsspatialandmultichannelphasevocoderspatialpvxspatialfreezeandtrajectorypy)
- [`src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_transient_preservation.py`](#srcpvxalgorithmsspatialandmultichannelphasevocoderspatialpvxspatialtransientpreservationpy)
- [`src/pvx/algorithms/spatial_dispatch.py`](#srcpvxalgorithmsspatialdispatchpy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/__init__.py`](#srcpvxalgorithmsspectraltimefrequencytransformsinitpy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/chirplet_transform_analysis.py`](#srcpvxalgorithmsspectraltimefrequencytransformschirplettransformanalysispy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/constant_q_transform_cqt_processing.py`](#srcpvxalgorithmsspectraltimefrequencytransformsconstantqtransformcqtprocessingpy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/multi_window_stft_fusion.py`](#srcpvxalgorithmsspectraltimefrequencytransformsmultiwindowstftfusionpy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/nsgt_based_processing.py`](#srcpvxalgorithmsspectraltimefrequencytransformsnsgtbasedprocessingpy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/reassigned_spectrogram_methods.py`](#srcpvxalgorithmsspectraltimefrequencytransformsreassignedspectrogrammethodspy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/synchrosqueezed_stft.py`](#srcpvxalgorithmsspectraltimefrequencytransformssynchrosqueezedstftpy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/variable_q_transform_vqt.py`](#srcpvxalgorithmsspectraltimefrequencytransformsvariableqtransformvqtpy)
- [`src/pvx/algorithms/spectral_time_frequency_transforms/wavelet_packet_processing.py`](#srcpvxalgorithmsspectraltimefrequencytransformswaveletpacketprocessingpy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/__init__.py`](#srcpvxalgorithmstimescaleandpitchcoreinitpy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/beat_synchronous_time_warping.py`](#srcpvxalgorithmstimescaleandpitchcorebeatsynchronoustimewarpingpy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/harmonic_percussive_split_tsm.py`](#srcpvxalgorithmstimescaleandpitchcoreharmonicpercussivesplittsmpy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/lp_psola.py`](#srcpvxalgorithmstimescaleandpitchcorelppsolapy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/multi_resolution_phase_vocoder.py`](#srcpvxalgorithmstimescaleandpitchcoremultiresolutionphasevocoderpy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/nonlinear_time_maps.py`](#srcpvxalgorithmstimescaleandpitchcorenonlineartimemapspy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/td_psola.py`](#srcpvxalgorithmstimescaleandpitchcoretdpsolapy)
- [`src/pvx/algorithms/time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py`](#srcpvxalgorithmstimescaleandpitchcorewsolawaveformsimilarityoverlapaddpy)
- [`src/pvx/augment/__init__.py`](#srcpvxaugmentinitpy)
- [`src/pvx/augment/codec.py`](#srcpvxaugmentcodecpy)
- [`src/pvx/augment/config.py`](#srcpvxaugmentconfigpy)
- [`src/pvx/augment/core.py`](#srcpvxaugmentcorepy)
- [`src/pvx/augment/gpu.py`](#srcpvxaugmentgpupy)
- [`src/pvx/augment/ir_database.py`](#srcpvxaugmentirdatabasepy)
- [`src/pvx/augment/noise.py`](#srcpvxaugmentnoisepy)
- [`src/pvx/augment/room.py`](#srcpvxaugmentroompy)
- [`src/pvx/augment/spectral.py`](#srcpvxaugmentspectralpy)
- [`src/pvx/augment/streaming.py`](#srcpvxaugmentstreamingpy)
- [`src/pvx/augment/time_domain.py`](#srcpvxaugmenttimedomainpy)
- [`src/pvx/cli/__init__.py`](#srcpvxcliinitpy)
- [`src/pvx/cli/catalog.py`](#srcpvxclicatalogpy)
- [`src/pvx/cli/hps_pitch_track.py`](#srcpvxclihpspitchtrackpy)
- [`src/pvx/cli/main.py`](#srcpvxclimainpy)
- [`src/pvx/cli/pvx.py`](#srcpvxclipvxpy)
- [`src/pvx/cli/pvx_augment.py`](#srcpvxclipvxaugmentpy)
- [`src/pvx/cli/pvx_helpers.py`](#srcpvxclipvxhelperspy)
- [`src/pvx/cli/pvx_pipeline.py`](#srcpvxclipvxpipelinepy)
- [`src/pvx/cli/pvxanalysis.py`](#srcpvxclipvxanalysispy)
- [`src/pvx/cli/pvxbandamp.py`](#srcpvxclipvxbandamppy)
- [`src/pvx/cli/pvxchordmapper.py`](#srcpvxclipvxchordmapperpy)
- [`src/pvx/cli/pvxcodec.py`](#srcpvxclipvxcodecpy)
- [`src/pvx/cli/pvxconform.py`](#srcpvxclipvxconformpy)
- [`src/pvx/cli/pvxdenoise.py`](#srcpvxclipvxdenoisepy)
- [`src/pvx/cli/pvxdeverb.py`](#srcpvxclipvxdeverbpy)
- [`src/pvx/cli/pvxenvelope.py`](#srcpvxclipvxenvelopepy)
- [`src/pvx/cli/pvxfilter.py`](#srcpvxclipvxfilterpy)
- [`src/pvx/cli/pvxformant.py`](#srcpvxclipvxformantpy)
- [`src/pvx/cli/pvxfreeze.py`](#srcpvxclipvxfreezepy)
- [`src/pvx/cli/pvxgain.py`](#srcpvxclipvxgainpy)
- [`src/pvx/cli/pvxharmmap.py`](#srcpvxclipvxharmmappy)
- [`src/pvx/cli/pvxharmonize.py`](#srcpvxclipvxharmonizepy)
- [`src/pvx/cli/pvxinharmonator.py`](#srcpvxclipvxinharmonatorpy)
- [`src/pvx/cli/pvxlayer.py`](#srcpvxclipvxlayerpy)
- [`src/pvx/cli/pvxmorph.py`](#srcpvxclipvxmorphpy)
- [`src/pvx/cli/pvxnoise.py`](#srcpvxclipvxnoisepy)
- [`src/pvx/cli/pvxnoisefilter.py`](#srcpvxclipvxnoisefilterpy)
- [`src/pvx/cli/pvxreshape.py`](#srcpvxclipvxreshapepy)
- [`src/pvx/cli/pvxresponse.py`](#srcpvxclipvxresponsepy)
- [`src/pvx/cli/pvxretune.py`](#srcpvxclipvxretunepy)
- [`src/pvx/cli/pvxring.py`](#srcpvxclipvxringpy)
- [`src/pvx/cli/pvxringfilter.py`](#srcpvxclipvxringfilterpy)
- [`src/pvx/cli/pvxringtvfilter.py`](#srcpvxclipvxringtvfilterpy)
- [`src/pvx/cli/pvxrir.py`](#srcpvxclipvxrirpy)
- [`src/pvx/cli/pvxspecaugment.py`](#srcpvxclipvxspecaugmentpy)
- [`src/pvx/cli/pvxspeccompander.py`](#srcpvxclipvxspeccompanderpy)
- [`src/pvx/cli/pvxtrajectoryreverb.py`](#srcpvxclipvxtrajectoryreverbpy)
- [`src/pvx/cli/pvxtransient.py`](#srcpvxclipvxtransientpy)
- [`src/pvx/cli/pvxtvfilter.py`](#srcpvxclipvxtvfilterpy)
- [`src/pvx/cli/pvxunison.py`](#srcpvxclipvxunisonpy)
- [`src/pvx/cli/pvxvoc.py`](#srcpvxclipvxvocpy)
- [`src/pvx/cli/pvxwarp.py`](#srcpvxclipvxwarppy)
- [`src/pvx/core/__init__.py`](#srcpvxcoreinitpy)
- [`src/pvx/core/analysis_store.py`](#srcpvxcoreanalysisstorepy)
- [`src/pvx/core/attribution.py`](#srcpvxcoreattributionpy)
- [`src/pvx/core/audio_metrics.py`](#srcpvxcoreaudiometricspy)
- [`src/pvx/core/common.py`](#srcpvxcorecommonpy)
- [`src/pvx/core/control_bus.py`](#srcpvxcorecontrolbuspy)
- [`src/pvx/core/feature_tracking.py`](#srcpvxcorefeaturetrackingpy)
- [`src/pvx/core/output_policy.py`](#srcpvxcoreoutputpolicypy)
- [`src/pvx/core/presets.py`](#srcpvxcorepresetspy)
- [`src/pvx/core/pvc_functions.py`](#srcpvxcorepvcfunctionspy)
- [`src/pvx/core/pvc_harmony.py`](#srcpvxcorepvcharmonypy)
- [`src/pvx/core/pvc_ops.py`](#srcpvxcorepvcopspy)
- [`src/pvx/core/pvc_resonators.py`](#srcpvxcorepvcresonatorspy)
- [`src/pvx/core/response_store.py`](#srcpvxcoreresponsestorepy)
- [`src/pvx/core/spatial_reverb.py`](#srcpvxcorespatialreverbpy)
- [`src/pvx/core/stereo.py`](#srcpvxcorestereopy)
- [`src/pvx/core/streaming.py`](#srcpvxcorestreamingpy)
- [`src/pvx/core/transients.py`](#srcpvxcoretransientspy)
- [`src/pvx/core/voc.py`](#srcpvxcorevocpy)
- [`src/pvx/core/voc_console.py`](#srcpvxcorevocconsolepy)
- [`src/pvx/core/voc_jobs.py`](#srcpvxcorevocjobspy)
- [`src/pvx/core/wsola.py`](#srcpvxcorewsolapy)
- [`src/pvx/integrations/__init__.py`](#srcpvxintegrationsinitpy)
- [`src/pvx/integrations/huggingface.py`](#srcpvxintegrationshuggingfacepy)
- [`src/pvx/integrations/pytorch.py`](#srcpvxintegrationspytorchpy)
- [`src/pvx/integrations/tensorflow.py`](#srcpvxintegrationstensorflowpy)
- [`src/pvx/metrics/__init__.py`](#srcpvxmetricsinitpy)
- [`src/pvx/metrics/coherence.py`](#srcpvxmetricscoherencepy)
- [`src/pvx/voc_cli.py`](#srcpvxvocclipy)
- [`src/pvxalgorithms/__init__.py`](#srcpvxalgorithmsinitpy)
- [`src/pvxalgorithms/base.py`](#srcpvxalgorithmsbasepy)
- [`src/pvxalgorithms/registry.py`](#srcpvxalgorithmsregistrypy)
- [`tests/test_algorithms_generated.py`](#teststestalgorithmsgeneratedpy)
- [`tests/test_alpha_release_ready.py`](#teststestalphareleasereadypy)
- [`tests/test_analysis_response_store.py`](#teststestanalysisresponsestorepy)
- [`tests/test_audio_metrics.py`](#teststestaudiometricspy)
- [`tests/test_augment_bench_gate.py`](#teststestaugmentbenchgatepy)
- [`tests/test_augment_mode.py`](#teststestaugmentmodepy)
- [`tests/test_augment_transforms.py`](#teststestaugmenttransformspy)
- [`tests/test_benchmark_metrics.py`](#teststestbenchmarkmetricspy)
- [`tests/test_benchmark_runner.py`](#teststestbenchmarkrunnerpy)
- [`tests/test_cli_regression.py`](#teststestcliregressionpy)
- [`tests/test_control_bus.py`](#teststestcontrolbuspy)
- [`tests/test_docs_coverage.py`](#teststestdocscoveragepy)
- [`tests/test_docs_pdf.py`](#teststestdocspdfpy)
- [`tests/test_dsp.py`](#teststestdsppy)
- [`tests/test_gpu_transforms.py`](#teststestgputransformspy)
- [`tests/test_integrations.py`](#teststestintegrationspy)
- [`tests/test_microtonal.py`](#teststestmicrotonalpy)
- [`tests/test_output_policy.py`](#teststestoutputpolicypy)
- [`tests/test_pvc_parity_benchmark.py`](#teststestpvcparitybenchmarkpy)
- [`tests/test_pvc_phase3_5.py`](#teststestpvcphase35py)
- [`tests/test_pvc_phase6.py`](#teststestpvcphase6py)
- [`tests/test_pvxvoc_cli.py`](#teststestpvxvocclipy)
- [`tests/test_transient_and_stereo.py`](#teststesttransientandstereopy)
- [`tests/test_voc_console.py`](#teststestvocconsolepy)
- [`tests/test_voc_jobs.py`](#teststestvocjobspy)

## `benchmarks/__init__.py`

**Purpose:** Benchmark utilities for pvx quality regression.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Benchmark utilities for pvx quality regression.
```

## `benchmarks/metrics.py`

**Purpose:** Objective metrics for pvx benchmark comparisons.

**Classes:** `OptionalMetricValue`
**Functions:** `_match_length`, `_principal_angle`, `_to_mono`, `_stft_complex`, `_stft_mag_db`, `_resample_signal`, `_onset_envelope`, `_detect_onsets`, `_match_events`, `_attack_time_ms`, `_f0_track_and_voicing`, `_hnr_track`, `_cross_correlation_lag_samples`, `_run_external_quality_tool`, `_proxy_quality_scalar`, `log_spectral_distance`, `modulation_spectrum_distance`, `transient_smear_score`, `signal_to_noise_ratio_db`, `si_sdr_db`, `spectral_convergence`, `envelope_correlation`, `rms_level_delta_db`, `crest_factor_delta_db`, `bandwidth_95_delta_hz`, `zero_crossing_rate_delta`, `dc_offset_delta`, `clipping_ratio_delta`, `integrated_lufs_delta_lu`, `short_term_lufs_delta_lu`, `loudness_range_delta_lu`, `true_peak_delta_dbtp`, `pesq_mos_lqo`, `stoi_score`, `visqol_mos_lqo`, `polqa_mos_lqo`, `peaq_odg`, `f0_rmse_cents`, `voicing_f1_score`, `harmonic_to_noise_ratio_drift_db`, `onset_precision_recall_f1`, `attack_time_error_ms`, `ild_drift_db`, `itd_drift_ms`, `interchannel_phase_deviation_by_band`, `phasiness_index`, `musical_noise_index`, `pre_echo_score`, `stereo_coherence_drift`

### Module Docstring

```text
Objective metrics for pvx benchmark comparisons.
```

## `benchmarks/run_augment_bench.py`

**Purpose:** Benchmark runner for pvx augmentation workflows.

**Classes:** None
**Functions:** `_load_manifest_jsonl`, `_required_error_count`, `_safe_float`, `_split_balance_l1`, `_summarize`, `_relative_metric_gate`, `_absolute_metric_gate`, `_json_safe`, `_flag_present`, `_load_profiles`, `_profile_names`, `_resolve_profile_path`, `_apply_profile_overrides`, `main`

**Help commands:** `python3 benchmarks/run_augment_bench.py`, `python3 benchmarks/run_augment_bench.py --help`

### Module Docstring

```text
Benchmark runner for pvx augmentation workflows.

Supports profile-driven benchmark presets (speech/music/noisy/stereo) with
both relative (baseline drift) and absolute metric gates.
```

## `benchmarks/run_augment_profile_suite.py`

**Purpose:** Run all augmentation benchmark profiles and summarize regression status.

**Classes:** None
**Functions:** `_load_profiles`, `_load_json`, `main`

**Help commands:** `python3 benchmarks/run_augment_profile_suite.py`, `python3 benchmarks/run_augment_profile_suite.py --help`

### Module Docstring

```text
Run all augmentation benchmark profiles and summarize regression status.
```

## `benchmarks/run_augment_wer_bench.py`

**Purpose:** Benchmark: WER impact of pvx augmentation on a speech recognition task.

**Classes:** None
**Functions:** `parse_args`, `get_device`, `build_pipelines`, `prepare_dataset`, `train_and_evaluate`, `main`

**Help commands:** `python3 benchmarks/run_augment_wer_bench.py`, `python3 benchmarks/run_augment_wer_bench.py --help`

### Module Docstring

```text
Benchmark: WER impact of pvx augmentation on a speech recognition task.

This script measures the effect of ``pvx.augment`` pipelines on ASR accuracy
by fine-tuning a small Wav2Vec2 model on LibriSpeech-clean-100 with and
without augmentation, then evaluating WER on the ``test-clean`` split.

Requirements
------------
    pip install "pvx[torch]" datasets transformers evaluate jiwer torchaudio

Usage
-----
    # Quick benchmark (~20 min on a single GPU)
    python benchmarks/run_augment_wer_bench.py --epochs 3 --max-train-samples 2000

    # Full benchmark (~2-4 hours on a single GPU)
    python benchmarks/run_augment_wer_bench.py --epochs 10

    # CPU-only (slow, for verification only)
    python benchmarks/run_augment_wer_bench.py --epochs 2 --max-train-samples 500 --device cpu

Results are saved as JSON to ``benchmarks/out_augment/wer_results.json``.
```

## `benchmarks/run_bench.py`

**Purpose:** Reproducible quality benchmark: pvx vs Rubber Band vs librosa baseline.

**Classes:** `TaskSpec`
**Functions:** `_sha256_bytes`, `_sha256_file`, `_parse_version`, `_collect_environment_metadata`, `_corpus_manifest_entries`, `_load_manifest`, `_write_manifest`, `_manifest_index`, `_validate_corpus_against_manifest`, `_prepare_dataset`, `_case_key`, `_row_case_key`, `_diagnose_metrics`, `_method_diagnostics`, `_pvx_bench_args`, `_read_audio`, `_write_audio`, `_match_channels`, `_to_mono`, `_align_pair`, `_generate_tiny_dataset`, `_run_pvx_cycle`, `_find_rubberband`, `_run_rubberband_cycle`, `_run_librosa_cycle`, `_compute_metrics`, `_aggregate`, `_render_markdown`, `_check_gate`, `main`

**Help commands:** `python3 benchmarks/run_bench.py`, `python3 benchmarks/run_bench.py --help`

### Module Docstring

```text
Reproducible quality benchmark: pvx vs Rubber Band vs librosa baseline.
```

## `benchmarks/run_pvc_parity.py`

**Purpose:** PVC-style parity benchmark suite for phase 3-7 operators.

**Classes:** `CaseSpec`
**Functions:** `_to_mono`, `_match_length`, `_flat_response`, `_tilted_response`, `_generate_input`, `_case_specs`, `_run_tvfilter_envelope`, `_run_ringtv_controlled`, `_run_analysis_response_function_chain`, `_compute_case_metrics`, `_aggregate`, `_render_markdown`, `_gate_failures`, `main`

**Help commands:** `python3 benchmarks/run_pvc_parity.py`, `python3 benchmarks/run_pvc_parity.py --help`

### Module Docstring

```text
PVC-style parity benchmark suite for phase 3-7 operators.
```

## `examples/asr_augmentation.py`

**Purpose:** ASR data augmentation example using pvx.augment.

**Classes:** None
**Functions:** `build_asr_pipeline`, `augment_directory`, `main`

**Help commands:** `python3 examples/asr_augmentation.py`, `python3 examples/asr_augmentation.py --help`

### Module Docstring

```text
ASR data augmentation example using pvx.augment.

Demonstrates how to build a reproducible augmentation pipeline for
automatic speech recognition training, apply it to a directory of WAV
files, and write a JSONL manifest for downstream training scripts.

Usage
-----
    python examples/asr_augmentation.py         --input-dir data/train_clean         --output-dir data/train_aug         --variants 3         --workers 8         --seed 1337
```

## `examples/huggingface_pipeline.py`

**Purpose:** HuggingFace datasets augmentation example using pvx.

**Classes:** None
**Functions:** `build_augment_pipeline`, `main`

**Help commands:** `python3 examples/huggingface_pipeline.py`, `python3 examples/huggingface_pipeline.py --help`

### Module Docstring

```text
HuggingFace datasets augmentation example using pvx.

Demonstrates how to load a speech dataset from the HuggingFace Hub,
apply pvx augmentation, and save the result for training.

Requirements: pip install datasets transformers pvx

Usage
-----
    # Augment CommonVoice English (replace with any HF audio dataset)
    python examples/huggingface_pipeline.py         --dataset mozilla-foundation/common_voice_13_0         --subset en         --split train         --output-dir data/cv_aug         --variants 3         --workers 8

    # Augment a local audio folder
    python examples/huggingface_pipeline.py         --local-dir data/my_audio         --output-dir data/my_audio_aug
```

## `examples/pytorch_dataset.py`

**Purpose:** PyTorch DataLoader example with pvx augmentation.

**Classes:** None
**Functions:** `build_train_pipeline`, `build_val_pipeline`, `main`

**Help commands:** `python3 examples/pytorch_dataset.py`, `python3 examples/pytorch_dataset.py --help`

### Module Docstring

```text
PyTorch DataLoader example with pvx augmentation.

Shows how to use PvxAugmentDataset in a complete PyTorch training loop
with multi-worker data loading, variable-length collation, and
deterministic validation.

Requirements: pip install torch torchaudio

Usage
-----
    python examples/pytorch_dataset.py         --train-dir data/train         --val-dir data/val         --epochs 5
```

## `examples/speech_enhancement_pairs.py`

**Purpose:** Speech enhancement training data generation using pvx.

**Classes:** None
**Functions:** `build_degradation_pipeline`, `main`

**Help commands:** `python3 examples/speech_enhancement_pairs.py`, `python3 examples/speech_enhancement_pairs.py --help`

### Module Docstring

```text
Speech enhancement training data generation using pvx.

Generates paired (clean, noisy) audio for training speech enhancement,
dereverberation, and noise suppression models.

For each clean utterance the script produces N degraded versions with
different noise/reverb conditions.  The output JSONL manifest contains
the clean path, noisy path, and all augmentation parameters — enabling
fully reproducible dataset generation and ablation studies.

Usage
-----
    python examples/speech_enhancement_pairs.py         --clean-dir data/clean_speech         --output-dir data/enhancement_pairs         --pairs 5         --workers 8         --seed 1337
```

## `scripts/scripts_ab_compare.py`

**Purpose:** Run an A/B pvx render comparison and emit metrics reports.

**Classes:** None
**Functions:** `_metrics`, `_render`, `main`

**Help commands:** `python3 scripts/scripts_ab_compare.py`, `python3 scripts/scripts_ab_compare.py --help`

### Module Docstring

```text
Run an A/B pvx render comparison and emit metrics reports.
```

## `scripts/scripts_alpha_check.py`

**Purpose:** Run alpha-release validation checks without relying on `make`.

**Classes:** None
**Functions:** `main`

**Help commands:** `python3 scripts/scripts_alpha_check.py`

### Module Docstring

```text
Run alpha-release validation checks without relying on `make`.
```

## `scripts/scripts_apply_attribution.py`

**Purpose:** Apply centralized attribution references to Python and Markdown files.

**Classes:** None
**Functions:** `_is_excluded`, `_relative_attribution_path`, `_insert_python_header`, `_insert_markdown_notice`, `apply_python_headers`, `apply_markdown_notices`, `main`

**Help commands:** `python3 scripts/scripts_apply_attribution.py`, `python3 scripts/scripts_apply_attribution.py --help`

### Module Docstring

```text
Apply centralized attribution references to Python and Markdown files.
```

## `scripts/scripts_benchmark_matrix.py`

**Purpose:** Benchmark matrix runner for pvxvoc transform/window/device combinations.

**Classes:** None
**Functions:** `_parse_csv_tokens`, `_run_case`, `main`

**Help commands:** `python3 scripts/scripts_benchmark_matrix.py`, `python3 scripts/scripts_benchmark_matrix.py --help`

### Module Docstring

```text
Benchmark matrix runner for pvxvoc transform/window/device combinations.
```

## `scripts/scripts_check_dependency_sync.py`

**Purpose:** Validate that runtime requirements.txt matches project runtime dependencies.

**Classes:** None
**Functions:** `normalize_name`, `read_requirements`, `read_pyproject_runtime_dependencies`, `main`

**Help commands:** `python3 scripts/scripts_check_dependency_sync.py`

### Module Docstring

```text
Validate that runtime requirements.txt matches project runtime dependencies.
```

## `scripts/scripts_generate_docs_extras.py`

**Purpose:** Generate advanced docs artifacts (coverage, limitations, benchmarks, citations, cookbook, architecture).

**Classes:** None
**Functions:** `git_commit_meta`, `generated_stamp_lines`, `logo_lines`, `attribution_section_lines`, `write_json`, `_string_literal`, `_simple_literal`, `_tool_name_for_path`, `_iter_cli_sources`, `collect_cli_flags`, `generate_cli_flags_reference`, `_unique_join`, `generate_algorithm_limitations`, `generate_cookbook`, `generate_architecture_doc`, `_ensure_2d`, `_align_audio_pair`, `_finite_mean`, `_finite_max`, `_make_benchmark_cases`, `_quality_score_0_100`, `_compute_roundtrip_metrics`, `_roundtrip_stft_istft`, `_benchmark_backend_case`, `_summarize_backend_rows`, `generate_benchmarks`, `_classify_reference_url`, `_extract_doi`, `_bib_escape`, `_bib_key`, `generate_citation_docs`, `generate_docs_contract`, `main`

**Help commands:** `python3 scripts/scripts_generate_docs_extras.py`, `python3 scripts/scripts_generate_docs_extras.py --help`

### Module Docstring

```text
Generate advanced docs artifacts (coverage, limitations, benchmarks, citations, cookbook, architecture).
```

## `scripts/scripts_generate_docs_pdf.py`

**Purpose:** Generate one combined PDF from all HTML documentation pages.

**Classes:** `SourcePage`, `ProgressBar`
**Functions:** `add_console_args`, `console_level`, `is_quiet`, `is_silent`, `log`, `html_sort_key`, `collect_html_pages`, `extract_title`, `extract_main_html`, `parse_source_page`, `_rewrite_internal_links`, `_extract_reference_count`, `_annotate_display_equations`, `build_combined_html`, `discover_chromium_executable`, `run_cmd`, `render_pdf_with_chromium`, `render_pdf_with_wkhtmltopdf`, `render_pdf_with_weasyprint`, `render_pdf_with_playwright`, `build_engine_registry`, `auto_engine_order`, `render_pdf`, `parse_args`, `main`

**Help commands:** `python3 scripts/scripts_generate_docs_pdf.py`, `python3 scripts/scripts_generate_docs_pdf.py --help`

### Module Docstring

```text
Generate one combined PDF from all HTML documentation pages.
```

## `scripts/scripts_generate_html_docs.py`

**Purpose:** Generate grouped HTML documentation for pvx algorithms and research references.

**Classes:** None
**Functions:** `git_commit_meta`, `scholar`, `_contains_out_of_scope_text`, `_is_out_of_scope_paper`, `_is_out_of_scope_glossary`, `slugify`, `dedupe_papers`, `_upgrade_paper_url`, `upgrade_paper_urls`, `load_extra_papers`, `load_glossary`, `infer_glossary_terms`, `glossary_links_html`, `load_json`, `classify_reference_url`, `window_entries`, `window_tradeoffs`, `_split_top_level_once`, `_extract_params_get_calls`, `extract_algorithm_param_specs`, `extract_algorithm_params`, `extract_module_cli_flags`, `collect_algorithm_module_flags`, `sample_value_from_default`, `format_sample_params`, `compute_unique_cli_flags`, `grouped_algorithms`, `html_logo_src`, `html_attribution_href`, `html_page`, `write_style_css`, `render_index`, `module_path_from_meta`, `render_group_pages`, `render_papers_page`, `render_glossary_page`, `render_math_page`, `render_windows_page`, `render_architecture_page`, `render_cli_flags_page`, `render_limitations_page`, `render_benchmarks_page`, `render_cookbook_page`, `render_citations_page`, `write_docs_root_index`, `main`

**Help commands:** `python3 scripts/scripts_generate_html_docs.py`

### Module Docstring

```text
Generate grouped HTML documentation for pvx algorithms and research references.
```

## `scripts/scripts_generate_python_docs.py`

**Purpose:** Generate comprehensive documentation for every Python file in the repository.

**Classes:** None
**Functions:** `logo_lines`, `attribution_section_lines`, `rel`, `safe_read`, `module_name_for_path`, `parse_module`, `cli_help`, `extract_algorithm_params`, `generate_algorithm_param_doc`, `generate_python_help_doc`, `main`

**Help commands:** `python3 scripts/scripts_generate_python_docs.py`, `python3 scripts/scripts_generate_python_docs.py --help`

### Module Docstring

```text
Generate comprehensive documentation for every Python file in the repository.
```

## `scripts/scripts_generate_theory_docs.py`

**Purpose:** Generate GitHub-renderable theory docs (math foundations + window reference).

**Classes:** None
**Functions:** `git_commit_meta`, `generated_stamp_lines`, `logo_lines`, `attribution_section_lines`, `window_entries`, `window_tradeoffs`, `window_samples`, `_first_local_minimum`, `compute_window_metrics`, `_polyline_points`, `_downsample_series`, `write_line_svg`, `_svg_plot_points`, `write_multiline_svg`, `_compressor_curve_db`, `_expander_curve_db`, `_limiter_curve_db`, `_softclip_cubic`, `generate_function_assets`, `_natural_cubic_spline_eval`, `_sample_interpolation_curve`, `_render_interpolation_svg`, `generate_interpolation_assets`, `generate_window_assets_and_metrics`, `write_math_foundations`, `write_window_reference`, `main`

**Help commands:** `python3 scripts/scripts_generate_theory_docs.py`

### Module Docstring

```text
Generate GitHub-renderable theory docs (math foundations + window reference).
```

## `scripts/scripts_install_man_pages.py`

**Purpose:** Generate and optionally install pvx man pages.

**Classes:** None
**Functions:** `_run_help`, `_roff_escape`, `_build_man_page`, `_write_pages`, `_install_pages`, `main`

**Help commands:** `python3 scripts/scripts_install_man_pages.py`, `python3 scripts/scripts_install_man_pages.py --help`

### Module Docstring

```text
Generate and optionally install pvx man pages.
```

## `scripts/scripts_quality_regression.py`

**Purpose:** Render a pvx case and compare objective metrics against a baseline.

**Classes:** None
**Functions:** `_metrics`, `_compare`, `main`

**Help commands:** `python3 scripts/scripts_quality_regression.py`, `python3 scripts/scripts_quality_regression.py --help`

### Module Docstring

```text
Render a pvx case and compare objective metrics against a baseline.
```

## `src/pvx/__init__.py`

**Purpose:** pvx package root.

**Classes:** None
**Functions:** None

### Module Docstring

```text
pvx package root.

Contains stable CLI entrypoints (`pvx.cli`), reusable DSP/runtime core (`pvx.core`),
and the large algorithm library (`pvx.algorithms`).
```

## `src/pvx/algorithms/__init__.py`

**Purpose:** Generated algorithm wrappers over shared pvx dispatch implementations.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Generated algorithm wrappers over shared pvx dispatch implementations.
```

## `src/pvx/algorithms/analysis_qa_and_automation/__init__.py`

**Purpose:** Analysis, QA, and Automation algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Analysis, QA, and Automation algorithm scaffolds.
```

## `src/pvx/algorithms/analysis_qa_and_automation/auto_parameter_tuning_bayesian_optimization.py`

**Purpose:** Auto-parameter tuning (Bayesian optimization).

**Algorithm ID:** `analysis_qa_and_automation.auto_parameter_tuning_bayesian_optimization`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/auto_parameter_tuning_bayesian_optimization.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/auto_parameter_tuning_bayesian_optimization.py --help`

### Module Docstring

```text
Auto-parameter tuning (Bayesian optimization).

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.auto_parameter_tuning_bayesian_optimization
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/analysis_qa_and_automation/batch_preset_recommendation_based_on_source_features.py`

**Purpose:** Batch preset recommendation based on source features.

**Algorithm ID:** `analysis_qa_and_automation.batch_preset_recommendation_based_on_source_features`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/batch_preset_recommendation_based_on_source_features.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/batch_preset_recommendation_based_on_source_features.py --help`

### Module Docstring

```text
Batch preset recommendation based on source features.

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.batch_preset_recommendation_based_on_source_features
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/analysis_qa_and_automation/clip_hum_buzz_artifact_detection.py`

**Purpose:** Clip/hum/buzz artifact detection.

**Algorithm ID:** `analysis_qa_and_automation.clip_hum_buzz_artifact_detection`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/clip_hum_buzz_artifact_detection.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/clip_hum_buzz_artifact_detection.py --help`

### Module Docstring

```text
Clip/hum/buzz artifact detection.

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.clip_hum_buzz_artifact_detection
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/analysis_qa_and_automation/key_chord_detection.py`

**Purpose:** Key/chord detection.

**Algorithm ID:** `analysis_qa_and_automation.key_chord_detection`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/key_chord_detection.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/key_chord_detection.py --help`

### Module Docstring

```text
Key/chord detection.

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.key_chord_detection
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/analysis_qa_and_automation/onset_beat_downbeat_tracking.py`

**Purpose:** Onset/beat/downbeat tracking.

**Algorithm ID:** `analysis_qa_and_automation.onset_beat_downbeat_tracking`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/onset_beat_downbeat_tracking.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/onset_beat_downbeat_tracking.py --help`

### Module Docstring

```text
Onset/beat/downbeat tracking.

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.onset_beat_downbeat_tracking
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/analysis_qa_and_automation/pesq_stoi_visqol_quality_metrics.py`

**Purpose:** PESQ/STOI/VISQOL quality metrics.

**Algorithm ID:** `analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/pesq_stoi_visqol_quality_metrics.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/pesq_stoi_visqol_quality_metrics.py --help`

### Module Docstring

```text
PESQ/STOI/VISQOL quality metrics.

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.pesq_stoi_visqol_quality_metrics
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/analysis_qa_and_automation/silence_speech_music_classifiers.py`

**Purpose:** Silence/speech/music classifiers.

**Algorithm ID:** `analysis_qa_and_automation.silence_speech_music_classifiers`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/silence_speech_music_classifiers.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/silence_speech_music_classifiers.py --help`

### Module Docstring

```text
Silence/speech/music classifiers.

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.silence_speech_music_classifiers
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/analysis_qa_and_automation/structure_segmentation_verse_chorus_sections.py`

**Purpose:** Structure segmentation (verse/chorus/sections).

**Algorithm ID:** `analysis_qa_and_automation.structure_segmentation_verse_chorus_sections`
**Theme:** `Analysis, QA, and Automation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/analysis_qa_and_automation/structure_segmentation_verse_chorus_sections.py`, `python3 src/pvx/algorithms/analysis_qa_and_automation/structure_segmentation_verse_chorus_sections.py --help`

### Module Docstring

```text
Structure segmentation (verse/chorus/sections).

Comprehensive module help:
- Theme: Analysis, QA, and Automation
- Algorithm ID: analysis_qa_and_automation.structure_segmentation_verse_chorus_sections
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/audio_dispatch.py`

**Purpose:** Restoration, separation, and loudness dispatch helpers for pvx wrappers.

**Classes:** None
**Functions:** `dispatch_separation`, `dispatch_denoise`, `dispatch_dereverb`, `_lufs_estimate`, `dispatch_dynamics`

### Module Docstring

```text
Restoration, separation, and loudness dispatch helpers for pvx wrappers.
```

## `src/pvx/algorithms/base.py`

**Purpose:** Shared DSP utilities and implementations for pvx algorithm modules.

**Classes:** `AlgorithmResult`
**Functions:** `coerce_audio`, `maybe_librosa`, `maybe_loudnorm`, `build_metadata`, `_resolve_metadata_status`, `_annotate_dispatch_metadata`, `normalize_peak`, `ensure_length`, `resample_length`, `envelope_follower`, `soft_clip`, `_resolve_transform_name`, `_stft_config`, `stft_multi`, `istft_multi`, `spectral_sharpen`, `spectral_blur`, `hpss_split`, `time_stretch`, `pitch_shift`, `overlap_add_frames`, `granular_time_stretch`, `spectral_gate`, `spectral_subtract_denoise`, `mmse_like_denoise`, `minimum_statistics_denoise`, `simple_declick`, `simple_declip`, `dereverb_decay_subtract`, `dereverb_wpe_style`, `compressor`, `upward_compressor`, `true_peak_limit`, `transient_shaper`, `spectral_dynamics`, `split_bands`, `multiband_compression`, `cross_synthesis`, `spectral_convolution`, `spectral_freeze`, `phase_randomize`, `formant_warp`, `resonator_bank`, `spectral_contrast_exaggerate`, `rhythmic_gate`, `ring_mod`, `spectral_tremolo`, `envelope_modulation`, `estimate_f0_track`, `nearest_scale_freq`, `variable_pitch_shift`, `detect_key_from_chroma`, `cqt_or_stft`, `icqt_or_istft`, `run_algorithm`

### Module Docstring

```text
Shared DSP utilities and implementations for pvx algorithm modules.
```

## `src/pvx/algorithms/creative_spectral_effects/__init__.py`

**Purpose:** Creative Spectral Effects algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Creative Spectral Effects algorithm scaffolds.
```

## `src/pvx/algorithms/creative_spectral_effects/cross_synthesis_vocoder.py`

**Purpose:** Cross-synthesis vocoder.

**Algorithm ID:** `creative_spectral_effects.cross_synthesis_vocoder`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/cross_synthesis_vocoder.py`, `python3 src/pvx/algorithms/creative_spectral_effects/cross_synthesis_vocoder.py --help`

### Module Docstring

```text
Cross-synthesis vocoder.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.cross_synthesis_vocoder
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/creative_spectral_effects/formant_painting_warping.py`

**Purpose:** Formant painting/warping.

**Algorithm ID:** `creative_spectral_effects.formant_painting_warping`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/formant_painting_warping.py`, `python3 src/pvx/algorithms/creative_spectral_effects/formant_painting_warping.py --help`

### Module Docstring

```text
Formant painting/warping.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.formant_painting_warping
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/creative_spectral_effects/phase_randomization_textures.py`

**Purpose:** Phase randomization textures.

**Algorithm ID:** `creative_spectral_effects.phase_randomization_textures`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/phase_randomization_textures.py`, `python3 src/pvx/algorithms/creative_spectral_effects/phase_randomization_textures.py --help`

### Module Docstring

```text
Phase randomization textures.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.phase_randomization_textures
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/creative_spectral_effects/resonator_filterbank_morphing.py`

**Purpose:** Resonator/filterbank morphing.

**Algorithm ID:** `creative_spectral_effects.resonator_filterbank_morphing`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/resonator_filterbank_morphing.py`, `python3 src/pvx/algorithms/creative_spectral_effects/resonator_filterbank_morphing.py --help`

### Module Docstring

```text
Resonator/filterbank morphing.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.resonator_filterbank_morphing
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/creative_spectral_effects/spectral_blur_smear.py`

**Purpose:** Spectral blur/smear.

**Algorithm ID:** `creative_spectral_effects.spectral_blur_smear`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/spectral_blur_smear.py`, `python3 src/pvx/algorithms/creative_spectral_effects/spectral_blur_smear.py --help`

### Module Docstring

```text
Spectral blur/smear.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.spectral_blur_smear
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/creative_spectral_effects/spectral_contrast_exaggeration.py`

**Purpose:** Spectral contrast exaggeration.

**Algorithm ID:** `creative_spectral_effects.spectral_contrast_exaggeration`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/spectral_contrast_exaggeration.py`, `python3 src/pvx/algorithms/creative_spectral_effects/spectral_contrast_exaggeration.py --help`

### Module Docstring

```text
Spectral contrast exaggeration.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.spectral_contrast_exaggeration
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/creative_spectral_effects/spectral_convolution_effects.py`

**Purpose:** Spectral convolution effects.

**Algorithm ID:** `creative_spectral_effects.spectral_convolution_effects`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/spectral_convolution_effects.py`, `python3 src/pvx/algorithms/creative_spectral_effects/spectral_convolution_effects.py --help`

### Module Docstring

```text
Spectral convolution effects.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.spectral_convolution_effects
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/creative_spectral_effects/spectral_freeze_banks.py`

**Purpose:** Spectral freeze banks.

**Algorithm ID:** `creative_spectral_effects.spectral_freeze_banks`
**Theme:** `Creative Spectral Effects`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/creative_spectral_effects/spectral_freeze_banks.py`, `python3 src/pvx/algorithms/creative_spectral_effects/spectral_freeze_banks.py --help`

### Module Docstring

```text
Spectral freeze banks.

Comprehensive module help:
- Theme: Creative Spectral Effects
- Algorithm ID: creative_spectral_effects.spectral_freeze_banks
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/__init__.py`

**Purpose:** Denoise and Restoration algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Denoise and Restoration algorithm scaffolds.
```

## `src/pvx/algorithms/denoise_and_restoration/declick_decrackle_median_wavelet_interpolation.py`

**Purpose:** Declick/decrackle (median/wavelet + interpolation).

**Algorithm ID:** `denoise_and_restoration.declick_decrackle_median_wavelet_interpolation`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/declick_decrackle_median_wavelet_interpolation.py`, `python3 src/pvx/algorithms/denoise_and_restoration/declick_decrackle_median_wavelet_interpolation.py --help`

### Module Docstring

```text
Declick/decrackle (median/wavelet + interpolation).

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.declick_decrackle_median_wavelet_interpolation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/declip_via_sparse_reconstruction.py`

**Purpose:** Declip via sparse reconstruction.

**Algorithm ID:** `denoise_and_restoration.declip_via_sparse_reconstruction`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/declip_via_sparse_reconstruction.py`, `python3 src/pvx/algorithms/denoise_and_restoration/declip_via_sparse_reconstruction.py --help`

### Module Docstring

```text
Declip via sparse reconstruction.

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.declip_via_sparse_reconstruction
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/diffusion_based_speech_audio_denoise.py`

**Purpose:** Diffusion-based speech/audio denoise.

**Algorithm ID:** `denoise_and_restoration.diffusion_based_speech_audio_denoise`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/diffusion_based_speech_audio_denoise.py`, `python3 src/pvx/algorithms/denoise_and_restoration/diffusion_based_speech_audio_denoise.py --help`

### Module Docstring

```text
Diffusion-based speech/audio denoise.

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.diffusion_based_speech_audio_denoise
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/log_mmse.py`

**Purpose:** Log-MMSE.

**Algorithm ID:** `denoise_and_restoration.log_mmse`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/log_mmse.py`, `python3 src/pvx/algorithms/denoise_and_restoration/log_mmse.py --help`

### Module Docstring

```text
Log-MMSE.

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.log_mmse
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/minimum_statistics_noise_tracking.py`

**Purpose:** Minimum-statistics noise tracking.

**Algorithm ID:** `denoise_and_restoration.minimum_statistics_noise_tracking`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/minimum_statistics_noise_tracking.py`, `python3 src/pvx/algorithms/denoise_and_restoration/minimum_statistics_noise_tracking.py --help`

### Module Docstring

```text
Minimum-statistics noise tracking.

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.minimum_statistics_noise_tracking
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/mmse_stsa.py`

**Purpose:** MMSE-STSA.

**Algorithm ID:** `denoise_and_restoration.mmse_stsa`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/mmse_stsa.py`, `python3 src/pvx/algorithms/denoise_and_restoration/mmse_stsa.py --help`

### Module Docstring

```text
MMSE-STSA.

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.mmse_stsa
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/rnnoise_style_denoiser.py`

**Purpose:** RNNoise-style denoiser.

**Algorithm ID:** `denoise_and_restoration.rnnoise_style_denoiser`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/rnnoise_style_denoiser.py`, `python3 src/pvx/algorithms/denoise_and_restoration/rnnoise_style_denoiser.py --help`

### Module Docstring

```text
RNNoise-style denoiser.

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.rnnoise_style_denoiser
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/denoise_and_restoration/wiener_denoising.py`

**Purpose:** Wiener denoising.

**Algorithm ID:** `denoise_and_restoration.wiener_denoising`
**Theme:** `Denoise and Restoration`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/denoise_and_restoration/wiener_denoising.py`, `python3 src/pvx/algorithms/denoise_and_restoration/wiener_denoising.py --help`

### Module Docstring

```text
Wiener denoising.

Comprehensive module help:
- Theme: Denoise and Restoration
- Algorithm ID: denoise_and_restoration.wiener_denoising
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/__init__.py`

**Purpose:** Dereverb and Room Correction algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Dereverb and Room Correction algorithm scaffolds.
```

## `src/pvx/algorithms/dereverb_and_room_correction/blind_deconvolution_dereverb.py`

**Purpose:** Blind deconvolution dereverb.

**Algorithm ID:** `dereverb_and_room_correction.blind_deconvolution_dereverb`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/blind_deconvolution_dereverb.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/blind_deconvolution_dereverb.py --help`

### Module Docstring

```text
Blind deconvolution dereverb.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.blind_deconvolution_dereverb
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/drr_guided_dereverb.py`

**Purpose:** DRR-guided dereverb.

**Algorithm ID:** `dereverb_and_room_correction.drr_guided_dereverb`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/drr_guided_dereverb.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/drr_guided_dereverb.py --help`

### Module Docstring

```text
DRR-guided dereverb.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.drr_guided_dereverb
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/late_reverb_suppression_via_coherence.py`

**Purpose:** Late reverb suppression via coherence.

**Algorithm ID:** `dereverb_and_room_correction.late_reverb_suppression_via_coherence`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/late_reverb_suppression_via_coherence.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/late_reverb_suppression_via_coherence.py --help`

### Module Docstring

```text
Late reverb suppression via coherence.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.late_reverb_suppression_via_coherence
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/multi_band_adaptive_deverb.py`

**Purpose:** Multi-band adaptive deverb.

**Algorithm ID:** `dereverb_and_room_correction.multi_band_adaptive_deverb`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/multi_band_adaptive_deverb.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/multi_band_adaptive_deverb.py --help`

### Module Docstring

```text
Multi-band adaptive deverb.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.multi_band_adaptive_deverb
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/neural_dereverb_module.py`

**Purpose:** Neural dereverb module.

**Algorithm ID:** `dereverb_and_room_correction.neural_dereverb_module`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/neural_dereverb_module.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/neural_dereverb_module.py --help`

### Module Docstring

```text
Neural dereverb module.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.neural_dereverb_module
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/room_impulse_inverse_filtering.py`

**Purpose:** Room impulse inverse filtering.

**Algorithm ID:** `dereverb_and_room_correction.room_impulse_inverse_filtering`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/room_impulse_inverse_filtering.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/room_impulse_inverse_filtering.py --help`

### Module Docstring

```text
Room impulse inverse filtering.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.room_impulse_inverse_filtering
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/spectral_decay_subtraction.py`

**Purpose:** Spectral decay subtraction.

**Algorithm ID:** `dereverb_and_room_correction.spectral_decay_subtraction`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/spectral_decay_subtraction.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/spectral_decay_subtraction.py --help`

### Module Docstring

```text
Spectral decay subtraction.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.spectral_decay_subtraction
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dereverb_and_room_correction/wpe_dereverberation.py`

**Purpose:** WPE dereverberation.

**Algorithm ID:** `dereverb_and_room_correction.wpe_dereverberation`
**Theme:** `Dereverb and Room Correction`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dereverb_and_room_correction/wpe_dereverberation.py`, `python3 src/pvx/algorithms/dereverb_and_room_correction/wpe_dereverberation.py --help`

### Module Docstring

```text
WPE dereverberation.

Comprehensive module help:
- Theme: Dereverb and Room Correction
- Algorithm ID: dereverb_and_room_correction.wpe_dereverberation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/__init__.py`

**Purpose:** Dynamics and Loudness algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Dynamics and Loudness algorithm scaffolds.
```

## `src/pvx/algorithms/dynamics_and_loudness/ebu_r128_normalization.py`

**Purpose:** EBU R128 normalization.

**Algorithm ID:** `dynamics_and_loudness.ebu_r128_normalization`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/ebu_r128_normalization.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/ebu_r128_normalization.py --help`

### Module Docstring

```text
EBU R128 normalization.

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.ebu_r128_normalization
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/itu_bs_1770_loudness_measurement_gating.py`

**Purpose:** ITU BS.1770 loudness measurement/gating.

**Algorithm ID:** `dynamics_and_loudness.itu_bs_1770_loudness_measurement_gating`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/itu_bs_1770_loudness_measurement_gating.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/itu_bs_1770_loudness_measurement_gating.py --help`

### Module Docstring

```text
ITU BS.1770 loudness measurement/gating.

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.itu_bs_1770_loudness_measurement_gating
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/lufs_target_mastering_chain.py`

**Purpose:** LUFS-target mastering chain.

**Algorithm ID:** `dynamics_and_loudness.lufs_target_mastering_chain`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/lufs_target_mastering_chain.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/lufs_target_mastering_chain.py --help`

### Module Docstring

```text
LUFS-target mastering chain.

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.lufs_target_mastering_chain
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/multi_band_compression.py`

**Purpose:** Multi-band compression.

**Algorithm ID:** `dynamics_and_loudness.multi_band_compression`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/multi_band_compression.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/multi_band_compression.py --help`

### Module Docstring

```text
Multi-band compression.

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.multi_band_compression
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/spectral_dynamics_bin_wise_compressor_expander.py`

**Purpose:** Spectral dynamics (bin-wise compressor/expander).

**Algorithm ID:** `dynamics_and_loudness.spectral_dynamics_bin_wise_compressor_expander`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/spectral_dynamics_bin_wise_compressor_expander.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/spectral_dynamics_bin_wise_compressor_expander.py --help`

### Module Docstring

```text
Spectral dynamics (bin-wise compressor/expander).

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.spectral_dynamics_bin_wise_compressor_expander
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/transient_shaping.py`

**Purpose:** Transient shaping.

**Algorithm ID:** `dynamics_and_loudness.transient_shaping`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/transient_shaping.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/transient_shaping.py --help`

### Module Docstring

```text
Transient shaping.

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.transient_shaping
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/true_peak_limiting.py`

**Purpose:** True-peak limiting.

**Algorithm ID:** `dynamics_and_loudness.true_peak_limiting`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/true_peak_limiting.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/true_peak_limiting.py --help`

### Module Docstring

```text
True-peak limiting.

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.true_peak_limiting
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/dynamics_and_loudness/upward_compression.py`

**Purpose:** Upward compression.

**Algorithm ID:** `dynamics_and_loudness.upward_compression`
**Theme:** `Dynamics and Loudness`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/dynamics_and_loudness/upward_compression.py`, `python3 src/pvx/algorithms/dynamics_and_loudness/upward_compression.py --help`

### Module Docstring

```text
Upward compression.

Comprehensive module help:
- Theme: Dynamics and Loudness
- Algorithm ID: dynamics_and_loudness.upward_compression
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/effects_dispatch.py`

**Purpose:** Creative, granular, and analysis dispatch helpers for pvx wrappers.

**Classes:** None
**Functions:** `dispatch_creative`, `dispatch_granular`, `dispatch_analysis`

### Module Docstring

```text
Creative, granular, and analysis dispatch helpers for pvx wrappers.
```

## `src/pvx/algorithms/granular_and_modulation/__init__.py`

**Purpose:** Granular and Modulation algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Granular and Modulation algorithm scaffolds.
```

## `src/pvx/algorithms/granular_and_modulation/am_fm_ring_modulation_blocks.py`

**Purpose:** AM/FM/ring modulation blocks.

**Algorithm ID:** `granular_and_modulation.am_fm_ring_modulation_blocks`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/am_fm_ring_modulation_blocks.py`, `python3 src/pvx/algorithms/granular_and_modulation/am_fm_ring_modulation_blocks.py --help`

### Module Docstring

```text
AM/FM/ring modulation blocks.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.am_fm_ring_modulation_blocks
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/granular_and_modulation/envelope_followed_modulation_routing.py`

**Purpose:** Envelope-followed modulation routing.

**Algorithm ID:** `granular_and_modulation.envelope_followed_modulation_routing`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/envelope_followed_modulation_routing.py`, `python3 src/pvx/algorithms/granular_and_modulation/envelope_followed_modulation_routing.py --help`

### Module Docstring

```text
Envelope-followed modulation routing.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.envelope_followed_modulation_routing
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/granular_and_modulation/formant_lfo_modulation.py`

**Purpose:** Formant LFO modulation.

**Algorithm ID:** `granular_and_modulation.formant_lfo_modulation`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/formant_lfo_modulation.py`, `python3 src/pvx/algorithms/granular_and_modulation/formant_lfo_modulation.py --help`

### Module Docstring

```text
Formant LFO modulation.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.formant_lfo_modulation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/granular_and_modulation/freeze_grain_morphing.py`

**Purpose:** Freeze-grain morphing.

**Algorithm ID:** `granular_and_modulation.freeze_grain_morphing`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/freeze_grain_morphing.py`, `python3 src/pvx/algorithms/granular_and_modulation/freeze_grain_morphing.py --help`

### Module Docstring

```text
Freeze-grain morphing.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.freeze_grain_morphing
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/granular_and_modulation/grain_cloud_pitch_textures.py`

**Purpose:** Grain-cloud pitch textures.

**Algorithm ID:** `granular_and_modulation.grain_cloud_pitch_textures`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/grain_cloud_pitch_textures.py`, `python3 src/pvx/algorithms/granular_and_modulation/grain_cloud_pitch_textures.py --help`

### Module Docstring

```text
Grain-cloud pitch textures.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.grain_cloud_pitch_textures
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/granular_and_modulation/granular_time_stretch_engine.py`

**Purpose:** Granular time-stretch engine.

**Algorithm ID:** `granular_and_modulation.granular_time_stretch_engine`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/granular_time_stretch_engine.py`, `python3 src/pvx/algorithms/granular_and_modulation/granular_time_stretch_engine.py --help`

### Module Docstring

```text
Granular time-stretch engine.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.granular_time_stretch_engine
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/granular_and_modulation/rhythmic_gate_stutter_quantizer.py`

**Purpose:** Rhythmic gate/stutter quantizer.

**Algorithm ID:** `granular_and_modulation.rhythmic_gate_stutter_quantizer`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/rhythmic_gate_stutter_quantizer.py`, `python3 src/pvx/algorithms/granular_and_modulation/rhythmic_gate_stutter_quantizer.py --help`

### Module Docstring

```text
Rhythmic gate/stutter quantizer.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.rhythmic_gate_stutter_quantizer
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/granular_and_modulation/spectral_tremolo.py`

**Purpose:** Spectral tremolo.

**Algorithm ID:** `granular_and_modulation.spectral_tremolo`
**Theme:** `Granular and Modulation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/granular_and_modulation/spectral_tremolo.py`, `python3 src/pvx/algorithms/granular_and_modulation/spectral_tremolo.py --help`

### Module Docstring

```text
Spectral tremolo.

Comprehensive module help:
- Theme: Granular and Modulation
- Algorithm ID: granular_and_modulation.spectral_tremolo
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/__init__.py`

**Purpose:** Pitch Detection and Tracking algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Pitch Detection and Tracking algorithm scaffolds.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/crepe_style_neural_f0.py`

**Purpose:** CREPE-style neural F0.

**Algorithm ID:** `pitch_detection_and_tracking.crepe_style_neural_f0`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/crepe_style_neural_f0.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/crepe_style_neural_f0.py --help`

### Module Docstring

```text
CREPE-style neural F0.

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.crepe_style_neural_f0
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/harmonic_product_spectrum_hps.py`

**Purpose:** Harmonic Product Spectrum (HPS).

**Algorithm ID:** `pitch_detection_and_tracking.harmonic_product_spectrum_hps`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/harmonic_product_spectrum_hps.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/harmonic_product_spectrum_hps.py --help`

### Module Docstring

```text
Harmonic Product Spectrum (HPS).

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.harmonic_product_spectrum_hps
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/pyin.py`

**Purpose:** pYIN.

**Algorithm ID:** `pitch_detection_and_tracking.pyin`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/pyin.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/pyin.py --help`

### Module Docstring

```text
pYIN.

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.pyin
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/rapt.py`

**Purpose:** RAPT.

**Algorithm ID:** `pitch_detection_and_tracking.rapt`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/rapt.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/rapt.py --help`

### Module Docstring

```text
RAPT.

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.rapt
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/subharmonic_summation.py`

**Purpose:** Subharmonic summation.

**Algorithm ID:** `pitch_detection_and_tracking.subharmonic_summation`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/subharmonic_summation.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/subharmonic_summation.py --help`

### Module Docstring

```text
Subharmonic summation.

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.subharmonic_summation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/swipe.py`

**Purpose:** SWIPE.

**Algorithm ID:** `pitch_detection_and_tracking.swipe`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/swipe.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/swipe.py --help`

### Module Docstring

```text
SWIPE.

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.swipe
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/viterbi_smoothed_pitch_contour_tracking.py`

**Purpose:** Viterbi-smoothed pitch contour tracking.

**Algorithm ID:** `pitch_detection_and_tracking.viterbi_smoothed_pitch_contour_tracking`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/viterbi_smoothed_pitch_contour_tracking.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/viterbi_smoothed_pitch_contour_tracking.py --help`

### Module Docstring

```text
Viterbi-smoothed pitch contour tracking.

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.viterbi_smoothed_pitch_contour_tracking
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_detection_and_tracking/yin.py`

**Purpose:** YIN.

**Algorithm ID:** `pitch_detection_and_tracking.yin`
**Theme:** `Pitch Detection and Tracking`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/pitch_detection_and_tracking/yin.py`, `python3 src/pvx/algorithms/pitch_detection_and_tracking/yin.py --help`

### Module Docstring

```text
YIN.

Comprehensive module help:
- Theme: Pitch Detection and Tracking
- Algorithm ID: pitch_detection_and_tracking.yin
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/pitch_dispatch.py`

**Purpose:** Pitch, retune, and transform dispatch helpers for pvx algorithm wrappers.

**Classes:** None
**Functions:** `dispatch_time_scale`, `dispatch_pitch_tracking`, `_scale_cents_from_name`, `dispatch_retune`, `dispatch_transforms`

### Module Docstring

```text
Pitch, retune, and transform dispatch helpers for pvx algorithm wrappers.
```

## `src/pvx/algorithms/registry.py`

**Purpose:** Registry for generated pvx algorithm wrappers backed by shared dispatch code.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Registry for generated pvx algorithm wrappers backed by shared dispatch code.
```

## `src/pvx/algorithms/retune_and_intonation/__init__.py`

**Purpose:** Retune and Intonation algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Retune and Intonation algorithm scaffolds.
```

## `src/pvx/algorithms/retune_and_intonation/adaptive_intonation_context_sensitive_intervals.py`

**Purpose:** Adaptive intonation (context-sensitive intervals).

**Algorithm ID:** `retune_and_intonation.adaptive_intonation_context_sensitive_intervals`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/adaptive_intonation_context_sensitive_intervals.py`, `python3 src/pvx/algorithms/retune_and_intonation/adaptive_intonation_context_sensitive_intervals.py --help`

### Module Docstring

```text
Adaptive intonation (context-sensitive intervals).

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.adaptive_intonation_context_sensitive_intervals
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/retune_and_intonation/chord_aware_retuning.py`

**Purpose:** Chord-aware retuning.

**Algorithm ID:** `retune_and_intonation.chord_aware_retuning`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/chord_aware_retuning.py`, `python3 src/pvx/algorithms/retune_and_intonation/chord_aware_retuning.py --help`

### Module Docstring

```text
Chord-aware retuning.

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.chord_aware_retuning
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/retune_and_intonation/just_intonation_mapping_per_key_center.py`

**Purpose:** Just intonation mapping per key center.

**Algorithm ID:** `retune_and_intonation.just_intonation_mapping_per_key_center`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/just_intonation_mapping_per_key_center.py`, `python3 src/pvx/algorithms/retune_and_intonation/just_intonation_mapping_per_key_center.py --help`

### Module Docstring

```text
Just intonation mapping per key center.

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.just_intonation_mapping_per_key_center
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/retune_and_intonation/key_aware_retuning_with_confidence_weighting.py`

**Purpose:** Key-aware retuning with confidence weighting.

**Algorithm ID:** `retune_and_intonation.key_aware_retuning_with_confidence_weighting`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/key_aware_retuning_with_confidence_weighting.py`, `python3 src/pvx/algorithms/retune_and_intonation/key_aware_retuning_with_confidence_weighting.py --help`

### Module Docstring

```text
Key-aware retuning with confidence weighting.

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.key_aware_retuning_with_confidence_weighting
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/retune_and_intonation/portamento_aware_retune_curves.py`

**Purpose:** Portamento-aware retune curves.

**Algorithm ID:** `retune_and_intonation.portamento_aware_retune_curves`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/portamento_aware_retune_curves.py`, `python3 src/pvx/algorithms/retune_and_intonation/portamento_aware_retune_curves.py --help`

### Module Docstring

```text
Portamento-aware retune curves.

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.portamento_aware_retune_curves
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/retune_and_intonation/scala_mts_scale_import_and_quantization.py`

**Purpose:** Scala/MTS scale import and quantization.

**Algorithm ID:** `retune_and_intonation.scala_mts_scale_import_and_quantization`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/scala_mts_scale_import_and_quantization.py`, `python3 src/pvx/algorithms/retune_and_intonation/scala_mts_scale_import_and_quantization.py --help`

### Module Docstring

```text
Scala/MTS scale import and quantization.

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.scala_mts_scale_import_and_quantization
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/retune_and_intonation/time_varying_cents_maps.py`

**Purpose:** Time-varying cents maps.

**Algorithm ID:** `retune_and_intonation.time_varying_cents_maps`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/time_varying_cents_maps.py`, `python3 src/pvx/algorithms/retune_and_intonation/time_varying_cents_maps.py --help`

### Module Docstring

```text
Time-varying cents maps.

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.time_varying_cents_maps
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/retune_and_intonation/vibrato_preserving_correction.py`

**Purpose:** Vibrato-preserving correction.

**Algorithm ID:** `retune_and_intonation.vibrato_preserving_correction`
**Theme:** `Retune and Intonation`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/retune_and_intonation/vibrato_preserving_correction.py`, `python3 src/pvx/algorithms/retune_and_intonation/vibrato_preserving_correction.py --help`

### Module Docstring

```text
Vibrato-preserving correction.

Comprehensive module help:
- Theme: Retune and Intonation
- Algorithm ID: retune_and_intonation.vibrato_preserving_correction
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/__init__.py`

**Purpose:** Separation and Decomposition algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Separation and Decomposition algorithm scaffolds.
```

## `src/pvx/algorithms/separation_and_decomposition/demucs_style_stem_separation_backend.py`

**Purpose:** Demucs-style stem separation backend.

**Algorithm ID:** `separation_and_decomposition.demucs_style_stem_separation_backend`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/demucs_style_stem_separation_backend.py`, `python3 src/pvx/algorithms/separation_and_decomposition/demucs_style_stem_separation_backend.py --help`

### Module Docstring

```text
Demucs-style stem separation backend.

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.demucs_style_stem_separation_backend
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/ica_bss_for_multichannel_stems.py`

**Purpose:** ICA/BSS for multichannel stems.

**Algorithm ID:** `separation_and_decomposition.ica_bss_for_multichannel_stems`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/ica_bss_for_multichannel_stems.py`, `python3 src/pvx/algorithms/separation_and_decomposition/ica_bss_for_multichannel_stems.py --help`

### Module Docstring

```text
ICA/BSS for multichannel stems.

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.ica_bss_for_multichannel_stems
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/nmf_decomposition.py`

**Purpose:** NMF decomposition.

**Algorithm ID:** `separation_and_decomposition.nmf_decomposition`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/nmf_decomposition.py`, `python3 src/pvx/algorithms/separation_and_decomposition/nmf_decomposition.py --help`

### Module Docstring

```text
NMF decomposition.

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.nmf_decomposition
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/probabilistic_latent_component_separation.py`

**Purpose:** Probabilistic latent component separation.

**Algorithm ID:** `separation_and_decomposition.probabilistic_latent_component_separation`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/probabilistic_latent_component_separation.py`, `python3 src/pvx/algorithms/separation_and_decomposition/probabilistic_latent_component_separation.py --help`

### Module Docstring

```text
Probabilistic latent component separation.

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.probabilistic_latent_component_separation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/rpca_hpss.py`

**Purpose:** RPCA HPSS.

**Algorithm ID:** `separation_and_decomposition.rpca_hpss`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/rpca_hpss.py`, `python3 src/pvx/algorithms/separation_and_decomposition/rpca_hpss.py --help`

### Module Docstring

```text
RPCA HPSS.

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.rpca_hpss
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/sinusoidal_residual_transient_decomposition.py`

**Purpose:** Sinusoidal+residual+transient decomposition.

**Algorithm ID:** `separation_and_decomposition.sinusoidal_residual_transient_decomposition`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/sinusoidal_residual_transient_decomposition.py`, `python3 src/pvx/algorithms/separation_and_decomposition/sinusoidal_residual_transient_decomposition.py --help`

### Module Docstring

```text
Sinusoidal+residual+transient decomposition.

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.sinusoidal_residual_transient_decomposition
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/tensor_decomposition_cp_tucker.py`

**Purpose:** Tensor decomposition (CP/Tucker).

**Algorithm ID:** `separation_and_decomposition.tensor_decomposition_cp_tucker`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/tensor_decomposition_cp_tucker.py`, `python3 src/pvx/algorithms/separation_and_decomposition/tensor_decomposition_cp_tucker.py --help`

### Module Docstring

```text
Tensor decomposition (CP/Tucker).

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.tensor_decomposition_cp_tucker
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/separation_and_decomposition/u_net_vocal_accompaniment_split.py`

**Purpose:** U-Net vocal/accompaniment split.

**Algorithm ID:** `separation_and_decomposition.u_net_vocal_accompaniment_split`
**Theme:** `Separation and Decomposition`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/separation_and_decomposition/u_net_vocal_accompaniment_split.py`, `python3 src/pvx/algorithms/separation_and_decomposition/u_net_vocal_accompaniment_split.py --help`

### Module Docstring

```text
U-Net vocal/accompaniment split.

Comprehensive module help:
- Theme: Separation and Decomposition
- Algorithm ID: separation_and_decomposition.u_net_vocal_accompaniment_split
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/__init__.py`

**Purpose:** Spatial and multichannel algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Spatial and multichannel algorithm scaffolds.
```

## `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/__init__.py`

**Purpose:** Spatial and multichannel: creative spatial fx.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Spatial and multichannel: creative spatial fx.
```

## `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/binaural_motion_trajectory_designer.py`

**Purpose:** Binaural motion trajectory designer.

**Algorithm ID:** `spatial_and_multichannel.binaural_motion_trajectory_designer`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/binaural_motion_trajectory_designer.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/binaural_motion_trajectory_designer.py --help`

### Module Docstring

```text
Binaural motion trajectory designer.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.binaural_motion_trajectory_designer
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/decorrelated_reverb_upmix.py`

**Purpose:** Decorrelated reverb upmix.

**Algorithm ID:** `spatial_and_multichannel.decorrelated_reverb_upmix`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/decorrelated_reverb_upmix.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/decorrelated_reverb_upmix.py --help`

### Module Docstring

```text
Decorrelated reverb upmix.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.decorrelated_reverb_upmix
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/rotating_speaker_doppler_field.py`

**Purpose:** Rotating-speaker Doppler field.

**Algorithm ID:** `spatial_and_multichannel.rotating_speaker_doppler_field`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/rotating_speaker_doppler_field.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/rotating_speaker_doppler_field.py --help`

### Module Docstring

```text
Rotating-speaker Doppler field.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.rotating_speaker_doppler_field
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spatial_freeze_resynthesis.py`

**Purpose:** Spatial freeze resynthesis.

**Algorithm ID:** `spatial_and_multichannel.spatial_freeze_resynthesis`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spatial_freeze_resynthesis.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spatial_freeze_resynthesis.py --help`

### Module Docstring

```text
Spatial freeze resynthesis.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.spatial_freeze_resynthesis
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spectral_spatial_granulator.py`

**Purpose:** Spectral spatial granulator.

**Algorithm ID:** `spatial_and_multichannel.spectral_spatial_granulator`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spectral_spatial_granulator.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/spectral_spatial_granulator.py --help`

### Module Docstring

```text
Spectral spatial granulator.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.spectral_spatial_granulator
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/stochastic_spatial_diffusion_cloud.py`

**Purpose:** Stochastic spatial diffusion cloud.

**Algorithm ID:** `spatial_and_multichannel.stochastic_spatial_diffusion_cloud`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/stochastic_spatial_diffusion_cloud.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/creative_spatial_fx/stochastic_spatial_diffusion_cloud.py --help`

### Module Docstring

```text
Stochastic spatial diffusion cloud.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.stochastic_spatial_diffusion_cloud
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/__init__.py`

**Purpose:** Spatial and multichannel: imaging and panning.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Spatial and multichannel: imaging and panning.
```

## `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/binaural_itd_ild_synthesis.py`

**Purpose:** Binaural ITD/ILD synthesis.

**Algorithm ID:** `spatial_and_multichannel.binaural_itd_ild_synthesis`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/binaural_itd_ild_synthesis.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/binaural_itd_ild_synthesis.py --help`

### Module Docstring

```text
Binaural ITD/ILD synthesis.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.binaural_itd_ild_synthesis
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/dbap_distance_based_amplitude_panning.py`

**Purpose:** DBAP (distance-based amplitude panning).

**Algorithm ID:** `spatial_and_multichannel.dbap_distance_based_amplitude_panning`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/dbap_distance_based_amplitude_panning.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/dbap_distance_based_amplitude_panning.py --help`

### Module Docstring

```text
DBAP (distance-based amplitude panning).

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.dbap_distance_based_amplitude_panning
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/phase_aligned_mid_side_field_rotation.py`

**Purpose:** Phase-aligned mid/side field rotation.

**Algorithm ID:** `spatial_and_multichannel.phase_aligned_mid_side_field_rotation`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/phase_aligned_mid_side_field_rotation.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/phase_aligned_mid_side_field_rotation.py --help`

### Module Docstring

```text
Phase-aligned mid/side field rotation.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.phase_aligned_mid_side_field_rotation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/stereo_width_frequency_dependent_control.py`

**Purpose:** Stereo width (frequency-dependent control).

**Algorithm ID:** `spatial_and_multichannel.stereo_width_frequency_dependent_control`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/stereo_width_frequency_dependent_control.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/stereo_width_frequency_dependent_control.py --help`

### Module Docstring

```text
Stereo width (frequency-dependent control).

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.stereo_width_frequency_dependent_control
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/transaural_crosstalk_cancellation.py`

**Purpose:** Transaural crosstalk cancellation.

**Algorithm ID:** `spatial_and_multichannel.transaural_crosstalk_cancellation`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/transaural_crosstalk_cancellation.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/transaural_crosstalk_cancellation.py --help`

### Module Docstring

```text
Transaural crosstalk cancellation.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.transaural_crosstalk_cancellation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/vbap_adaptive_panning.py`

**Purpose:** VBAP adaptive panning.

**Algorithm ID:** `spatial_and_multichannel.vbap_adaptive_panning`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/vbap_adaptive_panning.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/imaging_and_panning/vbap_adaptive_panning.py --help`

### Module Docstring

```text
VBAP adaptive panning.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.vbap_adaptive_panning
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/__init__.py`

**Purpose:** Spatial and multichannel: multichannel restoration.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Spatial and multichannel: multichannel restoration.
```

## `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/coherence_based_dereverb_multichannel.py`

**Purpose:** Coherence-based dereverb (multichannel).

**Algorithm ID:** `spatial_and_multichannel.coherence_based_dereverb_multichannel`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/coherence_based_dereverb_multichannel.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/coherence_based_dereverb_multichannel.py --help`

### Module Docstring

```text
Coherence-based dereverb (multichannel).

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.coherence_based_dereverb_multichannel
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/cross_channel_click_pop_repair.py`

**Purpose:** Cross-channel click/pop repair.

**Algorithm ID:** `spatial_and_multichannel.cross_channel_click_pop_repair`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/cross_channel_click_pop_repair.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/cross_channel_click_pop_repair.py --help`

### Module Docstring

```text
Cross-channel click/pop repair.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.cross_channel_click_pop_repair
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/microphone_array_calibration_tones.py`

**Purpose:** Microphone-array calibration tones.

**Algorithm ID:** `spatial_and_multichannel.microphone_array_calibration_tones`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/microphone_array_calibration_tones.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/microphone_array_calibration_tones.py --help`

### Module Docstring

```text
Microphone-array calibration tones.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.microphone_array_calibration_tones
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_noise_psd_tracking.py`

**Purpose:** Multichannel noise PSD tracking.

**Algorithm ID:** `spatial_and_multichannel.multichannel_noise_psd_tracking`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_noise_psd_tracking.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_noise_psd_tracking.py --help`

### Module Docstring

```text
Multichannel noise PSD tracking.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.multichannel_noise_psd_tracking
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_wiener_postfilter.py`

**Purpose:** Multichannel Wiener postfilter.

**Algorithm ID:** `spatial_and_multichannel.multichannel_wiener_postfilter`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_wiener_postfilter.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/multichannel_wiener_postfilter.py --help`

### Module Docstring

```text
Multichannel Wiener postfilter.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.multichannel_wiener_postfilter
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/phase_consistent_multichannel_denoise.py`

**Purpose:** Phase-consistent multichannel denoise.

**Algorithm ID:** `spatial_and_multichannel.phase_consistent_multichannel_denoise`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/phase_consistent_multichannel_denoise.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/multichannel_restoration/phase_consistent_multichannel_denoise.py --help`

### Module Docstring

```text
Phase-consistent multichannel denoise.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.phase_consistent_multichannel_denoise
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/__init__.py`

**Purpose:** Spatial and multichannel: phase vocoder spatial.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Spatial and multichannel: phase vocoder spatial.
```

## `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_directional_spectral_warp.py`

**Purpose:** pvx directional spectral warp.

**Algorithm ID:** `spatial_and_multichannel.pvx_directional_spectral_warp`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_directional_spectral_warp.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_directional_spectral_warp.py --help`

### Module Docstring

```text
pvx directional spectral warp.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.pvx_directional_spectral_warp
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interaural_coherence_shaping.py`

**Purpose:** pvx interaural coherence shaping.

**Algorithm ID:** `spatial_and_multichannel.pvx_interaural_coherence_shaping`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interaural_coherence_shaping.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interaural_coherence_shaping.py --help`

### Module Docstring

```text
pvx interaural coherence shaping.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.pvx_interaural_coherence_shaping
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interchannel_phase_locking.py`

**Purpose:** pvx interchannel phase locking.

**Algorithm ID:** `spatial_and_multichannel.pvx_interchannel_phase_locking`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interchannel_phase_locking.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_interchannel_phase_locking.py --help`

### Module Docstring

```text
pvx interchannel phase locking.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.pvx_interchannel_phase_locking
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_multichannel_time_alignment.py`

**Purpose:** pvx multichannel time alignment.

**Algorithm ID:** `spatial_and_multichannel.pvx_multichannel_time_alignment`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_multichannel_time_alignment.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_multichannel_time_alignment.py --help`

### Module Docstring

```text
pvx multichannel time alignment.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.pvx_multichannel_time_alignment
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_freeze_and_trajectory.py`

**Purpose:** pvx spatial freeze and trajectory.

**Algorithm ID:** `spatial_and_multichannel.pvx_spatial_freeze_and_trajectory`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_freeze_and_trajectory.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_freeze_and_trajectory.py --help`

### Module Docstring

```text
pvx spatial freeze and trajectory.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.pvx_spatial_freeze_and_trajectory
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_transient_preservation.py`

**Purpose:** pvx spatial transient preservation.

**Algorithm ID:** `spatial_and_multichannel.pvx_spatial_transient_preservation`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_transient_preservation.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/phase_vocoder_spatial/pvx_spatial_transient_preservation.py --help`

### Module Docstring

```text
pvx spatial transient preservation.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.pvx_spatial_transient_preservation
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_dispatch.py`

**Purpose:** Spatial algorithm dispatch helpers for pvx algorithm wrappers.

**Classes:** None
**Functions:** `_spatial_to_channels`, `_spatial_fractional_delay`, `_spatial_apply_delays`, `_spatial_circular_gains`, `_spatial_delay_by_xcorr`, `_spatial_estimate_channel_delays`, `_spatial_synthetic_rir`, `dispatch_spatial`

### Module Docstring

```text
Spatial algorithm dispatch helpers for pvx algorithm wrappers.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/__init__.py`

**Purpose:** Spectral and Time-Frequency Transforms algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Spectral and Time-Frequency Transforms algorithm scaffolds.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/chirplet_transform_analysis.py`

**Purpose:** Chirplet transform analysis.

**Algorithm ID:** `spectral_time_frequency_transforms.chirplet_transform_analysis`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/chirplet_transform_analysis.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/chirplet_transform_analysis.py --help`

### Module Docstring

```text
Chirplet transform analysis.

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.chirplet_transform_analysis
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/constant_q_transform_cqt_processing.py`

**Purpose:** Constant-Q Transform (CQT) processing.

**Algorithm ID:** `spectral_time_frequency_transforms.constant_q_transform_cqt_processing`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/constant_q_transform_cqt_processing.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/constant_q_transform_cqt_processing.py --help`

### Module Docstring

```text
Constant-Q Transform (CQT) processing.

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.constant_q_transform_cqt_processing
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/multi_window_stft_fusion.py`

**Purpose:** Multi-window STFT fusion.

**Algorithm ID:** `spectral_time_frequency_transforms.multi_window_stft_fusion`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/multi_window_stft_fusion.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/multi_window_stft_fusion.py --help`

### Module Docstring

```text
Multi-window STFT fusion.

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.multi_window_stft_fusion
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/nsgt_based_processing.py`

**Purpose:** NSGT-based processing.

**Algorithm ID:** `spectral_time_frequency_transforms.nsgt_based_processing`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/nsgt_based_processing.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/nsgt_based_processing.py --help`

### Module Docstring

```text
NSGT-based processing.

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.nsgt_based_processing
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/reassigned_spectrogram_methods.py`

**Purpose:** Reassigned spectrogram methods.

**Algorithm ID:** `spectral_time_frequency_transforms.reassigned_spectrogram_methods`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/reassigned_spectrogram_methods.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/reassigned_spectrogram_methods.py --help`

### Module Docstring

```text
Reassigned spectrogram methods.

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.reassigned_spectrogram_methods
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/synchrosqueezed_stft.py`

**Purpose:** Synchrosqueezed STFT.

**Algorithm ID:** `spectral_time_frequency_transforms.synchrosqueezed_stft`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/synchrosqueezed_stft.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/synchrosqueezed_stft.py --help`

### Module Docstring

```text
Synchrosqueezed STFT.

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.synchrosqueezed_stft
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/variable_q_transform_vqt.py`

**Purpose:** Variable-Q Transform (VQT).

**Algorithm ID:** `spectral_time_frequency_transforms.variable_q_transform_vqt`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/variable_q_transform_vqt.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/variable_q_transform_vqt.py --help`

### Module Docstring

```text
Variable-Q Transform (VQT).

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.variable_q_transform_vqt
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spectral_time_frequency_transforms/wavelet_packet_processing.py`

**Purpose:** Wavelet packet processing.

**Algorithm ID:** `spectral_time_frequency_transforms.wavelet_packet_processing`
**Theme:** `Spectral and Time-Frequency Transforms`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spectral_time_frequency_transforms/wavelet_packet_processing.py`, `python3 src/pvx/algorithms/spectral_time_frequency_transforms/wavelet_packet_processing.py --help`

### Module Docstring

```text
Wavelet packet processing.

Comprehensive module help:
- Theme: Spectral and Time-Frequency Transforms
- Algorithm ID: spectral_time_frequency_transforms.wavelet_packet_processing
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/__init__.py`

**Purpose:** Time-Scale and Pitch Core algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Time-Scale and Pitch Core algorithm scaffolds.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/beat_synchronous_time_warping.py`

**Purpose:** Beat-synchronous time warping.

**Algorithm ID:** `time_scale_and_pitch_core.beat_synchronous_time_warping`
**Theme:** `Time-Scale and Pitch Core`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/time_scale_and_pitch_core/beat_synchronous_time_warping.py`, `python3 src/pvx/algorithms/time_scale_and_pitch_core/beat_synchronous_time_warping.py --help`

### Module Docstring

```text
Beat-synchronous time warping.

Comprehensive module help:
- Theme: Time-Scale and Pitch Core
- Algorithm ID: time_scale_and_pitch_core.beat_synchronous_time_warping
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/harmonic_percussive_split_tsm.py`

**Purpose:** Harmonic/percussive split TSM.

**Algorithm ID:** `time_scale_and_pitch_core.harmonic_percussive_split_tsm`
**Theme:** `Time-Scale and Pitch Core`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/time_scale_and_pitch_core/harmonic_percussive_split_tsm.py`, `python3 src/pvx/algorithms/time_scale_and_pitch_core/harmonic_percussive_split_tsm.py --help`

### Module Docstring

```text
Harmonic/percussive split TSM.

Comprehensive module help:
- Theme: Time-Scale and Pitch Core
- Algorithm ID: time_scale_and_pitch_core.harmonic_percussive_split_tsm
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/lp_psola.py`

**Purpose:** LP-PSOLA.

**Algorithm ID:** `time_scale_and_pitch_core.lp_psola`
**Theme:** `Time-Scale and Pitch Core`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/time_scale_and_pitch_core/lp_psola.py`, `python3 src/pvx/algorithms/time_scale_and_pitch_core/lp_psola.py --help`

### Module Docstring

```text
LP-PSOLA.

Comprehensive module help:
- Theme: Time-Scale and Pitch Core
- Algorithm ID: time_scale_and_pitch_core.lp_psola
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/multi_resolution_phase_vocoder.py`

**Purpose:** Multi-resolution phase vocoder.

**Algorithm ID:** `time_scale_and_pitch_core.multi_resolution_phase_vocoder`
**Theme:** `Time-Scale and Pitch Core`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/time_scale_and_pitch_core/multi_resolution_phase_vocoder.py`, `python3 src/pvx/algorithms/time_scale_and_pitch_core/multi_resolution_phase_vocoder.py --help`

### Module Docstring

```text
Multi-resolution phase vocoder.

Comprehensive module help:
- Theme: Time-Scale and Pitch Core
- Algorithm ID: time_scale_and_pitch_core.multi_resolution_phase_vocoder
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/nonlinear_time_maps.py`

**Purpose:** Nonlinear time maps (curves, anchors, spline timing).

**Algorithm ID:** `time_scale_and_pitch_core.nonlinear_time_maps`
**Theme:** `Time-Scale and Pitch Core`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/time_scale_and_pitch_core/nonlinear_time_maps.py`, `python3 src/pvx/algorithms/time_scale_and_pitch_core/nonlinear_time_maps.py --help`

### Module Docstring

```text
Nonlinear time maps (curves, anchors, spline timing).

Comprehensive module help:
- Theme: Time-Scale and Pitch Core
- Algorithm ID: time_scale_and_pitch_core.nonlinear_time_maps
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/td_psola.py`

**Purpose:** TD-PSOLA.

**Algorithm ID:** `time_scale_and_pitch_core.td_psola`
**Theme:** `Time-Scale and Pitch Core`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/time_scale_and_pitch_core/td_psola.py`, `python3 src/pvx/algorithms/time_scale_and_pitch_core/td_psola.py --help`

### Module Docstring

```text
TD-PSOLA.

Comprehensive module help:
- Theme: Time-Scale and Pitch Core
- Algorithm ID: time_scale_and_pitch_core.td_psola
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py`

**Purpose:** WSOLA (Waveform Similarity Overlap-Add).

**Algorithm ID:** `time_scale_and_pitch_core.wsola_waveform_similarity_overlap_add`
**Theme:** `Time-Scale and Pitch Core`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see [docs/PVX_ALGORITHM_PARAMS.md](PVX_ALGORITHM_PARAMS.md).

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py`, `python3 src/pvx/algorithms/time_scale_and_pitch_core/wsola_waveform_similarity_overlap_add.py --help`

### Module Docstring

```text
WSOLA (Waveform Similarity Overlap-Add).

Comprehensive module help:
- Theme: Time-Scale and Pitch Core
- Algorithm ID: time_scale_and_pitch_core.wsola_waveform_similarity_overlap_add
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/augment/__init__.py`

**Purpose:** pvx.augment — Audio data augmentation Python API.

**Classes:** None
**Functions:** `asr_pipeline`, `music_pipeline`, `speech_enhancement_pipeline`, `contrastive_pipeline`

### Module Docstring

```text
pvx.augment — Audio data augmentation Python API.

.. warning::

   **Alpha release (0.1.0a1).**  This API is under active development.
   Public interfaces may change between minor versions until 1.0.
   Pin your dependency to an exact version (``pvx==0.1.0a1``) if you
   need stability, and please report issues on GitHub.

A composable, NumPy-native augmentation library built on the pvx DSP engine.
All transforms follow a uniform ``(audio, sr, seed=None) -> (audio, sr)``
interface and are fully reproducible given a fixed seed.

Quick start
-----------
>>> import soundfile as sf
>>> import numpy as np
>>> from pvx.augment import Pipeline, AddNoise, RoomSimulator, SpecAugment, GainPerturber
>>>
>>> audio, sr = sf.read("speech.wav", always_2d=False)
>>>
>>> pipeline = Pipeline([
...     GainPerturber(gain_db=(-3, 3), p=0.8),
...     RoomSimulator(rt60_range=(0.1, 0.6), wet_range=(0.2, 0.7), p=0.5),
...     AddNoise(snr_db=(15, 35), noise_type="pink", p=0.6),
...     SpecAugment(freq_mask_param=20, time_mask_param=40, p=0.5),
... ], seed=42)
>>>
>>> audio_aug, sr_out = pipeline(audio, sr)

Framework integrations
----------------------
PyTorch::

    from pvx.integrations.pytorch import PvxAugmentDataset
    dataset = PvxAugmentDataset(file_list, pipeline, sample_rate=16000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

HuggingFace datasets::

    from pvx.integrations.huggingface import make_augment_map_fn
    augment_fn = make_augment_map_fn(pipeline, audio_column="audio")
    ds_aug = ds.map(augment_fn, batched=False)

TensorFlow::

    from pvx.integrations.tensorflow import make_tf_augment_fn
    augment_fn = make_tf_augment_fn(pipeline, sample_rate=16000)
    tf_dataset = tf_dataset.map(augment_fn)
```

## `src/pvx/augment/codec.py`

**Purpose:** Codec and transmission degradation transforms.

**Classes:** `CodecDegradation`, `BitCrusher`, `BandwidthLimiter`
**Functions:** `_lowpass`, `_bandpass`, `_quantize`, `_resample_simulate`

### Module Docstring

```text
Codec and transmission degradation transforms.

Simulates lossy audio compression (mp3-like, telephone, VoIP), bit-depth
reduction (bit-crushing), and low-fidelity recording artifacts — all using
only NumPy and SciPy (no external codec binaries required).
```

## `src/pvx/augment/config.py`

**Purpose:** Declarative pipeline configuration for ``pvx.augment``.

**Classes:** None
**Functions:** `_load_manifest`, `_camel_to_snake`, `_builtin_transform_index`, `_torch_transform_index`, `_resolve_transform_class`, `_build_transform`, `load_pipeline`, `load_torch_pipeline`

### Module Docstring

```text
Declarative pipeline configuration for ``pvx.augment``.

Lets users describe an augmentation pipeline in YAML or JSON instead of
constructing transforms in Python. The same manifest format works for both
NumPy (``Pipeline``) and GPU (``TorchPipeline``) backends.

Manifest structure
------------------

.. code-block:: yaml

    seed: 1337
    p: 1.0
    transforms:
      - name: gain_perturber
        params:
          gain_db: [-6.0, 6.0]
          p: 0.8
      - name: add_noise
        params:
          snr_db: [15.0, 35.0]
          noise_type: pink
          p: 0.6
      - name: spec_augment
        params:
          freq_mask_param: 20
          time_mask_param: 30
          p: 0.5

The ``name`` field is resolved against the plugin registry first
(:func:`pvx.augment.get_transform`), then against built-in transform
classes by snake_case-stripped class name. ``params`` are forwarded
verbatim to the constructor.

Loading
-------

>>> from pvx.augment.config import load_pipeline
>>> pipeline = load_pipeline("recipes/asr_robust.yaml")
>>> audio_aug, sr = pipeline(audio, sr)

For the GPU backend:

>>> from pvx.augment.config import load_torch_pipeline
>>> torch_pipeline = load_torch_pipeline("recipes/asr_robust.yaml")
>>> tensor_aug = torch_pipeline(tensor, sr=16000)
```

## `src/pvx/augment/core.py`

**Purpose:** Core base classes for pvx augmentation transforms.

**Classes:** `Transform`, `Pipeline`, `OneOf`, `SomeOf`, `RandomApply`, `Identity`, `TransformResult`
**Functions:** `_to_2d`, `_from_2d`, `load_audio`, `save_audio`, `fingerprint_audio`, `register_transform`, `get_transform`, `list_transforms`, `_load_entry_point_plugins`, `_to_snake_case`

### Module Docstring

```text
Core base classes for pvx augmentation transforms.

All transforms operate on NumPy arrays with shape (channels, samples) or
(samples,) for mono. They return ``(audio, sample_rate)`` pairs and are
fully composable.
```

## `src/pvx/augment/gpu.py`

**Purpose:** GPU-accelerated augmentation transforms using PyTorch.

**Classes:** `TorchTransform`, `TorchGainPerturber`, `TorchAddNoise`, `TorchEQPerturber`, `TorchSpecAugment`, `TorchNormalizer`, `TorchClippingSimulator`, `TorchPipeline`, `TorchTimeStretch`, `TorchPitchShift`, `TorchRoomSimulator`, `TorchMixup`, `NumpyTransformAdapter`
**Functions:** `_require_torch`, `_torch_phase_vocoder`, `batch_process_files`

### Module Docstring

```text
GPU-accelerated augmentation transforms using PyTorch.

These transforms operate directly on ``torch.Tensor`` objects and can run
on CUDA, MPS, or CPU devices.  They avoid NumPy round-tripping and support
batched operation for maximum throughput during training.

.. warning::

   **Alpha release (0.1.0a1).**  This module is under active development.

Requirements
------------
``torch`` must be installed::

    pip install "pvx[torch]"

Usage
-----
>>> import torch
>>> from pvx.augment.gpu import TorchGainPerturber, TorchSpecAugment, TorchPipeline
>>>
>>> pipeline = TorchPipeline([
...     TorchGainPerturber(gain_db=(-6, 6), p=0.8),
...     TorchAddNoise(snr_db=(10, 30), noise_type="white", p=0.5),
...     TorchSpecAugment(freq_mask_param=27, time_mask_param=100, p=0.5),
... ])
>>>
>>> audio = torch.randn(16, 1, 48000)  # (batch, channels, samples)
>>> audio_aug = pipeline(audio, sr=16000)
```

## `src/pvx/augment/ir_database.py`

**Purpose:** Curated impulse response (IR) database integration.

**Classes:** `IRDatabase`
**Functions:** None

### Module Docstring

```text
Curated impulse response (IR) database integration.

Provides a simple downloader and cache manager for publicly available
impulse response collections, enabling physics-based room acoustics
augmentation via :class:`pvx.augment.ImpulseResponseConvolver`.

Supported databases
-------------------
- **EchoThief** — 115 real-world IRs from concert halls, churches,
  tunnels, parking garages, and other spaces.  CC BY license.
  http://www.echothief.com
- **OpenAIR** — University of York's open acoustic impulse response
  library with measured IRs from real spaces.
  https://openairlib.net

Usage
-----
>>> from pvx.augment.ir_database import IRDatabase
>>>
>>> db = IRDatabase(cache_dir="~/.pvx/ir_cache")
>>> db.download("echothief")
>>>
>>> # Use with ImpulseResponseConvolver
>>> from pvx.augment import ImpulseResponseConvolver
>>> aug = ImpulseResponseConvolver(db.ir_dir("echothief"), wet_range=(0.4, 1.0))
>>>
>>> # Or get specific IRs by room category
>>> halls = db.filter("echothief", category="hall")
>>> aug = ImpulseResponseConvolver(halls, wet_range=(0.5, 0.9))
```

## `src/pvx/augment/noise.py`

**Purpose:** Noise-based audio augmentation transforms.

**Classes:** `AddNoise`, `BackgroundMixer`, `ImpulseNoise`
**Functions:** `_db_to_linear`, `_linear_to_db`, `_rms`, `_scale_to_snr`, `_generate_white_noise`, `_generate_pink_noise`, `_generate_brown_noise`, `_generate_bandlimited_noise`

### Module Docstring

```text
Noise-based audio augmentation transforms.

Provides additive noise injection at controlled SNR levels, background
audio mixing, and impulse noise (click/pop) simulation.  All transforms
are NumPy-native and require only ``scipy`` as an optional dependency for
spectral-shaped noise generation.
```

## `src/pvx/augment/room.py`

**Purpose:** Room acoustics augmentation transforms.

**Classes:** `RoomSimulator`, `ImpulseResponseConvolver`
**Functions:** `_generate_synthetic_rir`

### Module Docstring

```text
Room acoustics augmentation transforms.

Provides synthetic room impulse response (RIR) approximation via an
exponentially-decaying noise model and convolution with arbitrary
user-supplied impulse response files.

The synthetic RIR generator is a lightweight statistical approximation
suitable for data augmentation — it does *not* perform physical room
modelling (image-source or ray-tracing).  For physics-based simulation,
use external tools such as ``pyroomacoustics`` or ``gpuRIR`` and load
the resulting IR files with :class:`ImpulseResponseConvolver`.

No external neural-network dependencies required.
```

## `src/pvx/augment/spectral.py`

**Purpose:** Spectral-domain augmentation transforms.

**Classes:** `SpecAugment`, `EQPerturber`, `SpectralNoise`, `PitchShiftSimple`
**Functions:** `_stft`, `_istft`

### Module Docstring

```text
Spectral-domain augmentation transforms.

Implements SpecAugment (frequency and time masking), random EQ perturbation,
spectral noise injection, and harmonic distortion — all operating in the
STFT domain with overlap-add resynthesis.
```

## `src/pvx/augment/streaming.py`

**Purpose:** Streaming / chunked augmentation for long-form audio.

**Classes:** None
**Functions:** `_hann_fade`, `stream_augment`, `stream_augment_file`, `stream_augment_directory`

### Module Docstring

```text
Streaming / chunked augmentation for long-form audio.

Processes audio files in overlapping chunks to keep memory usage bounded,
then crossfades the overlapping regions to avoid audible seams.

This is designed for podcasts, audiobooks, meetings, and other long-form
content where loading the entire file into memory is impractical.

Usage
-----
>>> from pvx.augment import Pipeline, AddNoise, GainPerturber
>>> from pvx.augment.streaming import stream_augment, stream_augment_file
>>>
>>> pipeline = Pipeline([
...     GainPerturber(gain_db=(-3, 3), p=0.8),
...     AddNoise(snr_db=(15, 35), noise_type="pink", p=0.5),
... ], seed=42)
>>>
>>> # Process a file with bounded memory
>>> stream_augment_file(
...     "podcast_2h.wav",
...     "podcast_2h_aug.wav",
...     pipeline=pipeline,
...     chunk_duration_s=30.0,
...     seed=42,
... )
>>>
>>> # Or process a large in-memory array
>>> for chunk_out in stream_augment(audio, sr, pipeline, chunk_duration_s=30.0):
...     # Each chunk_out is (audio_chunk, sr)
...     pass

Notes
-----
- Chunk boundaries use overlap-add with a crossfade to avoid clicks.
- Each chunk gets a deterministic seed derived from (global_seed, chunk_index).
- Transforms that require global context (e.g., Normalizer with mode="rms")
  will operate per-chunk.  For global normalization, use a two-pass approach.
- TimeStretch changes the duration of each chunk.  The crossfade logic
  accounts for output-length mismatches.
```

## `src/pvx/augment/time_domain.py`

**Purpose:** Time-domain audio augmentation transforms.

**Classes:** `GainPerturber`, `Normalizer`, `ClippingSimulator`, `TimeShift`, `Reverse`, `Fade`, `TrimSilence`, `FixedLengthCrop`, `TimeStretch`, `PitchShift`
**Functions:** `_has_torch`, `_has_torchaudio`, `_resolve_engine`, `_torchaudio_time_stretch`, `_torchaudio_pitch_shift`, `_pytorch_time_stretch`, `_pytorch_pitch_shift`, `_call_pvx_voc`

### Module Docstring

```text
Time-domain audio augmentation transforms.

Provides transforms that operate on the waveform directly: gain
perturbation, clipping, time shifting, fade in/out, silence trimming, and
reverse.  Production-quality time-stretch and pitch-shift wrappers that
call ``pvx voc`` are also provided here.
```

## `src/pvx/cli/__init__.py`

**Purpose:** CLI entrypoints for pvx tools.

**Classes:** None
**Functions:** None

### Module Docstring

```text
CLI entrypoints for pvx tools.
```

## `src/pvx/cli/catalog.py`

**Purpose:** Static command catalog for the unified pvx CLI.

**Classes:** `ToolSpec`
**Functions:** `build_tool_index`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.catalog --help`

### Module Docstring

```text
Static command catalog for the unified pvx CLI.
```

## `src/pvx/cli/hps_pitch_track.py`

**Purpose:** Track F0 and emit a pvx control-map CSV for pitch-follow pipelines.

**Classes:** None
**Functions:** `_read_audio`, `_acf_pitch_and_confidence`, `_estimate_reference_hz`, `_smooth`, `_track_pyin`, `_track_acf`, `build_parser`, `validate_args`, `_emit_csv`, `_derive_stretch_track`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.hps_pitch_track --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.hps_pitch_track [-h] [--output OUTPUT]
                                          [--backend {auto,pyin,acf}]
                                          [--fmin FMIN] [--fmax FMAX]
                                          [--frame-length FRAME_LENGTH]
                                          [--hop-size HOP_SIZE]
                                          [--ratio-reference {median,mean,first,hz}]
                                          [--reference-hz REFERENCE_HZ]
                                          [--ratio-min RATIO_MIN]
                                          [--ratio-max RATIO_MAX]
                                          [--smooth-frames SMOOTH_FRAMES]
                                          [--confidence-floor CONFIDENCE_FLOOR]
                                          [--emit {pitch_map,stretch_map,pitch_to_stretch}]
                                          [--stretch-from {pitch_ratio,inv_pitch_ratio,f0_hz}]
                                          [--stretch-scale STRETCH_SCALE]
                                          [--stretch-min STRETCH_MIN]
                                          [--stretch-max STRETCH_MAX]
                                          [--stretch STRETCH]
                                          [--feature-set {none,basic,advanced,all}]
                                          [--mfcc-count MFCC_COUNT]
                                          [--verbosity {silent,quiet,normal,verbose,debug}]
                                          [-v] [--quiet] [--silent]
                                          input

HPS/pyin-style pitch tracker that emits pvx control-map CSV to stdout.

positional arguments:
  input                 Input audio file path or '-' for stdin audio

options:
  -h, --help            show this help message and exit
  --output OUTPUT       Output CSV path (default: '-' for stdout)
  --backend {auto,pyin,acf}
                        Pitch backend (default: auto -> pyin if available, else acf)
  --fmin FMIN           Minimum F0 in Hz (default: 50)
  --fmax FMAX           Maximum F0 in Hz (default: 1200)
  --frame-length FRAME_LENGTH
                        Frame length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 256)
  --ratio-reference {median,mean,first,hz}
                        Reference for emitted pitch_ratio values (default: median voiced f0).
  --reference-hz REFERENCE_HZ
                        Reference frequency in Hz when --ratio-reference hz.
  --ratio-min RATIO_MIN
                        Lower clamp for emitted pitch_ratio (default: 0.25).
  --ratio-max RATIO_MAX
                        Upper clamp for emitted pitch_ratio (default: 4.0).
  --smooth-frames SMOOTH_FRAMES
                        Smoothing window for pitch_ratio frames (default: 5).
  --confidence-floor CONFIDENCE_FLOOR
                        Set confidence below this floor to 0.0 (default: 0.0).
  --emit {pitch_map,stretch_map,pitch_to_stretch}
                        Output mode: pitch_map (default), stretch_map, or pitch_to_stretch.
  --stretch-from {pitch_ratio,inv_pitch_ratio,f0_hz}
                        Source signal used to derive stretch in stretch-oriented emit modes (default: pitch_ratio).
  --stretch-scale STRETCH_SCALE
                        Scale factor for derived stretch tracks (default: 1.0).
  --stretch-min STRETCH_MIN
                        Lower clamp for emitted stretch in stretch-oriented modes (default: 0.25).
  --stretch-max STRETCH_MAX
                        Upper clamp for emitted stretch in stretch-oriented modes (default: 4.0).
  --stretch STRETCH     Emit constant stretch column value for --emit pitch_map (default: 1.0).
  --feature-set {none,basic,advanced,all}
                        Feature tracking preset emitted as extra CSV columns. none/basic/advanced/all (default: all).
  --mfcc-count MFCC_COUNT
                        Number of MFCC columns (mfcc_01..mfcc_N) when feature-set is advanced/all (default: 13).
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output

Examples:
  pvx pitch-track guide.wav --output guide_pitch.csv
  pvx pitch-track guide.wav --backend pyin --ratio-reference hz --reference-hz 440 --output guide_to_a440.csv
  pvx pitch-track guide.wav --emit pitch_to_stretch --output - | pvx voc target.wav --control-stdin --output followed.wav

Notes:
  - Default output columns include control map fields and feature tracks (for example: rms_db, spectral_flux, voicing_prob, MFCCs, MPEG-7-style descriptors).
  - Use --confidence-floor to gate unreliable pitch estimates.
```

### Module Docstring

```text
Track F0 and emit a pvx control-map CSV for pitch-follow pipelines.
```

## `src/pvx/cli/main.py`

**Purpose:** Compatibility entrypoint for the unified pvx CLI.

**Classes:** None
**Functions:** None

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.main --help`

### CLI Help Snapshot

```text
usage: pvx [-h] [command] ...

Unified CLI for pvx (audio quality first, speed second).
Use subcommands to access all existing pvx tools from one entrypoint.

positional arguments:
  command     Subcommand name, helper command, or input path (defaults to `voc` when an input path is provided)
  args        Arguments forwarded directly to the selected subcommand

options:
  -h, --help  show this help message and exit

Quick start:
  pvx voc input.wav --stretch 1.2 --output output.wav
  pvx input.wav --stretch 1.2 --output output.wav   # defaults to `voc`
  pvx follow guide.wav target.wav --output followed.wav --emit pitch_to_stretch
  pvx chain input.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output out.wav
  pvx stream input.wav --output out.wav --chunk-seconds 0.2 --time-stretch 2.0
  pvx stretch-budget input.wav --disk-budget 20GB --bit-depth 16 --requested-stretch 1000000
  pvx doctor
  pvx quickstart input.wav --output output.wav
  pvx safe input.wav --material mix --output output.wav
  pvx transforms
  pvx smoke --output smoke_out.wav
  pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337
  pvx augment-manifest validate aug_out/augment_manifest.jsonl
  pvx voc input.wav --output-dir out --lucky 8
  pvx list
  pvx examples basic
  pvx help voc

Available tool commands: voc, freeze, harmonize, conform, morph, warp, formant, transient, unison, denoise, deverb, retune, layer, pitch-track, analysis, response, envelope, reshape, filter, tvfilter, noisefilter, bandamp, spec-compander, ring, ringfilter, ringtvfilter, chordmapper, inharmonator, trajectory-reverb, noise, rir, codec, specaugment, gain
```

### Module Docstring

```text
Compatibility entrypoint for the unified pvx CLI.
```

## `src/pvx/cli/pvx.py`

**Purpose:** Unified top-level CLI for the pvx command suite.

**Classes:** None
**Functions:** `_load_entrypoint`, `_looks_like_audio_input`, `_tool_names_csv`, `print_tools`, `print_examples`, `_prompt_text`, `_prompt_choice`, `_print_command_preview`, `print_follow_examples`, `run_doctor_mode`, `run_quickstart_mode`, `run_safe_mode`, `run_transforms_mode`, `run_smoke_mode`, `_parse_split_ratios`, `_stable_seed_from_text`, `_augment_group_key`, `run_batch_gpu_mode`, `run_augment_mode`, `run_augment_manifest_mode`, `run_guided_mode`, `_token_flag`, `_extract_flag_value`, `_strip_flags`, `_replace_flag_value`, `_consume_lucky_options`, `_lucky_output_variant`, `_lucky_mastering_overrides`, `_lucky_tool_overrides`, `_run_lucky_tool_mode`, `_run_lucky_helper_mode`, `run_follow_mode`, `run_chain_mode`, `run_stream_mode`, `_parse_size_bytes`, `_format_bytes_human`, `_infer_output_format`, `_bytes_per_sample_from_subtype`, `run_stretch_budget_mode`, `dispatch_tool`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvx --help`

### CLI Help Snapshot

```text
usage: pvx [-h] [command] ...

Unified CLI for pvx (audio quality first, speed second).
Use subcommands to access all existing pvx tools from one entrypoint.

positional arguments:
  command     Subcommand name, helper command, or input path (defaults to `voc` when an input path is provided)
  args        Arguments forwarded directly to the selected subcommand

options:
  -h, --help  show this help message and exit

Quick start:
  pvx voc input.wav --stretch 1.2 --output output.wav
  pvx input.wav --stretch 1.2 --output output.wav   # defaults to `voc`
  pvx follow guide.wav target.wav --output followed.wav --emit pitch_to_stretch
  pvx chain input.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output out.wav
  pvx stream input.wav --output out.wav --chunk-seconds 0.2 --time-stretch 2.0
  pvx stretch-budget input.wav --disk-budget 20GB --bit-depth 16 --requested-stretch 1000000
  pvx doctor
  pvx quickstart input.wav --output output.wav
  pvx safe input.wav --material mix --output output.wav
  pvx transforms
  pvx smoke --output smoke_out.wav
  pvx augment data/*.wav --output-dir aug_out --variants-per-input 4 --intent asr_robust --seed 1337
  pvx augment-manifest validate aug_out/augment_manifest.jsonl
  pvx voc input.wav --output-dir out --lucky 8
  pvx list
  pvx examples basic
  pvx help voc

Available tool commands: voc, freeze, harmonize, conform, morph, warp, formant, transient, unison, denoise, deverb, retune, layer, pitch-track, analysis, response, envelope, reshape, filter, tvfilter, noisefilter, bandamp, spec-compander, ring, ringfilter, ringtvfilter, chordmapper, inharmonator, trajectory-reverb, noise, rir, codec, specaugment, gain
```

### Module Docstring

```text
Unified top-level CLI for the pvx command suite.
```

## `src/pvx/cli/pvx_augment.py`

**Purpose:** Augmentation helpers and subcommands for the unified pvx CLI.

**Classes:** None
**Functions:** `_expand_augment_inputs`, `_parse_split_ratios`, `_pick_split`, `_stable_seed_from_text`, `_augment_group_key`, `_token_flag`, `_flag_present`, `_load_augment_policy`, `_load_label_metadata`, `_source_metadata`, `_assign_balanced_split_for_groups`, `_sha256_file`, `_audio_audit_metrics`, `_manifest_required_errors`, `_load_manifest_jsonl`, `_merge_manifest_rows`, `_sample_augment_params`, `_render_job_pytorch`, `_augment_output_conflict`, `run_batch_gpu_mode`, `run_augment_mode`, `run_augment_manifest_mode`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvx_augment --help`

### Module Docstring

```text
Augmentation helpers and subcommands for the unified pvx CLI.
```

## `src/pvx/cli/pvx_helpers.py`

**Purpose:** Helper modes and utility flows for the unified pvx CLI.

**Classes:** None
**Functions:** `run_doctor_mode`, `run_quickstart_mode`, `_token_flag`, `run_safe_mode`, `run_transforms_mode`, `_build_smoke_signal`, `run_smoke_mode`, `run_guided_mode`, `_extract_flag_value`, `_strip_flags`, `_replace_flag_value`, `_consume_lucky_options`, `_lucky_output_variant`, `_lucky_mastering_overrides`, `_lucky_tool_overrides`, `_run_lucky_tool_mode`, `_run_lucky_helper_mode`, `_parse_size_bytes`, `_format_bytes_human`, `_infer_output_format`, `_bytes_per_sample_from_subtype`, `run_stretch_budget_mode`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvx_helpers --help`

### Module Docstring

```text
Helper modes and utility flows for the unified pvx CLI.
```

## `src/pvx/cli/pvx_pipeline.py`

**Purpose:** Pipeline-oriented helper commands for the unified pvx CLI.

**Classes:** `_BytesStdin`
**Functions:** `print_follow_examples`, `_extract_follow_example_request`, `_split_pipeline_stages`, `_token_flag`, `_extract_flag_value`, `_run_stage_command`, `_run_stage_capture_stdout`, `_patched_stdin_bytes`, `run_follow_mode`, `run_chain_mode`, `run_stream_mode`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvx_pipeline --help`

### Module Docstring

```text
Pipeline-oriented helper commands for the unified pvx CLI.
```

## `src/pvx/cli/pvxanalysis.py`

**Purpose:** Persist and inspect reusable phase-vocoder analysis artifacts (PVXAN).

**Classes:** None
**Functions:** `_normalize_argv`, `_default_output_path`, `_print_summary`, `_write_json`, `build_parser`, `_validate_create_args`, `_run_create`, `_run_inspect`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxanalysis --help`

### CLI Help Snapshot

```text
usage: pvx analysis [-h] {create,inspect} ...

Create and inspect PVXAN analysis artifacts for reusable phase-vocoder workflows.

positional arguments:
  {create,inspect}
    create          Analyze audio and save PVXAN artifact
    inspect         Inspect existing PVXAN artifact

options:
  -h, --help        show this help message and exit

Examples:
  pvx analysis create input.wav --output input.pvxan.npz --n-fft 4096 --hop-size 256
  pvx analysis inspect input.pvxan.npz
  pvxanalysis input.wav --output input.pvxan.npz

Notes:
  - When command is omitted, create mode is assumed for convenience.
  - PVXAN stores complex STFT payloads (channels x frames x bins) in compressed NPZ format.
```

### Module Docstring

```text
Persist and inspect reusable phase-vocoder analysis artifacts (PVXAN).
```

## `src/pvx/cli/pvxbandamp.py`

**Purpose:** Response-peak band emphasis wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxbandamp --help`

### CLI Help Snapshot

```text
usage: pvx bandamp [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
                   [--output-format OUTPUT_FORMAT] [--stdout] [--overwrite]
                   [--dry-run]
                   [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                   [--quiet] [--silent] [--normalize {none,peak,rms}]
                   [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                   [--target-lufs TARGET_LUFS]
                   [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                   [--compressor-ratio COMPRESSOR_RATIO]
                   [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                   [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                   [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                   [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                   [--expander-ratio EXPANDER_RATIO]
                   [--expander-attack-ms EXPANDER_ATTACK_MS]
                   [--expander-release-ms EXPANDER_RELEASE_MS]
                   [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                   [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                   [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                   [--compander-attack-ms COMPANDER_ATTACK_MS]
                   [--compander-release-ms COMPANDER_RELEASE_MS]
                   [--compander-makeup-db COMPANDER_MAKEUP_DB]
                   [--limiter-threshold LIMITER_THRESHOLD]
                   [--soft-clip-level SOFT_CLIP_LEVEL]
                   [--soft-clip-type {tanh,arctan,cubic}]
                   [--soft-clip-drive SOFT_CLIP_DRIVE]
                   [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                   [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                   [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                   [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                   [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                   [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                   [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                   [--kaiser-beta KAISER_BETA]
                   [--transform {fft,dft,czt,dct,dst,hartley}]
                   [--phase-engine {propagate,hybrid,random}]
                   [--ambient-phase-mix AMBIENT_PHASE_MIX]
                   [--phase-random-seed PHASE_RANDOM_SEED]
                   [--onset-time-credit]
                   [--onset-credit-pull ONSET_CREDIT_PULL]
                   [--onset-credit-max ONSET_CREDIT_MAX] [--no-onset-realign]
                   [--no-center] [--device {auto,cpu,cuda}]
                   [--cuda-device CUDA_DEVICE]
                   [--operator {filter,tvfilter,noisefilter,bandamp,spec-compander}]
                   --response RESPONSE [--response-mix RESPONSE_MIX]
                   [--dry-mix DRY_MIX] [--response-gain-db RESPONSE_GAIN_DB]
                   [--transpose-semitones TRANSPOSE_SEMITONES]
                   [--shift-bins SHIFT_BINS] [--tv-map TV_MAP]
                   [--tv-key TV_KEY]
                   [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                   [--tv-order TV_ORDER] [--noise-floor NOISE_FLOOR]
                   [--band-gain-db BAND_GAIN_DB]
                   [--band-width-bins BAND_WIDTH_BINS]
                   [--peak-count PEAK_COUNT]
                   [--comp-threshold-db COMP_THRESHOLD_DB]
                   [--comp-ratio COMP_RATIO] [--expand-ratio EXPAND_RATIO]
                   inputs [inputs ...]

Response-driven spectral processing using PVXRF artifacts.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _filter)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,qu
... [truncated]
```

### Module Docstring

```text
Response-peak band emphasis wrapper.
```

## `src/pvx/cli/pvxchordmapper.py`

**Purpose:** Chordmapper wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxchordmapper --help`

### CLI Help Snapshot

```text
usage: pvx chordmapper [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                       [--suffix SUFFIX] [--output-format OUTPUT_FORMAT]
                       [--stdout] [--overwrite] [--dry-run]
                       [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                       [--quiet] [--silent] [--normalize {none,peak,rms}]
                       [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                       [--target-lufs TARGET_LUFS]
                       [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                       [--compressor-ratio COMPRESSOR_RATIO]
                       [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                       [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                       [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                       [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                       [--expander-ratio EXPANDER_RATIO]
                       [--expander-attack-ms EXPANDER_ATTACK_MS]
                       [--expander-release-ms EXPANDER_RELEASE_MS]
                       [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                       [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                       [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                       [--compander-attack-ms COMPANDER_ATTACK_MS]
                       [--compander-release-ms COMPANDER_RELEASE_MS]
                       [--compander-makeup-db COMPANDER_MAKEUP_DB]
                       [--limiter-threshold LIMITER_THRESHOLD]
                       [--soft-clip-level SOFT_CLIP_LEVEL]
                       [--soft-clip-type {tanh,arctan,cubic}]
                       [--soft-clip-drive SOFT_CLIP_DRIVE]
                       [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                       [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                       [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                       [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                       [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                       [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                       [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                       [--kaiser-beta KAISER_BETA]
                       [--transform {fft,dft,czt,dct,dst,hartley}]
                       [--phase-engine {propagate,hybrid,random}]
                       [--ambient-phase-mix AMBIENT_PHASE_MIX]
                       [--phase-random-seed PHASE_RANDOM_SEED]
                       [--onset-time-credit]
                       [--onset-credit-pull ONSET_CREDIT_PULL]
                       [--onset-credit-max ONSET_CREDIT_MAX]
                       [--no-onset-realign] [--no-center]
                       [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                       [--operator {chordmapper,inharmonator}]
                       [--root-hz ROOT_HZ]
                       [--chord {aug,dim,dom7,maj7,major,min7,minor,power,sus2,sus4}]
                       [--strength STRENGTH]
                       [--tolerance-cents TOLERANCE_CENTS]
                       [--boost-db BOOST_DB] [--attenuation ATTENUATION]
                       [--inharmonic-f0-hz INHARMONIC_F0_HZ]
                       [--inharmonicity INHARMONICITY]
                       [--inharmonic-mix INHARMONIC_MIX] [--dry-mix DRY_MIX]
                       inputs [inputs ...]

Harmonic/chord mapping and inharmonic spectral warping.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _harm)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for
... [truncated]
```

### Module Docstring

```text
Chordmapper wrapper.
```

## `src/pvx/cli/pvxcodec.py`

**Purpose:** pvxcodec — simulate lossy codec and transmission degradation.

**Classes:** None
**Functions:** `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxcodec --help`

### CLI Help Snapshot

```text
usage: pvxcodec [-h] --output OUTPUT
                [--codec {mp3_low,mp3_medium,telephone,voip_narrow,voip_wide,am_radio,lo_fi,random}]
                [--bits BITS] [--seed SEED] [--bit-depth {16,24,32,float}]
                input

Simulate lossy codec artifacts (MP3, telephone, VoIP) without codec binaries.

positional arguments:
  input                 Input audio file

options:
  -h, --help            show this help message and exit
  --output, -o OUTPUT   Output audio file
  --codec {mp3_low,mp3_medium,telephone,voip_narrow,voip_wide,am_radio,lo_fi,random}
                        Codec preset to simulate (default: none — use --bits
                        for raw bit-crush)
  --bits BITS           Bit depth for raw bit-crushing (4-16). Applied after
                        codec if both specified.
  --seed SEED           Random seed (default: 0)
  --bit-depth {16,24,32,float}
                        Output file bit depth (default: 24)

pvxcodec — simulate lossy codec and transmission degradation.

Applies bandwidth limiting, sample-rate decimation, and bit-depth
quantization to simulate the artifacts of real-world codec processing
(MP3, telephone, VoIP, AM radio) without requiring external codec
binaries.

Available presets
-----------------
  mp3_low       ~32 kbps MP3 (11 kHz bandwidth, 12-bit quantization)
  mp3_medium    ~128 kbps MP3 (16 kHz bandwidth, 14-bit quantization)
  telephone     POTS telephone band (300-3400 Hz, 8 kHz, 8-bit)
  voip_narrow   Narrow-band VoIP (50-4000 Hz, 8 kHz, 8-bit)
  voip_wide     Wide-band VoIP (50-7000 Hz, 16 kHz, 12-bit)
  am_radio      AM radio (200-5000 Hz, 8-bit)
  lo_fi         Extreme lo-fi (8 kHz bandwidth, 6-bit)
  random        Pick a random preset each run

Examples
--------
pvxcodec speech.wav --codec telephone --output speech_phone.wav
pvxcodec music.wav --codec mp3_low --output music_lossy.wav
pvxcodec speech.wav --codec random --seed 7 --output out.wav
pvxcodec music.wav --bits 8 --output music_crushed.wav
```

### Module Docstring

```text
pvxcodec — simulate lossy codec and transmission degradation.

Applies bandwidth limiting, sample-rate decimation, and bit-depth
quantization to simulate the artifacts of real-world codec processing
(MP3, telephone, VoIP, AM radio) without requiring external codec
binaries.

Available presets
-----------------
  mp3_low       ~32 kbps MP3 (11 kHz bandwidth, 12-bit quantization)
  mp3_medium    ~128 kbps MP3 (16 kHz bandwidth, 14-bit quantization)
  telephone     POTS telephone band (300-3400 Hz, 8 kHz, 8-bit)
  voip_narrow   Narrow-band VoIP (50-4000 Hz, 8 kHz, 8-bit)
  voip_wide     Wide-band VoIP (50-7000 Hz, 16 kHz, 12-bit)
  am_radio      AM radio (200-5000 Hz, 8-bit)
  lo_fi         Extreme lo-fi (8 kHz bandwidth, 6-bit)
  random        Pick a random preset each run

Examples
--------
pvxcodec speech.wav --codec telephone --output speech_phone.wav
pvxcodec music.wav --codec mp3_low --output music_lossy.wav
pvxcodec speech.wav --codec random --seed 7 --output out.wav
pvxcodec music.wav --bits 8 --output music_crushed.wav
```

## `src/pvx/cli/pvxconform.py`

**Purpose:** Conform timing and pitch to a user-provided segment map.

**Classes:** None
**Functions:** `expand_segments`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxconform --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxconform [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                     [--suffix SUFFIX]
                                     [--output-format OUTPUT_FORMAT]
                                     [--stdout] [--overwrite] [--dry-run]
                                     [--verbosity {silent,quiet,normal,verbose,debug}]
                                     [-v] [--quiet] [--silent]
                                     [--normalize {none,peak,rms}]
                                     [--peak-dbfs PEAK_DBFS]
                                     [--rms-dbfs RMS_DBFS]
                                     [--target-lufs TARGET_LUFS]
                                     [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                     [--compressor-ratio COMPRESSOR_RATIO]
                                     [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                     [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                     [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                     [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                     [--expander-ratio EXPANDER_RATIO]
                                     [--expander-attack-ms EXPANDER_ATTACK_MS]
                                     [--expander-release-ms EXPANDER_RELEASE_MS]
                                     [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                     [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                     [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                     [--compander-attack-ms COMPANDER_ATTACK_MS]
                                     [--compander-release-ms COMPANDER_RELEASE_MS]
                                     [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                     [--limiter-threshold LIMITER_THRESHOLD]
                                     [--soft-clip-level SOFT_CLIP_LEVEL]
                                     [--soft-clip-type {tanh,arctan,cubic}]
                                     [--soft-clip-drive SOFT_CLIP_DRIVE]
                                     [--hard-clip-level HARD_CLIP_LEVEL]
                                     [--clip] [--subtype SUBTYPE]
                                     [--bit-depth {inherit,16,24,32f}]
                                     [--dither {none,tpdf}]
                                     [--dither-seed DITHER_SEED]
                                     [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                     [--metadata-policy {none,sidecar,copy}]
                                     [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                     [--hop-size HOP_SIZE]
                                     [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                     [--kaiser-beta KAISER_BETA]
                                     [--transform {fft,dft,czt,dct,dst,hartley}]
                                     [--phase-engine {propagate,hybrid,random}]
                                     [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                     [--phase-random-seed PHASE_RANDOM_SEED]
                                     [--onset-time-credit]
                                     [--onset-credit-pull ONSET_CREDIT_PULL]
                                     [--onset-credit-max ONSET_CREDIT_MAX]
                                     [--no-onset-realign] [--no-center]
                                     [--device {auto,cpu,cuda}]
                                     [--cuda-device CUDA_DEVICE] --map MAP
                                     [--crossfade-ms CROSSFADE_MS]
                                     [--resample-mode {auto,fft,linear}]
                                     inputs [inputs ...]

Conform audio to a CSV map. CSV columns: start_sec,end_sec,stretch, and one pitch field: pitch_semitones or pitch_cents or pitch_ratio (pitch_ratio accepts decimals, fractions like 3/2, or expressions like 2^(1/12))

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --outp
... [truncated]
```

### Module Docstring

```text
Conform timing and pitch to a user-provided segment map.
```

## `src/pvx/cli/pvxdenoise.py`

**Purpose:** Phase-consistent spectral denoiser.

**Classes:** None
**Functions:** `smooth_mask`, `denoise_channel`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxdenoise --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxdenoise [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                     [--suffix SUFFIX]
                                     [--output-format OUTPUT_FORMAT]
                                     [--stdout] [--overwrite] [--dry-run]
                                     [--verbosity {silent,quiet,normal,verbose,debug}]
                                     [-v] [--quiet] [--silent]
                                     [--normalize {none,peak,rms}]
                                     [--peak-dbfs PEAK_DBFS]
                                     [--rms-dbfs RMS_DBFS]
                                     [--target-lufs TARGET_LUFS]
                                     [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                     [--compressor-ratio COMPRESSOR_RATIO]
                                     [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                     [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                     [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                     [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                     [--expander-ratio EXPANDER_RATIO]
                                     [--expander-attack-ms EXPANDER_ATTACK_MS]
                                     [--expander-release-ms EXPANDER_RELEASE_MS]
                                     [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                     [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                     [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                     [--compander-attack-ms COMPANDER_ATTACK_MS]
                                     [--compander-release-ms COMPANDER_RELEASE_MS]
                                     [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                     [--limiter-threshold LIMITER_THRESHOLD]
                                     [--soft-clip-level SOFT_CLIP_LEVEL]
                                     [--soft-clip-type {tanh,arctan,cubic}]
                                     [--soft-clip-drive SOFT_CLIP_DRIVE]
                                     [--hard-clip-level HARD_CLIP_LEVEL]
                                     [--clip] [--subtype SUBTYPE]
                                     [--bit-depth {inherit,16,24,32f}]
                                     [--dither {none,tpdf}]
                                     [--dither-seed DITHER_SEED]
                                     [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                     [--metadata-policy {none,sidecar,copy}]
                                     [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                     [--hop-size HOP_SIZE]
                                     [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                     [--kaiser-beta KAISER_BETA]
                                     [--transform {fft,dft,czt,dct,dst,hartley}]
                                     [--phase-engine {propagate,hybrid,random}]
                                     [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                     [--phase-random-seed PHASE_RANDOM_SEED]
                                     [--onset-time-credit]
                                     [--onset-credit-pull ONSET_CREDIT_PULL]
                                     [--onset-credit-max ONSET_CREDIT_MAX]
                                     [--no-onset-realign] [--no-center]
                                     [--device {auto,cpu,cuda}]
                                     [--cuda-device CUDA_DEVICE]
                                     [--noise-seconds NOISE_SECONDS]
                                     [--noise-file NOISE_FILE]
                                     [--reduction-db REDUCTION_DB]
                                     [--floor FLOOR] [--smooth SMOOTH]
                                     inputs [inputs ...]

Spectral subtraction denoiser

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path
... [truncated]
```

### Module Docstring

```text
Phase-consistent spectral denoiser.
```

## `src/pvx/cli/pvxdeverb.py`

**Purpose:** Spectral tail suppression for dereverberation-like cleanup.

**Classes:** None
**Functions:** `deverb_channel`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxdeverb --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxdeverb [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                    [--suffix SUFFIX]
                                    [--output-format OUTPUT_FORMAT] [--stdout]
                                    [--overwrite] [--dry-run]
                                    [--verbosity {silent,quiet,normal,verbose,debug}]
                                    [-v] [--quiet] [--silent]
                                    [--normalize {none,peak,rms}]
                                    [--peak-dbfs PEAK_DBFS]
                                    [--rms-dbfs RMS_DBFS]
                                    [--target-lufs TARGET_LUFS]
                                    [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                    [--compressor-ratio COMPRESSOR_RATIO]
                                    [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                    [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                    [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                    [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                    [--expander-ratio EXPANDER_RATIO]
                                    [--expander-attack-ms EXPANDER_ATTACK_MS]
                                    [--expander-release-ms EXPANDER_RELEASE_MS]
                                    [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                    [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                    [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                    [--compander-attack-ms COMPANDER_ATTACK_MS]
                                    [--compander-release-ms COMPANDER_RELEASE_MS]
                                    [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                    [--limiter-threshold LIMITER_THRESHOLD]
                                    [--soft-clip-level SOFT_CLIP_LEVEL]
                                    [--soft-clip-type {tanh,arctan,cubic}]
                                    [--soft-clip-drive SOFT_CLIP_DRIVE]
                                    [--hard-clip-level HARD_CLIP_LEVEL]
                                    [--clip] [--subtype SUBTYPE]
                                    [--bit-depth {inherit,16,24,32f}]
                                    [--dither {none,tpdf}]
                                    [--dither-seed DITHER_SEED]
                                    [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                    [--metadata-policy {none,sidecar,copy}]
                                    [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                    [--hop-size HOP_SIZE]
                                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                    [--kaiser-beta KAISER_BETA]
                                    [--transform {fft,dft,czt,dct,dst,hartley}]
                                    [--phase-engine {propagate,hybrid,random}]
                                    [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                    [--phase-random-seed PHASE_RANDOM_SEED]
                                    [--onset-time-credit]
                                    [--onset-credit-pull ONSET_CREDIT_PULL]
                                    [--onset-credit-max ONSET_CREDIT_MAX]
                                    [--no-onset-realign] [--no-center]
                                    [--device {auto,cpu,cuda}]
                                    [--cuda-device CUDA_DEVICE]
                                    [--strength STRENGTH] [--decay DECAY]
                                    [--floor FLOOR]
                                    inputs [inputs ...]

Reduce spectral tails / reverberant smear

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _deverb)
  --output-format OUTPUT_FORMAT
                        Output extension/form
... [truncated]
```

### Module Docstring

```text
Spectral tail suppression for dereverberation-like cleanup.
```

## `src/pvx/cli/pvxenvelope.py`

**Purpose:** PVC-style envelope function-stream generator.

**Classes:** None
**Functions:** `build_parser`, `_resolve_output_format`, `_render_payload`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxenvelope --help`

### CLI Help Snapshot

```text
usage: pvx envelope [-h]
                    [--mode {adsr,ramp,exp,sine,triangle,square,saw_up,saw_down}]
                    --duration DURATION [--rate RATE] [--start START]
                    [--peak PEAK] [--sustain SUSTAIN] [--end END]
                    [--attack-sec ATTACK_SEC] [--decay-sec DECAY_SEC]
                    [--release-sec RELEASE_SEC] [--exp-curve EXP_CURVE]
                    [--sine-cycles SINE_CYCLES] [--frequency-hz FREQUENCY_HZ]
                    [--sine-phase-rad SINE_PHASE_RAD]
                    [--duty-cycle DUTY_CYCLE] [--min MIN_VALUE]
                    [--max MAX_VALUE] [--key KEY] [--format {csv,json,auto}]
                    [--output OUTPUT] [--stdout]
                    [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                    [--quiet] [--silent]

Generate deterministic envelope/LFO control-rate maps for pvx CSV/JSON workflows.

options:
  -h, --help            show this help message and exit
  --mode, --wave {adsr,ramp,exp,sine,triangle,square,saw_up,saw_down}
                        Envelope/LFO waveform mode
  --duration DURATION   Envelope duration in seconds
  --rate RATE           Control points per second (default: 20)
  --start, --center START
                        Start/center value (mode-dependent; for periodic modes this is the center/DC offset)
  --peak, --amplitude PEAK
                        Peak/depth/amplitude value (mode-dependent; periodic modes use this as waveform amplitude)
  --sustain SUSTAIN     Sustain value for ADSR mode
  --end END             End value for ADSR/ramp/exp modes
  --attack-sec ATTACK_SEC
                        ADSR attack in seconds
  --decay-sec DECAY_SEC
                        ADSR decay in seconds
  --release-sec RELEASE_SEC
                        ADSR release in seconds
  --exp-curve EXP_CURVE
                        Exponential curve steepness for exp mode
  --sine-cycles, --cycles SINE_CYCLES
                        Waveform cycles across full duration (periodic modes)
  --frequency-hz, --freq FREQUENCY_HZ
                        Waveform frequency in Hz for periodic modes (converted to cycles = frequency * duration)
  --sine-phase-rad, --phase-rad, --phase SINE_PHASE_RAD
                        Initial waveform phase in radians for periodic modes
  --duty-cycle DUTY_CYCLE
                        Duty cycle in (0,1) for square mode (default: 0.5)
  --min MIN_VALUE       Optional value clamp minimum
  --max MAX_VALUE       Optional value clamp maximum
  --key KEY             Output control column/key name (default: value)
  --format {csv,json,auto}
                        Output format (default: auto from extension or csv for stdout)
  --output, --out OUTPUT
                        Output path. Default: stdout.
  --stdout              Write map to stdout
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output

Examples:
  pvx envelope --mode adsr --duration 8 --rate 20 --attack-sec 0.2 --decay-sec 0.6 --sustain 1.1 --release-sec 1.0 --key stretch --output stretch_env.csv
  pvx envelope --mode ramp --duration 6 --rate 10 --start 1.0 --end 0.5 --key pitch_ratio --stdout | pvx voc input.wav --stretch 1.0 --pitch-map-stdin --route pitch_ratio=pitch_ratio --output out.wav
  pvx lfo --wave sine --duration 12 --rate 30 --center 1.0 --amplitude 0.25 --cycles 6 --min 0.75 --max 1.25 --key stretch --output stretch_lfo.json
  pvx lfo --wave triangle --duration 8 --frequency-hz 0.5 --center 1.0 --amplitude 0.2 --key stretch --output stretch_triangle.csv
  pvx lfo --wave square --duration 8 --frequency-hz 2.0 --center 1.0 --amplitude 0.3 --duty-cycle 0.35 --key pitch_ratio --output pitch_square.csv

Notes:
  - Generated maps are control-rate signals (not sample-rate audio).
  - Use --key to emit directly as stretch or pitch_ratio for pvx voc map workflows.
  - Use `pvx lfo` as a shorthand alias for `pvx envelope` when authoring periodic control signals.
```

### Module Docstring

```text
PVC-style envelope function-stream generator.
```

## `src/pvx/cli/pvxfilter.py`

**Purpose:** PVC-inspired response-driven spectral operators.

**Classes:** None
**Functions:** `build_parser`, `run_filter_cli`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxfilter --help`

### CLI Help Snapshot

```text
usage: pvx filter [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
                  [--output-format OUTPUT_FORMAT] [--stdout] [--overwrite]
                  [--dry-run]
                  [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                  [--quiet] [--silent] [--normalize {none,peak,rms}]
                  [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                  [--target-lufs TARGET_LUFS]
                  [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                  [--compressor-ratio COMPRESSOR_RATIO]
                  [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                  [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                  [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                  [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                  [--expander-ratio EXPANDER_RATIO]
                  [--expander-attack-ms EXPANDER_ATTACK_MS]
                  [--expander-release-ms EXPANDER_RELEASE_MS]
                  [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                  [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                  [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                  [--compander-attack-ms COMPANDER_ATTACK_MS]
                  [--compander-release-ms COMPANDER_RELEASE_MS]
                  [--compander-makeup-db COMPANDER_MAKEUP_DB]
                  [--limiter-threshold LIMITER_THRESHOLD]
                  [--soft-clip-level SOFT_CLIP_LEVEL]
                  [--soft-clip-type {tanh,arctan,cubic}]
                  [--soft-clip-drive SOFT_CLIP_DRIVE]
                  [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                  [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                  [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                  [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                  [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                  [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                  [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                  [--kaiser-beta KAISER_BETA]
                  [--transform {fft,dft,czt,dct,dst,hartley}]
                  [--phase-engine {propagate,hybrid,random}]
                  [--ambient-phase-mix AMBIENT_PHASE_MIX]
                  [--phase-random-seed PHASE_RANDOM_SEED]
                  [--onset-time-credit]
                  [--onset-credit-pull ONSET_CREDIT_PULL]
                  [--onset-credit-max ONSET_CREDIT_MAX] [--no-onset-realign]
                  [--no-center] [--device {auto,cpu,cuda}]
                  [--cuda-device CUDA_DEVICE]
                  [--operator {filter,tvfilter,noisefilter,bandamp,spec-compander}]
                  --response RESPONSE [--response-mix RESPONSE_MIX]
                  [--dry-mix DRY_MIX] [--response-gain-db RESPONSE_GAIN_DB]
                  [--transpose-semitones TRANSPOSE_SEMITONES]
                  [--shift-bins SHIFT_BINS] [--tv-map TV_MAP]
                  [--tv-key TV_KEY]
                  [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                  [--tv-order TV_ORDER] [--noise-floor NOISE_FLOOR]
                  [--band-gain-db BAND_GAIN_DB]
                  [--band-width-bins BAND_WIDTH_BINS]
                  [--peak-count PEAK_COUNT]
                  [--comp-threshold-db COMP_THRESHOLD_DB]
                  [--comp-ratio COMP_RATIO] [--expand-ratio EXPAND_RATIO]
                  inputs [inputs ...]

Response-driven spectral processing using PVXRF artifacts.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _filter)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console
... [truncated]
```

### Module Docstring

```text
PVC-inspired response-driven spectral operators.
```

## `src/pvx/cli/pvxformant.py`

**Purpose:** Formant processing tool with optional pitch shifting.

**Classes:** None
**Functions:** `shift_envelope`, `formant_process_channel`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxformant --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxformant [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                     [--suffix SUFFIX]
                                     [--output-format OUTPUT_FORMAT]
                                     [--stdout] [--overwrite] [--dry-run]
                                     [--verbosity {silent,quiet,normal,verbose,debug}]
                                     [-v] [--quiet] [--silent]
                                     [--normalize {none,peak,rms}]
                                     [--peak-dbfs PEAK_DBFS]
                                     [--rms-dbfs RMS_DBFS]
                                     [--target-lufs TARGET_LUFS]
                                     [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                     [--compressor-ratio COMPRESSOR_RATIO]
                                     [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                     [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                     [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                     [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                     [--expander-ratio EXPANDER_RATIO]
                                     [--expander-attack-ms EXPANDER_ATTACK_MS]
                                     [--expander-release-ms EXPANDER_RELEASE_MS]
                                     [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                     [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                     [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                     [--compander-attack-ms COMPANDER_ATTACK_MS]
                                     [--compander-release-ms COMPANDER_RELEASE_MS]
                                     [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                     [--limiter-threshold LIMITER_THRESHOLD]
                                     [--soft-clip-level SOFT_CLIP_LEVEL]
                                     [--soft-clip-type {tanh,arctan,cubic}]
                                     [--soft-clip-drive SOFT_CLIP_DRIVE]
                                     [--hard-clip-level HARD_CLIP_LEVEL]
                                     [--clip] [--subtype SUBTYPE]
                                     [--bit-depth {inherit,16,24,32f}]
                                     [--dither {none,tpdf}]
                                     [--dither-seed DITHER_SEED]
                                     [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                     [--metadata-policy {none,sidecar,copy}]
                                     [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                     [--hop-size HOP_SIZE]
                                     [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                     [--kaiser-beta KAISER_BETA]
                                     [--transform {fft,dft,czt,dct,dst,hartley}]
                                     [--phase-engine {propagate,hybrid,random}]
                                     [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                     [--phase-random-seed PHASE_RANDOM_SEED]
                                     [--onset-time-credit]
                                     [--onset-credit-pull ONSET_CREDIT_PULL]
                                     [--onset-credit-max ONSET_CREDIT_MAX]
                                     [--no-onset-realign] [--no-center]
                                     [--device {auto,cpu,cuda}]
                                     [--cuda-device CUDA_DEVICE]
                                     [--pitch-shift-semitones PITCH_SHIFT_SEMITONES]
                                     [--pitch-shift-cents PITCH_SHIFT_CENTS]
                                     [--formant-shift-ratio FORMANT_SHIFT_RATIO]
                                     [--mode {shift,preserve}]
                                     [--formant-lifter FORMANT_LIFTER]
                                     [--formant-max-gain-db FORMANT_MAX_GAIN_DB]
                                     [--resample-mode {auto,fft,linear}]
                                     inputs [inputs ...]

Formant shift/preserve processor

positional arguments:
  inpu
... [truncated]
```

### Module Docstring

```text
Formant processing tool with optional pitch shifting.
```

## `src/pvx/cli/pvxfreeze.py`

**Purpose:** Spectral freeze tool built on pvx phase-vocoder primitives.

**Classes:** None
**Functions:** `_principal_angle`, `freeze_channel`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxfreeze --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxfreeze [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                    [--suffix SUFFIX]
                                    [--output-format OUTPUT_FORMAT] [--stdout]
                                    [--overwrite] [--dry-run]
                                    [--verbosity {silent,quiet,normal,verbose,debug}]
                                    [-v] [--quiet] [--silent]
                                    [--normalize {none,peak,rms}]
                                    [--peak-dbfs PEAK_DBFS]
                                    [--rms-dbfs RMS_DBFS]
                                    [--target-lufs TARGET_LUFS]
                                    [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                    [--compressor-ratio COMPRESSOR_RATIO]
                                    [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                    [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                    [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                    [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                    [--expander-ratio EXPANDER_RATIO]
                                    [--expander-attack-ms EXPANDER_ATTACK_MS]
                                    [--expander-release-ms EXPANDER_RELEASE_MS]
                                    [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                    [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                    [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                    [--compander-attack-ms COMPANDER_ATTACK_MS]
                                    [--compander-release-ms COMPANDER_RELEASE_MS]
                                    [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                    [--limiter-threshold LIMITER_THRESHOLD]
                                    [--soft-clip-level SOFT_CLIP_LEVEL]
                                    [--soft-clip-type {tanh,arctan,cubic}]
                                    [--soft-clip-drive SOFT_CLIP_DRIVE]
                                    [--hard-clip-level HARD_CLIP_LEVEL]
                                    [--clip] [--subtype SUBTYPE]
                                    [--bit-depth {inherit,16,24,32f}]
                                    [--dither {none,tpdf}]
                                    [--dither-seed DITHER_SEED]
                                    [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                    [--metadata-policy {none,sidecar,copy}]
                                    [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                    [--hop-size HOP_SIZE]
                                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                    [--kaiser-beta KAISER_BETA]
                                    [--transform {fft,dft,czt,dct,dst,hartley}]
                                    [--phase-engine {propagate,hybrid,random}]
                                    [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                    [--phase-random-seed PHASE_RANDOM_SEED]
                                    [--onset-time-credit]
                                    [--onset-credit-pull ONSET_CREDIT_PULL]
                                    [--onset-credit-max ONSET_CREDIT_MAX]
                                    [--no-onset-realign] [--no-center]
                                    [--device {auto,cpu,cuda}]
                                    [--cuda-device CUDA_DEVICE]
                                    [--freeze-time FREEZE_TIME]
                                    [--duration DURATION] [--random-phase]
                                    [--phase-mode {instantaneous,bin,hold}]
                                    inputs [inputs ...]

Freeze a spectral slice into a sustained output

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (
... [truncated]
```

### Module Docstring

```text
Spectral freeze tool built on pvx phase-vocoder primitives.
```

## `src/pvx/cli/pvxgain.py`

**Purpose:** pvxgain — apply random or fixed gain perturbation to audio.

**Classes:** None
**Functions:** `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxgain --help`

### CLI Help Snapshot

```text
usage: pvxgain [-h] --output OUTPUT [--gain GAIN] [--normalize {peak,rms}]
               [--target TARGET] [--seed SEED] [--bit-depth {16,24,32,float}]
               input

Apply random or fixed gain perturbation or normalization.

positional arguments:
  input                 Input audio file

options:
  -h, --help            show this help message and exit
  --output, -o OUTPUT   Output audio file
  --gain GAIN           Gain in dB. Single value (fixed) or 'min,max' (random
                        range). Mutually exclusive with --normalize.
  --normalize {peak,rms}
                        Normalization mode. Mutually exclusive with --gain.
  --target TARGET       Target level in dBFS for normalization (default: -1.0)
  --seed SEED           Random seed (default: 0)
  --bit-depth {16,24,32,float}
                        Output bit depth (default: 24)

pvxgain — apply random or fixed gain perturbation to audio.

A simple utility for volume normalization and random loudness perturbation.
Useful as a pre-processing or augmentation step in any audio ML pipeline.

Examples
--------
# Random gain -6 to +6 dB:
pvxgain speech.wav --gain -6,6 --output speech_gain.wav

# Normalize to -1 dBFS peak:
pvxgain speech.wav --normalize peak --target -1 --output normalized.wav

# Normalize to -20 dBFS RMS:
pvxgain speech.wav --normalize rms --target -20 --output normalized.wav

# Batch random loudness for dataset:
for f in dataset/*.wav; do
    pvxgain "$f" --gain -6,6 --seed $RANDOM --output augmented/"$(basename $f)"
done
```

### Module Docstring

```text
pvxgain — apply random or fixed gain perturbation to audio.

A simple utility for volume normalization and random loudness perturbation.
Useful as a pre-processing or augmentation step in any audio ML pipeline.

Examples
--------
# Random gain -6 to +6 dB:
pvxgain speech.wav --gain -6,6 --output speech_gain.wav

# Normalize to -1 dBFS peak:
pvxgain speech.wav --normalize peak --target -1 --output normalized.wav

# Normalize to -20 dBFS RMS:
pvxgain speech.wav --normalize rms --target -20 --output normalized.wav

# Batch random loudness for dataset:
for f in dataset/*.wav; do
    pvxgain "$f" --gain -6,6 --seed $RANDOM --output augmented/"$(basename $f)"
done
```

## `src/pvx/cli/pvxharmmap.py`

**Purpose:** PVC-inspired harmonic/chord spectral mapping CLI.

**Classes:** None
**Functions:** `build_parser`, `run_harmmap_cli`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxharmmap --help`

### CLI Help Snapshot

```text
usage: pvx chordmapper [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                       [--suffix SUFFIX] [--output-format OUTPUT_FORMAT]
                       [--stdout] [--overwrite] [--dry-run]
                       [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                       [--quiet] [--silent] [--normalize {none,peak,rms}]
                       [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                       [--target-lufs TARGET_LUFS]
                       [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                       [--compressor-ratio COMPRESSOR_RATIO]
                       [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                       [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                       [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                       [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                       [--expander-ratio EXPANDER_RATIO]
                       [--expander-attack-ms EXPANDER_ATTACK_MS]
                       [--expander-release-ms EXPANDER_RELEASE_MS]
                       [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                       [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                       [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                       [--compander-attack-ms COMPANDER_ATTACK_MS]
                       [--compander-release-ms COMPANDER_RELEASE_MS]
                       [--compander-makeup-db COMPANDER_MAKEUP_DB]
                       [--limiter-threshold LIMITER_THRESHOLD]
                       [--soft-clip-level SOFT_CLIP_LEVEL]
                       [--soft-clip-type {tanh,arctan,cubic}]
                       [--soft-clip-drive SOFT_CLIP_DRIVE]
                       [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                       [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                       [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                       [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                       [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                       [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                       [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                       [--kaiser-beta KAISER_BETA]
                       [--transform {fft,dft,czt,dct,dst,hartley}]
                       [--phase-engine {propagate,hybrid,random}]
                       [--ambient-phase-mix AMBIENT_PHASE_MIX]
                       [--phase-random-seed PHASE_RANDOM_SEED]
                       [--onset-time-credit]
                       [--onset-credit-pull ONSET_CREDIT_PULL]
                       [--onset-credit-max ONSET_CREDIT_MAX]
                       [--no-onset-realign] [--no-center]
                       [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                       [--operator {chordmapper,inharmonator}]
                       [--root-hz ROOT_HZ]
                       [--chord {aug,dim,dom7,maj7,major,min7,minor,power,sus2,sus4}]
                       [--strength STRENGTH]
                       [--tolerance-cents TOLERANCE_CENTS]
                       [--boost-db BOOST_DB] [--attenuation ATTENUATION]
                       [--inharmonic-f0-hz INHARMONIC_F0_HZ]
                       [--inharmonicity INHARMONICITY]
                       [--inharmonic-mix INHARMONIC_MIX] [--dry-mix DRY_MIX]
                       inputs [inputs ...]

Harmonic/chord mapping and inharmonic spectral warping.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _harm)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for
... [truncated]
```

### Module Docstring

```text
PVC-inspired harmonic/chord spectral mapping CLI.
```

## `src/pvx/cli/pvxharmonize.py`

**Purpose:** Multi-voice harmonizer built from phase-vocoder pitch shifts.

**Classes:** None
**Functions:** `pan_stereo`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxharmonize --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxharmonize [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                       [--suffix SUFFIX]
                                       [--output-format OUTPUT_FORMAT]
                                       [--stdout] [--overwrite] [--dry-run]
                                       [--verbosity {silent,quiet,normal,verbose,debug}]
                                       [-v] [--quiet] [--silent]
                                       [--normalize {none,peak,rms}]
                                       [--peak-dbfs PEAK_DBFS]
                                       [--rms-dbfs RMS_DBFS]
                                       [--target-lufs TARGET_LUFS]
                                       [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                       [--compressor-ratio COMPRESSOR_RATIO]
                                       [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                       [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                       [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                       [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                       [--expander-ratio EXPANDER_RATIO]
                                       [--expander-attack-ms EXPANDER_ATTACK_MS]
                                       [--expander-release-ms EXPANDER_RELEASE_MS]
                                       [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                       [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                       [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                       [--compander-attack-ms COMPANDER_ATTACK_MS]
                                       [--compander-release-ms COMPANDER_RELEASE_MS]
                                       [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                       [--limiter-threshold LIMITER_THRESHOLD]
                                       [--soft-clip-level SOFT_CLIP_LEVEL]
                                       [--soft-clip-type {tanh,arctan,cubic}]
                                       [--soft-clip-drive SOFT_CLIP_DRIVE]
                                       [--hard-clip-level HARD_CLIP_LEVEL]
                                       [--clip] [--subtype SUBTYPE]
                                       [--bit-depth {inherit,16,24,32f}]
                                       [--dither {none,tpdf}]
                                       [--dither-seed DITHER_SEED]
                                       [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                       [--metadata-policy {none,sidecar,copy}]
                                       [--n-fft N_FFT]
                                       [--win-length WIN_LENGTH]
                                       [--hop-size HOP_SIZE]
                                       [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                       [--kaiser-beta KAISER_BETA]
                                       [--transform {fft,dft,czt,dct,dst,hartley}]
                                       [--phase-engine {propagate,hybrid,random}]
                                       [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                       [--phase-random-seed PHASE_RANDOM_SEED]
                                       [--onset-time-credit]
                                       [--onset-credit-pull ONSET_CREDIT_PULL]
                                       [--onset-credit-max ONSET_CREDIT_MAX]
                                       [--no-onset-realign] [--no-center]
                                       [--device {auto,cpu,cuda}]
                                       [--cuda-device CUDA_DEVICE]
                                       [--intervals INTERVALS]
                                       [--intervals-cents INTERVALS_CENTS]
                                       [--gains GAINS] [--pans PANS]
                                       [--force-stereo]
                                       [--resample-mode {auto,fft,linear}]
                                       inputs [inputs ...]

Generate harmonized voices from an input

positional arguments:
  inputs                Input files/globs or '-' f
... [truncated]
```

### Module Docstring

```text
Multi-voice harmonizer built from phase-vocoder pitch shifts.
```

## `src/pvx/cli/pvxinharmonator.py`

**Purpose:** Inharmonator wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxinharmonator --help`

### CLI Help Snapshot

```text
usage: pvx inharmonator [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                        [--suffix SUFFIX] [--output-format OUTPUT_FORMAT]
                        [--stdout] [--overwrite] [--dry-run]
                        [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                        [--quiet] [--silent] [--normalize {none,peak,rms}]
                        [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                        [--target-lufs TARGET_LUFS]
                        [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                        [--compressor-ratio COMPRESSOR_RATIO]
                        [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                        [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                        [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                        [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                        [--expander-ratio EXPANDER_RATIO]
                        [--expander-attack-ms EXPANDER_ATTACK_MS]
                        [--expander-release-ms EXPANDER_RELEASE_MS]
                        [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                        [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                        [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                        [--compander-attack-ms COMPANDER_ATTACK_MS]
                        [--compander-release-ms COMPANDER_RELEASE_MS]
                        [--compander-makeup-db COMPANDER_MAKEUP_DB]
                        [--limiter-threshold LIMITER_THRESHOLD]
                        [--soft-clip-level SOFT_CLIP_LEVEL]
                        [--soft-clip-type {tanh,arctan,cubic}]
                        [--soft-clip-drive SOFT_CLIP_DRIVE]
                        [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                        [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                        [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                        [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                        [--metadata-policy {none,sidecar,copy}]
                        [--n-fft N_FFT] [--win-length WIN_LENGTH]
                        [--hop-size HOP_SIZE]
                        [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                        [--kaiser-beta KAISER_BETA]
                        [--transform {fft,dft,czt,dct,dst,hartley}]
                        [--phase-engine {propagate,hybrid,random}]
                        [--ambient-phase-mix AMBIENT_PHASE_MIX]
                        [--phase-random-seed PHASE_RANDOM_SEED]
                        [--onset-time-credit]
                        [--onset-credit-pull ONSET_CREDIT_PULL]
                        [--onset-credit-max ONSET_CREDIT_MAX]
                        [--no-onset-realign] [--no-center]
                        [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                        [--operator {chordmapper,inharmonator}]
                        [--root-hz ROOT_HZ]
                        [--chord {aug,dim,dom7,maj7,major,min7,minor,power,sus2,sus4}]
                        [--strength STRENGTH]
                        [--tolerance-cents TOLERANCE_CENTS]
                        [--boost-db BOOST_DB] [--attenuation ATTENUATION]
                        [--inharmonic-f0-hz INHARMONIC_F0_HZ]
                        [--inharmonicity INHARMONICITY]
                        [--inharmonic-mix INHARMONIC_MIX] [--dry-mix DRY_MIX]
                        inputs [inputs ...]

Harmonic/chord mapping and inharmonic spectral warping.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _harm)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        C
... [truncated]
```

### Module Docstring

```text
Inharmonator wrapper.
```

## `src/pvx/cli/pvxlayer.py`

**Purpose:** Layered harmonic/percussive processing with independent controls.

**Classes:** None
**Functions:** `hpss_masks`, `split_hpss`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxlayer --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxlayer [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                   [--suffix SUFFIX]
                                   [--output-format OUTPUT_FORMAT] [--stdout]
                                   [--overwrite] [--dry-run]
                                   [--verbosity {silent,quiet,normal,verbose,debug}]
                                   [-v] [--quiet] [--silent]
                                   [--normalize {none,peak,rms}]
                                   [--peak-dbfs PEAK_DBFS]
                                   [--rms-dbfs RMS_DBFS]
                                   [--target-lufs TARGET_LUFS]
                                   [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                   [--compressor-ratio COMPRESSOR_RATIO]
                                   [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                   [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                   [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                   [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                   [--expander-ratio EXPANDER_RATIO]
                                   [--expander-attack-ms EXPANDER_ATTACK_MS]
                                   [--expander-release-ms EXPANDER_RELEASE_MS]
                                   [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                   [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                   [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                   [--compander-attack-ms COMPANDER_ATTACK_MS]
                                   [--compander-release-ms COMPANDER_RELEASE_MS]
                                   [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                   [--limiter-threshold LIMITER_THRESHOLD]
                                   [--soft-clip-level SOFT_CLIP_LEVEL]
                                   [--soft-clip-type {tanh,arctan,cubic}]
                                   [--soft-clip-drive SOFT_CLIP_DRIVE]
                                   [--hard-clip-level HARD_CLIP_LEVEL]
                                   [--clip] [--subtype SUBTYPE]
                                   [--bit-depth {inherit,16,24,32f}]
                                   [--dither {none,tpdf}]
                                   [--dither-seed DITHER_SEED]
                                   [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                   [--metadata-policy {none,sidecar,copy}]
                                   [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                   [--hop-size HOP_SIZE]
                                   [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                   [--kaiser-beta KAISER_BETA]
                                   [--transform {fft,dft,czt,dct,dst,hartley}]
                                   [--phase-engine {propagate,hybrid,random}]
                                   [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                   [--phase-random-seed PHASE_RANDOM_SEED]
                                   [--onset-time-credit]
                                   [--onset-credit-pull ONSET_CREDIT_PULL]
                                   [--onset-credit-max ONSET_CREDIT_MAX]
                                   [--no-onset-realign] [--no-center]
                                   [--device {auto,cpu,cuda}]
                                   [--cuda-device CUDA_DEVICE]
                                   [--harmonic-stretch HARMONIC_STRETCH]
                                   [--harmonic-pitch-semitones HARMONIC_PITCH_SEMITONES]
                                   [--harmonic-pitch-cents HARMONIC_PITCH_CENTS]
                                   [--percussive-stretch PERCUSSIVE_STRETCH]
                                   [--percussive-pitch-semitones PERCUSSIVE_PITCH_SEMITONES]
                                   [--percussive-pitch-cents PERCUSSIVE_PITCH_CENTS]
                                   [--harmonic-gain HARMONIC_GAIN]
                                   [--percussive-gain PERCUSSIVE_GAIN]
                                   [--harmonic-kernel HARMONIC_KERNEL]
                                   [--percus
... [truncated]
```

### Module Docstring

```text
Layered harmonic/percussive processing with independent controls.
```

## `src/pvx/cli/pvxmorph.py`

**Purpose:** Spectral morphing between two input files.

**Classes:** `MorphControlSignal`
**Functions:** `_normalize_control_points`, `_parse_csv_control_points`, `_parse_json_control_points`, `_load_control_signal`, `_parse_scalar_or_control`, `_sample_cubic_local`, `_smoothstep`, `_smootherstep`, `_exp_ease`, `_sample_piecewise_ease`, `_sample_control_signal`, `match_channels`, `_phase_blend`, `_resolve_phase_mix_curve`, `_safe_rms`, `_framewise_envelope`, `_mask_from_modulator`, `morph_pair`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxmorph --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxmorph [-h] [-o OUTPUT] [--stdout]
                                   [--output-format OUTPUT_FORMAT]
                                   [--alpha ALPHA]
                                   [--blend-mode {linear,geometric,magnitude_b_phase_a,magnitude_a_phase_b,carrier_a_envelope_b,carrier_b_envelope_a,carrier_a_mask_b,carrier_b_mask_a,product,max_mag,min_mag}]
                                   [--phase-mix PHASE_MIX]
                                   [--interp {none,linear,nearest,cubic,polynomial,exponential,s_curve,smootherstep}]
                                   [--order ORDER]
                                   [--mask-exponent MASK_EXPONENT]
                                   [--envelope-lifter ENVELOPE_LIFTER]
                                   [--normalize-energy]
                                   [--normalize {none,peak,rms}]
                                   [--peak-dbfs PEAK_DBFS]
                                   [--rms-dbfs RMS_DBFS]
                                   [--target-lufs TARGET_LUFS]
                                   [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                   [--compressor-ratio COMPRESSOR_RATIO]
                                   [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                   [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                   [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                   [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                   [--expander-ratio EXPANDER_RATIO]
                                   [--expander-attack-ms EXPANDER_ATTACK_MS]
                                   [--expander-release-ms EXPANDER_RELEASE_MS]
                                   [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                   [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                   [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                   [--compander-attack-ms COMPANDER_ATTACK_MS]
                                   [--compander-release-ms COMPANDER_RELEASE_MS]
                                   [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                   [--limiter-threshold LIMITER_THRESHOLD]
                                   [--soft-clip-level SOFT_CLIP_LEVEL]
                                   [--soft-clip-type {tanh,arctan,cubic}]
                                   [--soft-clip-drive SOFT_CLIP_DRIVE]
                                   [--hard-clip-level HARD_CLIP_LEVEL]
                                   [--clip] [--subtype SUBTYPE]
                                   [--bit-depth {inherit,16,24,32f}]
                                   [--dither {none,tpdf}]
                                   [--dither-seed DITHER_SEED]
                                   [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                   [--metadata-policy {none,sidecar,copy}]
                                   [--overwrite]
                                   [--verbosity {silent,quiet,normal,verbose,debug}]
                                   [-v] [--quiet] [--silent] [--n-fft N_FFT]
                                   [--win-length WIN_LENGTH]
                                   [--hop-size HOP_SIZE]
                                   [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                   [--kaiser-beta KAISER_BETA]
                                   [--transform {fft,dft,czt,dct,dst,hartley}]
                                   [--phase-engine {propagate,hybrid,random}]
                                   [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                   [--phase-random-seed PHASE_RANDOM_SEED]
                                   [--onset-time-credit]
                                   [--onset-credit-pull ONSET_CREDIT_PULL]
                                   [--onset-credit-max ONSET_CREDIT_MAX]
                                   [--no-onset-realign] [--no-center]
                                   [--device {auto,cpu,cuda}]
                                   [--cuda-device CUDA_DEVICE]
                                   input_a input_b

Morph two audio files in the STFT domain

positional arguments:
  input_a               Input A path or 
... [truncated]
```

### Module Docstring

```text
Spectral morphing between two input files.
```

## `src/pvx/cli/pvxnoise.py`

**Purpose:** pvxnoise — add synthetic noise to audio at a controlled SNR.

**Classes:** None
**Functions:** `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxnoise --help`

### CLI Help Snapshot

```text
usage: pvxnoise [-h] --output OUTPUT [--snr SNR]
                [--noise-type {white,pink,brown,gaussian,bandlimited}]
                [--band BAND] [--background-dir BACKGROUND_DIR] [--seed SEED]
                [--bit-depth {16,24,32,float}]
                input

Add synthetic or background noise to audio at a controlled SNR.

positional arguments:
  input                 Input audio file

options:
  -h, --help            show this help message and exit
  --output, -o OUTPUT   Output audio file
  --snr SNR             Signal-to-noise ratio in dB. Single value for fixed
                        SNR or 'min,max' for uniform sampling. (default: 20)
  --noise-type {white,pink,brown,gaussian,bandlimited}
                        Noise type (default: white)
  --band BAND           Low,high frequency band in Hz for bandlimited noise
                        (default: 300,4000)
  --background-dir BACKGROUND_DIR
                        Directory of background audio files to mix instead of
                        synthetic noise
  --seed SEED           Random seed (default: 0)
  --bit-depth {16,24,32,float}
                        Output bit depth (default: 24)

pvxnoise — add synthetic noise to audio at a controlled SNR.

Supports white, pink, brown, and band-limited noise injection with
precise signal-to-noise ratio control.  Useful for building robust ASR,
speaker verification, and speech enhancement datasets.

Examples
--------
# Add pink noise at 20 dB SNR:
pvxnoise speech.wav --snr 20 --noise-type pink --output speech_noisy.wav

# Random SNR between 5-30 dB, white noise:
pvxnoise speech.wav --snr 5,30 --noise-type white --output speech_noisy.wav

# Band-limited noise (telephone band):
pvxnoise speech.wav --snr 15 --noise-type bandlimited --band 300,3400 --output out.wav

# Background noise mixing from a directory:
pvxnoise speech.wav --background-dir noise_samples/ --snr 10,25 --output out.wav
```

### Module Docstring

```text
pvxnoise — add synthetic noise to audio at a controlled SNR.

Supports white, pink, brown, and band-limited noise injection with
precise signal-to-noise ratio control.  Useful for building robust ASR,
speaker verification, and speech enhancement datasets.

Examples
--------
# Add pink noise at 20 dB SNR:
pvxnoise speech.wav --snr 20 --noise-type pink --output speech_noisy.wav

# Random SNR between 5-30 dB, white noise:
pvxnoise speech.wav --snr 5,30 --noise-type white --output speech_noisy.wav

# Band-limited noise (telephone band):
pvxnoise speech.wav --snr 15 --noise-type bandlimited --band 300,3400 --output out.wav

# Background noise mixing from a directory:
pvxnoise speech.wav --background-dir noise_samples/ --snr 10,25 --output out.wav
```

## `src/pvx/cli/pvxnoisefilter.py`

**Purpose:** Response-profile noise filter wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxnoisefilter --help`

### CLI Help Snapshot

```text
usage: pvx noisefilter [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                       [--suffix SUFFIX] [--output-format OUTPUT_FORMAT]
                       [--stdout] [--overwrite] [--dry-run]
                       [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                       [--quiet] [--silent] [--normalize {none,peak,rms}]
                       [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                       [--target-lufs TARGET_LUFS]
                       [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                       [--compressor-ratio COMPRESSOR_RATIO]
                       [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                       [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                       [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                       [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                       [--expander-ratio EXPANDER_RATIO]
                       [--expander-attack-ms EXPANDER_ATTACK_MS]
                       [--expander-release-ms EXPANDER_RELEASE_MS]
                       [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                       [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                       [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                       [--compander-attack-ms COMPANDER_ATTACK_MS]
                       [--compander-release-ms COMPANDER_RELEASE_MS]
                       [--compander-makeup-db COMPANDER_MAKEUP_DB]
                       [--limiter-threshold LIMITER_THRESHOLD]
                       [--soft-clip-level SOFT_CLIP_LEVEL]
                       [--soft-clip-type {tanh,arctan,cubic}]
                       [--soft-clip-drive SOFT_CLIP_DRIVE]
                       [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                       [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                       [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                       [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                       [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                       [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                       [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                       [--kaiser-beta KAISER_BETA]
                       [--transform {fft,dft,czt,dct,dst,hartley}]
                       [--phase-engine {propagate,hybrid,random}]
                       [--ambient-phase-mix AMBIENT_PHASE_MIX]
                       [--phase-random-seed PHASE_RANDOM_SEED]
                       [--onset-time-credit]
                       [--onset-credit-pull ONSET_CREDIT_PULL]
                       [--onset-credit-max ONSET_CREDIT_MAX]
                       [--no-onset-realign] [--no-center]
                       [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                       [--operator {filter,tvfilter,noisefilter,bandamp,spec-compander}]
                       --response RESPONSE [--response-mix RESPONSE_MIX]
                       [--dry-mix DRY_MIX]
                       [--response-gain-db RESPONSE_GAIN_DB]
                       [--transpose-semitones TRANSPOSE_SEMITONES]
                       [--shift-bins SHIFT_BINS] [--tv-map TV_MAP]
                       [--tv-key TV_KEY]
                       [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                       [--tv-order TV_ORDER] [--noise-floor NOISE_FLOOR]
                       [--band-gain-db BAND_GAIN_DB]
                       [--band-width-bins BAND_WIDTH_BINS]
                       [--peak-count PEAK_COUNT]
                       [--comp-threshold-db COMP_THRESHOLD_DB]
                       [--comp-ratio COMP_RATIO] [--expand-ratio EXPAND_RATIO]
                       inputs [inputs ...]

Response-driven spectral processing using PVXRF artifacts.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _filter)
  --output-format OUTPUT_FORMAT
                        Output extension/f
... [truncated]
```

### Module Docstring

```text
Response-profile noise filter wrapper.
```

## `src/pvx/cli/pvxreshape.py`

**Purpose:** PVC-style function-stream reshaper for control maps.

**Classes:** None
**Functions:** `build_parser`, `_resolve_output_format`, `_resolve_input_format`, `_render_payload`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxreshape --help`

### CLI Help Snapshot

```text
usage: pvx reshape [-h] [--key KEY] [--output-key OUTPUT_KEY]
                   --operation {scale,offset,clip,pow,normalize,invert,smooth,time-scale,time-shift,resample}
                   [--factor FACTOR] [--offset OFFSET] [--min MIN_VALUE]
                   [--max MAX_VALUE] [--exponent EXPONENT] [--window WINDOW]
                   [--target-min TARGET_MIN] [--target-max TARGET_MAX]
                   [--rate RATE]
                   [--interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                   [--order ORDER] [--input-format {auto,csv,json}]
                   [--format {auto,csv,json}] [--output OUTPUT] [--stdout]
                   [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                   [--quiet] [--silent]
                   input

Transform control-rate CSV/JSON maps for pvx routing and modulation workflows.

positional arguments:
  input                 Input control map path or '-' for stdin

options:
  -h, --help            show this help message and exit
  --key KEY             Control column/key to read (default: value)
  --output-key OUTPUT_KEY
                        Output control key. Default: same as --key
  --operation {scale,offset,clip,pow,normalize,invert,smooth,time-scale,time-shift,resample}
                        Reshape operation
  --factor FACTOR       Scale factor for scale/time-scale ops
  --offset OFFSET       Offset for offset/time-shift ops
  --min MIN_VALUE       Minimum clamp value
  --max MAX_VALUE       Maximum clamp value
  --exponent EXPONENT   Exponent for pow operation
  --window WINDOW       Window size for smooth operation
  --target-min TARGET_MIN
                        Target minimum for normalize operation
  --target-max TARGET_MAX
                        Target maximum for normalize operation
  --rate RATE           Resample control rate (Hz) for resample operation
  --interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}
                        Interpolation mode
  --order ORDER         Polynomial order for --interp polynomial
  --input-format {auto,csv,json}
                        Input format override (default: auto by suffix or csv for stdin)
  --format {auto,csv,json}
                        Output format (default: auto from --output suffix or csv for stdout)
  --output, --out OUTPUT
                        Output file path. Default: stdout.
  --stdout              Write reshaped map to stdout
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output

Examples:
  pvx reshape stretch_env.csv --key stretch --operation scale --factor 1.2 --output stretch_scaled.csv
  pvx reshape map.csv --key pitch_ratio --operation clip --min 0.5 --max 2.0 --stdout | pvx voc input.wav --pitch-map-stdin --route pitch_ratio=pitch_ratio --output out.wav
  pvx reshape alpha_curve.csv --operation resample --rate 50 --interp polynomial --order 5 --output alpha_curve_dense.csv
  pvx reshape stretch_env.csv --operation smooth --window 9 --output stretch_smooth.csv

Notes:
  - Use --key to select which control column to reshape.
  - For stdin input, pass '-' as INPUT and optionally set --input-format.
```

### Module Docstring

```text
PVC-style function-stream reshaper for control maps.
```

## `src/pvx/cli/pvxresponse.py`

**Purpose:** Create and inspect reusable frequency-response artifacts (PVXRF).

**Classes:** None
**Functions:** `_normalize_argv`, `_default_output_path`, `_print_summary`, `_write_json`, `build_parser`, `_run_create`, `_run_inspect`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxresponse --help`

### CLI Help Snapshot

```text
usage: pvx response [-h] {create,inspect} ...

Create and inspect PVXRF frequency-response artifacts derived from PVXAN analyses.

positional arguments:
  {create,inspect}
    create          Derive response artifact from analysis artifact
    inspect         Inspect existing PVXRF artifact

options:
  -h, --help        show this help message and exit

Examples:
  pvx response create input.pvxan.npz --output input.pvxrf.npz --method median --normalize peak
  pvx response inspect input.pvxrf.npz
  pvxresponse input.pvxan.npz --phase-mode zero --normalize rms

Notes:
  - When command is omitted, create mode is assumed for convenience.
  - PVXRF stores per-channel response magnitude/phase vectors and frequency bins.
```

### Module Docstring

```text
Create and inspect reusable frequency-response artifacts (PVXRF).
```

## `src/pvx/cli/pvxretune.py`

**Purpose:** Monophonic retuning with phase-vocoder segment processing.

**Classes:** None
**Functions:** `freq_to_midi`, `midi_to_freq`, `normalize_octave_cents`, `nearest_scale_freq`, `overlap_add`, `collect_f0_values`, `recommend_root_hz`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxretune --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxretune [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                    [--suffix SUFFIX]
                                    [--output-format OUTPUT_FORMAT] [--stdout]
                                    [--overwrite] [--dry-run]
                                    [--verbosity {silent,quiet,normal,verbose,debug}]
                                    [-v] [--quiet] [--silent]
                                    [--normalize {none,peak,rms}]
                                    [--peak-dbfs PEAK_DBFS]
                                    [--rms-dbfs RMS_DBFS]
                                    [--target-lufs TARGET_LUFS]
                                    [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                    [--compressor-ratio COMPRESSOR_RATIO]
                                    [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                    [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                    [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                    [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                    [--expander-ratio EXPANDER_RATIO]
                                    [--expander-attack-ms EXPANDER_ATTACK_MS]
                                    [--expander-release-ms EXPANDER_RELEASE_MS]
                                    [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                    [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                    [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                    [--compander-attack-ms COMPANDER_ATTACK_MS]
                                    [--compander-release-ms COMPANDER_RELEASE_MS]
                                    [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                    [--limiter-threshold LIMITER_THRESHOLD]
                                    [--soft-clip-level SOFT_CLIP_LEVEL]
                                    [--soft-clip-type {tanh,arctan,cubic}]
                                    [--soft-clip-drive SOFT_CLIP_DRIVE]
                                    [--hard-clip-level HARD_CLIP_LEVEL]
                                    [--clip] [--subtype SUBTYPE]
                                    [--bit-depth {inherit,16,24,32f}]
                                    [--dither {none,tpdf}]
                                    [--dither-seed DITHER_SEED]
                                    [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                    [--metadata-policy {none,sidecar,copy}]
                                    [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                    [--hop-size HOP_SIZE]
                                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                    [--kaiser-beta KAISER_BETA]
                                    [--transform {fft,dft,czt,dct,dst,hartley}]
                                    [--phase-engine {propagate,hybrid,random}]
                                    [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                    [--phase-random-seed PHASE_RANDOM_SEED]
                                    [--onset-time-credit]
                                    [--onset-credit-pull ONSET_CREDIT_PULL]
                                    [--onset-credit-max ONSET_CREDIT_MAX]
                                    [--no-onset-realign] [--no-center]
                                    [--device {auto,cpu,cuda}]
                                    [--cuda-device CUDA_DEVICE] [--root ROOT]
                                    [--root-hz ROOT_HZ] [--recommend-root]
                                    [--scale {chromatic,major,minor,pentatonic}]
                                    [--scale-cents SCALE_CENTS]
                                    [--strength STRENGTH]
                                    [--chunk-ms CHUNK_MS]
                                    [--overlap-ms OVERLAP_MS]
                                    [--a4-reference-hz A4_REFERENCE_HZ]
                                    [--f0-min F0_MIN] [--f0-max F0_MAX]
                                    [--resample-mode {auto,fft,linear}]
                                    inputs [inputs ...]

Monophonic retun
... [truncated]
```

### Module Docstring

```text
Monophonic retuning with phase-vocoder segment processing.
```

## `src/pvx/cli/pvxring.py`

**Purpose:** PVC-inspired ring and resonator operators.

**Classes:** None
**Functions:** `build_parser`, `run_ring_cli`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxring --help`

### CLI Help Snapshot

```text
usage: pvx ring [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
                [--output-format OUTPUT_FORMAT] [--stdout] [--overwrite]
                [--dry-run] [--verbosity {silent,quiet,normal,verbose,debug}]
                [-v] [--quiet] [--silent] [--normalize {none,peak,rms}]
                [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                [--target-lufs TARGET_LUFS]
                [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                [--compressor-ratio COMPRESSOR_RATIO]
                [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                [--expander-ratio EXPANDER_RATIO]
                [--expander-attack-ms EXPANDER_ATTACK_MS]
                [--expander-release-ms EXPANDER_RELEASE_MS]
                [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                [--compander-attack-ms COMPANDER_ATTACK_MS]
                [--compander-release-ms COMPANDER_RELEASE_MS]
                [--compander-makeup-db COMPANDER_MAKEUP_DB]
                [--limiter-threshold LIMITER_THRESHOLD]
                [--soft-clip-level SOFT_CLIP_LEVEL]
                [--soft-clip-type {tanh,arctan,cubic}]
                [--soft-clip-drive SOFT_CLIP_DRIVE]
                [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                [--kaiser-beta KAISER_BETA]
                [--transform {fft,dft,czt,dct,dst,hartley}]
                [--phase-engine {propagate,hybrid,random}]
                [--ambient-phase-mix AMBIENT_PHASE_MIX]
                [--phase-random-seed PHASE_RANDOM_SEED] [--onset-time-credit]
                [--onset-credit-pull ONSET_CREDIT_PULL]
                [--onset-credit-max ONSET_CREDIT_MAX] [--no-onset-realign]
                [--no-center] [--device {auto,cpu,cuda}]
                [--cuda-device CUDA_DEVICE]
                [--operator {ring,ringfilter,ringtvfilter}]
                [--frequency-hz FREQUENCY_HZ] [--depth DEPTH] [--mix MIX]
                [--feedback FEEDBACK] [--resonance-hz RESONANCE_HZ]
                [--resonance-q RESONANCE_Q] [--resonance-mix RESONANCE_MIX]
                [--resonance-decay RESONANCE_DECAY] [--tv-map TV_MAP]
                [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                [--tv-order TV_ORDER]
                inputs [inputs ...]

Ring modulation and resonator filtering with PVC-style controls.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _ring)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization mode
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS when --normalize peak
  --rms-dbfs RMS_DBFS   Target RMS dBFS when --normalize rms
  --target-lufs TARGET_LUFS
             
... [truncated]
```

### Module Docstring

```text
PVC-inspired ring and resonator operators.
```

## `src/pvx/cli/pvxringfilter.py`

**Purpose:** Ring + resonator filter wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxringfilter --help`

### CLI Help Snapshot

```text
usage: pvx ringfilter [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
                      [--output-format OUTPUT_FORMAT] [--stdout] [--overwrite]
                      [--dry-run]
                      [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                      [--quiet] [--silent] [--normalize {none,peak,rms}]
                      [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                      [--target-lufs TARGET_LUFS]
                      [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                      [--compressor-ratio COMPRESSOR_RATIO]
                      [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                      [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                      [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                      [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                      [--expander-ratio EXPANDER_RATIO]
                      [--expander-attack-ms EXPANDER_ATTACK_MS]
                      [--expander-release-ms EXPANDER_RELEASE_MS]
                      [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                      [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                      [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                      [--compander-attack-ms COMPANDER_ATTACK_MS]
                      [--compander-release-ms COMPANDER_RELEASE_MS]
                      [--compander-makeup-db COMPANDER_MAKEUP_DB]
                      [--limiter-threshold LIMITER_THRESHOLD]
                      [--soft-clip-level SOFT_CLIP_LEVEL]
                      [--soft-clip-type {tanh,arctan,cubic}]
                      [--soft-clip-drive SOFT_CLIP_DRIVE]
                      [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                      [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                      [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                      [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                      [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                      [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                      [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                      [--kaiser-beta KAISER_BETA]
                      [--transform {fft,dft,czt,dct,dst,hartley}]
                      [--phase-engine {propagate,hybrid,random}]
                      [--ambient-phase-mix AMBIENT_PHASE_MIX]
                      [--phase-random-seed PHASE_RANDOM_SEED]
                      [--onset-time-credit]
                      [--onset-credit-pull ONSET_CREDIT_PULL]
                      [--onset-credit-max ONSET_CREDIT_MAX]
                      [--no-onset-realign] [--no-center]
                      [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                      [--operator {ring,ringfilter,ringtvfilter}]
                      [--frequency-hz FREQUENCY_HZ] [--depth DEPTH]
                      [--mix MIX] [--feedback FEEDBACK]
                      [--resonance-hz RESONANCE_HZ]
                      [--resonance-q RESONANCE_Q]
                      [--resonance-mix RESONANCE_MIX]
                      [--resonance-decay RESONANCE_DECAY] [--tv-map TV_MAP]
                      [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                      [--tv-order TV_ORDER]
                      inputs [inputs ...]

Ring modulation and resonator filtering with PVC-style controls.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _ring)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail
... [truncated]
```

### Module Docstring

```text
Ring + resonator filter wrapper.
```

## `src/pvx/cli/pvxringtvfilter.py`

**Purpose:** Time-varying ring + resonator filter wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxringtvfilter --help`

### CLI Help Snapshot

```text
usage: pvx ringtvfilter [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                        [--suffix SUFFIX] [--output-format OUTPUT_FORMAT]
                        [--stdout] [--overwrite] [--dry-run]
                        [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                        [--quiet] [--silent] [--normalize {none,peak,rms}]
                        [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                        [--target-lufs TARGET_LUFS]
                        [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                        [--compressor-ratio COMPRESSOR_RATIO]
                        [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                        [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                        [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                        [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                        [--expander-ratio EXPANDER_RATIO]
                        [--expander-attack-ms EXPANDER_ATTACK_MS]
                        [--expander-release-ms EXPANDER_RELEASE_MS]
                        [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                        [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                        [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                        [--compander-attack-ms COMPANDER_ATTACK_MS]
                        [--compander-release-ms COMPANDER_RELEASE_MS]
                        [--compander-makeup-db COMPANDER_MAKEUP_DB]
                        [--limiter-threshold LIMITER_THRESHOLD]
                        [--soft-clip-level SOFT_CLIP_LEVEL]
                        [--soft-clip-type {tanh,arctan,cubic}]
                        [--soft-clip-drive SOFT_CLIP_DRIVE]
                        [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                        [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                        [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                        [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                        [--metadata-policy {none,sidecar,copy}]
                        [--n-fft N_FFT] [--win-length WIN_LENGTH]
                        [--hop-size HOP_SIZE]
                        [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                        [--kaiser-beta KAISER_BETA]
                        [--transform {fft,dft,czt,dct,dst,hartley}]
                        [--phase-engine {propagate,hybrid,random}]
                        [--ambient-phase-mix AMBIENT_PHASE_MIX]
                        [--phase-random-seed PHASE_RANDOM_SEED]
                        [--onset-time-credit]
                        [--onset-credit-pull ONSET_CREDIT_PULL]
                        [--onset-credit-max ONSET_CREDIT_MAX]
                        [--no-onset-realign] [--no-center]
                        [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                        [--operator {ring,ringfilter,ringtvfilter}]
                        [--frequency-hz FREQUENCY_HZ] [--depth DEPTH]
                        [--mix MIX] [--feedback FEEDBACK]
                        [--resonance-hz RESONANCE_HZ]
                        [--resonance-q RESONANCE_Q]
                        [--resonance-mix RESONANCE_MIX]
                        [--resonance-decay RESONANCE_DECAY] [--tv-map TV_MAP]
                        [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                        [--tv-order TV_ORDER]
                        inputs [inputs ...]

Ring modulation and resonator filtering with PVC-style controls.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _ring)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,
... [truncated]
```

### Module Docstring

```text
Time-varying ring + resonator filter wrapper.
```

## `src/pvx/cli/pvxrir.py`

**Purpose:** pvxrir — simulate room acoustics via room impulse response convolution.

**Classes:** None
**Functions:** `build_parser`, `_parse_range`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxrir --help`

### CLI Help Snapshot

```text
usage: pvxrir [-h] --output OUTPUT [--rt60 RT60] [--wet WET] [--drr DRR]
              [--pre-delay-ms PRE_DELAY_MS] [--ir-file IR_FILE]
              [--ir-dir IR_DIR] [--ir-database ID] [--category CATEGORY]
              [--list-databases] [--cache-dir CACHE_DIR] [--no-trim]
              [--seed SEED] [--bit-depth {16,24,32,float}]
              input

Simulate room acoustics via impulse response convolution.

positional arguments:
  input                 Input audio file

options:
  -h, --help            show this help message and exit
  --output, -o OUTPUT   Output audio file
  --rt60 RT60           Reverberation time in seconds. Single value or
                        'min,max' range (default: 0.2,1.5)
  --wet WET             Wet/dry mix 0-1. Single value or 'min,max' range
                        (default: 0.3,0.7)
  --drr DRR             Direct-to-reverb ratio in dB. Single value or
                        'min,max' range (default: 3,12)
  --pre-delay-ms PRE_DELAY_MS
                        Pre-delay in ms before the reverberant tail (default:
                        5.0)
  --ir-file IR_FILE     Path to an impulse response WAV file (overrides
                        synthetic generation)
  --ir-dir IR_DIR       Directory of impulse response files (one is chosen
                        randomly)
  --ir-database ID      Use IRs from a curated database (auto-downloads on
                        first use). Available: echothief, mit_kemar
  --category CATEGORY   Filter IRs by category when using --ir-database (e.g.,
                        hall, church, outdoor, room, large)
  --list-databases      List available IR databases and exit
  --cache-dir CACHE_DIR
                        Cache directory for downloaded IR databases (default:
                        ~/.pvx/ir_cache)
  --no-trim             Do not trim output to input length (include reverb
                        tail)
  --seed SEED           Random seed (default: 0)
  --bit-depth {16,24,32,float}
                        Output bit depth (default: 24)

pvxrir — simulate room acoustics via room impulse response convolution.

Convolves audio with a synthetic or user-provided room impulse response
to simulate the acoustic characteristics of different room environments.
Useful for building reverb-robust ASR, speaker verification, and source
separation datasets.

Examples
--------
# Synthetic room: RT60 between 0.3 and 1.2 s
pvxrir speech.wav --rt60 0.3,1.2 --wet 0.4,0.7 --output speech_reverb.wav

# Fixed reverb parameters:
pvxrir speech.wav --rt60 0.5 --wet 0.6 --output speech_reverb.wav

# Convolve with a real IR file:
pvxrir speech.wav --ir-file concert_hall.wav --wet 0.5 --output out.wav

# Convolve with a random IR from a directory:
pvxrir speech.wav --ir-dir irs/ --wet 0.3,0.8 --output out.wav

# Use a curated IR database (auto-downloads on first use):
pvxrir speech.wav --ir-database echothief --wet 0.5 --output out.wav

# Filter by room category:
pvxrir speech.wav --ir-database echothief --category hall --wet 0.6 --output out.wav

# List available databases:
pvxrir input.wav --list-databases --output /dev/null
```

### Module Docstring

```text
pvxrir — simulate room acoustics via room impulse response convolution.

Convolves audio with a synthetic or user-provided room impulse response
to simulate the acoustic characteristics of different room environments.
Useful for building reverb-robust ASR, speaker verification, and source
separation datasets.

Examples
--------
# Synthetic room: RT60 between 0.3 and 1.2 s
pvxrir speech.wav --rt60 0.3,1.2 --wet 0.4,0.7 --output speech_reverb.wav

# Fixed reverb parameters:
pvxrir speech.wav --rt60 0.5 --wet 0.6 --output speech_reverb.wav

# Convolve with a real IR file:
pvxrir speech.wav --ir-file concert_hall.wav --wet 0.5 --output out.wav

# Convolve with a random IR from a directory:
pvxrir speech.wav --ir-dir irs/ --wet 0.3,0.8 --output out.wav

# Use a curated IR database (auto-downloads on first use):
pvxrir speech.wav --ir-database echothief --wet 0.5 --output out.wav

# Filter by room category:
pvxrir speech.wav --ir-database echothief --category hall --wet 0.6 --output out.wav

# List available databases:
pvxrir input.wav --list-databases --output /dev/null
```

## `src/pvx/cli/pvxspecaugment.py`

**Purpose:** pvxspecaugment — apply SpecAugment frequency and time masking.

**Classes:** None
**Functions:** `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxspecaugment --help`

### CLI Help Snapshot

```text
usage: pvxspecaugment [-h] --output OUTPUT [--freq-mask FREQ_MASK]
                      [--time-mask TIME_MASK]
                      [--num-freq-masks NUM_FREQ_MASKS]
                      [--num-time-masks NUM_TIME_MASKS] [--fill FILL]
                      [--n-fft N_FFT] [--hop-length HOP_LENGTH] [--seed SEED]
                      [--bit-depth {16,24,32,float}]
                      input

Apply SpecAugment frequency and time masking to audio.

positional arguments:
  input                 Input audio file

options:
  -h, --help            show this help message and exit
  --output, -o OUTPUT   Output audio file
  --freq-mask FREQ_MASK
                        Maximum frequency mask width F in bins (default: 27)
  --time-mask TIME_MASK
                        Maximum time mask width T in frames (default: 100)
  --num-freq-masks NUM_FREQ_MASKS
                        Number of frequency masks (default: 2)
  --num-time-masks NUM_TIME_MASKS
                        Number of time masks (default: 2)
  --fill FILL           Fill value for masked regions: 0, mean, or a float
                        (default: 0)
  --n-fft N_FFT         FFT size (default: 512)
  --hop-length HOP_LENGTH
                        STFT hop length in samples (default: 128)
  --seed SEED           Random seed (default: 0)
  --bit-depth {16,24,32,float}
                        Output bit depth (default: 24)

pvxspecaugment — apply SpecAugment frequency and time masking.

Implements the SpecAugment policy (Park et al., 2019) for training robust
speech and audio models.  Applies random frequency and time masks directly
to the STFT magnitude spectrum, then reconstructs the waveform.

References
----------
Park, D. S., et al. (2019). "SpecAugment: A Simple Data Augmentation
Method for Automatic Speech Recognition." Interspeech.
https://arxiv.org/abs/1904.08779

Examples
--------
# Standard SpecAugment LB policy:
pvxspecaugment speech.wav --freq-mask 30 --time-mask 40 --num-masks 2 --output out.wav

# LD policy (fewer, narrower masks):
pvxspecaugment speech.wav --freq-mask 27 --time-mask 70 --num-masks 1 --output out.wav

# LibriSpeech LB policy:
pvxspecaugment speech.wav --freq-mask 30 --time-mask 100 --num-freq-masks 2 --num-time-masks 2 --output out.wav
```

### Module Docstring

```text
pvxspecaugment — apply SpecAugment frequency and time masking.

Implements the SpecAugment policy (Park et al., 2019) for training robust
speech and audio models.  Applies random frequency and time masks directly
to the STFT magnitude spectrum, then reconstructs the waveform.

References
----------
Park, D. S., et al. (2019). "SpecAugment: A Simple Data Augmentation
Method for Automatic Speech Recognition." Interspeech.
https://arxiv.org/abs/1904.08779

Examples
--------
# Standard SpecAugment LB policy:
pvxspecaugment speech.wav --freq-mask 30 --time-mask 40 --num-masks 2 --output out.wav

# LD policy (fewer, narrower masks):
pvxspecaugment speech.wav --freq-mask 27 --time-mask 70 --num-masks 1 --output out.wav

# LibriSpeech LB policy:
pvxspecaugment speech.wav --freq-mask 30 --time-mask 100 --num-freq-masks 2 --num-time-masks 2 --output out.wav
```

## `src/pvx/cli/pvxspeccompander.py`

**Purpose:** Response-referenced spectral compander wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxspeccompander --help`

### CLI Help Snapshot

```text
usage: pvx spec-compander [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                          [--suffix SUFFIX] [--output-format OUTPUT_FORMAT]
                          [--stdout] [--overwrite] [--dry-run]
                          [--verbosity {silent,quiet,normal,verbose,debug}]
                          [-v] [--quiet] [--silent]
                          [--normalize {none,peak,rms}]
                          [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                          [--target-lufs TARGET_LUFS]
                          [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                          [--compressor-ratio COMPRESSOR_RATIO]
                          [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                          [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                          [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                          [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                          [--expander-ratio EXPANDER_RATIO]
                          [--expander-attack-ms EXPANDER_ATTACK_MS]
                          [--expander-release-ms EXPANDER_RELEASE_MS]
                          [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                          [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                          [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                          [--compander-attack-ms COMPANDER_ATTACK_MS]
                          [--compander-release-ms COMPANDER_RELEASE_MS]
                          [--compander-makeup-db COMPANDER_MAKEUP_DB]
                          [--limiter-threshold LIMITER_THRESHOLD]
                          [--soft-clip-level SOFT_CLIP_LEVEL]
                          [--soft-clip-type {tanh,arctan,cubic}]
                          [--soft-clip-drive SOFT_CLIP_DRIVE]
                          [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                          [--subtype SUBTYPE]
                          [--bit-depth {inherit,16,24,32f}]
                          [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                          [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                          [--metadata-policy {none,sidecar,copy}]
                          [--n-fft N_FFT] [--win-length WIN_LENGTH]
                          [--hop-size HOP_SIZE]
                          [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                          [--kaiser-beta KAISER_BETA]
                          [--transform {fft,dft,czt,dct,dst,hartley}]
                          [--phase-engine {propagate,hybrid,random}]
                          [--ambient-phase-mix AMBIENT_PHASE_MIX]
                          [--phase-random-seed PHASE_RANDOM_SEED]
                          [--onset-time-credit]
                          [--onset-credit-pull ONSET_CREDIT_PULL]
                          [--onset-credit-max ONSET_CREDIT_MAX]
                          [--no-onset-realign] [--no-center]
                          [--device {auto,cpu,cuda}]
                          [--cuda-device CUDA_DEVICE]
                          [--operator {filter,tvfilter,noisefilter,bandamp,spec-compander}]
                          --response RESPONSE [--response-mix RESPONSE_MIX]
                          [--dry-mix DRY_MIX]
                          [--response-gain-db RESPONSE_GAIN_DB]
                          [--transpose-semitones TRANSPOSE_SEMITONES]
                          [--shift-bins SHIFT_BINS] [--tv-map TV_MAP]
                          [--tv-key TV_KEY]
                          [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                          [--tv-order TV_ORDER] [--noise-floor NOISE_FLOOR]
                          [--band-gain-db BAND_GAIN_DB]
                          [--band-width-bins BAND_WIDTH_BINS]
                          [--peak-count PEAK_COUNT]
                          [--comp-threshold-db COMP_THRESHOLD_DB]
                          [--comp-ratio COMP_RATIO]
                          [--expand-ratio EXPAND_RATIO]
                          inputs [inputs ...]

Response-driven spectral processing using PVXRF artifacts.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OU
... [truncated]
```

### Module Docstring

```text
Response-referenced spectral compander wrapper.
```

## `src/pvx/cli/pvxtrajectoryreverb.py`

**Purpose:** Trajectory-aware multichannel convolution reverb for mono sources.

**Classes:** None
**Functions:** `build_parser`, `_to_mono`, `_normalize_coordinate_option_args`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxtrajectoryreverb --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxtrajectoryreverb [-h] [-o OUTPUT_DIR]
                                              [--output OUTPUT]
                                              [--suffix SUFFIX]
                                              [--output-format OUTPUT_FORMAT]
                                              [--stdout] [--overwrite]
                                              [--dry-run]
                                              [--verbosity {silent,quiet,normal,verbose,debug}]
                                              [-v] [--quiet] [--silent]
                                              [--normalize {none,peak,rms}]
                                              [--peak-dbfs PEAK_DBFS]
                                              [--rms-dbfs RMS_DBFS]
                                              [--target-lufs TARGET_LUFS]
                                              [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                              [--compressor-ratio COMPRESSOR_RATIO]
                                              [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                              [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                              [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                              [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                              [--expander-ratio EXPANDER_RATIO]
                                              [--expander-attack-ms EXPANDER_ATTACK_MS]
                                              [--expander-release-ms EXPANDER_RELEASE_MS]
                                              [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                              [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                              [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                              [--compander-attack-ms COMPANDER_ATTACK_MS]
                                              [--compander-release-ms COMPANDER_RELEASE_MS]
                                              [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                              [--limiter-threshold LIMITER_THRESHOLD]
                                              [--soft-clip-level SOFT_CLIP_LEVEL]
                                              [--soft-clip-type {tanh,arctan,cubic}]
                                              [--soft-clip-drive SOFT_CLIP_DRIVE]
                                              [--hard-clip-level HARD_CLIP_LEVEL]
                                              [--clip] [--subtype SUBTYPE]
                                              [--bit-depth {inherit,16,24,32f}]
                                              [--dither {none,tpdf}]
                                              [--dither-seed DITHER_SEED]
                                              [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                              [--metadata-policy {none,sidecar,copy}]
                                              --ir IR
                                              [--coord-system {cartesian,spherical}]
                                              --start START --end END
                                              [--speaker-angles SPEAKER_ANGLES]
                                              [--trajectory-shape {linear,ease-in,ease-out,ease-in-out}]
                                              [--distance-law {none,inverse,inverse-square}]
                                              [--normalize-gains]
                                              [--no-normalize-gains]
                                              [--wet WET] [--dry DRY]
                                              inputs [inputs ...]

Convolve mono source with multichannel impulse response and move source from A to B.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _trajrev)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        
... [truncated]
```

### Module Docstring

```text
Trajectory-aware multichannel convolution reverb for mono sources.
```

## `src/pvx/cli/pvxtransient.py`

**Purpose:** Transient-aware time/pitch processing.

**Classes:** None
**Functions:** `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxtransient --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxtransient [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                       [--suffix SUFFIX]
                                       [--output-format OUTPUT_FORMAT]
                                       [--stdout] [--overwrite] [--dry-run]
                                       [--verbosity {silent,quiet,normal,verbose,debug}]
                                       [-v] [--quiet] [--silent]
                                       [--normalize {none,peak,rms}]
                                       [--peak-dbfs PEAK_DBFS]
                                       [--rms-dbfs RMS_DBFS]
                                       [--target-lufs TARGET_LUFS]
                                       [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                       [--compressor-ratio COMPRESSOR_RATIO]
                                       [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                       [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                       [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                       [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                       [--expander-ratio EXPANDER_RATIO]
                                       [--expander-attack-ms EXPANDER_ATTACK_MS]
                                       [--expander-release-ms EXPANDER_RELEASE_MS]
                                       [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                       [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                       [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                       [--compander-attack-ms COMPANDER_ATTACK_MS]
                                       [--compander-release-ms COMPANDER_RELEASE_MS]
                                       [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                       [--limiter-threshold LIMITER_THRESHOLD]
                                       [--soft-clip-level SOFT_CLIP_LEVEL]
                                       [--soft-clip-type {tanh,arctan,cubic}]
                                       [--soft-clip-drive SOFT_CLIP_DRIVE]
                                       [--hard-clip-level HARD_CLIP_LEVEL]
                                       [--clip] [--subtype SUBTYPE]
                                       [--bit-depth {inherit,16,24,32f}]
                                       [--dither {none,tpdf}]
                                       [--dither-seed DITHER_SEED]
                                       [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                       [--metadata-policy {none,sidecar,copy}]
                                       [--n-fft N_FFT]
                                       [--win-length WIN_LENGTH]
                                       [--hop-size HOP_SIZE]
                                       [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                       [--kaiser-beta KAISER_BETA]
                                       [--transform {fft,dft,czt,dct,dst,hartley}]
                                       [--phase-engine {propagate,hybrid,random}]
                                       [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                       [--phase-random-seed PHASE_RANDOM_SEED]
                                       [--onset-time-credit]
                                       [--onset-credit-pull ONSET_CREDIT_PULL]
                                       [--onset-credit-max ONSET_CREDIT_MAX]
                                       [--no-onset-realign] [--no-center]
                                       [--device {auto,cpu,cuda}]
                                       [--cuda-device CUDA_DEVICE]
                                       [--time-stretch TIME_STRETCH]
                                       [--target-duration TARGET_DURATION]
                                       [--pitch-shift-semitones PITCH_SHIFT_SEMITONES]
                                       [--pitch-shift-cents PITCH_SHIFT_CENTS]
                                       [--pitch-shift-ratio PITCH_SHIFT_RATIO]
                                       [--transient-threshold TRANSIENT_THRESHOLD]
                                       [
... [truncated]
```

### Module Docstring

```text
Transient-aware time/pitch processing.
```

## `src/pvx/cli/pvxtvfilter.py`

**Purpose:** Time-varying response filter wrapper.

**Classes:** None
**Functions:** `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxtvfilter --help`

### CLI Help Snapshot

```text
usage: pvx tvfilter [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
                    [--output-format OUTPUT_FORMAT] [--stdout] [--overwrite]
                    [--dry-run]
                    [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                    [--quiet] [--silent] [--normalize {none,peak,rms}]
                    [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS]
                    [--target-lufs TARGET_LUFS]
                    [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                    [--compressor-ratio COMPRESSOR_RATIO]
                    [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                    [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                    [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                    [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                    [--expander-ratio EXPANDER_RATIO]
                    [--expander-attack-ms EXPANDER_ATTACK_MS]
                    [--expander-release-ms EXPANDER_RELEASE_MS]
                    [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                    [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                    [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                    [--compander-attack-ms COMPANDER_ATTACK_MS]
                    [--compander-release-ms COMPANDER_RELEASE_MS]
                    [--compander-makeup-db COMPANDER_MAKEUP_DB]
                    [--limiter-threshold LIMITER_THRESHOLD]
                    [--soft-clip-level SOFT_CLIP_LEVEL]
                    [--soft-clip-type {tanh,arctan,cubic}]
                    [--soft-clip-drive SOFT_CLIP_DRIVE]
                    [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                    [--subtype SUBTYPE] [--bit-depth {inherit,16,24,32f}]
                    [--dither {none,tpdf}] [--dither-seed DITHER_SEED]
                    [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                    [--metadata-policy {none,sidecar,copy}] [--n-fft N_FFT]
                    [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                    [--kaiser-beta KAISER_BETA]
                    [--transform {fft,dft,czt,dct,dst,hartley}]
                    [--phase-engine {propagate,hybrid,random}]
                    [--ambient-phase-mix AMBIENT_PHASE_MIX]
                    [--phase-random-seed PHASE_RANDOM_SEED]
                    [--onset-time-credit]
                    [--onset-credit-pull ONSET_CREDIT_PULL]
                    [--onset-credit-max ONSET_CREDIT_MAX] [--no-onset-realign]
                    [--no-center] [--device {auto,cpu,cuda}]
                    [--cuda-device CUDA_DEVICE]
                    [--operator {filter,tvfilter,noisefilter,bandamp,spec-compander}]
                    --response RESPONSE [--response-mix RESPONSE_MIX]
                    [--dry-mix DRY_MIX] [--response-gain-db RESPONSE_GAIN_DB]
                    [--transpose-semitones TRANSPOSE_SEMITONES]
                    [--shift-bins SHIFT_BINS] [--tv-map TV_MAP]
                    [--tv-key TV_KEY]
                    [--tv-interp {none,stairstep,nearest,linear,cubic,polynomial,exponential,s_curve,smootherstep}]
                    [--tv-order TV_ORDER] [--noise-floor NOISE_FLOOR]
                    [--band-gain-db BAND_GAIN_DB]
                    [--band-width-bins BAND_WIDTH_BINS]
                    [--peak-count PEAK_COUNT]
                    [--comp-threshold-db COMP_THRESHOLD_DB]
                    [--comp-ratio COMP_RATIO] [--expand-ratio EXPAND_RATIO]
                    inputs [inputs ...]

Response-driven spectral processing using PVXRF artifacts.

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _filter)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve a
... [truncated]
```

### Module Docstring

```text
Time-varying response filter wrapper.
```

## `src/pvx/cli/pvxunison.py`

**Purpose:** Create unison width via micro-detuned phase-vocoder voices.

**Classes:** None
**Functions:** `cents_to_ratio`, `pan_gains`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxunison --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxunison [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                    [--suffix SUFFIX]
                                    [--output-format OUTPUT_FORMAT] [--stdout]
                                    [--overwrite] [--dry-run]
                                    [--verbosity {silent,quiet,normal,verbose,debug}]
                                    [-v] [--quiet] [--silent]
                                    [--normalize {none,peak,rms}]
                                    [--peak-dbfs PEAK_DBFS]
                                    [--rms-dbfs RMS_DBFS]
                                    [--target-lufs TARGET_LUFS]
                                    [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                    [--compressor-ratio COMPRESSOR_RATIO]
                                    [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                    [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                    [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                    [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                    [--expander-ratio EXPANDER_RATIO]
                                    [--expander-attack-ms EXPANDER_ATTACK_MS]
                                    [--expander-release-ms EXPANDER_RELEASE_MS]
                                    [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                    [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                    [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                    [--compander-attack-ms COMPANDER_ATTACK_MS]
                                    [--compander-release-ms COMPANDER_RELEASE_MS]
                                    [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                    [--limiter-threshold LIMITER_THRESHOLD]
                                    [--soft-clip-level SOFT_CLIP_LEVEL]
                                    [--soft-clip-type {tanh,arctan,cubic}]
                                    [--soft-clip-drive SOFT_CLIP_DRIVE]
                                    [--hard-clip-level HARD_CLIP_LEVEL]
                                    [--clip] [--subtype SUBTYPE]
                                    [--bit-depth {inherit,16,24,32f}]
                                    [--dither {none,tpdf}]
                                    [--dither-seed DITHER_SEED]
                                    [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                    [--metadata-policy {none,sidecar,copy}]
                                    [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                    [--hop-size HOP_SIZE]
                                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                    [--kaiser-beta KAISER_BETA]
                                    [--transform {fft,dft,czt,dct,dst,hartley}]
                                    [--phase-engine {propagate,hybrid,random}]
                                    [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                    [--phase-random-seed PHASE_RANDOM_SEED]
                                    [--onset-time-credit]
                                    [--onset-credit-pull ONSET_CREDIT_PULL]
                                    [--onset-credit-max ONSET_CREDIT_MAX]
                                    [--no-onset-realign] [--no-center]
                                    [--device {auto,cpu,cuda}]
                                    [--cuda-device CUDA_DEVICE]
                                    [--voices VOICES]
                                    [--detune-cents DETUNE_CENTS]
                                    [--width WIDTH] [--dry-mix DRY_MIX]
                                    [--resample-mode {auto,fft,linear}]
                                    inputs [inputs ...]

Stereo unison thickener

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX      
... [truncated]
```

### Module Docstring

```text
Create unison width via micro-detuned phase-vocoder voices.
```

## `src/pvx/cli/pvxvoc.py`

**Purpose:** CLI entrypoint wrapper for the phase-vocoder tool.

**Classes:** None
**Functions:** None

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxvoc --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxvoc [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                                 [--output-format OUTPUT_FORMAT]
                                 [--out OUTPUT] [--overwrite] [--dry-run]
                                 [--stdout]
                                 [--verbosity {silent,quiet,normal,verbose,debug}]
                                 [-v] [--quiet] [--silent]
                                 [--preset {none,default,vocal,ambient,extreme,vocal_studio,drums_safe,extreme_ambient,stereo_coherent}]
                                 [--example {all,basic,vocal,ambient,extreme,drums_safe,stereo_coherent,hybrid,benchmark,gpu,pipeline,csv}]
                                 [--guided] [--stretch STRETCH] [--gpu]
                                 [--cpu]
                                 [--quality-profile {neutral,speech,music,percussion,ambient,extreme}]
                                 [--auto-profile]
                                 [--auto-profile-lookahead-seconds AUTO_PROFILE_LOOKAHEAD_SECONDS]
                                 [--auto-transform] [--n-fft N_FFT]
                                 [--win-length WIN_LENGTH]
                                 [--hop-size HOP_SIZE]
                                 [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                 [--kaiser-beta KAISER_BETA]
                                 [--transform {fft,dft,czt,dct,dst,hartley}]
                                 [--no-center]
                                 [--phase-locking {off,identity}]
                                 [--phase-engine {propagate,hybrid,random}]
                                 [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                 [--phase-random-seed PHASE_RANDOM_SEED]
                                 [--transient-preserve]
                                 [--transient-threshold TRANSIENT_THRESHOLD]
                                 [--fourier-sync]
                                 [--fourier-sync-min-fft FOURIER_SYNC_MIN_FFT]
                                 [--fourier-sync-max-fft FOURIER_SYNC_MAX_FFT]
                                 [--fourier-sync-smooth FOURIER_SYNC_SMOOTH]
                                 [--multires-fusion]
                                 [--multires-ffts MULTIRES_FFTS]
                                 [--multires-weights MULTIRES_WEIGHTS]
                                 [--device {auto,cpu,cuda}]
                                 [--cuda-device CUDA_DEVICE]
                                 [--time-stretch TIME_STRETCH]
                                 [--target-duration TARGET_DURATION]
                                 [--stretch-mode {auto,standard,multistage}]
                                 [--extreme-time-stretch]
                                 [--extreme-stretch-threshold EXTREME_STRETCH_THRESHOLD]
                                 [--max-stage-stretch MAX_STAGE_STRETCH]
                                 [--onset-time-credit]
                                 [--onset-credit-pull ONSET_CREDIT_PULL]
                                 [--onset-credit-max ONSET_CREDIT_MAX]
                                 [--no-onset-realign] [--ambient-preset]
                                 [--auto-segment-seconds AUTO_SEGMENT_SECONDS]
                                 [--checkpoint-dir CHECKPOINT_DIR]
                                 [--checkpoint-id CHECKPOINT_ID] [--resume]
                                 [--interp {none,linear,nearest,cubic,polynomial,exponential,s_curve,smootherstep}]
                                 [--order ORDER]
                                 [--transient-mode {off,reset,hybrid,wsola}]
                                 [--transient-sensitivity TRANSIENT_SENSITIVITY]
                                 [--transient-protect-ms TRANSIENT_PROTECT_MS]
                                 [--transient-crossfade-ms TRANSIENT_CROSSFADE_MS]
                                 [--stereo-mode {independent,mid_side_lock,ref_channel_lock}]
                                 [--ref-channel REF_CHANNEL]
                                 [--coherence-strength COHERENCE_STRENGTH]
                                 [--pitch-shift-semitones PITCH_SHIFT_SEMITONES |
                                 --pitch-shift-cents PITCH_SHIFT_CENTS |
                                 --pitch-shift-ratio PITCH_SHIFT_RATIO |
    
... [truncated]
```

### Module Docstring

```text
CLI entrypoint wrapper for the phase-vocoder tool.
```

## `src/pvx/cli/pvxwarp.py`

**Purpose:** Time-warp an input according to a user-provided stretch map.

**Classes:** None
**Functions:** `fill_stretch_segments`, `build_parser`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.cli.pvxwarp --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.cli.pvxwarp [-h] [-o OUTPUT_DIR] [--output OUTPUT]
                                  [--suffix SUFFIX]
                                  [--output-format OUTPUT_FORMAT] [--stdout]
                                  [--overwrite] [--dry-run]
                                  [--verbosity {silent,quiet,normal,verbose,debug}]
                                  [-v] [--quiet] [--silent]
                                  [--normalize {none,peak,rms}]
                                  [--peak-dbfs PEAK_DBFS]
                                  [--rms-dbfs RMS_DBFS]
                                  [--target-lufs TARGET_LUFS]
                                  [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                                  [--compressor-ratio COMPRESSOR_RATIO]
                                  [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                                  [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                                  [--compressor-makeup-db COMPRESSOR_MAKEUP_DB]
                                  [--expander-threshold-db EXPANDER_THRESHOLD_DB]
                                  [--expander-ratio EXPANDER_RATIO]
                                  [--expander-attack-ms EXPANDER_ATTACK_MS]
                                  [--expander-release-ms EXPANDER_RELEASE_MS]
                                  [--compander-threshold-db COMPANDER_THRESHOLD_DB]
                                  [--compander-compress-ratio COMPANDER_COMPRESS_RATIO]
                                  [--compander-expand-ratio COMPANDER_EXPAND_RATIO]
                                  [--compander-attack-ms COMPANDER_ATTACK_MS]
                                  [--compander-release-ms COMPANDER_RELEASE_MS]
                                  [--compander-makeup-db COMPANDER_MAKEUP_DB]
                                  [--limiter-threshold LIMITER_THRESHOLD]
                                  [--soft-clip-level SOFT_CLIP_LEVEL]
                                  [--soft-clip-type {tanh,arctan,cubic}]
                                  [--soft-clip-drive SOFT_CLIP_DRIVE]
                                  [--hard-clip-level HARD_CLIP_LEVEL] [--clip]
                                  [--subtype SUBTYPE]
                                  [--bit-depth {inherit,16,24,32f}]
                                  [--dither {none,tpdf}]
                                  [--dither-seed DITHER_SEED]
                                  [--true-peak-max-dbtp TRUE_PEAK_MAX_DBTP]
                                  [--metadata-policy {none,sidecar,copy}]
                                  [--n-fft N_FFT] [--win-length WIN_LENGTH]
                                  [--hop-size HOP_SIZE]
                                  [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                                  [--kaiser-beta KAISER_BETA]
                                  [--transform {fft,dft,czt,dct,dst,hartley}]
                                  [--phase-engine {propagate,hybrid,random}]
                                  [--ambient-phase-mix AMBIENT_PHASE_MIX]
                                  [--phase-random-seed PHASE_RANDOM_SEED]
                                  [--onset-time-credit]
                                  [--onset-credit-pull ONSET_CREDIT_PULL]
                                  [--onset-credit-max ONSET_CREDIT_MAX]
                                  [--no-onset-realign] [--no-center]
                                  [--device {auto,cpu,cuda}]
                                  [--cuda-device CUDA_DEVICE] --map MAP
                                  [--crossfade-ms CROSSFADE_MS]
                                  [--resample-mode {auto,fft,linear}]
                                  inputs [inputs ...]

Apply variable time-stretch map from CSV

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only). Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _warp)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping); require
... [truncated]
```

### Module Docstring

```text
Time-warp an input according to a user-provided stretch map.
```

## `src/pvx/core/__init__.py`

**Purpose:** Core DSP/runtime internals shared by pvx CLI tools.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Core DSP/runtime internals shared by pvx CLI tools.
```

## `src/pvx/core/analysis_store.py`

**Purpose:** Persistent phase-vocoder analysis artifact storage.

**Classes:** `AnalysisArtifact`
**Functions:** `_utc_now_iso`, `_as_complex_spectrum`, `_canonical_meta`, `analysis_digest`, `summarize_analysis_artifact`, `analyze_audio`, `save_analysis_artifact`, `load_analysis_artifact`

### Module Docstring

```text
Persistent phase-vocoder analysis artifact storage.

PVXAN schema:
- container: NumPy NPZ (compressed)
- required members:
  - meta_json: UTF-8 JSON metadata payload
  - spectrum_real: float64 array, shape (channels, frames, bins)
  - spectrum_imag: float64 array, shape (channels, frames, bins)
```

## `src/pvx/core/attribution.py`

**Purpose:** Centralized attribution text shared across pvx code and documentation.

**Classes:** None
**Functions:** `python_header_reference`, `markdown_notice`, `html_notice`

### Module Docstring

```text
Centralized attribution text shared across pvx code and documentation.
```

## `src/pvx/core/audio_metrics.py`

**Purpose:** Shared audio metric summaries and ASCII table rendering.

**Classes:** `AudioMetricSummary`
**Functions:** `_to_mono`, `_to_2d`, `_resample_1d_linear`, `_resample_audio_linear`, `_match_length`, `_principal_angle`, `_stft_complex`, `_stft_mag_db`, `_onset_envelope`, `_dbfs`, `_spectral_centroid_and_bw95`, `summarize_audio_metrics`, `_format_float`, `_ascii_table`, `render_audio_metrics_table`, `summarize_audio_comparison_metrics`, `render_audio_comparison_table`

### Module Docstring

```text
Shared audio metric summaries and ASCII table rendering.
```

## `src/pvx/core/common.py`

**Purpose:** Shared helpers for pvx DSP command-line tools.

**Classes:** `SegmentSpec`, `StatusBar`
**Functions:** `add_console_args`, `build_examples_epilog`, `console_level`, `is_quiet`, `is_silent`, `log_message`, `log_error`, `build_status_bar`, `add_common_io_args`, `add_output_policy_args`, `add_vocoder_args`, `build_vocoder_config`, `validate_vocoder_args`, `resolve_inputs`, `read_audio`, `finalize_audio`, `write_output`, `print_input_output_metrics_table`, `default_output_path`, `_stream_format_name`, `parse_float_list`, `semitone_to_ratio`, `cents_to_ratio`, `time_pitch_shift_channel`, `time_pitch_shift_audio`, `read_segment_csv`, `concat_with_crossfade`, `ensure_runtime`

### Module Docstring

```text
Shared helpers for pvx DSP command-line tools.
```

## `src/pvx/core/control_bus.py`

**Purpose:** Control-bus routing helpers for time-varying CSV maps.

**Classes:** `ControlRoute`
**Functions:** `normalize_control_name`, `_parse_finite_float`, `_parse_signal_name`, `parse_control_route`, `parse_control_routes`, `_source_column_candidates`, `_parse_row_float`, `_read_source_value`, `_eval_route`, `apply_control_routes_csv`

### Module Docstring

```text
Control-bus routing helpers for time-varying CSV maps.
```

## `src/pvx/core/feature_tracking.py`

**Purpose:** Frame-level feature tracking for control-rate audio modulation maps.

**Classes:** None
**Functions:** `_safe_div`, `_hz_to_mel`, `_mel_to_hz`, `_mel_filterbank`, `_dct_type2`, `_frame`, `_acf_peak_ratio`, `_estimate_formants_lpc`, `_estimate_tempo_bpm`, `_estimate_inharmonicity`, `extract_feature_tracks`, `feature_subset`, `as_serializable_columns`

### Module Docstring

```text
Frame-level feature tracking for control-rate audio modulation maps.
```

## `src/pvx/core/output_policy.py`

**Purpose:** Shared output policy helpers for bit depth, dither, true-peak, and metadata sidecars.

**Classes:** None
**Functions:** `db_to_amplitude`, `_resample_linear_1d`, `true_peak_dbtp`, `enforce_true_peak_limit`, `resolve_output_subtype`, `subtype_bit_depth`, `apply_dither_if_needed`, `prepare_output_audio`, `source_metadata`, `write_metadata_sidecar`, `validate_output_policy_args`

### Module Docstring

```text
Shared output policy helpers for bit depth, dither, true-peak, and metadata sidecars.
```

## `src/pvx/core/presets.py`

**Purpose:** Preset definitions for pvx processing intent modes.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Preset definitions for pvx processing intent modes.
```

## `src/pvx/core/pvc_functions.py`

**Purpose:** PVC-style function-stream utilities for control-rate map authoring.

**Classes:** None
**Functions:** `_sanitize_times_values`, `_auto_format_from_path`, `parse_control_points_payload`, `load_control_points`, `dump_control_points_csv`, `dump_control_points_json`, `generate_envelope_points`, `_moving_average`, `reshape_control_points`

### Module Docstring

```text
PVC-style function-stream utilities for control-rate map authoring.

Phase 6 coverage:
- envelope: generate deterministic control trajectories
- reshape: transform existing control trajectories
```

## `src/pvx/core/pvc_harmony.py`

**Purpose:** PVC-inspired harmonic/chord spectral mapping for pvx.

**Classes:** None
**Functions:** `_coerce_audio`, `chord_mapper_mask`, `_inharmonic_inverse_map`, `_interp_mag_phase_from_freq`, `process_harmony_operator`

### Module Docstring

```text
PVC-inspired harmonic/chord spectral mapping for pvx.

Phase 5 coverage:
- chordmapper
- inharmonator
```

## `src/pvx/core/pvc_ops.py`

**Purpose:** PVC-inspired response-driven spectral operators for pvx.

**Classes:** None
**Functions:** `_piecewise_segment_fraction`, `_smoothstep`, `_smootherstep`, `_exp_ease`, `db_to_amp`, `_coerce_audio`, `_resize_curve`, `_shift_response_curve`, `_read_rows_from_map`, `load_scalar_control_points`, `evaluate_scalar_control`, `_frame_times`, `_blend_dry_wet`, `_compute_band_shape`, `process_response_operator`

### Module Docstring

```text
PVC-inspired response-driven spectral operators for pvx.

Phase 3 coverage:
- filter
- tvfilter
- noisefilter
- bandamp
- spec-compander
```

## `src/pvx/core/pvc_resonators.py`

**Purpose:** PVC-inspired ring/resonator operators for pvx.

**Classes:** None
**Functions:** `_coerce_audio`, `_sample_times`, `_ring_modulate`, `_resonant_peak_filter`, `process_ring_operator`

### Module Docstring

```text
PVC-inspired ring/resonator operators for pvx.

Phase 4 coverage:
- ring
- ringfilter
- ringtvfilter
```

## `src/pvx/core/response_store.py`

**Purpose:** Persistent frequency-response artifact storage.

**Classes:** `ResponseArtifact`
**Functions:** `_canonical_meta`, `_moving_average_1d`, `_aggregate_magnitude`, `_aggregate_phase`, `_normalize_magnitude`, `response_digest`, `summarize_response_artifact`, `response_from_analysis`, `save_response_artifact`, `load_response_artifact`

### Module Docstring

```text
Persistent frequency-response artifact storage.

PVXRF schema:
- container: NumPy NPZ (compressed)
- required members:
  - meta_json: UTF-8 JSON metadata payload
  - frequencies_hz: float64 array, shape (bins,)
  - magnitude: float64 array, shape (channels, bins)
  - phase: float64 array, shape (channels, bins)
```

## `src/pvx/core/spatial_reverb.py`

**Purpose:** Trajectory-aware multichannel convolution reverb helpers.

**Classes:** None
**Functions:** `_parse_float_triplet`, `parse_coordinate`, `parse_speaker_angles`, `default_speaker_angles`, `_angles_to_unit_vectors`, `_shape_curve`, `compute_trajectory_gains`, `_fft_convolve_or_fallback`, `apply_multichannel_trajectory_reverb`, `resample_audio_linear`

### Module Docstring

```text
Trajectory-aware multichannel convolution reverb helpers.
```

## `src/pvx/core/stereo.py`

**Purpose:** Stereo/multichannel helper utilities.

**Classes:** None
**Functions:** `validate_ref_channel`, `lr_to_ms`, `ms_to_lr`

### Module Docstring

```text
Stereo/multichannel helper utilities.
```

## `src/pvx/core/streaming.py`

**Purpose:** Stateful chunked streaming helpers for the unified pvx CLI.

**Classes:** None
**Functions:** `_read_audio`, `_build_config`, `_resolve_voc_args_for_stream`, `_chunk_core_extract`, `_concat_exact`, `run_stateful_stream`

### Module Docstring

```text
Stateful chunked streaming helpers for the unified pvx CLI.
```

## `src/pvx/core/transients.py`

**Purpose:** Transient analysis and segmentation helpers for hybrid pvx modes.

**Classes:** `TransientFeatures`, `TransientRegion`
**Functions:** `_principal`, `_normalize_robust`, `_frame_signal`, `compute_transient_features`, `pick_onset_frames`, `_mask_to_regions`, `_enforce_min_region_samples`, `build_transient_mask`, `map_mask_to_output`, `smooth_binary_mask`, `detect_transient_regions`

### Module Docstring

```text
Transient analysis and segmentation helpers for hybrid pvx modes.
```

## `src/pvx/core/voc.py`

**Purpose:** Multi-channel phase vocoder CLI for time and pitch manipulation.

**Classes:** `VocoderConfig`, `PitchConfig`, `ControlSegment`, `DynamicControlRef`, `DynamicControlSignal`, `JobResult`, `FourierSyncPlan`, `AudioBlockResult`, `RuntimeConfig`
**Functions:** `db_to_amplitude`, `cents_to_ratio`, `_eval_numeric_expr`, `parse_numeric_expression`, `parse_pitch_ratio_value`, `_is_power_of_two`, `parse_numeric_list`, `parse_int_list`, `_looks_like_control_signal_reference`, `looks_like_control_signal_reference`, `_parse_scalar_cli_value`, `_parse_int_cli_value`, `parse_int_cli_value`, `_parse_control_signal_value`, `_coerce_control_interp`, `_control_value_column_candidates`, `_deduplicate_points`, `_normalize_control_points`, `_parse_csv_control_points`, `_parse_json_control_points`, `load_dynamic_control_signal`, `_smoothstep`, `_smootherstep`, `_exp_ease`, `_piecewise_ease_sample`, `_sample_dynamic_signal`, `estimate_content_features`, `suggest_quality_profile`, `apply_quality_profile_overrides`, `resolve_transform_auto`, `_has_cupy`, `_is_cupy_array`, `_array_module`, `_to_numpy`, `_to_runtime_array`, `_as_float`, `_as_bool`, `_i0`, `normalize_transform_name`, `transform_bin_count`, `_analysis_angular_velocity`, `_transform_requires_scipy`, `ensure_transform_backend_available`, `validate_transform_available`, `_resize_or_pad_1d`, `_onesided_to_full_spectrum`, `_forward_transform_numpy`, `_inverse_transform_numpy`, `_forward_transform`, `_inverse_transform`, `add_runtime_args`, `runtime_config`, `configure_runtime`, `configure_runtime_from_args`, `ensure_runtime_dependencies`, `principal_angle`, `_cosine_series_window`, `_bartlett_window`, `_bohman_window`, `_cosine_window`, `_sine_window`, `_triangular_window`, `_bartlett_hann_window`, `_tukey_window`, `_parzen_window`, `_lanczos_window`, `_welch_window`, `_gaussian_window`, `_general_gaussian_window`, `_exponential_window`, `_cauchy_window`, `_cosine_power_window`, `_hann_poisson_window`, `_general_hamming_window`, `_kaiser_window`, `make_window`, `pad_for_framing`, `stft`, `istft`, `scaled_win_length`, `resize_spectrum_bins`, `smooth_series`, `regularize_frame_lengths`, `fill_nan_with_nearest`, `lock_fft_length_to_f0`, `build_fourier_sync_plan`, `compute_transient_flags`, `build_output_time_steps`, `create_phase_rng`, `draw_random_phase`, `apply_phase_engine`, `find_spectral_peaks`, `apply_identity_phase_locking`, `phase_vocoder_time_stretch`, `phase_vocoder_time_stretch_fourier_sync`, `compute_multistage_stretches`, `phase_vocoder_time_stretch_multistage`, `stretch_channel_with_strategy`, `phase_vocoder_time_stretch_multires_fusion`, `linear_resample_1d`, `resample_1d`, `force_length`, `estimate_f0_autocorrelation`, `normalize_audio`, `_envelope_coeff`, `_envelope_follower`, `_estimate_lufs_or_rms_db`, `_apply_compressor`, `_apply_expander`, `_apply_compander`, `_apply_limiter`, `_apply_soft_clip`, `add_mastering_args`, `validate_mastering_args`, `apply_mastering_chain`, `cepstral_envelope`, `apply_formant_preservation`, `choose_pitch_ratio`, `_parse_optional_float`, `parse_control_segments_csv`, `apply_control_confidence_policy`, `smooth_control_ratios`, `expand_control_segments`, `load_control_segments`, `_lock_channel_phase_to_reference`, `process_audio_block`, `resolve_base_stretch`, `build_vocoder_config_from_args`, `_finalize_dynamic_segment_values`, `build_dynamic_control_segments`, `compute_output_path`, `_stream_format_name`, `_read_audio_input`, `read_audio_input`, `_write_audio_output`, `concat_audio_chunks`, `build_uniform_control_segments`, `resolve_checkpoint_context`, `load_checkpoint_chunk`, `save_checkpoint_chunk`, `write_manifest`, `process_file`, `force_length_multi`, `resample_multi`, `validate_args`, `build_parser`, `run_guided_mode`, `expand_inputs`, `collect_cli_flags`, `apply_named_preset`, `main`

**Help commands:** `PYTHONPATH=src python3 -m pvx.core.voc --help`

### CLI Help Snapshot

```text
usage: python3 -m pvx.core.voc [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                               [--output-format OUTPUT_FORMAT] [--out OUTPUT]
                               [--overwrite] [--dry-run] [--stdout]
                               [--verbosity {silent,quiet,normal,verbose,debug}]
                               [-v] [--quiet] [--silent]
                               [--preset {none,default,vocal,ambient,extreme,vocal_studio,drums_safe,extreme_ambient,stereo_coherent}]
                               [--example {all,basic,vocal,ambient,extreme,drums_safe,stereo_coherent,hybrid,benchmark,gpu,pipeline,csv}]
                               [--guided] [--stretch STRETCH] [--gpu] [--cpu]
                               [--quality-profile {neutral,speech,music,percussion,ambient,extreme}]
                               [--auto-profile]
                               [--auto-profile-lookahead-seconds AUTO_PROFILE_LOOKAHEAD_SECONDS]
                               [--auto-transform] [--n-fft N_FFT]
                               [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                               [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                               [--kaiser-beta KAISER_BETA]
                               [--transform {fft,dft,czt,dct,dst,hartley}]
                               [--no-center] [--phase-locking {off,identity}]
                               [--phase-engine {propagate,hybrid,random}]
                               [--ambient-phase-mix AMBIENT_PHASE_MIX]
                               [--phase-random-seed PHASE_RANDOM_SEED]
                               [--transient-preserve]
                               [--transient-threshold TRANSIENT_THRESHOLD]
                               [--fourier-sync]
                               [--fourier-sync-min-fft FOURIER_SYNC_MIN_FFT]
                               [--fourier-sync-max-fft FOURIER_SYNC_MAX_FFT]
                               [--fourier-sync-smooth FOURIER_SYNC_SMOOTH]
                               [--multires-fusion]
                               [--multires-ffts MULTIRES_FFTS]
                               [--multires-weights MULTIRES_WEIGHTS]
                               [--device {auto,cpu,cuda}]
                               [--cuda-device CUDA_DEVICE]
                               [--time-stretch TIME_STRETCH]
                               [--target-duration TARGET_DURATION]
                               [--stretch-mode {auto,standard,multistage}]
                               [--extreme-time-stretch]
                               [--extreme-stretch-threshold EXTREME_STRETCH_THRESHOLD]
                               [--max-stage-stretch MAX_STAGE_STRETCH]
                               [--onset-time-credit]
                               [--onset-credit-pull ONSET_CREDIT_PULL]
                               [--onset-credit-max ONSET_CREDIT_MAX]
                               [--no-onset-realign] [--ambient-preset]
                               [--auto-segment-seconds AUTO_SEGMENT_SECONDS]
                               [--checkpoint-dir CHECKPOINT_DIR]
                               [--checkpoint-id CHECKPOINT_ID] [--resume]
                               [--interp {none,linear,nearest,cubic,polynomial,exponential,s_curve,smootherstep}]
                               [--order ORDER]
                               [--transient-mode {off,reset,hybrid,wsola}]
                               [--transient-sensitivity TRANSIENT_SENSITIVITY]
                               [--transient-protect-ms TRANSIENT_PROTECT_MS]
                               [--transient-crossfade-ms TRANSIENT_CROSSFADE_MS]
                               [--stereo-mode {independent,mid_side_lock,ref_channel_lock}]
                               [--ref-channel REF_CHANNEL]
                               [--coherence-strength COHERENCE_STRENGTH]
                               [--pitch-shift-semitones PITCH_SHIFT_SEMITONES |
                               --pitch-shift-cents PITCH_SHIFT_CENTS |
                               --pitch-shift-ratio PITCH_SHIFT_RATIO |
                               --target-f0 TARGET_F0]
                               [--analysis-channel {first,mix}]
                               [--f0-min F0_MIN] [--f0-max F0_MAX]
                               [--pitch-mode {standard,formant-
... [truncated]
```

### Module Docstring

```text
Multi-channel phase vocoder CLI for time and pitch manipulation.
```

## `src/pvx/core/voc_console.py`

**Purpose:** Console, preset, and example helpers for the pvx voc CLI.

**Classes:** `ProgressBar`
**Functions:** `add_console_args`, `console_level`, `is_quiet`, `is_silent`, `log_message`, `log_error`, `clone_args_namespace`, `collect_cli_flags`, `print_cli_examples`, `apply_named_preset`

### Module Docstring

```text
Console, preset, and example helpers for the pvx voc CLI.
```

## `src/pvx/core/voc_jobs.py`

**Purpose:** Job orchestration helpers for pvx voc renders.

**Classes:** None
**Functions:** `compute_output_path`, `_stream_format_name`, `_hash_file_contents`, `_metadata_sidecar_path`, `_checkpoint_chunk_meta_path`, `_read_audio_input`, `_write_audio_output`, `concat_audio_chunks`, `build_uniform_control_segments`, `_checkpoint_job_id`, `_render_fingerprint`, `resolve_checkpoint_context`, `load_checkpoint_chunk`, `save_checkpoint_chunk`, `write_manifest`, `_resolve_output_path`, `_ensure_writable_target`, `_preflight_file_writes`, `_consume_prefetched_audio`, `_preflight_checkpoint_targets`, `process_file`

### Module Docstring

```text
Job orchestration helpers for pvx voc renders.
```

## `src/pvx/core/wsola.py`

**Purpose:** Deterministic WSOLA time-stretch primitives for transient handling.

**Classes:** None
**Functions:** `_safe_window`, `wsola_time_stretch`

### Module Docstring

```text
Deterministic WSOLA time-stretch primitives for transient handling.
```

## `src/pvx/integrations/__init__.py`

**Purpose:** pvx.integrations — Framework adapters for pvx augmentation pipelines.

**Classes:** None
**Functions:** None

### Module Docstring

```text
pvx.integrations — Framework adapters for pvx augmentation pipelines.

Provides thin, zero-dependency-on-framework-in-core adapters so that
pvx augmentation pipelines slot cleanly into PyTorch, HuggingFace
datasets, and TensorFlow data pipelines.

Usage
-----
Each submodule guards its framework import so only the framework you
actually use needs to be installed::

    # PyTorch
    from pvx.integrations.pytorch import PvxAugmentDataset

    # HuggingFace
    from pvx.integrations.huggingface import make_augment_map_fn

    # TensorFlow
    from pvx.integrations.tensorflow import make_tf_augment_fn
```

## `src/pvx/integrations/huggingface.py`

**Purpose:** HuggingFace datasets integration for pvx augmentation pipelines.

**Classes:** `HFAugmentMapper`
**Functions:** `make_augment_map_fn`, `augment_dataset`

### Module Docstring

```text
HuggingFace datasets integration for pvx augmentation pipelines.

Provides ``map()``-compatible functions so that pvx augmentation slots
cleanly into HuggingFace ``datasets.Dataset`` workflows with full
reproducibility and multiprocessing support.

Requirements
------------
``datasets`` must be installed separately::

    pip install datasets

Usage
-----
>>> from datasets import load_dataset
>>> from pvx.augment import Pipeline, AddNoise, RoomSimulator
>>> from pvx.integrations.huggingface import make_augment_map_fn
>>>
>>> ds = load_dataset("speech_commands", split="train")
>>>
>>> pipeline = Pipeline([
...     AddNoise(snr_db=(10, 30), p=0.5),
...     RoomSimulator(rt60_range=(0.1, 0.8), p=0.4),
... ], seed=42)
>>>
>>> augment_fn = make_augment_map_fn(
...     pipeline,
...     audio_column="audio",
...     sr_column=None,      # infer from audio dict
...     output_column="audio_aug",
... )
>>> ds_aug = ds.map(augment_fn, batched=False, num_proc=4)
```

## `src/pvx/integrations/pytorch.py`

**Purpose:** PyTorch integration for pvx augmentation pipelines.

**Classes:** `PvxAugmentDataset`, `PvxAugmentTransform`, `AudioCollator`
**Functions:** `_require_torch`

### Module Docstring

```text
PyTorch integration for pvx augmentation pipelines.

Provides ``torch.utils.data.Dataset`` subclasses and collate helpers so
that pvx augmentation pipelines slot directly into PyTorch training loops
with no boilerplate.

Requirements
------------
``torch`` and ``torchaudio`` must be installed separately::

    pip install torch torchaudio

Usage
-----
>>> from pvx.augment import Pipeline, AddNoise, RoomSimulator, SpecAugment
>>> from pvx.integrations.pytorch import PvxAugmentDataset
>>>
>>> pipeline = Pipeline([
...     AddNoise(snr_db=(10, 30), p=0.5),
...     RoomSimulator(rt60_range=(0.1, 0.8), p=0.4),
...     SpecAugment(freq_mask_param=30, time_mask_param=50, p=0.5),
... ], seed=42)
>>>
>>> file_list = ["speech1.wav", "speech2.wav", ...]
>>> dataset = PvxAugmentDataset(
...     file_list,
...     pipeline=pipeline,
...     sample_rate=16000,
...     labels=labels,       # optional list of int labels
...     duration_s=3.0,      # optional fixed crop
... )
>>> loader = torch.utils.data.DataLoader(
...     dataset, batch_size=32, num_workers=4,
...     collate_fn=PvxAugmentDataset.collate_fn,
... )
```

## `src/pvx/integrations/tensorflow.py`

**Purpose:** TensorFlow / tf.data integration for pvx augmentation pipelines.

**Classes:** None
**Functions:** `_require_tf`, `make_tf_augment_fn`, `make_tf_augment_map_fn`, `pvx_augment_tf_dataset`

### Module Docstring

```text
TensorFlow / tf.data integration for pvx augmentation pipelines.

Provides helpers to wrap pvx augmentation pipelines inside ``tf.data``
pipelines using ``tf.py_function`` for gradient-free data augmentation.

Requirements
------------
``tensorflow`` must be installed separately::

    pip install tensorflow

Usage
-----
>>> import tensorflow as tf
>>> from pvx.augment import Pipeline, AddNoise, RoomSimulator
>>> from pvx.integrations.tensorflow import make_tf_augment_fn, pvx_augment_tf_dataset
>>>
>>> pipeline = Pipeline([
...     AddNoise(snr_db=(10, 30), p=0.5),
...     RoomSimulator(rt60_range=(0.1, 0.8), p=0.4),
... ], seed=42)
>>>
>>> # Wrap a tf.data dataset
>>> tf_dataset = tf.data.Dataset.from_tensor_slices(audio_array)
>>> tf_dataset = pvx_augment_tf_dataset(tf_dataset, pipeline, sample_rate=16000)
>>>
>>> # Or get a map function for manual use
>>> augment_fn = make_tf_augment_fn(pipeline, sample_rate=16000)
>>> tf_dataset = tf_dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
```

## `src/pvx/metrics/__init__.py`

**Purpose:** Objective metric helpers for pvx benchmarks and tests.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Objective metric helpers for pvx benchmarks and tests.
```

## `src/pvx/metrics/coherence.py`

**Purpose:** Inter-channel coherence drift metrics.

**Classes:** None
**Functions:** `_principal`, `_stft`, `interchannel_coherence_drift`, `stereo_coherence_drift_score`

### Module Docstring

```text
Inter-channel coherence drift metrics.
```

## `src/pvx/voc_cli.py`

**Purpose:** CLI entrypoint for the phase-vocoder tool.

**Classes:** None
**Functions:** `_prompt_text`, `_prompt_choice`, `run_guided_mode`, `expand_inputs`, `_build_explain_plan`, `_manifest_entries`, `_preflight_manifest_target`, `main`

**Help commands:** `python3 src/pvx/voc_cli.py`, `python3 src/pvx/voc_cli.py --help`

### Module Docstring

```text
CLI entrypoint for the phase-vocoder tool.
```

## `src/pvxalgorithms/__init__.py`

**Purpose:** Compatibility shim for `pvxalgorithms` namespace.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Compatibility shim for `pvxalgorithms` namespace.

Use `pvx.algorithms` as the canonical import path.
```

## `src/pvxalgorithms/base.py`

**Purpose:** Compatibility shim for `pvxalgorithms.base`.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Compatibility shim for `pvxalgorithms.base`.
```

## `src/pvxalgorithms/registry.py`

**Purpose:** Compatibility shim for `pvxalgorithms.registry`.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Compatibility shim for `pvxalgorithms.registry`.
```

## `tests/test_algorithms_generated.py`

**Purpose:** Regression smoke tests for generated pvx algorithm wrappers.

**Classes:** `TestGeneratedAlgorithms`
**Functions:** None

**Help commands:** `python3 tests/test_algorithms_generated.py`

### Module Docstring

```text
Regression smoke tests for generated pvx algorithm wrappers.

This test verifies that every algorithm listed in `pvx.algorithms.registry`
is importable and can process a synthetic stereo signal while returning
finite 2D output and honest dispatch metadata.
```

## `tests/test_alpha_release_ready.py`

**Purpose:** Direct seam tests for alpha-release refactors.

**Classes:** `TestCLIHelperSeams`, `TestVocCompatibilitySeams`
**Functions:** None

**Help commands:** `python3 tests/test_alpha_release_ready.py`

### Module Docstring

```text
Direct seam tests for alpha-release refactors.
```

## `tests/test_analysis_response_store.py`

**Purpose:** Unit tests for PVXAN/PVXRF artifact storage and determinism.

**Classes:** `TestAnalysisResponseStore`
**Functions:** None

**Help commands:** `python3 tests/test_analysis_response_store.py`

### Module Docstring

```text
Unit tests for PVXAN/PVXRF artifact storage and determinism.
```

## `tests/test_audio_metrics.py`

**Purpose:** Tests for shared audio metric table utilities.

**Classes:** `TestAudioMetrics`
**Functions:** None

**Help commands:** `python3 tests/test_audio_metrics.py`

### Module Docstring

```text
Tests for shared audio metric table utilities.
```

## `tests/test_augment_bench_gate.py`

**Purpose:** Regression tests for directional augment benchmark gates.

**Classes:** `TestAugmentBenchGate`
**Functions:** None

**Help commands:** `python3 tests/test_augment_bench_gate.py`

### Module Docstring

```text
Regression tests for directional augment benchmark gates.
```

## `tests/test_augment_mode.py`

**Purpose:** Unit tests for pvx augment helper mode.

**Classes:** `TestAugmentMode`
**Functions:** None

**Help commands:** `python3 tests/test_augment_mode.py`

### Module Docstring

```text
Unit tests for pvx augment helper mode.
```

## `tests/test_augment_transforms.py`

**Purpose:** Unit tests for pvx.augment transform library.

**Classes:** `TestPipelineCore`, `TestAddNoise`, `TestImpulseNoise`, `TestRoomSimulator`, `TestCodecDegradation`, `TestBitCrusher`, `TestSpecAugment`, `TestEQPerturber`, `TestGainPerturber`, `TestNormalizer`, `TestClippingSimulator`, `TestTimeShift`, `TestReverse`, `TestFade`, `TestFixedLengthCrop`, `TestPresetFactories`, `TestEngineSelection`, `TestPluginSystem`, `TestPipelineConfig`
**Functions:** `_mono`, `_stereo`, `_sine`

**Help commands:** `python3 tests/test_augment_transforms.py`

### Module Docstring

```text
Unit tests for pvx.augment transform library.

Tests cover: shape preservation, reproducibility, per-transform correctness,
pipeline composition, and combinators.
```

## `tests/test_benchmark_metrics.py`

**Purpose:** Unit tests for benchmark metric primitives.

**Classes:** `TestBenchmarkMetrics`
**Functions:** None

**Help commands:** `python3 tests/test_benchmark_metrics.py`

### Module Docstring

```text
Unit tests for benchmark metric primitives.
```

## `tests/test_benchmark_runner.py`

**Purpose:** Tests for benchmark runner profile selection.

**Classes:** `TestBenchmarkRunnerProfiles`
**Functions:** None

**Help commands:** `python3 tests/test_benchmark_runner.py`

### Module Docstring

```text
Tests for benchmark runner profile selection.
```

## `tests/test_cli_regression.py`

**Purpose:** CLI regression tests for pvxvoc end-to-end workflows.

**Classes:** `TestCLIRegression`
**Functions:** `write_stereo_tone`, `write_mono_tone`, `write_mono_glide`, `write_mono_complex`, `write_multichannel_ir`

**Help commands:** `python3 tests/test_cli_regression.py`

### Module Docstring

```text
CLI regression tests for pvxvoc end-to-end workflows.

Coverage includes:
- baseline multi-channel pitch/time behavior
- dry-run behavior with existing outputs
- microtonal cents-shift CLI path
- non-power-of-two Fourier-sync mode
- a numeric DSP snapshot metric for drift detection
```

## `tests/test_control_bus.py`

**Purpose:** Unit tests for control-bus routing helpers.

**Classes:** `TestControlBus`
**Functions:** None

**Help commands:** `python3 tests/test_control_bus.py`

### Module Docstring

```text
Unit tests for control-bus routing helpers.
```

## `tests/test_docs_coverage.py`

**Purpose:** Documentation coverage checks for CLI flags.

**Classes:** None
**Functions:** `_string_literal`, `_iter_cli_sources`, `_tool_name_for_path`, `extract_flags_from_code`, `load_doc_pairs`, `test_cli_flag_docs_match_parser_definitions`, `test_readme_long_flags_exist_in_parser_sources`

### Module Docstring

```text
Documentation coverage checks for CLI flags.
```

## `tests/test_docs_pdf.py`

**Purpose:** Tests for scripts/scripts_generate_docs_pdf.py helpers.

**Classes:** `TestDocsPdfHelpers`
**Functions:** None

**Help commands:** `python3 tests/test_docs_pdf.py`

### Module Docstring

```text
Tests for scripts/scripts_generate_docs_pdf.py helpers.
```

## `tests/test_dsp.py`

**Purpose:** DSP unit tests for core vocoder and analysis primitives.

**Classes:** `TestPhaseVocoderDSP`
**Functions:** `spectral_centroid`

**Help commands:** `python3 tests/test_dsp.py`

### Module Docstring

```text
DSP unit tests for core vocoder and analysis primitives.

These tests validate transform length behavior, F0 estimation, transient
handling, formant-preserving correction, Fourier-sync operation, runtime
selection, and support for all registered analysis windows.
```

## `tests/test_gpu_transforms.py`

**Purpose:** Tests for GPU-accelerated augmentation transforms (pvx.augment.gpu).

**Classes:** `TestTorchGainPerturber`, `TestTorchAddNoise`, `TestTorchEQPerturber`, `TestTorchSpecAugment`, `TestTorchNormalizer`, `TestTorchClippingSimulator`, `TestTorchPipeline`, `TestNumpyTransformAdapter`, `TestTorchTimeStretch`, `TestTorchPitchShift`, `TestTorchRoomSimulator`, `TestTorchMixup`
**Functions:** `_random_audio`, `_sine_audio`

### Module Docstring

```text
Tests for GPU-accelerated augmentation transforms (pvx.augment.gpu).
```

## `tests/test_integrations.py`

**Purpose:** Smoke tests for pvx.integrations.

**Classes:** `TestHuggingFaceIntegration`, `TestTensorFlowIntegrationImport`, `TestPyTorchIntegrationImport`
**Functions:** `_mono`

**Help commands:** `python3 tests/test_integrations.py`

### Module Docstring

```text
Smoke tests for pvx.integrations.

Tests that the integration modules import cleanly, the map functions work
with pure-NumPy inputs, and framework-specific functionality is properly
guarded behind optional import checks.
```

## `tests/test_microtonal.py`

**Purpose:** Microtonal feature tests across CSV mapping, retune, and CLI pitch paths.

**Classes:** `TestMicrotonalSupport`
**Functions:** `write_text`

**Help commands:** `python3 tests/test_microtonal.py`

### Module Docstring

```text
Microtonal feature tests across CSV mapping, retune, and CLI pitch paths.

Ensures cents/ratio/semitone mapping behavior remains stable and that
microtonal pitch controls produce expected conversion outputs.
```

## `tests/test_output_policy.py`

**Purpose:** Unit tests for shared output policy helpers.

**Classes:** `TestOutputPolicy`
**Functions:** `_make_args`

**Help commands:** `python3 tests/test_output_policy.py`, `python3 tests/test_output_policy.py --help`

### Module Docstring

```text
Unit tests for shared output policy helpers.
```

## `tests/test_pvc_parity_benchmark.py`

**Purpose:** Tests for PVC parity benchmark runner and gate logic.

**Classes:** `TestPVCParityBenchmark`
**Functions:** None

**Help commands:** `python3 tests/test_pvc_parity_benchmark.py`

### Module Docstring

```text
Tests for PVC parity benchmark runner and gate logic.
```

## `tests/test_pvc_phase3_5.py`

**Purpose:** Unit tests for PVC-inspired Phase 3-5 core operators.

**Classes:** `TestPVCPhase3To5`
**Functions:** None

**Help commands:** `python3 tests/test_pvc_phase3_5.py`

### Module Docstring

```text
Unit tests for PVC-inspired Phase 3-5 core operators.
```

## `tests/test_pvc_phase6.py`

**Purpose:** Unit tests for PVC-inspired Phase 6 function-stream utilities.

**Classes:** `TestPVCPhase6Utilities`, `TestPVCPhase6Cli`
**Functions:** None

**Help commands:** `python3 tests/test_pvc_phase6.py`

### Module Docstring

```text
Unit tests for PVC-inspired Phase 6 function-stream utilities.
```

## `tests/test_pvxvoc_cli.py`

**Purpose:** Regression tests for the dedicated pvxvoc CLI module.

**Classes:** `TestPvxvocCli`
**Functions:** None

**Help commands:** `python3 tests/test_pvxvoc_cli.py`

### Module Docstring

```text
Regression tests for the dedicated pvxvoc CLI module.
```

## `tests/test_transient_and_stereo.py`

**Purpose:** Tests for hybrid transient processing and stereo coherence modes.

**Classes:** `TestTransientHybridAndStereo`
**Functions:** `_build_args`, `_phase_drift_internal`

**Help commands:** `python3 tests/test_transient_and_stereo.py`

### Module Docstring

```text
Tests for hybrid transient processing and stereo coherence modes.
```

## `tests/test_voc_console.py`

**Purpose:** Focused regression tests for the extracted voc console helpers.

**Classes:** `TestVocConsole`
**Functions:** None

**Help commands:** `python3 tests/test_voc_console.py`

### Module Docstring

```text
Focused regression tests for the extracted voc console helpers.
```

## `tests/test_voc_jobs.py`

**Purpose:** Focused regression tests for extracted voc job helpers.

**Classes:** `TestVocJobs`
**Functions:** None

**Help commands:** `python3 tests/test_voc_jobs.py`

### Module Docstring

```text
Focused regression tests for extracted voc job helpers.
```

## Attribution

 See [`ATTRIBUTION.md`](../ATTRIBUTION.md).
