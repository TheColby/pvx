# Python File Documentation and Help

## Acronym Primer

- application programming interface (API)
- command-line interface (CLI)
- path environment variable (PATH)
- digital signal processing (DSP)
- short-time Fourier transform (STFT)
- fast Fourier transform (FFT)
- comma-separated values (CSV)
- JavaScript Object Notation (JSON)
- central processing unit (CPU)
- graphics processing unit (GPU)
- Compute Unified Device Architecture (CUDA)
- root-mean-square (RMS)
- signal-to-noise ratio (SNR)
- fundamental frequency (F0)
- independent component analysis (ICA)
- non-negative matrix factorization (NMF)

Comprehensive reference for every Python file in this repository.

Total Python files documented: **204**

## Contents

- [`HPS-pitch-track.py`](#HPS-pitch-trackpy)
- [`benchmarks/__init__.py`](#benchmarksinitpy)
- [`benchmarks/metrics.py`](#benchmarksmetricspy)
- [`benchmarks/run_bench.py`](#benchmarksrunbenchpy)
- [`main.py`](#mainpy)
- [`pvx.py`](#pvxpy)
- [`pvxalgorithms/__init__.py`](#pvxalgorithmsinitpy)
- [`pvxalgorithms/base.py`](#pvxalgorithmsbasepy)
- [`pvxalgorithms/registry.py`](#pvxalgorithmsregistrypy)
- [`pvxcommon.py`](#pvxcommonpy)
- [`pvxconform.py`](#pvxconformpy)
- [`pvxdenoise.py`](#pvxdenoisepy)
- [`pvxdeverb.py`](#pvxdeverbpy)
- [`pvxformant.py`](#pvxformantpy)
- [`pvxfreeze.py`](#pvxfreezepy)
- [`pvxharmonize.py`](#pvxharmonizepy)
- [`pvxlayer.py`](#pvxlayerpy)
- [`pvxmorph.py`](#pvxmorphpy)
- [`pvxretune.py`](#pvxretunepy)
- [`pvxtransient.py`](#pvxtransientpy)
- [`pvxunison.py`](#pvxunisonpy)
- [`pvxvoc.py`](#pvxvocpy)
- [`pvxwarp.py`](#pvxwarppy)
- [`scripts_ab_compare.py`](#scriptsabcomparepy)
- [`scripts_benchmark_matrix.py`](#scriptsbenchmarkmatrixpy)
- [`scripts_generate_docs_extras.py`](#scriptsgeneratedocsextraspy)
- [`scripts_generate_docs_pdf.py`](#scriptsgeneratedocspdfpy)
- [`scripts_generate_html_docs.py`](#scriptsgeneratehtmldocspy)
- [`scripts_generate_python_docs.py`](#scriptsgeneratepythondocspy)
- [`scripts_generate_theory_docs.py`](#scriptsgeneratetheorydocspy)
- [`scripts_quality_regression.py`](#scriptsqualityregressionpy)
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
- [`src/pvx/cli/__init__.py`](#srcpvxcliinitpy)
- [`src/pvx/cli/hps_pitch_track.py`](#srcpvxclihpspitchtrackpy)
- [`src/pvx/cli/main.py`](#srcpvxclimainpy)
- [`src/pvx/cli/pvx.py`](#srcpvxclipvxpy)
- [`src/pvx/cli/pvxconform.py`](#srcpvxclipvxconformpy)
- [`src/pvx/cli/pvxdenoise.py`](#srcpvxclipvxdenoisepy)
- [`src/pvx/cli/pvxdeverb.py`](#srcpvxclipvxdeverbpy)
- [`src/pvx/cli/pvxformant.py`](#srcpvxclipvxformantpy)
- [`src/pvx/cli/pvxfreeze.py`](#srcpvxclipvxfreezepy)
- [`src/pvx/cli/pvxharmonize.py`](#srcpvxclipvxharmonizepy)
- [`src/pvx/cli/pvxlayer.py`](#srcpvxclipvxlayerpy)
- [`src/pvx/cli/pvxmorph.py`](#srcpvxclipvxmorphpy)
- [`src/pvx/cli/pvxretune.py`](#srcpvxclipvxretunepy)
- [`src/pvx/cli/pvxtransient.py`](#srcpvxclipvxtransientpy)
- [`src/pvx/cli/pvxunison.py`](#srcpvxclipvxunisonpy)
- [`src/pvx/cli/pvxwarp.py`](#srcpvxclipvxwarppy)
- [`src/pvx/core/__init__.py`](#srcpvxcoreinitpy)
- [`src/pvx/core/audio_metrics.py`](#srcpvxcoreaudiometricspy)
- [`src/pvx/core/common.py`](#srcpvxcorecommonpy)
- [`src/pvx/core/output_policy.py`](#srcpvxcoreoutputpolicypy)
- [`src/pvx/core/presets.py`](#srcpvxcorepresetspy)
- [`src/pvx/core/stereo.py`](#srcpvxcorestereopy)
- [`src/pvx/core/streaming.py`](#srcpvxcorestreamingpy)
- [`src/pvx/core/transients.py`](#srcpvxcoretransientspy)
- [`src/pvx/core/voc.py`](#srcpvxcorevocpy)
- [`src/pvx/core/wsola.py`](#srcpvxcorewsolapy)
- [`src/pvx/metrics/__init__.py`](#srcpvxmetricsinitpy)
- [`src/pvx/metrics/coherence.py`](#srcpvxmetricscoherencepy)
- [`src/pvxalgorithms/__init__.py`](#srcpvxalgorithmsinitpy)
- [`src/pvxalgorithms/base.py`](#srcpvxalgorithmsbasepy)
- [`src/pvxalgorithms/registry.py`](#srcpvxalgorithmsregistrypy)
- [`tests/test_algorithms_generated.py`](#teststestalgorithmsgeneratedpy)
- [`tests/test_audio_metrics.py`](#teststestaudiometricspy)
- [`tests/test_benchmark_metrics.py`](#teststestbenchmarkmetricspy)
- [`tests/test_benchmark_runner.py`](#teststestbenchmarkrunnerpy)
- [`tests/test_cli_regression.py`](#teststestcliregressionpy)
- [`tests/test_docs_coverage.py`](#teststestdocscoveragepy)
- [`tests/test_docs_pdf.py`](#teststestdocspdfpy)
- [`tests/test_dsp.py`](#teststestdsppy)
- [`tests/test_microtonal.py`](#teststestmicrotonalpy)
- [`tests/test_output_policy.py`](#teststestoutputpolicypy)
- [`tests/test_transient_and_stereo.py`](#teststesttransientandstereopy)

## `HPS-pitch-track.py`

**Purpose:** Compatibility wrapper for the HPS pitch tracker CLI.

**Classes:** None
**Functions:** None

**Help commands:** `python3 HPS-pitch-track.py --help`

### CLI Help Snapshot

```text
usage: HPS-pitch-track.py [-h] [--output OUTPUT] [--backend {auto,pyin,acf}]
                          [--fmin FMIN] [--fmax FMAX]
                          [--frame-length FRAME_LENGTH] [--hop-size HOP_SIZE]
                          [--ratio-reference {median,mean,first,hz}]
                          [--reference-hz REFERENCE_HZ]
                          [--ratio-min RATIO_MIN] [--ratio-max RATIO_MAX]
                          [--smooth-frames SMOOTH_FRAMES]
                          [--confidence-floor CONFIDENCE_FLOOR]
                          [--stretch STRETCH]
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
                        Pitch backend (default: auto -> pyin if available,
                        else acf)
  --fmin FMIN           Minimum F0 in Hz (default: 50)
  --fmax FMAX           Maximum F0 in Hz (default: 1200)
  --frame-length FRAME_LENGTH
                        Frame length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 256)
  --ratio-reference {median,mean,first,hz}
                        Reference for emitted pitch_ratio values (default:
                        median voiced f0).
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
  --stretch STRETCH     Emit constant stretch column value (default: 1.0).
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
```

### Module Docstring

```text
Compatibility wrapper for the HPS pitch tracker CLI.
```

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

## `benchmarks/run_bench.py`

**Purpose:** Reproducible quality benchmark: pvx vs Rubber Band vs librosa baseline.

**Classes:** `TaskSpec`
**Functions:** `_sha256_bytes`, `_sha256_file`, `_parse_version`, `_collect_environment_metadata`, `_corpus_manifest_entries`, `_load_manifest`, `_write_manifest`, `_manifest_index`, `_validate_corpus_against_manifest`, `_prepare_dataset`, `_case_key`, `_diagnose_metrics`, `_method_diagnostics`, `_pvx_bench_args`, `_read_audio`, `_write_audio`, `_match_channels`, `_to_mono`, `_align_pair`, `_generate_tiny_dataset`, `_run_pvx_cycle`, `_find_rubberband`, `_run_rubberband_cycle`, `_run_librosa_cycle`, `_compute_metrics`, `_aggregate`, `_render_markdown`, `_check_gate`, `main`

**Help commands:** `python3 benchmarks/run_bench.py`, `python3 benchmarks/run_bench.py --help`

### Module Docstring

```text
Reproducible quality benchmark: pvx vs Rubber Band vs librosa baseline.
```

## `main.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 main.py --help`

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
  pvx chain input.wav --pipeline "voc --stretch 1.2 | formant --mode preserve" --output out.wav
  pvx stream input.wav --output out.wav --chunk-seconds 0.2 --time-stretch 2.0
  pvx list
  pvx examples basic
  pvx help voc

Available tool commands: voc, freeze, harmonize, conform, morph, warp, formant, transient, unison, denoise, deverb, retune, layer, pitch-track
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.main` after the
src-layout migration.
```

## `pvx.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvx.py`

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvx` after the
src-layout migration.
```

## `pvxalgorithms/__init__.py`

**Purpose:** Compatibility shim for `pvxalgorithms` namespace.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Compatibility shim for `pvxalgorithms` namespace.

Use `pvx.algorithms` as the canonical import path.
```

## `pvxalgorithms/base.py`

**Purpose:** Compatibility shim for `pvxalgorithms.base`.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Compatibility shim for `pvxalgorithms.base`.
```

## `pvxalgorithms/registry.py`

**Purpose:** Compatibility shim for `pvxalgorithms.registry`.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Compatibility shim for `pvxalgorithms.registry`.
```

## `pvxcommon.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxcommon.py`

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.core.common` after the
src-layout migration.
```

## `pvxconform.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxconform.py --help`

### CLI Help Snapshot

```text
usage: pvxconform.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                     --map MAP [--crossfade-ms CROSSFADE_MS]
                     [--resample-mode {auto,fft,linear}]
                     inputs [inputs ...]

Conform audio to a CSV map. CSV columns: start_sec,end_sec,stretch, and one
pitch field: pitch_semitones or pitch_cents or pitch_ratio (pitch_ratio
accepts decimals, fractions like 3/2, or expressions like 2^(1/12))

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _conform)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
  --rms-dbfs RMS_DBFS   Targ
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxconform` after the
src-layout migration.
```

## `pvxdenoise.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxdenoise.py --help`

### CLI Help Snapshot

```text
usage: pvxdenoise.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                     [--noise-seconds NOISE_SECONDS] [--noise-file NOISE_FILE]
                     [--reduction-db REDUCTION_DB] [--floor FLOOR]
                     [--smooth SMOOTH]
                     inputs [inputs ...]

Spectral subtraction denoiser

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _denoise)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
                        Integrated loudness target in LUFS
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxdenoise` after the
src-layout migration.
```

## `pvxdeverb.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxdeverb.py --help`

### CLI Help Snapshot

```text
usage: pvxdeverb.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                    [--cuda-device CUDA_DEVICE] [--strength STRENGTH]
                    [--decay DECAY] [--floor FLOOR]
                    inputs [inputs ...]

Reduce spectral tails / reverberant smear

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _deverb)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
                        Integrated loudness target in LUFS
  --compressor-threshold-db COMPRESSOR_THRESHOLD_DB
                        Enable compressor above threshold dBFS
  --compressor-ratio COMPRES
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxdeverb` after the
src-layout migration.
```

## `pvxformant.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxformant.py --help`

### CLI Help Snapshot

```text
usage: pvxformant.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _formant)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization mo
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxformant` after the
src-layout migration.
```

## `pvxfreeze.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxfreeze.py --help`

### CLI Help Snapshot

```text
usage: pvxfreeze.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                    [--cuda-device CUDA_DEVICE] [--freeze-time FREEZE_TIME]
                    [--duration DURATION] [--random-phase]
                    inputs [inputs ...]

Freeze a spectral slice into a sustained output

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _freeze)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
                        Integrated loudness target in LUFS
  --compressor-threshold-db COMPRESSOR_THRESHOLD_DB
                        Enable compressor above threshold dBFS
  --compr
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxfreeze` after the
src-layout migration.
```

## `pvxharmonize.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxharmonize.py --help`

### CLI Help Snapshot

```text
usage: pvxharmonize.py [-h] [-o OUTPUT_DIR] [--output OUTPUT]
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
                       [--intervals INTERVALS]
                       [--intervals-cents INTERVALS_CENTS] [--gains GAINS]
                       [--pans PANS] [--force-stereo]
                       [--resample-mode {auto,fft,linear}]
                       inputs [inputs ...]

Generate harmonized voices from an input

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _harm)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
 
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxharmonize` after the
src-layout migration.
```

## `pvxlayer.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxlayer.py --help`

### CLI Help Snapshot

```text
usage: pvxlayer.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                   [--harmonic-stretch HARMONIC_STRETCH]
                   [--harmonic-pitch-semitones HARMONIC_PITCH_SEMITONES]
                   [--harmonic-pitch-cents HARMONIC_PITCH_CENTS]
                   [--percussive-stretch PERCUSSIVE_STRETCH]
                   [--percussive-pitch-semitones PERCUSSIVE_PITCH_SEMITONES]
                   [--percussive-pitch-cents PERCUSSIVE_PITCH_CENTS]
                   [--harmonic-gain HARMONIC_GAIN]
                   [--percussive-gain PERCUSSIVE_GAIN]
                   [--harmonic-kernel HARMONIC_KERNEL]
                   [--percussive-kernel PERCUSSIVE_KERNEL]
                   [--resample-mode {auto,fft,linear}]
                   inputs [inputs ...]

Split into harmonic/percussive layers and process independently

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _layer)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for ext
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxlayer` after the
src-layout migration.
```

## `pvxmorph.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxmorph.py --help`

### CLI Help Snapshot

```text
usage: pvxmorph.py [-h] [-o OUTPUT] [--stdout] [--output-format OUTPUT_FORMAT]
                   [--alpha ALPHA] [--normalize {none,peak,rms}]
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
                   [--metadata-policy {none,sidecar,copy}] [--overwrite]
                   [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                   [--quiet] [--silent] [--n-fft N_FFT]
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
                   input_a input_b

Morph two audio files in the STFT domain

positional arguments:
  input_a               Input A path or '-' for stdin
  input_b               Input B path or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output file path
  --stdout              Write processed audio to stdout stream (for piping);
                        equivalent to -o -
  --output-format OUTPUT_FORMAT
                        Output extension/format; for --stdout defaults to wav
  --alpha ALPHA         Morph amount 0..1 (0=A, 1=B)
  --normalize {none,peak,rms}
                        Output normalization mode
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS when --normalize peak
  --rms-dbfs RMS_DBFS   Target RMS dBFS when --normalize rms
  --target-lufs TARGET_LUFS
                        Integrated loudness target in LUFS
  --compressor-threshold-db COMPRESSOR_THRESHOLD_DB
                        Enable compressor above threshold dBFS
  --compressor-ratio COMPRESSOR_RATIO
                        Compressor ratio (>=1)
  --compressor-attack-ms COMPRESSOR_ATTACK_MS
                        Compressor attack time in ms
  --compressor-release-ms COMPRESSOR_RELEASE_MS
                        Compressor release time in ms
  --compressor-makeup-db COMPRESSOR_MAKEUP_DB
                        Compressor makeup gain in dB
  --expander-threshold-db EXPANDER_THRESHOLD_DB
                        Enable downward expander below threshold dBFS
  --expander-ratio EXPANDER_RATIO
                        Expander ratio (>=1)
  --expander-attack-ms EXPANDER_ATTACK_MS
                        Expander attack time in ms
  --expander-release-ms EXPANDER_REL
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxmorph` after the
src-layout migration.
```

## `pvxretune.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxretune.py --help`

### CLI Help Snapshot

```text
usage: pvxretune.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                    [--cuda-device CUDA_DEVICE] [--root ROOT]
                    [--scale {chromatic,major,minor,pentatonic}]
                    [--scale-cents SCALE_CENTS] [--strength STRENGTH]
                    [--chunk-ms CHUNK_MS] [--overlap-ms OVERLAP_MS]
                    [--f0-min F0_MIN] [--f0-max F0_MAX]
                    [--resample-mode {auto,fft,linear}]
                    inputs [inputs ...]

Monophonic retune toward a musical scale

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _retune)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
  --rms-dbfs RMS_DBFS   Target RMS dB
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxretune` after the
src-layout migration.
```

## `pvxtransient.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxtransient.py --help`

### CLI Help Snapshot

```text
usage: pvxtransient.py [-h] [-o OUTPUT_DIR] [--output OUTPUT]
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
                       [--time-stretch TIME_STRETCH]
                       [--target-duration TARGET_DURATION]
                       [--pitch-shift-semitones PITCH_SHIFT_SEMITONES]
                       [--pitch-shift-cents PITCH_SHIFT_CENTS]
                       [--pitch-shift-ratio PITCH_SHIFT_RATIO]
                       [--transient-threshold TRANSIENT_THRESHOLD]
                       [--resample-mode {auto,fft,linear}]
                       inputs [inputs ...]

Transient-preserving phase-vocoder processor

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _trans)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent     
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxtransient` after the
src-layout migration.
```

## `pvxunison.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxunison.py --help`

### CLI Help Snapshot

```text
usage: pvxunison.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                    [--cuda-device CUDA_DEVICE] [--voices VOICES]
                    [--detune-cents DETUNE_CENTS] [--width WIDTH]
                    [--dry-mix DRY_MIX] [--resample-mode {auto,fft,linear}]
                    inputs [inputs ...]

Stereo unison thickener

positional arguments:
  inputs                Input files/globs or '-' for stdin

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --output, --out OUTPUT
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _unison)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
                        Integrated loudness target in LUFS
  --compressor-threshold-db COMPRESSOR_THRESHOLD_DB
                       
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxunison` after the
src-layout migration.
```

## `pvxvoc.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxvoc.py --help`

### CLI Help Snapshot

```text
usage: pvxvoc.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                 [--output-format OUTPUT_FORMAT] [--out OUTPUT] [--overwrite]
                 [--dry-run] [--stdout]
                 [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                 [--quiet] [--silent]
                 [--preset {none,default,vocal,ambient,extreme,vocal_studio,drums_safe,extreme_ambient,stereo_coherent}]
                 [--example {all,basic,vocal,ambient,extreme,drums_safe,stereo_coherent,hybrid,benchmark,gpu,pipeline,csv}]
                 [--guided] [--stretch STRETCH] [--gpu] [--cpu]
                 [--quality-profile {neutral,speech,music,percussion,ambient,extreme}]
                 [--auto-profile]
                 [--auto-profile-lookahead-seconds AUTO_PROFILE_LOOKAHEAD_SECONDS]
                 [--auto-transform] [--n-fft N_FFT] [--win-length WIN_LENGTH]
                 [--hop-size HOP_SIZE]
                 [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                 [--kaiser-beta KAISER_BETA]
                 [--transform {fft,dft,czt,dct,dst,hartley}] [--no-center]
                 [--phase-locking {off,identity}]
                 [--phase-engine {propagate,hybrid,random}]
                 [--ambient-phase-mix AMBIENT_PHASE_MIX]
                 [--phase-random-seed PHASE_RANDOM_SEED]
                 [--transient-preserve]
                 [--transient-threshold TRANSIENT_THRESHOLD] [--fourier-sync]
                 [--fourier-sync-min-fft FOURIER_SYNC_MIN_FFT]
                 [--fourier-sync-max-fft FOURIER_SYNC_MAX_FFT]
                 [--fourier-sync-smooth FOURIER_SYNC_SMOOTH]
                 [--multires-fusion] [--multires-ffts MULTIRES_FFTS]
                 [--multires-weights MULTIRES_WEIGHTS]
                 [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                 [--time-stretch TIME_STRETCH]
                 [--target-duration TARGET_DURATION]
                 [--stretch-mode {auto,standard,multistage}]
                 [--extreme-time-stretch]
                 [--extreme-stretch-threshold EXTREME_STRETCH_THRESHOLD]
                 [--max-stage-stretch MAX_STAGE_STRETCH] [--onset-time-credit]
                 [--onset-credit-pull ONSET_CREDIT_PULL]
                 [--onset-credit-max ONSET_CREDIT_MAX] [--no-onset-realign]
                 [--ambient-preset]
                 [--auto-segment-seconds AUTO_SEGMENT_SECONDS]
                 [--checkpoint-dir CHECKPOINT_DIR]
                 [--checkpoint-id CHECKPOINT_ID] [--resume]
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
                 --target-f0 TARGET_F0] [--analysis-channel {first,mix}]
                 [--f0-min F0_MIN] [--f0-max F0_MAX]
                 [--pitch-mode {standard,formant-preserving}]
                 [--formant-lifter FORMANT_LIFTER]
                 [--formant-strength FORMANT_STRENGTH]
                 [--formant-max-gain-db FORMANT_MAX_GAIN_DB]
                 [--pitch-map PITCH_MAP] [--pitch-map-stdin]
                 [--pitch-follow-stdin] [--pitch-conf-min PITCH_CONF_MIN]
                 [--pitch-lowconf-mode {hold,unity,interp}]
                 [--pitch-map-smooth-ms PITCH_MAP_SMOOTH_MS]
                 [--pitch-map-crossfade-ms PITCH_MAP_CROSSFADE_MS]
                 [--target-sample-rate TARGET_SAMPLE_RATE]
                 [--resample-mode {auto,fft,linear}]
                 [--normalize {none,peak,rms}] [--peak-dbfs PEAK_DBFS]
                 [--rms-dbfs RMS_DBFS] [--target-lufs TARGET_LUFS]
                 [--compressor-threshold-db COMPRESSOR_THRESHOLD_DB]
                 [--compressor-ratio COMPRESSOR_RATIO]
                 [--compressor-attack-ms COMPRESSOR_ATTACK_MS]
                 [--compressor-release-ms COMPRESSOR_RELEASE_MS]
                 [--compressor-makeup-db COMPRESSOR_M
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.core.voc` after the
src-layout migration.
```

## `pvxwarp.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 pvxwarp.py --help`

### CLI Help Snapshot

```text
usage: pvxwarp.py [-h] [-o OUTPUT_DIR] [--output OUTPUT] [--suffix SUFFIX]
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
                        Explicit output file path (single-input mode only).
                        Alias: --out
  --suffix SUFFIX       Output filename suffix (default: _warp)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --stdout              Write processed audio to stdout stream (for piping);
                        requires exactly one input
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
                        Integrated loudness target in LUFS
  --compressor-threshold-db COMPRESSOR_THRESHOLD_DB
                        Enable compressor above threshold dBFS
  --compressor-ratio COMPRESSOR_RATIO
                        Compressor ratio (>
... [truncated]
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxwarp` after the
src-layout migration.
```

## `scripts_ab_compare.py`

**Purpose:** Run an A/B pvx render comparison and emit metrics reports.

**Classes:** None
**Functions:** `_metrics`, `_render`, `main`

**Help commands:** `python3 scripts_ab_compare.py`, `python3 scripts_ab_compare.py --help`

### Module Docstring

```text
Run an A/B pvx render comparison and emit metrics reports.
```

## `scripts_benchmark_matrix.py`

**Purpose:** Benchmark matrix runner for pvxvoc transform/window/device combinations.

**Classes:** None
**Functions:** `_parse_csv_tokens`, `_run_case`, `main`

**Help commands:** `python3 scripts_benchmark_matrix.py`, `python3 scripts_benchmark_matrix.py --help`

### Module Docstring

```text
Benchmark matrix runner for pvxvoc transform/window/device combinations.
```

## `scripts_generate_docs_extras.py`

**Purpose:** Generate advanced docs artifacts (coverage, limitations, benchmarks, citations, cookbook, architecture).

**Classes:** None
**Functions:** `git_commit_meta`, `generated_stamp_lines`, `acronym_primer_lines`, `write_json`, `_string_literal`, `_simple_literal`, `_tool_name_for_path`, `_iter_cli_sources`, `collect_cli_flags`, `generate_cli_flags_reference`, `_unique_join`, `generate_algorithm_limitations`, `generate_cookbook`, `generate_architecture_doc`, `_spectral_distance_db`, `_snr_db`, `_make_signal`, `_benchmark_backend`, `generate_benchmarks`, `_classify_reference_url`, `_extract_doi`, `_bib_escape`, `_bib_key`, `generate_citation_docs`, `generate_docs_contract`, `main`

**Help commands:** `python3 scripts_generate_docs_extras.py`, `python3 scripts_generate_docs_extras.py --help`

### Module Docstring

```text
Generate advanced docs artifacts (coverage, limitations, benchmarks, citations, cookbook, architecture).
```

## `scripts_generate_docs_pdf.py`

**Purpose:** Generate one combined PDF from all HTML documentation pages.

**Classes:** `SourcePage`, `ProgressBar`
**Functions:** `add_console_args`, `console_level`, `is_quiet`, `is_silent`, `log`, `html_sort_key`, `collect_html_pages`, `extract_title`, `extract_main_html`, `parse_source_page`, `_rewrite_internal_links`, `_extract_reference_count`, `_annotate_display_equations`, `build_combined_html`, `discover_chromium_executable`, `run_cmd`, `render_pdf_with_chromium`, `render_pdf_with_wkhtmltopdf`, `render_pdf_with_weasyprint`, `render_pdf_with_playwright`, `build_engine_registry`, `auto_engine_order`, `render_pdf`, `parse_args`, `main`

**Help commands:** `python3 scripts_generate_docs_pdf.py`, `python3 scripts_generate_docs_pdf.py --help`

### Module Docstring

```text
Generate one combined PDF from all HTML documentation pages.
```

## `scripts_generate_html_docs.py`

**Purpose:** Generate grouped HTML documentation for pvx algorithms and research references.

**Classes:** None
**Functions:** `git_commit_meta`, `scholar`, `_contains_out_of_scope_text`, `_is_out_of_scope_paper`, `_is_out_of_scope_glossary`, `slugify`, `dedupe_papers`, `_upgrade_paper_url`, `upgrade_paper_urls`, `load_extra_papers`, `load_glossary`, `infer_glossary_terms`, `glossary_links_html`, `load_json`, `classify_reference_url`, `window_entries`, `window_tradeoffs`, `_split_top_level_once`, `_extract_params_get_calls`, `extract_algorithm_param_specs`, `extract_algorithm_params`, `extract_module_cli_flags`, `collect_algorithm_module_flags`, `sample_value_from_default`, `format_sample_params`, `compute_unique_cli_flags`, `grouped_algorithms`, `acronym_primer_html`, `html_page`, `write_style_css`, `render_index`, `module_path_from_meta`, `render_group_pages`, `render_papers_page`, `render_glossary_page`, `render_math_page`, `render_windows_page`, `render_architecture_page`, `render_cli_flags_page`, `render_limitations_page`, `render_benchmarks_page`, `render_cookbook_page`, `render_citations_page`, `write_docs_root_index`, `main`

**Help commands:** `python3 scripts_generate_html_docs.py`

### Module Docstring

```text
Generate grouped HTML documentation for pvx algorithms and research references.
```

## `scripts_generate_python_docs.py`

**Purpose:** Generate comprehensive documentation for every Python file in the repository.

**Classes:** None
**Functions:** `acronym_primer_lines`, `rel`, `safe_read`, `parse_module`, `cli_help`, `extract_algorithm_params`, `generate_algorithm_param_doc`, `generate_python_help_doc`, `main`

**Help commands:** `python3 scripts_generate_python_docs.py`, `python3 scripts_generate_python_docs.py --help`

### Module Docstring

```text
Generate comprehensive documentation for every Python file in the repository.
```

## `scripts_generate_theory_docs.py`

**Purpose:** Generate GitHub-renderable theory docs (math foundations + window reference).

**Classes:** None
**Functions:** `git_commit_meta`, `generated_stamp_lines`, `acronym_primer_lines`, `window_entries`, `window_tradeoffs`, `window_samples`, `_first_local_minimum`, `compute_window_metrics`, `_polyline_points`, `_downsample_series`, `write_line_svg`, `generate_window_assets_and_metrics`, `write_math_foundations`, `write_window_reference`, `main`

**Help commands:** `python3 scripts_generate_theory_docs.py`

### Module Docstring

```text
Generate GitHub-renderable theory docs (math foundations + window reference).
```

## `scripts_quality_regression.py`

**Purpose:** Render a pvx case and compare objective metrics against a baseline.

**Classes:** None
**Functions:** `_metrics`, `_compare`, `main`

**Help commands:** `python3 scripts_quality_regression.py`, `python3 scripts_quality_regression.py --help`

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

**Purpose:** Generated algorithm scaffolds for proposed pvx roadmap features.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Generated algorithm scaffolds for proposed pvx roadmap features.
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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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

## `src/pvx/algorithms/base.py`

**Purpose:** Shared DSP utilities and implementations for pvx algorithm modules.

**Classes:** `AlgorithmResult`
**Functions:** `coerce_audio`, `maybe_librosa`, `maybe_loudnorm`, `build_metadata`, `normalize_peak`, `ensure_length`, `resample_length`, `envelope_follower`, `soft_clip`, `_resolve_transform_name`, `_stft_config`, `stft_multi`, `istft_multi`, `spectral_sharpen`, `spectral_blur`, `hpss_split`, `time_stretch`, `pitch_shift`, `overlap_add_frames`, `granular_time_stretch`, `spectral_gate`, `spectral_subtract_denoise`, `mmse_like_denoise`, `minimum_statistics_denoise`, `simple_declick`, `simple_declip`, `dereverb_decay_subtract`, `dereverb_wpe_style`, `compressor`, `upward_compressor`, `true_peak_limit`, `transient_shaper`, `spectral_dynamics`, `split_bands`, `multiband_compression`, `cross_synthesis`, `spectral_convolution`, `spectral_freeze`, `phase_randomize`, `formant_warp`, `resonator_bank`, `spectral_contrast_exaggerate`, `rhythmic_gate`, `ring_mod`, `spectral_tremolo`, `envelope_modulation`, `estimate_f0_track`, `nearest_scale_freq`, `variable_pitch_shift`, `detect_key_from_chroma`, `cqt_or_stft`, `icqt_or_istft`, `_dispatch_time_scale`, `_dispatch_pitch_tracking`, `_scale_cents_from_name`, `_dispatch_retune`, `_dispatch_transforms`, `_dispatch_separation`, `_dispatch_denoise`, `_dispatch_dereverb`, `_lufs_estimate`, `_dispatch_dynamics`, `_dispatch_creative`, `_dispatch_granular`, `_dispatch_analysis`, `_spatial_to_channels`, `_spatial_fractional_delay`, `_spatial_apply_delays`, `_spatial_circular_gains`, `_spatial_delay_by_xcorr`, `_spatial_estimate_channel_delays`, `_spatial_synthetic_rir`, `_dispatch_spatial`, `run_algorithm`

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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

## `src/pvx/algorithms/registry.py`

**Purpose:** Registry for generated pvx algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Registry for generated pvx algorithm scaffolds.
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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/pvx_ALGORITHM_PARAMS.md`.

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

## `src/pvx/cli/__init__.py`

**Purpose:** CLI entrypoints for pvx tools.

**Classes:** None
**Functions:** None

### Module Docstring

```text
CLI entrypoints for pvx tools.
```

## `src/pvx/cli/hps_pitch_track.py`

**Purpose:** Track F0 and emit a pvx control-map CSV for pitch-follow pipelines.

**Classes:** None
**Functions:** `_read_audio`, `_acf_pitch_and_confidence`, `_estimate_reference_hz`, `_smooth`, `_track_pyin`, `_track_acf`, `build_parser`, `validate_args`, `_emit_csv`, `main`

**Help commands:** `python3 src/pvx/cli/hps_pitch_track.py`, `python3 src/pvx/cli/hps_pitch_track.py --help`

### Module Docstring

```text
Track F0 and emit a pvx control-map CSV for pitch-follow pipelines.
```

## `src/pvx/cli/main.py`

**Purpose:** Compatibility entrypoint for the unified pvx CLI.

**Classes:** None
**Functions:** None

**Help commands:** `python3 src/pvx/cli/main.py`

### Module Docstring

```text
Compatibility entrypoint for the unified pvx CLI.
```

## `src/pvx/cli/pvx.py`

**Purpose:** Unified top-level CLI for the pvx command suite.

**Classes:** `ToolSpec`
**Functions:** `_tool_index`, `_load_entrypoint`, `_looks_like_audio_input`, `_tool_names_csv`, `print_tools`, `print_examples`, `_prompt_text`, `_prompt_choice`, `_print_command_preview`, `run_guided_mode`, `_split_pipeline_stages`, `_token_flag`, `_run_stage_command`, `run_chain_mode`, `run_stream_mode`, `dispatch_tool`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvx.py`, `python3 src/pvx/cli/pvx.py --help`

### Module Docstring

```text
Unified top-level CLI for the pvx command suite.
```

## `src/pvx/cli/pvxconform.py`

**Purpose:** Conform timing and pitch to a user-provided segment map.

**Classes:** None
**Functions:** `expand_segments`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxconform.py`, `python3 src/pvx/cli/pvxconform.py --help`

### Module Docstring

```text
Conform timing and pitch to a user-provided segment map.
```

## `src/pvx/cli/pvxdenoise.py`

**Purpose:** Phase-consistent spectral denoiser.

**Classes:** None
**Functions:** `smooth_mask`, `denoise_channel`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxdenoise.py`, `python3 src/pvx/cli/pvxdenoise.py --help`

### Module Docstring

```text
Phase-consistent spectral denoiser.
```

## `src/pvx/cli/pvxdeverb.py`

**Purpose:** Spectral tail suppression for dereverberation-like cleanup.

**Classes:** None
**Functions:** `deverb_channel`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxdeverb.py`, `python3 src/pvx/cli/pvxdeverb.py --help`

### Module Docstring

```text
Spectral tail suppression for dereverberation-like cleanup.
```

## `src/pvx/cli/pvxformant.py`

**Purpose:** Formant processing tool with optional pitch shifting.

**Classes:** None
**Functions:** `shift_envelope`, `formant_process_channel`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxformant.py`, `python3 src/pvx/cli/pvxformant.py --help`

### Module Docstring

```text
Formant processing tool with optional pitch shifting.
```

## `src/pvx/cli/pvxfreeze.py`

**Purpose:** Spectral freeze tool built on pvx phase-vocoder primitives.

**Classes:** None
**Functions:** `freeze_channel`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxfreeze.py`, `python3 src/pvx/cli/pvxfreeze.py --help`

### Module Docstring

```text
Spectral freeze tool built on pvx phase-vocoder primitives.
```

## `src/pvx/cli/pvxharmonize.py`

**Purpose:** Multi-voice harmonizer built from phase-vocoder pitch shifts.

**Classes:** None
**Functions:** `pan_stereo`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxharmonize.py`, `python3 src/pvx/cli/pvxharmonize.py --help`

### Module Docstring

```text
Multi-voice harmonizer built from phase-vocoder pitch shifts.
```

## `src/pvx/cli/pvxlayer.py`

**Purpose:** Layered harmonic/percussive processing with independent controls.

**Classes:** None
**Functions:** `hpss_masks`, `split_hpss`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxlayer.py`, `python3 src/pvx/cli/pvxlayer.py --help`

### Module Docstring

```text
Layered harmonic/percussive processing with independent controls.
```

## `src/pvx/cli/pvxmorph.py`

**Purpose:** Spectral morphing between two input files.

**Classes:** None
**Functions:** `match_channels`, `morph_pair`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxmorph.py`, `python3 src/pvx/cli/pvxmorph.py --help`

### Module Docstring

```text
Spectral morphing between two input files.
```

## `src/pvx/cli/pvxretune.py`

**Purpose:** Monophonic retuning with phase-vocoder segment processing.

**Classes:** None
**Functions:** `freq_to_midi`, `midi_to_freq`, `normalize_octave_cents`, `nearest_scale_freq`, `overlap_add`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxretune.py`, `python3 src/pvx/cli/pvxretune.py --help`

### Module Docstring

```text
Monophonic retuning with phase-vocoder segment processing.
```

## `src/pvx/cli/pvxtransient.py`

**Purpose:** Transient-aware time/pitch processing.

**Classes:** None
**Functions:** `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxtransient.py`, `python3 src/pvx/cli/pvxtransient.py --help`

### Module Docstring

```text
Transient-aware time/pitch processing.
```

## `src/pvx/cli/pvxunison.py`

**Purpose:** Create unison width via micro-detuned phase-vocoder voices.

**Classes:** None
**Functions:** `cents_to_ratio`, `pan_gains`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxunison.py`, `python3 src/pvx/cli/pvxunison.py --help`

### Module Docstring

```text
Create unison width via micro-detuned phase-vocoder voices.
```

## `src/pvx/cli/pvxwarp.py`

**Purpose:** Time-warp an input according to a user-provided stretch map.

**Classes:** None
**Functions:** `fill_stretch_segments`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/pvxwarp.py`, `python3 src/pvx/cli/pvxwarp.py --help`

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
**Functions:** `add_console_args`, `console_level`, `is_quiet`, `is_silent`, `log_message`, `log_error`, `build_status_bar`, `add_common_io_args`, `add_output_policy_args`, `add_vocoder_args`, `build_vocoder_config`, `validate_vocoder_args`, `resolve_inputs`, `read_audio`, `finalize_audio`, `write_output`, `print_input_output_metrics_table`, `default_output_path`, `_stream_format_name`, `parse_float_list`, `semitone_to_ratio`, `cents_to_ratio`, `time_pitch_shift_channel`, `time_pitch_shift_audio`, `read_segment_csv`, `concat_with_crossfade`, `ensure_runtime`

### Module Docstring

```text
Shared helpers for pvx DSP command-line tools.
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

**Classes:** `VocoderConfig`, `PitchConfig`, `ControlSegment`, `JobResult`, `FourierSyncPlan`, `AudioBlockResult`, `RuntimeConfig`, `ProgressBar`
**Functions:** `add_console_args`, `console_level`, `is_quiet`, `is_silent`, `log_message`, `log_error`, `clone_args_namespace`, `collect_cli_flags`, `print_cli_examples`, `apply_named_preset`, `_prompt_text`, `_prompt_choice`, `run_guided_mode`, `db_to_amplitude`, `cents_to_ratio`, `_eval_numeric_expr`, `parse_numeric_expression`, `parse_pitch_ratio_value`, `_is_power_of_two`, `parse_numeric_list`, `parse_int_list`, `estimate_content_features`, `suggest_quality_profile`, `apply_quality_profile_overrides`, `resolve_transform_auto`, `_has_cupy`, `_is_cupy_array`, `_array_module`, `_to_numpy`, `_to_runtime_array`, `_as_float`, `_as_bool`, `_i0`, `normalize_transform_name`, `transform_bin_count`, `_analysis_angular_velocity`, `_transform_requires_scipy`, `ensure_transform_backend_available`, `validate_transform_available`, `_resize_or_pad_1d`, `_onesided_to_full_spectrum`, `_forward_transform_numpy`, `_inverse_transform_numpy`, `_forward_transform`, `_inverse_transform`, `add_runtime_args`, `runtime_config`, `configure_runtime`, `configure_runtime_from_args`, `ensure_runtime_dependencies`, `principal_angle`, `_cosine_series_window`, `_bartlett_window`, `_bohman_window`, `_cosine_window`, `_sine_window`, `_triangular_window`, `_bartlett_hann_window`, `_tukey_window`, `_parzen_window`, `_lanczos_window`, `_welch_window`, `_gaussian_window`, `_general_gaussian_window`, `_exponential_window`, `_cauchy_window`, `_cosine_power_window`, `_hann_poisson_window`, `_general_hamming_window`, `_kaiser_window`, `make_window`, `pad_for_framing`, `stft`, `istft`, `scaled_win_length`, `resize_spectrum_bins`, `smooth_series`, `regularize_frame_lengths`, `fill_nan_with_nearest`, `lock_fft_length_to_f0`, `build_fourier_sync_plan`, `compute_transient_flags`, `build_output_time_steps`, `create_phase_rng`, `draw_random_phase`, `apply_phase_engine`, `find_spectral_peaks`, `apply_identity_phase_locking`, `phase_vocoder_time_stretch`, `phase_vocoder_time_stretch_fourier_sync`, `compute_multistage_stretches`, `phase_vocoder_time_stretch_multistage`, `stretch_channel_with_strategy`, `phase_vocoder_time_stretch_multires_fusion`, `linear_resample_1d`, `resample_1d`, `force_length`, `estimate_f0_autocorrelation`, `normalize_audio`, `_envelope_coeff`, `_envelope_follower`, `_estimate_lufs_or_rms_db`, `_apply_compressor`, `_apply_expander`, `_apply_compander`, `_apply_limiter`, `_apply_soft_clip`, `add_mastering_args`, `validate_mastering_args`, `apply_mastering_chain`, `cepstral_envelope`, `apply_formant_preservation`, `choose_pitch_ratio`, `_parse_optional_float`, `parse_control_segments_csv`, `apply_control_confidence_policy`, `smooth_control_ratios`, `expand_control_segments`, `load_control_segments`, `_lock_channel_phase_to_reference`, `process_audio_block`, `resolve_base_stretch`, `compute_output_path`, `_stream_format_name`, `_read_audio_input`, `_write_audio_output`, `concat_audio_chunks`, `build_uniform_control_segments`, `_checkpoint_job_id`, `resolve_checkpoint_context`, `load_checkpoint_chunk`, `save_checkpoint_chunk`, `write_manifest`, `process_file`, `force_length_multi`, `resample_multi`, `validate_args`, `build_parser`, `expand_inputs`, `main`

**Help commands:** `python3 src/pvx/core/voc.py`, `python3 src/pvx/core/voc.py --help`

### Module Docstring

```text
Multi-channel phase vocoder CLI for time and pitch manipulation.
```

## `src/pvx/core/wsola.py`

**Purpose:** Deterministic WSOLA time-stretch primitives for transient handling.

**Classes:** None
**Functions:** `_safe_window`, `wsola_time_stretch`

### Module Docstring

```text
Deterministic WSOLA time-stretch primitives for transient handling.
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

**Purpose:** Regression smoke tests for all generated pvx algorithm modules.

**Classes:** `TestGeneratedAlgorithms`
**Functions:** None

**Help commands:** `python3 tests/test_algorithms_generated.py`

### Module Docstring

```text
Regression smoke tests for all generated pvx algorithm modules.

This test verifies that every algorithm listed in `pvxalgorithms.registry`
is importable and can process a synthetic stereo signal while returning
finite 2D output and implemented metadata status.
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
**Functions:** `write_stereo_tone`, `write_mono_tone`

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

## `tests/test_docs_coverage.py`

**Purpose:** Documentation coverage checks for CLI flags.

**Classes:** None
**Functions:** `_string_literal`, `_iter_cli_sources`, `_tool_name_for_path`, `extract_flags_from_code`, `load_doc_pairs`, `test_cli_flag_docs_match_parser_definitions`, `test_readme_long_flags_exist_in_parser_sources`

### Module Docstring

```text
Documentation coverage checks for CLI flags.
```

## `tests/test_docs_pdf.py`

**Purpose:** Tests for scripts_generate_docs_pdf.py helpers.

**Classes:** `TestDocsPdfHelpers`
**Functions:** None

**Help commands:** `python3 tests/test_docs_pdf.py`

### Module Docstring

```text
Tests for scripts_generate_docs_pdf.py helpers.
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

## `tests/test_transient_and_stereo.py`

**Purpose:** Tests for hybrid transient processing and stereo coherence modes.

**Classes:** `TestTransientHybridAndStereo`
**Functions:** `_build_args`, `_phase_drift_internal`

**Help commands:** `python3 tests/test_transient_and_stereo.py`

### Module Docstring

```text
Tests for hybrid transient processing and stereo coherence modes.
```
