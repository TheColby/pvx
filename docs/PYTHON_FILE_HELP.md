# Python File Documentation and Help

Comprehensive reference for every Python file in this repository.

Total Python files documented: **154**

## Contents

- [`main.py`](#mainpy)
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
- [`scripts_generate_python_docs.py`](#scriptsgeneratepythondocspy)
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
- [`src/pvx/algorithms/spatial_and_multichannel/ambisonics_encode_decode.py`](#srcpvxalgorithmsspatialandmultichannelambisonicsencodedecodepy)
- [`src/pvx/algorithms/spatial_and_multichannel/binaural_hrtf_rendering.py`](#srcpvxalgorithmsspatialandmultichannelbinauralhrtfrenderingpy)
- [`src/pvx/algorithms/spatial_and_multichannel/gcc_phat_localization.py`](#srcpvxalgorithmsspatialandmultichannelgccphatlocalizationpy)
- [`src/pvx/algorithms/spatial_and_multichannel/gsc_beamforming.py`](#srcpvxalgorithmsspatialandmultichannelgscbeamformingpy)
- [`src/pvx/algorithms/spatial_and_multichannel/mid_side_adaptive_widening.py`](#srcpvxalgorithmsspatialandmultichannelmidsideadaptivewideningpy)
- [`src/pvx/algorithms/spatial_and_multichannel/mvdr_beamforming.py`](#srcpvxalgorithmsspatialandmultichannelmvdrbeamformingpy)
- [`src/pvx/algorithms/spatial_and_multichannel/stereo_decorrelation_for_width.py`](#srcpvxalgorithmsspatialandmultichannelstereodecorrelationforwidthpy)
- [`src/pvx/algorithms/spatial_and_multichannel/upmix_downmix_with_phase_coherent_routing.py`](#srcpvxalgorithmsspatialandmultichannelupmixdownmixwithphasecoherentroutingpy)
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
- [`src/pvx/cli/main.py`](#srcpvxclimainpy)
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
- [`src/pvx/core/common.py`](#srcpvxcorecommonpy)
- [`src/pvx/core/voc.py`](#srcpvxcorevocpy)
- [`src/pvxalgorithms/__init__.py`](#srcpvxalgorithmsinitpy)
- [`src/pvxalgorithms/base.py`](#srcpvxalgorithmsbasepy)
- [`src/pvxalgorithms/registry.py`](#srcpvxalgorithmsregistrypy)
- [`tests/test_algorithms_generated.py`](#teststestalgorithmsgeneratedpy)
- [`tests/test_cli_regression.py`](#teststestcliregressionpy)
- [`tests/test_dsp.py`](#teststestdsppy)
- [`tests/test_microtonal.py`](#teststestmicrotonalpy)

## `main.py`

**Purpose:** Compatibility wrapper.

**Classes:** None
**Functions:** None

**Help commands:** `python3 main.py --help`

### CLI Help Snapshot

```text
usage: main.py [-h] [--list-tools] [--list-algorithm-package] [--show-docs]

pvx toolkit navigator (tool list, algorithm package, and docs index)

options:
  -h, --help            show this help message and exit
  --list-tools          Print the primary pvx CLI tools
  --list-algorithm-package
                        Print location of pvxalgorithms package and registry
                        files
  --show-docs           Print key generated documentation paths
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.main` after the
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
usage: pvxconform.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                     [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                     [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                     [--quiet] [--silent] [--normalize {none,peak,rms}]
                     [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                     [--subtype SUBTYPE] [--n-fft N_FFT]
                     [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                     [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                     [--kaiser-beta KAISER_BETA] [--no-center]
                     [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                     --map MAP [--crossfade-ms CROSSFADE_MS]
                     [--resample-mode {auto,fft,linear}]
                     inputs [inputs ...]

Conform audio to a CSV map. CSV columns: start_sec,end_sec,stretch, and one
pitch field: pitch_semitones or pitch_cents or pitch_ratio

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _conform)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --map MAP             CSV map path
  --crossfade-ms CROSSFADE_MS
                        Segment crossfade in milliseconds
  --resample-mode {auto,fft,linear}
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
usage: pvxdenoise.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                     [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                     [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                     [--quiet] [--silent] [--normalize {none,peak,rms}]
                     [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                     [--subtype SUBTYPE] [--n-fft N_FFT]
                     [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                     [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                     [--kaiser-beta KAISER_BETA] [--no-center]
                     [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                     [--noise-seconds NOISE_SECONDS] [--noise-file NOISE_FILE]
                     [--reduction-db REDUCTION_DB] [--floor FLOOR]
                     [--smooth SMOOTH]
                     inputs [inputs ...]

Spectral subtraction denoiser

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _denoise)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --noise-seconds NOISE_SECONDS
                        Noise profile duration from start
  --noise-file NOISE_FILE
                        Optional external noise reference
  --reduction-db REDUCTION_DB
                        Reduction strength in dB
  --floor FLOOR         Noise floor multiplier
  --smooth SMOOTH       Temporal smoothing frames
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
usage: pvxdeverb.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                    [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                    [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                    [--quiet] [--silent] [--normalize {none,peak,rms}]
                    [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                    [--subtype SUBTYPE] [--n-fft N_FFT]
                    [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                    [--kaiser-beta KAISER_BETA] [--no-center]
                    [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                    [--strength STRENGTH] [--decay DECAY] [--floor FLOOR]
                    inputs [inputs ...]

Reduce spectral tails / reverberant smear

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _deverb)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --strength STRENGTH   Tail suppression strength 0..1
  --decay DECAY         Tail memory decay 0..1
  --floor FLOOR         Per-bin floor multiplier
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
usage: pvxformant.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                     [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                     [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                     [--quiet] [--silent] [--normalize {none,peak,rms}]
                     [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                     [--subtype SUBTYPE] [--n-fft N_FFT]
                     [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                     [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                     [--kaiser-beta KAISER_BETA] [--no-center]
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
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _formant)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --pitch-shift-semitones PITCH_SHIFT_SEMITONES
                        Optional pitch shift before formant stage
  --pitch-shift-cents PITCH_SHIFT_CENTS
                        Additional microtonal pitch shift in cents before
                        formant stage
  --formant-shift-ratio FORMANT_SHIFT_RATIO
                        Formant ratio (>1 up, <1 down)
  --mode {shift,preserve}
  --formant-lifter FORMANT_LIFTER
  --formant-max-gain-db FORMANT_MAX_GAIN_DB
  --resample-mode {auto,fft,linear}
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
usage: pvxfreeze.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                    [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                    [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                    [--quiet] [--silent] [--normalize {none,peak,rms}]
                    [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                    [--subtype SUBTYPE] [--n-fft N_FFT]
                    [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                    [--kaiser-beta KAISER_BETA] [--no-center]
                    [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                    [--freeze-time FREEZE_TIME] [--duration DURATION]
                    [--random-phase]
                    inputs [inputs ...]

Freeze a spectral slice into a sustained output

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _freeze)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 256)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --freeze-time FREEZE_TIME
                        Freeze anchor time in seconds
  --duration DURATION   Output freeze duration in seconds
  --random-phase        Add subtle phase randomization per frame
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
usage: pvxharmonize.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                       [--output-format OUTPUT_FORMAT] [--overwrite]
                       [--dry-run]
                       [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                       [--quiet] [--silent] [--normalize {none,peak,rms}]
                       [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                       [--subtype SUBTYPE] [--n-fft N_FFT]
                       [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                       [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                       [--kaiser-beta KAISER_BETA] [--no-center]
                       [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                       [--intervals INTERVALS]
                       [--intervals-cents INTERVALS_CENTS] [--gains GAINS]
                       [--pans PANS] [--force-stereo]
                       [--resample-mode {auto,fft,linear}]
                       inputs [inputs ...]

Generate harmonized voices from an input

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _harm)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --intervals INTERVALS
                        Comma-separated semitone intervals per voice (supports
                        fractional values)
  --intervals-cents INTERVALS_CENTS
                        Optional cents offsets per voice, added to --intervals
                        (e.g. 0,14,-12)
  --gains GAINS         Optional comma-separated linear gain per voice
  --pans PANS           Optional comma-separated pan per voice [-1..1]
  --force-stereo        Mix result as stereo with panning
  --resample-mode {auto,fft,linear}
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
usage: pvxlayer.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                   [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                   [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                   [--quiet] [--silent] [--normalize {none,peak,rms}]
                   [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                   [--subtype SUBTYPE] [--n-fft N_FFT]
                   [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                   [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                   [--kaiser-beta KAISER_BETA] [--no-center]
                   [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
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
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _layer)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --harmonic-stretch HARMONIC_STRETCH
  --harmonic-pitch-semitones HARMONIC_PITCH_SEMITONES
  --harmonic-pitch-cents HARMONIC_PITCH_CENTS
  --percussive-stretch PERCUSSIVE_STRETCH
  --percussive-pitch-semitones PERCUSSIVE_PITCH_SEMITONES
  --percussive-pitch-cents PERCUSSIVE_PITCH_CENTS
  --harmonic-gain HARMONIC_GAIN
  --percussive-gain PERCUSSIVE_GAIN
  --harmonic-kernel HARMONIC_KERNEL
  --percussive-kernel PERCUSSIVE_KERNEL
  --resample-mode {auto,fft,linear}
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
usage: pvxmorph.py [-h] [-o OUTPUT] [--alpha ALPHA]
                   [--normalize {none,peak,rms}] [--peak-dbfs PEAK_DBFS]
                   [--rms-dbfs RMS_DBFS] [--clip] [--subtype SUBTYPE]
                   [--overwrite]
                   [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                   [--quiet] [--silent] [--n-fft N_FFT]
                   [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                   [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                   [--kaiser-beta KAISER_BETA] [--no-center]
                   [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                   input_a input_b

Morph two audio files in the STFT domain

positional arguments:
  input_a               Input A
  input_b               Input B

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output file path
  --alpha ALPHA         Morph amount 0..1 (0=A, 1=B)
  --normalize {none,peak,rms}
  --peak-dbfs PEAK_DBFS
  --rms-dbfs RMS_DBFS
  --clip
  --subtype SUBTYPE
  --overwrite
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
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
usage: pvxretune.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                    [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                    [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                    [--quiet] [--silent] [--normalize {none,peak,rms}]
                    [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                    [--subtype SUBTYPE] [--n-fft N_FFT]
                    [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                    [--kaiser-beta KAISER_BETA] [--no-center]
                    [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                    [--root ROOT] [--scale {chromatic,major,minor,pentatonic}]
                    [--scale-cents SCALE_CENTS] [--strength STRENGTH]
                    [--chunk-ms CHUNK_MS] [--overlap-ms OVERLAP_MS]
                    [--f0-min F0_MIN] [--f0-max F0_MAX]
                    [--resample-mode {auto,fft,linear}]
                    inputs [inputs ...]

Monophonic retune toward a musical scale

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _retune)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --root ROOT           Scale root note (C,C#,D,...,B)
  --scale {chromatic,major,minor,pentatonic}
                        Named scale for 12-TET quantization
  --scale-cents SCALE_CENTS
                        Optional comma-separated microtonal scale degrees in
                        cents within one octave, relative to --root (example:
                        0,90,204,294,408,498,612,702,816,906,1020,1110)
  --strength STRENGTH   Correction strength 0..1
  --chunk-ms CHUNK_MS   Analysis/process chunk duration in ms
  --overlap-ms OVERLAP_MS
                        Chunk overlap in ms
  --f0-min F0_MIN
  --f0-max F0_MAX
  --resample-mode {auto,fft,linear}
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
usage: pvxtransient.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                       [--output-format OUTPUT_FORMAT] [--overwrite]
                       [--dry-run]
                       [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                       [--quiet] [--silent] [--normalize {none,peak,rms}]
                       [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                       [--subtype SUBTYPE] [--n-fft N_FFT]
                       [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                       [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                       [--kaiser-beta KAISER_BETA] [--no-center]
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
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _trans)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 256)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --time-stretch TIME_STRETCH
  --target-duration TARGET_DURATION
                        Target duration in seconds
  --pitch-shift-semitones PITCH_SHIFT_SEMITONES
  --pitch-shift-cents PITCH_SHIFT_CENTS
                        Optional microtonal pitch shift in cents (added to
                        --pitch-shift-semitones)
  --pitch-shift-ratio PITCH_SHIFT_RATIO
  --transient-threshold TRANSIENT_THRESHOLD
  --resample-mode {auto,fft,linear}
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
usage: pvxunison.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                    [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                    [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                    [--quiet] [--silent] [--normalize {none,peak,rms}]
                    [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                    [--subtype SUBTYPE] [--n-fft N_FFT]
                    [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                    [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                    [--kaiser-beta KAISER_BETA] [--no-center]
                    [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                    [--voices VOICES] [--detune-cents DETUNE_CENTS]
                    [--width WIDTH] [--dry-mix DRY_MIX]
                    [--resample-mode {auto,fft,linear}]
                    inputs [inputs ...]

Stereo unison thickener

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _unison)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --voices VOICES       Number of unison voices
  --detune-cents DETUNE_CENTS
                        Total detune span in cents
  --width WIDTH         Stereo width multiplier 0..2
  --dry-mix DRY_MIX     Dry signal mix amount
  --resample-mode {auto,fft,linear}
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
                 [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                 [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                 [--quiet] [--silent] [--n-fft N_FFT]
                 [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                 [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                 [--kaiser-beta KAISER_BETA] [--no-center]
                 [--phase-locking {off,identity}] [--transient-preserve]
                 [--transient-threshold TRANSIENT_THRESHOLD] [--fourier-sync]
                 [--fourier-sync-min-fft FOURIER_SYNC_MIN_FFT]
                 [--fourier-sync-max-fft FOURIER_SYNC_MAX_FFT]
                 [--fourier-sync-smooth FOURIER_SYNC_SMOOTH]
                 [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                 [--time-stretch TIME_STRETCH]
                 [--target-duration TARGET_DURATION]
                 [--pitch-shift-semitones PITCH_SHIFT_SEMITONES |
                 --pitch-shift-cents PITCH_SHIFT_CENTS |
                 --pitch-shift-ratio PITCH_SHIFT_RATIO |
                 --target-f0 TARGET_F0] [--analysis-channel {first,mix}]
                 [--f0-min F0_MIN] [--f0-max F0_MAX]
                 [--pitch-mode {standard,formant-preserving}]
                 [--formant-lifter FORMANT_LIFTER]
                 [--formant-strength FORMANT_STRENGTH]
                 [--formant-max-gain-db FORMANT_MAX_GAIN_DB]
                 [--target-sample-rate TARGET_SAMPLE_RATE]
                 [--resample-mode {auto,fft,linear}]
                 [--normalize {none,peak,rms}] [--peak-dbfs PEAK_DBFS]
                 [--rms-dbfs RMS_DBFS] [--clip] [--subtype SUBTYPE]
                 inputs [inputs ...]

Phase-vocoder CLI for multi-file, multi-channel time stretching and pitch
shifting.

positional arguments:
  inputs                Input audio files (wav/flac/aiff/ogg/etc)

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Directory for output files (default: same directory as
                        each input)
  --suffix SUFFIX       Suffix appended to output filename stem (default: _pv)
  --output-format OUTPUT_FORMAT
                        Output format/extension (e.g. wav, flac, aiff).
                        Default: keep input extension.
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve settings without writing files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output

STFT / vocoder parameters:
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type (default: hann)
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable center padding in STFT/ISTFT
  --phase-locking {off,identity}
                        Inter-bin phase locking mode for transient fidelity
                        (default: identity)
  --transient-preserve  Enable transient phase resets based on spectral flux
  --transient-threshold TRANSIENT_THRESHOLD
                        Spectral-flux multiplier for transient detection
                        (default: 2.0)
  --fourier-sync        Enable 
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
usage: pvxwarp.py [-h] [-o OUTPUT_DIR] [--suffix SUFFIX]
                  [--output-format OUTPUT_FORMAT] [--overwrite] [--dry-run]
                  [--verbosity {silent,quiet,normal,verbose,debug}] [-v]
                  [--quiet] [--silent] [--normalize {none,peak,rms}]
                  [--peak-dbfs PEAK_DBFS] [--rms-dbfs RMS_DBFS] [--clip]
                  [--subtype SUBTYPE] [--n-fft N_FFT]
                  [--win-length WIN_LENGTH] [--hop-size HOP_SIZE]
                  [--window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}]
                  [--kaiser-beta KAISER_BETA] [--no-center]
                  [--device {auto,cpu,cuda}] [--cuda-device CUDA_DEVICE]
                  --map MAP [--crossfade-ms CROSSFADE_MS]
                  [--resample-mode {auto,fft,linear}]
                  inputs [inputs ...]

Apply variable time-stretch map from CSV

positional arguments:
  inputs                Input files or glob patterns

options:
  -h, --help            show this help message and exit
  -o, --output-dir OUTPUT_DIR
                        Output directory
  --suffix SUFFIX       Output filename suffix (default: _warp)
  --output-format OUTPUT_FORMAT
                        Output extension/format
  --overwrite           Overwrite existing outputs
  --dry-run             Resolve and print, but do not write files
  --verbosity {silent,quiet,normal,verbose,debug}
                        Console verbosity level
  -v, --verbose         Increase verbosity (repeat for extra detail)
  --quiet               Reduce output and hide status bars
  --silent              Suppress all console output
  --normalize {none,peak,rms}
                        Output normalization
  --peak-dbfs PEAK_DBFS
                        Target peak dBFS for peak normalization
  --rms-dbfs RMS_DBFS   Target RMS dBFS for RMS normalization
  --clip                Hard clip output to [-1, 1]
  --subtype SUBTYPE     libsndfile subtype (PCM_16, PCM_24, FLOAT, etc)
  --n-fft N_FFT         FFT size (default: 2048)
  --win-length WIN_LENGTH
                        Window length in samples (default: 2048)
  --hop-size HOP_SIZE   Hop size in samples (default: 512)
  --window {hann,hamming,blackman,blackmanharris,nuttall,flattop,blackman_nuttall,exact_blackman,sine,bartlett,boxcar,triangular,bartlett_hann,tukey,tukey_0p1,tukey_0p25,tukey_0p75,tukey_0p9,parzen,lanczos,welch,gaussian_0p25,gaussian_0p35,gaussian_0p45,gaussian_0p55,gaussian_0p65,general_gaussian_1p5_0p35,general_gaussian_2p0_0p35,general_gaussian_3p0_0p35,general_gaussian_4p0_0p35,exponential_0p25,exponential_0p5,exponential_1p0,cauchy_0p5,cauchy_1p0,cauchy_2p0,cosine_power_2,cosine_power_3,cosine_power_4,hann_poisson_0p5,hann_poisson_1p0,hann_poisson_2p0,general_hamming_0p50,general_hamming_0p60,general_hamming_0p70,general_hamming_0p80,bohman,cosine,kaiser,rect}
                        Window type
  --kaiser-beta KAISER_BETA
                        Kaiser window beta parameter used when --window kaiser
                        (default: 14.0)
  --no-center           Disable centered framing
  --device {auto,cpu,cuda}
                        Compute device: auto (prefer CUDA), cpu, or cuda
  --cuda-device CUDA_DEVICE
                        CUDA device index used when --device is auto/cuda
                        (default: 0)
  --map MAP             CSV map with start_sec,end_sec,stretch
  --crossfade-ms CROSSFADE_MS
  --resample-mode {auto,fft,linear}
```

### Module Docstring

```text
Compatibility wrapper.

This root module forwards imports/execution to `pvx.cli.pvxwarp` after the
src-layout migration.
```

## `scripts_generate_python_docs.py`

**Purpose:** Generate comprehensive documentation for every Python file in the repository.

**Classes:** None
**Functions:** `rel`, `safe_read`, `parse_module`, `cli_help`, `extract_algorithm_params`, `generate_algorithm_param_doc`, `generate_python_help_doc`, `main`

**Help commands:** `python3 scripts_generate_python_docs.py`, `python3 scripts_generate_python_docs.py --help`

### Module Docstring

```text
Generate comprehensive documentation for every Python file in the repository.
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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Functions:** `coerce_audio`, `maybe_librosa`, `maybe_loudnorm`, `build_metadata`, `normalize_peak`, `ensure_length`, `resample_length`, `envelope_follower`, `soft_clip`, `stft_multi`, `istft_multi`, `spectral_sharpen`, `spectral_blur`, `hpss_split`, `time_stretch`, `pitch_shift`, `overlap_add_frames`, `granular_time_stretch`, `spectral_gate`, `spectral_subtract_denoise`, `mmse_like_denoise`, `minimum_statistics_denoise`, `simple_declick`, `simple_declip`, `dereverb_decay_subtract`, `dereverb_wpe_style`, `compressor`, `upward_compressor`, `true_peak_limit`, `transient_shaper`, `spectral_dynamics`, `split_bands`, `multiband_compression`, `mid_side_widen`, `allpass_decorrelate`, `binaural_render`, `ambisonics_encode_decode`, `simple_beamform`, `gcc_phat_delay`, `cross_synthesis`, `spectral_convolution`, `spectral_freeze`, `phase_randomize`, `formant_warp`, `resonator_bank`, `spectral_contrast_exaggerate`, `rhythmic_gate`, `ring_mod`, `spectral_tremolo`, `envelope_modulation`, `estimate_f0_track`, `nearest_scale_freq`, `variable_pitch_shift`, `detect_key_from_chroma`, `cqt_or_stft`, `icqt_or_istft`, `_dispatch_time_scale`, `_dispatch_pitch_tracking`, `_scale_cents_from_name`, `_dispatch_retune`, `_dispatch_transforms`, `_dispatch_separation`, `_dispatch_denoise`, `_dispatch_dereverb`, `_lufs_estimate`, `_dispatch_dynamics`, `_dispatch_spatial`, `_dispatch_creative`, `_dispatch_granular`, `_dispatch_analysis`, `run_algorithm`

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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

**Purpose:** Spatial and Multichannel algorithm scaffolds.

**Classes:** None
**Functions:** None

### Module Docstring

```text
Spatial and Multichannel algorithm scaffolds.
```

## `src/pvx/algorithms/spatial_and_multichannel/ambisonics_encode_decode.py`

**Purpose:** Ambisonics encode/decode.

**Algorithm ID:** `spatial_and_multichannel.ambisonics_encode_decode`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/ambisonics_encode_decode.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/ambisonics_encode_decode.py --help`

### Module Docstring

```text
Ambisonics encode/decode.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.ambisonics_encode_decode
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/binaural_hrtf_rendering.py`

**Purpose:** Binaural HRTF rendering.

**Algorithm ID:** `spatial_and_multichannel.binaural_hrtf_rendering`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/binaural_hrtf_rendering.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/binaural_hrtf_rendering.py --help`

### Module Docstring

```text
Binaural HRTF rendering.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.binaural_hrtf_rendering
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/gcc_phat_localization.py`

**Purpose:** GCC-PHAT localization.

**Algorithm ID:** `spatial_and_multichannel.gcc_phat_localization`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/gcc_phat_localization.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/gcc_phat_localization.py --help`

### Module Docstring

```text
GCC-PHAT localization.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.gcc_phat_localization
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/gsc_beamforming.py`

**Purpose:** GSC beamforming.

**Algorithm ID:** `spatial_and_multichannel.gsc_beamforming`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/gsc_beamforming.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/gsc_beamforming.py --help`

### Module Docstring

```text
GSC beamforming.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.gsc_beamforming
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/mid_side_adaptive_widening.py`

**Purpose:** Mid/Side adaptive widening.

**Algorithm ID:** `spatial_and_multichannel.mid_side_adaptive_widening`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/mid_side_adaptive_widening.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/mid_side_adaptive_widening.py --help`

### Module Docstring

```text
Mid/Side adaptive widening.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.mid_side_adaptive_widening
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/mvdr_beamforming.py`

**Purpose:** MVDR beamforming.

**Algorithm ID:** `spatial_and_multichannel.mvdr_beamforming`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/mvdr_beamforming.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/mvdr_beamforming.py --help`

### Module Docstring

```text
MVDR beamforming.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.mvdr_beamforming
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/stereo_decorrelation_for_width.py`

**Purpose:** Stereo decorrelation for width.

**Algorithm ID:** `spatial_and_multichannel.stereo_decorrelation_for_width`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/stereo_decorrelation_for_width.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/stereo_decorrelation_for_width.py --help`

### Module Docstring

```text
Stereo decorrelation for width.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.stereo_decorrelation_for_width
- Primary API: process(audio, sample_rate, **params) -> AlgorithmResult
- Backend: delegates to pvx.algorithms.base.run_algorithm()

This module is both importable and executable.
When executed directly, it prints verbose help text describing purpose,
I/O contract, and parameter-routing behavior.
```

## `src/pvx/algorithms/spatial_and_multichannel/upmix_downmix_with_phase_coherent_routing.py`

**Purpose:** Upmix/downmix with phase-coherent routing.

**Algorithm ID:** `spatial_and_multichannel.upmix_downmix_with_phase_coherent_routing`
**Theme:** `Spatial and Multichannel`
**Primary API:** `process(audio, sample_rate, **params) -> AlgorithmResult`
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

**Classes:** None
**Functions:** `process`, `module_help_text`, `build_parser`, `main`

**Help commands:** `python3 src/pvx/algorithms/spatial_and_multichannel/upmix_downmix_with_phase_coherent_routing.py`, `python3 src/pvx/algorithms/spatial_and_multichannel/upmix_downmix_with_phase_coherent_routing.py --help`

### Module Docstring

```text
Upmix/downmix with phase-coherent routing.

Comprehensive module help:
- Theme: Spatial and Multichannel
- Algorithm ID: spatial_and_multichannel.upmix_downmix_with_phase_coherent_routing
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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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
**Parameter docs:** see `/Users/cleider/dev/pvx/docs/PVX_ALGORITHM_PARAMS.md`.

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

## `src/pvx/cli/main.py`

**Purpose:** Top-level project helper CLI.

**Classes:** None
**Functions:** `build_parser`, `main`

**Help commands:** `python3 src/pvx/cli/main.py`, `python3 src/pvx/cli/main.py --help`

### Module Docstring

```text
Top-level project helper CLI.

This file exists as a lightweight entrypoint that points users to the
specialized `pvx*` command-line tools and the generated algorithm library.
Run `python3 main.py --help` to view quick navigation commands.
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

## `src/pvx/core/common.py`

**Purpose:** Shared helpers for pvx DSP command-line tools.

**Classes:** `SegmentSpec`, `StatusBar`
**Functions:** `add_console_args`, `console_level`, `is_quiet`, `is_silent`, `log_message`, `log_error`, `build_status_bar`, `add_common_io_args`, `add_vocoder_args`, `build_vocoder_config`, `validate_vocoder_args`, `resolve_inputs`, `read_audio`, `finalize_audio`, `write_output`, `default_output_path`, `parse_float_list`, `semitone_to_ratio`, `cents_to_ratio`, `time_pitch_shift_channel`, `time_pitch_shift_audio`, `read_segment_csv`, `concat_with_crossfade`, `ensure_runtime`

### Module Docstring

```text
Shared helpers for pvx DSP command-line tools.
```

## `src/pvx/core/voc.py`

**Purpose:** Multi-channel phase vocoder CLI for time and pitch manipulation.

**Classes:** `VocoderConfig`, `PitchConfig`, `JobResult`, `FourierSyncPlan`, `RuntimeConfig`, `ProgressBar`
**Functions:** `add_console_args`, `console_level`, `is_quiet`, `is_silent`, `log_message`, `log_error`, `db_to_amplitude`, `cents_to_ratio`, `_has_cupy`, `_is_cupy_array`, `_array_module`, `_to_numpy`, `_to_runtime_array`, `_as_float`, `_as_bool`, `_i0`, `add_runtime_args`, `runtime_config`, `configure_runtime`, `configure_runtime_from_args`, `ensure_runtime_dependencies`, `principal_angle`, `_cosine_series_window`, `_bartlett_window`, `_bohman_window`, `_cosine_window`, `_sine_window`, `_triangular_window`, `_bartlett_hann_window`, `_tukey_window`, `_parzen_window`, `_lanczos_window`, `_welch_window`, `_gaussian_window`, `_general_gaussian_window`, `_exponential_window`, `_cauchy_window`, `_cosine_power_window`, `_hann_poisson_window`, `_general_hamming_window`, `_kaiser_window`, `make_window`, `pad_for_framing`, `stft`, `istft`, `scaled_win_length`, `resize_spectrum_bins`, `smooth_series`, `regularize_frame_lengths`, `fill_nan_with_nearest`, `lock_fft_length_to_f0`, `build_fourier_sync_plan`, `compute_transient_flags`, `find_spectral_peaks`, `apply_identity_phase_locking`, `phase_vocoder_time_stretch`, `phase_vocoder_time_stretch_fourier_sync`, `linear_resample_1d`, `resample_1d`, `force_length`, `estimate_f0_autocorrelation`, `normalize_audio`, `cepstral_envelope`, `apply_formant_preservation`, `choose_pitch_ratio`, `resolve_base_stretch`, `compute_output_path`, `process_file`, `force_length_multi`, `resample_multi`, `validate_args`, `build_parser`, `expand_inputs`, `main`

**Help commands:** `python3 src/pvx/core/voc.py`, `python3 src/pvx/core/voc.py --help`

### Module Docstring

```text
Multi-channel phase vocoder CLI for time and pitch manipulation.
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

## `tests/test_cli_regression.py`

**Purpose:** CLI regression tests for pvxvoc end-to-end workflows.

**Classes:** `TestCLIRegression`
**Functions:** `write_stereo_tone`

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
