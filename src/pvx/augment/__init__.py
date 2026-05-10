"""pvx.augment — Audio data augmentation Python API.

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
"""

from __future__ import annotations

# Codec / degradation
from .codec import (
    BandwidthLimiter,
    BitCrusher,
    CodecDegradation,
)

# Core primitives
from .core import (
    Identity,
    OneOf,
    Pipeline,
    RandomApply,
    SomeOf,
    Transform,
    TransformResult,
    fingerprint_audio,
    get_transform,
    list_transforms,
    load_audio,
    register_transform,
    save_audio,
)

# Noise
from .noise import (
    AddNoise,
    BackgroundMixer,
    ImpulseNoise,
)

# Room / reverb
from .room import (
    ImpulseResponseConvolver,
    RoomSimulator,
)

# Spectral
from .spectral import (
    EQPerturber,
    PitchShiftSimple,
    SpecAugment,
    SpectralNoise,
)

# Time domain
from .time_domain import (
    AudioDropout,
    ClippingSimulator,
    Fade,
    FixedLengthCrop,
    GainPerturber,
    Normalizer,
    PitchShift,
    Reverse,
    StereoWidener,
    TimeShift,
    TimeStretch,
    Tremolo,
    TrimSilence,
)

__all__ = [
    "AddNoise",
    "AudioDropout",
    "BackgroundMixer",
    "BandwidthLimiter",
    "BitCrusher",
    "ClippingSimulator",
    "CodecDegradation",
    "EQPerturber",
    "Fade",
    "FixedLengthCrop",
    "GainPerturber",
    "Identity",
    "ImpulseNoise",
    "ImpulseResponseConvolver",
    "Normalizer",
    "OneOf",
    "Pipeline",
    "PitchShift",
    "PitchShiftSimple",
    "RandomApply",
    "Reverse",
    "RoomSimulator",
    "SomeOf",
    "SpecAugment",
    "SpectralNoise",
    "StereoWidener",
    "TimeShift",
    "TimeStretch",
    "Transform",
    "TransformResult",
    "Tremolo",
    "TrimSilence",
    "fingerprint_audio",
    "get_transform",
    "list_transforms",
    "load_audio",
    "register_transform",
    "save_audio",
]

# ---------------------------------------------------------------------------
# Convenience factory: intent presets
# ---------------------------------------------------------------------------


def asr_pipeline(seed: int = 42) -> Pipeline:
    """Return a ready-to-use augmentation pipeline tuned for ASR training.

    Applies mild perturbations that preserve phonetic content while
    improving robustness to microphone quality, room acoustics, and
    background noise.

    Parameters
    ----------
    seed:
        Default seed for reproducibility.

    Returns
    -------
    Pipeline
    """
    return Pipeline(
        [
            GainPerturber(gain_db=(-3.0, 3.0), p=0.8),
            RoomSimulator(rt60_range=(0.05, 0.6), wet_range=(0.1, 0.5), p=0.4),
            AddNoise(snr_db=(15.0, 40.0), noise_type="pink", p=0.5),
            CodecDegradation(codec="random", p=0.2),
            SpecAugment(
                freq_mask_param=20, time_mask_param=30, num_freq_masks=1, num_time_masks=1, p=0.5
            ),
        ],
        seed=seed,
    )


def music_pipeline(seed: int = 42) -> Pipeline:
    """Return an augmentation pipeline tuned for music information retrieval.

    Applies moderate pitch, timing, and spectral perturbations suitable for
    training beat trackers, chord estimators, and genre classifiers.

    Parameters
    ----------
    seed:
        Default seed for reproducibility.

    Returns
    -------
    Pipeline
    """
    return Pipeline(
        [
            GainPerturber(gain_db=(-6.0, 6.0), p=0.9),
            EQPerturber(n_bands=4, gain_db_range=(-6.0, 6.0), p=0.6),
            RoomSimulator(rt60_range=(0.1, 1.5), wet_range=(0.1, 0.5), p=0.3),
            OneOf(
                [
                    AddNoise(snr_db=(20.0, 40.0), noise_type="white"),
                    ImpulseNoise(rate=1.0, p=1.0),
                    Identity(),
                ],
                p=0.4,
            ),
            SpecAugment(
                freq_mask_param=30, time_mask_param=50, num_freq_masks=2, num_time_masks=2, p=0.4
            ),
        ],
        seed=seed,
    )


def speech_enhancement_pipeline(seed: int = 42) -> Pipeline:
    """Return an augmentation pipeline for speech enhancement model training.

    Generates challenging noisy/reverberant conditions for training
    denoising and dereverb models.  The augmented audio serves as
    *input* (noisy); pair it with the clean original as *target*.

    Parameters
    ----------
    seed:
        Default seed for reproducibility.

    Returns
    -------
    Pipeline
    """
    return Pipeline(
        [
            GainPerturber(gain_db=(-6.0, 6.0), p=1.0),
            RoomSimulator(rt60_range=(0.1, 2.0), wet_range=(0.2, 0.9), p=0.7),
            AddNoise(snr_db=(0.0, 20.0), noise_type="pink", p=0.8),
            OneOf(
                [
                    CodecDegradation(codec="voip_narrow"),
                    CodecDegradation(codec="telephone"),
                    Identity(),
                ],
                p=0.3,
            ),
            ImpulseNoise(rate=0.5, amplitude_range=(0.02, 0.15), p=0.15),
        ],
        seed=seed,
    )


def contrastive_pipeline(seed: int = 42) -> tuple[Pipeline, Pipeline]:
    """Return two correlated augmentation pipelines for self-supervised learning.

    The two pipelines share the same augmentation types but use different
    seeds, producing two statistically-related but distinct views of the
    same audio clip — suitable for contrastive objectives like SimCLR,
    MoCo, or BYOL.

    Parameters
    ----------
    seed:
        Base seed; view A uses *seed*, view B uses *seed + 1*.

    Returns
    -------
    tuple[Pipeline, Pipeline]
        ``(pipeline_a, pipeline_b)``
    """

    def _build(s: int) -> Pipeline:
        return Pipeline(
            [
                GainPerturber(gain_db=(-6.0, 6.0), p=0.8),
                RoomSimulator(rt60_range=(0.05, 1.0), wet_range=(0.1, 0.6), p=0.5),
                AddNoise(snr_db=(10.0, 35.0), noise_type="pink", p=0.5),
                EQPerturber(n_bands=3, gain_db_range=(-6.0, 6.0), p=0.5),
                SpecAugment(
                    freq_mask_param=25,
                    time_mask_param=40,
                    num_freq_masks=2,
                    num_time_masks=2,
                    p=0.5,
                ),
                TimeShift(shift=(-0.1, 0.1), p=0.3),
            ],
            seed=s,
        )

    return _build(seed), _build(seed + 1)


# GPU-accelerated transforms (optional — requires torch)
# Guarded import: only available when PyTorch is installed.
try:
    from .gpu import (
        NumpyTransformAdapter,
        TorchAddNoise,
        TorchClippingSimulator,
        TorchEQPerturber,
        TorchGainPerturber,
        TorchMixup,
        TorchNormalizer,
        TorchPipeline,
        TorchPitchShift,
        TorchRoomSimulator,
        TorchSpecAugment,
        TorchTimeStretch,
        TorchTransform,
        batch_process_files,
    )

    __all__ += [
        "NumpyTransformAdapter",
        "TorchAddNoise",
        "TorchClippingSimulator",
        "TorchEQPerturber",
        "TorchGainPerturber",
        "TorchMixup",
        "TorchNormalizer",
        "TorchPipeline",
        "TorchPitchShift",
        "TorchRoomSimulator",
        "TorchSpecAugment",
        "TorchTimeStretch",
        "TorchTransform",
        "batch_process_files",
    ]
except ImportError:
    pass  # PyTorch not installed — GPU transforms unavailable

# Streaming augmentation
from .streaming import (
    stream_augment,
    stream_augment_directory,
    stream_augment_file,
)

__all__ += [
    "stream_augment",
    "stream_augment_directory",
    "stream_augment_file",
]

# IR database (no extra dependencies)
from .ir_database import IRDatabase

__all__ += ["IRDatabase"]

# YAML/JSON pipeline configuration
from .config import load_pipeline, load_torch_pipeline

__all__ += ["load_pipeline", "load_torch_pipeline"]

# Version
__version__ = "0.1.0a1"
