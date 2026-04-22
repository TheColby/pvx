# pvx.augment

NumPy-native, composable audio augmentation API. All transforms follow the
uniform `(audio, sr, seed=None) -> (audio, sr)` interface and are fully
reproducible given a fixed seed.

## Pipeline & core primitives

::: pvx.augment.Pipeline
::: pvx.augment.Transform
::: pvx.augment.OneOf
::: pvx.augment.SomeOf
::: pvx.augment.RandomApply
::: pvx.augment.Identity

## Registry

::: pvx.augment.register_transform
::: pvx.augment.get_transform
::: pvx.augment.list_transforms

## Intent presets

::: pvx.augment.asr_pipeline
::: pvx.augment.music_pipeline
::: pvx.augment.speech_enhancement_pipeline
::: pvx.augment.contrastive_pipeline

## Noise

::: pvx.augment.AddNoise
::: pvx.augment.BackgroundMixer
::: pvx.augment.ImpulseNoise

## Room / reverb

::: pvx.augment.RoomSimulator
::: pvx.augment.ImpulseResponseConvolver

## Codec / degradation

::: pvx.augment.CodecDegradation
::: pvx.augment.BitCrusher
::: pvx.augment.BandwidthLimiter

## Spectral

::: pvx.augment.SpecAugment
::: pvx.augment.EQPerturber
::: pvx.augment.SpectralNoise
::: pvx.augment.PitchShiftSimple

## Time domain

::: pvx.augment.GainPerturber
::: pvx.augment.Normalizer
::: pvx.augment.ClippingSimulator
::: pvx.augment.TimeShift
::: pvx.augment.Reverse
::: pvx.augment.Fade
::: pvx.augment.TrimSilence
::: pvx.augment.FixedLengthCrop
::: pvx.augment.TimeStretch
::: pvx.augment.PitchShift

## I/O helpers

::: pvx.augment.load_audio
::: pvx.augment.save_audio
::: pvx.augment.fingerprint_audio
