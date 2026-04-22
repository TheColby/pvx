"""Declarative pipeline configuration for ``pvx.augment``.

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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import Pipeline, Transform, get_transform, list_transforms


def _load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON manifest into a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pipeline manifest not found: {p}")

    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load .yaml manifests. Install with: pip install pyyaml"
            ) from exc
        manifest = yaml.safe_load(text)
    elif suffix == ".json":
        manifest = json.loads(text)
    else:
        # Try YAML first, fall back to JSON
        try:
            import yaml

            manifest = yaml.safe_load(text)
        except ImportError:
            manifest = json.loads(text)

    if not isinstance(manifest, dict):
        raise ValueError(
            f"manifest must be a mapping at the top level, got {type(manifest).__name__}"
        )
    return manifest


def _camel_to_snake(name: str) -> str:
    import re

    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _builtin_transform_index() -> dict[str, type]:
    """Return ``{snake_case_name: cls}`` for all built-in pvx.augment transforms."""
    from . import (
        AddNoise,
        BackgroundMixer,
        BandwidthLimiter,
        BitCrusher,
        ClippingSimulator,
        CodecDegradation,
        EQPerturber,
        Fade,
        FixedLengthCrop,
        GainPerturber,
        Identity,
        ImpulseNoise,
        ImpulseResponseConvolver,
        Normalizer,
        PitchShift,
        PitchShiftSimple,
        Reverse,
        RoomSimulator,
        SpecAugment,
        SpectralNoise,
        TimeShift,
        TimeStretch,
        TrimSilence,
    )

    classes = [
        AddNoise,
        BackgroundMixer,
        BandwidthLimiter,
        BitCrusher,
        ClippingSimulator,
        CodecDegradation,
        EQPerturber,
        Fade,
        FixedLengthCrop,
        GainPerturber,
        Identity,
        ImpulseNoise,
        ImpulseResponseConvolver,
        Normalizer,
        PitchShift,
        PitchShiftSimple,
        Reverse,
        RoomSimulator,
        SpecAugment,
        SpectralNoise,
        TimeShift,
        TimeStretch,
        TrimSilence,
    ]
    return {_camel_to_snake(cls.__name__): cls for cls in classes}


def _torch_transform_index() -> dict[str, type]:
    """Return ``{snake_case_name: cls}`` for GPU TorchTransform classes (if torch installed)."""
    try:
        from . import (
            TorchAddNoise,
            TorchClippingSimulator,
            TorchEQPerturber,
            TorchGainPerturber,
            TorchMixup,
            TorchNormalizer,
            TorchPitchShift,
            TorchRoomSimulator,
            TorchSpecAugment,
            TorchTimeStretch,
        )
    except ImportError:
        return {}

    classes = [
        TorchAddNoise,
        TorchClippingSimulator,
        TorchEQPerturber,
        TorchGainPerturber,
        TorchMixup,
        TorchNormalizer,
        TorchPitchShift,
        TorchRoomSimulator,
        TorchSpecAugment,
        TorchTimeStretch,
    ]
    # Strip the leading "torch_" so YAML uses the same name as the CPU version
    out: dict[str, type] = {}
    for cls in classes:
        snake = _camel_to_snake(cls.__name__)
        out[snake] = cls
        if snake.startswith("torch_"):
            out[snake[len("torch_") :]] = cls
    return out


def _resolve_transform_class(name: str, *, gpu: bool) -> type:
    """Resolve a transform name to a class.

    Lookup order:
      1. plugin registry (:func:`pvx.augment.get_transform`)
      2. GPU-only built-ins (when ``gpu=True``)
      3. CPU built-ins (always)
    """
    # 1. Plugin registry
    try:
        return get_transform(name)
    except KeyError:
        pass

    # 2. GPU built-ins
    if gpu:
        torch_index = _torch_transform_index()
        if name in torch_index:
            return torch_index[name]

    # 3. CPU built-ins
    cpu_index = _builtin_transform_index()
    if name in cpu_index:
        return cpu_index[name]

    available = sorted(
        set(cpu_index) | set(_torch_transform_index() if gpu else {}) | set(list_transforms())
    )
    raise KeyError(f"Unknown transform: {name!r}. Available: {', '.join(available) or '(none)'}")


def _build_transform(spec: dict[str, Any], *, gpu: bool):
    if "name" not in spec:
        raise ValueError(f"transform spec missing required 'name' field: {spec!r}")
    name = str(spec["name"])
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(
            f"transform '{name}' params must be a mapping, got {type(params).__name__}"
        )
    # YAML lists become Python lists; many transforms expect tuples for ranges.
    # Convert top-level lists of length 2 to tuples for parameters that look like ranges.
    converted: dict[str, Any] = {}
    for key, value in params.items():
        if (
            isinstance(value, list)
            and len(value) == 2
            and all(isinstance(v, (int, float)) for v in value)
        ):
            converted[key] = tuple(value)
        else:
            converted[key] = value
    cls = _resolve_transform_class(name, gpu=gpu)
    return cls(**converted)


def load_pipeline(path: str | Path) -> Pipeline:
    """Build a CPU :class:`pvx.augment.Pipeline` from a YAML/JSON manifest.

    Parameters
    ----------
    path:
        Path to a ``.yaml``, ``.yml``, or ``.json`` manifest.

    Returns
    -------
    Pipeline
        A configured NumPy-backed pipeline.
    """
    manifest = _load_manifest(path)
    seed = manifest.get("seed")
    p = float(manifest.get("p", 1.0))
    transforms_spec = manifest.get("transforms")
    if not isinstance(transforms_spec, list) or not transforms_spec:
        raise ValueError("manifest 'transforms' must be a non-empty list")

    transforms: list[Transform] = []
    for spec in transforms_spec:
        if not isinstance(spec, dict):
            raise ValueError(f"each transform entry must be a mapping, got {spec!r}")
        transforms.append(_build_transform(spec, gpu=False))

    return Pipeline(transforms, seed=seed, p=p)


def load_torch_pipeline(path: str | Path):
    """Build a :class:`pvx.augment.gpu.TorchPipeline` from a YAML/JSON manifest.

    Same manifest format as :func:`load_pipeline`. Transform names resolve
    to GPU-native ``Torch*`` classes when available, falling back to the
    plugin registry.
    """
    try:
        from .gpu import TorchPipeline
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for load_torch_pipeline. Install with: pip install 'pvx[torch]'"
        ) from exc

    manifest = _load_manifest(path)
    seed = manifest.get("seed")
    transforms_spec = manifest.get("transforms")
    if not isinstance(transforms_spec, list) or not transforms_spec:
        raise ValueError("manifest 'transforms' must be a non-empty list")

    transforms = []
    for spec in transforms_spec:
        if not isinstance(spec, dict):
            raise ValueError(f"each transform entry must be a mapping, got {spec!r}")
        transforms.append(_build_transform(spec, gpu=True))

    return TorchPipeline(transforms, seed=seed)
