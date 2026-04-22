
"""Core base classes for pvx augmentation transforms.

All transforms operate on NumPy arrays with shape (channels, samples) or
(samples,) for mono. They return ``(audio, sample_rate)`` pairs and are
fully composable.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Audio shape utilities
# ---------------------------------------------------------------------------


def _to_2d(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(channels, samples)`` array and a flag for mono input."""
    if audio.ndim == 1:
        return audio[np.newaxis, :], True
    if audio.ndim == 2:
        return audio, False
    raise ValueError(f"audio must be 1-D or 2-D, got shape {audio.shape}")


def _from_2d(audio: np.ndarray, was_mono: bool) -> np.ndarray:
    if was_mono:
        return audio[0]
    return audio


# ---------------------------------------------------------------------------
# Base Transform
# ---------------------------------------------------------------------------


class Transform(ABC):
    """Abstract base class for a single augmentation transform.

    Parameters
    ----------
    p:
        Probability that this transform is applied when called.  Default 1.0.
    """

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        self.p = float(p)

    # ------------------------------------------------------------------
    def __call__(
        self,
        audio: np.ndarray,
        sr: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Apply transform with probability ``self.p``.

        Parameters
        ----------
        audio:
            Input audio as a NumPy float32/float64 array with shape
            ``(samples,)`` or ``(channels, samples)``.
        sr:
            Sample rate in Hz.
        seed:
            Optional integer seed for reproducibility.

        Returns
        -------
        tuple[np.ndarray, int]
            ``(augmented_audio, sample_rate)`` — sample rate may change.
        """
        rng = np.random.default_rng(seed)
        if rng.random() > self.p:
            return audio, sr
        return self.apply(audio, sr, rng)

    # ------------------------------------------------------------------
    @abstractmethod
    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        """Apply the transform unconditionally.

        Override this in subclasses.
        """

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({params})"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Sequential composition of transforms applied left-to-right.

    Parameters
    ----------
    transforms:
        Ordered list of :class:`Transform` (or any callable accepting
        ``(audio, sr, seed=...)``).
    seed:
        Default seed used when :meth:`__call__` is invoked without one.
    p:
        Probability that the whole pipeline is applied.
    """

    def __init__(
        self,
        transforms: Sequence[Transform | Any],
        seed: int | None = None,
        p: float = 1.0,
    ) -> None:
        self.transforms = list(transforms)
        self.seed = seed
        self.p = float(p)

    # ------------------------------------------------------------------
    def __call__(
        self,
        audio: np.ndarray,
        sr: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Run all transforms in sequence.

        Each transform receives an independent child seed derived from
        the parent seed so runs are fully reproducible.
        """
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        if rng.random() > self.p:
            return audio, sr
        for t in self.transforms:
            child_seed = int(rng.integers(0, 2**31))
            audio, sr = t(audio, sr, seed=child_seed)
        return audio, sr

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        inner = "\n  ".join(repr(t) for t in self.transforms)
        return f"Pipeline(p={self.p},\n  {inner}\n)"

    def __len__(self) -> int:
        return len(self.transforms)

    def append(self, transform: Transform) -> Pipeline:
        """Return a new Pipeline with *transform* appended."""
        return Pipeline(self.transforms + [transform], seed=self.seed, p=self.p)


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------


class OneOf(Transform):
    """Apply exactly one transform chosen at random from a list.

    Parameters
    ----------
    transforms:
        List of transforms to choose from.
    weights:
        Optional relative weights for each transform.  If ``None`` all
        transforms are equally probable.
    p:
        Probability that *any* transform is applied.
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
        weights: Sequence[float] | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if not transforms:
            raise ValueError("transforms must not be empty")
        self.transforms = list(transforms)
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            self.weights: np.ndarray | None = w / w.sum()
        else:
            self.weights = None

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        idx = int(rng.choice(len(self.transforms), p=self.weights))
        child_seed = int(rng.integers(0, 2**31))
        return self.transforms[idx](audio, sr, seed=child_seed)


class SomeOf(Transform):
    """Apply *k* randomly chosen transforms (without replacement).

    Parameters
    ----------
    transforms:
        Pool of transforms.
    k:
        Number of transforms to apply.  Defaults to selecting all of them
        in a shuffled order (equivalent to a shuffled :class:`Pipeline`).
    p:
        Probability that any transforms are applied.
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
        k: int | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.transforms = list(transforms)
        self.k = k

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        k = self.k if self.k is not None else len(self.transforms)
        k = min(k, len(self.transforms))
        idxs = rng.choice(len(self.transforms), size=k, replace=False)
        for idx in idxs:
            child_seed = int(rng.integers(0, 2**31))
            audio, sr = self.transforms[int(idx)](audio, sr, seed=child_seed)
        return audio, sr


class RandomApply(Transform):
    """Apply a single transform with probability *p*.

    Equivalent to wrapping a transform with ``p < 1.0``, but useful for
    wrapping a :class:`Pipeline` or :class:`OneOf` that has its own ``p=1.0``.
    """

    def __init__(self, transform: Transform | Pipeline, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.transform = transform

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        child_seed = int(rng.integers(0, 2**31))
        return self.transform(audio, sr, seed=child_seed)


# ---------------------------------------------------------------------------
# Identity / Passthrough
# ---------------------------------------------------------------------------


class Identity(Transform):
    """Pass audio through unchanged."""

    def apply(
        self,
        audio: np.ndarray,
        sr: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, int]:
        return audio, sr


# ---------------------------------------------------------------------------
# TransformResult helper
# ---------------------------------------------------------------------------


class TransformResult:
    """Container returned when ``return_metadata=True`` is requested."""

    __slots__ = ("audio", "elapsed_s", "params", "sr", "transform_name")

    def __init__(
        self,
        audio: np.ndarray,
        sr: int,
        params: dict[str, Any],
        transform_name: str,
        elapsed_s: float = 0.0,
    ) -> None:
        self.audio = audio
        self.sr = sr
        self.params = params
        self.transform_name = transform_name
        self.elapsed_s = elapsed_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform": self.transform_name,
            "sr": self.sr,
            "params": self.params,
            "elapsed_s": round(self.elapsed_s, 4),
        }


# ---------------------------------------------------------------------------
# Audio loading helper used by all submodules
# ---------------------------------------------------------------------------


def load_audio(
    path: str | Path,
    target_sr: int | None = None,
    mono: bool = False,
    dtype: str = "float32",
) -> tuple[np.ndarray, int]:
    """Load audio from a file, optionally resampling and converting to mono.

    Parameters
    ----------
    path:
        Path to audio file.
    target_sr:
        If provided, resample to this rate using ``scipy.signal.resample_poly``.
    mono:
        If ``True`` mix down to mono.
    dtype:
        NumPy dtype for the output array.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(audio, sample_rate)`` where audio shape is ``(channels, samples)``
        (or ``(samples,)`` if *mono* is ``True``).
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype=dtype, always_2d=True)
    # soundfile returns (samples, channels); transpose to (channels, samples).
    audio = audio.T.copy()

    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)

    if target_sr is not None and target_sr != sr:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(target_sr, sr)
        up, down = target_sr // g, sr // g
        resampled = np.stack(
            [resample_poly(ch, up, down).astype(dtype) for ch in audio],
            axis=0,
        )
        audio = resampled
        sr = target_sr

    return audio, sr


def save_audio(
    path: str | Path,
    audio: np.ndarray,
    sr: int,
    fmt: str = "WAV",
    subtype: str = "PCM_16",
) -> None:
    """Write audio to a file.

    Parameters
    ----------
    path:
        Output file path.
    audio:
        Array with shape ``(channels, samples)`` or ``(samples,)``.
    sr:
        Sample rate.
    fmt:
        Soundfile format string (e.g. ``"WAV"``, ``"FLAC"``).
    subtype:
        Soundfile subtype (e.g. ``"PCM_24"``, ``"FLOAT"``).
    """
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if audio.ndim == 2:
        data = audio.T  # (samples, channels)
    else:
        data = audio
    sf.write(str(path), data, sr, format=fmt, subtype=subtype)


def fingerprint_audio(audio: np.ndarray, sr: int) -> str:
    """Return a short hex digest of audio content for manifest tracking."""
    h = hashlib.sha256()
    h.update(sr.to_bytes(4, "little"))
    h.update(audio.tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Plugin registry for custom transforms
# ---------------------------------------------------------------------------

_TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}


def register_transform(
    name: str | None = None,
) -> type[Transform]:
    """Register a custom Transform class in the global registry.

    Can be used as a decorator or called directly.

    Examples
    --------
    As a decorator::

        @register_transform("my_reverb")
        class MyReverb(Transform):
            def apply(self, audio, sr, rng):
                ...

    Direct call::

        register_transform("my_reverb")(MyReverb)

    Parameters
    ----------
    name:
        Registry key.  Defaults to the class name in snake_case.

    Returns
    -------
    The original class, unmodified.
    """

    def decorator(cls: type[Transform]) -> type[Transform]:
        key = name if name is not None else _to_snake_case(cls.__name__)
        if not issubclass(cls, Transform):
            raise TypeError(f"register_transform expects a Transform subclass, got {cls!r}")
        _TRANSFORM_REGISTRY[key] = cls
        return cls

    return decorator


def get_transform(name: str) -> type[Transform]:
    """Look up a registered transform by name.

    Parameters
    ----------
    name:
        Registry key (as passed to :func:`register_transform`).

    Returns
    -------
    type[Transform]
        The registered Transform class.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in _TRANSFORM_REGISTRY:
        available = ", ".join(sorted(_TRANSFORM_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown transform: {name!r}. Registered: {available}")
    return _TRANSFORM_REGISTRY[name]


def list_transforms() -> dict[str, type[Transform]]:
    """Return a copy of the current transform registry."""
    return dict(_TRANSFORM_REGISTRY)


def _load_entry_point_plugins() -> None:
    """Discover and load transforms registered via setuptools entry points.

    Third-party packages can register transforms by adding an entry point
    in the ``pvx.augment.transforms`` group::

        [project.entry-points."pvx.augment.transforms"]
        my_reverb = "my_package.transforms:MyReverb"

    Each entry point's value should be a :class:`Transform` subclass.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return  # Python < 3.10 fallback not needed (requires-python >=3.10)

    eps = entry_points()
    # Python 3.12+ returns SelectableGroups; 3.10-3.11 returns dict
    if isinstance(eps, dict):
        group = eps.get("pvx.augment.transforms", [])
    else:
        group = eps.select(group="pvx.augment.transforms")

    for ep in group:
        try:
            cls = ep.load()
            if isinstance(cls, type) and issubclass(cls, Transform):
                _TRANSFORM_REGISTRY.setdefault(ep.name, cls)
        except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
            pass  # Silently skip broken plugins


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re

    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


# Auto-discover entry-point plugins at import time
_load_entry_point_plugins()
