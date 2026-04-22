"""PyTorch integration for pvx augmentation pipelines.

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
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Guard: do not import torch at module level so that pvx works even
# without PyTorch installed.


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for pvx.integrations.pytorch. Install it with: pip install torch"
        ) from exc


# ---------------------------------------------------------------------------
# PvxAugmentDataset
# ---------------------------------------------------------------------------


class PvxAugmentDataset:
    """A ``torch.utils.data.Dataset`` backed by pvx augmentation pipelines.

    Supports variable-length audio (with optional fixed-length cropping),
    per-item seeding for deterministic validation sets, and optional label
    passthrough.

    Parameters
    ----------
    files:
        List of audio file paths.
    pipeline:
        A ``pvx.augment.Pipeline`` (or any callable ``(audio, sr, seed) ->
        (audio, sr)``).  Pass ``None`` to disable augmentation.
    sample_rate:
        Target sample rate.  Audio is resampled if needed.
    labels:
        Optional list of integer labels (one per file).
    duration_s:
        If set, crop/pad each clip to this length in seconds.
    mono:
        If ``True`` mix down to mono before augmentation.
    augment_on_val:
        If ``False`` and ``is_val=True`` is passed to :meth:`__getitem__`,
        augmentation is skipped (useful for deterministic validation).
    seed_offset:
        Added to item index to form the per-item seed.
    return_path:
        If ``True`` include the file path in the returned dict.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        pipeline=None,
        sample_rate: int = 16000,
        labels: Sequence[int] | None = None,
        duration_s: float | None = None,
        mono: bool = True,
        augment_on_val: bool = False,
        seed_offset: int = 0,
        return_path: bool = False,
    ) -> None:
        torch = _require_torch()

        self.files = [Path(f) for f in files]
        self.pipeline = pipeline
        self.sample_rate = int(sample_rate)
        self.labels = list(labels) if labels is not None else None
        self.duration_s = duration_s
        self.mono = mono
        self.augment_on_val = augment_on_val
        self.seed_offset = int(seed_offset)
        self.return_path = return_path
        self._torch = torch

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int, *, is_val: bool = False) -> dict[str, Any]:
        from pvx.augment.core import load_audio

        path = self.files[idx]
        audio, sr = load_audio(path, target_sr=self.sample_rate, mono=self.mono)

        # Fixed-length crop / pad
        if self.duration_s is not None:
            target = int(self.duration_s * sr)
            n_samp = audio.shape[-1]
            if n_samp >= target:
                audio = audio[..., :target]
            else:
                pad = target - n_samp
                if audio.ndim == 1:
                    audio = np.pad(audio, (0, pad))
                else:
                    audio = np.pad(audio, ((0, 0), (0, pad)))

        # Augment
        if self.pipeline is not None and (not is_val or self.augment_on_val):
            seed = idx + self.seed_offset
            audio, sr = self.pipeline(audio, sr, seed=seed)

        # Ensure float32 tensor with shape (channels, samples) or (samples,)
        tensor = self._torch.from_numpy(np.asarray(audio, dtype=np.float32))

        item: dict[str, Any] = {"audio": tensor, "sr": sr}
        if self.labels is not None:
            item["label"] = self._torch.tensor(self.labels[idx], dtype=self._torch.long)
        if self.return_path:
            item["path"] = str(path)
        return item

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate variable-length audio into a padded batch tensor.

        Pads all clips to the length of the longest clip in the batch.
        Returns a dict with keys: ``audio`` (B, C, T) or (B, T),
        ``lengths``, and optionally ``label``, ``sr``, ``path``.
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        max_len = max(item["audio"].shape[-1] for item in batch)
        padded = []
        lengths = []
        for item in batch:
            a = item["audio"]
            lengths.append(a.shape[-1])
            pad_amt = max_len - a.shape[-1]
            if a.ndim == 1:
                padded.append(F.pad(a, (0, pad_amt)))
            else:
                padded.append(F.pad(a, (0, pad_amt)))

        result: dict[str, Any] = {
            "audio": torch.stack(padded, dim=0),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "sr": batch[0]["sr"],
        }
        if "label" in batch[0]:
            result["label"] = torch.stack([item["label"] for item in batch])
        if "path" in batch[0]:
            result["path"] = [item["path"] for item in batch]
        return result

    # ------------------------------------------------------------------
    def as_torch_dataset(self):
        """Return ``self`` wrapped as a ``torch.utils.data.Dataset``."""
        _require_torch()
        from torch.utils.data import Dataset

        class _TorchWrapper(Dataset):
            def __init__(inner, parent):
                inner._parent = parent

            def __len__(inner):
                return len(inner._parent)

            def __getitem__(inner, idx):
                return inner._parent[idx]

        return _TorchWrapper(self)


# ---------------------------------------------------------------------------
# PvxAugmentTransform — torchvision-style transform callable
# ---------------------------------------------------------------------------


class PvxAugmentTransform:
    """A torchvision-compatible callable transform wrapping a pvx pipeline.

    Accepts a ``torch.Tensor`` of shape ``(C, T)`` or ``(T,)`` and returns
    an augmented tensor of the same shape.

    Parameters
    ----------
    pipeline:
        pvx augmentation pipeline or any ``(audio, sr, seed) -> (audio, sr)``
        callable.
    sample_rate:
        Sample rate assumed for the input tensor.
    seed:
        Optional fixed seed (for deterministic test-time evaluation).

    Examples
    --------
    >>> transform = PvxAugmentTransform(pipeline, sample_rate=16000)
    >>> # Use with torchaudio datasets
    >>> ds = torchaudio.datasets.LIBRISPEECH("data/", transform=transform)
    """

    def __init__(self, pipeline, sample_rate: int = 16000, seed: int | None = None) -> None:
        self.pipeline = pipeline
        self.sample_rate = int(sample_rate)
        self.seed = seed
        self._counter = 0

    def __call__(self, tensor) -> Any:
        import torch

        numpy_audio = tensor.numpy() if hasattr(tensor, "numpy") else np.asarray(tensor)
        seed = self.seed if self.seed is not None else (self._counter % (2**31))
        self._counter += 1
        audio_aug, _ = self.pipeline(numpy_audio, self.sample_rate, seed=seed)
        return torch.from_numpy(np.asarray(audio_aug, dtype=np.float32))


# ---------------------------------------------------------------------------
# AudioCollator — for DataLoader with variable-length batches
# ---------------------------------------------------------------------------


class AudioCollator:
    """DataLoader collate function for variable-length audio tensors.

    Parameters
    ----------
    pad_value:
        Value used for padding.
    return_lengths:
        Whether to include the true length of each clip.

    Examples
    --------
    >>> loader = DataLoader(
    ...     dataset,
    ...     collate_fn=AudioCollator(return_lengths=True),
    ...     batch_size=16,
    ... )
    """

    def __init__(self, pad_value: float = 0.0, return_lengths: bool = True) -> None:
        self.pad_value = float(pad_value)
        self.return_lengths = return_lengths

    def __call__(self, batch: list[Any]) -> Any:
        import torch
        import torch.nn.functional as F  # noqa: N812

        if isinstance(batch[0], dict):
            return PvxAugmentDataset.collate_fn(batch)

        # List of tensors
        tensors = batch
        max_len = max(t.shape[-1] for t in tensors)
        padded = []
        lengths = []
        for t in tensors:
            lengths.append(t.shape[-1])
            pad = max_len - t.shape[-1]
            padded.append(F.pad(t, (0, pad), value=self.pad_value))
        out = torch.stack(padded, 0)
        if self.return_lengths:
            return out, torch.tensor(lengths, dtype=torch.long)
        return out
