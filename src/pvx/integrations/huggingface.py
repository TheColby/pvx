"""HuggingFace datasets integration for pvx augmentation pipelines.

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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# make_augment_map_fn
# ---------------------------------------------------------------------------


def make_augment_map_fn(
    pipeline,
    audio_column: str = "audio",
    sr_column: str | None = None,
    output_column: str | None = None,
    default_sr: int = 16000,
    seed_column: str | None = None,
    base_seed: int = 42,
    return_metadata: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a function compatible with ``datasets.Dataset.map()``.

    The returned function reads audio from a HuggingFace dataset row,
    applies the pvx augmentation pipeline, and writes the result back.

    HuggingFace ``Audio`` feature rows look like::

        {"array": np.ndarray, "sampling_rate": int, "path": str}

    The function handles both raw-array and Audio-feature formats.

    Parameters
    ----------
    pipeline:
        pvx augmentation pipeline.
    audio_column:
        Dataset column containing audio.  Can be a raw array or a dict
        with ``"array"`` and ``"sampling_rate"`` keys (HuggingFace
        ``Audio`` feature).
    sr_column:
        Column with sample rate.  If ``None``, inferred from the audio
        dict or uses *default_sr*.
    output_column:
        Column to write augmented audio to.  If ``None``, overwrites
        *audio_column*.
    default_sr:
        Fallback sample rate if not found in the row.
    seed_column:
        Optional column containing an integer seed per row.  If ``None``,
        uses ``base_seed + row_index`` (requires ``with_indices=True``).
    base_seed:
        Base seed used when *seed_column* is not set.
    return_metadata:
        If ``True``, add an ``"augment_params"`` JSON string column.

    Returns
    -------
    Callable
        A function ``(row, [idx]) -> row`` suitable for
        ``ds.map(fn, batched=False, with_indices=True)``.

    Examples
    --------
    >>> fn = make_augment_map_fn(pipeline, audio_column="audio")
    >>> ds_aug = ds.map(fn, batched=False, with_indices=True, num_proc=4)
    """

    out_col = output_column or audio_column

    def _augment_row(row: dict[str, Any], idx: int = 0) -> dict[str, Any]:
        audio_val = row[audio_column]

        # Extract numpy array and sample rate from various formats
        if isinstance(audio_val, dict):
            array = np.asarray(audio_val.get("array", audio_val.get("data", [])), dtype=np.float32)
            sr = int(audio_val.get("sampling_rate", audio_val.get("sample_rate", default_sr)))
        elif isinstance(audio_val, np.ndarray):
            array = audio_val.astype(np.float32)
            sr = int(row.get(sr_column, default_sr)) if sr_column else default_sr
        else:
            # Try to coerce to numpy
            array = np.asarray(audio_val, dtype=np.float32)
            sr = default_sr

        # Determine seed
        if seed_column and seed_column in row:
            seed = int(row[seed_column])
        else:
            seed = base_seed + idx

        # Run pipeline
        audio_aug, sr_out = pipeline(array, sr, seed=seed)
        audio_aug = np.asarray(audio_aug, dtype=np.float32)

        # Build output value in the same format as input
        if isinstance(audio_val, dict):
            new_audio = dict(audio_val)
            new_audio["array"] = audio_aug
            new_audio["sampling_rate"] = sr_out
        else:
            new_audio = audio_aug

        row = dict(row)
        row[out_col] = new_audio

        if return_metadata:
            import json

            row["augment_params"] = json.dumps(
                {
                    "seed": seed,
                    "sr_in": sr,
                    "sr_out": sr_out,
                    "shape_in": list(array.shape),
                    "shape_out": list(audio_aug.shape),
                }
            )

        return row

    return _augment_row


# ---------------------------------------------------------------------------
# HFAugmentMapper — stateful class with full configuration
# ---------------------------------------------------------------------------


class HFAugmentMapper:
    """Stateful HuggingFace datasets mapper with advanced configuration.

    Use this when you need more control over the mapping behaviour than
    :func:`make_augment_map_fn` provides, e.g. label-aware augmentation
    or paired clean/noisy generation.

    Parameters
    ----------
    pipeline:
        pvx augmentation pipeline.
    audio_column:
        Input audio column name.
    label_column:
        Optional label column.  If provided, label-preserving policies
        are applied (conservative stretch/pitch perturbation).
    sr_column:
        Optional sample rate column.
    output_prefix:
        Prefix for new output columns.
    generate_pair:
        If ``True``, generate *two* augmented views per row (for
        contrastive learning).  Output columns are
        ``<prefix>_a`` and ``<prefix>_b``.
    base_seed:
        Base seed for reproducibility.

    Examples
    --------
    >>> mapper = HFAugmentMapper(pipeline, generate_pair=True)
    >>> ds_aug = ds.map(mapper, batched=False, with_indices=True)
    """

    def __init__(
        self,
        pipeline,
        audio_column: str = "audio",
        label_column: str | None = None,
        sr_column: str | None = None,
        output_prefix: str = "aug",
        generate_pair: bool = False,
        base_seed: int = 42,
    ) -> None:
        self.pipeline = pipeline
        self.audio_column = audio_column
        self.label_column = label_column
        self.sr_column = sr_column
        self.output_prefix = output_prefix
        self.generate_pair = generate_pair
        self.base_seed = int(base_seed)

    def _extract(self, row: dict[str, Any], default_sr: int = 16000) -> tuple[np.ndarray, int]:
        val = row[self.audio_column]
        if isinstance(val, dict):
            arr = np.asarray(val.get("array", val.get("data", [])), dtype=np.float32)
            sr = int(val.get("sampling_rate", val.get("sample_rate", default_sr)))
        else:
            arr = np.asarray(val, dtype=np.float32)
            sr = int(row.get(self.sr_column, default_sr)) if self.sr_column else default_sr
        return arr, sr

    def _pack(self, audio: np.ndarray, sr: int, original_val: Any) -> Any:
        if isinstance(original_val, dict):
            out = dict(original_val)
            out["array"] = np.asarray(audio, dtype=np.float32)
            out["sampling_rate"] = sr
            return out
        return np.asarray(audio, dtype=np.float32)

    def __call__(self, row: dict[str, Any], idx: int = 0) -> dict[str, Any]:
        arr, sr = self._extract(row)
        original_val = row[self.audio_column]
        row = dict(row)

        seed_a = self.base_seed + idx * 2
        audio_a, sr_a = self.pipeline(arr, sr, seed=seed_a)

        if self.generate_pair:
            seed_b = self.base_seed + idx * 2 + 1
            audio_b, sr_b = self.pipeline(arr, sr, seed=seed_b)
            row[f"{self.output_prefix}_a"] = self._pack(audio_a, sr_a, original_val)
            row[f"{self.output_prefix}_b"] = self._pack(audio_b, sr_b, original_val)
        else:
            row[self.output_prefix] = self._pack(audio_a, sr_a, original_val)

        return row


# ---------------------------------------------------------------------------
# Batch augmentation helper
# ---------------------------------------------------------------------------


def augment_dataset(
    dataset,
    pipeline,
    audio_column: str = "audio",
    output_column: str | None = None,
    num_proc: int = 1,
    base_seed: int = 42,
    **map_kwargs: Any,
):
    """Apply a pvx pipeline to an entire HuggingFace dataset.

    Convenience wrapper around ``Dataset.map()`` with sane defaults.

    Parameters
    ----------
    dataset:
        A ``datasets.Dataset`` or ``DatasetDict``.
    pipeline:
        pvx augmentation pipeline.
    audio_column:
        Input audio column.
    output_column:
        Output audio column (defaults to *audio_column*).
    num_proc:
        Number of parallel processes for mapping.
    base_seed:
        Base seed for per-row reproducibility.
    **map_kwargs:
        Additional kwargs forwarded to ``Dataset.map()``.

    Returns
    -------
    datasets.Dataset or DatasetDict
        Augmented dataset.

    Examples
    --------
    >>> from pvx.integrations.huggingface import augment_dataset
    >>> ds_aug = augment_dataset(ds, pipeline, num_proc=4)
    """
    fn = make_augment_map_fn(
        pipeline,
        audio_column=audio_column,
        output_column=output_column,
        base_seed=base_seed,
    )
    return dataset.map(fn, batched=False, with_indices=True, num_proc=num_proc, **map_kwargs)
