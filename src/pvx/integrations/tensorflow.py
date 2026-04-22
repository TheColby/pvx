
"""TensorFlow / tf.data integration for pvx augmentation pipelines.

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
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _require_tf():
    try:
        import tensorflow as tf

        return tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for pvx.integrations.tensorflow. "
            "Install it with: pip install tensorflow"
        ) from exc


# ---------------------------------------------------------------------------
# make_tf_augment_fn
# ---------------------------------------------------------------------------


def make_tf_augment_fn(
    pipeline,
    sample_rate: int = 16000,
    seed: int | None = None,
) -> Any:
    """Return a ``tf.data``-compatible map function using ``tf.py_function``.

    The returned function wraps the pvx pipeline in a ``tf.py_function``
    call so it can be used inside a ``tf.data.Dataset.map()`` call.  Note
    that ``tf.py_function`` runs in eager mode and will not be traced by
    TF's graph compiler — this is expected for data preprocessing.

    Parameters
    ----------
    pipeline:
        pvx augmentation pipeline.
    sample_rate:
        Sample rate assumed for audio tensors.
    seed:
        Optional fixed seed.  If ``None`` a counter is used.

    Returns
    -------
    callable
        A function ``(audio_tensor) -> audio_tensor`` compatible with
        ``tf.data.Dataset.map()``.

    Examples
    --------
    >>> fn = make_tf_augment_fn(pipeline, sample_rate=16000)
    >>> ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)
    """
    tf = _require_tf()
    _counter = [0]

    def _augment_py(audio_np: np.ndarray) -> np.ndarray:
        s = seed if seed is not None else (_counter[0] % (2**31))
        _counter[0] += 1
        audio_aug, _ = pipeline(audio_np.astype(np.float32), sample_rate, seed=s)
        return np.asarray(audio_aug, dtype=np.float32)

    def augment_fn(audio_tensor):
        result = tf.py_function(
            func=_augment_py,
            inp=[audio_tensor],
            Tout=tf.float32,
        )
        result.set_shape(audio_tensor.shape)
        return result

    return augment_fn


# ---------------------------------------------------------------------------
# make_tf_augment_map_fn — for dict-based datasets
# ---------------------------------------------------------------------------


def make_tf_augment_map_fn(
    pipeline,
    audio_key: str = "audio",
    sr_key: str | None = None,
    output_key: str | None = None,
    default_sr: int = 16000,
    seed: int | None = None,
) -> Any:
    """Return a map function for dict-structured ``tf.data`` datasets.

    Parameters
    ----------
    pipeline:
        pvx augmentation pipeline.
    audio_key:
        Key in the dataset dict containing the audio tensor.
    sr_key:
        Optional key for sample rate.
    output_key:
        Key to write augmented audio to (defaults to *audio_key*).
    default_sr:
        Fallback sample rate.
    seed:
        Optional fixed seed.

    Returns
    -------
    callable
        A function ``(features_dict) -> features_dict``.

    Examples
    --------
    >>> fn = make_tf_augment_map_fn(pipeline, audio_key="audio")
    >>> ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)
    """
    tf = _require_tf()
    out_key = output_key or audio_key
    _counter = [0]

    def _augment_py(audio_np: np.ndarray, sr_np: np.ndarray) -> np.ndarray:
        sr = int(sr_np) if sr_np.ndim == 0 else default_sr
        s = seed if seed is not None else (_counter[0] % (2**31))
        _counter[0] += 1
        audio_aug, _ = pipeline(audio_np.astype(np.float32), sr, seed=s)
        return np.asarray(audio_aug, dtype=np.float32)

    def augment_fn(features: dict) -> dict:
        audio = features[audio_key]
        if sr_key and sr_key in features:
            sr_tensor = tf.cast(features[sr_key], tf.int32)
            sr_np_fn = lambda: tf.py_function(
                func=_augment_py,
                inp=[audio, sr_tensor],
                Tout=tf.float32,
            )
            result = sr_np_fn()
        else:
            sr_val = tf.constant(default_sr, dtype=tf.int32)
            result = tf.py_function(
                func=_augment_py,
                inp=[audio, sr_val],
                Tout=tf.float32,
            )
        result.set_shape(audio.shape)
        features = dict(features)
        features[out_key] = result
        return features

    return augment_fn


# ---------------------------------------------------------------------------
# pvx_augment_tf_dataset — convenience wrapper
# ---------------------------------------------------------------------------


def pvx_augment_tf_dataset(
    dataset,
    pipeline,
    audio_key: str = "audio",
    sample_rate: int = 16000,
    seed: int | None = None,
    num_parallel_calls: int | None = None,
):
    """Apply a pvx pipeline to a ``tf.data.Dataset``.

    Parameters
    ----------
    dataset:
        A ``tf.data.Dataset`` where each element is either a tensor or a
        dict of tensors.
    pipeline:
        pvx augmentation pipeline.
    audio_key:
        If the dataset yields dicts, this is the audio key.
    sample_rate:
        Sample rate assumed for audio tensors.
    seed:
        Optional fixed seed.
    num_parallel_calls:
        Forwarded to ``Dataset.map()``.  Defaults to ``tf.data.AUTOTUNE``.

    Returns
    -------
    tf.data.Dataset
        Dataset with augmentation applied.

    Examples
    --------
    >>> ds_aug = pvx_augment_tf_dataset(ds, pipeline, sample_rate=16000)
    >>> for batch in ds_aug.batch(32):
    ...     train_step(batch)
    """
    tf = _require_tf()
    parallelism = num_parallel_calls if num_parallel_calls is not None else tf.data.AUTOTUNE

    # Peek at first element to determine if it's a dict or tensor
    for sample in dataset.take(1):
        is_dict = isinstance(sample, dict)
        break
    else:
        is_dict = False

    if is_dict:
        fn = make_tf_augment_map_fn(
            pipeline, audio_key=audio_key, default_sr=sample_rate, seed=seed
        )
    else:
        fn = make_tf_augment_fn(pipeline, sample_rate=sample_rate, seed=seed)

    return dataset.map(fn, num_parallel_calls=parallelism)
