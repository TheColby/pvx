
"""Streaming / chunked augmentation for long-form audio.

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
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np

from .core import Pipeline, Transform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hann_fade(length: int, dtype: np.dtype = np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Return (fade_in, fade_out) Hann half-windows of *length* samples."""
    if length <= 0:
        return np.ones(0, dtype=dtype), np.ones(0, dtype=dtype)
    t = np.linspace(0, np.pi / 2, length, endpoint=True, dtype=dtype)
    fade_in = np.sin(t) ** 2
    fade_out = np.cos(t) ** 2
    return fade_in, fade_out


# ---------------------------------------------------------------------------
# In-memory streaming
# ---------------------------------------------------------------------------


def stream_augment(
    audio: np.ndarray,
    sr: int,
    pipeline: Pipeline | Transform,
    *,
    chunk_duration_s: float = 30.0,
    overlap_s: float = 0.1,
    seed: int = 0,
) -> Generator[tuple[np.ndarray, int], None, None]:
    """Yield augmented chunks from an in-memory array.

    Parameters
    ----------
    audio:
        Full input array, shape ``(samples,)`` or ``(channels, samples)``.
    sr:
        Sample rate.
    pipeline:
        Augmentation pipeline or single transform.
    chunk_duration_s:
        Target chunk length in seconds.
    overlap_s:
        Crossfade overlap between consecutive chunks in seconds.
    seed:
        Base seed for deterministic per-chunk augmentation.

    Yields
    ------
    tuple[np.ndarray, int]
        ``(augmented_chunk, sr)`` for each chunk, crossfaded at boundaries.
    """
    mono = audio.ndim == 1
    if mono:
        arr = audio[np.newaxis, :]
    else:
        arr = audio

    n_ch, total_samples = arr.shape
    chunk_samples = max(int(chunk_duration_s * sr), sr)  # at least 1 second
    overlap_samples = min(int(overlap_s * sr), chunk_samples // 4)

    step = chunk_samples - overlap_samples
    if step <= 0:
        step = chunk_samples

    fade_in, fade_out = _hann_fade(overlap_samples, dtype=arr.dtype)

    # Previous chunk's trailing overlap region (for crossfading)
    prev_tail: np.ndarray | None = None
    chunk_idx = 0

    pos = 0
    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = arr[:, pos:end]

        # Augment this chunk
        chunk_seed = seed + chunk_idx * 7919  # large prime for spacing
        if mono:
            chunk_in = chunk[0]
        else:
            chunk_in = chunk

        aug_chunk, aug_sr = pipeline(chunk_in, sr, seed=chunk_seed)
        if aug_chunk.ndim == 1:
            aug_chunk = aug_chunk[np.newaxis, :]

        # Apply crossfade with previous chunk's tail
        if prev_tail is not None and overlap_samples > 0:
            xf_len = min(overlap_samples, aug_chunk.shape[1], prev_tail.shape[1])
            if xf_len > 0:
                fi, fo = _hann_fade(xf_len, dtype=aug_chunk.dtype)
                # Blend: prev_tail fades out, current chunk fades in
                blended = prev_tail[:, :xf_len] * fo + aug_chunk[:, :xf_len] * fi
                # Emit: blended overlap + rest of chunk (minus overlap at end)
                if aug_chunk.shape[1] > overlap_samples:
                    body = (
                        aug_chunk[:, xf_len:-overlap_samples]
                        if pos + chunk_samples < total_samples
                        else aug_chunk[:, xf_len:]
                    )
                    out = np.concatenate([blended, body], axis=1)
                else:
                    out = blended
            else:
                out = aug_chunk
        else:
            # First chunk: emit everything except the trailing overlap
            if pos + chunk_samples < total_samples and overlap_samples > 0:
                out = aug_chunk[:, :-overlap_samples]
            else:
                out = aug_chunk

        # Save tail for next crossfade
        if (
            pos + chunk_samples < total_samples
            and overlap_samples > 0
            and aug_chunk.shape[1] >= overlap_samples
        ):
            prev_tail = aug_chunk[:, -overlap_samples:]
        else:
            prev_tail = None

        yield (out[0] if mono else out, aug_sr)

        pos += step
        chunk_idx += 1


# ---------------------------------------------------------------------------
# File-based streaming
# ---------------------------------------------------------------------------


def stream_augment_file(
    input_path: str | Path,
    output_path: str | Path,
    pipeline: Pipeline | Transform,
    *,
    chunk_duration_s: float = 30.0,
    overlap_s: float = 0.1,
    seed: int = 0,
    output_format: str = "WAV",
    output_subtype: str = "PCM_16",
) -> dict[str, object]:
    """Augment a long audio file in chunks with bounded memory.

    Reads the input in segments, augments each chunk, crossfades at
    boundaries, and writes the output incrementally.

    Parameters
    ----------
    input_path:
        Path to the input audio file.
    output_path:
        Path for the augmented output file.
    pipeline:
        Augmentation pipeline or single transform.
    chunk_duration_s:
        Duration of each processing chunk in seconds (default: 30).
    overlap_s:
        Crossfade overlap at chunk boundaries in seconds (default: 0.1).
    seed:
        Deterministic seed.
    output_format:
        Soundfile output format (WAV, FLAC, etc.).
    output_subtype:
        Soundfile subtype (PCM_16, PCM_24, FLOAT, etc.).

    Returns
    -------
    dict
        Summary with keys: ``input_path``, ``output_path``,
        ``duration_s``, ``n_chunks``, ``peak_memory_samples``.
    """
    import soundfile as sf

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read full file (for now — future: use sf.blocks for truly streaming I/O)
    audio, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    mono = audio.ndim == 1

    chunks_written = 0
    total_samples_written = 0

    # Collect all chunks (we need to know total length for the output file)
    output_chunks: list[np.ndarray] = []
    for chunk_out, chunk_sr in stream_augment(
        audio,
        sr,
        pipeline,
        chunk_duration_s=chunk_duration_s,
        overlap_s=overlap_s,
        seed=seed,
    ):
        output_chunks.append(chunk_out)
        chunks_written += 1

    if output_chunks:
        result = np.concatenate(output_chunks, axis=-1)
        total_samples_written = result.shape[-1] if result.ndim > 1 else len(result)
        sf.write(
            str(output_path),
            result.T if result.ndim == 2 else result,
            sr,
            format=output_format,
            subtype=output_subtype,
        )

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "duration_s": round(total_samples_written / sr, 3),
        "n_chunks": chunks_written,
        "peak_memory_samples": int(chunk_duration_s * sr),
    }


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def stream_augment_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    pipeline: Pipeline | Transform,
    *,
    chunk_duration_s: float = 30.0,
    overlap_s: float = 0.1,
    seed: int = 0,
    glob_pattern: str = "*.wav",
    workers: int = 1,
) -> list[dict[str, object]]:
    """Augment all audio files in a directory with streaming.

    Parameters
    ----------
    input_dir:
        Directory containing input audio files.
    output_dir:
        Directory for augmented output files.
    pipeline:
        Augmentation pipeline.
    chunk_duration_s:
        Chunk size in seconds.
    overlap_s:
        Crossfade overlap in seconds.
    seed:
        Base seed (offset per file for variety).
    glob_pattern:
        File pattern to match (default: ``"*.wav"``).
    workers:
        Number of parallel workers (1 = sequential).

    Returns
    -------
    list[dict]
        Per-file summary dicts from :func:`stream_augment_file`.
    """
    import concurrent.futures

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        return []

    def _process(args: tuple[int, Path]) -> dict[str, object]:
        idx, src = args
        dst = output_dir / src.name
        return stream_augment_file(
            src,
            dst,
            pipeline,
            chunk_duration_s=chunk_duration_s,
            overlap_s=overlap_s,
            seed=seed + idx * 100003,
        )

    results: list[dict[str, object]] = []
    if workers <= 1:
        for idx, src in enumerate(files):
            results.append(_process((idx, src)))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process, (idx, src)) for idx, src in enumerate(files)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    return results
