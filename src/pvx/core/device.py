# Copyright (c) 2026 Colby Leider and contributors. See ATTRIBUTION.md.
"""GPU/CUDA device abstraction and runtime configuration helpers."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Literal

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency guard
    np = None

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

DeviceMode = Literal["auto", "cpu", "cuda"]

@dataclass(frozen=True)
class RuntimeConfig:
    requested_device: DeviceMode
    active_device: Literal["cpu", "cuda"]
    cuda_device: int
    fallback_reason: str | None = None


_RUNTIME_CONFIG = RuntimeConfig(
    requested_device="auto",
    active_device="cpu",
    cuda_device=0,
    fallback_reason=None,
)


def _has_cupy() -> bool:
    return cp is not None


def _is_cupy_array(value: Any) -> bool:
    return _has_cupy() and isinstance(value, cp.ndarray)


def _array_module(value: Any):
    if _is_cupy_array(value):
        return cp
    return np


def _to_numpy(value: Any):
    if _is_cupy_array(value):
        return cp.asnumpy(value)
    return value


def _to_runtime_array(value: Any):
    if _RUNTIME_CONFIG.active_device != "cuda":
        return value
    if _is_cupy_array(value):
        return value
    return cp.asarray(value)


def runtime_config() -> RuntimeConfig:
    return _RUNTIME_CONFIG


def configure_runtime(
    device: DeviceMode = "auto",
    cuda_device: int = 0,
    *,
    verbose: bool = False,
) -> RuntimeConfig:
    global _RUNTIME_CONFIG

    if cuda_device < 0:
        raise ValueError("--cuda-device must be >= 0")

    requested = device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device mode: {device}")

    if requested == "cpu":
        _RUNTIME_CONFIG = RuntimeConfig(
            requested_device="cpu",
            active_device="cpu",
            cuda_device=cuda_device,
            fallback_reason=None,
        )
        return _RUNTIME_CONFIG

    if not _has_cupy():
        reason = "CuPy is not installed"
        if requested == "cuda":
            raise RuntimeError("CUDA mode requires CuPy. Install a matching `cupy-cudaXXx` package.")
        _RUNTIME_CONFIG = RuntimeConfig(
            requested_device="auto",
            active_device="cpu",
            cuda_device=cuda_device,
            fallback_reason=reason,
        )
        if verbose:
            print(f"[info] {reason}; using CPU backend", file=sys.stderr)
        return _RUNTIME_CONFIG

    try:
        cp.cuda.Device(cuda_device).use()
        _ = cp.cuda.runtime.getDevice()
    except Exception as exc:
        reason = f"CUDA device init failed: {exc}"
        if requested == "cuda":
            raise RuntimeError(reason) from exc
        _RUNTIME_CONFIG = RuntimeConfig(
            requested_device="auto",
            active_device="cpu",
            cuda_device=cuda_device,
            fallback_reason=reason,
        )
        if verbose:
            print(f"[info] {reason}; using CPU backend", file=sys.stderr)
        return _RUNTIME_CONFIG

    requested_device: DeviceMode = "auto" if requested == "auto" else "cuda"
    _RUNTIME_CONFIG = RuntimeConfig(
        requested_device=requested_device,
        active_device="cuda",
        cuda_device=cuda_device,
        fallback_reason=None,
    )
    if verbose:
        print(f"[info] Using CUDA backend on device {cuda_device}", file=sys.stderr)
    return _RUNTIME_CONFIG


def configure_runtime_from_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
    *,
    verbose: bool = False,
) -> RuntimeConfig:
    try:
        return configure_runtime(
            device=getattr(args, "device", "auto"),
            cuda_device=getattr(args, "cuda_device", 0),
            verbose=verbose,
        )
    except Exception as exc:
        if parser is not None:
            parser.error(str(exc))
        raise
