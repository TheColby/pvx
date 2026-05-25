"""GPU-accelerated augmentation transforms using PyTorch.

These transforms operate directly on ``torch.Tensor`` objects and can run
on CUDA, MPS, or CPU devices.  They avoid NumPy round-tripping and support
batched operation for maximum throughput during training.

.. warning::

   **Alpha release (0.1.0a1).**  This module is under active development.

Requirements
------------
``torch`` must be installed::

    pip install "pvx[torch]"

Usage
-----
>>> import torch
>>> from pvx.augment.gpu import TorchGainPerturber, TorchSpecAugment, TorchPipeline
>>>
>>> pipeline = TorchPipeline([
...     TorchGainPerturber(gain_db=(-6, 6), p=0.8),
...     TorchAddNoise(snr_db=(10, 30), noise_type="white", p=0.5),
...     TorchSpecAugment(freq_mask_param=27, time_mask_param=100, p=0.5),
... ])
>>>
>>> audio = torch.randn(16, 1, 48000)  # (batch, channels, samples)
>>> audio_aug = pipeline(audio, sr=16000)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


def _require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for pvx.augment.gpu. Install with: pip install 'pvx[torch]'"
        ) from exc


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class TorchTransform(ABC):
    """Base class for GPU-accelerated augmentation transforms.

    All transforms accept tensors of shape ``(batch, channels, samples)``
    or ``(channels, samples)`` or ``(samples,)`` and return the same shape.

    Parameters
    ----------
    p:
        Probability of applying this transform per batch element.
    """

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        self.p = float(p)

    def __call__(self, audio, sr: int, generator=None):
        """Apply transform with probability ``self.p``.

        Parameters
        ----------
        audio:
            ``torch.Tensor`` of shape ``(B, C, T)``, ``(C, T)``, or ``(T,)``.
        sr:
            Sample rate in Hz.
        generator:
            Optional ``torch.Generator`` for reproducibility.

        Returns
        -------
        torch.Tensor
            Augmented audio with the same shape as input.
        """
        torch = _require_torch()

        # Normalize to (B, C, T)
        original_ndim = audio.ndim
        if audio.ndim == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif audio.ndim == 2:
            audio = audio.unsqueeze(0)  # (1, C, T)

        B = audio.shape[0]

        if self.p < 1.0:
            mask = torch.rand(B, device=audio.device, generator=generator) < self.p
            if not mask.any():
                return self._restore_shape(audio, original_ndim)
            result = audio.clone()
            augmented = self.apply(audio[mask], sr, generator)
            result[mask] = augmented
        else:
            result = self.apply(audio, sr, generator)

        return self._restore_shape(result, original_ndim)

    @staticmethod
    def _restore_shape(audio, original_ndim: int):
        if original_ndim == 1:
            return audio.squeeze(0).squeeze(0)
        if original_ndim == 2:
            return audio.squeeze(0)
        return audio

    @abstractmethod
    def apply(self, audio, sr: int, generator=None):
        """Apply transform to a ``(B, C, T)`` tensor."""

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({params})"


# ---------------------------------------------------------------------------
# TorchGainPerturber
# ---------------------------------------------------------------------------


class TorchGainPerturber(TorchTransform):
    """GPU-accelerated random gain perturbation.

    Parameters
    ----------
    gain_db:
        ``(min_db, max_db)`` range for random gain in decibels.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        gain_db: tuple[float, float] = (-6.0, 6.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.gain_db_min = float(gain_db[0])
        self.gain_db_max = float(gain_db[1])

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B = audio.shape[0]
        gain_db = torch.empty(B, 1, 1, device=audio.device).uniform_(
            self.gain_db_min, self.gain_db_max, generator=generator
        )
        scale = 10.0 ** (gain_db / 20.0)
        return audio * scale


# ---------------------------------------------------------------------------
# TorchAddNoise
# ---------------------------------------------------------------------------


class TorchAddNoise(TorchTransform):
    """GPU-accelerated additive noise injection.

    Supports ``"white"``, ``"pink"``, and ``"brown"`` noise types.

    Parameters
    ----------
    snr_db:
        ``(min_snr, max_snr)`` range in dB for the signal-to-noise ratio.
    noise_type:
        ``"white"``, ``"pink"``, or ``"brown"``.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        snr_db: tuple[float, float] = (10.0, 30.0),
        noise_type: str = "white",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.snr_db_min = float(snr_db[0])
        self.snr_db_max = float(snr_db[1])
        if noise_type not in ("white", "pink", "brown"):
            raise ValueError(f"noise_type must be white/pink/brown, got {noise_type!r}")
        self.noise_type = noise_type

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape

        # Generate noise
        if self.noise_type == "white":
            noise = torch.randn(B, C, T, device=audio.device, generator=generator)
        elif self.noise_type in ("pink", "brown"):
            # Spectral shaping in frequency domain on GPU
            noise = torch.randn(B, C, T, device=audio.device, generator=generator)
            N = T // 2 + 1
            freqs = torch.arange(1, N + 1, device=audio.device, dtype=torch.float32)
            if self.noise_type == "pink":
                spectrum_scale = 1.0 / torch.sqrt(freqs)  # 1/f^0.5
            else:  # brown
                spectrum_scale = 1.0 / freqs  # 1/f
            spectrum_scale[0] = 1.0  # DC component
            noise_fft = torch.fft.rfft(noise, dim=-1)
            noise_fft = noise_fft * spectrum_scale.unsqueeze(0).unsqueeze(0)
            noise = torch.fft.irfft(noise_fft, n=T, dim=-1)
        else:
            noise = torch.randn(B, C, T, device=audio.device, generator=generator)

        # Compute per-sample SNR
        snr_db = torch.empty(B, 1, 1, device=audio.device).uniform_(
            self.snr_db_min, self.snr_db_max, generator=generator
        )

        # Scale noise to desired SNR
        signal_power = (audio**2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
        noise_power = (noise**2).mean(dim=-1, keepdim=True).clamp(min=1e-10)
        target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        scale = torch.sqrt(target_noise_power / noise_power)
        return audio + noise * scale


# ---------------------------------------------------------------------------
# TorchEQPerturber
# ---------------------------------------------------------------------------


class TorchEQPerturber(TorchTransform):
    """GPU-accelerated parametric EQ perturbation via frequency-domain shaping.

    Applies random gain at randomly chosen frequency bands in the STFT
    domain. This is a fast approximation of multi-band parametric EQ.

    Parameters
    ----------
    n_bands:
        Number of EQ bands to apply.
    gain_db_range:
        ``(min_db, max_db)`` for each band.
    q_range:
        ``(min_q, max_q)`` quality factor range (higher = narrower band).
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        n_bands: int = 4,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        q_range: tuple[float, float] = (0.5, 4.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.n_bands = int(n_bands)
        self.gain_db_min = float(gain_db_range[0])
        self.gain_db_max = float(gain_db_range[1])
        self.q_min = float(q_range[0])
        self.q_max = float(q_range[1])

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape

        n_fft = 2048
        hop_length = n_fft // 4
        N = n_fft // 2 + 1

        # Build per-batch EQ curve
        freqs = torch.linspace(0, sr / 2, N, device=audio.device)
        eq_curve = torch.ones(B, N, device=audio.device)

        for _ in range(self.n_bands):
            center_hz = torch.empty(B, device=audio.device).uniform_(
                60.0, sr / 2 - 100, generator=generator
            )
            gain_db = torch.empty(B, device=audio.device).uniform_(
                self.gain_db_min, self.gain_db_max, generator=generator
            )
            q = torch.empty(B, device=audio.device).uniform_(
                self.q_min, self.q_max, generator=generator
            )
            bandwidth = center_hz / q.clamp(min=0.1)

            # Gaussian-shaped EQ band
            diff = (freqs.unsqueeze(0) - center_hz.unsqueeze(1)) / bandwidth.unsqueeze(1)
            band_shape = torch.exp(-0.5 * diff**2)
            band_gain = 10.0 ** (gain_db.unsqueeze(1) / 20.0)
            eq_curve = eq_curve * (1.0 + band_shape * (band_gain - 1.0))

        # Apply in frequency domain per channel
        result_channels = []
        for ch in range(C):
            spec = torch.stft(
                audio[:, ch, :],
                n_fft=n_fft,
                hop_length=hop_length,
                return_complex=True,
                window=torch.hann_window(n_fft, device=audio.device),
            )  # (B, N, frames)
            spec = spec * eq_curve.unsqueeze(-1)
            reconstructed = torch.istft(
                spec,
                n_fft=n_fft,
                hop_length=hop_length,
                length=T,
                window=torch.hann_window(n_fft, device=audio.device),
            )
            result_channels.append(reconstructed)

        return torch.stack(result_channels, dim=1)


# ---------------------------------------------------------------------------
# TorchSpecAugment
# ---------------------------------------------------------------------------


class TorchSpecAugment(TorchTransform):
    """GPU-accelerated SpecAugment (Park et al. 2019).

    Applies frequency and time masking in the STFT domain, entirely on GPU.

    Parameters
    ----------
    freq_mask_param:
        Maximum number of frequency bins to mask per mask.
    time_mask_param:
        Maximum number of time frames to mask per mask.
    num_freq_masks:
        Number of frequency masks to apply.
    num_time_masks:
        Number of time masks to apply.
    fill_value:
        Value to fill masked regions (0.0 or use ``"mean"``).
    n_fft:
        FFT size for STFT.
    hop_length:
        Hop length for STFT.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        fill_value: float | str = 0.0,
        n_fft: int = 1024,
        hop_length: int = 256,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.freq_mask_param = int(freq_mask_param)
        self.time_mask_param = int(time_mask_param)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)
        self.fill_value = fill_value
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape

        window = torch.hann_window(self.n_fft, device=audio.device)

        result_channels = []
        for ch in range(C):
            spec = torch.stft(
                audio[:, ch, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
                window=window,
            )  # (B, freq_bins, time_frames)
            n_freq, n_time = spec.shape[1], spec.shape[2]

            mag = spec.abs()
            phase = spec.angle()

            fill = 0.0
            if self.fill_value == "mean":
                fill = mag.mean().item()
            elif isinstance(self.fill_value, (int, float)):
                fill = float(self.fill_value)

            # Frequency masking
            for _ in range(self.num_freq_masks):
                f = torch.randint(
                    0,
                    self.freq_mask_param + 1,
                    (B,),
                    device=audio.device,
                    generator=generator,
                )
                f0 = torch.randint(
                    0,
                    max(n_freq - self.freq_mask_param, 1),
                    (B,),
                    device=audio.device,
                    generator=generator,
                )
                for b in range(B):
                    f_end = min(int(f0[b] + f[b]), n_freq)
                    mag[b, int(f0[b]) : f_end, :] = fill

            # Time masking
            for _ in range(self.num_time_masks):
                t = torch.randint(
                    0,
                    self.time_mask_param + 1,
                    (B,),
                    device=audio.device,
                    generator=generator,
                )
                t0 = torch.randint(
                    0,
                    max(n_time - self.time_mask_param, 1),
                    (B,),
                    device=audio.device,
                    generator=generator,
                )
                for b in range(B):
                    t_end = min(int(t0[b] + t[b]), n_time)
                    mag[b, :, int(t0[b]) : t_end] = fill

            # Reconstruct
            spec_masked = torch.polar(mag, phase)
            reconstructed = torch.istft(
                spec_masked,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=T,
                window=window,
            )
            result_channels.append(reconstructed)

        return torch.stack(result_channels, dim=1)


# ---------------------------------------------------------------------------
# TorchNormalizer
# ---------------------------------------------------------------------------


class TorchNormalizer(TorchTransform):
    """GPU-accelerated peak or RMS normalization.

    Parameters
    ----------
    mode:
        ``"peak"`` or ``"rms"``.
    target_db:
        Target level in dBFS.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        mode: str = "peak",
        target_db: float = -1.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if mode not in ("peak", "rms"):
            raise ValueError(f"mode must be 'peak' or 'rms', got {mode!r}")
        self.mode = mode
        self.target_db = float(target_db)

    def apply(self, audio, sr: int, generator=None):
        target = 10.0 ** (self.target_db / 20.0)
        if self.mode == "peak":
            current = audio.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        else:
            current = audio.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-10)
        scale = target / current
        return audio * scale


# ---------------------------------------------------------------------------
# TorchClippingSimulator
# ---------------------------------------------------------------------------


class TorchClippingSimulator(TorchTransform):
    """GPU-accelerated clipping simulation.

    Parameters
    ----------
    percentile:
        ``(min, max)`` percentile range for clipping threshold.
    mode:
        ``"hard"`` for rectangular clipping, ``"soft"`` for tanh saturation.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        percentile: tuple[float, float] = (90.0, 99.5),
        mode: str = "hard",
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.pct_min = float(percentile[0])
        self.pct_max = float(percentile[1])
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft', got {mode!r}")
        self.mode = mode

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B = audio.shape[0]
        pct = torch.empty(B, device=audio.device).uniform_(
            self.pct_min / 100.0, self.pct_max / 100.0, generator=generator
        )
        # Compute per-sample threshold via quantile (one per batch element)
        flat = audio.abs().reshape(B, -1)
        thresholds = []
        for b in range(B):
            thresholds.append(torch.quantile(flat[b], pct[b]))
        th = torch.stack(thresholds).reshape(B, 1, 1)

        if self.mode == "hard":
            return audio.clamp(-th, th)
        return torch.tanh(audio / (th + 1e-8)) * th


# ---------------------------------------------------------------------------
# TorchPipeline
# ---------------------------------------------------------------------------


class TorchPipeline:
    """Sequential composition of GPU transforms.

    Parameters
    ----------
    transforms:
        Ordered list of :class:`TorchTransform` instances.
    seed:
        Optional seed for a shared ``torch.Generator``.

    Examples
    --------
    >>> pipeline = TorchPipeline([
    ...     TorchGainPerturber(gain_db=(-6, 6)),
    ...     TorchAddNoise(snr_db=(10, 30)),
    ...     TorchSpecAugment(freq_mask_param=27),
    ... ], seed=42)
    >>> audio_aug = pipeline(audio_batch, sr=16000)
    """

    def __init__(
        self,
        transforms: Sequence[TorchTransform],
        seed: int | None = None,
    ) -> None:
        self.transforms = list(transforms)
        self.seed = seed

    def __call__(self, audio, sr: int, seed: int | None = None):
        torch = _require_torch()
        s = seed if seed is not None else self.seed
        gen = None
        if s is not None:
            gen = torch.Generator(device=audio.device)
            gen.manual_seed(s)

        for t in self.transforms:
            audio = t(audio, sr, generator=gen)
        return audio

    def __repr__(self) -> str:
        inner = "\n  ".join(repr(t) for t in self.transforms)
        return f"TorchPipeline(\n  {inner}\n)"

    def __len__(self) -> int:
        return len(self.transforms)


# ---------------------------------------------------------------------------
# Adapter: wrap NumPy transforms for batched GPU usage
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TorchPhaseVocoder — core engine for time-stretch on GPU
# ---------------------------------------------------------------------------


def _torch_phase_vocoder(
    audio_1d, stretch: float, n_fft: int = 2048, hop_length: int = 512, window=None
):
    """Phase vocoder time-stretch operating on a 1-D torch tensor.

    Uses a vectorized implementation: magnitudes are interpolated in a
    single batched operation, and phase advances are computed via
    ``torch.cumsum`` instead of a Python-level per-frame loop.

    Parameters
    ----------
    audio_1d:
        ``(T,)`` float tensor.
    stretch:
        Stretch factor (>1 = slower, <1 = faster).
    n_fft:
        FFT window size.
    hop_length:
        Analysis hop size.
    window:
        Window tensor of length *n_fft*.  Defaults to Hann.

    Returns
    -------
    torch.Tensor
        Time-stretched 1-D tensor.
    """
    import torch

    device = audio_1d.device
    if window is None:
        window = torch.hann_window(n_fft, device=device)

    input_length = int(audio_1d.shape[-1])
    target_length = max(int(round(input_length * stretch)), 1)

    # STFT
    spec = torch.stft(
        audio_1d,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )  # (n_bins, n_frames)
    n_bins, n_frames = spec.shape

    # Output time steps (fractional indices into analysis frames).
    # Choose enough output frames so that the iSTFT can produce ``target_length``
    # samples (centered STFT: output_len = (n_out - 1) * hop).
    n_out_frames = max(target_length // hop_length + 1, 2)
    time_steps = (
        torch.arange(n_out_frames, device=device, dtype=torch.float32)
        * (n_frames - 1)
        / max(n_out_frames - 1, 1)
    )

    # Frame indices for interpolation (vectorized)
    frame_idx = time_steps.long()  # (n_out,)
    frac = time_steps - frame_idx.float()  # (n_out,)

    left = frame_idx.clamp(max=n_frames - 1)  # (n_out,)
    right = (frame_idx + 1).clamp(max=n_frames - 1)  # (n_out,)

    # Gather magnitudes and angles for all frames at once
    spec_mag = spec.abs()  # (n_bins, n_frames)
    spec_angle = spec.angle()  # (n_bins, n_frames)

    mag_left = spec_mag[:, left]  # (n_bins, n_out)
    mag_right = spec_mag[:, right]  # (n_bins, n_out)
    frac_2d = frac.unsqueeze(0)  # (1, n_out) for broadcasting

    # Interpolated magnitudes — fully vectorized
    mag_interp = (1.0 - frac_2d) * mag_left + frac_2d * mag_right  # (n_bins, n_out)

    # Expected phase advance per hop (analysis angular velocity)
    omega = (
        2.0
        * torch.pi
        * torch.arange(n_bins, device=device, dtype=torch.float32)
        * hop_length
        / n_fft
    )  # (n_bins,)

    # Instantaneous frequency deviation for each output frame
    angle_left = spec_angle[:, left]  # (n_bins, n_out)
    angle_right = spec_angle[:, right]  # (n_bins, n_out)
    dphi = angle_right - angle_left - omega.unsqueeze(1)  # (n_bins, n_out)

    # Principal angle wrap: dphi ∈ [-π, π)
    dphi = dphi - 2.0 * torch.pi * torch.round(dphi / (2.0 * torch.pi))

    # Phase advance per frame = omega + dphi
    phase_advance = omega.unsqueeze(1) + dphi  # (n_bins, n_out)

    # Phase accumulation via cumulative sum
    # phase[0] = angle(spec[:, 0])
    # phase[i] = phase[0] + sum(phase_advance[:, 1:i+1])
    initial_phase = spec_angle[:, 0:1]  # (n_bins, 1)
    # For frame 0, use the initial phase directly; for frames 1..N-1,
    # accumulate the phase advances.
    cumulative = torch.cumsum(phase_advance[:, 1:], dim=1)  # (n_bins, n_out-1)
    phase = torch.cat([initial_phase, initial_phase + cumulative], dim=1)  # (n_bins, n_out)

    # Reconstruct complex STFT from magnitude + phase
    out_spec = torch.polar(mag_interp, phase)  # (n_bins, n_out)

    # iSTFT — pass exact target length so the output is precisely
    # ``round(input_length * stretch)`` samples.
    result = torch.istft(
        out_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=target_length,
    )
    return result


# ---------------------------------------------------------------------------
# TorchTimeStretch
# ---------------------------------------------------------------------------


class TorchTimeStretch(TorchTransform):
    """GPU-accelerated time-stretch via a native PyTorch phase vocoder.

    No subprocess or pvx CLI required — runs entirely on the GPU.

    Parameters
    ----------
    rate:
        Stretch factor or ``(min, max)`` range.
        > 1.0 = slower (longer), < 1.0 = faster (shorter).
    n_fft:
        FFT window size (default 2048).
    hop_length:
        Hop size (default 512).
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> aug = TorchTimeStretch(rate=(0.8, 1.25))
    >>> audio_out = aug(audio_batch, sr=16000)
    """

    def __init__(
        self,
        rate: float | tuple[float, float] = (0.8, 1.25),
        n_fft: int = 2048,
        hop_length: int = 512,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(rate, (int, float)):
            self.rate_min = self.rate_max = float(rate)
        else:
            self.rate_min, self.rate_max = float(rate[0]), float(rate[1])
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, _T = audio.shape
        window = torch.hann_window(self.n_fft, device=audio.device)

        # Sample one rate per batch element
        rates = torch.empty(B, device=audio.device).uniform_(
            self.rate_min, self.rate_max, generator=generator
        )

        results = []
        for b in range(B):
            rate = rates[b].item()
            channels = []
            for ch in range(C):
                stretched = _torch_phase_vocoder(
                    audio[b, ch],
                    stretch=rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=window,
                )
                channels.append(stretched)
            # All channels have the same output length from same rate
            channels[0].shape[0]
            results.append(torch.stack(channels, dim=0))  # (C, T')

        # Pad/crop to uniform length (longest in batch)
        max_len = max(r.shape[-1] for r in results)
        padded = []
        for r in results:
            if r.shape[-1] < max_len:
                r = torch.nn.functional.pad(r, (0, max_len - r.shape[-1]))
            else:
                r = r[..., :max_len]
            padded.append(r)

        return torch.stack(padded, dim=0)


# ---------------------------------------------------------------------------
# TorchPitchShift
# ---------------------------------------------------------------------------


class TorchPitchShift(TorchTransform):
    """GPU-accelerated pitch shift via phase vocoder + resampling.

    Pitch is shifted by time-stretching then resampling to the original
    length, entirely on the GPU with no subprocess calls.

    Parameters
    ----------
    semitones:
        Shift amount or ``(min, max)`` range in semitones.
    n_fft:
        FFT window size.
    hop_length:
        Hop size.
    p:
        Probability of applying this transform.

    Examples
    --------
    >>> aug = TorchPitchShift(semitones=(-2, 2))
    >>> audio_out = aug(audio_batch, sr=16000)
    """

    def __init__(
        self,
        semitones: float | tuple[float, float] = (-2.0, 2.0),
        n_fft: int = 2048,
        hop_length: int = 512,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        if isinstance(semitones, (int, float)):
            self.st_min = self.st_max = float(semitones)
        else:
            self.st_min, self.st_max = float(semitones[0]), float(semitones[1])
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape
        window = torch.hann_window(self.n_fft, device=audio.device)

        semitones = torch.empty(B, device=audio.device).uniform_(
            self.st_min, self.st_max, generator=generator
        )

        results = []
        for b in range(B):
            st = semitones[b].item()
            # pitch_ratio = 2^(st/12); to pitch up, time-stretch by ratio then resample down
            pitch_ratio = 2.0 ** (st / 12.0)
            stretch = pitch_ratio  # stretch by pitch_ratio, then resample to original length

            channels = []
            for ch in range(C):
                stretched = _torch_phase_vocoder(
                    audio[b, ch],
                    stretch=stretch,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=window,
                )
                # Resample back to original length to preserve duration
                resampled = (
                    torch.nn.functional.interpolate(
                        stretched.unsqueeze(0).unsqueeze(0),  # (1, 1, T')
                        size=T,
                        mode="linear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )  # (T,)
                channels.append(resampled)
            results.append(torch.stack(channels, dim=0))

        return torch.stack(results, dim=0)


# ---------------------------------------------------------------------------
# TorchRoomSimulator
# ---------------------------------------------------------------------------


class TorchRoomSimulator(TorchTransform):
    """GPU-accelerated synthetic reverb via FFT convolution.

    Generates an exponential-decay noise RIR on the GPU and convolves
    with the input signal using ``torch.fft`` — no scipy required.

    This is a statistical approximation suitable for data augmentation,
    not a physics-based room model.

    Parameters
    ----------
    rt60_range:
        ``(min_s, max_s)`` reverberation time.
    wet_range:
        ``(min, max)`` wet/dry mix (0 = dry, 1 = fully wet).
    drr_db_range:
        ``(min, max)`` direct-to-reverb ratio in dB.
    p:
        Probability of applying this transform.
    """

    def __init__(
        self,
        rt60_range: tuple[float, float] = (0.2, 1.5),
        wet_range: tuple[float, float] = (0.3, 0.8),
        drr_db_range: tuple[float, float] = (3.0, 12.0),
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.rt60_min, self.rt60_max = float(rt60_range[0]), float(rt60_range[1])
        self.wet_min, self.wet_max = float(wet_range[0]), float(wet_range[1])
        self.drr_min, self.drr_max = float(drr_db_range[0]), float(drr_db_range[1])

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, C, T = audio.shape

        rt60 = torch.empty(B, device=audio.device).uniform_(
            self.rt60_min, self.rt60_max, generator=generator
        )
        wet = torch.empty(B, device=audio.device).uniform_(
            self.wet_min, self.wet_max, generator=generator
        )
        drr_db = torch.empty(B, device=audio.device).uniform_(
            self.drr_min, self.drr_max, generator=generator
        )

        results = []
        for b in range(B):
            rt60_s = rt60[b].item()
            rir_len = max(int(rt60_s * sr), 64)

            # Generate synthetic RIR: direct impulse + exponential decay noise
            decay_rate = 3.0 * 2.302585 / rt60_s  # 3*ln(10)
            t = torch.arange(rir_len, device=audio.device, dtype=torch.float32) / sr
            envelope = torch.exp(-decay_rate * t)
            noise = torch.randn(rir_len, device=audio.device, generator=generator)
            tail = noise * envelope

            direct_amp = 10.0 ** (drr_db[b].item() / 20.0)
            rir = tail.clone()
            rir[0] = rir[0] + direct_amp

            # Normalize
            rir = rir / (rir.abs().max() + 1e-8)

            # FFT convolution per channel
            conv_len = T + rir_len - 1
            fft_len = 1
            while fft_len < conv_len:
                fft_len *= 2

            rir_fft = torch.fft.rfft(rir, n=fft_len)

            ch_results = []
            for ch in range(C):
                sig_fft = torch.fft.rfft(audio[b, ch], n=fft_len)
                wet_sig = torch.fft.irfft(sig_fft * rir_fft, n=fft_len)[:T]
                w = wet[b].item()
                mixed = (1.0 - w) * audio[b, ch] + w * wet_sig
                ch_results.append(mixed)
            results.append(torch.stack(ch_results, dim=0))

        return torch.stack(results, dim=0)


# ---------------------------------------------------------------------------
# TorchMixup
# ---------------------------------------------------------------------------


class TorchMixup(TorchTransform):
    """GPU-accelerated Mixup augmentation (Zhang et al. 2018).

    Mixes pairs of audio within the batch using a Beta-distributed
    lambda, producing interpolated training examples.

    Parameters
    ----------
    alpha:
        Beta distribution parameter.  Lower values bias lambda toward 0
        and 1 (less mixing); higher values produce more uniform mixing.
    p:
        Probability of applying mixup to the batch.
    """

    def __init__(self, alpha: float = 0.4, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.alpha = float(alpha)

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        B, _C, _T = audio.shape
        if B < 2:
            return audio

        # Shuffle indices for pairing
        perm = torch.randperm(B, device=audio.device, generator=generator)

        # Sample lambda from Beta(alpha, alpha)
        # Use numpy for beta sampling (not available in torch.distributions on all devices)
        import numpy as np

        lam = float(np.random.default_rng().beta(self.alpha, self.alpha))
        lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5 (original dominates)

        mixed = lam * audio + (1.0 - lam) * audio[perm]
        return mixed


# ---------------------------------------------------------------------------
# Adapter: wrap NumPy transforms for batched GPU usage
# ---------------------------------------------------------------------------


class NumpyTransformAdapter(TorchTransform):
    """Wrap any NumPy-based ``pvx.augment.Transform`` for use in a
    :class:`TorchPipeline`.

    This is a convenience adapter — it moves data to CPU, applies the
    NumPy transform per-sample, and moves back to the original device.
    Use native ``Torch*`` transforms for maximum GPU throughput.

    Parameters
    ----------
    transform:
        A ``pvx.augment.Transform`` instance.
    p:
        Override probability (uses the original transform's p if not set).
    """

    def __init__(self, transform, p: float | None = None) -> None:
        actual_p = p if p is not None else getattr(transform, "p", 1.0)
        super().__init__(p=actual_p)
        self.transform = transform

    def apply(self, audio, sr: int, generator=None):
        torch = _require_torch()
        import numpy as np

        device = audio.device
        B, _C, _T = audio.shape
        results = []
        for b in range(B):
            np_audio = audio[b].cpu().numpy()
            seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
            aug_audio, _ = self.transform(np_audio, sr, seed=seed)
            results.append(torch.from_numpy(np.asarray(aug_audio, dtype=np.float32)))

        return torch.stack(results, dim=0).to(device)


# ---------------------------------------------------------------------------
# Batch GPU file processing
# ---------------------------------------------------------------------------


def batch_process_files(
    file_paths: Sequence[str],
    pipeline: TorchPipeline,
    *,
    sr: int = 16000,
    output_dir: str | None = None,
    output_suffix: str = "_aug",
    batch_size: int = 16,
    device: str = "auto",
    seed: int = 0,
    max_length_s: float = 30.0,
    output_format: str = "WAV",
    output_subtype: str = "PCM_16",
    progress: bool = True,
) -> list[dict[str, object]]:
    """Load multiple files, process them on GPU in batches, and write results.

    This function maximizes GPU throughput by batching multiple audio files
    into a single tensor and processing them through the TorchPipeline
    simultaneously.

    Parameters
    ----------
    file_paths:
        List of input audio file paths.
    pipeline:
        A :class:`TorchPipeline` to apply.
    sr:
        Target sample rate.  All files are resampled to this rate.
    output_dir:
        Directory for output files.  If ``None``, writes next to originals.
    output_suffix:
        Suffix appended before extension (default: ``"_aug"``).
    batch_size:
        Number of files processed simultaneously on GPU.
    device:
        ``"auto"`` (CUDA > MPS > CPU), ``"cuda"``, ``"mps"``, or ``"cpu"``.
    seed:
        Base seed for reproducibility.
    max_length_s:
        Maximum duration in seconds.  Files are truncated or zero-padded.
    output_format:
        Soundfile format (WAV, FLAC, etc.).
    output_subtype:
        Soundfile subtype (PCM_16, PCM_24, FLOAT, etc.).
    progress:
        Print progress messages.

    Returns
    -------
    list[dict]
        Per-file result dicts with keys: ``input``, ``output``, ``status``.
    """
    torch = _require_torch()
    from pathlib import Path

    import numpy as np

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    max_samples = int(max_length_s * sr)
    results: list[dict[str, object]] = []

    # Process in batches
    for batch_start in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[batch_start : batch_start + batch_size]
        len(batch_paths)

        if progress:
            end = min(batch_start + batch_size, len(file_paths))
            print(f"[batch_gpu] Processing files {batch_start + 1}-{end} of {len(file_paths)}...")

        # Load and pad/truncate to uniform length
        audio_list: list[np.ndarray] = []
        original_lengths: list[int] = []

        for fpath in batch_paths:
            try:
                from pvx.augment.core import load_audio

                audio_np, _file_sr = load_audio(fpath, target_sr=sr, mono=False)
                # Ensure (C, T) shape
                if audio_np.ndim == 1:
                    audio_np = audio_np[np.newaxis, :]
                C, T = audio_np.shape
                original_lengths.append(T)

                # Pad or truncate
                if max_samples < T:
                    audio_np = audio_np[:, :max_samples]
                elif max_samples > T:
                    pad = np.zeros((C, max_samples - T), dtype=audio_np.dtype)
                    audio_np = np.concatenate([audio_np, pad], axis=1)

                audio_list.append(audio_np)
            except (OSError, ValueError, RuntimeError) as exc:
                results.append(
                    {
                        "input": str(fpath),
                        "output": None,
                        "status": f"error: {exc}",
                    }
                )
                audio_list.append(np.zeros((1, max_samples), dtype=np.float32))
                original_lengths.append(0)

        # Stack into (B, C, T) tensor
        # All must have same channel count — use max channels
        max_ch = max(a.shape[0] for a in audio_list)
        padded = []
        for a in audio_list:
            if a.shape[0] < max_ch:
                pad_ch = np.zeros((max_ch - a.shape[0], a.shape[1]), dtype=a.dtype)
                a = np.concatenate([a, pad_ch], axis=0)
            padded.append(a)

        batch_tensor = torch.from_numpy(np.stack(padded, axis=0)).to(dev)

        # Create generator for reproducibility
        gen = torch.Generator(device=dev if dev.type != "mps" else "cpu")
        gen.manual_seed(seed + batch_start)

        # Apply pipeline
        with torch.no_grad():
            augmented = pipeline(batch_tensor, sr=sr, generator=gen)

        # Write results
        aug_np = augmented.cpu().numpy()

        for i, fpath in enumerate(batch_paths):
            # Skip files that had errors during loading
            if original_lengths[i] == 0:
                continue

            p = Path(fpath)
            out_name = f"{p.stem}{output_suffix}{p.suffix}"
            if output_dir:
                out_path = Path(output_dir) / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = p.parent / out_name

            # Trim to original length
            orig_len = min(original_lengths[i], max_samples)
            audio_out = aug_np[i, :, :orig_len]

            import soundfile as sf

            sf.write(
                str(out_path),
                audio_out.T,
                sr,
                format=output_format,
                subtype=output_subtype,
            )
            results.append(
                {
                    "input": str(fpath),
                    "output": str(out_path),
                    "status": "ok",
                }
            )

    if progress:
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"[batch_gpu] Done — {ok}/{len(file_paths)} files processed on {dev}")

    return results
