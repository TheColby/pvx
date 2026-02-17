# PVX Mathematical Foundations

_Generated from commit `a2d5649` (commit date: 2026-02-17T11:08:46-05:00)._

This document explains the core signal-processing equations used by PVX, with plain-English interpretation.
All equations are written in GitHub-renderable LaTeX and are intended to render directly in normal GitHub Markdown view.

## Notation

- Analysis hop: $H_a$
- Synthesis hop: $H_s$
- FFT size: $N$
- Frame index: $t$
- Frequency-bin index: $k$
- Frame phase: $\phi_t[k]$
- Bin-center angular frequency: $\omega_k=2\pi k/N$

## 1. STFT Analysis and Synthesis

Analysis transform:

$$
X_t[k]=\sum_{n=0}^{N-1} x[n+tH_a]w[n]e^{-j2\pi kn/N}
$$

Plain English: each frame `t` is windowed by `w[n]` and transformed into complex frequency bins `k`.

Weighted overlap-add synthesis:

$$
\hat{x}[n]=\frac{\sum_t \hat{x}_t[n-tH_s]w[n-tH_s]}{\sum_t w^2[n-tH_s]+\varepsilon}
$$

Plain English: reconstructed frames are overlap-added, then normalized by accumulated window energy.

## 2. Phase-Vocoder Frequency and Phase Update

$$
\Delta\phi_t[k]=\operatorname{princarg}\left(\phi_t[k]-\phi_{t-1}[k]-\omega_kH_a\right)
$$
$$
\hat{\omega}_t[k]=\omega_k+\frac{\Delta\phi_t[k]}{H_a}
$$
$$
\hat{\phi}_t[k]=\hat{\phi}_{t-1}[k]+\hat{\omega}_t[k]H_s
$$

Plain English: PVX estimates true per-bin frequency from wrapped phase deviation, then re-accumulates phase at the synthesis hop.

## 3. Time Stretch and Pitch Ratio

Pitch shift ratio from semitones/cents:

$$
r_{\text{pitch}}=2^{\Delta s/12}=2^{\Delta c/1200}
$$

Plain English: semitone and cent controls map to the same multiplicative ratio.

## 4. Microtonal Retune Mapping

For a detected frequency $f$, PVX scale-aware retuning chooses the nearest permitted scale target $f_{\text{scale}}$ and applies:

$$
\Delta s = 12\log_2\left(\frac{f_{\text{scale}}}{f}\right)
$$

Plain English: each frame is shifted only as much as needed to land on the selected microtonal scale degree.

## 5. Loudness and Dynamics

LUFS target gain:

$$
g_{\text{LUFS}}=10^{(L_{\text{target}}-L_{\text{in}})/20},\qquad y[n]=g_{\text{LUFS}}x[n]
$$

Compressor gain law (conceptual):

$$
g(x)=\begin{cases}
1, & |x|\le T \
\left(\frac{T+(|x|-T)/R}{|x|}\right), & |x|>T
\end{cases}
$$

Plain English: level is reduced above threshold $T$ by ratio $R$, then makeup/loudness stages may be applied.

## 6. Spatial and Multichannel Core Equations

Delay-and-sum beamforming:

$$
y[n]=\frac{1}{M}\sum_{m=1}^{M}x_m[n-\tau_m]
$$

MVDR beamforming:

$$
\mathbf{w}=\frac{\mathbf{R}^{-1}\mathbf{d}}{\mathbf{d}^H\mathbf{R}^{-1}\mathbf{d}}
$$

FOA ambisonic encoding:

$$
W=\frac{s}{\sqrt{2}},\quad X=s\cos\theta\cos\varphi,\quad Y=s\sin\theta\cos\varphi,\quad Z=s\sin\varphi
$$

Plain English: PVX spatial modules combine array steering, covariance-weighted beamforming, and ambisonic transforms with phase-consistent DSP.
