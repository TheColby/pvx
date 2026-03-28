<p align="center"><img src="../assets/pvx_logo.png" alt="pvx logo" width="192" /></p>

# pvx Mathematical Foundations

_Generated from commit `0f90deb` (commit date: 2026-03-02T19:16:48Z)._

This document explains the core signal-processing equations used by pvx, with plain-English interpretation.
All equations are written in GitHub-renderable LaTeX and are intended to render directly in normal GitHub Markdown view.

## Notation

- Analysis hop: $H_a$
- Synthesis hop: $H_s$
- FFT size: $N$
- Frame index: $t$
- Frequency-bin index: $k$
- Frame phase: $\phi_t[k]$
- Bin-center angular frequency: $\omega_k=2\pi k/N$
- Selected transform backend: $m \in \{\text{fft},\text{dft},\text{czt},\text{dct},\text{dst},\text{hartley}\}$

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

## 2. Transform Backend Families and Tradeoffs

pvx allows selecting the per-frame transform with `--transform` in STFT/ISTFT paths.
The same overlap-add framework is used, but per-bin meaning and phase behavior differ by transform family.

Complex Fourier family (`fft`, `dft`):

$$
X_t[k]=\sum_{n=0}^{N-1} x_t[n]e^{-j2\pi kn/N},\qquad
x_t[n]=\frac{1}{N}\sum_{k=0}^{N-1}X_t[k]e^{j2\pi kn/N}
$$

Chirp-Z (`czt`, Bluestein-style contour):

$$
X_t[k]=\sum_{n=0}^{N-1}x_t[n]A^{-n}W^{nk}
$$

With pvx defaults $A=1$ and $W=e^{-j2\pi/N}$, CZT matches DFT samples but uses a different numerical path.

DCT-II / IDCT-II (`dct`):

$$
C_t[k]=\alpha_k\sum_{n=0}^{N-1}x_t[n]\cos\left(\frac{\pi}{N}(n+\tfrac{1}{2})k\right)
$$

DST-II / IDST-II (`dst`):

$$
S_t[k]=\beta_k\sum_{n=0}^{N-1}x_t[n]\sin\left(\frac{\pi}{N}(n+\tfrac{1}{2})(k+1)\right)
$$

Discrete Hartley (`hartley`):

$$
H_t[k]=\sum_{n=0}^{N-1}x_t[n]\,\mathrm{cas}\left(\frac{2\pi kn}{N}\right),\quad
\mathrm{cas}(\theta)=\cos(\theta)+\sin(\theta)
$$

| Transform | Strengths | Tradeoffs | Recommended use cases |
| --- | --- | --- | --- |
| `fft` | Fastest common path, stable, native one-sided complex bins, best CUDA coverage. | Needs careful window/hop tuning for extreme nonstationarity. | Default for most time-stretch, pitch-shift, and batch processing. |
| `dft` | Canonical Fourier definition, deterministic parity baseline versus FFT implementations. | Usually slower than `fft`, little practical quality advantage in normal runs. | Cross-checking, verification, or research comparisons. |
| `czt` | Useful alternative numerical path for awkward frame sizes; can be robust for prime/non-power sizes. | Requires SciPy; typically CPU-only and slower than FFT. | Edge-case frame-size experiments and diagnostic reruns. |
| `dct` | Real-valued compact basis with strong energy compaction for smooth envelopes. | Loses explicit complex phase; less transparent for strict phase-coherent resynthesis. | Spectral shaping, denoise-like creative processing, coefficient-domain experiments. |
| `dst` | Real-valued odd-symmetry basis; emphasizes different boundary behavior than DCT. | Same phase limitations as DCT; content-dependent coloration. | Creative timbre variants and odd-symmetry analysis experiments. |
| `hartley` | Real transform using `cas`, often helpful for CPU-friendly exploratory analysis. | Different bin semantics vs complex STFT; can alter artifact profile. | Alternative real-basis phase-vocoder experiments and pedagogical comparisons. |

Plain English: choose `fft` first, then move to other transforms when you specifically need different numerical behavior, basis structure, or artifact character.

### Sample Transform Use Cases

```bash
pvx voc dialog.wav --transform fft --time-stretch 1.08 --transient-preserve --output-dir out
pvx voc test_tones.wav --transform dft --time-stretch 1.00 --output-dir out
pvx voc archival_take.wav --transform czt --n-fft 1536 --win-length 1536 --hop-size 384 --output-dir out
pvx voc strings.wav --transform dct --pitch-shift-cents -17 --soft-clip-level 0.94 --output-dir out
pvx voc percussion.wav --transform dst --time-stretch 0.92 --output-dir out
pvx voc synth.wav --transform hartley --phase-locking off --time-stretch 1.30 --output-dir out
```

## 3. Phase-Vocoder Frequency and Phase Update

$$
\Delta\phi_t[k]=\mathrm{princarg}\left(\phi_t[k]-\phi_{t-1}[k]-\omega_kH_a\right)
$$
$$
\hat{\omega}_t[k]=\omega_k+\frac{\Delta\phi_t[k]}{H_a}
$$
$$
\hat{\phi}_t[k]=\hat{\phi}_{t-1}[k]+\hat{\omega}_t[k]H_s
$$

Plain English: pvx estimates true per-bin frequency from wrapped phase deviation, then re-accumulates phase at the synthesis hop.

## 4. Time Stretch and Pitch Ratio

Pitch shift ratio from semitones/cents:

$$
r_{\text{pitch}}=2^{\Delta s/12}=2^{\Delta c/1200}
$$

Plain English: semitone and cent controls map to the same multiplicative ratio.

## 5. Dynamic Control Interpolation (CSV/JSON)

When a numeric flag receives a CSV/JSON control track, pvx samples a control function $u(t)$ over render time.

Polynomial mode fits a global polynomial of degree $d$:

$$
u(t)=\sum_{i=0}^{d} a_i t^i,\qquad d=\min(\texttt{order}, M-1)
$$

where $M$ is the number of control points. In command-line interface (CLI) usage, `--order` accepts any integer $\ge 1$.
The effective degree is automatically capped to avoid over-specification when there are too few points.

Nearest/linear/cubic/exponential/S-curve are local interpolation modes; `none` is sample-and-hold (stairstep).
`exponential` uses piecewise exponential easing and `s_curve`/`smootherstep` use Hermite S-shaped easing between adjacent control points.
Legend used in each plot: blue solid line = interpolated control curve $u(t)$, red dashed line = piecewise connection of control points, red circles labeled $p_i$ = original control points.

| Interpolation mode | CLI form | Example plot |
| --- | --- | --- |
| none (stairstep) | `--interp none` | ![none (stairstep)](assets/interpolation/interp_none.svg) |
| nearest | `--interp nearest` | ![nearest](assets/interpolation/interp_nearest.svg) |
| linear | `--interp linear` | ![linear](assets/interpolation/interp_linear.svg) |
| cubic | `--interp cubic` | ![cubic](assets/interpolation/interp_cubic.svg) |
| exponential | `--interp exponential` | ![exponential](assets/interpolation/interp_exponential.svg) |
| S-curve (smoothstep) | `--interp s_curve` | ![S-curve (smoothstep)](assets/interpolation/interp_s_curve.svg) |
| S-curve (smootherstep) | `--interp smootherstep` | ![S-curve (smootherstep)](assets/interpolation/interp_smootherstep.svg) |
| polynomial order 1 | `--interp polynomial --order 1` | ![polynomial order 1](assets/interpolation/interp_polynomial_order_1.svg) |
| polynomial order 2 | `--interp polynomial --order 2` | ![polynomial order 2](assets/interpolation/interp_polynomial_order_2.svg) |
| polynomial order 3 | `--interp polynomial --order 3` | ![polynomial order 3](assets/interpolation/interp_polynomial_order_3.svg) |
| polynomial order 5 | `--interp polynomial --order 5` | ![polynomial order 5](assets/interpolation/interp_polynomial_order_5.svg) |

## 6. Microtonal Retune Mapping

For a detected frequency $f$, pvx scale-aware retuning chooses the nearest permitted scale target $f_{\text{scale}}$ and applies:

$$
\Delta s = 12\log_2\left(\frac{f_{\text{scale}}}{f}\right)
$$

Plain English: each frame is shifted only as much as needed to land on the selected microtonal scale degree.

## 7. Loudness and Dynamics

LUFS target gain:

$$
g_{\text{LUFS}}=10^{(L_{\text{target}}-L_{\text{in}})/20},\qquad y[n]=g_{\text{LUFS}}x[n]
$$

Compressor gain law (conceptual):

$$
g(x)=\begin{cases}
1, & |x|\le T \\
\left(\frac{T+(|x|-T)/R}{|x|}\right), & |x|>T
\end{cases}
$$

Plain English: level is reduced above threshold $T$ by ratio $R$, then makeup/loudness stages may be applied.

## 8. Spatial Coherence and Channel Alignment

Inter-channel phase difference:

$$
\Delta\phi_{ij}(k,t)=\phi_i(k,t)-\phi_j(k,t)
$$

Phase-drift objective:

$$
J=\sum_{k,t}\left|\Delta\phi^{\text{out}}_{ij}(k,t)-\Delta\phi^{\text{in}}_{ij}(k,t)\right|
$$

Channel delay estimate by phase-weighted cross-correlation:

$$
\tau^*=\arg\max_\tau\,\mathcal{F}^{-1}\left\{\frac{X_i(\omega)X_j^*(\omega)}{|X_i(\omega)X_j^*(\omega)|+\varepsilon}\right\}(\tau)
$$

Plain English: pvx spatial modules prioritize channel coherence, stable image cues, and robust delay alignment for multichannel processing chains.

## 9. Function Graph Gallery

The plots below visualize core transfer functions and parameter curves used across pvx processing modules.

| Function family | Plot | Why this matters |
| --- | --- | --- |
| Pitch ratio from semitones | ![Pitch ratio from semitones](assets/functions/pitch_ratio_vs_semitones.svg) | `r = 2^(semitones/12)` conversion used by pitch-shift flags. |
| Pitch ratio from cents | ![Pitch ratio from cents](assets/functions/pitch_ratio_vs_cents.svg) | `r = 2^(cents/1200)` conversion for microtonal cent controls. |
| Dynamics transfer functions | ![Dynamics transfer functions](assets/functions/dynamics_transfer_curves.svg) | Compressor/expander/limiter conceptual transfer curves used in mastering stages. |
| Soft-clip transfer functions | ![Soft-clip transfer functions](assets/functions/softclip_transfer_functions.svg) | Shows `tanh`, `arctan`, and `cubic` soft clipping behavior. |
| Morph blend magnitude curves | ![Morph blend magnitude curves](assets/functions/morph_blend_magnitude_curves.svg) | Compares `linear`, `geometric`, `product`, `max_mag`, and `min_mag` blend behaviors. |
| Mask exponent response | ![Mask exponent response](assets/functions/mask_exponent_curves.svg) | Shows how `--mask-exponent` shapes cross-synthesis mask gain. |
| Phase mix to output angle | ![Phase mix to output angle](assets/functions/phase_mix_angle_curve.svg) | Example of complex-vector phase blending used by morph phase mix. |

## Attribution

Copyright (c) 2026 Colby Leider and contributors. See [`ATTRIBUTION.md`](../ATTRIBUTION.md).
