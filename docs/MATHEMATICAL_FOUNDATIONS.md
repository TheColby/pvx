# pvx Mathematical Foundations

This document explains the core DSP math used in pvx with equation-first detail and plain-English interpretation.
All formulas are written in GitHub-compatible LaTeX.

## 0) Notation

- Time-domain signal: $x[n]$
- Output signal: $y[n]$
- Frame index: $t$
- Sample index inside frame: $n$
- Frequency-bin index: $k$
- FFT size: $N$
- Analysis hop: $H_a$
- Synthesis hop: $H_s$
- Window: $w[n]$
- Analysis phase: $\phi_t[k]$
- Synthesis phase: $\hat{\phi}_t[k]$
- Bin-center angular frequency: $\omega_k = 2\pi k/N$
- Small numerical guard: $\varepsilon > 0$

---

## 1) STFT Analysis and Synthesis

Analysis:

$$
X_t[k] = \sum_{n=0}^{N-1} x[n+tH_a]\,w[n]\,e^{-j2\pi kn/N}
$$

Magnitude-phase form:

$$
X_t[k] = |X_t[k]|e^{j\phi_t[k]}
$$

Frame-domain synthesis:

$$
\tilde{x}_t[n] = \frac{1}{N}\sum_{k=0}^{N-1}\hat{X}_t[k]e^{j2\pi kn/N}
$$

Weighted overlap-add (WOLA):

$$
y[m] =
\frac{\sum_t \tilde{x}_t[m-tH_s]\,w[m-tH_s]}
{\sum_t w^2[m-tH_s] + \varepsilon}
$$

Plain English:
- STFT maps short windowed slices to complex spectra.
- ISTFT overlap-add reconstructs time-domain audio and normalizes window energy.

---

## 2) COLA and Reconstruction Constraints

Constant overlap-add condition (conceptual):

$$
\sum_t w[m-tH] = C
$$

Weighted variant used in practice:

$$
\sum_t w^2[m-tH] \approx C_w
$$

If this denominator approaches zero in any region, reconstruction becomes unstable, which is why pvx uses explicit denominator guards.

---

## 3) Phase-Vocoder Instantaneous Frequency

Wrapped phase deviation:

$$
\Delta\phi_t[k] =
\mathrm{wrap}_{(-\pi,\pi]}
\left(
\phi_t[k]-\phi_{t-1}[k]-\omega_k H_a
\right)
$$

Instantaneous angular frequency estimate:

$$
\hat{\omega}_t[k] = \omega_k + \frac{\Delta\phi_t[k]}{H_a}
$$

Synthesis phase accumulation:

$$
\hat{\phi}_t[k] = \hat{\phi}_{t-1}[k] + \hat{\omega}_t[k]H_s
$$

Synthesis spectrum:

$$
\hat{X}_t[k] = \hat{M}_t[k]e^{j\hat{\phi}_t[k]}
$$

Plain English:
- pvx estimates per-bin true frequency from phase differences, then accumulates phase using the synthesis hop.

---

## 4) Time Stretch Mapping

Nominal stretch ratio:

$$
r_t = \frac{H_s}{H_a}
$$

Output duration approximation:

$$
T_{\mathrm{out}} \approx r_t T_{\mathrm{in}}
$$

Segment-wise stretch map:

$$
r_t =
\begin{cases}
r_1, & t \in [\tau_0,\tau_1) \\
r_2, & t \in [\tau_1,\tau_2) \\
\cdots
\end{cases}
$$

Multistage decomposition for large ratios:

$$
r_{\mathrm{total}} = \prod_{i=1}^{S} r_i
$$

Plain English:
- Large extreme ratios are safer when factored into multiple moderate stages.

---

## 5) Identity Phase Locking

Let $p_t$ be the local spectral peak bin for frame $t$ and neighborhood $\mathcal{N}(p_t)$.

Relative analysis phase of a neighbor bin $q$:

$$
\delta_t[q] = \phi_t[q] - \phi_t[p_t]
$$

Locked synthesis phase:

$$
\hat{\phi}_t[q] = \hat{\phi}_t[p_t] + \delta_t[q], \quad q \in \mathcal{N}(p_t)
$$

Plain English:
- Neighbors follow peak-bin motion to reduce local phase shearing and blur.

---

## 6) Transient Detection Math

### 6.1 Spectral Flux

With magnitude $M_t[k]=|X_t[k]|$:

$$
F_t = \sqrt{\sum_k \left(\max(0, M_t[k]-M_{t-1}[k])\right)^2}
$$

### 6.2 High-Frequency Content

$$
H_t = \sum_k k\,M_t[k]
$$

### 6.3 Broadbandness (flatness-like ratio)

$$
B_t = \frac{\exp\left(\frac{1}{K}\sum_k \log(M_t[k]+\varepsilon)\right)}
{\frac{1}{K}\sum_k M_t[k] + \varepsilon}
$$

### 6.4 Combined onset score

$$
S_t = \alpha_F \tilde{F}_t + \alpha_H \tilde{H}_t + \alpha_B \tilde{B}_t
$$

where tildes denote normalized feature tracks.

Plain English:
- pvx combines multiple cues so onset detection is less brittle than a single feature.

---

## 7) WSOLA Similarity and Alignment

Given reference segment $r[n]$ and candidate $c_\tau[n]$ shifted by lag $\tau$:

$$
\rho(\tau)=
\frac{\sum_n r[n]\,c_\tau[n]}
{\sqrt{\sum_n r[n]^2}\sqrt{\sum_n c_\tau[n]^2}+\varepsilon}
$$

Best lag:

$$
\tau^\star = \arg\max_{\tau\in\Omega} \rho(\tau)
$$

Overlap-add with crossfade window $g[n]\in[0,1]$:

$$
y[n] = g[n]\,a[n] + (1-g[n])\,b[n]
$$

Plain English:
- WSOLA finds the best local waveform offset before stitching blocks.

---

## 8) Hybrid PV + WSOLA Stitching

Let transient mask be $m[n]\in\{0,1\}$ (1 for transient region).

PV output: $y_{\mathrm{pv}}[n]$, WSOLA output: $y_{\mathrm{ws}}[n]$.

Soft region blending:

$$
y[n] = (1-\lambda[n])\,y_{\mathrm{pv}}[n] + \lambda[n]\,y_{\mathrm{ws}}[n]
$$

where $\lambda[n]$ is obtained from transient mask and crossfade smoothing.

Plain English:
- Steady-state gets PV coherence, transient zones get time-domain attack preservation.

---

## 9) Stereo and Multichannel Coherence

### 9.1 Mid/Side transform

$$
M = \frac{L+R}{\sqrt{2}}, \qquad S = \frac{L-R}{\sqrt{2}}
$$

Inverse:

$$
L = \frac{M+S}{\sqrt{2}}, \qquad R = \frac{M-S}{\sqrt{2}}
$$

### 9.2 Reference-channel lock (conceptual)

For channel $c$ phase increment $\Delta\phi_t^{(c)}[k]$ and reference channel $r$:

$$
\Delta\phi_{t,\mathrm{locked}}^{(c)}[k]
=
(1-\gamma)\Delta\phi_t^{(c)}[k]
\gamma\,\Delta\phi_t^{(r)}[k]
$$

with coherence strength $\gamma\in[0,1]$.

Plain English:
- $\gamma=0$ means independent channels; $\gamma=1$ means fully reference-driven increments.

---

## 10) Transform Backends Used in pvx

### 10.1 DFT/FFT

$$
X[k]=\sum_{n=0}^{N-1}x[n]e^{-j2\pi kn/N},\qquad
x[n]=\frac{1}{N}\sum_{k=0}^{N-1}X[k]e^{j2\pi kn/N}
$$

### 10.2 CZT

$$
X[k] = \sum_{n=0}^{N-1} x[n]A^{-n}W^{nk}
$$

### 10.3 DCT-II

$$
C[k]=\alpha_k\sum_{n=0}^{N-1}x[n]\cos\left(\frac{\pi}{N}(n+\tfrac12)k\right)
$$

### 10.4 DST-II

$$
S[k]=\beta_k\sum_{n=0}^{N-1}x[n]\sin\left(\frac{\pi}{N}(n+\tfrac12)(k+1)\right)
$$

### 10.5 Hartley

$$
H[k]=\sum_{n=0}^{N-1}x[n]\left[\cos\left(\frac{2\pi kn}{N}\right)+\sin\left(\frac{2\pi kn}{N}\right)\right]
$$

Plain English:
- FFT is the default production path; other transforms are useful for analysis and controlled experiments.

---

## 11) Pitch Ratio, Semitones, Cents, and Microtonal Targets

Semitones to ratio:

$$
r = 2^{\Delta s/12}
$$

Cents to ratio:

$$
r = 2^{\Delta c/1200}
$$

General $N$-TET step mapping:

$$
r = 2^{m/N}
$$

Just ratio:

$$
r = \frac{p}{q},\qquad p,q\in\mathbb{Z}^+
$$

Detected F0 retune amount:

$$
\Delta s = 12\log_2\left(\frac{f_{\mathrm{target}}}{f_{\mathrm{detected}}+\varepsilon}\right)
$$

Plain English:
- pvx accepts floating, rational, and expression-based pitch ratios for microtonal control.

---

## 12) Formant Preservation (Cepstral Envelope)

Log magnitude spectrum:

$$
L[k] = \log(|X[k]|+\varepsilon)
$$

Real cepstrum:

$$
c[n] = \frac{1}{N}\sum_{k=0}^{N-1}L[k]e^{j2\pi kn/N}
$$

Liftered envelope cepstrum:

$$
c_e[n] =
\begin{cases}
c[n], & |n|\le n_c \\
0, & \text{otherwise}
\end{cases}
$$

Reconstructed envelope:

$$
E[k] = \exp\left(\sum_{n=0}^{N-1}c_e[n]e^{-j2\pi kn/N}\right)
$$

Plain English:
- Low-quefrency cepstral structure approximates vocal tract envelope; pvx uses this to reduce formant drift.

---

## 13) Dynamics and Loudness

### 13.1 LUFS target gain

$$
g_{\mathrm{LUFS}} = 10^{(L_{\mathrm{target}}-L_{\mathrm{in}})/20}
$$

$$
y[n]=g_{\mathrm{LUFS}}x[n]
$$

### 13.2 Compressor (static curve)

For level $u$ in dBFS, threshold $T$, ratio $R$:

$$
u_{\mathrm{out}} =
\begin{cases}
u, & u\le T\\
T+\frac{u-T}{R}, & u>T
\end{cases}
$$

### 13.3 Soft clip example

$$
y = \tanh(\alpha x)
$$

Plain English:
- pvx mastering stages are optional and intended for conservative post-processing.

---

## 14) Objective Metrics in Benchmarking and Runtime Compare

### 14.1 SNR

$$
\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}\frac{\mathbb{E}[x^2]}{\mathbb{E}[(x-\hat{x})^2]+\varepsilon}
$$

### 14.2 SI-SDR

$$
\alpha = \frac{\langle \hat{x},x\rangle}{\langle x,x\rangle+\varepsilon}
$$

$$
\mathrm{SI\text{-}SDR}_{\mathrm{dB}} =
10\log_{10}
\frac{\|\alpha x\|_2^2+\varepsilon}{\|\hat{x}-\alpha x\|_2^2+\varepsilon}
$$

### 14.3 Log-spectral distance

$$
\mathrm{LSD}=
\sqrt{
\frac{1}{TK}
\sum_{t,k}
\left(
20\log_{10}|X_t[k]|-
20\log_{10}|\hat{X}_t[k]|
\right)^2
}
$$

### 14.4 Spectral convergence

$$
\mathrm{SC}=
\frac{\|\,|X|-|\hat{X}|\,\|_F}{\|\,|X|\,\|_F+\varepsilon}
$$

### 14.5 Envelope correlation

For onset envelopes $e$ and $\hat{e}$:

$$
\rho =
\frac{\langle e-\bar{e},\hat{e}-\bar{\hat{e}}\rangle}
{\|e-\bar{e}\|_2\|\hat{e}-\bar{\hat{e}}\|_2+\varepsilon}
$$

Plain English:
- pvx tracks both waveform-level and perceptual-proxy indicators; quality is judged from a metric set, not one number.

---

## 15) Stereo Quality Metrics

Interaural level difference:

$$
\mathrm{ILD}_{\mathrm{dB}} = 20\log_{10}\frac{\mathrm{RMS}(L)}{\mathrm{RMS}(R)}
$$

Interaural time difference estimate:

$$
\tau^\star = \arg\max_{\tau\in[-\tau_{\max},\tau_{\max}]}
\sum_n L[n]R[n-\tau]
$$

Inter-channel phase deviation (binwise):

$$
\Delta\phi_{\mathrm{IPD}}[k,t] =
\mathrm{wrap}_{(-\pi,\pi]}
\left(
(\phi_L-\phi_R)_{\mathrm{out}}
-
(\phi_L-\phi_R)_{\mathrm{in}}
\right)
$$

Plain English:
- these metrics quantify stereo image stability under time/pitch processing.

---

## 16) Artifact-Proxy Metrics

Phasiness proxy:

$$
P = 1 - \left|
\frac{1}{T}\sum_t e^{j\Delta\phi_t}
\right|
$$

Musical-noise proxy from residual flatness and sparsity:

$$
M \propto
\left(1-\frac{\exp(\mathbb{E}[\log R])}{\mathbb{E}[R]+\varepsilon}\right)
\cdot
\mathrm{Var}(R)
$$

Pre-echo score near onsets:

$$
E_{\mathrm{pre}} =
\mathbb{E}_i
\left[
\max\left(
0,\,
\frac{P_{\mathrm{pre,out}}^{(i)}}{P_{\mathrm{post,out}}^{(i)}+\varepsilon}
-
\frac{P_{\mathrm{pre,in}}^{(i)}}{P_{\mathrm{post,in}}^{(i)}+\varepsilon}
\right)
\right]
$$

Plain English:
- these are practical diagnostics for common phase-vocoder failure modes.

---

## 17) Complexity and Practical Scaling

FFT frame complexity:

$$
\mathcal{O}(N\log N)
$$

Per-second frame count:

$$
F \approx \frac{f_s}{H_a}
$$

Approximate transform workload per second per channel:

$$
W \propto \frac{f_s}{H_a}\,N\log N
$$

Multichannel scaling:

$$
W_{\mathrm{total}} \propto C\,W
$$

Plain English:
- runtime rises with more channels, larger FFTs, and smaller hops; pvx still prioritizes quality-safe settings first.

---

## 18) Quality-First Parameter Couplings

Useful coupling heuristics:

$$
\text{overlap ratio} = 1 - \frac{H_a}{N}
$$

Higher overlap ratio generally improves continuity but increases compute cost.

For extreme stretch:

$$
r_{\mathrm{stage}} \le r_{\max,\mathrm{stage}}
\quad\Rightarrow\quad
S \ge \left\lceil \frac{\log r_{\mathrm{total}}}{\log r_{\max,\mathrm{stage}}} \right\rceil
$$

Transient crossfade sample count:

$$
N_{\mathrm{xfade}} = \left\lfloor \frac{f_s\,t_{\mathrm{xfade,ms}}}{1000} \right\rfloor
$$

Plain English:
- these couplings explain why pvx defaults toward stability and artifact control before speed optimization.

---

## 19) Window Families, Leakage, and Resolution

Generalized cosine window family:

$$
w[n] = \sum_{m=0}^{M} a_m \cos\left(\frac{2\pi m n}{N-1}\right), \quad 0 \le n < N
$$

Examples:

$$
w_{\mathrm{hann}}[n] = 0.5 - 0.5\cos\left(\frac{2\pi n}{N-1}\right)
$$

$$
w_{\mathrm{hamming}}[n] = 0.54 - 0.46\cos\left(\frac{2\pi n}{N-1}\right)
$$

Equivalent noise bandwidth (ENBW) in bins:

$$
\mathrm{ENBW} =
\frac{N\sum_{n=0}^{N-1} w[n]^2}
{\left(\sum_{n=0}^{N-1} w[n]\right)^2}
$$

Main-lobe width and sidelobe level trade off against each other; lower sidelobes generally mean wider main lobes.

Plain English:
- window choice controls leakage rejection versus frequency selectivity, which directly affects phase-vocoder artifact character.

---

## 20) Transient Phase Reset Logic

Let binary onset mask be $m_t \in \{0,1\}$ for frame $t$.

One reset strategy:

$$
\hat{\phi}_t[k] =
\begin{cases}
\phi_t[k], & m_t = 1 \\
\hat{\phi}_{t-1}[k] + \hat{\omega}_t[k]H_s, & m_t = 0
\end{cases}
$$

Soft reset blend:

$$
\hat{\phi}_t[k] = \beta_t\phi_t[k] + (1-\beta_t)\left(\hat{\phi}_{t-1}[k] + \hat{\omega}_t[k]H_s\right)
$$

with $\beta_t \in [0,1]$ increasing near transient boundaries.

Plain English:
- resetting or blending phase at onsets reduces attack blur at the cost of some long-range phase continuity.

---

## 21) Crossfades and Click Suppression

Linear crossfade between streams $a[n]$ and $b[n]$:

$$
y[n] = (1-\lambda[n])a[n] + \lambda[n]b[n]
$$

Equal-power variant:

$$
y[n] = \cos\theta[n]\,a[n] + \sin\theta[n]\,b[n], \quad \theta[n]\in[0,\pi/2]
$$

Smooth ramp example with $L$-sample boundary:

$$
\lambda[n] = \frac{1}{2}\left(1-\cos\frac{\pi n}{L-1}\right), \quad 0 \le n < L
$$

Plain English:
- windowed crossfades at region boundaries are essential for click-free PV/WSOLA stitching.

---

## 22) Time-Varying Pitch Control Smoothing

Given raw pitch-ratio control $r_t$, one-pole smoothing:

$$
\tilde{r}_t = \alpha r_t + (1-\alpha)\tilde{r}_{t-1}, \quad \alpha \in (0,1]
$$

Confidence-gated control:

$$
r_t^\star =
\begin{cases}
r_t, & c_t \ge c_{\min} \\
r_{t-1}^\star, & c_t < c_{\min} \text{ and hold mode}
\end{cases}
$$

Microtonal row interpolation between $(t_i,r_i)$ and $(t_{i+1},r_{i+1})$:

$$
r(t)=r_i + \frac{t-t_i}{t_{i+1}-t_i}(r_{i+1}-r_i)
$$

Plain English:
- control smoothing prevents zipper noise and confidence gating avoids unstable pitch commands.

---

## 23) Ratio Parsing: Float, Rational, and Expressions

Rational ratio:

$$
r = \frac{p}{q}, \quad p,q\in\mathbb{Z}_{>0}
$$

Equal-tempered semitone mapping:

$$
r = 2^{s/12}
$$

General $N$-TET step mapping:

$$
r = 2^{u/N}
$$

Cent mapping:

$$
r = 2^{c/1200}
$$

Plain English:
- pvx pitch controls are unified as a ratio stream regardless of how the user specifies interval values.

---

## 24) Analysis-Synthesis Error and Invertibility Diagnostics

Reconstruction residual:

$$
e[n] = y[n] - x[n]
$$

Relative error norm:

$$
\eta = \frac{\lVert e \rVert_2}{\lVert x \rVert_2 + \varepsilon}
$$

Identity test target:

$$
\text{stretch}=1,\ \text{pitch}=0
\quad\Rightarrow\quad
\eta \text{ should be small}
$$

Plain English:
- even perfect-parameter identity is not exact in floating-point STFT pipelines, but residual should remain low and stable.

---

## 25) Dynamics Chain Transfer Curves

Level in dB: $x_{\mathrm{dB}}$, threshold $T$, ratio $R$.

Compressor static curve:

$$
y_{\mathrm{dB}}=
\begin{cases}
x_{\mathrm{dB}}, & x_{\mathrm{dB}}\le T \\
T + \frac{x_{\mathrm{dB}}-T}{R}, & x_{\mathrm{dB}} > T
\end{cases}
$$

Expander (downward) example:

$$
y_{\mathrm{dB}}=
\begin{cases}
T + R_e(x_{\mathrm{dB}}-T), & x_{\mathrm{dB}} < T,\ R_e>1 \\
x_{\mathrm{dB}}, & x_{\mathrm{dB}} \ge T
\end{cases}
$$

Limiter ceiling:

$$
y_{\mathrm{dB}} = \min(x_{\mathrm{dB}}, C)
$$

Plain English:
- mastering controls are nonlinear amplitude maps after core time/pitch processing.

---

## 26) Why Output SNR Can Be Low After Intentional Time/Pitch Changes

Conventional sample-aligned SNR:

$$
\mathrm{SNR} = 10\log_{10}\frac{\sum_n x[n]^2}{\sum_n (x[n]-y[n])^2 + \varepsilon}
$$

If $y$ has intentionally shifted timing or pitch, direct subtraction is not a perceptual quality test.

Aligned SNR idea (after coarse alignment operator $\mathcal{A}$):

$$
\mathrm{SNR}_{\mathrm{aligned}} =
10\log_{10}\frac{\sum_n x[n]^2}
{\sum_n (x[n]-\mathcal{A}(y)[n])^2 + \varepsilon}
$$

Plain English:
- low raw SNR does not necessarily mean poor sound quality when the output is intentionally transformed.

---

## 27) Multi-Objective Quality Score (Benchmark Aggregation)

Let normalized metric vector be $m_i$ with positive directionality.

Weighted score:

$$
Q = \sum_{i=1}^{P} w_i m_i,\quad \sum_i w_i=1,\ w_i\ge0
$$

Constraint-style gating:

$$
m_i \ge \tau_i \ \forall i \in \mathcal{C}
$$

Plain English:
- pvx benchmark gates should enforce minimum quality floors first, then optimize composite score second.

---

## 28) Chained-Module Error Propagation

For chain of $J$ modules:

$$
y = (f_J \circ f_{J-1} \circ \cdots \circ f_1)(x)
$$

Local perturbation at stage $j$: $\delta_j$.

First-order propagated bound (conceptual):

$$
\lVert \delta_{\mathrm{out}} \rVert
\lesssim
\sum_{j=1}^{J}
\left(
\prod_{k=j+1}^{J} L_k
\right)
\lVert \delta_j \rVert
$$

where $L_k$ are local Lipschitz-like gain factors.

Plain English:
- long command pipelines compound artifacts, so each stage should be conservative.

---

## 29) Sample-Rate-Aware Parameter Scaling

If a parameter is time-based in milliseconds $t_{\mathrm{ms}}$:

$$
N_t = \left\lfloor \frac{f_s t_{\mathrm{ms}}}{1000} \right\rfloor
$$

To keep equivalent temporal behavior when moving from $f_{s1}$ to $f_{s2}$:

$$
N_{t,2} = N_{t,1}\frac{f_{s2}}{f_{s1}}
$$

Frequency bin spacing:

$$
\Delta f = \frac{f_s}{N}
$$

Plain English:
- porting presets across sample rates requires re-scaling sample-count parameters and revisiting FFT-size frequency resolution.

---

## 30) Practical Delta Tables Used in Console Metrics

Given metric value $m_{\mathrm{in}}$ and $m_{\mathrm{out}}$:

$$
\Delta m = m_{\mathrm{out}} - m_{\mathrm{in}}
$$

Relative percent change (when denominator safe):

$$
\Delta_{\%} = 100\cdot\frac{m_{\mathrm{out}}-m_{\mathrm{in}}}{|m_{\mathrm{in}}|+\varepsilon}
$$

Signed interpretation convention:
- $\Delta > 0$: increased metric value after processing.
- $\Delta < 0$: decreased metric value after processing.

Plain English:
- showing input, output, and delta side-by-side prevents ambiguity when interpreting quality diagnostics.
