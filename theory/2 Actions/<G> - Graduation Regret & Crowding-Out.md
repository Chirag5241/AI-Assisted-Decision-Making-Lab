# <G> - Graduation Regret & The Crowding-Out Law

This document derives the exact closed-form expression for **expected graduation regret** under autonomous test selection and proves the **Crowding-Out Law**: how uncalibrated distractions catastrophically corrupt autonomous clinical decisions.

---

## 1. Closed-Form Expected Graduation Regret

At graduation ($t > T$), the clinician observes presenting symptoms $s$, autonomously selects test panel $\hat{F} = \hat{F}(g) \subseteq [n]$ ($|\hat{F}| \le c$), observes $x_{\hat{F}}$, and diagnoses Action 1 iff:
$$W := g_{\hat{F}}^\top x_{\hat{F}} + g_{\hat{F}^c}^\top \mu_{\hat{F}^c}(s) > 0$$

The true optimal diagnosis is Action 1 iff the ground-truth advantage $D := d^\top x > 0$.
The clinical regret incurred is:
$$\ell_{\text{grad}}(x, s; g) = |D| \cdot \mathbb{1}\big[\operatorname{sgn}(W) \neq \operatorname{sgn}(D)\big]$$

### Theorem: Exact Graduation Regret via Sheppard's Arcsine Law
In the canonical standardized setting ($x \sim \mathcal{N}(0, I_n)$, independent test residual uncertainties):
1. The pair $(D, W)$ is jointly zero-mean Gaussian with covariance $\operatorname{Cov}(D, W) = d_{\hat{F}}^\top g_{\hat{F}}$, variance $\sigma_D^2 = \|d\|_2^2$, and variance $\sigma_W^2 = \|g_{\hat{F}}\|_2^2$.
2. The Pearson correlation coefficient is:
   $$\rho(\hat{F}(g), g) := \frac{\operatorname{Cov}(D, W)}{\sigma_D \sigma_W} = \frac{\sum_{j \in \hat{F}(g)} d_j g_j}{\|d\|_2 \sqrt{\sum_{j \in \hat{F}(g)} g_j^2}}$$
3. The exact expected graduation regret is:
   $$\mathcal{L}_{\text{grad}}(g) := \mathbb{E}_{x, s}[\ell_{\text{grad}}(x, s; g)] = \frac{\|d\|_2}{\sqrt{2\pi}} \left( 1 - \rho(\hat{F}(g), g) \right)$$

### Proof:
Define standardized variables $U := D / \|d\|_2 \sim \mathcal{N}(0, 1)$ and $V := W / \|g_{\hat{F}}\|_2 \sim \mathcal{N}(0, 1)$ with correlation $\rho = \operatorname{Cov}(U, V) = \frac{d_{\hat{F}}^\top g_{\hat{F}}}{\|d\|_2 \|g_{\hat{F}}\|_2}$.
Then:
$$\ell_{\text{grad}} = \|d\|_2 |U| \mathbb{1}[U V < 0]$$
By Sheppard's classical bivariate normal dichotomy theorem:
$$\mathbb{E}[|U| \mathbb{1}[UV < 0]] = \int_{-\infty}^0 \int_0^\infty u \cdot f_{U,V}(u, v; \rho) \, du \, dv = \frac{1 - \rho}{\sqrt{2\pi}}$$
Multiplying by $\|d\|_2$ completes the proof. $\blacksquare$

---

## 2. The Anatomy of Clinical Correlation

The correlation ratio $\rho(\hat{F}(g), g)$ governs everything. Notice its mathematical structure:
$$\rho(\hat{F}(g), g) = \frac{\sum_{j \in \hat{F}(g)} d_j g_j}{\|d\|_2 \sqrt{\sum_{j \in \hat{F}(g)} g_j^2}}$$
where $\hat{F}(g) = \arg\max_{|F| \le c} \sum_{j \in F} g_j^2$.

This exposes the **two fundamental failure modes** of an autonomous doctor:

### Failure Mode 1: The Reverse-Inference Catastrophe (Wrong Signs)
If an ordered biomarker $j \in \hat{F}(g)$ has inverted sign ($g_j d_j < 0$), its contribution to the numerator is **negative**:
$$d_j g_j = - |d_j| |g_j| < 0$$
This actively pulls $\rho$ toward $-1$, causing the doctor to systematically prescribe the contraindicated treatment.

### Failure Mode 2: The Crowding-Out Catastrophe (Loud Distractions)
Suppose biomarker $k$ is physiologically irrelevant ($d_k = 0$), but the clinician starts with a massive misconception ($|g_k^{(0)}| = M \gg 1$).
* **Slot Theft:** Because $g_k^2 \approx M^2$, biomarker $k$ enters the top-$c$ set $\hat{F}(g)$, displacing a genuinely predictive biomarker $j^*$ ($d_{j^*}^2 > 0$).
* **Numerator Contribution:** Biomarker $k$ contributes exactly **zero** to the numerator ($d_k g_k = 0$).
* **Noise Inflation:** Biomarker $k$ adds $g_k^2$ to the denominator $\sqrt{\sum g_j^2}$, diluting the signal of all other legitimate tests in the panel!

---

## 3. The Crowding-Out Law

> [!IMPORTANT] Theorem: The Crowding-Out Inequality
> Let $j^* \in [n]$ be a decisive biomarker with $d_{j^*} > 0$ and correctly calibrated belief $g_{j^*} = d_{j^*}$. 
> Let $k \in [n]$ be a loud distraction with $d_k = 0$ and uncalibrated belief $|g_k| = M$.
> Suppose $c = 1$. Then:
> 1. If $M > d_{j^*}$, the clinician autonomously orders test $k$: $\hat{F} = \{k\}$.
> 2. The resulting correlation is:
>    $$\rho(\{k\}, g) = \frac{d_k g_k}{\|d\|_2 |g_k|} = 0$$
> 3. The graduation regret is:
>    $$\mathcal{L}_{\text{grad}} = \frac{\|d\|_2}{\sqrt{2\pi}} \approx 0.3989 \|d\|_2$$
> 4. If test $k$ were extinguished ($M < d_{j^*}$), the clinician would autonomously order $j^*$, achieving correlation $\rho = \frac{d_{j^*}}{\|d\|_2} > 0$ and strictly lower regret:
>    $$\mathcal{L}_{\text{grad}} = \frac{\|d\|_2 - d_{j^*}}{\sqrt{2\pi}} < \frac{\|d\|_2}{\sqrt{2\pi}}$$

### Crucial Pedagogical Consequence:
The AI preceptor cannot simply teach the clinician good biomarkers. **The AI must actively spend training budget extinguishing loud distractions**, because a single loud distraction will hijack the clinician's autonomous test panel and destroy clinical accuracy at graduation!
