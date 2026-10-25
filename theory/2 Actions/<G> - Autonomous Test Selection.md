# <G> - Autonomous Test Selection: Value of Information & Greedy Ranking

This document derives the clinician's **autonomous test-ordering rule** $F_{\mathcal{H}}^*(s; g)$ at graduation. 

We prove that under Gaussian biomarker posteriors, maximizing the Bayesian Value of Information (VoI) is strictly equivalent to maximizing the variance of the perceived utility difference, yielding an exact, symptom-invariant **greedy ranking rule** based on subjective feature salience $g_j^2 \sigma_j^2$.

---

## 1. Decision Variable Under Partial Testing

Suppose the clinician selects test panel $F \subseteq [n]$ with $|F| \le c$.
The lab unveils true values $x_F$. For unmeasured biomarkers $j \notin F$, the clinician imputes the conditional expectation $\mu_j(s) = \mathbb{E}[x_j \mid s]$.

The clinician's subjective utility difference is:
$$V(F, x_F, s; g) = \sum_{j \in F} g_j x_j + \sum_{j \notin F} g_j \mu_j(s) = g^\top \mu(s) + g_F^\top \big(x_F - \mu_F(s)\big)$$

Conditioned on triage symptoms $s$, $x_F - \mu_F(s) \sim \mathcal{N}(0, \Gamma_{FF})$.
Therefore, $V(F, x_F, s; g)$ is a scalar Gaussian random variable:
$$V(F) \sim \mathcal{N}\big(\mu_V(s), \; \sigma_V^2(F)\big)$$
where:
* $\mu_V(s) := g^\top \mu(s) \in \mathbb{R}$ (the prior expected decision advantage given symptoms alone)
* $\sigma_V^2(F) := g_F^\top \Gamma_{FF} g_F \ge 0$ (the variance resolved by running test panel $F$)

The clinician predicts Action 1 if and only if $V(F) > 0$.

---

## 2. Derivation of the Autonomous Selection Rule

Before ordering tests, the clinician evaluates candidate panels $F$ by their perceived **Value of Information (VoI)**:
$$\text{VoI}(F \mid s) := \mathbb{E}_{x_F \mid s} \left[ \max\big(\hat{u}_1(F), \hat{u}_2(F)\big) \right] - \max\left( \mathbb{E}[\hat{u}_1 \mid s], \; \mathbb{E}[\hat{u}_2 \mid s] \right)$$

Using the elementary identity $\max(a, b) = \frac{a+b}{2} + \frac{|a-b|}{2}$, this simplifies to:
$$\text{VoI}(F \mid s) = \frac{1}{2} \left( \mathbb{E}_{x_F \mid s}\big[|V(F)|\big] - |\mu_V(s)| \right)$$

### Lemma: Strict Monotonicity of Folded Normal Expectation
For any Gaussian variable $Z \sim \mathcal{N}(\mu, \sigma^2)$, the expected absolute value is given analytically by:
$$\mathbb{E}[|Z|] = 2\sigma \, \phi\left(\frac{\mu}{\sigma}\right) + \mu \left( 2\Phi\left(\frac{\mu}{\sigma}\right) - 1 \right)$$
where $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$ is the standard normal PDF and $\Phi(z)$ is the standard normal CDF.
Furthermore, for any fixed $\mu \in \mathbb{R}$, $\mathbb{E}[|Z|]$ is strictly increasing in $\sigma > 0$:
$$\frac{\partial}{\partial \sigma} \mathbb{E}[|Z|] = 2 \, \phi\left(\frac{\mu}{\sigma}\right) > 0$$

### Proof:
Differentiating with respect to $\sigma$:
$$\frac{\partial}{\partial \sigma}\left[ 2\sigma \phi\left(\frac{\mu}{\sigma}\right) + \mu\left(2\Phi\left(\frac{\mu}{\sigma}\right) - 1\right) \right] = 2\phi\left(\frac{\mu}{\sigma}\right) + 2\sigma \phi'\left(\frac{\mu}{\sigma}\right)\left(-\frac{\mu}{\sigma^2}\right) + 2\mu \phi\left(\frac{\mu}{\sigma}\right)\left(-\frac{\mu}{\sigma^2}\right)$$
Since $\phi'(z) = -z \phi(z)$, the second term is $2\sigma \left(-\frac{\mu}{\sigma}\phi\left(\frac{\mu}{\sigma}\right)\right)\left(-\frac{\mu}{\sigma^2}\right) = \frac{2\mu^2}{\sigma^2}\phi\left(\frac{\mu}{\sigma}\right)$.
The third term is $-\frac{2\mu^2}{\sigma^2}\phi\left(\frac{\mu}{\sigma}\right)$.
These two terms cancel exactly, leaving:
$$\frac{\partial}{\partial \sigma} \mathbb{E}[|Z|] = 2\phi\left(\frac{\mu}{\sigma}\right) > 0 \quad \forall \sigma > 0 \quad \blacksquare$$

---

## 3. The Autonomous Test-Ordering Theorem

> [!NOTE] Theorem: Autonomous Selection is Symptom-Invariant Quadratic Maximization
> Let residual covariance $\Gamma$ be invariant to symptoms $s$. Then for any symptom presentation $s$, the clinician's autonomous test panel choice is uniquely given by:
> $$\hat{F}(g) = \arg\max_{|F| \le c} g_F^\top \Gamma_{FF} g_F$$
> In particular, when residual uncertainties are independent ($\Gamma = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)$), the objective decouples additively:
> $$g_F^\top \Gamma_{FF} g_F = \sum_{j \in F} g_j^2 \sigma_j^2$$
> and the clinician greedily orders the $c$ biomarkers with the largest values of subjective salience $S_j(g) := g_j^2 \sigma_j^2$.

### Proof:
By the Lemma, $\text{VoI}(F \mid s) = \frac{1}{2}(\mathbb{E}[|V(F)|] - |\mu_V(s)|)$ is strictly increasing in $\sigma_V(F) = \sqrt{g_F^\top \Gamma_{FF} g_F}$. Since $\mu_V(s)$ does not depend on $F$, maximizing VoI is equivalent to maximizing $\sigma_V^2(F) = g_F^\top \Gamma_{FF} g_F$. When $\Gamma = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)$, this is $\sum_{j \in F} g_j^2 \sigma_j^2$, which is maximized by selecting the $c$ largest components. $\blacksquare$

---

## 4. The Diagnostic Crowding-Out Phenomenon

This theorem exposes the precise cognitive vulnerability of an uncalibrated clinician:

> [!WARNING] The Crowding-Out Mechanism
> The clinician orders biomarker $j$ solely based on **perceived magnitude squared $g_j^2$**, completely blind to whether the belief sign is correct ($\operatorname{sgn}(g_j) = \operatorname{sgn}(d_j)$) or whether the biomarker is a complete distraction ($d_j = 0$).
>
> If an irrelevant biomarker has initial misconception $|g_j^{(0)}| > \max_{k \notin \hat{F}} |g_k|$, it will **displace** a genuinely life-saving biomarker from the test panel $\hat{F}$, stealing one of the doctor's scarce $c$ test slots!
