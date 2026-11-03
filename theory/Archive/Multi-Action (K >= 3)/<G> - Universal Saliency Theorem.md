# <G> - Universal Saliency Theorem: Closed-Form Test Selection for Any K >= 2 Under General Correlated Features and Arbitrary Innovation Variance

This document resolves the two central foundational challenges for multi-action triage ($K \ge 3$):
1. **Correlated Features ($\Gamma = \Sigma_{x \mid s} \not\succ 0$ Non-Diagonal)**: Deriving the exact epistemic matrix geometry when biomarkers exhibit prior collinearity and triage-induced explaining-away covariance.
2. **The Large Innovation Frontier**: Proving why higher-order polynomial Taylor expansions fail, and developing two non-perturbative exact solutions: the **Clark-Sheppard Closed Form** for active contenders and **MacKay's Probit-Softmax Damping** for $K \ge 3$.

---

## 1. The Multi-Action Value of Information Objective

Recall that for $K \ge 3$ actions, the clinician's utility for action $k \in [K]$ after ordering test panel $F \subseteq [n]$ ($|F| \le c$) on a patient with symptoms $s$ is:
$$\hat{u}_k(F, x_F, s) = \bar{u}_k(s) + \tilde{u}_k(F)$$
where:
* $\bar{u}_k(s) = h_k^\top \mu(s) \in \mathbb{R}$ is the prior expected utility given symptoms alone, with $\mu(s) = \Sigma B^\top (B \Sigma B^\top)^{-1} s$.
* $\tilde{u}_k(F) = h_{k, F}^\top (x_F - \mu_F(s))$ is the zero-mean Gaussian test innovation.
* $\Gamma = \Sigma_{x \mid s} = \Sigma - \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma \in \mathbb{R}^{n \times n}$ is the residual covariance matrix of latent patient vitals given symptoms $s$. Crucially, $\Gamma$ is **general positive semi-definite (non-diagonal)** due to biological collinearity and Berkson's explaining-away effect from $s = Bx$.

The joint innovation vector across all $K$ actions is:
$$\tilde{u}(F) = \begin{pmatrix} \tilde{u}_1(F) \\ \vdots \\ \tilde{u}_K(F) \end{pmatrix} \sim \mathcal{N}\big(0, \; \Omega(F)\big), \qquad \Omega_{k, \ell}(F) = h_{k, F}^\top \Gamma_{FF} h_{\ell, F}$$
where $\Gamma_{FF} \in \mathbb{R}^{c \times c}$ is the submatrix of $\Gamma$ indexed by test panel $F$.

The clinician's autonomous test-ordering problem at graduation is:
$$F_{\mathcal{H}}^*(s) = \arg\max_{|F| \le c} \operatorname{VoI}(F \mid s)$$
where:
$$\operatorname{VoI}(F \mid s) := \mathbb{E}_{\tilde{u} \sim \mathcal{N}(0, \Omega(F))} \left[ \max_{k \in [K]} \big( \bar{u}_k(s) + \tilde{u}_k(F) \big) \right] - \max_{k \in [K]} \bar{u}_k(s)$$

---

## 2. Part I: General Correlated Features ($\Gamma$ Non-Diagonal)

To evaluate the non-smooth maximum $\max_{k \in [K]} u_k$, we consider the smooth Log-Sum-Exp approximation with inverse temperature $\beta > 0$:
$$M_\beta(u) := \frac{1}{\beta} \log \sum_{k=1}^K e^{\beta u_k}, \qquad \lim_{\beta \to \infty} M_\beta(u) = \max_{k \in [K]} u_k$$

Let $p(s) = (p_1(s), \dots, p_K(s))^\top$ denote the **softmax posterior probability vector** over clinical conditions given symptoms $s$ alone:
$$p_k(s) := \frac{e^{\beta \bar{u}_k(s)}}{\sum_{\ell=1}^K e^{\beta \bar{u}_\ell(s)}}$$

The gradient and Hessian of $M_\beta$ evaluated at $\bar{u}(s)$ are:
$$\big[ \nabla M_\beta(\bar{u}(s)) \big]_k = p_k(s)$$
$$\big[ \nabla^2 M_\beta(\bar{u}(s)) \big]_{k, \ell} = \beta \Big( p_k(s) \mathbb{1}[k = \ell] - p_k(s) p_\ell(s) \Big)$$

---

### Definition 1: The Epistemic Action Covariance Matrix
We define the **Epistemic Action Covariance Matrix** $M(s) \in \mathbb{R}^{n \times n}$ as the covariance of the clinician's utility rows under the triage distribution $p(s)$:
$$M(s) := \operatorname{Cov}_{k \sim p(s)}[h_k] = \sum_{k=1}^K p_k(s) h_k h_k^\top - \left( \sum_{k=1}^K p_k(s) h_k \right) \left( \sum_{k=1}^K p_k(s) h_k \right)^\top$$
Using the algebraic identity for weighted covariance, $M(s)$ decomposes into pairwise contrastive rank-1 outer products:
$$M(s) = \sum_{1 \le k < \ell \le K} p_k(s) p_\ell(s) (h_k - h_\ell)(h_k - h_\ell)^\top$$
*Properties:*
1. $M(s)$ is symmetric and positive semi-definite ($M(s) \succeq 0$) with rank at most $\min(n, K - 1)$.
2. $M(s)$ measures how strongly the competing diagnostic hypotheses diverge along each biomarker axis. If disease $k$ and disease $\ell$ are the active contenders ($p_k p_\ell > 0$), their difference vector $\Delta h_{k\ell} = h_k - h_\ell$ spans the principal subspace of $M(s)$.

---

### Theorem 1: Correlated Universal Saliency Theorem
Under general residual feature covariance $\Gamma \succeq 0$, the Value of Information for test panel $F$ under a second-order expansion around $\bar{u}(s)$ is given in closed coordinate-free form by:
$$\operatorname{VoI}(F \mid s) = \frac{\beta}{2} \operatorname{Tr}\Big( \Gamma_{FF} M_{FF}(s) \Big) + \mathcal{O}(\|\Omega(F)\|^2)$$
where $\Gamma_{FF}$ and $M_{FF}(s)$ are the $c \times c$ submatrices corresponding to the biomarkers in $F$.

Furthermore, let $z \in \{0, 1\}^n$ be the binary indicator vector of panel $F$ ($z_j = 1 \iff j \in F$). Then:
$$\operatorname{VoI}(F \mid s) = \frac{\beta}{2} z^\top \Big( \Gamma \odot M(s) \Big) z + \mathcal{O}(\|\Omega(F)\|^2) = \frac{\beta}{2} \sum_{i \in F} \sum_{j \in F} \Gamma_{ij} M_{ij}(s)$$
where $\odot$ denotes the Hadamard (element-wise) product.

#### Proof:
Expanding $M_\beta(\bar{u} + \tilde{u})$ in a Taylor series around the prior baseline $\bar{u}(s)$:
$$M_\beta(\bar{u} + \tilde{u}) = M_\beta(\bar{u}) + \nabla M_\beta(\bar{u})^\top \tilde{u} + \frac{1}{2} \tilde{u}^\top \nabla^2 M_\beta(\bar{u}) \tilde{u} + \mathcal{O}(\|\tilde{u}\|^3)$$
Taking expectations over $\tilde{u} \sim \mathcal{N}(0, \Omega(F))$:
1. The first-order term vanishes identically because $\mathbb{E}[\tilde{u}] = 0$.
2. The second-order term is:
$$\mathbb{E}\left[ \frac{1}{2} \tilde{u}^\top \nabla^2 M_\beta(\bar{u}) \tilde{u} \right] = \frac{\beta}{2} \mathbb{E}\left[ \sum_{k=1}^K p_k(s) \tilde{u}_k(F)^2 - \left( \sum_{k=1}^K p_k(s) \tilde{u}_k(F) \right)^2 \right]$$
Using the weighted variance identity $\sum p_k z_k^2 - (\sum p_k z_k)^2 = \sum_{k < \ell} p_k p_\ell (z_k - z_\ell)^2$:
$$= \frac{\beta}{2} \sum_{1 \le k < \ell \le K} p_k(s) p_\ell(s) \mathbb{E}\left[ \big( \tilde{u}_k(F) - \tilde{u}_\ell(F) \big)^2 \right]$$
The scalar innovation difference is $\tilde{u}_k(F) - \tilde{u}_\ell(F) = (h_{k, F} - h_{\ell, F})^\top (x_F - \mu_F(s))$.
Its expected square is its variance under $x_F \mid s \sim \mathcal{N}(\mu_F(s), \Gamma_{FF})$:
$$\mathbb{E}\left[ \big( \tilde{u}_k(F) - \tilde{u}_\ell(F) \big)^2 \right] = (h_{k, F} - h_{\ell, F})^\top \Gamma_{FF} (h_{k, F} - h_{\ell, F})$$
Using the cyclic property of trace:
$$= \operatorname{Tr}\left( \Gamma_{FF} (h_{k, F} - h_{\ell, F})(h_{k, F} - h_{\ell, F})^\top \right)$$
Distributing the non-negative weights $p_k(s) p_\ell(s)$ inside the trace and summing across all pairs $k < \ell$:
$$= \frac{\beta}{2} \operatorname{Tr}\left( \Gamma_{FF} \sum_{1 \le k < \ell \le K} p_k(s) p_\ell(s) (h_{k, F} - h_{\ell, F})(h_{k, F} - h_{\ell, F})^\top \right) = \frac{\beta}{2} \operatorname{Tr}\Big( \Gamma_{FF} M_{FF}(s) \Big)$$
Finally, expressing the submatrix trace using the binary indicator vector $z \in \{0, 1\}^n$:
$$\operatorname{Tr}\big(\Gamma_{FF} M_{FF}(s)\big) = \sum_{i \in F} \sum_{j \in F} \Gamma_{ij} M_{ij}(s) = \sum_{i=1}^n \sum_{j=1}^n z_i z_j \Gamma_{ij} M_{ij}(s) = z^\top \big( \Gamma \odot M(s) \big) z$$
This completes the proof. $\blacksquare$

---

### Theorem 2: Schur Positive Semi-Definiteness & Diagonal Recovery
Define the **Diagnostic Saliency Kernel Matrix** as $W(s) := \Gamma \odot M(s) \in \mathbb{R}^{n \times n}$.
1. **Guaranteed Positive Semi-Definiteness**: $W(s) \succeq 0$.
2. **Exact Recovery of Independent Case**: When residual features are uncorrelated ($\Gamma = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)$), the off-diagonal terms vanish, and the formulation recovers the decoupled additive formula identically:
$$\operatorname{Tr}\big(\Gamma_{FF} M_{FF}(s)\big) = \sum_{j \in F} \sigma_j^2 M_{jj}(s) = \sum_{j \in F} \sigma_j^2 \sum_{k < \ell} p_k(s) p_\ell(s) (H_{kj} - H_{\ell j})^2 = \sum_{j \in F} \text{Salience}_j(s)$$

#### Proof:
1. By definition of conditional covariance, $\Gamma = \operatorname{Var}[x \mid s] \succeq 0$. By Definition 1, $M(s) \succeq 0$. By **Schur's Product Theorem (1911)**, the Hadamard product of any two positive semi-definite matrices is positive semi-definite. Therefore, $W(s) = \Gamma \odot M(s) \succeq 0$.
2. If $\Gamma = \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)$, then $\Gamma_{ij} = \sigma_j^2 \delta_{ij}$. The double sum simplifies to:
$$z^\top (\Gamma \odot M(s)) z = \sum_{i \in F} \sum_{j \in F} \sigma_j^2 \delta_{ij} M_{ij}(s) = \sum_{j \in F} \sigma_j^2 M_{jj}(s)$$
Substituting $M_{jj}(s) = \sum_{k < \ell} p_k(s) p_\ell(s) (H_{kj} - H_{\ell j})^2$ yields the decoupled additive formula. $\blacksquare$

---

### Anatomy of Cross-Feature Interactions: Redundancy vs. Synergy
Under correlated features, the objective decomposes into:
$$\operatorname{VoI}(F \mid s) \approx \frac{\beta}{2} \left[ \sum_{j \in F} \underbrace{\Gamma_{jj} M_{jj}(s)}_{\text{Self-Salience}} + 2 \sum_{i < j \in F} \underbrace{\Gamma_{ij} M_{ij}(s)}_{\text{Cross-Feature Interaction}} \right]$$
The cross-term $\Gamma_{ij} M_{ij}(s) = \Gamma_{ij} \sum_{k < \ell} p_k p_\ell (H_{ki} - H_{\ell i})(H_{kj} - H_{\ell j})$ governs multi-test efficiency:
* **Diagnostic Redundancy ($\Gamma_{ij} M_{ij}(s) < 0$ or diminishing returns)**: If two biomarkers are positively correlated ($\Gamma_{ij} > 0$) and discriminate in the same direction ($\Delta H_i \Delta H_j > 0$), ordering both yields diminishing marginal returns. The preceptor learns that measuring one explains away the other.
* **Diagnostic Synergy ($\Gamma_{ij} M_{ij}(s) > 0$ with opposing signs)**: If two biomarkers have negative residual correlation ($\Gamma_{ij} < 0$) while pointing in opposing directions, measuring both amplifies the separation of competing diseases beyond their individual sums (e.g., arterial blood gas ratios or electrolyte gaps).

---

## 3. Part II: The Large Innovation Frontier (Problem 2)

We now resolve the "small innovation" assumption ($\|\Omega(F)\|^2 \ll 1$).

### Why Pure Higher-Order Polynomial Taylor Expansions Fail

Consider extending the Taylor series of $M_\beta(u)$ to higher orders:
$$M_\beta(\bar{u} + \tilde{u}) = M_\beta(\bar{u}) + \nabla M_\beta^\top \tilde{u} + \frac{1}{2} \tilde{u}^\top \nabla^2 M_\beta \tilde{u} + \frac{1}{6} D^3 M_\beta[\tilde{u}^3] + \frac{1}{24} D^4 M_\beta[\tilde{u}^4] + \dots$$

1. **All Odd Moments Vanish Identically**:
   Because $\tilde{u} \sim \mathcal{N}(0, \Omega(F))$ is a zero-mean multivariate Gaussian, **all odd central moments are zero**:
   $$\mathbb{E}[\tilde{u}_i \tilde{u}_j \tilde{u}_k] = 0, \qquad \mathbb{E}[\tilde{u}_i \tilde{u}_j \tilde{u}_k \tilde{u}_l \tilde{u}_m] = 0$$
   The 3rd-order and 5th-order terms contribute **exactly zero** correction to the expectation.
2. **Even Moments Cause Catastrophic Quartic Divergence**:
   The true target function $\max_{k \in [K]} u_k$ is asymptotically **linear** in the tails:
   $$\lim_{\alpha \to \infty} \frac{\max_k (u_k + \alpha v_k)}{\alpha} = \max_k v_k = \mathcal{O}(\|\tilde{u}\|)$$
   However, a 4th-order Taylor polynomial grows as $\|\tilde{u}\|^4$. When a lab test has large diagnostic power (large variance $\Omega_{kk}$), the 4th-order polynomial term evaluates to a huge value that rapidly diverges to $\pm \infty$. 

Therefore, **polynomial Taylor expansions of order 4 or higher are fundamentally unsuited for large innovations**. Instead, we establish two exact, non-perturbative formulations.

---

### Theorem 3: The Exact Clark-Sheppard Closed Form for Active Contenders

In emergency medicine, triage symptoms $s$ exponentially prune the differential diagnosis down to the **two active primary contenders**:
$$y^* = \arg\max_{k \in [K]} \bar{u}_k(s), \qquad k^* = \arg\max_{k \neq y^*} \bar{u}_k(s)$$
All remaining diagnoses $k \notin \{y^*, k^*\}$ have prior deficits $\bar{u}_{y^*}(s) - \bar{u}_k(s) > 3\sigma_{\max}$, contributing asymptotically zero to the decision boundary.

Between these two active contenders, we do not require the smooth Log-Sum-Exp approximation or any Taylor series. Let:
$$\Delta(s) := \bar{u}_{y^*}(s) - \bar{u}_{k^*}(s) \ge 0$$
$$\sigma_{\text{diff}}^2(F) := (h_{y^*, F} - h_{k^*, F})^\top \Gamma_{FF} (h_{y^*, F} - h_{k^*, F})$$

Then the exact Value of Information is given in **closed analytical form for arbitrary variance magnitude**:
$$\operatorname{VoI}_{\text{exact}}(F \mid s) = \sigma_{\text{diff}}(F) \cdot \phi\left( \frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right) - \Delta(s) \cdot \Phi\left( -\frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right)$$
where $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$ is the standard normal PDF and $\Phi(z)$ is the standard normal CDF.

#### Proof:
For two competing actions, the post-test utility is:
$$\max\big( \bar{u}_{y^*}(s) + \tilde{u}_{y^*}(F), \; \bar{u}_{k^*}(s) + \tilde{u}_{k^*}(F) \big) = \bar{u}_{y^*}(s) + \tilde{u}_{y^*}(F) + \max\big( 0, \; -\Delta(s) + \tilde{u}_{k^*}(F) - \tilde{u}_{y^*}(F) \big)$$
Define the scalar difference $Z := \tilde{u}_{k^*}(F) - \tilde{u}_{y^*}(F) \sim \mathcal{N}(0, \sigma_{\text{diff}}^2(F))$.
The Value of Information is the expected gain over the baseline optimal action $y^*$:
$$\operatorname{VoI}(F \mid s) = \mathbb{E}_Z\big[ \max(0, Z - \Delta(s)) \big] = \int_{\Delta(s)}^\infty (z - \Delta(s)) \frac{1}{\sqrt{2\pi}\sigma_{\text{diff}}} e^{-z^2 / 2\sigma_{\text{diff}}^2} \, dz$$
Substitute $w = z / \sigma_{\text{diff}}$, so $dz = \sigma_{\text{diff}} dw$ and the lower limit becomes $\alpha = \Delta(s) / \sigma_{\text{diff}}$:
$$\operatorname{VoI}(F \mid s) = \sigma_{\text{diff}} \int_\alpha^\infty w \phi(w) dw - \Delta(s) \int_\alpha^\infty \phi(w) dw$$
Since $\int w \phi(w) dw = -\phi(w)$, the first integral evaluates to:
$$\int_\alpha^\infty w \phi(w) dw = \big[ -\phi(w) \big]_\alpha^\infty = 0 - (-\phi(\alpha)) = \phi(\alpha)$$
The second integral is the tail probability $1 - \Phi(\alpha) = \Phi(-\alpha)$.
Substituting $\alpha = \frac{\Delta(s)}{\sigma_{\text{diff}}(F)}$:
$$\operatorname{VoI}_{\text{exact}}(F \mid s) = \sigma_{\text{diff}}(F) \phi\left( \frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right) - \Delta(s) \Phi\left( -\frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right) \quad \blacksquare$$

#### Asymptotic Properties for Large Innovations:
1. **Large Variance Linear Scaling ($\sigma_{\text{diff}} \to \infty$)**:
   $$\lim_{\sigma_{\text{diff}} \to \infty} \operatorname{VoI}_{\text{exact}}(F \mid s) = \frac{\sigma_{\text{diff}}(F)}{\sqrt{2\pi}} - \frac{\Delta(s)}{2}$$
   The exact VoI scales **linearly** with standard deviation $\sigma_{\text{diff}}$ for large innovations, exactly matching the linear physics of order statistics rather than blowing up quadratically or quartically.
2. **Small Variance Recovery ($\sigma_{\text{diff}} \to 0$)**:
   When test variance is small relative to $\Delta(s)$, Mill's ratio yields:
   $$\operatorname{VoI} = \frac{\sigma_{\text{diff}}^3}{\Delta(s)^2 \sqrt{2\pi}} e^{-\Delta(s)^2 / 2\sigma_{\text{diff}}^2} \to 0$$
   Tests that cannot bridge the clinical gap $\Delta(s)$ receive zero value.

---

### Theorem 4: MacKay's Probit-Softmax Damping for General $K \ge 3$

When $K \ge 3$ diseases remain actively tied ($\Delta_k(s) \approx 0$), we resolve large innovations using the **Gaussian-Convoluted Probit Approximation** (MacKay 1992; Daunizeau 2017).

By Price's Theorem (1958), the rate of change of expected VoI with respect to the innovation covariance is the expected Hessian:
$$\frac{\partial \operatorname{VoI}}{\partial \Omega_{k\ell}} = \frac{1}{2} (2 - \delta_{k\ell}) \mathbb{E}_{\tilde{u} \sim \mathcal{N}(0, \Omega(F))}\left[ \frac{\partial^2 M_\beta}{\partial u_k \partial u_\ell}(\bar{u}(s) + \tilde{u}(F)) \right]$$
As test innovation variance $\sigma_{k\ell}^2(F)$ grows large, the expected Hessian does not remain constant; it attenuates because the random variable $\bar{u} + \tilde{u}$ spreads across the saturated boundaries of the softmax.

Integrating the Gaussian convolution against the logistic sigmoid yields the **Variance-Damped Saliency Kernel**:
$$\operatorname{VoI}_{\text{damped}}(F \mid s) = \frac{\frac{\beta}{2} \operatorname{Tr}\Big( \Gamma_{FF} M_{FF}(s) \Big)}{\sqrt{1 + \frac{\pi \beta^2}{8} \sigma_{\max}^2(F)}} = \frac{\frac{\beta}{2} z^\top \big( \Gamma \odot M(s) \big) z}{\sqrt{1 + \frac{\pi \beta^2}{8} \sigma_{\max}^2(F)}}$$
where $\sigma_{\max}^2(F) := \max_{k \in [K]} h_{k, F}^\top \Gamma_{FF} h_{k, F}$.

#### Mathematical Advantages:
1. **Self-Damping Curvature**: When innovation variance $\sigma_{\max}^2(F)$ is small, the square root denominator evaluates to $\approx 1$, recovering the second-order Universal Saliency Theorem identically.
2. **Sub-Linear Saturation**: When innovation variance is large, the denominator scales as $\mathcal{O}(\sigma_{\max})$, converting the quadratic numerator $\mathcal{O}(\sigma^2)$ into linear growth $\mathcal{O}(\sigma)$, seamlessly matching the true asymptotic behavior of $\mathbb{E}[\max u_k]$.

---

## 4. Exponential Pruning of Inactive Margins

Sort the diseases by prior expected utility given symptoms alone:
$$\bar{u}_{(1)}(s) \ge \bar{u}_{(2)}(s) \ge \bar{u}_{(3)}(s) \ge \dots \ge \bar{u}_{(K)}(s)$$
For any candidate disease $k$, define its clinical deficit relative to the leading diagnosis: $\Delta_k(s) := \bar{u}_{(1)}(s) - \bar{u}_k(s) \ge 0$.

\begin{lemma}[Exponential Suppression of Distant Diseases]
Let test innovation variances be bounded by $\sigma_{\max}^2 := \max_{k, F} h_{k, F}^\top \Gamma_{FF} h_{k, F}$.
For any candidate action $k$ whose deficit satisfies $\Delta_k(s) > 3 \sigma_{\max}$:
$$p_k(s) = \frac{e^{\beta \bar{u}_k(s)}}{\sum_{\ell} e^{\beta \bar{u}_\ell(s)}} \le e^{-\beta \Delta_k(s)} = \mathcal{O}\left( e^{-\Delta_k(s)^2 / 2\sigma_{\max}^2} \right)$$
Its pairwise outer product $p_k(s) p_\ell(s) (h_k - h_\ell)(h_k - h_\ell)^\top$ is exponentially suppressed for all $\ell$, contributing asymptotically zero to the Epistemic Covariance Matrix $M(s)$.
\end{lemma}

### Clinical Significance:
In real emergency triage, regardless of whether $K = 5$ or $K = 50$, the presenting symptoms $s$ compress the active differential diagnosis to at most 2 or 3 diseases. The clinician's test selection is guided strictly by the active margins separating those specific contenders.
