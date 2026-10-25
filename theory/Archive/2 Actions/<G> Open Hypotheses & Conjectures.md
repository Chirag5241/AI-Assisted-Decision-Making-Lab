# <G> Open Hypotheses & Conjectures

This note collects the primary unresolved theoretical conjectures, computational questions, and potential research directions for the project.

---

## 1. Ex-Post Value Monotonicity

* **Conjecture**: For general non-increasing learning functions $\psi(m)$, the optimal ex-post discounted regret $V(m)$ is monotonically non-increasing in each coordinate:
  $$V(m + e_i) \le V(m) \quad \forall i \in [n], \forall m \in \mathbb{N}^n$$
* **Current Status**:
  * **Proved** for the discrete flip transition $m_i \to \tau_i$ (where $\operatorname{sgn}(g_i)$ flips from $-\operatorname{sgn}(d_i)$ to $+\operatorname{sgn}(d_i)$) via conditional Gaussian coupling ([[<G> Ex-Post Opportunistic Teaching (c = 1)]]).
  * Verified numerically across $>10^5$ value iteration states with zero violations.
  * Open for general continuous contraction increments when $c > 1$.

---

## 2. Tightness of the Chaperone Transient Bound

* **Conjecture**: Under one-shot learning ($\psi(m) = \mathbf{1}[m=0]$) and ex-ante feature budget $c \ge 2$, the optimal transient length satisfies:
  $$L \le \min\{c, 2\}$$
* **Current Status**:
  * **Proved**: $L \le c$ ([[<G> Multi-Feature Chaperoning (c >= 2)]]).
  * Evaluated across 1,500 random instances with $c \in \{2, 3\}$: transient length was 0, 1, or 2, and **never reached 3**.
  * Open: Can an adversarial instance with $c=3$ force a 3-step domino chaperone sequence, or does $L \le 2$ hold universally?

---

## 3. Price of Stationarity under Geometric Learning

* **Question**: For geometric learning $\psi(m) = \alpha^{-m}$ with $K=2$ and independent features, how much worse can the best stationary policy perform compared to the optimal dynamic schedule?
  $$\text{PoS} := \frac{\inf_{\pi \in \Pi_{\text{stat}}} J(\delta, \pi)}{\inf_{\pi \in \Pi} J(\delta, \pi)}$$
* **Current Status**:
  * In Noti et al. (EC 2025), $\text{PoS} = 1$ under MSE loss.
  * Under decision regret, numerical adversarial search found instances with $\text{PoS} > 17\times$.
  * Open: Is $\text{PoS}$ bounded by a constant depending on $n$ and $c$, or is it unbounded as $\delta \to 1$ (paralleling Guan et al.'s result for correlated features)?

---

## 4. Complexity of Unassisted Graduation Allocation

* **Question**: What is the computational complexity of optimizing the unassisted graduation cosine objective under a budget $T$?
  $$\max_{\sum m_j = T} \frac{g(m)^\top d}{\|g(m)\| \|d\|}$$
* **Current Status**:
  * Random single-feature graduation (G2) is weakly NP-hard and admits an FPTAS via 0–1 Knapsack ([[<G> Curriculum Allocation (Knapsack Equivalence)]]).
  * Unassisted graduation (G1) is nonlinear and non-separable due to the Euclidean norm $\|g(m)\|$ in the denominator. Does it admit a pseudo-polynomial DP or PTAS?

---

## 5. Multi-Action Scaffolding for $K \ge 3$

* **Question**: For $K \ge 3$ actions, what is the geometric characterization of row error profiles where selecting an intermediate action menu size $1 < |A_t| < K$ is strictly optimal?
* **Current Status**:
  * For $K=2$, single-row debiasing is proved to accelerate sign flips by avoiding row-error cancellation ([[<G> Targeted Row Debiasing (|A_t| < K)]]).
  * Extension to general $K \times n$ utility matrices $U$ and mental models $H$ remains open.
