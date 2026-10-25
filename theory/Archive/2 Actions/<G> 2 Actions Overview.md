# <G> 2 Actions Overview

This note outlines the analytical reduction for the **binary action setting** ($K=2$), which serves as the core testbed for the mathematical theory of pedagogical AI assistance.

---

## 1. Environment & Coordinate Reduction

Let the two actions be $\{1, 2\}$, with ground truth utility vectors $u_1, u_2 \in \mathbb{R}^n$ and human internal belief vectors $h_1^{(t)}, h_2^{(t)} \in \mathbb{R}^n$.
Define the **ground truth difference vector** and the **perceived difference vector**:
$$d := u_1 - u_2 \in \mathbb{R}^n \setminus \{0\}, \qquad g^{(t)} := h_1^{(t)} - h_2^{(t)} \in \mathbb{R}^n$$

* **Optimal action**: In state $x \sim \mathcal{N}(0, I_n)$, action 1 is optimal iff $u_1^\top x > u_2^\top x \iff d^\top x > 0$.
* **Decision regret**: For any predicted action $\hat{y} \in \{1, 2\}$, the instantaneous regret is:
  $$\ell_{\text{regret}}(x, \hat{y}) = |d^\top x| \cdot \mathbf{1}[\hat{y} \neq y^*(x)]$$

---

## 2. Learning Dynamics of the Difference Vector

When both actions are presented ($A_t = \{1, 2\}$), the exposure count $m_{kj}(t)$ for both rows increases symmetrically: $m_{1j}(t) = m_{2j}(t) =: m_j(t)$.
Under the contraction dynamic $h_{kj}^{(t+1)} - u_{kj} = \psi(m_{kj}(t))(h_{kj}^{(0)} - u_{kj})$, the perceived difference vector evolves deterministically:
$$g_j(m_j) = (1 - \psi(m_j)) d_j + \psi(m_j) g_j^{(0)}$$
where $\psi: \mathbb{N} \to [0, 1]$ is non-increasing with $\psi(0) = 1$ and $\lim_{m \to \infty} \psi(m) = 0$.

### Permanent Sign-Flip Threshold $\tau_j$
A feature $j$ has an **initially wrong sign** (a "bad" feature) if $g_j^{(0)} d_j < 0$.
Because $g_j(m) = d_j + \psi(m)(g_j^{(0)} - d_j)$, $g_j(m)$ achieves the correct sign $\operatorname{sgn}(g_j(m)) = \operatorname{sgn}(d_j)$ once:
$$\psi(m_j) < \frac{|d_j|}{|d_j| + |g_j^{(0)}|}$$
We define the **flip count** $\tau_j$:
$$\tau_j := \begin{cases} 0 & \text{if } g_j^{(0)} d_j > 0 \text{ (initially good)} \\ \min\left\{m \ge 0 : \psi(m) < \frac{|d_j|}{|d_j| + |g_j^{(0)}|}\right\} & \text{if } g_j^{(0)} d_j < 0 \text{ (initially bad)} \end{cases}$$

---

## 3. Exact One-Round Ex-Ante Expected Regret

For any fixed feature subset $F \subseteq [n]$ chosen *ex-ante* (independently of the realization $x^{(t)}$), the human predicts $\hat{y} = 1$ iff $g_F^\top x_F > 0$.
Define:
$$U := \frac{d^\top x}{\|d\|} \sim \mathcal{N}(0, 1), \qquad V := \frac{g_F^\top x_F}{\|g_F\|} \sim \mathcal{N}(0, 1)$$
The correlation between $U$ and $V$ is the angular alignment:
$$r(F, g) := \operatorname{Corr}(U, V) = \frac{g_F^\top d_F}{\|g_F\| \|d\|} \in [-1, 1]$$
Since $\ell(x, \hat{y}) = \|d\| \cdot |U| \cdot \mathbf{1}[U V < 0]$, evaluating the Gaussian integral $\mathbb{E}[|U|\mathbf{1}[UV < 0]] = \frac{1 - r}{\sqrt{2\pi}}$ yields the exact closed-form expected regret:

$$\Lambda_F(g) = \frac{\|d\|}{\sqrt{2\pi}} \left( 1 - \frac{g_F^\top d_F}{\|g_F\| \|d\|} \right)$$

### Single-Feature Specialization ($|F| = 1$)
When only a single feature $F = \{j\}$ is shown, the magnitude $|g_j|$ cancels out in the correlation ratio:
$$r(\{j\}, g) = \operatorname{sgn}(g_j) \frac{d_j}{\|d\|}$$
Thus, $\Lambda_{\{j\}}(g_j)$ collapses to **only two discrete values**:
$$\Lambda_{\{j\}}(g_j) = \begin{cases} a_j := \frac{\|d\| - |d_j|}{\sqrt{2\pi}} & \text{if } \operatorname{sgn}(g_j) = \operatorname{sgn}(d_j) \text{ (correct/good)} \\ b_j := \frac{\|d\| + |d_j|}{\sqrt{2\pi}} & \text{if } \operatorname{sgn}(g_j) = -\operatorname{sgn}(d_j) \text{ (inverted/bad)} \end{cases}$$
The regret gap eliminated upon flipping feature $j$ is:
$$\Delta_j := b_j - a_j = \sqrt{\frac{2}{\pi}} |d_j|$$

---

## 4. Organization of Results

The theoretical results for $K=2$ are organized into two primary phases:

* **Training Phase ($t \le T$)**:
  * [[<G> Training Overview]]: Overview of pedagogical mechanisms.
  * [[<G> Ex-Post Opportunistic Teaching (c = 1)]]: Zero-regret availability via Sheppard's arcsine law and optimal Bellman threshold policy.
  * [[<G> Targeted Row Debiasing (|A_t| < K)]]: Scaffolding via action menu restriction to accelerate sign flips.
  * [[<G> Multi-Feature Chaperoning (c >= 2)]]: Multi-feature buffering and transient bounds.

* **Graduation Phase ($t > T$)**:
  * [[<G> Graduation Overview]]: Curriculum design for eventual independence.
  * [[<G> Curriculum Allocation (Knapsack Equivalence)]]: Mapping budget allocation to 0–1 Knapsack and loud-distraction debiasing.
  * [[<G> Contiguous Block Scheduling]]: Gittins-like block index for sequence ordering.
  * [[<G> Parameter MSE vs Decision Regret]]: Fundamental divergence between parameter estimation and decision quality.

* **Open Directions**:
  * [[<G> Open Hypotheses & Conjectures]]: Unresolved conjectures and computational questions.
