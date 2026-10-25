# <G> Ex-Ante c=1 Strategy

This note formalizes the **Ex-Ante, Single-Feature ($c = 1$)** training strategy for binary actions ($K=2$).

---

## 1. Problem Setting

At each training round $t \in \{0, \dots, T-1\}$, the algorithm $\mathcal{A}$ must select a single feature $F_t = \{j\} \subset [n]$ **before** observing the environment state $x^{(t)} \sim \mathcal{N}(0, I_n)$.
Both actions are presented: $A_t = \{1, 2\}$.
The human observes $x_j^{(t)}$ and predicts $\hat{y} = 1$ iff $g_j^{(t)} x_j^{(t)} > 0$.

---

## 2. One-Round Expected Regret: Discrete Arm Collapse

Let $d = u_1 - u_2$ and $g^{(t)} = h_1^{(t)} - h_2^{(t)}$.
Under $x \sim \mathcal{N}(0, I_n)$, the expected decision regret for showing feature $j$ collapses to only **two discrete values** based solely on whether $\operatorname{sgn}(g_j(m_j)) = \operatorname{sgn}(d_j)$:

$$\Lambda_{\{j\}}(g_j(m_j)) = \begin{cases} a_j := \frac{\|d\| - |d_j|}{\sqrt{2\pi}} & \text{if } m_j \ge \tau_j \text{ (good sign)} \\ b_j := \frac{\|d\| + |d_j|}{\sqrt{2\pi}} & \text{if } m_j < \tau_j \text{ (bad sign)} \end{cases}$$

where $\tau_j = \min\{m \ge 0 : \psi(m) < \frac{|d_j|}{|d_j| + |g_j^{(0)}|}\}$ is the permanent flip count, and the regret reduction gap is:
$$\Delta_j := b_j - a_j = \sqrt{\frac{2}{\pi}} |d_j|$$

---

## 3. Structural Theorems

### Theorem 1: Contiguous Block Property
Any optimal sequence of feature presentations in this setting schedules feature $j$ in a **single contiguous block** of length $m_j$. Features are never interleaved.

### Theorem 2: The Block Priority Index
The optimal priority ordering among bad features to be taught is determined by sorting according to the Gittins-like block index:
$$\nu_j(\tau_j) := \frac{\sum_{s=0}^{\tau_j - 1} \delta^s b_j + \sum_{s=\tau_j}^\infty \delta^s a_j}{\frac{1}{1 - \delta}} = a_j + (1 - \delta^{\tau_j})\sqrt{\frac{2}{\pi}}|d_j|$$

---

## 4. Algorithmic Decision Rule for $\mathcal{A}$

### The Permanent Crutch vs. Teaching Tradeoff
Let $j^*_{\text{good}} = \arg\min_{k : \tau_k = 0} a_k = \arg\max_{k : \tau_k = 0} |d_k|$ be the best initially aligned feature (minimal immediate loss without any learning needed).

1. **When to Teach**: $\mathcal{A}$ should invest in teaching a bad feature $j$ over the permanent crutch $j^*_{\text{good}}$ if and only if:
   $$\nu_j(\tau_j) < a_{j^*_{\text{good}}} \iff \sqrt{\frac{2}{\pi}}\big(|d_{j^*_{\text{good}}}| - |d_j|\big) + (1 - \delta^{\tau_j})\sqrt{\frac{2}{\pi}}|d_j| < 0$$
   $$\iff \delta^{\tau_j} |d_j| > |d_{j^*_{\text{good}}}|$$
   * **Interpretation**: The discounted asymptotic benefit of the learned feature ($\delta^{\tau_j} |d_j|$) must strictly exceed the baseline quality of the available crutch ($|d_{j^*_{\text{good}}}|$).

2. **Algorithm**:
   * If $\max_{j \in B} \delta^{\tau_j} |d_j| \le |d_{j^*_{\text{good}}}|$, the optimal policy is a **permanent crutch**: show $j^*_{\text{good}}$ forever.
   * Otherwise, sort all candidate bad features satisfying $\delta^{\tau_j} |d_j| > |d_{j^*_{\text{good}}}|$ in increasing order of index $\nu_j(\tau_j)$, teach each in a contiguous block of $\tau_j$ rounds, and then switch permanently to the newly best learned feature.
