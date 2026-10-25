# <G> Curriculum Allocation (Knapsack Equivalence)

This note establishes how the AI should allocate a finite training budget $T$ across features ($\sum_{j=1}^n m_j = T$) under two distinct graduation regimes: **Random Single-Feature Graduation** and **Unassisted Full-State Graduation**.

---

## 1. Single-Feature Random Graduation (G2) $\equiv$ 0–1 Knapsack

Suppose that at graduation, nature or the user's task queries a single feature $j$ drawn from a task distribution $q = (q_1, \dots, q_n)$, and the human acts using only that feature.
The expected graduation regret is:
$$\mathcal{L}_{\text{grad}}(m) = \sum_{j=1}^n q_j \Lambda_{\{j\}}(g_j(m_j))$$
Let $B := \{j \in [n] : \operatorname{sgn}(g_j^{(0)}) = -\operatorname{sgn}(d_j)\}$ denote the set of initially bad features.
Recall that for $j \notin B$, $\Lambda_{\{j\}}(g_j(m_j)) \equiv a_j$ for all $m_j \ge 0$.
For $j \in B$, the single-feature regret takes only two values:
$$\Lambda_{\{j\}}(g_j(m_j)) = b_j - \Delta_j \cdot \mathbf{1}[m_j \ge \tau_j], \qquad \Delta_j = \sqrt{\frac{2}{\pi}} |d_j|$$
where $\tau_j = \min\{m : \psi(m) < \frac{|d_j|}{|d_j| + |g_j^{(0)}|}\}$ is the exposure threshold required to permanently flip feature $j$'s sign.

### Theorem: Knapsack Equivalence
When training regret weight $\lambda \to 0$:
$$\min_{\sum m_j = T} \mathcal{L}_{\text{grad}}(m) = \mathcal{L}_{\text{grad}}(\mathbf{0}) - \sqrt{\frac{2}{\pi}} \max_{S \subseteq B} \left\{ \sum_{j \in S} q_j |d_j| \quad \text{s.t.} \quad \sum_{j \in S} \tau_j \le T \right\}$$

#### Proof:
Flipping any subset of bad features $S \subseteq B$ requires allocating $m_j \ge \tau_j$ for all $j \in S$. The minimal exposure cost to flip the set $S$ is exactly $\sum_{j \in S} \tau_j$. If $\sum_{j \in S} \tau_j \le T$, the allocation is feasible (spending any residual exposures arbitrarily). The regret reduction achieved is $\sum_{j \in S} q_j \Delta_j = \sqrt{\frac{2}{\pi}} \sum_{j \in S} q_j |d_j|$. This is the standard 0–1 Knapsack objective where each bad feature $j \in B$ has **value** $v_j = q_j |d_j|$ and **weight** $w_j = \tau_j$.

### Algorithmic & Complexity Consequences:
* **NP-Hardness**: The curriculum allocation problem is weakly NP-hard when weights $\tau_j$ and budget $T$ are given in binary.
* **Exact DP**: Solvable in pseudo-polynomial time $O(|B| T)$ via standard dynamic programming.
* **FPTAS**: Admits a Fully Polynomial-Time Approximation Scheme achieving $(1 - \epsilon)$-optimal curriculum allocation in time $O(|B|^3 / \epsilon)$.

---

## 2. Unassisted Full-State Graduation (G1): Extinguishing Loud Distractions

Under **Unassisted Graduation**, the graduated human receives no algorithmic help and must predict using the full state vector $x \in \mathbb{R}^n$.
The graduation loss is determined by the cosine similarity between $g(m)$ and $d$:
$$\mathcal{L}_{\text{grad}}^{\text{G1}}(m) = \frac{\|d\|}{\sqrt{2\pi}} \left( 1 - \frac{g(m)^\top d}{\|g(m)\| \|d\|} \right)$$

### Proposition: Unassisted Graduation Can Strictly Prioritize Irrelevant Features
Under unassisted graduation, an **irrelevant feature** ($d_j = 0$) can strictly dominate every relevant feature ($d_i > 0$).

#### Proof (Explicit Counterexample):
* Consider $n = 2$, with true weights $d = (1, 0)$ (feature 1 is fully relevant, feature 2 is completely irrelevant).
* Let initial beliefs be $g^{(0)} = (1, M)$ with large $M > 0$.
* The human already knows the true weight of feature 1 ($g_1(0) = d_1 = 1$). Teaching feature 1 changes nothing: $g_1(m_1) \equiv 1$.
* Feature 2 is a **"loud distraction"**: while $d_2 = 0$, the human mistakenly believes it has weight $g_2(m_2) = M \psi(m_2)$.
* The correlation ratio is:
  $$\frac{g(m)^\top d}{\|g(m)\| \|d\|} = \frac{1 \cdot 1 + M\psi(m_2) \cdot 0}{\sqrt{1 + M^2 \psi(m_2)^2} \cdot 1} = \frac{1}{\sqrt{1 + M^2 \psi(m_2)^2}}$$
* This quantity is **strictly increasing in $m_2$** and completely invariant to $m_1$.
* Therefore, for any budget $T \ge 1$, the **unique optimal curriculum** is:
  $$m_1^* = 0, \qquad m_2^* = T$$

> [!IMPORTANT] Takeaway
> In unassisted graduation, eliminating human misconceptions on irrelevant "noise" features can be vastly more important than reinforcing already-understood signal features.
