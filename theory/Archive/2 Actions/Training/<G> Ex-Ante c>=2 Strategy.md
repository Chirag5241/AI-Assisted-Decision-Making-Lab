# <G> Ex-Ante c>=2 Strategy

This note formalizes the **Ex-Ante, Multi-Feature ($c \ge 2$)** training strategy for binary actions ($K=2$).

---

## 1. Problem Setting

At each training round $t$, $\mathcal{A}$ chooses a subset $F_t \subseteq [n]$ with $|F_t| \le c$ **before** observing $x^{(t)} \sim \mathcal{N}(0, I_n)$.
The human observes $x_{F_t}^{(t)}$ and predicts $\hat{y} = 1$ iff:
$$\sum_{j \in F_t} g_j^{(t)} x_j^{(t)} > 0$$

---

## 2. One-Round Expected Regret: Angular Correlation

Unlike $c=1$, the expected regret is non-linear and coupled across features:
$$\Lambda_F(g) = \frac{\|d\|}{\sqrt{2\pi}} \left( 1 - \frac{g_F^\top d_F}{\|g_F\| \|d\|} \right)$$
where:
$$g_F^\top d_F = \sum_{j \in F} g_j d_j, \qquad \|g_F\| = \sqrt{\sum_{j \in F} g_j^2}$$
* Regret is minimized by maximizing the cosine correlation $r(F, g) = \frac{g_F^\top d_F}{\|g_F\| \|d\|}$.

---

## 3. Structural Theorems

### Theorem 1: The Chaperone Inequality
Let $F^*$ be a set of $c$ bad features ($g_j^{(0)} d_j < 0$).
Let $k \notin F^*$ be a chaperone feature with $d_k > 0$ and $g_k^{(0)} = G > 0$.
For any subset $F' \subset F^*$ with $|F'| = c - 1$:
$$\text{If } \quad G \cdot d_k > -(g_{F'}^{(0)})^\top d_{F'},$$
then presenting $\{k\} \cup F'$ achieves $r(\{k\} \cup F', g^{(0)}) > 0 > r(F^*, g^{(0)})$, strictly lowering regret while teaching $F'$.

### Theorem 2: Bounded Transient Length
Under one-shot learning ($\psi(m) = \mathbf{1}[m=0]$), every optimal schedule reaches its stationary set $F_{\text{final}}$ in at most:
$$L \le |F_{\text{final}}| \le c \quad \text{transient rounds}$$

---

## 4. Algorithmic Decision Rule for $\mathcal{A}$

```
                          Optimal Suffix Selection
                 F* = argmax_{|F| <= c} ||d_F|| (Top-c |d_j|)
                                    │
                                    ▼
                         Is F* already learned?
                               /          \
                             YES           NO
                            /                \
                     Stationary:          Dynamic Chaperoning:
                     Show F* forever      Pair unlearned j in F* with
                                          best chaperone in S_t
```

### Algorithm: Greedy Chaperoned Curriculum
1. **Identify Target Stationary Set**:
   $F^* = \arg\max_{|F| \le c} \sum_{j \in F} d_j^2$ (the $c$ features with largest $|d_j|$).
2. **Dynamic Transient (at round $t$)**:
   * If all features in $F^*$ are learned, show $F^*$ forever.
   * Otherwise, select the unlearned feature $j^* \in F^* \setminus S_t$ with largest $|d_{j^*}|$.
   * Fill the remaining $c-1$ slots with features from $S_t$ (or good initial features) that maximize the aggregate correlation $r(\{j^*\} \cup C, g^{(t)})$.
   * Show $\{j^*\} \cup C$ to advance $j^*$ while keeping decision regret bounded.
