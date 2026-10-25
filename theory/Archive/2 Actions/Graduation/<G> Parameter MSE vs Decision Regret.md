# <G> Parameter MSE vs Decision Regret

This note directly addresses the question raised in `Problem Description.md`: **Should the AI optimize for parameter accuracy ($\|H - U\|_F^2$) or decision regret?**

We prove that parameter estimation error and decision regret can **diverge catastrophically**: a schedule designed to minimize parameter MSE can destroy cognitive proportionality, resulting in graduation decision regret that is **strictly worse than teaching nothing at all**.

---

## 1. Formulation of the Two Objectives

Fix budget $T$ and allocation $m = (m_1, \dots, m_n)$ with $\sum_{j=1}^n m_j = T$:

1. **Parameter MSE Loss**:
   $$\mathcal{L}_{\text{norm}}(m) := \|g(m) - d\|_2^2 = \sum_{j=1}^n \psi(m_j)^2 \big(g_j^{(0)} - d_j\big)^2$$

2. **Unassisted Decision Regret Loss**:
   $$\mathcal{L}_{\text{grad}}^{\text{G1}}(m) := \frac{\|d\|}{\sqrt{2\pi}} \left( 1 - \frac{g(m)^\top d}{\|g(m)\| \|d\|} \right)$$

---

## 2. Parameter MSE: Optimal Water-Filling

### Proposition: Greedy Water-Filling is Optimal for Convex $\psi^2$
If $\psi(m)^2$ is convex on $\mathbb{N}$ (marginal reductions $\psi(m)^2 - \psi(m+1)^2$ are non-increasing), the parameter error $\mathcal{L}_{\text{norm}}(m)$ is strictly minimized by **greedy water-filling**:
* At each step, allocate an exposure to the feature $j$ maximizing the marginal reduction:
  $$\Delta \mathcal{L}_{\text{norm}}(j, m_j) = \big(g_j^{(0)} - d_j\big)^2 \Big( \psi(m_j)^2 - \psi(m_j + 1)^2 \Big)$$

---

## 3. The Divergence Theorem

### Theorem: The Norm-Minimizer Can Perform Worse Than Teaching Nothing
There exists a family of instances where the parameter norm-minimizing schedule incurs strictly higher unassisted graduation regret than the regret-minimizing schedule, and is **strictly worse than doing nothing ($T = 0$)**:

$$\mathcal{L}_{\text{grad}}^{\text{G1}}(m^{\text{regret}}) < \mathcal{L}_{\text{grad}}^{\text{G1}}(\mathbf{0}) < \mathcal{L}_{\text{grad}}^{\text{G1}}(m^{\text{norm}})$$

#### Proof by Explicit Construction:
Let $n = 3$, $d = (1, 1, 1)$, and initial beliefs $g^{(0)} = (M, M, -1)$ with large $M > 3$.
Assume one-shot learning ($\psi(m) = \mathbf{1}[m=0]$) and a training budget of $T = 1$ exposure.

1. **Parameter Error Comparison**:
   * Teaching feature 1 (or 2) sets $g_1 = 1$, leaving parameter error:
     $$\|g - d\|_2^2 = (1 - 1)^2 + (M - 1)^2 + (-1 - 1)^2 = (M - 1)^2 + 4$$
   * Teaching feature 3 sets $g_3 = 1$, leaving parameter error:
     $$\|g - d\|_2^2 = (M - 1)^2 + (M - 1)^2 + (1 - 1)^2 = 2(M - 1)^2$$
   * Since $M > 3$, $(M - 1)^2 + 4 < 2(M - 1)^2$.
   * **Conclusion**: The norm-minimizer **always teaches feature 1 (or 2)**.

2. **Decision Regret Comparison (Cosine Alignment)**:
   * **Teach feature 1**: $g = (1, M, -1)$.
     $$\cos(g, d) = \frac{1 + M - 1}{\sqrt{3(1 + M^2 + 1)}} = \frac{M}{\sqrt{3(M^2 + 2)}} \xrightarrow{M \to \infty} \frac{1}{\sqrt{3}} \approx 0.577$$
   * **Teach nothing ($T = 0$)**: $g = (M, M, -1)$.
     $$\cos(g, d) = \frac{M + M - 1}{\sqrt{3(M^2 + M^2 + 1)}} = \frac{2M - 1}{\sqrt{3(2M^2 + 1)}} \xrightarrow{M \to \infty} \frac{2}{\sqrt{6}} = \sqrt{\frac{2}{3}} \approx 0.816$$
   * **Teach feature 3 (Regret-optimal)**: $g = (M, M, 1)$.
     $$\cos(g, d) = \frac{M + M + 1}{\sqrt{3(M^2 + M^2 + 1)}} = \frac{2M + 1}{\sqrt{3(2M^2 + 1)}} \xrightarrow{M \to \infty} \frac{2}{\sqrt{6}} = \sqrt{\frac{2}{3}} \approx 0.816$$

3. **Asymptotic Regret Gap**:
   Since expected regret is $\frac{\|d\|}{\sqrt{2\pi}}(1 - \cos)$:
   $$\lim_{M \to \infty} \left[ \mathcal{L}_{\text{grad}}^{\text{G1}}(m^{\text{norm}}) - \mathcal{L}_{\text{grad}}^{\text{G1}}(m^{\text{regret}}) \right] = \frac{\sqrt{2/3} - \sqrt{1/3}}{\sqrt{2\pi}} \|d\| \approx \mathbf{0.095 \|d\|}$$

---

## 4. Why Does This Occur? (The Mechanism)

```
       True Vector: d = (1, 1, 1)   [Diagonal]
       
       Initial Belief: g^(0) = (M, M, -1)
       -> Features 1 & 2 are balanced with each other (M : M). 
          The human understands their relative tradeoff perfectly!
          
       Norm Minimizer: teaches Feature 1 -> g = (1, M, -1)
       -> Destroys proportionality! Feature 2 now overpowers Feature 1 by M : 1.
          Decision boundaries are completely warped.
```

* **Scale Invariance of Decisions**: Discrete decision regret depends only on the sign of $g^\top x$. Scaling all beliefs by a constant does not change decisions at all ($\cos(\alpha g, d) = \cos(g, d)$).
* **Norm Penalizes Scale, Not Distortion**: Parameter MSE focuses entirely on pulling $M \to 1$. In doing so on one feature, it shatters the **relative ratio** between features 1 and 2, introducing severe directional bias into the decision boundary.

> [!CAUTION] Design Takeaway
> **Never optimize pedagogical AI assistance using parameter MSE as a proxy for decision competency.** The AI must optimize directly for decision regret to preserve inter-feature calibration.
