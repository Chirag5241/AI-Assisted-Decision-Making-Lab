# <G> - Multi-Action Pedagogical Algorithm: The Margin-Cover Curriculum (MCC)

This document develops the **Margin-Cover Curriculum (MCC)** algorithm for training a clinician across $K \ge 3$ actions.

---

## 1. The Multi-Action Pedagogical Dilemma

In the multi-action setting, the AI preceptor cannot simply train a single difference vector.
Across the patient population, different patients arrive with different triage presentations $s \sim \mathcal{D}_s$, activating different disease margins $(k_1(s), k_2(s))$.

If an AI uses a standard greedy policy, it will train the clinician on common conditions (e.g., standard viral infections) while leaving **rare, high-stakes disease margins completely uncalibrated**.

At graduation, when a rare cardiac or neurological patient arrives, the clinician will suffer from **fatal diagnostic blind spots**.

---

## 2. The Margin Risk Graph

Let $\mathcal{M} = \{(i, j) : 1 \le i < j \le K\}$ denote the set of all $\binom{K}{2}$ pairwise clinical margins.
For each margin $(i, j) \in \mathcal{M}$:
1. **Activation Probability**:
   $$q_{ij} := P_{s \sim \mathcal{D}_s}\big( \{k_1(s), k_2(s)\} = \{i, j\} \big)$$
2. **Clinical Stakes (Regret Severity)**:
   $$S_{ij} := \mathbb{E}\big[ |(u_i - u_j)^\top x| \;\big|\; \text{margin } (i, j) \text{ active} \big]$$
3. **Margin Vulnerability**:
   $$V_{ij}(H) := \| (h_i - h_j) - (u_i - u_j) \|_2$$

The total expected clinical risk of margin $(i, j)$ is:
$$\text{Risk}_{ij}(H) = q_{ij} \cdot S_{ij} \cdot V_{ij}(H)$$

---

## 3. The Margin-Cover Curriculum (MCC) Algorithm

Rather than naively applying knapsack solvers, the MCC algorithm directly minimizes the integrated population risk across all clinical margins:

```
Algorithm: Margin-Cover Curriculum (MCC)
Input: Patient distribution D_s, true utility matrix U, initial mental model H^(0), budget T
Output: Optimal pedagogical training policy pi*

1. Initialize exposure matrix M = zeros(K, n)
2. Build Margin Risk Graph: compute q_ij and S_ij for all (i, j) in M
3. For round t = 1 to T:
4.    Identify the Most Dangerous Uncalibrated Margin:
5.        (i*, j*) = argmax_{(i, j)} Risk_ij(H(M))
6.    Identify the Critical Decisive Biomarker for this margin:
7.        k* = argmax_{k in [n]} |(u_i* - u_j*)_k - (h_i* - h_j*)_k| * sigma_k
8.    Observe patient s^(t), x^(t):
9.        If k* in Z(x^(t)) (Zero-regret opportunity available):
10.           Present feature k* at zero regret!
11.       Else if stakes |D^(t)| < Threshold:
12.           Present feature k* (Paid low-stakes teaching)
13.       Else:
14.           Present safest calibrated feature to protect patient
15.   Update exposure matrix M and clinician matrix H
16. Return trained matrix H^(T) and policy log
```

### Theoretical Guarantees:
1. **Zero Blind Spots**: The MCC algorithm guarantees that no high-stakes margin is starved of exposures, bounding maximum graduation regret:
   $$\max_{(i, j)} \text{Regret}_{ij} \le O\left( \psi(T / |\mathcal{M}|) \right)$$
2. **Safe Exploration**: By coupling margin prioritization with the ex-post zero-regret set $Z(x^{(t)})$, the algorithm achieves high training efficiency without compromising patient care.
