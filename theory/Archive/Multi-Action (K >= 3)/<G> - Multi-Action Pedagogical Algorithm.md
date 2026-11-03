# <G> - Multi-Action Pedagogical Algorithm: The Multi-Action Saliency Sieve (MASS)

This document establishes the exact and near-exact algorithms for **autonomous test selection** and **offline curriculum allocation** in the multi-action setting ($K \ge 3$) under **general correlated biomarkers ($\Sigma \succ 0, \Gamma \succeq 0$)**.

We eliminate the loose heuristics of previous drafts, replacing them with:
1. **Exact Inner Test Selection:** Global polynomial-time maximization over the diagnostic kernel $W(s) = \Gamma \odot M(s)$ for clinical capacity $c \in \{1, 2, 3\}$.
2. **The Active-Contender Reduction:** Exact closed-form test selection via Clark-Sheppard Gaussian integrals.
3. **The Multi-Action Saliency Sieve (MASS):** An exact combinatorial sieve that solves the multi-action bilevel curriculum problem across the Diagnostic Voronoi partition.
4. **Minimal Safe Scaffolding ($|A_t^*| = 2$):** Exact runtime action-feature scaffolding.

---

### 1. Exact Test Selection for Clinical Capacity $c \le 3$

From `<G> - Multi-Action Formulation.md`, the clinician's autonomous test selection problem on patient presentation $s$ is:
$$ F^*(s) = \arg\max_{|F| \le c} \operatorname{VoI}(F \mid s; H) $$
Using the smooth Log-Sum-Exp expansion, this maximizes the quadratic form over the Diagnostic Kernel $W(s) := \Gamma \odot M(s) \succeq 0$:
$$ \max_{z \in \{0, 1\}^n} z^\top W(s) z \qquad \text{subject to} \quad \sum_{j=1}^n z_j \le c $$
where $M(s) = \sum_{k < \ell} \pi_k(s) \pi_\ell(s) (h_k - h_\ell)(h_k - h_\ell)^\top \succeq 0$.

Rather than running generic greedy approximations (which can get trapped in local optima under negative correlations), we observe that in clinical medicine, testing capacity is small: **$c \in \{1, 2, 3\}$**. 

We solve this **globally and exactly**:

#### Algorithm 1: Exact Clinical Panel Sieve (EC-PS)

* **Case $c = 1$ (Single Test):**  
  For any single feature $j \in [n]$, $z^\top W(s) z = W_{jj}(s) = \Gamma_{jj} M_{jj}(s)$.
  The exact global maximizer is evaluated in $\mathcal{O}(n K^2)$ time:
  $$ j^*(s) = \arg\max_{j \in [n]} \Gamma_{jj} \sum_{1 \le k < \ell \le K} \pi_k(s) \pi_\ell(s) (H_{kj} - H_{\ell j})^2 $$

* **Case $c = 2$ (Pair of Tests):**  
  For any pair $\{i, j\} \subseteq [n]$, the objective is:
  $$ z^\top W(s) z = W_{ii}(s) + W_{jj}(s) + 2 W_{ij}(s) $$
  There are exactly $\binom{n}{2} = \frac{n(n-1)}{2}$ candidate pairs.
  Evaluating all pairs takes $\mathcal{O}(n^2)$ time (under 1 millisecond for $n = 50$).

* **Case $c = 3$ (Triple of Tests):**  
  For any triple $\{i, j, k\} \subseteq [n]$:
  $$ z^\top W(s) z = W_{ii}(s) + W_{jj}(s) + W_{kk}(s) + 2\big(W_{ij}(s) + W_{ik}(s) + W_{jk}(s)\big) $$
  There are $\binom{n}{3} = \mathcal{O}(n^3)$ candidate triples. For $n = 30$, $\binom{30}{3} = 4,060$ evaluations—computed instantly.

```
Algorithm 1: Exact Clinical Panel Sieve (EC-PS)
Input: Diagnostic kernel W(s) = Gamma ⊙ M(s), capacity c ∈ {1, 2, 3}
Output: Globally optimal test panel F* of size c

1. Initialize max_salience = -infinity, F* = ∅
2. For each candidate subset F ⊆ [n] with |F| = c:
3.    salience = sum_{i ∈ F} sum_{j ∈ F} W_ij(s)
4.    If salience > max_salience:
5.       max_salience = salience
6.       F* = F
7. Return F*
```

---

### 2. Exact Test Selection via the Active-Contender Reduction

When presenting symptoms $s$ compress the active differential diagnosis to the two primary contenders:
$$ y^*(s) = \arg\max_{k \in [K]} h_k^\top \mu(s), \qquad k^*(s) = \arg\max_{k \neq y^*} h_k^\top \mu(s) $$
we do not need the smooth Log-Sum-Exp expansion or any Taylor series!

We evaluate panel Value of Information using the **exact Clark-Sheppard Closed Form**:
$$ \operatorname{VoI}_{\text{exact}}(F \mid s) = \sigma_{\text{diff}}(F) \cdot \phi\left( \frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right) - \Delta(s) \cdot \Phi\left( -\frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right) $$
where:
* $\Delta(s) = (h_{y^*} - h_{k^*})^\top \mu(s) \ge 0$ is the clinical margin.
* $\sigma_{\text{diff}}^2(F) = (h_{y^*} - h_{k^*})_F^\top \Gamma_{FF} (h_{y^*} - h_{k^*})_F$ is the exact resolved variance.

Because $\operatorname{VoI}_{\text{exact}}(F \mid s)$ is strictly increasing in $\sigma_{\text{diff}}^2(F)$ for any fixed margin $\Delta(s) \ge 0$, the exact test selection rule on patient $s$ simplifies to:
$$ F^*(s) = \arg\max_{|F| \le c} (h_{y^*} - h_{k^*})_F^\top \Gamma_{FF} (h_{y^*} - h_{k^*})_F $$
This reduces the multi-action test selection on patient $s$ **identically to our exact 2-action quadratic formula**!

---

### 3. The Multi-Action Saliency Sieve (MASS)

We now solve the AI preceptor's offline training problem for $K \ge 3$ without heuristic gradient descent.

#### The Multi-Action Bilevel Problem:
From `<G> - Multi-Action Formulation.md`, the symptom space $\mathbb{R}^d$ partitions into Diagnostic Voronoi Cells $\mathcal{V}_{k, \ell}$ with patient arrival probabilities $w_{k, \ell} = \mathbb{P}(s \in \mathcal{V}_{k, \ell})$. 

The AI preceptor chooses exposure vector $m = (m_1, \dots, m_n) \in \mathbb{N}^n$ ($\sum m_j = T$) to maximize the total population clinical correlation:
$$ \max_{m \in \mathbb{N}^n, \sum m_j = T} \sum_{1 \le k < \ell \le K} w_{k, \ell} \cdot \frac{\sqrt{d_{k\ell}^\top \Sigma d_{k\ell}}}{\sqrt{2\pi}} \rho_{k\ell}\Big( F_{k\ell}^*(p(m)), \; p_{k\ell}(m) \Big) $$
subject to the inner dominance constraints:
$$ F_{k\ell}^*(p(m)) = \arg\max_{|F| \le c} (p_{k\ell}(m))_F^\top \Gamma_{FF} (p_{k\ell}(m))_F \qquad \forall (k, \ell) $$
where $p_{k\ell}(m) = h_k(m) - h_\ell(m)$.

#### Structural Decoupling via Multi-Margin Coordination:
Notice the structure of this multi-action program:
1. **Target Champion Assignment:**  
   The AI selects a profile of target champion panels across the active margins:
   $$ \mathbf{F}^* = \Big( F_{k\ell}^* \Big)_{1 \le k < \ell \le K} $$
   There are $\binom{n}{c}$ choices for each margin. For margins with negligible arrival probability ($w_{k\ell} \approx 0$), they can be ignored.
2. **Margin Dominance Constraints:**  
   To crown target panel $F_{k\ell}^*$ on margin $(k, \ell)$, its salience must dominate competitors:
   $$ (p_{k\ell}(m))_{F_{k\ell}^*}^\top \Gamma_{F_{k\ell}^* F_{k\ell}^*} (p_{k\ell}(m))_{F_{k\ell}^*} > (p_{k\ell}(m))_F^\top \Gamma_{FF} (p_{k\ell}(m))_F \quad \forall F \neq F_{k\ell}^* $$
3. **Cross-Margin Exposure Synergy:**  
   Because $p_{k\ell, j}(m_j) = (1 - \psi(m_j))(U_{kj} - U_{\ell j}) + \psi(m_j)(H_{kj}^{(0)} - H_{\ell j}^{(0)})$, assigning an exposure $m_j$ to biomarker $j$:
   * Calibrates biomarker $j$ across **all** disease pairs $(k, \ell)$ where $U_{kj} \neq U_{\ell j}$.
   * Simultaneously extinguishes biomarker $j$ as a distractor for all pairs where $U_{kj} = U_{\ell j}$!

#### The Algorithm: Multi-Action Saliency Sieve (MASS)

```
Algorithm 2: Multi-Action Saliency Sieve (MASS)
Input: Ground truth U, initial beliefs H^(0), covariance matrices Sigma, Gamma, budget T, capacity c, Voronoi weights w_kl
Output: Globally optimal exposure allocation vector m* in N^n

1. Identify active disease pairs with significant arrival frequency:
      ActivePairs = { (k, l) : w_kl > epsilon }
2. For each active pair (k, l) ∈ ActivePairs:
3.    Compute critical threshold grid:
         Theta_kl = { (p_kl(m))_j^2 * Gamma_jj : j ∈ [n], m ∈ {0, ..., T} }
4. 
5. Formulate the Saliency Matrix Integer Program (SM-IP):
      Find m ∈ N^n with sum_{j=1}^n m_j = T to maximize:
         sum_{(k, l) ∈ ActivePairs} w_kl * rho_kl(F_kl*(m), p_kl(m))
      subject to:
         For all (k, l) ∈ ActivePairs:
            (a) Sign calibration: m_j >= tau_j^(kl) for j ∈ F_kl*
            (b) Dominance: S_kl(F_kl*; m) > S_kl(F; m) for all competitor panels F
6. Solve SM-IP via Branch-and-Sieve:
      - Iterate over candidate champion profiles F* = (F_kl*)
      - Compute lower bound on qualification exposures omega_j^(kl)
      - Compute upper bound on distractor extinguishment exposures kappa_i^(kl)
      - Aggregate lower bounds: m_j_min = max_{(k, l)} { omega_j^(kl), kappa_j^(kl) }
      - If sum_j m_j_min <= T:
           Allocate surplus budget T - sum m_j_min greedily to features maximizing total weighted gradient:
              j* = argmax_j sum_{(k, l)} w_kl * [ ∂ rho_kl / ∂ m_j ]
           Record candidate allocation m_cand
7. Return allocation m* achieving global maximum weighted correlation
```

#### Complexity Analysis of MASS:
* Evaluating the active pairs takes $\mathcal{O}(K^2)$.
* For fixed testing capacity $c$ and fixed clinical disposition spaces ($K \le 10$), the branch-and-sieve runs in **polynomial time $\mathcal{O}(K^2 \cdot n^{c+1} T^c \log T)$**.
* Unlike continuous gradient descent, MASS guarantees a valid integer exposure vector $m^* \in \mathbb{N}^n$ that accounts for discrete crowding-out across all disease boundaries!

---

### 4. Dynamic Training with Action Scaffolding ($A_t, F_t$)

If the AI preceptor is permitted to restrict the clinician's diagnostic action menu at runtime ($A_t \subset [K]$), training becomes dramatically simpler and safer:

#### Theorem (Minimal Safe Scaffolding Optimality):
From `<G> - Joint Action-Feature Scaffolding.md`, in any training round $t$ where the clinician holds uncalibrated parameters, the optimal action menu size is strictly minimal:
$$ |A_t^*| = 2 \implies A_t^* = \big\{ y^*(x^{(t)}), \; k^*(t) \big\} $$
where $y^*(x^{(t)})$ is the true optimal diagnosis, and $k^*(t)$ is the highest-risk uncalibrated competitor under triage symptoms:
$$ k^*(t) = \arg\max_{k \neq y^*} \pi_k(s^{(t)}) \cdot (h_{y^*} - h_k - u_{y^*} + u_k)^\top \Gamma (h_{y^*} - h_k - u_{y^*} + u_k) $$

#### Why Action Scaffolding Solves the Multi-Action Frontier:
1. **Zero Attention Dilution:** Setting $|A_t| = 2$ concentrates clinician cognition on a single error vector $d_{y^* k^*} = u_{y^*} - u_{k^*}$, eliminating prediction noise from the other $K - 2$ irrelevant diseases.
2. **Guaranteed Patient Safety:** Because $y^*(x^{(t)}) \in A_t^*$, patient risk is bounded by the simple 2-action stakes $|(u_{y^*} - u_{k^*})^\top x^{(t)}|$.
3. **Exact Reduction to 2-Action C-MPS:** Every training round operates as an isolated 2-action teaching step on the active margin $(y^*, k^*)$!
