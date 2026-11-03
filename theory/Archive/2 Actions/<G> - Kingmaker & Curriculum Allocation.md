# <G> - Kingmaker & Curriculum Allocation: Exact Bilevel Optimization Under General Correlated Biomarkers

This document establishes the exact, polynomial-time offline curriculum allocation algorithm for the binary decision setting ($K = 2$) under **general correlated biomarkers ($\Sigma \succ 0, \Gamma \succeq 0$)** and arbitrary testing capacity $c \ge 1$.

We first define the concept of **Salience** from first principles, formulate the **Bilevel Program**, explain why heuristic knapsacks fail, and develop the **Correlated Monotone Panel Sieve (C-MPS)** algorithm.

---

### 1. Conceptual Foundation: What is "Salience"?

In cognitive psychology and behavioral economics, **salience** describes the degree to which an external stimulus stands out and captures attention. 

In our clinical decision framework, from [[2. Test Selection.md]], when a clinician observes triage symptoms $s$, she autonomously orders the lab panel $F^*(p) \subseteq [n]$ ($|F^*| \le c$) that maximizes perceived Value of Information:
$$ F^*(p) = \arg\max_{|F| \le c} p_F^\top \Gamma_{FF} p_F $$
where $\Gamma = \operatorname{Var}[x \mid s] \succeq 0$ is the residual uncertainty after triage, and $p = h_1 - h_2 \in \mathbb{R}^n$ is the clinician's vector of perceived utility differences.

#### A. Single-Feature Salience ($c = 1$)
For an individual biomarker $j \in [n]$, its objective score is:
$$ S_j(p) := p_j^2 \Gamma_{jj} $$
Notice that $S_j(p)$ is the product of two distinct quantities:
1. **$p_j^2 = (h_{1j} - h_{2j})^2$ (Subjective Importance Squared):**  
   How heavily the clinician *believes* biomarker $j$ sways the decision between Action 1 and Action 2. If the clinician believes biomarker $j$ is irrelevant ($p_j \approx 0$), its salience is zero.
2. **$\Gamma_{jj} = \operatorname{Var}[x_j \mid s]$ (Unresolved Clinical Uncertainty):**  
   The variance of biomarker $j$ that remains unknown after triage symptoms $s$. If biomarker $j$ is already completely determined by symptoms ($\Gamma_{jj} \approx 0$), its salience is zero.

> [!NOTE] Definition: Clinical Salience
> **Salience** is the **perceived variance resolved** by observing a test:
> $$ \text{Salience} = (\text{Perceived Decision Weight})^2 \times (\text{Uncertainty of the Test}) $$
> In a clinician's mind, a test is highly salient if they believe it is decisive for treatment **and** they do not already know its outcome.

#### B. Panel Salience Under Correlated Biomarkers ($c \ge 1$)
When testing capacity allows $c \ge 1$ tests, the clinician evaluates candidate panels $F \subseteq [n]$ with $|F| \le c$ using the quadratic form:
$$ S(F; p) := p_F^\top \Gamma_{FF} p_F = \sum_{j \in F} p_j^2 \Gamma_{jj} + 2 \sum_{i < j \in F} p_i p_j \Gamma_{ij} $$
Notice the fundamental effect of **correlation ($\Gamma_{ij} \neq 0$)**:
* **Synergistic Tests ($\Gamma_{ij} p_i p_j > 0$):**  
  Two tests whose correlation aligns with their perceived decision weights produce *super-additive salience*. Together, they stand out more than the sum of their individual parts.
* **Redundant Tests ($\Gamma_{ij} p_i p_j < 0$ or small eigenvalues in $\Gamma_{FF}$):**  
  Two tests measuring overlapping physiological mechanisms yield diminishing returns. The quadratic form penalizes collinear redundancy.

---

### 2. The Bilevel Curriculum Optimization Problem

The AI preceptor has a total training budget of $T \in \mathbb{N}$ exposures to distribute across the $n$ biomarkers during the training phase ($t = 1, \dots, T$). 

Let $m = (m_1, \dots, m_n) \in \mathbb{N}^n$ denote the integer exposure vector, satisfying:
$$ \sum_{j=1}^n m_j = T $$

#### A. Clinician Belief Update Under Exposure $m$
From `Problem Description.md`, each exposure $m_j$ to biomarker $j$ contracts the clinician's perceived difference toward the ground truth $d_j = u_{1j} - u_{2j}$:
$$ p_j(m_j) = \big(1 - \psi(m_j)\big) d_j + \psi(m_j) p_j^{(0)} $$
where $\psi: \mathbb{N} \to (0, 1]$ is a strictly decreasing contraction factor with $\psi(0) = 1$ and $\lim_{m \to \infty} \psi(m) = 0$.
* If biomarker $j$ receives zero training exposures ($m_j = 0$), the clinician remains trapped in their initial misconception: $p_j(0) = p_j^{(0)}$.
* If biomarker $j$ receives infinite training exposures ($m_j \to \infty$), the belief converges to truth: $p_j(\infty) = d_j$.

#### B. Autonomous Test Selection at Graduation (The Inner Problem)
At graduation ($t > T$), triage symptoms $s = Bx$ are observed. The clinician autonomously orders the panel $F^*(p(m))$ maximizing perceived panel salience:
$$ F^*(p(m)) = \arg\max_{|F| \le c} p_F(m)^\top \Gamma_{FF} p_F(m) $$

#### C. Expected Clinical Graduation Regret (The Outer Objective)
From [[3. Graduation Regret.md]], the clinician's expected graduation regret across arriving patients is:
$$ \mathcal{L}_{\text{grad}}(p(m)) = \frac{\sqrt{d^\top \Sigma d}}{\sqrt{2\pi}} \Big( 1 - \rho\big(F^*(p(m)), p(m)\big) \Big) $$
where the exact clinical correlation achieved at graduation is:
$$ \rho\big(F, p\big) = \frac{d^\top (\Sigma - \Gamma) p + p_F^\top \Gamma_{F, :} \, d}{\sqrt{d^\top \Sigma d} \;\; \sqrt{p^\top (\Sigma - \Gamma) p + p_F^\top \Gamma_{FF} p_F}} $$

#### D. The AI Preceptor's Optimization Program
Because minimizing $\mathcal{L}_{\text{grad}}$ is mathematically identical to maximizing clinical correlation $\rho$, the AI preceptor solves the bilevel program:
$$ 
\begin{aligned}
\max_{m \in \mathbb{N}^n} \quad & \rho\big(F^*(p(m)), p(m)\big) \\
\text{subject to} \quad & \sum_{j=1}^n m_j = T, \quad m_j \in \mathbb{N} \\
& F^*(p(m)) = \arg\max_{|F| \le c} p_F(m)^\top \Gamma_{FF} p_F(m)
\end{aligned}
$$

---

### 3. Why Standard Knapsack & Greedy Algorithms Fail

In standard supervised machine learning, curriculum design is modeled as a knapsack problem: evaluate the marginal gain of training each feature, and allocate exposures greedily to features with the highest marginal return.

This standard approach completely collapses here due to three structural barriers:

#### 1. The Crowding-Out Barrier (The Need to Extinguish Zero-Utility Distractors)
Suppose biomarker $k$ is a pure distractor: it carries zero residual diagnostic value orthogonal to symptoms ($\Gamma_{k, :} d = 0$).
* A standard knapsack algorithm evaluates the direct clinical return of biomarker $k$: since $d_k = 0$, its marginal clinical payoff is 0. A greedy knapsack assigns $m_k = 0$.
* But suppose the clinician begins with an uncalibrated misconception on test $k$: $|p_k^{(0)}| = M \gg 0$.
* If the AI assigns $m_k = 0$, then at graduation, the clinician's belief remains unattenuated: $p_k(0) = M$.
* The perceived salience of distractor $k$ is $M^2 \Gamma_{kk}$, which is massive!
* At graduation, distractor $k$ enters the clinician's autonomous panel $F^*$, **crowding out and displacing a genuinely life-saving biomarker**!
* **The AI is forced to spend budget on distractors ($m_k > 0$) solely to extinguish their perceived salience**, even though distractors provide zero positive clinical value!

#### 2. Inner Combinatorial Discontinuity (Abrupt Panel Jumps)
The mapping from allocation $m$ to the chosen panel $F^*(p(m))$ is discontinuous. A single extra exposure $m_j \to m_j + 1$ can cause biomarker $j$'s salience to cross the threshold of another biomarker, causing an abrupt discrete jump in the chosen panel $F^*$.

#### 3. Correlated Cross-Coupling
Under general $\Gamma \succeq 0$, exposures to biomarker $i$ change $p_i(m_i)$, which simultaneously modifies the interaction term $2 p_i(m_i) p_j(m_j) \Gamma_{ij}$ for all correlated biomarkers $j$. You cannot evaluate the value of training biomarker $i$ in isolation!

---

### 4. Structural Decomposition: Candidate Champion Panels

Rather than searching the entire exponential space of exposure allocations $m \in \mathbb{N}^n$ simultaneously, we decompose the bilevel problem by **Candidate Champion Panels**.

Let $\mathcal{F}_c := \{F \subseteq [n] : |F| = c\}$ denote the collection of all candidate panels of size $c$.
Notice that $|\mathcal{F}_c| = \binom{n}{c}$. For typical clinical diagnostic budgets (e.g. $c \in \{1, 2, 3\}$ and $n \in \{10, \dots, 50\}$), $\binom{n}{c}$ is polynomial and modest in size:
* For $c = 1$: $\binom{n}{1} = n$ panels.
* For $c = 2$: $\binom{n}{2} = \frac{n(n-1)}{2}$ panels (e.g., $n=20 \implies 190$ panels).
* For $c = 3$: $\binom{n}{3} = \frac{n(n-1)(n-2)}{6}$ panels (e.g., $n=20 \implies 1,140$ panels).

For any candidate panel $F^* \in \mathcal{F}_c$, the AI asks two sequential questions:
1. **Realizability:** What is the minimum exposure vector $m \in \mathbb{N}^n$ required to make $F^*$ dominate every competing panel $F \neq F^*$?
2. **Optimality:** Subject to $F^*$ winning, how should surplus training exposures be allocated to maximize correlation $\rho(F^*, p(m))$?

---

### 5. The Correlated Dominance Condition

For a specific candidate panel $F^* \in \mathcal{F}_c$ to be the clinician's unique autonomous choice at graduation, its perceived panel salience must strictly exceed the salience of every competitor panel $F \in \mathcal{F}_c \setminus \{F^*\}$:
$$ p_{F^*}^\top \Gamma_{F^* F^*} p_{F^*} > p_F^\top \Gamma_{FF} p_F \qquad \forall F \in \mathcal{F}_c \setminus \{F^*\} $$

#### Exact Salience Difference Decomposition:
Let $F$ be any competing panel of size $c$. We partition $F^*$ and $F$ into:
* $A := F^* \cap F$ (the set of shared biomarkers, with $|A| = a < c$).
* $J := F^* \setminus F$ (the biomarkers unique to $F^*$, with $|J| = c - a \ge 1$).
* $K := F \setminus F^*$ (the biomarkers unique to competitor $F$, with $|K| = c - a \ge 1$).

We expand the salience difference $\Delta(F^*, F; m) := S(F^*; p(m)) - S(F; p(m))$:
$$ 
\begin{aligned}
\Delta(F^*, F; m) &= \Big( p_A^\top \Gamma_{AA} p_A + p_J^\top \Gamma_{JJ} p_J + 2 p_J^\top \Gamma_{JA} p_A \Big) - \Big( p_A^\top \Gamma_{AA} p_A + p_K^\top \Gamma_{KK} p_K + 2 p_K^\top \Gamma_{KA} p_A \Big) \\
&= \underbrace{\Big( p_J^\top \Gamma_{JJ} p_J + 2 p_J^\top \Gamma_{JA} p_A \Big)}_{\text{Advantage from Champion Features } J} \;-\; \underbrace{\Big( p_K^\top \Gamma_{KK} p_K + 2 p_K^\top \Gamma_{KA} p_A \Big)}_{\text{Challenge from Competitor Features } K}
\end{aligned}
$$
Notice the beauty of this algebraic expansion:
* **The shared variance $p_A^\top \Gamma_{AA} p_A$ cancels out completely!**
* Competitor $F$ only threatens champion $F^*$ through the features $K \subseteq [n] \setminus F^*$ that lie **outside** the champion panel.

---

### 6. Decoupling Budget: Champion Calibration vs. Competitor Extinguishment

For any candidate champion panel $F^*$, the exposure vector $m = (m_1, \dots, m_n)$ splits naturally into two disjoint sets:
1. **Champion Exposures ($m_{F^*} = \{m_j : j \in F^*\}$):**  
   These exposures calibrate the champion panel's weights toward truth ($p_{F^*} \to d_{F^*}$). This directly boosts the clinical covariance numerator $p_{F^*}^\top \Gamma_{F^*, :} d$ in $\rho(F^*, p)$.
2. **Competitor Extinguishment Exposures ($m_{F^{*c}} = \{m_k : k \notin F^*\}$):**  
   These exposures do not appear in the champion panel $F^*$. Their **sole purpose** is to shrink competitor beliefs $p_k(m_k)$ so that the challenge term $\big( p_K^\top \Gamma_{KK} p_K + 2 p_K^\top \Gamma_{KA} p_A \big)$ is suppressed below champion salience!

#### A. The Minimum Extinguishment Vector
Fix any candidate champion exposure subvector $m_{F^*} \in \mathbb{N}^c$ satisfying the sign-correction threshold ($m_j \ge \tau_j$ for all $j \in F^*$).
The champion panel achieves fixed salience:
$$ S^* := p_{F^*}(m_{F^*})^\top \Gamma_{F^* F^*} p_{F^*}(m_{F^*}) $$
To suppress every competitor panel $F \neq F^*$ below $S^*$, the outside exposures $m_{F^{*c}} \in \mathbb{N}^{n-c}$ must satisfy:
$$ p_F(m)^\top \Gamma_{FF} p_F(m) < S^* \qquad \forall F \in \mathcal{F}_c \setminus \{F^*\} $$

Because belief error $|p_k(m_k) - d_k| = \psi(m_k) |p_k^{(0)} - d_k|$ is monotonically non-increasing in $m_k$, competitor saliences shrink as outside exposures increase.
We define the **Minimum Extinguishment Cost** for champion panel $F^*$ under calibration $m_{F^*}$:
$$ \operatorname{Cost}_{\text{ext}}(F^*; m_{F^*}) := \min_{m_{F^{*c}} \in \mathbb{N}^{n-c}} \left\{ \sum_{k \notin F^*} m_k \quad \text{s.t.} \quad \max_{F \neq F^*} p_F(m)^\top \Gamma_{FF} p_F(m) < S^* \right\} $$

#### B. Exact Realizability Condition Under Budget $T$
A candidate champion panel $F^*$ can be made the clinician's autonomous choice at graduation under budget $T$ if and only if there exists a champion allocation $m_{F^*} \in \mathbb{N}^c$ such that:
$$ \sum_{j \in F^*} m_j + \operatorname{Cost}_{\text{ext}}(F^*; m_{F^*}) \le T $$

---

### 7. The Correlated Monotone Panel Sieve (C-MPS) Algorithm

We now present the complete, exact algorithm that solves the general correlated bilevel curriculum problem in polynomial time.

#### Algorithm Structure:
1. **Outer Loop over Panels:** Enumerate all $\binom{n}{c} = \mathcal{O}(n^c)$ candidate champion panels $F^* \in \mathcal{F}_c$.
2. **Inner Search over Champion Allocations:** For a fixed panel $F^*$, enumerate integer partitions of champion budget $\sum_{j \in F^*} m_j \le T$, which is $\binom{T + c}{c} = \mathcal{O}(T^c)$ allocations.
3. **Extinguishment Sieve:** For each champion allocation $m_{F^*}$:
   * Compute champion salience $S^* = p_{F^*}(m_{F^*})^\top \Gamma_{F^* F^*} p_{F^*}(m_{F^*})$.
   * Compute the minimum extinguishment cost $\kappa(F^*, m_{F^*})$ in $\mathcal{O}(T^c n^c)$ time.
   * If total cost $\sum_{j \in F^*} m_j + \kappa(F^*, m_{F^*}) \le T$:
     * The panel configuration is **realizable**.
     * Allocate surplus exposures $T - \text{Total Cost}$ greedily to the champion feature $j \in F^*$ maximizing marginal clinical correlation $\rho(F^*, p(m))$.
     * Evaluate the exact correlation $\rho(F^*, p(m^*))$.
4. **Global Maximizer:** Output the allocation vector $m^*$ achieving the highest clinical correlation.

```
Algorithm: Correlated Monotone Panel Sieve (C-MPS)
Input: Ground-truth d, initial beliefs p^(0), covariance matrices Sigma, Gamma, budget T, capacity c
Output: Globally optimal exposure allocation vector m* in N^n

1. Initialize best_rho = -infinity, best_m = zeros(n)
2. For each candidate champion panel F* in collection of all panels of size c:
3.    # Check if champion sign-correction is achievable
4.    tau_cost = sum_{j in F*} tau_j
5.    If tau_cost > T: continue   # Skip unrealizable panels
6.    
7.    For each champion budget allocation m_{F*} with m_j >= tau_j and sum_{j in F*} m_j <= T:
8.        Compute champion salience:
9.           S* = p_{F*}(m_{F*})^T Gamma_{F* F*} p_{F*}(m_{F*})
10.       
11.       # Determine minimum extinguishment exposures for outside features
12.       m_outside, cost_ext = compute_extinguishment(F*, S*, budget = T - sum(m_{F*}))
13.       If cost_ext is None: continue   # Unrealizable within budget
14.       
15.       total_cost = sum(m_{F*}) + cost_ext
16.       If total_cost <= T:
17.           surplus = T - total_cost
18.           m_cand = construct_full_vector(m_{F*}, m_outside)
19.           # Greedily distribute surplus budget to champion features
20.           m_cand[F*] += allocate_surplus(surplus, F*, Gamma, Sigma, d, p(m_cand))
21.           
22.           rho_cand = compute_correlation(F*, p(m_cand), Gamma, Sigma, d)
23.           If rho_cand > best_rho:
24.               best_rho = rho_cand
25.               best_m = m_cand
26.
27. Return best_m
```

---

### 8. Computational Complexity Analysis

Let us analyze the exact time complexity of the C-MPS algorithm:
* **Number of candidate panels:** $\binom{n}{c} \le \frac{n^c}{c!} = \mathcal{O}(n^c)$.
* **Champion exposure partitions:** The number of ways to allocate budget to $c$ features is $\binom{T + c}{c} \le \frac{(T+c)^c}{c!} = \mathcal{O}(T^c)$.
* **Extinguishment evaluation:** For each configuration, computing $\kappa(F^*, m_{F^*})$ takes $\mathcal{O}(T^c n^c)$ time.
* **Total Algorithm Runtime:**
  $$ \mathcal{O}\Big( n^c \cdot T^c \cdot T^c n^c \Big) = \mathcal{O}\Big( n^{2c} T^{2c} \Big) $$
  *(For single-test capacity $c = 1$, this collapses to $\mathcal{O}(n^2 T \log T)$ via decoupled 1D binary search).*

#### Why this is a foundational result:
1. **Polynomial for Fixed Capacity:** For any fixed clinical testing capacity $c$ (e.g. $c = 1, 2, 3$), the runtime is strictly polynomial in both $n$ and $T$.
2. **Guaranteed Global Optimality:** Solves the bilevel curriculum problem exactly without heuristic relaxations.
3. **Clinical Interpretability:** Decomposes the AI's teaching budget transparently into qualification, competitor extinguishment, and belief sharpening.
