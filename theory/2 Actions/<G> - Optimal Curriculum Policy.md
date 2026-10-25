# <G> - Optimal Curriculum Policy: The Kingmaker Theorem & Dynamic Teaching

This document formally solves the AI preceptor's optimal training policy $\pi^*$ for the binary clinical setting ($K=2$).

We formulate the bilevel curriculum problem, solve the $c=1$ capacity case in closed form via the **Kingmaker Theorem**, establish the combinatorial knapsack structure for general $c$, and derive the dynamic state-dependent ex-post Bellman policy.

---

## 1. The Bilevel Curriculum Allocation Problem

Let $T \in \mathbb{N}$ be the total training exposure budget. The AI must choose exposure allocations $m = (m_1, \dots, m_n) \in \mathbb{N}^n$ with $\sum_{j=1}^n m_j = T$ to minimize graduation regret.

Recalling that minimizing $\mathcal{L}_{\text{grad}}(g(m))$ is equivalent to maximizing correlation $\rho(\hat{F}(g(m)), g(m))$, the AI's optimization problem is:

$$\max_{m \in \mathbb{N}^n} \; \rho(m) := \frac{\sum_{j \in \hat{F}(g(m))} d_j g_j(m_j)}{\|d\|_2 \sqrt{\sum_{j \in \hat{F}(g(m))} g_j(m_j)^2}}$$
$$\text{subject to: } \quad \sum_{j=1}^n m_j = T$$
$$\hat{F}(g(m)) = \arg\max_{|F| \le c} \sum_{j \in F} g_j(m_j)^2$$
$$g_j(m_j) = \big(1 - \psi(m_j)\big) d_j + \psi(m_j) g_j^{(0)}$$

This is a **bilevel combinatorial program**:
* **Outer Level (The AI):** Allocates the training budget $m$ to shape the clinician's future mental model.
* **Inner Level (The Graduated Clinician):** Autonomously selects the top-$c$ test panel $\hat{F}$ based on subjective salience $g_j(m_j)^2$.

---

## 2. Solving the Single-Test Capacity Setting ($c = 1$): The Kingmaker Theorem

When $c = 1$, the graduated clinician autonomously selects a single test:
$$j^* = \arg\max_{j \in [n]} g_j(m_j)^2$$
For any single feature $j$, the magnitude $|g_j|$ cancels out of the correlation:
$$\rho(\{j\}, g(m)) = \operatorname{sgn}\big(d_j g_j(m_j)\big) \frac{|d_j|}{\|d\|_2}$$

Thus, if feature $j$ is chosen and has correct sign, it yields clinical correlation $\frac{|d_j|}{\|d\|_2}$. If it has the wrong sign, it yields $-\frac{|d_j|}{\|d\|_2}$. If $d_j = 0$, it yields $0$.

### The Kingmaker Feasibility Conditions
For a target feature $j \in [n]$ with $d_j \neq 0$ to be the **autonomous champion** at graduation, the AI must ensure:
1. **Sign Flip:** The belief sign must match the truth:
   $$m_j \ge \tau_j := \min\left\{ m : \psi(m) < \frac{|d_j|}{|d_j| + |g_j^{(0)}|} \right\} \quad (\tau_j = 0 \text{ if initially correct})$$
2. **Autonomy Dominance (Winning the Salience Race):** Feature $j$ must have strictly greater subjective salience than every other feature $k \neq j$:
   $$g_j(m_j)^2 > g_k(m_k)^2 \quad \forall k \neq j$$

For every other feature $k \neq j$, if its unattenuated initial salience $|g_k^{(0)}| \ge |g_j(m_j)|$, the AI **must allocate exposures to feature $k$** to extinguish it below $|g_j(m_j)|$!

### Definition: Extinguishment Threshold
For target salience $\theta = |g_j(m_j)|$, the minimum exposures required to suppress feature $k$ below $\theta$ is:
$$\kappa_k(\theta) := \min\left\{ m \in \mathbb{N} : |g_k(m)| < \theta \right\}$$
(and $\kappa_k(\theta) = 0$ if $|g_k^{(0)}| < \theta$).

### Theorem: The Kingmaker Curriculum Cost Function
For any candidate feature $j \in [n]$ and target allocation $m_j \ge \tau_j$, the minimum total training budget required to make feature $j$ the autonomous champion is:
$$\text{Cost}(j, m_j) := m_j + \sum_{k \neq j} \kappa_k\big(|g_j(m_j)|\big)$$
A feature $j$ is **achievable as an autonomous champion** under budget $T$ if and only if:
$$\min_{m_j \ge \tau_j} \text{Cost}(j, m_j) \le T$$

---

## 3. The Kingmaker Algorithm ($O(n^2 \log T)$ Exact Solution for $c=1$)

```
Algorithm: Kingmaker Optimal Curriculum (c = 1)
Input: True difference d, initial beliefs g^(0), contraction psi, budget T
Output: Optimal champion j*, optimal training allocation m*

1. BestCorrelation = -infinity; BestAllocation = None; BestChampion = None
2. For each candidate feature j in {1, ..., n} with d_j != 0:
3.    For each candidate allocation m_j in {tau_j, ..., T}:
4.        theta = |(1 - psi(m_j)) d_j + psi(m_j) g_j^(0)|
5.        If theta == 0: continue
6.        TotalCost = m_j
7.        Allocation = {j: m_j}
8.        For each other feature k != j:
9.            Find smallest m_k >= 0 such that |(1 - psi(m_k)) d_k + psi(m_k) g_k^(0)| < theta
10.           TotalCost += m_k
11.           Allocation[k] = m_k
12.       If TotalCost <= T:
13.           CandidateCorrelation = |d_j| / ||d||_2
14.           If CandidateCorrelation > BestCorrelation:
15.               BestCorrelation = CandidateCorrelation
16.               BestChampion = j
17.               Allocate remaining (T - TotalCost) exposures to m_j
18.               BestAllocation = Allocation
19. Return BestChampion, BestAllocation, BestCorrelation
```

> [!TIP] Complexity & Optimality
> The Kingmaker Algorithm evaluates at most $n$ candidate features across at most $T$ allocations. For each pair, finding $\kappa_k(\theta)$ takes $O(\log T)$ via binary search (since $|g_k(m)|$ contracts monotonically for distractions). Total runtime is **$O(n^2 \log T)$**, yielding the globally optimal curriculum in milliseconds.

---

## 4. General Test Capacity ($c \ge 2$): The Chaperoned Knapsack

When $c \ge 2$, the AI can select a **panel of champions** $\mathcal{C}^* \subset [n]$ with $|\mathcal{C}^*| = c$.
The optimization problem requires:
1. Flipping the signs of all $j \in \mathcal{C}^*$ ($m_j \ge \tau_j$).
2. Ensuring that $\min_{j \in \mathcal{C}^*} g_j(m_j)^2 > \max_{k \notin \mathcal{C}^*} g_k(m_k)^2$ (panel dominance).
3. Maximizing the panel cosine similarity:
   $$\max_{\mathcal{C} \subset [n], |\mathcal{C}|=c} \frac{\sum_{j \in \mathcal{C}} d_j g_j(m_j)}{\sqrt{\sum_{j \in \mathcal{C}} g_j(m_j)^2}}$$

### Greedy Salience Water-Filling (Approximation for Large $n$):
1. **Target Selection:** Identify the top-$c$ features with largest $d_j^2$.
2. **Sign Rectification:** Spend $\sum_{j \in \mathcal{C}^*} \tau_j$ to flip their signs.
3. **Distraction Squashing:** For any distracting feature $k \notin \mathcal{C}^*$ with salience $g_k^2 > \min_{j \in \mathcal{C}^*} g_j^2$, greedily allocate exposures to feature $k$ until its salience falls below the panel threshold.

---

## 5. Dynamic Ex-Post Policy During Training

During the training phase ($t \le T$), if the AI observes the realized state $x^{(t)}$ before selecting the training panel $F_t$:

### Zero-Regret Teaching via Sheppard's Arcsine Law
Showing feature $j$ incurs zero immediate regret on patient $x^{(t)}$ if and only if:
$$\operatorname{sgn}(g_j x_j^{(t)}) = \operatorname{sgn}(d^\top x^{(t)})$$
By Sheppard's Arcsine Law, even an uncalibrated / wrong-sign feature provides a **zero-regret teaching opportunity** with probability:
$$P(\text{Zero-Regret Teaching}) = \frac{1}{2} - \frac{1}{\pi} \arcsin\left( \frac{|d_j|}{\|d\|_2} \right) > 0$$

### The Bellman Stakes Threshold Policy
Let $V(m)$ be the optimal cost-to-go from count vector $m$, and let $\Delta V_j(m) = \delta V(m + e_j) - \delta V(m)$.
In round $t$ with patient stakes $|D^{(t)}| = |d^\top x^{(t)}|$:
* **Opportunistic Free Teaching:** If the target teaching feature belongs to the zero-regret set $Z(x^{(t)})$, teach it immediately at **zero regret**, regardless of how high the stakes are.
* **Paid Low-Stakes Teaching:** If the target feature does not belong to $Z(x^{(t)})$, teach it if and only if the patient stakes are smaller than the future curriculum value gain:
  $$|D^{(t)}| < \Delta V_{\text{safe}} - \Delta V_{\text{target}}$$
