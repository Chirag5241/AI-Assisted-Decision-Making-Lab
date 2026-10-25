# <G> - Kingmaker & Curriculum Allocation: Monotone Panel Sieve Algorithm

This document establishes the optimal curriculum allocation algorithm for the binary decision setting ($K = 2$) across arbitrary test capacity $c \ge 1$. 

Rather than relying on standard heuristic relaxations or generic knapsack reductions, we develop the **Monotone Panel Sieve (MPS)** algorithm, which exploits the exact bilevel geometry of autonomous test selection.

---

## 1. The Bilevel Curriculum Optimization Problem

Let $T \in \mathbb{N}$ be the total exposure budget. The AI preceptor chooses exposure counts $m = (m_1, \dots, m_n) \in \mathbb{N}^n$ with $\sum_{j=1}^n m_j = T$.
Under exposure vector $m$, the clinician's calibrated belief vector is:
$$p_j(m_j) = \big(1 - \psi(m_j)\big) d_j + \psi(m_j) p_j^{(0)}$$

At graduation, the clinician autonomously orders the panel $\hat{F}(p(m))$ maximizing perceived salience:
$$\hat{F}(p(m)) = \arg\max_{|F| \le c} \sum_{j \in F} p_j(m_j)^2 \sigma_j^2$$
The true expected clinical correlation achieved at graduation is:
$$\rho(m) = \frac{\sum_{j \in \hat{F}(p(m))} d_j p_j(m_j)}{\|d\|_2 \sqrt{\sum_{j \in \hat{F}(p(m))} p_j(m_j)^2}}$$

The AI solves the bilevel optimization problem:
$$\max_{m \in \mathbb{N}^n} \rho(m) \quad \text{subject to: } \quad \sum_{j=1}^n m_j = T, \quad \hat{F}(p(m)) = \arg\max_{|F| \le c} \sum_{j \in F} p_j(m_j)^2 \sigma_j^2$$

---

## 2. Why Generic Knapsack / Greedy Algorithms Fail

Generic 0–1 Knapsack and greedy water-filling fail on this problem for three structural reasons:
1. **Inner-Level Combinatorial Selection**: A marginal increase $m_j \to m_j + 1$ does not simply improve feature $j$; it can abruptly cause feature $j$ to enter $\hat{F}$ or displace another feature from $\hat{F}$.
2. **Non-Separable Ratio Objective**: The objective $\rho(m)$ is a ratio of sums. Maximizing the numerator independently can inflate the denominator, reducing overall correlation.
3. **The Crowding-Out Barrier**: A distraction feature $k$ ($d_k = 0$) has zero marginal value in a standard knapsack, yet spending budget on $k$ is **strictly necessary** to prevent it from stealing a slot in $\hat{F}$.

---

## 3. Structural Geometry: The Monotone Saliency Separation

For any target panel $F^* \subseteq [n]$ with $|F^*| = c$ to be autonomously chosen by the graduated clinician, there must exist a **salience separation threshold** $\theta \ge 0$ such that:
$$\min_{j \in F^*} p_j(m_j)^2 \sigma_j^2 \ge \theta > \max_{k \notin F^*} p_k(m_k)^2 \sigma_k^2$$

### Definition: Minimum Extinguishment Function
For any feature $k \in [n]$ and target threshold $\theta \ge 0$, define the minimum exposure required to suppress feature $k$ below $\theta$:
$$\kappa_k(\theta) := \min\Big\{ m \in \mathbb{N} : p_k(m)^2 \sigma_k^2 < \theta \Big\}$$
Note that if $p_k^{(0)2} \sigma_k^2 < \theta$, then $\kappa_k(\theta) = 0$. Because $|p_k(m)|$ contracts monotonically for distractions ($d_k = 0$), $\kappa_k(\theta)$ is monotonically non-increasing in $\theta$.

### Definition: Minimum Qualification Function
For any candidate champion feature $j \in [n]$ and target threshold $\theta \ge 0$, define the minimum exposure required for feature $j$ to achieve correct sign AND exceed threshold $\theta$:
$$\omega_j(\theta) := \min\Big\{ m \ge \tau_j : p_j(m)^2 \sigma_j^2 \ge \theta \Big\}$$
(where $\omega_j(\theta) = \infty$ if feature $j$ cannot attain salience $\theta$ for any $m$).

---

## 4. The Monotone Panel Sieve (MPS) Algorithm

\begin{theorem}[Exact Panel Realizability]
A candidate panel $F^* \subseteq [n]$ ($|F^*| = c$) can be made the clinician's autonomous choice at graduation if and only if there exists a threshold $\theta > 0$ such that the total panel qualification and extinguishment cost satisfies:
$$\text{Cost}(F^*, \theta) := \sum_{j \in F^*} \omega_j(\theta) + \sum_{k \notin F^*} \kappa_k(\theta) \le T$$
\end{theorem}


### The MPS Algorithm Description
Rather than searching the exponential space of all allocation vectors $m \in \mathbb{N}^n$:
1. **Critical Threshold Discretization**: The continuous threshold $\theta$ only needs to be evaluated at the discrete values $\{\theta_{j, m} = p_j(m)^2 \sigma_j^2 : j \in [n], m \in \{0, \dots, T\}\}$. There are at most $n(T+1)$ critical thresholds.
2. **Threshold Sieve**: For each critical threshold $\theta$:
   - Compute $\omega_j(\theta)$ for all $j \in [n]$ and $\kappa_k(\theta)$ for all $k \in [n]$.
   - Filter the candidate features with $\omega_j(\theta) < \infty$.
   - Select the optimal panel $F^*$ of size $c$ that maximizes $\frac{\sum_{j \in F^*} d_j p_j(\omega_j(\theta))}{\sqrt{\sum_{j \in F^*} p_j(\omega_j(\theta))^2}}$ subject to $\text{Cost}(F^*, \theta) \le T$.
3. **Surplus Allocation**: Any leftover exposures $T - \text{Cost}(F^*, \theta)$ are greedily assigned to features in $F^*$ with the largest marginal alignment gain $\frac{\partial \rho}{\partial m_j}$.

### Complexity
For fixed $c$, sorting and sieving across $O(nT)$ critical thresholds takes **$O(n^{c+1} T \log T)$** time, finding the globally optimal curriculum without relying on generic heuristics.
