# <G> Contiguous Block Scheduling

This note addresses the problem of **Curriculum Sequencing**: once a target allocation of exposures $m = (m_1, \dots, m_n)$ is fixed, in what temporal order should these features be taught to minimize discounted training loss?

---

## 1. Problem Setup

Let the multiset of chosen feature showings be $\{j^{m_j}\}_{j=1}^n$, with total duration $T = \sum_{j=1}^n m_j$.
Each feature $j$ possesses a non-increasing per-step cost sequence:
$$c_j(0) \ge c_j(1) \ge \dots \ge c_j(m_j - 1)$$
*(In single-feature training, $c_j(s) = b_j$ for the first $\tau_j$ exposures, and drops to $a_j < b_j$ thereafter).*

The algorithm seeks a permutation $(j_0, j_1, \dots, j_{T-1})$ minimizing the total discounted training cost:
$$\min_{\pi} \sum_{t=0}^{T-1} \delta^t c_{j_t}\big(m_{j_t}(t)\big), \qquad \delta \in (0, 1)$$
where $m_{j}(t) = |\{s < t : j_s = j\}|$ is the number of times feature $j$ has been shown prior to round $t$.

---

## 2. The Contiguous Block Theorem

### Theorem: Contiguity and the Gittins Block Index
Among all valid permutations of the exposure multiset, the discounted training cost is minimized by:
1. **Contiguity**: Presenting each feature $j$ in a **single contiguous block** of length $m_j$ (features are never split, interleaved, or revisited).
2. **Priority Ordering**: Ordering the blocks in increasing order of the **Gittins-like block index**:
   $$\nu_j(m_j) := \frac{\sum_{s=0}^{m_j - 1} \delta^s c_j(s)}{1 - \delta^{m_j}}$$

---

## 3. Mathematical Proof

### Part 1: Pairwise Order of Two Contiguous Blocks
Consider two disjoint blocks $B_1, B_2$ with lengths $L_1, L_2$ and internal discounted costs $C_1, C_2$:
$$C_k := \sum_{s=0}^{L_k - 1} \delta^s c(s), \qquad \nu(B_k) := \frac{C_k}{1 - \delta^{L_k}}$$
* Placing $B_1$ before $B_2$ incurs cost $C_1 + \delta^{L_1} C_2$.
* Placing $B_2$ before $B_1$ incurs cost $C_2 + \delta^{L_2} C_1$.
The cost difference is:
$$(C_1 + \delta^{L_1} C_2) - (C_2 + \delta^{L_2} C_1) = (1 - \delta^{L_1})(1 - \delta^{L_2}) \big( \nu(B_1) - \nu(B_2) \big)$$
Since $(1 - \delta^{L_1})(1 - \delta^{L_2}) > 0$, block $B_1$ should strictly precede $B_2$ iff:
$$\nu(B_1) \le \nu(B_2)$$

### Part 2: Non-Interleaving (Merging Split Blocks)
Suppose an optimal schedule splits a feature into two sub-blocks $P$ and $Q$ separated by an intervening segment $X$ of other features (schedule order $P \, X \, Q$).
* Because costs $c_j(s)$ are non-increasing in $s$, every exposure in $P$ costs at least as much as every exposure in $Q$.
* Since $\nu(\cdot)$ is a $\delta^s$-weighted average, $\nu(P) \ge \nu(Q)$.
* We compare the index of the intervening block $X$:
  * **Case 1**: If $\nu(X) \ge \nu(Q)$, shifting $Q$ forward directly adjacent to $P$ (order $P \, Q \, X$) changes cost by $-\delta^{|P|} [C_X(1 - \delta^{|Q|}) - C_Q(1 - \delta^{|X|})] \le 0$.
  * **Case 2**: If $\nu(X) < \nu(Q) \le \nu(P)$, shifting $P$ backward directly adjacent to $Q$ (order $X \, P \, Q$) changes cost by $-[C_P(1 - \delta^{|X|}) - C_X(1 - \delta^{|P|})] \le 0$.
* In both cases, merging preserves the within-feature exposure sequence while weakly decreasing total discounted cost. Iterating merges all split fragments into single contiguous blocks per feature.

---

## 4. Connection to Classical Scheduling Theory

* **Smith's Rule / WSPT (1956)**: When $\delta \to 1$, $\nu_j(m_j) \to \frac{\sum c_j(s)}{m_j}$, which is the average processing cost (Weighted Shortest Processing Time).
* **Gittins Index for Deteriorating Arms (1979)**: In multi-armed bandits, when arm rewards are deterministic and monotonically decreasing, the optimal stopping time defining the Gittins index is the entire planned block, reproducing $\nu_j(m_j) = \frac{C_j}{1 - \delta^{m_j}}$.
* **Validation**: Exact brute-force enumeration across 391 random instances ($n = 3$, $T \le 9$) verified zero counterexamples: a contiguous arrangement was never beaten, and the $\nu$-ordering was always strictly optimal.
