# <G> Ex-Post c>=2 Strategy

This note formalizes the **Ex-Post, Multi-Feature ($c \ge 2$)** training strategy for binary actions ($K=2$), directly resolving the core questions posed in the project's running document.

---

## 1. Problem Setting

At round $t$, nature realizes state $x^{(t)} \sim \mathcal{N}(0, I_n)$.
The algorithm $\mathcal{A}$ observes $x^{(t)}$ and knows the true optimal action $a \in \{1, 2\}$ and suboptimal action $\bar{a}$.
$\mathcal{A}$ selects a subset $F_t \subseteq [n]$ of size $|F_t| \le c$ to show to $\mathcal{H}$.

The human predicts correctly ($y^* = \hat{y} = a$) iff:
$$\sum_{j \in F_t} \big(h_{aj}^{(t)} - h_{\bar{a}j}^{(t)}\big) x_j^{(t)} > 0$$

---

## 2. Realized Contribution Vectors & The Correct Set $C_t$

Define the **state-realized directional contribution** of each feature $j$:
$$v_j := \big(h_{aj}^{(t)} - h_{\bar{a}j}^{(t)}\big) x_j^{(t)}$$
* If $v_j > 0$, feature $j$ pushes the human toward the **correct** action on state $x^{(t)}$.
* If $v_j < 0$, feature $j$ pushes the human toward the **incorrect** action on state $x^{(t)}$.

A subset $F \subseteq [n]$ induces a correct decision iff $\sum_{j \in F} v_j > 0$.
We define the collection of correct subsets:
$$C_t := \left\{ F \subseteq [n] : |F| \le c, \ \sum_{j \in F} v_j > 0 \right\}$$

### $O(n \log n)$ Feasibility Algorithm: Is $C_t \neq \emptyset$?
To determine if zero regret is achievable on state $x^{(t)}$:
1. Compute contributions $v_1, \dots, v_n$.
2. Sort $v$ in descending order: $v_{(1)} \ge v_{(2)} \ge \dots \ge v_{(n)}$.
3. Sum the top $c$ values: $S_{\max} := \sum_{k=1}^c \max\{0, v_{(k)}\}$.
4. **Result**: $C_t \neq \emptyset$ if and only if $S_{\max} > 0$ and $v_{(1)} > 0$.

---

## 3. Algorithmic Decision Rules for $\mathcal{A}$

```
                      Feasibility Test: Is C_t empty?
                                    │
               ┌────────────────────┴────────────────────┐
               ▼                                         ▼
         C_t is EMPTY                              C_t is NON-EMPTY
               │                                         │
     Immediate regret is                       Zero regret is achievable
     FIXED at |d^T x| for                      by ANY subset F in C_t.
     EVERY feasible subset.                              │
               │                                         ▼
               ▼                               Solve Constrained Chaperone:
     Max-Learning Strategy:                    min_{F in C_t} sum_{j in F} \nu_j
     Show c features with                      (Opportunistic State Chaperoning)
     minimal future cost \nu_j                           │
     (Teach most divergent items)                        ▼
                                               Compare with stakes:
                                               Paid teaching if stakes |D|
                                               are smaller than gap!
```

---

## 4. Resolution of the Central Questions

### Question 1: What does $\mathcal{A}$ do when $|C_t| = 0$?
* **Situation**: No subset of size $\le c$ can produce a positive sum. The human will inevitably make a mistake and select $\bar{a}$.
* **Immediate Regret**: Exactly $|(u_a - u_{\bar{a}})^\top x^{(t)}| = |d^\top x^{(t)}|$, **invariant to the subset chosen**.
* **Optimal Action**: Since the immediate loss is fixed, $\mathcal{A}$ enjoys a "free pass" to maximize long-term learning:
  $$F_t^* = \arg\min_{F \subseteq [n], |F| \le c} \delta V(m + \mathbf{1}_F)$$
  Under coordinate-separable continuation approximations, $\mathcal{A}$ displays the **$c$ features with the lowest $\nu_j(m)$** (the most divergent unlearned features).

### Question 2: Which subset to show when $|C_t| \ge 1$?
* **Situation**: Multiple subsets achieve zero immediate regret.
* **Optimal Action**: $\mathcal{A}$ solves the **Opportunistic State-Level Chaperone Problem**:
  $$\min_{F \subseteq [n], |F| \le c} \sum_{j \in F} \nu_j(m) \quad \text{subject to} \quad \sum_{j \in F} v_j > 0$$
* **Mechanics**:
  * If a high-priority unlearned feature $j^*$ has $v_{j^*} < 0$ on this state, $\mathcal{A}$ includes one or more helper features $k$ with $v_k > |v_{j^*}|$.
  * This guarantees $\sum_{j \in F} v_j > 0$, ensuring **zero immediate regret while forcing the human to observe and learn $j^*$**.
* **Stakes Comparison**: If stakes $|d^\top x^{(t)}|$ are smaller than the continuation gap between the best infeasible set and the best feasible set:
  $$|d^\top x^{(t)}| < \min_{F \in C_t} \sum_{j \in F} \nu_j - \min_{F \notin C_t} \sum_{j \in F} \nu_j$$
  $\mathcal{A}$ deliberately pays the small regret to teach an even more valuable combination.
