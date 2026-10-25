# <G> Multi-Feature Chaperoning (c >= 2)

This note develops the theory of **Multi-Feature Chaperoning** when the AI assistant can show subsets of features of size $|F_t| \le c$ with $c \ge 2$ under ex-ante presentation.

---

## 1. The Chaperone Concept

When $c = 1$, showing a bad feature ($g_j^{(0)} d_j < 0$) inevitably forces the human to make a decision based on an inverted signal, incurring high regret.
When $c \ge 2$, the algorithm can pair a bad feature with an **anchor or chaperone feature**—a feature the human understands well or over-weights in the correct direction—so that the linear combination:
$$\sum_{j \in F_t} g_j^{(t)} x_j$$
remains positively correlated with the true utility margin $d^\top x$, **protecting decision quality while teaching the bad feature**.

---

## 2. The Chaperone Inequality

Assume one-shot learning ($\psi(m) = \mathbf{1}[m = 0]$) and $K = 2$.
Let the human's epistemic state be the set of taught features $S_t \subseteq [n]$ ($g_j = d_j$ for $j \in S_t$, and $g_j = g_j^{(0)}$ for $j \notin S_t$).

### Theorem: The Chaperone Inequality
Let $F^*$ be a candidate set of $c$ features all believed with the wrong sign ($g_j^{(0)} d_j < 0$ for all $j \in F^*$).
Let $k \notin F^*$ be a chaperone feature with $d_k > 0$ and strong positive belief $g_k^{(0)} = G > 0$.
For any sub-collection $F' \subset F^*$ with $|F'| = c - 1$:

$$\text{If } \quad G \cdot d_k > -(g_{F'}^{(0)})^\top d_{F'},$$

then presenting the composite set $\{k\} \cup F'$ at $t=0$ achieves **positive correlation**:
$$r(\{k\} \cup F', g^{(0)}) > 0 > r(F^*, g^{(0)})$$
This strictly lowers round-0 regret while successfully teaching all features in $F'$.

#### Proof:
By definition of the correlation ratio:
$$r(\{k\} \cup F', g^{(0)}) \propto g_{\{k\} \cup F'}^{(0)\top} d_{\{k\} \cup F'} = G d_k + (g_{F'}^{(0)})^\top d_{F'} > 0$$
In contrast, $r(F^*, g^{(0)}) \propto (g_{F^*}^{(0)})^\top d_{F^*} < 0$ because every component term is negative.
Since one-round regret $\Lambda_F(g) = \frac{\|d\|}{\sqrt{2\pi}}(1 - r)$ is strictly decreasing in $r$, the chaperoned set achieves strictly lower regret.

### Consumption of Chaperones:
Once the chaperone feature $k$ is displayed, it is learned ($g_k \to d_k$). If its initial leverage came from being over-weighted ($G \gg d_k$), its chaperone capacity is **consumed** after one exposure.

#### Verified Numerical Example ($n=3, c=2$):
* Parameters: $d = (3, 3, 0.2)$, $g^{(0)} = (-1, -1, 50)$.
  * Feature 0 and 1 are target high-importance features ($d_0 = d_1 = 3$), but have wrong signs ($g_0 = g_1 = -1$).
  * Feature 2 is a weak feature ($d_2 = 0.2$), but the human has a massive positive belief ($g_2 = 50$).
* **Round-0 Regret**:
  * Unchaperoned set $\{0, 1\}$: Correlation $r = -0.999$, Regret $= 3.387$.
  * Chaperoned set $\{0, 2\}$: Correlation $r = +0.033$, Regret $= 1.639$ (cuts regret by more than half!).
* **Optimal Trajectory**:
  $$\{0, 2\} \longrightarrow \{0, 1\} \longrightarrow \{0, 1\} \longrightarrow \dots$$
  Feature 2 chaperones the learning of feature 0 in round 0; in round 1, feature 0 is now learned and chaperones feature 1, reaching the optimal stationary set $F^* = \{0, 1\}$ in transient length $L = 2$.

---

## 3. Transient Bounds & Stationarity

Under one-shot learning, an optimal schedule consists of a **transient phase** (in which the taught set $S_t$ strictly grows), followed by an infinite **stationary phase** where a fixed set $F_{\text{final}} \subseteq S_{\text{final}}$ is displayed forever.

### Theorem: Exact Transient Bound $L \le c$
Every optimal schedule reaches its stationary set $F_{\text{final}}$ in at most:
$$L \le |F_{\text{final}}| \le c \quad \text{transient rounds}$$

#### Proof Sketch:
* Once a taught set $S_L$ is formed, the optimal stationary suffix chooses the $c$ features in $S_L$ with the largest $d_j^2$.
* If an optimal schedule spent a transient round $t$ teaching an auxiliary feature $k \notin F_{\text{final}}$ solely to use $k$ as a learned chaperone in a later round $s$, one could have instead taught an element of $F_{\text{final}}$ directly, since every element $j^* \in F_{\text{final}}$ satisfies $d_{j^*}^2 \ge d_k^2$.
* Thus, in any minimal optimal trajectory, every transient round must strictly increase the count of learned target features $|S_t \cap F_{\text{final}}|$, forcing $L \le |F_{\text{final}}| \le c$.

---

## 4. Empirical Behavior & The Length $\le 2$ Phenomenon

In extensive Monte Carlo evaluations across 1,500 random instances ($n \in \{4, 5, 6\}$, $c \in \{2, 3\}$, $\delta \in \{0.5, 0.8, 0.95, 0.99\}$):

| Budget $c$ | Length $L = 0$ (Stationary) | Length $L = 1$ | Length $L = 2$ | Length $L \ge 3$ |
| :---: | :---: | :---: | :---: | :---: |
| **$c = 2$** | 554 instances | 201 instances | 10 instances | **0** |
| **$c = 3$** | 512 instances | 213 instances | 10 instances | **0** |

* Across all 1,500 instances, the optimal transient length **never exceeded 2**, even when $c = 3$.
* This motivates the open conjecture: Is the optimal transient length bounded by $\min\{c, 2\}$ under generic conditions? (See [[<G> Open Hypotheses & Conjectures]]).
