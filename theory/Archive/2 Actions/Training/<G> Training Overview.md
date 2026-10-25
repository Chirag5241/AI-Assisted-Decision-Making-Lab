# <G> Training Overview

During the **Training Phase** ($t = 0, \dots, T-1$), nature draws environment states $x^{(t)} \sim \mathcal{N}(0, I_n)$, and the algorithm $\mathcal{A}$ dynamically guides human decision-making by choosing:
1. An **action menu** $A_t \subseteq [K]$ ($a_{\min} \le |A_t| \le a_{\max}$)
2. A **feature subset** $F_t \subseteq [n]$ ($|F_t| \le c$)

The objective during training is to balance immediate decision regret with long-term human learning so that the human reaches maximal competency upon graduation.

---

## 🗺️ The $2 \times 2$ Framework of Feature Selection

The problem of feature selection naturally decomposes along two structural dimensions:
1. **Information Timing**: **Ex-Ante** (chosen before $x^{(t)}$ is realized) vs. **Ex-Post** (chosen after $x^{(t)}$ is observed).
2. **Cognitive Capacity**: **$c = 1$** (single feature) vs. **$c \ge 2$** (multi-feature combinations).

```
                          Information Timing
                     Ex-Ante                      Ex-Post
              (Before x is drawn)           (After x is drawn)
           ┌─────────────────────────────┬─────────────────────────────┐
           │ [[<G> Ex-Ante c=1 Strategy]]│ [[<G> Ex-Post c=1 Strategy]]│
           │ • Two-state arm regret      │ • Sheppard's Arcsine Law    │
  c = 1    │ • Gittins / Knapsack index  │ • Bellman Threshold Rule    │
           │ • Crutch vs. Teach tradeoff │ • Zero-regret in 1-2^(1-n)  │
Capacity   ├─────────────────────────────┼─────────────────────────────┤
           │ [[<G> Ex-Ante c>=2 Strategy]│ [[<G> Ex-Post c>=2 Strategy]│
  c >= 2   │ • Combinatorial Chaperoning │ • C_t empty vs non-empty    │
           │ • Correlation buffering     │ • State-dependent Chaperone │
           │ • Transient length <= c     │ • Max-Learning on mistakes  │
           └─────────────────────────────┴─────────────────────────────┘
```

---

## 📑 Detailed Breakdown of Modules

### 1. [[<G> Ex-Ante c=1 Strategy]]
* Decision regret collapses to only two values ($a_j$ if sign is correct, $b_j$ if sign is inverted).
* Features are taught in contiguous blocks ordered by a Gittins-like block index.
* Resolves the **Permanent Crutch Threshold**: when to show a safe known feature forever vs. when to invest in teaching an unlearned feature.

### 2. [[<G> Ex-Ante c>=2 Strategy]]
* Decision regret depends non-linearly on the angular cosine correlation $r(F, g)$.
* Introduces the **Chaperone Inequality**: using strong, accurate features to buffer the negative impact of unlearned features, achieving positive correlation while teaching.
* Proves that any optimal trajectory reaches its stationary set in at most $L \le c$ transient rounds.

### 3. [[<G> Ex-Post c=1 Strategy]]
* The AI observes $x^{(t)}$ before selecting the feature.
* **Sheppard's Arcsine Law**: Bad features provide free zero-regret teaching opportunities with probability $\frac{1}{2} - \frac{1}{\pi}\arcsin\frac{|d_j|}{\|d\|}$. Even when all features have the wrong sign, a zero-regret feature exists with probability $1 - 2^{1-n}$.
* **Bellman Threshold Policy**: Characterizes free teaching at any stakes when $Z(x) \neq \emptyset$, and max-learning debiasing when $Z(x) = \emptyset$.

### 4. [[<G> Ex-Post c>=2 Strategy]]
* Directly resolves the core dilemmas from `project.tex`:
  * **When $C_t = \emptyset$ (Zero-regret impossible)**: Immediate regret is fixed at $|d^\top x|$ for every feasible subset; $\mathcal{A}$ has a "free pass" to select the $c$ features with maximum learning value.
  * **When $C_t \neq \emptyset$ (Zero-regret possible)**: $\mathcal{A}$ solves the **Constrained Chaperone Problem**, dynamically pairing unlearned features with positive contribution features on this specific state.

### 5. [[<G> Action Menu Selection and Scaffolding]]
* Analyzes the tradeoff between forced actions ($|A_t| = 1$) and full menu choice ($|A_t| = 2$).
* Proves why showing only the answer fails as a global strategy (starves unselected rows of learning, crippling unassisted graduation).
* Establishes when action menu restriction is strictly optimal: **asymmetric row debiasing** and **high-stakes error shielding**.
