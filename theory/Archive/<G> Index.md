# <G> Project Map

Welcome to the **CS 598 Project Vault on Pedagogical AI Assistance**.

This repository organizes the theoretical foundations, algorithms, and proofs for designing AI assistants that actively educate human decision-makers across repeated interactions.

---

## 🗺️ Vault Structure

### 📌 Core Problem & Literature
* [[CS 598 Project/Archive/Problem Description]]: The foundational mathematical problem statement, two-phase interaction (Training $\to$ Graduation), and open questions.
* [[<G> Related Work]]: Comprehensive mapping to Noti et al. (EC 2025), Guan et al. (2026), Machine Teaching, CIRL / Assistive Bandits, Bayesian Persuasion, and Cognitive Scaffolding.

---

### ♟️ 2 Actions Setting ($K = 2$)
* [[<G> 2 Actions Overview]]: Coordinate reduction to difference vectors $d = u_1 - u_2$ and $g = h_1 - h_2$, and the closed-form one-round ex-ante expected regret $\Lambda_F(g)$.

#### 📚 Training Phase (`2 Actions/Training/`)
* [[<G> Training Overview]]: Conceptual guide to the $2 \times 2$ information and capacity matrix.
* [[<G> Ex-Ante c=1 Strategy]]: Two-state discrete regret, Gittins block index, permanent crutch threshold.
* [[<G> Ex-Ante c>=2 Strategy]]: Angular correlation, combinatorial chaperoning, transient bound $L \le c$.
* [[<G> Ex-Post c=1 Strategy]]: State observation, Sheppard's arcsine law ($2^{1-n}$ zero-regret availability), Bellman threshold policy on stakes $|D|$, and value monotonicity.
* [[<G> Ex-Post c>=2 Strategy]]: Realized contributions $v_j$, feasibility of $C_t$, max-learning when $C_t = \emptyset$, and opportunistic state-level chaperoning when $C_t \neq \emptyset$.
* [[<G> Action Menu Selection and Scaffolding]]: The tradeoffs between forced actions ($|A_t| = 1$) and full menu choice ($|A_t| = 2$); asymmetric row debiasing and high-stakes error shielding.

#### 🎓 Graduation Phase (`2 Actions/Graduation/`)
* [[<G> Graduation Overview]]: Models of graduation (Unassisted G1, Random single-feature G2, Parameter MSE G3).
* [[<G> Curriculum Allocation (Knapsack Equivalence)]]: 0–1 Knapsack equivalence under random graduation, FPTAS, and extinguishing loud distractions in unassisted graduation.
* [[<G> Contiguous Block Scheduling]]: Proving contiguous block presentation is optimal; sorting blocks via the Gittins-like block priority index $\nu_j(m_j)$.
* [[<G> Parameter MSE vs Decision Regret]]: Proof of fundamental divergence—why optimizing parameter MSE can perform worse than teaching nothing.

#### 🔬 Open Directions (`2 Actions/`)
* [[<G> Open Hypotheses & Conjectures]]: Open questions regarding global value monotonicity, transient length $\le 2$, price of stationarity, and multi-action scaffolding ($K \ge 3$).
