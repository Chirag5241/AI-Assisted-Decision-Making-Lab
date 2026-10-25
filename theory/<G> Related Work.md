# <G> Related Work

This document maps the theoretical landscape surrounding **Pedagogical AI Assistance**, situating the current project relative to foundational literature, adjacent fields in machine learning, economics, and cognitive science, and identifying exact mathematical reductions.

---

## 1. Direct Predecessors

### [1] AI-Assisted Decision Making with Human Learning
* **Authors**: Gali Noti, Kate Donahue, Jon Kleinberg, Sigal Oren (ACM EC 2025; arXiv:2502.13062)
* **Model**:
  * Continuous scalar target: $y = c + \sum_{i=1}^n a_i x_i$, with independent standardized features $x_i \sim (0, 1)$.
  * At each round, the algorithm chooses feature subset $A_t \subseteq [n]$ with $|A_t| \le k$.
  * Human predicts $\hat{y}_t = \bar{c} + \sum_{i \in A_t} h_{i,t} x_i$, incurring MSE loss:
    $$\text{MSE}(A_t, h_t) = (c - \bar{c})^2 + \sum_{i \notin A_t} a_i^2 + \sum_{i \in A_t} (a_i - h_{i,t})^2$$
  * Human updates beliefs via $\phi$-convergent dynamic: $(a_i - h_{i,t})^2 = \phi(m_i(t))(a_i - h_{i,0})^2$.
* **Key Findings**:
  * Under feature independence and MSE, **stationary policies ($A_t = A^*$ for all $t$) are globally optimal**.
  * Feature values decouple additively, allowing an $O(n \log n)$ greedy algorithm for the top-$k$ features.
  * Identifies two core tradeoffs: *Informativeness vs. Divergence* and *Fixed vs. Growth Mindset*.
* **Limitations Addressed in this Project**:
  * Noti et al. inadvertently models AI assistance as a **permanent crutch**—because the policy is stationary, the human never learns unselected features and can never graduate to unassisted decision-making.
  * Assumes continuous MSE regression rather than discrete decision-making and regret.

---

### [2] Human Decision-Making with AI Assistance under Correlated Features
* **Authors**: Yanru Guan, Naveen Raman, Fei Fang (arXiv:2606.20628, 2026)
* **Model**:
  * Extends Noti et al. to correlated features $x \sim \mathcal{N}(0, \Sigma)$, where the human uses Gaussian conditional imputation: $\hat{x}_t = \mathbb{E}[x \mid x_{S_t}]$.
* **Key Findings**:
  * Feature correlation breaks stationarity: stationary policies can perform arbitrarily poorly (unbounded approximation ratio as $\delta \to 1$).
  * The optimal policy has an **Explore-then-Commit** structure: an initial dynamic exploration phase of length $T_d$ followed by an infinite stationary suffix.
  * Proves NP-hardness via reduction from Cubic Vertex Cover (with no FPTAS).
  * Develops finite-horizon DP and truncated-horizon approximations with explicit error bounds.
* **Limitations Addressed in this Project**:
  * Still restricted to continuous scalar prediction under MSE loss; does not address discrete action selection, decision regret, action menus, or graduation curricula.

---

## 2. Comparison with Adjacent Disciplines

```
                                 Pedagogical AI Assistance
                    (Discrete Actions, Feature & Menu Curation, Regret)
                                            │
        ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
        ▼                   ▼                               ▼                   ▼
┌───────────────┐   ┌───────────────┐               ┌───────────────┐   ┌───────────────┐
│    Machine    │   │   Assistive   │               │   Bayesian    │   │  Educational  │
│   Teaching    │   │    Bandits    │               │  Persuasion   │   │  Scaffolding  │
│  (Zhu et al.) │   │  & CIRL / IRL │               │  (Information │   │   & Fading    │
│               │   │ (Hadfield-M.) │               │    Design)    │   │  (Vygotsky)   │
└───────────────┘   └───────────────┘               └───────────────┘   └───────────────┘
```

### A. Machine Teaching (Zhu 2015; Singla et al.; Liu et al.)
* **The Machine Teaching Framework**: Finds the minimal training dataset $D = \{(x_s, y_s)\}$ (the *teaching dimension*) to steer a learner's parameter vector $w$ into an $\epsilon$-ball around target $w^*$.
* **How our problem differs**:
  1. **Nature vs. Teacher generation**: In machine teaching, the teacher synthesizes or selects arbitrary states $x$. In our setting, **nature draws the environment state $x^{(t)}$**, and the AI can only choose which *features* and *action alternatives* to reveal.
  2. **Regret vs. Parameter Error**: Machine teaching penalizes teaching rounds or parameter distance $\|w - w^*\|$. Here, the objective is **online decision regret** on live operational tasks. As proven in [[<G> Parameter MSE vs Decision Regret]], parameter convergence and regret minimization fundamentally diverge.

### B. Assistive Multi-Armed Bandits & CIRL (Hadfield-Menell et al. 2016; Chan et al. 2019)
* **The CIRL Framework**: In Cooperative Inverse Reinforcement Learning and Assistive Bandits, an AI agent helps a human. The human knows the reward function, and the AI infers it from human actions to assist.
* **The Inversion in our Framework**:
  * In our setting, **the AI knows the ground-truth utility matrix $U$**, while the **human has an inaccurate mental model $H_t$**.
  * The AI acts as an instructor/coach shaping the human's information exposure, rather than an apprentice inferring human desires.

### C. Information Design & Bayesian Persuasion (Kamenica & Gentzkow 2011; Ely 2017)
* **The Persuasion Framework**: An information sender designs an experiment/signaling scheme to influence the action of a rational Bayesian receiver without altering payoff incentives.
* **How our problem differs**:
  * In Bayesian persuasion, the receiver is static and Bayesian; signals update the receiver's belief about the *current instance state*.
  * In our setting, information disclosure induces **endogenous cognitive learning** ($\phi$-convergence of the human's utility model $H_t$). Revealing a feature permanently modifies how the human acts on *all future states*.

### D. Educational Psychology: Scaffolding, Fading, and Cognitive Forcing
* **Concepts**:
  * **Scaffolding (Wood, Bruner, Ross 1976; Vygotsky)**: Temporary cognitive structures enabling a learner to solve problems beyond their unassisted capacity.
  * **Fading**: The systematic removal of scaffolding as competence increases, avoiding learned helplessness / automation bias.
* **Mathematical Formalization**:
  * Our framework provides the first rigorous mathematical formulation of scaffolding via **Action Menu Restriction ($|A_t| < K$)** and **Multi-Feature Chaperoning**, transitioning explicitly into a **Graduation Phase** ($t > T$).

---

## 3. Reductions to Canonical Theoretical Problems

| Problem Component | Mathematical Formulation | Canonical Reduction | Complexity / Property |
| :--- | :--- | :--- | :--- |
| **Ex-Ante Curriculum under Random Graduation** | $\max_{S \subseteq B} \sum_{j \in S} q_j \|d_j\| \quad \text{s.t.} \sum_{j \in S} \tau_j \le T$ | **$0$–$1$ Knapsack Problem** | NP-hard (weak), $O(nT)$ pseudo-poly DP, FPTAS. |
| **Discounted Sequence Scheduling** | $\min_{\pi} \sum_{t=0}^{T-1} \delta^t c_{j_t}(m_{j_t}(t))$ for non-increasing costs | **Single-Machine Discounted Scheduling / Gittins Index** | Contiguous blocks optimal; sorted by index $\nu_j = \frac{\sum \delta^s c(s)}{1 - \delta^L}$. |
| **Ex-Post Zero-Regret Probability** | $\mathbb{P}(\operatorname{sgn}(d_j x_j) \neq \operatorname{sgn}(d^\top x))$ for $x \sim \mathcal{N}(0, I_n)$ | **Sheppard's Arcsine Law (1899)** | Closed form: $\frac{1}{2} - \frac{1}{\pi}\arcsin\left(\frac{\|d_j\|}{\|d\|}\right)$. |
| **Parameter-Norm Minimization** | $\min \sum_j \psi(m_j)^2 (g_j^{(0)} - d_j)^2 \quad \text{s.t.} \sum m_j = T$ | **Convex Resource Allocation / Water-Filling** | Exactly optimal via greedy water-filling for convex $\psi^2$. |

---

## 4. Summary Matrix of Model Features

| Dimension | Noti et al. (2025) | Guan et al. (2026) | This Project (Pedagogical AI) |
| :--- | :--- | :--- | :--- |
| **Decision Task** | Continuous prediction ($y \in \mathbb{R}$) | Continuous prediction ($y \in \mathbb{R}$) | Discrete action selection ($\hat{y} \in [K]$) |
| **Loss Function** | MSE: $(y - \hat{y})^2$ | MSE: $(y - \hat{y})^2$ | Decision Regret: $\max_k u_k^\top x - u_{\hat{y}}^\top x$ |
| **Feature Space** | Independent | Correlated ($\mathcal{N}(0, \Sigma)$) | Independent (or general $\Sigma$), $x \sim \mathcal{N}(0, I_n)$ |
| **Action Menu Control** | None ($K=1$) | None ($K=1$) | Dynamic subsets $A_t \subseteq [K]$ ($a_{\min} \le \|A_t\| \le a_{\max}$) |
| **Feature Control** | Ex-ante subset $F_t$ | Ex-ante subset $F_t$ | Ex-ante & Ex-post subsets $F_t \subseteq [n]$ |
| **Optimal Policy** | Stationary ($A_t = A^*$) | Explore-then-Commit ($T_d$ then commit) | Dynamic Curriculum $\to$ Graduation |
| **Human Autonomy Goal** | None (Permanent crutch) | None (Permanent commit) | **Graduation to unassisted competency** |
