# <G> Graduation Overview

The **Graduation Phase** ($t > T$) models the human decision-maker after completing an instructional curriculum of $T$ rounds. 

The central design goal is **fostering human autonomy**: ensuring the AI does not become a permanent crutch, and that the human acquires genuine competency to make high-stakes decisions with minimal or no ongoing assistance.

---

## 1. Formulation of the Curriculum Problem

The algorithm operates with a budget of $T$ training exposures $\sum_{j=1}^n m_j = T$.
The overall objective minimizes a weighted combination of training regret and discounted graduation regret:
$$\min_{\pi} \left[ \lambda \sum_{t=1}^T \ell_{\text{reg}}(x^{(t)}, \hat{y}_t) + \sum_{t > T} \delta^t \ell_{\text{grad}}(x^{(t)}, \hat{y}_t) \right]$$
When the designer prioritizes ultimate competence, the training penalty is negligible ($\lambda \to 0$), decoupling the problem into two distinct optimization steps:
1. **Curriculum Allocation**: Which features $j$ to allocate the $T$ exposures to (finding count vector $m = (m_1, \dots, m_n)$).
2. **Curriculum Sequencing**: In what chronological order to present the multiset of chosen features during training to minimize discounted training loss.

---

## 2. Models of Graduation Evaluation

Depending on the operational deployment, three primary graduation evaluation models are analyzed:

| Model | Formal Loss $\ell_{\text{grad}}$ | Meaning & Setting |
| :--- | :--- | :--- |
| **G1: Unassisted Full-State Graduation** | $\Lambda_{[n]}(g(m)) = \frac{\|d\|}{\sqrt{2\pi}}\left(1 - \frac{g(m)^\top d}{\|g(m)\| \|d\|}\right)$ | The human receives **no algorithmic help** at graduation, observing all features and using internal beliefs $g(m)$. |
| **G2: Random Single-Feature Graduation** | $\sum_{j=1}^n q_j \Lambda_{\{j\}}(g_j(m_j))$ | At each graduation round, an arbitrary task draws a single feature $j \sim q$ for the decision. |
| **G3: Parameter MSE Graduation** | $\mathcal{L}_{\text{norm}}(m) = \|g(m) - d\|_2^2 = \sum_{j=1}^n \psi(m_j)^2 (g_j^{(0)} - d_j)^2$ | Evaluates how accurately the human has internalized the true parameter values. |

---

## 3. Key Findings in Graduation Theory

### 1. [[<G> Curriculum Allocation (Knapsack Equivalence)]]
* Under random single-feature graduation (G2), deciding which bad features to flip permanently subject to $\sum \tau_j \le T$ is mathematically equivalent to the **$0$–$1$ Knapsack Problem**.
* Under unassisted graduation (G1), the AI can prioritize extinguishing **"loud distractions"**—irrelevant features ($d_j = 0$) with massive misconceptions ($|g_j^{(0)}| \gg 1$)—over genuinely informative features.

### 2. [[<G> Contiguous Block Scheduling]]
* For any chosen target allocation $m = (m_1, \dots, m_n)$, the optimal sequence for presenting features during training schedules each feature in a **single contiguous block**.
* The blocks are arranged in increasing order of a **Gittins-like block priority index**:
  $$\nu_j(m_j) := \frac{\sum_{s=0}^{m_j - 1} \delta^s c_j(s)}{1 - \delta^{m_j}}$$

### 3. [[<G> Parameter MSE vs Decision Regret]]
* Resolves the open question in the problem description regarding whether to train or evaluate on parameter MSE ($\|H - U\|_F^2$):
* **Crucial Divergence**: Minimizing parameter MSE via greedy water-filling can **destroy inter-feature proportionality**, producing unassisted graduation regret that is **strictly worse than teaching nothing at all**.
