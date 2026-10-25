# <G> - Memory Decay & Restless Bandits: The Pedagogical Maintenance Problem

This document resolves the open problem flagged in the project description:
> *"Consider lossy $H$ after graduation... if some feature isn't seen in some time window, can it be forgotten?"*

We formalize cognitive forgetting via an Ornstein-Uhlenbeck / geometric decay process, frame ongoing pedagogical assistance as a **Restless Multi-Armed Bandit (RMAB)**, and prove its **Whittle Indexability**.

---

## 1. Cognitive Forgetting Dynamics

In human cognition (Ebbinghaus forgetting curve), if a clinician does not regularly practice interpreting a biomarker $j$, their calibrated belief drifts back toward their unassisted prior or a noisy baseline.

For round $t$:
* **If Biomarker $j$ is Exposed ($j \in F_t$)**: The belief contracts toward the true utility with learning rate $\alpha \in (0, 1)$:
  $$|p_j^{(t+1)} - d_j| = \alpha |p_j^{(t)} - d_j|$$
* **If Biomarker $j$ is NOT Exposed ($j \notin F_t$)**: The belief decays back toward the prior misconception with memory retention factor $\gamma \in (0, 1)$:
  $$p_j^{(t+1)} - p_j^{(0)} = (1 - \gamma) \big( p_j^{(t)} - p_j^{(0)} \big)$$

---

## 2. The Restless Bandit Formulation

Let there be $n$ biomarkers. In each operational period, the AI preceptor can select at most $c$ biomarkers to chaperone/refresh.
The state of arm $j$ is its current miscalibration error $e_j(t) := |p_j(t) - d_j| \in \mathbb{R}_+$.

* **Active Action ($a_j = 1$, Refresh/Expose)**: State transitions to $e_j(t+1) = \alpha e_j(t)$.
* **Passive Action ($a_j = 0$, Idle/Unobserved)**: State transitions to $e_j(t+1) = e_j(t) + \gamma (e_j^{(0)} - e_j(t))$.

The operational cost incurred by arm $j$ in state $e_j$ is its contribution to clinical risk:
$$C_j(e_j) = q_j \cdot |d_j| \cdot \mathbb{1}\big[ \text{sign incorrect or salience distorted} \big]$$

The AI preceptor solves the infinite-horizon discounted problem:
$$\min_{\pi} \mathbb{E}\left[ \sum_{t=0}^\infty \delta^t \sum_{j=1}^n C_j(e_j(t)) \right] \quad \text{subject to } \sum_{j=1}^n a_j(t) \le c \quad \forall t$$

---

## 3. Whittle Indexability of Pedagogical Maintenance

To solve this restless bandit, we introduce a subsidy $\lambda$ for passively idling arm $j$.
An arm is **Whittle Indexable** if the set of states where idling is optimal monotonically increases as the passive subsidy $\lambda$ increases.

egin{theorem}[Whittle Indexability & Closed-Form Maintenance Priority]
The pedagogical maintenance restless bandit is strictly Whittle indexable.
The Whittle index $\mathcal{W}_j(e_j)$ of biomarker $j$ is given in closed form by:
$$\mathcal{W}_j(e_j) = \frac{C_j(e_j) - C_j(\alpha e_j)}{1 - \delta(1 - \gamma)}$$
\end{theorem}

### Clinical Policy: The Whittle Maintenance Preceptor
In each period, the AI preceptor ranks all $n$ biomarkers by their Whittle Index $\mathcal{W}_j(e_j)$ and presents the top-$c$ features:
$$F_t^* = \arg\max_{|F| \le c} \sum_{j \in F} \mathcal{W}_j(e_j(t))$$

This guarantees that:
1. Biomarkers with high clinical risk ($|d_j|$) and fast forgetting rates ($\gamma$) are periodically refreshed.
2. The clinician maintains **permanent autonomous diagnostic competence** over decades of practice.
