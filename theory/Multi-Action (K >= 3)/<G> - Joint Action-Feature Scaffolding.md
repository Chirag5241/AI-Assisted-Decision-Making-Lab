# <G> - Joint Action-Feature Scaffolding: Optimal (A_t, F_t) Curriculum Policy

This document formalizes the theory and optimal algorithm for **Joint Action-Feature Scaffolding** for multi-action differential diagnosis ($K \ge 3$).

When the AI preceptor can simultaneously restrict the candidate diagnostic menu $A_t \subset [K]$ ($a_{\min} \le |A_t| \le a_{\max} < K$) and select the test panel $F_t \subseteq [n]$ ($|F_t| \le c$), how should $(A_t, F_t)$ be jointly scheduled to optimize autonomous clinical competence?

---

## 1. Mathematical Formulation of Joint Scaffolding

At each training round $t = 1, \dots, T$:
1. Patient arrives with latent vitals $x^{(t)} \sim \mathcal{N}(0, \Sigma)$ and presenting symptoms $s^{(t)} = B x^{(t)}$.
   The true optimal action is $y^*(x^{(t)}) = \arg\max_{k \in [K]} u_k^\top x^{(t)}$.
2. The AI preceptor presents:
   * A **diagnostic action menu** $A_t \subseteq [K]$ with $2 \le |A_t| \le a_{\max} < K$.
   * A **diagnostic test panel** $F_t \subseteq [n]$ with $|F_t| \le c$.
3. The clinician observes $A_t$ and revealed biomarkers $x_{F_t}^{(t)}$, and chooses:
   $$\hat{y}_t = \arg\max_{k \in A_t} \sum_{j \in F_t} H_{kj}^{(t)} x_j$$
   incurring immediate decision regret:
   $$\ell_{\text{reg}}(x^{(t)}, \hat{y}_t) = \max_{k \in [K]} u_k^\top x^{(t)} - u_{\hat{y}_t}^\top x^{(t)}$$
4. **Targeted Grid Updating Rule**:
   Crucially, belief calibration occurs **only for the exposed grid** $(k, j) \in A_t \times F_t$:
   $$H_{kj}^{(t+1)} - U_{kj} = \psi\big(m_{kj}(t) + 1\big) \big(H_{kj}^{(0)} - U_{kj}\big) \quad \forall (k, j) \in A_t \times F_t$$
   For all unexposed pairs $(k, j) \notin A_t \times F_t$, beliefs remain frozen: $H_{kj}^{(t+1)} = H_{kj}^{(t)}$.

---

## 2. The Core Scaffolding Trade-Offs

Joint scaffolding introduces three coupled dynamics:

### 1. The Safety Constraint (Patient Shielding)
If the AI excludes the true optimal action $y^*(x^{(t)}) \notin A_t$, the clinician is forced to pick a suboptimal action, causing **guaranteed immediate regret**:
$$\ell_{\text{reg}} \ge u_{y^*}^\top x^{(t)} - \max_{k \in A_t} u_k^\top x^{(t)} > 0$$
Therefore, any safe pedagogical policy must ensure:
$$y^*(x^{(t)}) \in A_t \quad \text{almost surely.}$$

### 2. The Row Isolation Benefit (Debiasing Acceleration)
When a clinician holds a severe misconception on a specific disease row $k$ ($H_{kj} \gg U_{kj}$), exposing a large menu ($|A_t| = K$) allows other rows to interfere, diluting the error signal.
By restricting the menu to $A_t = \{y^*(x^{(t)}), k\}$, the AI **isolates row $k$**, forcing the clinician to confront the misconception directly and accelerating sign flips:
$$\tau^{\text{scaffold}} < \tau^{\text{full-menu}}$$

### 3. The Grid Budget Allocation
The AI has an operational budget of at most $a_{\max} \times c$ cell updates per round. Allocating menu slots to already-calibrated disease rows wastes exposures that should be spent debiasing critical margins.

---

## 3. The Optimal Pairwise Scaffolding Curriculum (PSC)

egin{theorem}[Optimality of Minimal Safe Scaffolding]
Assume training regret weight $\lambda > 0$ and non-increasing contraction $\psi$. In any round $t$ where the clinician holds uncalibrated rows, the optimal action menu size is minimal:
$$|A_t^*| = a_{\min} = 2$$
The optimal action menu is uniquely given by:
$$A_t^* = \big\{ y^*(x^{(t)}), \; k^*(t) \big\}$$
where $k^*(t)$ is the highest-risk uncalibrated competitor:
$$k^*(t) = \arg\max_{k \neq y^*(x^{(t)})} \text{Risk}\big(y^*(x^{(t)}), k; H^{(t)}\big)$$
\end{theorem}

egin{proof}
Let $y^* = y^*(x^{(t)})$. To guarantee zero forced regret from menu truncation, $y^* \in A_t$.
Suppose $|A_t| > 2$. Adding any third action $\ell \notin \{y^*, k^*\}$ to $A_t$:
1. Cannot improve immediate safety, since $y^*$ is already present.
2. Diverts clinician attention and allows row $\ell$ to absorb credit/blame, weakening the contraction step on $k^*$.
3. Increases the flip threshold $\tau$ for row $k^*$ by Proposition on Row Debiasing.
Therefore, setting $A_t^* = \{y^*, k^*\}$ strictly minimizes both immediate risk and time-to-debias. $\blacksquare$
\end{proof}

---

## 4. The Joint PSC Algorithm

```
Algorithm: Pairwise Scaffolding Curriculum (PSC)
Input: Patient stream (x^(t), s^(t)), true U, initial H^(0), limits (c, a_min=2)
Output: Online sequence of action menus A_t and feature panels F_t

For round t = 1 to T:
    1. Determine ground-truth optimal action:
       y* = argmax_{k in [K]} u_k^T x^(t)
       
    2. Identify the most critical contending disease row:
       k* = argmax_{k != y*} Risk_{y*, k}(H^(t))
       Set Action Menu: A_t = {y*, k*}
       
    3. Define Active Margin Difference:
       p^(active) = h_{y*}^(t) - h_{k*)^(t)
       d^(active) = u_{y*} - u_{k*}
       
    4. Select Feature Panel F_t (|F_t| <= c):
       Using the Universal Saliency Gradient, select the c biomarkers
       maximizing debiasing power between row y* and row k*:
       F_t = argmax_{|F| <= c} sum_{j in F} (H_{y*, j}^(t) - H_{k*, j}^(t) - d_j^(active))^2 * sigma_j^2
       
    5. Present (A_t, F_t) to clinician; observe choice y_hat and reward
    6. Update exposed grid (k, j) in A_t x F_t
```

### Theoretical Guarantees:
1. **Zero Truncation Regret**: The clinician is never deprived of the optimal treatment choice ($y^* \in A_t$).
2. **Maximal Debiasing Rate**: Restricting to binary candidate menus achieves the theoretical upper bound on row contraction speed.
3. **Seamless Graduation**: At round $T$, all pairwise disease margins are verified calibrated, guaranteeing that unassisted full-menu graduation ($A_{\text{grad}} = [K]$) operates with minimal regret.
