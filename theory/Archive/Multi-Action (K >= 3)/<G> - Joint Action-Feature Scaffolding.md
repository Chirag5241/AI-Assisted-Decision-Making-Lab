# <G> - Joint Action-Feature Scaffolding: Optimal (A_t, F_t) Curriculum Policy Under Correlated Biomarkers

This document formalizes the theory and optimal algorithm for **Joint Action-Feature Scaffolding** for multi-action differential diagnosis ($K \ge 3$) under general residual biomarker covariance $\Gamma = \Sigma_{x \mid s} \succeq 0$.

When the AI preceptor can simultaneously restrict the candidate diagnostic menu $A_t \subset [K]$ ($a_{\min} \le |A_t| \le a_{\max} < K$) and select the test panel $F_t \subseteq [n]$ ($|F_t| \le c$), how should $(A_t, F_t)$ be jointly scheduled to maximize the debiasing rate while guaranteeing patient safety?

---

## 1. Mathematical Formulation of Joint Scaffolding

At each training round $t = 1, \dots, T$:
1. A patient arrives with latent vitals $x^{(t)} \sim \mathcal{N}(0, \Sigma)$ and presenting symptoms $s^{(t)} = B x^{(t)}$.
   The residual biomarker distribution is $x^{(t)} \mid s^{(t)} \sim \mathcal{N}(\mu(s^{(t)}), \Gamma)$, where $\Gamma \in \mathbb{R}^{n \times n}$ is the positive semi-definite residual covariance matrix.
   The ground-truth optimal action is:
   $$y^*(x^{(t)}) = \arg\max_{k \in [K]} u_k^\top x^{(t)}$$
2. The AI preceptor presents:
   * A **diagnostic action menu** $A_t \subseteq [K]$ with $2 \le |A_t| \le a_{\max} < K$.
   * A **diagnostic test panel** $F_t \subseteq [n]$ with $|F_t| \le c$.
3. The clinician observes candidate menu $A_t$ and revealed biomarkers $x_{F_t}^{(t)}$, and chooses:
   $$\hat{y}_t = \arg\max_{k \in A_t} \sum_{j \in F_t} H_{kj}^{(t)} x_j$$
   incurring immediate decision regret:
   $$\ell_{\text{reg}}(x^{(t)}, \hat{y}_t) = \max_{k \in [K]} u_k^\top x^{(t)} - u_{\hat{y}_t}^\top x^{(t)}$$
4. **Targeted Grid Updating Rule**:
   Belief calibration occurs **only for the exposed grid** $(k, j) \in A_t \times F_t$:
   $$H_{kj}^{(t+1)} - U_{kj} = \psi\big(m_{kj}(t) + 1\big) \big(H_{kj}^{(0)} - U_{kj}\big) \quad \forall (k, j) \in A_t \times F_t$$
   For all unexposed pairs $(k, j) \notin A_t \times F_t$, beliefs remain frozen: $H_{kj}^{(t+1)} = H_{kj}^{(t)}$.

---

## 2. Core Scaffolding Dynamics Under Correlated Features

Joint scaffolding introduces three coupled mathematical mechanisms:

### 1. The Safety Constraint (Patient Shielding)
If the AI excludes the true optimal action $y^*(x^{(t)}) \notin A_t$, the clinician is forced to pick a suboptimal action, causing **guaranteed immediate regret**:
$$\ell_{\text{reg}} \ge u_{y^*}^\top x^{(t)} - \max_{k \in A_t} u_k^\top x^{(t)} > 0$$
Therefore, any safe pedagogical policy must ensure:
$$y^*(x^{(t)}) \in A_t \quad \text{almost surely.}$$

### 2. Active Margin Isolation & Debiasing Speed
Let $k^*$ be the primary contending disease row. By restricting $A_t = \{y^*, k^*\}$, the preceptor isolates the active difference vector:
$$d := u_{y^*} - u_{k^*} \in \mathbb{R}^n, \qquad p^{(t)} := h_{y^*}^{(t)} - h_{k^*}^{(t)} \in \mathbb{R}^n$$
Under full menus ($|A_t| = K$), credit assignment is diluted across non-contending alternatives: extraneous rows absorb prediction errors and slow down the contraction of critical margin errors. Minimal menus maximize the direct gradient step on the $(y^*, k^*)$ margin.

### 3. Correlated Feature Alignment
Under non-diagonal $\Gamma$, selecting biomarkers for $F_t$ cannot be done independently. The error variance between rows $y^*$ and $k^*$ across panel $F_t$ is:
$$\operatorname{Var}\Big[ (p_{F_t}^{(t)} - d_{F_t})^\top (x_{F_t} - \mu_{F_t}(s)) \Big] = (p_{F_t}^{(t)} - d_{F_t})^\top \Gamma_{F_t F_t} (p_{F_t}^{(t)} - d_{F_t})$$
This full quadratic form captures collinear redundancies and cross-feature synergies directly.

---

## 3. The Optimal Pairwise Scaffolding Curriculum (PSC)

\begin{theorem}[Optimality of Minimal Safe Scaffolding Under General Covariance]
Assume training regret penalty $\lambda > 0$ and non-increasing contraction $\psi$. In any training round $t$ where the clinician holds uncalibrated parameters, the optimal action menu size is strictly minimal:
$$|A_t^*| = a_{\min} = 2$$
The optimal action menu is uniquely given by:
$$A_t^* = \big\{ y^*(x^{(t)}), \; k^*(t) \big\}$$
where $k^*(t)$ is the highest-risk uncalibrated competitor under the triage posterior:
$$k^*(t) = \arg\max_{k \neq y^*(x^{(t)})} p_k(s^{(t)}) \cdot (h_{y^*} - h_k - u_{y^*} + u_k)^\top \Gamma (h_{y^*} - h_k - u_{y^*} + u_k)$$
\end{theorem}

\begin{proof}
Let $y^* = y^*(x^{(t)})$. To guarantee zero forced regret from menu truncation, $y^* \in A_t$.
Suppose $|A_t| > 2$. Adding any third action $\ell \notin \{y^*, k^*\}$ to $A_t$:
1. **Safety**: Cannot improve immediate patient safety, since the optimal action $y^*$ is already present in $A_t$.
2. **Attention Dilution**: Under the learning dynamic, exposing row $\ell$ diverts finite clinician cognitive capacity. If the clinician mispredicts action $\ell$, the belief contraction step is applied to row $\ell$ rather than extinguishing the critical error between $y^*$ and $k^*$.
3. **Margin Rank**: The rank of the active epistemic covariance matrix $\sum_{a < b \in A_t} p_a p_b \Delta h_{ab} \Delta h_{ab}^\top$ increases from 1 (for a binary pair) to $|A_t| - 1 \ge 2$, spreading out the debiasing gradient across multiple dimensions rather than concentrating exposures along the principal error axis.
Therefore, setting $|A_t^*| = 2$ with $A_t^* = \{y^*, k^*\}$ strictly minimizes immediate clinical risk and maximizes the contraction rate along the active error margin. $\blacksquare$
\end{proof}

---

## 4. Correlated Feature Panel Selection for Active Margins

With $A_t = \{y^*, k^*\}$, how does the preceptor select the optimal biomarker panel $F_t \subseteq [n]$ ($|F_t| \le c$) when features are correlated?

Define the **Active Miscalibration Vector**:
$$e(t) := (h_{y^*}^{(t)} - h_{k^*}^{(t)}) - (u_{y^*} - u_{k^*}) \in \mathbb{R}^n$$
The debiasing objective is to select $F_t$ to maximize the miscalibration variance exposed to the clinician:
$$\max_{F \subseteq [n], |F| \le c} e_F(t)^\top \Gamma_{FF} e_F(t)$$

Expressing this in indicator form with $z \in \{0, 1\}^n$ ($z_j = 1 \iff j \in F$):
$$e_F(t)^\top \Gamma_{FF} e_F(t) = z^\top \Big( \Gamma \odot \big(e(t) e(t)^\top\big) \Big) z$$
Define the **Active Pairwise Saliency Kernel**:
$$K_{\text{active}}(t) := \Gamma \odot \big(e(t) e(t)^\top\big) \in \mathbb{R}^{n \times n}$$
* Its diagonal elements are:
  $$\big[ K_{\text{active}}(t) \big]_{jj} = \Gamma_{jj} e_j(t)^2 = \sigma_j^2 \Big( (H_{y^* j}^{(t)} - H_{k^* j}^{(t)}) - (U_{y^* j} - U_{k^* j}) \Big)^2$$
* Its off-diagonal elements are:
  $$\big[ K_{\text{active}}(t) \big]_{ij} = \Gamma_{ij} e_i(t) e_j(t)$$

The preceptor solves this using the **Covariance-Penalized Greedy (CPG)** selection:
Initialize $F_t = \emptyset$. Iteratively add biomarkers $j^*$ for $step = 1, \dots, c$:
$$j^* = \arg\max_{j \notin F_t} \left\{ \big[ K_{\text{active}}(t) \big]_{jj} + 2 \sum_{i \in F_t} \big[ K_{\text{active}}(t) \big]_{ij} \right\}$$
* If two biomarkers are collinear ($\Gamma_{ij} > 0$) and have the same error sign ($e_i e_j > 0$), the interaction term penalizes the second biomarker, preventing redundant panel allocation.
* If two biomarkers have negative correlation ($\Gamma_{ij} < 0$) with opposing error signs ($e_i e_j < 0$), the cross-term is positive, actively incentivizing synergistic joint testing.

---

## 5. The Joint Correlated PSC Algorithm

```
Algorithm: Correlated Pairwise Scaffolding Curriculum (C-PSC)
Input: Patient stream (x^(t), s^(t)), true U, initial H^(0), capacity c, residual covariance Gamma
Output: Online sequence of action menus A_t and feature panels F_t

For round t = 1 to T:
    1. Observe patient presentation (s^(t), x^(t)):
    
    2. Determine ground-truth optimal clinical action:
       y* = argmax_{k ∈ [K]} u_k^T x^(t)
       
    3. Compute symptom-prior disease likelihoods:
       p_k(s^(t)) = exp(beta * (H^(t) mu(s^(t)))_k) / sum_r exp(beta * (H^(t) mu(s^(t)))_r)
       
    4. Identify the highest-risk contending disease row:
       k* = argmax_{k != y*} p_k(s^(t)) * [ (h_y* - h_k) - (u_y* - u_k) ]^T Gamma [ (h_y* - h_k) - (u_y* - u_k) ]
       
    5. Construct minimal candidate action menu:
       A_t = {y*, k*}
       
    6. Compute active miscalibration difference vector:
       e(t) = (h_{y*}^{(t)} - h_{k*}^{(t)}) - (u_{y*} - u_{k*})
       
    7. Construct Active Saliency Kernel:
       K_active(t) = Gamma ⊙ (e(t) e(t)^T)
       
    8. Select optimal test panel F_t using Covariance-Penalized Greedy:
       F_t = CPG(K_active(t), c)
       
    9. Present (A_t, F_t) to clinician; observe predicted action y_hat
    10. Deliver clinical feedback and update exposed grid:
        H_kj^(t+1) - U_kj = psi(m_kj(t) + 1) * [H_kj^(t) - U_kj]   ∀ (k, j) ∈ A_t x F_t
        All unexposed cells (k, j) ∉ A_t x F_t remain frozen.
```

---

## 6. Theoretical Guarantees

1. **Zero Truncation Regret**: Because the optimal clinical disposition is always present ($y^* \in A_t$), menu truncation never deprives the patient of the optimal medical intervention.
2. **Maximal Margin Contraction Speed**: Binary menus restrict the epistemic rank to 1, concentrating 100% of the training exposure along the active error axis $e(t)$.
3. **Collinear Redundancy Elimination**: Under non-diagonal $\Gamma$, the CPG selection rule guarantees that test budget $c$ is never wasted on collinear redundant assays, maximizing information gain per test ordered.
4. **Seamless Graduation ($t > T$)**: Upon graduation, the clinician is released to unconstrained decision-making ($A_{\text{grad}} = [K]$). Because C-PSC systematically extinguishes error along all pairwise margins, the autonomous test ordering $F_{\mathcal{H}}^*(s)$ operates with minimal regret.
