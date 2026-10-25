# <G> - Multi-Action Formulation: Differential Diagnosis ($K \ge 3$)

This document formalizes the multi-action setting ($K \ge 3$), capturing the true clinical complexity of emergency triage across multiple competing disease categories.

---

## 1. Clinical State, Triage Symptoms, and Action Space

1. **Latent Patient State**: $x \sim \mathcal{N}(0, \Sigma) \in \mathbb{R}^n$ represents the patient's internal physiological and biochemical markers.
2. **Presenting Symptoms**: $s = B x \in \mathbb{R}^d$ ($d \ll n$) represents triage vital signs and chief complaints.
   $$x \mid s \sim \mathcal{N}\big(\mu(s), \; \Gamma\big)$$
   where $\mu(s) = \Sigma B^\top (B \Sigma B^\top)^{-1} s$ and $\Gamma = \Sigma_{x \mid s}$.
3. **Action Space**: The clinician chooses among $K \ge 3$ distinct clinical treatments/dispositions:
   $$\mathcal{A} = \{1, 2, \dots, K\}$$
   Examples: Action 1 = *Cardiac Catheterization*, Action 2 = *Pulmonary Anticoagulation*, Action 3 = *Appendectomy*, Action $K$ = *Discharge*.
4. **Ground Truth Utilities**: $U \in \mathbb{R}^{K \times n}$ with rows $u_1^\top, \dots, u_K^\top$.
   The true utility of action $k$ is $u_k(x) = u_k^\top x$.
   Optimal action: $y^*(x) = \arg\max_{k \in [K]} u_k^\top x$.
   Regret: $\ell_{\text{reg}}(x, \hat{y}) = \max_{k \in [K]} u_k^\top x - u_{\hat{y}}^\top x$.

---

## 2. Model-Based Clinician Decision-Making ($K \ge 3$)

The clinician maintains an internal mental model matrix $H^{(t)} \in \mathbb{R}^{K \times n}$ with rows $h_1^{(t)\top}, \dots, h_K^{(t)\top}$.

### Prior Expected Utility Vector (Given Symptoms $s$)
Before ordering any lab tests, the clinician's prior expectation for each action is:
$$\bar{u}(s) := H^{(t)} \mu(s) = \begin{pmatrix} h_1^\top \mu(s) \\ \vdots \\ h_K^\top \mu(s) \end{pmatrix} \in \mathbb{R}^K$$

### Post-Test Utility Vector (Given Tests $F$)
When panel $F \subseteq [n]$ ($|F| \le c$) is unveiled:
$$\hat{u}_k(F, x_F, s) = \bar{u}_k(s) + \tilde{u}_k(F)$$
where the test innovation is:
$$\tilde{u}_k(F) := h_{k, F}^\top \big(x_F - \mu_F(s)\big)$$
The joint innovation vector across all $K$ actions is a zero-mean Gaussian vector:
$$\tilde{u}(F) = \begin{pmatrix} \tilde{u}_1(F) \\ \vdots \\ \tilde{u}_K(F) \end{pmatrix} \sim \mathcal{N}\big(0, \; \Omega(F)\big), \qquad \Omega_{k, \ell}(F) = h_{k, F}^\top \Gamma_{FF} h_{\ell, F}$$

The clinician diagnoses:
$$\hat{y} = \arg\max_{k \in [K]} \big( \bar{u}_k(s) + \tilde{u}_k(F) \big)$$

---

## 3. The Autonomous Test-Ordering Problem ($K \ge 3$)

At graduation, the clinician autonomously selects:
$$F_{\mathcal{H}}^*(s) = \arg\max_{|F| \le c} \mathbb{E}_{\tilde{u} \sim \mathcal{N}(0, \Omega(F))} \left[ \max_{k \in [K]} \big( \bar{u}_k(s) + \tilde{u}_k(F) \big) \right]$$

> [!NOTE] The Symptom-Dependency Frontier
> Unlike the binary setting ($K=2$), the expectation $\mathbb{E}[\max_k (\bar{u}_k(s) + \tilde{u}_k)]$ **cannot be reduced to a single variance term**.
> The baseline utility vector $\bar{u}(s) = H \mu(s)$ determines which actions are viable contenders, making the optimal test panel $F_{\mathcal{H}}^*(s)$ **strictly dependent on the presenting symptoms $s$**.
