# <G> - Multi-Action Formulation: Differential Diagnosis ($K \ge 3$) Under General Correlated Biomarkers

This document establishes the complete, rigorous mathematical foundation for multi-action differential diagnosis ($K \ge 3$) under **general correlated biomarkers ($\Sigma \succ 0, \Gamma \succeq 0$)**.

We derive the clinician's post-test decision variable, characterize the **Symptom-Dependent Value of Information**, prove the **Active-Contender Reduction**, establish the **Diagnostic Voronoi Partition**, and prove the **Diagnostic Margin Decomposition Theorem**, which expresses multi-action graduation regret as a convex combination of 2-action regrets.

---

### 1. The Multi-Action Clinical Decision Setting

In real-world emergency triage, a physician does not merely make a binary decision (e.g. admit vs. discharge). She must differentiate among $K \ge 3$ competing clinical interventions:
$$\mathcal{A} = \{1, 2, \dots, K\}$$
*(e.g., Action 1 = Cardiac Catheterization, Action 2 = Pulmonary Anticoagulation, Action 3 = Emergency Appendectomy, Action $K$ = Discharge Home).*

#### A. Patient Biology and Triage Symptoms
1. **Latent Physiological State:**  
   The patient's true biological state is $x \sim \mathcal{N}(0, \Sigma) \in \mathbb{R}^n$, with covariance $\Sigma \succ 0$.
2. **Presenting Triage Symptoms:**  
   The front-desk nurse observes triage symptoms $s = Bx \in \mathbb{R}^d$ ($d \ll n$). From Gaussian conditioning:
   $$ x \mid s \sim \mathcal{N}\big(\mu(s), \; \Gamma\big) $$
   where:
   $$ \mu(s) = \Sigma B^\top (B \Sigma B^\top)^{-1} s = \Pi x \qquad \text{with } \Pi := \Sigma B^\top (B \Sigma B^\top)^{-1} B \in \mathbb{R}^{n \times n} $$
   $$ \Gamma = \Sigma - \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma = \Sigma (I_n - \Pi^\top) \succeq 0 $$
   *Note:* Even if raw vitals were independent a priori ($\Sigma = I_n$), $\Gamma$ is **inherently non-diagonal** due to Berkson's explaining-away effect (e.g., blood pressure $s$ couples cardiac output $x_1$ with systemic vascular resistance $x_2$).

#### B. Ground-Truth Utility Matrix
Nature assigns utilities via matrix $U \in \mathbb{R}^{K \times n}$, whose rows are $u_1^\top, \dots, u_K^\top \in \mathbb{R}^{1 \times n}$.
* The true utility of taking action $k$ on patient $x$ is:
  $$ u_k(x) := u_k^\top x = \sum_{j=1}^n U_{kj} x_j $$
* Nature's optimal action is:
  $$ y^*(x) := \arg\max_{k \in [K]} u_k^\top x $$
* When an action $\hat{y} \in [K]$ is selected, the patient incurs decision regret equal to the utility deficit:
  $$ \ell(x, \hat{y}) := \max_{k \in [K]} u_k^\top x - u_{\hat{y}}^\top x $$

#### C. Clinician's Subjective Mental Model
The clinician maintains an internal belief matrix $H \in \mathbb{R}^{K \times n}$, with rows $h_1^\top, \dots, h_K^\top$.
* Before ordering tests, the clinician's **prior expected utility** for action $k$ given symptoms $s$ is:
  $$ \bar{u}_k(s) := h_k^\top \mu(s) = h_k^\top \Pi x $$
* The clinician's prior expected utility vector across all $K$ actions is:
  $$ \bar{u}(s) := H \mu(s) = \begin{pmatrix} h_1^\top \mu(s) \\ \vdots \\ h_K^\top \mu(s) \end{pmatrix} \in \mathbb{R}^K $$

#### D. Post-Test Decision Variable Under Panel $F$
The clinician orders test panel $F \subseteq [n]$ with $|F| \le c$, and observes true values $x_F$.
For unmeasured biomarkers $j \notin F$, she imputes the conditional expectation $\mu_j(s)$.
Her post-test perceived utility for action $k$ is:
$$ 
\begin{aligned}
\hat{u}_k(F, x_F, s) &= \sum_{j \in F} H_{kj} x_j + \sum_{j \notin F} H_{kj} \mu_j(s) \\
&= h_k^\top \mu(s) + h_{k, F}^\top \big(x_F - \mu_F(s)\big) \\
&= \bar{u}_k(s) + \tilde{u}_k(F)
\end{aligned}
$$
where $\tilde{u}_k(F) := h_{k, F}^\top (x_F - \mu_F(s))$ is the **zero-mean Gaussian test innovation**.

The joint innovation vector across all $K$ actions is a zero-mean Gaussian vector:
$$ \tilde{u}(F) = \begin{pmatrix} \tilde{u}_1(F) \\ \vdots \\ \tilde{u}_K(F) \end{pmatrix} \sim \mathcal{N}\big(0, \; \Omega(F)\big) \in \mathbb{R}^K $$
where the $K \times K$ innovation covariance matrix is:
$$ \Omega_{k, \ell}(F) = \operatorname{Cov}\big(\tilde{u}_k(F), \tilde{u}_\ell(F) \mid s\big) = h_{k, F}^\top \Gamma_{FF} h_{\ell, F} $$

The clinician diagnoses action:
$$ \hat{y}(F, x_F, s) = \arg\max_{k \in [K]} \hat{u}_k(F, x_F, s) = \arg\max_{k \in [K]} \big( \bar{u}_k(s) + \tilde{u}_k(F) \big) $$

---

### 2. Autonomous Test Selection at Graduation ($t > T$)

Before ordering tests, the clinician evaluates candidate panels $F \subseteq [n]$ ($|F| \le c$) by their perceived **Value of Information (VoI)**:
$$ \operatorname{VoI}(F \mid s; H) := \mathbb{E}_{x_F \mid s}\left[ \max_{k \in [K]} \hat{u}_k(F, x_F, s) \right] - \max_{k \in [K]} \bar{u}_k(s) $$
The autonomous test selection rule at graduation is:
$$ F_{\mathcal{H}}^*(s; H) = \arg\max_{|F| \le c} \operatorname{VoI}(F \mid s; H) $$

#### Theorem 1 (Correlated Saliency Kernel Representation):
Let $M_\beta(u) := \frac{1}{\beta} \log \sum_{k=1}^K e^{\beta u_k}$ be the smooth Log-Sum-Exp approximation with inverse temperature $\beta > 0$. Let $\pi(s) = (\pi_1(s), \dots, \pi_K(s))^\top$ be the clinician's prior softmax disease probability vector:
$$ \pi_k(s) := \frac{e^{\beta \bar{u}_k(s)}}{\sum_{r=1}^K e^{\beta \bar{u}_r(s)}} = \frac{e^{\beta h_k^\top \mu(s)}}{\sum_{r=1}^K e^{\beta h_r^\top \mu(s)}} $$
Under a second-order expansion around $\bar{u}(s)$, the Value of Information for test panel $F$ is given by the quadratic form:
$$ \operatorname{VoI}(F \mid s; H) \approx \frac{\beta}{2} \sum_{i \in F} \sum_{j \in F} \Gamma_{ij} M_{ij}(s) = \frac{\beta}{2} z^\top \Big( \Gamma \odot M(s) \Big) z $$
where $z \in \{0, 1\}^n$ is the indicator vector of panel $F$ ($z_j = 1 \iff j \in F$), $\odot$ is the Hadamard product, and $M(s) \in \mathbb{R}^{n \times n}$ is the **Epistemic Action Covariance Matrix**:
$$ M(s) := \sum_{1 \le k < \ell \le K} \pi_k(s) \pi_\ell(s) (h_k - h_\ell)(h_k - h_\ell)^\top \succeq 0 $$

#### Proof:
Expand $M_\beta(\bar{u} + \tilde{u})$ in a Taylor series around $\bar{u}(s)$:
$$ M_\beta(\bar{u} + \tilde{u}) = M_\beta(\bar{u}) + \nabla M_\beta(\bar{u})^\top \tilde{u} + \frac{1}{2} \tilde{u}^\top \nabla^2 M_\beta(\bar{u}) \tilde{u} + \mathcal{O}(\|\tilde{u}\|^3) $$
Taking expectations over $\tilde{u} \sim \mathcal{N}(0, \Omega(F))$:
1. $\mathbb{E}[\nabla M_\beta^\top \tilde{u}] = 0$ since $\tilde{u}$ is zero-mean.
2. The Hessian of $M_\beta$ is $[\nabla^2 M_\beta]_{k\ell} = \beta (\pi_k \delta_{k\ell} - \pi_k \pi_\ell)$.
   Using the weighted variance identity $\sum \pi_k z_k^2 - (\sum \pi_k z_k)^2 = \sum_{k < \ell} \pi_k \pi_\ell (z_k - z_\ell)^2$:
   $$ \mathbb{E}\left[ \frac{1}{2} \tilde{u}^\top \nabla^2 M_\beta \tilde{u} \right] = \frac{\beta}{2} \sum_{k < \ell} \pi_k(s) \pi_\ell(s) \mathbb{E}\left[ (\tilde{u}_k - \tilde{u}_\ell)^2 \right] $$
3. Since $\tilde{u}_k - \tilde{u}_\ell = (h_{k, F} - h_{\ell, F})^\top (x_F - \mu_F(s))$, its expected square is:
   $$ \mathbb{E}[(\tilde{u}_k - \tilde{u}_\ell)^2] = (h_k - h_\ell)_F^\top \Gamma_{FF} (h_k - h_\ell)_F $$
4. Substituting into the sum and pulling $\Gamma_{FF}$ outside the trace yields $\frac{\beta}{2} \operatorname{Tr}(\Gamma_{FF} M_{FF}(s)) = \frac{\beta}{2} z^\top (\Gamma \odot M(s)) z$. $\blacksquare$

---

### 3. The Active-Contender Reduction (Clark-Sheppard Closed Form)

In clinical medicine, presenting symptoms $s$ immediately eliminate irrelevant diseases. For any patient, symptoms isolate the **two active primary contenders**:
$$ y^*(s) = \arg\max_{k \in [K]} \bar{u}_k(s), \qquad k^*(s) = \arg\max_{k \neq y^*} \bar{u}_k(s) $$
For all remaining diseases $r \notin \{y^*(s), k^*(s)\}$, their prior deficit $\bar{u}_{y^*}(s) - \bar{u}_r(s)$ is so large that $\pi_r(s) \approx 0$.

#### Theorem 2 (Local Active-Pair Reduction):
When symptoms $s$ prune the diagnosis down to the top two contenders $y^*$ and $k^*$, the multi-action test selection problem **collapses locally to an EXACT 2-action selection**:
$$ F^*(s) = \arg\max_{|F| \le c} (h_{y^*} - h_{k^*})_F^\top \Gamma_{FF} (h_{y^*} - h_{k^*})_F $$
and the exact Value of Information is given in **closed analytical form for arbitrary variance magnitude** by:
$$ \operatorname{VoI}_{\text{exact}}(F \mid s) = \sigma_{\text{diff}}(F) \cdot \phi\left( \frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right) - \Delta(s) \cdot \Phi\left( -\frac{\Delta(s)}{\sigma_{\text{diff}}(F)} \right) $$
where:
* $\Delta(s) := \bar{u}_{y^*}(s) - \bar{u}_{k^*}(s) \ge 0$ is the clinical margin between the top two diseases.
* $\sigma_{\text{diff}}^2(F) := (h_{y^*} - h_{k^*})_F^\top \Gamma_{FF} (h_{y^*} - h_{k^*})_F$ is the resolved variance across panel $F$.

#### Proof:
Identical to Section 4 of `2. Test Selection.md` by defining $V(F) := \hat{u}_{y^*}(F) - \hat{u}_{k^*}(F) \sim \mathcal{N}(\Delta(s), \sigma_{\text{diff}}^2(F))$ and integrating $\mathbb{E}[\max(0, V(F))]$. $\blacksquare$

---

### 4. Diagnostic Geometry: The Voronoi Partition of Symptom Space

Because test selection depends on which two diseases are tied, the symptom space $\mathbb{R}^d$ partitions into convex polyhedral cones called **Diagnostic Voronoi Cells**.

#### Definition: Diagnostic Voronoi Cell ($\mathcal{V}_{k, \ell}$)
For any pair of diseases $1 \le k < \ell \le K$, the Diagnostic Voronoi Cell $\mathcal{V}_{k, \ell} \subset \mathbb{R}^d$ is the region of presenting symptoms where disease $k$ and disease $\ell$ are the two top contenders:
$$ \mathcal{V}_{k, \ell} := \Big\{ s \in \mathbb{R}^d : \min\big(\bar{u}_k(s), \bar{u}_\ell(s)\big) > \max_{r \notin \{k, \ell\}} \bar{u}_r(s) \Big\} $$

#### Properties of the Diagnostic Partition:
1. **Disjoint Covering:** The collection of cells $\{\mathcal{V}_{k, \ell} : 1 \le k < \ell \le K\}$ partitions the symptom space $\mathbb{R}^d$ almost everywhere (ties between 3 or more diseases occur on a measure-zero set).
2. **Patient Arrival Frequency ($w_{k, \ell}$):**  
   Since $s = Bx \sim \mathcal{N}(0, B \Sigma B^\top)$, the probability that an arriving patient presents with a diagnostic dilemma between disease $k$ and disease $\ell$ is:
   $$ w_{k, \ell} := \mathbb{P}\big(s \in \mathcal{V}_{k, \ell}\big) $$
   Note that $\sum_{1 \le k < \ell \le K} w_{k, \ell} = 1$. These weights represent the clinical epidemiology of the hospital!
3. **Local Panel Specialization:**  
   Within cell $\mathcal{V}_{k, \ell}$, the clinician autonomously orders the panel specialized for separating disease $k$ from disease $\ell$:
   $$ F_{k, \ell}^* := \arg\max_{|F| \le c} (h_k - h_\ell)_F^\top \Gamma_{FF} (h_k - h_\ell)_F $$

---

### 5. Closed-Form Multi-Action Graduation Regret

We now establish the master closed-form expression for expected graduation regret when $K \ge 3$.

#### Theorem 3 (The Diagnostic Margin Decomposition Theorem):
Under general correlated vitals $x \sim \mathcal{N}(0, \Sigma)$ and triage symptoms $s = Bx$, the expected multi-action graduation regret of a clinician with belief matrix $H$ decomposes into a **convex combination of pairwise 2-action graduation regrets**:
$$ \mathcal{L}_{\text{grad}}(H) := \mathbb{E}_{x, s}[\ell(x, \hat{y})] = \sum_{1 \le k < \ell \le K} w_{k, \ell} \cdot \mathcal{L}_{k, \ell}\big(F_{k, \ell}^*, \; h_k - h_\ell\big) $$
where each pairwise regret $\mathcal{L}_{k, \ell}$ is given in exact closed form by the Sheppard Arcsine formula from [[3. Graduation Regret.md]]:
$$ \mathcal{L}_{k, \ell}\big(F_{k, \ell}^*, \; p_{k\ell}\big) = \frac{\sqrt{d_{k\ell}^\top \Sigma d_{k\ell}}}{\sqrt{2\pi}} \Big( 1 - \rho_{k\ell}\big(F_{k\ell}^*, p_{k\ell}\big) \Big) $$
where:
* $d_{k\ell} := u_k - u_\ell \in \mathbb{R}^n$ is the ground-truth pairwise advantage.
* $p_{k\ell} := h_k - h_\ell \in \mathbb{R}^n$ is the clinician's perceived pairwise difference.
* $\rho_{k\ell}$ is the Pearson clinical correlation for pair $(k, \ell)$:
  $$ \rho_{k\ell}(F_{k\ell}^*, p_{k\ell}) = \frac{d_{k\ell}^\top (\Sigma - \Gamma) p_{k\ell} + (p_{k\ell})_{F_{k\ell}^*}^\top \Gamma_{F_{k\ell}^*, :} \, d_{k\ell}}{\sqrt{d_{k\ell}^\top \Sigma d_{k\ell}} \;\; \sqrt{p_{k\ell}^\top (\Sigma - \Gamma) p_{k\ell} + (p_{k\ell})_{F_{k\ell}^*}^\top \Gamma_{F_{k\ell}^* F_{k\ell}^*} (p_{k\ell})_{F_{k\ell}^*}}} $$

#### Proof:
By the Law of Total Expectation, conditioning on the triage Voronoi cell $s \in \mathcal{V}_{k, \ell}$:
$$ \mathcal{L}_{\text{grad}}(H) = \sum_{1 \le k < \ell \le K} \mathbb{P}(s \in \mathcal{V}_{k, \ell}) \cdot \mathbb{E}\big[ \ell(x, \hat{y}) \mid s \in \mathcal{V}_{k, \ell} \big] $$
Within cell $\mathcal{V}_{k, \ell}$, all other $K - 2$ actions are sub-optimal by at least $\Delta_{\text{prune}}$, so the decision simplifies to $\hat{y} \in \{k, \ell\}$.
The loss is $\ell = |(u_k - u_\ell)^\top x| \cdot \mathbb{1}[\operatorname{sgn}(V(F_{k\ell}^*)) \neq \operatorname{sgn}(d_{k\ell}^\top x)]$.
Applying the Master Graduation Regret Theorem from `3. Graduation Regret.md` directly to this pairwise subproblem yields $\frac{\sqrt{d_{k\ell}^\top \Sigma d_{k\ell}}}{\sqrt{2\pi}} (1 - \rho_{k\ell})$. Weighting by arrival probability $w_{k, \ell}$ completes the proof. $\blacksquare$

---

### 6. The Multi-Action Bilevel Curriculum Program

With the graduation regret decomposed into explicit pairwise correlations, the AI preceptor's offline training optimization problem becomes:

$$ 
\begin{aligned}
\min_{m \in \mathbb{N}^n} \quad & \sum_{1 \le k < \ell \le K} w_{k, \ell} \cdot \frac{\sqrt{d_{k\ell}^\top \Sigma d_{k\ell}}}{\sqrt{2\pi}} \Big( 1 - \rho_{k\ell}\big(F_{k\ell}^*(p(m)), \; p_{k\ell}(m)\big) \Big) \\
\text{subject to} \quad & \sum_{j=1}^n m_j = T, \quad m_j \in \mathbb{N} \\
& F_{k\ell}^*(p(m)) = \arg\max_{|F| \le c} (p_{k\ell}(m))_F^\top \Gamma_{FF} (p_{k\ell}(m))_F \quad \forall (k, \ell)
\end{aligned}
$$

This formulation reveals the central challenge of multi-action training:
> [!IMPORTANT] The Multi-Margin Coupling
> An exposure $m_j$ spent on biomarker $j$ updates column $j$ of belief matrix $H$, which simultaneously alters the perceived difference $p_{k\ell} = h_k - h_\ell$ across **all disease pairs that depend on biomarker $j$**!
> The optimal curriculum must balance calibration and distractor extinguishment across multiple active margins simultaneously.
