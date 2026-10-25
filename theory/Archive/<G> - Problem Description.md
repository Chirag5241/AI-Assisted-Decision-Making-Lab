# Pedagogical AI Assistance in Clinical Diagnostic Triage
## Problem Description & Mathematical Framework

Consider a human clinician $\mathcal{H}$ (e.g., an emergency department physician or resident) interacting with an AI preceptor/assistant $\mathcal{A}$ across repeated patient cases. The central design challenge is **fostering human autonomy**: training the clinician to both **order the right diagnostic tests** and **correctly interpret the results**, so that when algorithmic assistance is withdrawn at graduation, the clinician does not suffer from diagnostic deskilling or rely on the AI as a permanent crutch.

---

## 1. Environment, Patients, and Ground Truth

At each round $t = 1, 2, \dots$:
1. **Latent Patient State (Vitals & Biomarkers):** Nature draws an internal, unobserved biological state $x^{(t)} \in \mathbb{R}^n$ according to a prior distribution $\mathcal{D} = \mathcal{N}(0, \Sigma)$. The vector $x^{(t)}$ contains all internal physiological parameters, lab markers (e.g., troponin, arterial blood gas, specific enzyme levels), and histological scores.
2. **Presenting Symptoms (Triage Presentation):** Before any tests are run, a lossy, zero-cost observation $s^{(t)} = B x^{(t)} \in \mathbb{R}^d$ ($d \ll n$, $B \in \mathbb{R}^{d \times n}$) is revealed to both the human and the AI. Symptoms represent immediate physical complaints (e.g., acute chest pain, dyspnea, fever, blood pressure).
   - Conditioned on $s^{(t)}$, the latent biomarkers follow a Gaussian posterior:
     $$x \mid s \sim \mathcal{N}\big(\mu(s), \; \Sigma_{x \mid s}\big)$$
     where $\mu(s) = \Sigma B^\top (B \Sigma B^\top)^{-1} s$ and $\Sigma_{x \mid s} = \Sigma - \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma$.
     We denote the residual uncertainty covariance by $\Gamma := \Sigma_{x \mid s} \in \mathbb{R}^{n \times n}$.
3. **Clinical Actions & Ground-Truth Utility:** The physician must choose a clinical action $k \in [K]$ (e.g., diagnosis, immediate surgery, prescription, discharge). The true utility of action $k$ on patient $x$ is linear in the latent vitals:
   $$u_k(x) = u_k^\top x, \qquad U = \begin{bmatrix} - u_1^\top - \\ \vdots \\ - u_K^\top - \end{bmatrix} \in \mathbb{R}^{K \times n}$$
   The true optimal action is $y^*(x) = \arg\max_{k \in [K]} u_k^\top x$.
4. **Clinical Decision Regret:** If action $\hat{y} \in [K]$ is selected, the incurred regret is:
   $$\ell_{\text{reg}}(x, \hat{y}) = \max_{k \in [K]} u_k^\top x - u_{\hat{y}}^\top x \ge 0$$

---

## 2. The Clinical Capacity Constraint

* Due to constraints on financial cost, time, invasiveness, and cognitive bandwidth, the clinician can order at most $c$ diagnostic tests ($c \ll n$).
* Ordering panel $F \subseteq [n]$ with $|F| \le c$ unveils the true values of the selected biomarkers $x_F \in \mathbb{R}^{|F|}$.
* The unmeasured biomarkers $F^c = [n] \setminus F$ remain latent; inference over them relies on the symptom-conditioned expectation $\mu_{F^c}(s)$.

---

## 3. The Human Decision-Maker: Model-Based Cognition

The clinician maintains an internal mental model of disease utilities $H^{(t)} \in \mathbb{R}^{K \times n}$ with rows $h_1^{(t)\top}, \dots, h_K^{(t)\top}$.

### A. Diagnostic Decision Rule
Given symptoms $s$, ordered panel $F$, and unveiled results $x_F$, the clinician forms the subjective utility estimate for action $k$:
$$\hat{u}_k(F, x_F, s; H^{(t)}) = \sum_{j \in F} H_{kj}^{(t)} x_j + \sum_{j \notin F} H_{kj}^{(t)} \mu_j(s)$$
The clinician chooses diagnosis:
$$\hat{y}\big(F, x_F, s; H^{(t)}\big) = \arg\max_{k \in [K]} \hat{u}_k(F, x_F, s; H^{(t)})$$

### B. Autonomous Test-Selection Rule (Value of Information)
When practicing without AI assistance, the clinician must decide which tests $F$ to order based on presenting symptoms $s$.
A rational clinician selects the test panel that maximizes their **Perceived Value of Information (VoI)** under mental model $H^{(t)}$:
$$F_{\mathcal{H}}^*(s; H^{(t)}) = \arg\max_{|F| \le c} \mathbb{E}_{x_F \mid s} \left[ \max_{k \in [K]} \hat{u}_k(F, x_F, s; H^{(t)}) \right]$$

> [!NOTE] Inherent Coupling
> Feature selection ($F$) and diagnostic interpretation ($h$) are derived from the **same underlying matrix $H^{(t)}$**. Correcting $H^{(t)} \to U$ simultaneously teaches the clinician which tests to order and how to interpret their results.

---

## 4. Two-Phase Protocol: Training to Autonomous Graduation

### Phase 1: Pedagogical Training Phase ($t = 1, \dots, T$)
For rounds $t = 1, \dots, T$:
1. Patient arrives: $(x^{(t)}, s^{(t)})$ realized.
2. The AI assistant $\mathcal{A}$ selects and recommends the diagnostic panel $F_t \subseteq [n]$ ($|F_t| \le c$).
3. The lab tests are executed, unveiling $x_{F_t}^{(t)}$.
4. The clinician diagnoses $\hat{y}_t = \hat{y}(F_t, x_{F_t}^{(t)}, s^{(t)}; H^{(t)})$.
5. Clinical outcome / corrective feedback is observed. The clinician's mental model updates for all measured features $j \in F_t$:
   $$H_{kj}^{(t+1)} - U_{kj} = \psi\big(m_{kj}(t)\big) \big(H_{kj}^{(0)} - U_{kj}\big)$$
   where $m_{kj}(t) = |\{ s < t : k \in A_s, j \in F_s\}|$ is the exposure count, and $\psi: \mathbb{N} \to [0, 1]$ is a non-increasing contraction function with $\psi(0) = 1$ and $\lim_{m \to \infty} \psi(m) = 0$. For unmeasured features $j \notin F_t$, beliefs remain unchanged: $H_{kj}^{(t+1)} = H_{kj}^{(t)}$.

### Phase 2: Autonomous Graduation Phase ($t > T$)
At round $T$, the AI is completely withdrawn. For all future rounds $t > T$:
1. Patient arrives with symptoms $s^{(t)}$.
2. The clinician **autonomously selects the diagnostic panel** using their learned mental model:
   $$\hat{F}_t = F_{\mathcal{H}}^*(s^{(t)}; H^{(T)}) \quad (|\hat{F}_t| \le c)$$
3. Lab tests unveil $x_{\hat{F}_t}^{(t)}$.
4. The clinician diagnoses $\hat{y}_t = \hat{y}(\hat{F}_t, x_{\hat{F}_t}^{(t)}, s^{(t)}; H^{(T)})$.
5. The system incurs autonomous graduation regret:
   $$\ell_{\text{grad}}(x^{(t)}, s^{(t)}; H^{(T)}) = \max_{k \in [K]} u_k^\top x^{(t)} - u_{\hat{y}_t}^\top x^{(t)}$$

---

## 5. The AI Preceptor's Objective

The AI designer chooses the training policy $\pi = (F_1, \dots, F_T)$ to minimize the combined objective:
$$\min_{\pi} \mathcal{J}(\pi) := \lambda \sum_{t=1}^T \mathbb{E}\big[\ell_{\text{reg}}(x^{(t)}, \hat{y}_t)\big] + \sum_{t > T} \delta^t \mathbb{E}\big[\ell_{\text{grad}}(x^{(t)}, s^{(t)}; H^{(T)})\big]$$
where $\lambda \in [0, 1)$ governs training-phase risk sensitivity, and $\delta \in (0, 1)$ is the graduation discount factor.
When the preceptor prioritizes long-term autonomous competence ($\lambda \to 0$), the problem decouples into finding the optimal training curriculum allocation $m^* = (m_1, \dots, m_n)$ followed by optimal training sequencing.
