# <G> Project Map: Pedagogical AI Assistance in Clinical Triage

Welcome to the **CS 598 Project Vault on Pedagogical AI Assistance in Clinical Diagnostic Triage**.

This vault contains the streamlined, non-redundant theoretical foundations, structural theorems, and novel algorithms for designing AI preceptors that train clinicians to **autonomous diagnostic competence after graduation**.

---

## 🗺️ Streamlined Reading Flow (Zero Redundancy)

### 📌 1. Foundations & Problem Formulation
* [[Project Roadmap & Goals]]: Inventory of completed files, active review queue, and publication targets.
* [[Problem Description]]: Foundational clinical formulation: latent vitals $x$, symptoms $s=Bx$, budget $c$, autonomous graduation.
* [[<G> Related Work]]: Mapping to Noti et al. (ACM EC 2025), Guan et al. (2026), Machine Teaching, CIRL, and Clinical Deskilling.

---

### 🩺 2. Binary Decision Setting ($K = 2$) — The Theorem Engine
* [[2 Actions Reduction]]: Coordinate reduction to $d = u_1 - u_2$, $p = h_1 - h_2$, and epistemic classification.
* [[<G> - Autonomous Test Selection]]: Derivation of $F_{\mathcal{H}}^*(s)$ via Value of Information; proof of symptom-invariance for fixed $K=2$.
* [[<G> - Graduation Regret & Crowding-Out]]: Exact Sheppard regret formula; proof of the Crowding-Out Law for loud distractions.
* [[<G> - Kingmaker & Curriculum Allocation]]: The **Monotone Panel Sieve (MPS)** algorithm; exact non-knapsack curriculum optimization for general $c$.
* [[<G> - Ex-Post Dynamic Teaching]]: Runtime state observation; zero-regret teaching opportunities; Bellman stakes threshold rule.

---

### 🏥 3. Multi-Action Triage ($K \ge 3$) — Clinical Reality
* [[<G> - Multi-Action Formulation]]: Formalization of multi-class triage; latent vitals $x$, presenting symptoms $s$, and innovation covariance $\Omega(F)$.
* [[<G> - Universal Saliency Theorem]]: **Closed-form Log-Sum-Exp Hessian expansion** proving universal margin-weighted saliency and exponential pruning for any $K \ge 2$.
* [[<G> - Multi-Action Pedagogical Algorithm]]: The upgraded **Universal Saliency Gradient (USG)** algorithm tracking exact population VoI gradients.
* [[<G> - Joint Action-Feature Scaffolding]]: Optimal joint $(A_t, F_t)$ curriculum policy; proof that minimal binary candidate menus maximize safe debiasing rate.

---

### 🔬 4. Cognitive Extensions & Empirical Verification
* [[<G> - Memory Decay & Restless Bandits]]: Solving cognitive forgetting via **Restless Multi-Armed Bandits** and proving Whittle Indexability.
* [[<G> - Empirical Simulation Results]]: Full Monte Carlo simulations verifying Theorems 1–4 ($R^2 = 0.99958$), crowding-out transitions, and MPS superiority.

---

### 📦 Archive
* `Archive/`: Preserved historical drafts, superseded approximations, and earlier notes.
