# <G> Project Map: Pedagogical AI Assistance in Clinical Triage

Welcome to the **CS 598 Project Vault on Pedagogical AI Assistance in Clinical Diagnostic Triage**.

This vault organizes the theoretical foundations, structural theorems, and novel algorithms for designing AI preceptors that train clinicians across repeated interactions to ensure **autonomous diagnostic competence after graduation**.

---

## 🗺️ Vault Structure & Reading Flow

### 📌 Project Status & Foundation
* [[Project Roadmap & Goals]]: Inventory of completed files, review queue, active targets, and roadmap.
* [[Problem Description]]: Foundational clinical formulation: latent vitals $x$, symptoms $s=Bx$, budget $c$, autonomous graduation.
* [[<G> Related Work]]: Mapping to Noti et al. (ACM EC 2025), Guan et al. (2026), Machine Teaching, CIRL, and Deskilling.

---

### 🩺 1. Binary Decision Setting ($K = 2$) — The Theorem Engine
* [[2 Actions Reduction]]: Coordinate reduction to $d = u_1 - u_2$, $p = h_1 - h_2$, and epistemic classification.
* [[<G> - Autonomous Test Selection]]: Derivation of $F_{\\mathcal{H}}^*(s)$ via Value of Information; proof of symptom-invariance for fixed $K=2$.
* [[<G> - Graduation Regret & Crowding-Out]]: Exact Sheppard regret formula; proof of the Crowding-Out Law for loud distractions.
* [[<G> - Kingmaker & Curriculum Allocation]]: The **Monotone Panel Sieve (MPS)** algorithm; exact non-knapsack curriculum optimization.
* [[<G> - Ex-Post Dynamic Teaching]]: Runtime state observation; zero-regret teaching opportunities; Bellman stakes threshold rule.

---

### 🏥 2. Multi-Action Triage ($K \ge 3$) — Clinical Reality
* [[<G> - Multi-Action Formulation]]: Formalization of multi-class triage; latent vitals $x$, presenting symptoms $s$, and innovation covariance $\\Omega(F)$.
* [[<G> - Active Margin Theory]]: Proof of the **Active Margin Theorem**; how triage symptoms $s$ prune irrelevant diseases and dynamically dictate test choice.
* [[<G> - Multi-Action Pedagogical Algorithm]]: The **Margin-Cover Curriculum (MCC)** algorithm for training balanced clinical matrices across disease categories.

---

### 📊 3. Empirical Verification & Simulations
* [[<G> - Empirical Simulation Results]]: Full Monte Carlo simulations verifying Theorems 1–4, Sheppard formula, crowding-out phase transitions, and the Monotone Panel Sieve (MPS) performance.

---

### 📦 Archive
* `Archive/`: Historical drafts and baseline regression formulations.
