# Project Roadmap & Research Goals

This document tracks the current status of the project, inventories finalized versus drafted modules, and details active theoretical and algorithmic targets.

---

## 1. Inventory of Files & Review Queue (Non-Redundant)

All redundant and superseded drafts have been moved to `Archive/`. The active reading queue consists strictly of distinct, non-overlapping modules:

| Module | File | Status | Core Contribution |
| :--- | :--- | :--- | :--- |
| **Problem Setup** | `Problem Description.md` | **Finalized (Human Written)** | Core clinical CDS framework: latent vitals $x$, symptoms $s=Bx$, budget $c$, autonomous graduation. |
| **K = 2 Setup** | `2 Actions/2 Actions Reduction.md` | **Finalized (Human Written)** | Binary coordinate reduction ($d = u_1 - u_2$, $p = h_1 - h_2$), epistemic contraction, feature classification. |
| **K = 2 VoI** | `2 Actions/<G> - Autonomous Test Selection.md` | **Ready for Review** | Value of Information folded Gaussian derivation; proof of symptom-invariance for fixed $K=2$. |
| **K = 2 Regret** | `2 Actions/<G> - Graduation Regret & Crowding-Out.md` | **Ready for Review** | Exact Sheppard regret formula; proof of the Crowding-Out Law for loud distractions. |
| **K = 2 Curriculum**| `2 Actions/<G> - Kingmaker & Curriculum Allocation.md` | **Ready for Review** | The **Monotone Panel Sieve (MPS)** algorithm; exact non-knapsack curriculum optimization for general $c$. |
| **K = 2 Runtime** | `2 Actions/<G> - Ex-Post Dynamic Teaching.md` | **Ready for Review** | Dynamic runtime teaching algorithm; Bellman threshold policy on stakes $|d^\top x|$; zero-regret teaching. |
| **K >= 3 Setup** | `Multi-Action (K >= 3)/<G> - Multi-Action Formulation.md` | **Ready for Review** | Formalization of $K \ge 3$ differential diagnosis; symptom-conditioned Gaussian innovation vector $\tilde{u}$. |
| **K >= 3 Saliency**| `Multi-Action (K >= 3)/<G> - Universal Saliency Theorem.md` | **Ready for Review** | **Closed-form Log-Sum-Exp Hessian expansion** proving universal margin-weighted saliency and exponential pruning for any $K \ge 2$. |
| **K >= 3 Curriculum**| `Multi-Action (K >= 3)/<G> - Multi-Action Pedagogical Algorithm.md` | **Ready for Review** | The upgraded **Universal Saliency Gradient (USG)** algorithm tracking exact population VoI gradients. |
| **Joint Scaffolding**| `Multi-Action (K >= 3)/<G> - Joint Action-Feature Scaffolding.md` | **Ready for Review** | Optimal joint $(A_t, F_t)$ curriculum policy; proof that minimal binary candidate menus maximize safe debiasing rate. |
| **Memory Decay** | `Theory Extensions/<G> - Memory Decay & Restless Bandits.md` | **Ready for Review** | Solving cognitive forgetting via **Restless Multi-Armed Bandits**; proof of Whittle Indexability. |
| **Simulations** | `Simulations/<G> - Empirical Simulation Results.md` | **Ready for Review** | Full Monte Carlo simulations verifying Theorems 1–4 ($R^2 = 0.99958$), crowding-out transitions, and MPS. |

---

## 2. Active Targets & Open Directions

* [ ] **Human Review**: Review the clean queue above from `<G>` drafts to finalized modules.
* [ ] **Non-Gaussian Robustness (Copula Transport / Berry-Esseen)**: Establish formal bounds for discrete vital sign scores (e.g., categorical Glasgow Coma Scale) via central limit projections.
* [ ] **Empirical Benchmark on MIMIC-IV**: Adapt simulation scripts to run on real-world emergency triage EHR data (e.g., MIMIC-IV ED triage vitals and ICD-10 dispositions).
* [ ] **Conference Paper Draft**: Package the theory and empirical simulations into an ACM EC / NeurIPS format monograph.
