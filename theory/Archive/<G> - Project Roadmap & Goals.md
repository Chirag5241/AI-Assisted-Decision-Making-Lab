# Project Roadmap & Research Goals

This document tracks the current status of the project, inventories finalized versus drafted modules, and details the active theoretical and algorithmic targets.

---

## 1. Inventory of Files & Review Status

| File | Status | Description |
| :--- | :--- | :--- |
| `Problem Description.md` | **Finalized (Human Written)** | Core clinical CDS framework: latent vitals $x$, symptoms $s=Bx$, budget $c$, autonomous graduation. |
| `2 Actions/2 Actions Reduction.md` | **Finalized (Human Written)** | Binary coordinate reduction ($d = u_1 - u_2$, $p = h_1 - h_2$), epistemic contraction, feature classification. |
| `2 Actions/<G> - Autonomous Test Selection.md` | **Draft (Under Review)** | Value of Information folded Gaussian derivation; proof of symptom-invariance for fixed $K=2$. |
| `2 Actions/<G> - Graduation Regret & Crowding-Out.md` | **Draft (Under Review)** | Exact Sheppard regret formula; proof of the Crowding-Out Law for loud distractions. |
| `2 Actions/<G> - Kingmaker & Curriculum Allocation.md` | **Draft (New Theory)** | Novel Monotone Panel Sieve algorithm; exact non-knapsack curriculum optimization for general $c$. |
| `2 Actions/<G> - Ex-Post Dynamic Teaching.md` | **Draft (New Theory)** | Dynamic runtime teaching algorithm; Bellman threshold policy on stakes $|d^\top x|$; value monotonicity. |
| `Multi-Action (K >= 3)/<G> - Multi-Action Formulation.md` | **Draft (New Theory)** | Formalization of $K \ge 3$ differential diagnosis; symptom-conditioned Gaussian innovation vector $\tilde{u}$. |
| `Multi-Action (K >= 3)/<G> - Active Margin Theory.md` | **Draft (New Theory)** | Theorem proving how symptoms $s$ activate top competing disease margins and dictate test selection $\hat{F}(s)$. |
| `Multi-Action (K >= 3)/<G> - Multi-Action Pedagogical Algorithm.md` | **Draft (New Theory)** | Margin-Cover Curriculum (MCC) algorithm for training multi-class diagnostic matrices without blind spots. |

---

## 2. Core Research Goals & Theoretical Targets

### Goal 1: Beyond Standard Reductions (Novel Algorithmic Contributions)
* We do **not** merely reduce curriculum allocation to standard 0–1 Knapsack and call an existing solver.
* We develop **domain-tailored algorithms** that exploit the structural properties unique to pedagogical assistance:
  1. **Monotone Saliency Separation**: In autonomous graduation, chosen panels are determined by relative salience ordering $p_j(m_j)^2 > p_k(m_k)^2$, creating an exact threshold geometry.
  2. **Contraction Duality**: The deterministic contraction $p_j(m_j) = (1 - \psi(m_j))d_j + \psi(m_j)p_j^{(0)}$ imposes strict monotonicity constraints that eliminate vast swaths of the combinatorial search space.
  3. **Zero-Regret State Coupling**: Under ex-post state observation, Sheppard's arcsine law enables free teaching at zero operational cost.

### Goal 2: Multi-Action Triage ($K \ge 3$)
* Establish why $K=2$ is symptom-invariant (single specialized clinic) whereas $K \ge 3$ is **dynamically symptom-driven** (general emergency triage).
* Prove the **Active Margin Theorem**: symptoms $s$ filter the $K$ choices down to the active differential diagnosis, reducing multi-action test selection to pairwise margin saliency.
* Formulate the **Margin-Cover Curriculum (MCC)** algorithm to schedule teaching across multiple competing diseases.

### Goal 3: Publications & Translational Product Impact
* **Target Venues**: ACM EC (Economics and Computation), NeurIPS, or CHIL (Conference on Health, Inference, and Learning).
* **Clinical Product**: Pedagogical Clinical Decision Support (P-CDS)—an EHR residency preceptor that eliminates physician deskilling and automation bias by training doctors to autonomous competence.
