# Project Roadmap & Research Goals

This document tracks the current status of the project, inventories finalized versus AI-generated modules, and details active research targets.

---

## 1. Inventory of Files & Review Queue

The active repository consists of the following modules:

| Module | File | Status | Brief Summary |
| :--- | :--- | :--- | :--- |
| **Problem Setup** | `Problem Description.md` | Finalized | Mathematical framework: state $x \sim \mathcal{N}(0, \Sigma)$, observation $s=Bx$, budget $c$, utilities $U$, beliefs $H$, contraction dynamic $\psi$. |
| **K = 2 Reduction** | `2 Actions/1. 2 Actions Reduction.md` | Finalized | Difference vectors $d = u_1 - u_2$, $p = h_1 - h_2$; deterministic contraction dynamic; classification into informative, misunderstood, and distraction features. |
| **K = 2 Test Selection** | `2 Actions/2. Test Selection.md` | Finalized | Value of Information folded Gaussian derivation; proof that autonomous test selection $F^* = \arg\max p_F^\top \Gamma_{FF} p_F$ is symptom-independent. |
| **K = 2 Regret** | `2 Actions/3. Graduation Regret.md` | Finalized | Exact closed-form graduation regret via Sheppard's arcsine formula: $\mathcal{L}_{\text{grad}} = \sqrt{\frac{d^\top \Sigma d}{2\pi}}(1 - \rho(F^*, p))$. |
| **K = 2 Curriculum** | `2 Actions/4. Curriculum Allocation.md` | Finalized | Bilevel optimization program ($\lambda = 0$); feature salience $S(F; p)$; minimum cost of extinguishment $\kappa(F)$; polynomial-time candidate set sieve. |
| **K = 2 Dynamic Assistance** | `2 Actions/<G> - 5. Dynamic Assistance (Lambda > 0).md` | AI written -- to be reviewed | Dual objective ($\lambda > 0$); symptom prior momentum vs. innovation; zero-regret set $Z(x, s)$; Full Oracle DP; Symptom Clairvoyant DP; Online Dynamic Preceptor (ODP). |
| **K >= 2 Reduction** | `Multi-Action (K >= 3)/<G> - 1. Multi-Action Reduction.md` | AI written -- to be reviewed | Pairwise difference reduction ($d_{k\ell}, p_{k\ell}$); deterministic matrix contraction; cross-margin feature coupling phenomenon. |
| **K >= 2 Policy** | `Multi-Action (K >= 3)/<G> - 2. Multi-Action Test Selection.md` | AI written -- to be reviewed | Exact relative VoI representation; exact VoI sandwich bounds; active-contender pinching; autonomous deployment policy $s \mapsto F^*(s)$. |
| **K >= 2 Regret** | `Multi-Action (K >= 3)/<G> - 3. Multi-Action Graduation Regret.md` | AI written -- to be reviewed | Pointwise pairwise regret upper bound; observation space partition into polyhedral cones $\mathcal{S}_{k\ell}$; closed-form pairwise decomposition. |
| **K >= 2 Clairvoyant** | `Multi-Action (K >= 3)/<G> - 4. Dynamic Assistance - Clairvoyant.md` | AI written -- to be reviewed | Full Oracle DP ($x_{1:T}$ known) with zero-cost debiasing; Symptom Clairvoyant DP ($s_{1:T}$ known); active contender risk; Pedagogical Information Hierarchy. |
| **K >= 2 Online** | `Multi-Action (K >= 3)/<G> - 5. Dynamic Assistance - Online.md` | AI written -- to be reviewed | Multi-Action Online Preceptor (M-AOP); Pedagogical Index $Q_t$; real-time Shield vs. Teach mode switching; competitive ratio analysis. |

---

## 2. Active Targets & Next Steps

* **Human Review:** Review AI-written modules (`<G>` files) across binary dynamic assistance and multi-action theory.
* **Simulation Verification:** Implement Monte Carlo rollouts validating the theoretical regret bounds and benchmarking the Online Preceptor against Oracle, Symptom Clairvoyant, and baseline policies.
* **Conference Paper Preparation:** Package the finalized theoretical framework and empirical results into a conference submission draft.
