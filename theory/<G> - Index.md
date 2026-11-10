# <G> Project Map: Pedagogical AI Assistance

This index outlines the active, streamlined reading flow for the CS 598 project on **Pedagogical AI Assistance and Feature-Revealing Curricula**.

---

## 1. Problem Formulation & Overview

* [[Problem Description]]: Core mathematical framework: latent state $x \sim \mathcal{N}(0, \Sigma)$, observation $s=Bx$, budget $c$, utilities $U$, beliefs $H$, contraction dynamic $\psi$.
* [[Project Roadmap & Goals]]: Inventory of completed files, active review queue, and next steps.
* [[<G> Related Work]]: Mapping to literature in machine teaching, cooperative inverse reinforcement learning, and AI assistance.

---

## 2. Binary Decision Setting ($K = 2$)

* [[1. 2 Actions Reduction]]: Reduction to difference vector $d = u_1 - u_2$, perceived vector $p = h_1 - h_2$, and feature classification.
* [[2. Test Selection]]: Perceived Value of Information derivation; proof that test selection $F^* = \arg\max p_F^\top \Gamma_{FF} p_F$ is symptom-independent for $K=2$.
* [[3. Graduation Regret]]: Exact closed-form expected graduation regret (bivariate-normal loss identity): $\mathcal{L}_{\text{grad}} = \sqrt{\frac{d^\top \Sigma d}{2\pi}}(1 - \rho(F^*, p))$.
* [[4. Curriculum Allocation]]: Offline curriculum allocation ($\lambda = 0$); feature salience $S(F; p)$; minimum cost of extinguishment $\kappa(F)$; polynomial-time candidate set sieve.
* [[<G> - 5. Dynamic Assistance (Lambda > 0)]]: Dynamic assistance when training mistakes carry cost ($\lambda > 0$); symptom prior momentum vs. innovation; zero-regret teachable moments; Full Oracle DP, Symptom Clairvoyant DP, and Online Dynamic Preceptor (ODP, heuristic).
* [[<G> - 6. Missing Assistance (Paused Learning)]]: The model can vanish at any round (survival $q$); graduation at a random time; every assisted round is charged $\lambda R$ plus hazard weight $w\,L$ on the state it leaves; closed-form costs when $B = 0$, $\Sigma$ diagonal, $c = 1$.
* [[<G> - 7. Continued Human Learning]]: Learning continues while alone: deterministic solo map $G(m) = m + e_{F^*(p(m))}$; solo value $W(m)$; distractions extinguish themselves, blind spots are never learned; solo loss can rise.
* [[<G> - 8. Returning Assistance]]: Intermittent model (stay $q$, return $r$) re-plans from $G^k(m)$; paused learning reduces to reweighted note 6; handoff-safe states and the model/human division of labour.

---

## 3. Multi-Action Setting ($K \ge 2$)

* [[<G> - 1. Multi-Action Reduction]]: Reduction to pairwise difference vectors $d_{k\ell}$ and $p_{k\ell}$; deterministic belief contraction; cross-margin feature coupling.
* [[<G> - 2. Multi-Action Test Selection]]: Exact relative VoI representation; exact VoI sandwich bounds; active-contender reduction; autonomous deployment policy $s \mapsto F^*(s)$.
* [[<G> - 3. Multi-Action Graduation Regret]]: Pointwise pairwise regret upper bound; partition of observation space into polyhedral cones $\mathcal{S}_{k\ell}$; closed-form active-contender decomposition.
* [[<G> - 4. Dynamic Assistance - Clairvoyant]]: Offline dynamic assistance benchmarks ($\lambda > 0$); Full Oracle DP ($x_{1:T}$ known) with zero-cost debiasing; Symptom Clairvoyant DP ($s_{1:T}$ known); active contender risk; Pedagogical Information Hierarchy.
* [[<G> - 5. Dynamic Assistance - Online]]: Real-time online pedagogical assistance ($\lambda > 0$); Multi-Action Online Preceptor (M-AOP); Online Pedagogical Index $Q_t$; real-time Shield vs. Teach mode switching; competitive-ratio claim retracted (open).
* [[<G> - 6. Adaptive Tutoring & Placement Test]]: Adaptive tutoring without prior knowledge of $H^{(0)}$ or $\psi$; unassisted multiclass placement testing via polyhedral slicing; scale-invariance theorem; dynamic stopping time $\tau$ (adaptive graduation).

---

## 4. LLM Assistance (Natural Language Scaffolding)

* [[<G> - 1. Problem Formulation & Two-Stage Architecture]]: Pure decision-theoretic formulation of LLM advice $M_t = (\alpha_t, a_t, \hat{k}_t)$, continuous active engagement $G_j(t)$, cognitive offloading law, and the two-stage (Controller + Verbalizer) architecture.
* [[<G> - 2. Optimal Scaffolding Policy (Known Parameters)]]: Optimal dynamic policy under known parameters; LLM Preceptor Index $\nu_j(t)$; water-filling attention $a_t^*$; Socratic switching threshold $\alpha_t^*$; Vygotsky fading theorem; proof of the "Helpful AI Trap."
* [[<G> - 3. Adaptive Estimation & In-Context Scaffolding (Unknown Parameters)]]: Learning student parameters online; diagnostic polyhedral probing for $H^{(0)}$; latency regression for working-memory capacity $C$; sequential Bayesian particle filtering; dynamic graduation stopping time $\tau$.
* [[<G> - 4. Out-of-the-Box LLM Verbalizer & Prompt Engine]]: Stage 2 prompt compiler; universal system prompt template enforcing $(\alpha_t^*, a_t^*, \hat{k}_t^*)$ while adapting to user style preferences $(V, \tau_{\text{tone}})$; concrete Spanish *ser* vs. *estar* execution transcripts.
* [[<G> - 5. Core Tradeoffs & Fundamental Theorems]]: Five foundational theorems distinguishing pedagogical discrete assistance from continuous MSE: The Kingmaker non-monotonicity trap, Dangerous Medicine trade-off, Price of Helpfulness Pareto frontier, Zero-Regret Teachable Moments, and breakdown of policy stationarity.
* [[<G> - 6. Latent Preference Alignment & Online Learning]]: Personalized steerable alignment via continuous latent preference vector $z \in \mathbb{R}^p$; online Bayesian / Kalman updating from interaction telemetry; Stage 2 constrained alignment optimization.

---

## 5. Meetings

* [[<C> - 2026-10-26 Correlations, then Lossy Observation]]: Next steps — correlated $x$ first, then a fixed lossy observation $s = Bx$; whiteboard and action items.

---

## 6. Archive

* `Archive/`: Preserved historical drafts, superseded derivations, and earlier working notes.
