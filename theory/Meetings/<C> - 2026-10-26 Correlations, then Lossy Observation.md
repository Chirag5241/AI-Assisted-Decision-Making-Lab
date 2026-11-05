# <C> 2026-10-26 — Correlations, then Lossy Observation

Meeting notes (voice transcript + whiteboard). `<C>` = drafted by Claude; check before trusting.

![[2026-10-26 Whiteboard.jpg]]

---

## 1. Decisions

1. **Stage 1 — correlated $x$ only.** Keep the current framing of [[Problem Description]]; change only $\Sigma \ne I$. Theory first, then code + simulation.
2. **Stage 2 — lossy observation.** $\mathcal{A}$ picks $F_t$ from a lossy view of $x$ *before* $x_{F_t}$ is revealed.
3. **The observation map is fixed.** Deterministic, set by the environment, same for every round; $x$ varies, the map does not. Learning the map was raised and **not** adopted.
4. **Across runs, vary the map.** Compare several maps to see how $\mathcal{A}$ responds (figure direction; metrics not agreed).
5. **Cross-review AI-drafted theory** (`<G>`, `<C>`) before it counts.
6. **Share via GitHub**, not AirDrop.

Why staged: isolate correlation before adding partial observation; keep the amount of AI-drafted math to review small.

---

## 2. Whiteboard → vault notation

The board reuses $H$ for the human, the belief matrix, and the observation map. In the vault:

| Board | Vault | Meaning |
| :--- | :--- | :--- |
| "vitals" $x \in \mathbb{R}^n$ | $x \sim \mathcal{N}(0, \Sigma)$ | latent state |
| $H(x)$, "lossy, observable to $\mathcal{H}$" | $s = Bx$ | symptoms; fixed $B$ per environment |
| $A(H(x))$ | $\mathcal{A}(s^{(t)})$ | selector sees $s$, not $x$ |
| $F_t \subseteq [n],\ \lvert F_t \rvert < c$ | $\lvert F_t \rvert \le c$ | budget — **strict vs. inclusive unconfirmed** |
| $\{x_i : i \in F_t\}$ "seen by $\mathcal{H}$" | $x_{F_t}$ | revealed tests |
| $h^t(\{x_i : i \in F\}) \to$ learning | $H^{(t)} \to H^{(t+1)}$ | belief update on shown features |
| $H(x) \to \hat{x}$ | $\hat{x}$ | human's filled-in state |

Unplaced fragments: $U : F \to \Delta_a$, $L$, $a_i$.

So stage 2 is not new machinery: the vault already has $s = Bx$. Stage 1 is the vault with $B = 0$ (no symptoms) and general $\Sigma$.

---

## 3. Stage 1 — correlated $x$

**Setup (proposed).** [[Problem Description]] with $B = 0$, $x \sim \mathcal{N}(0, \Sigma)$, everything else unchanged ($U$, $c$, $\psi$, $\lambda$, $\delta$, graduation).

**Modeling choice to make first.** [[Problem Description]] fills hidden features with $\mu_j(s)$ — from $s$ only, never from $x_F$. Under correlation that throws information away. Options:

| Fill-in for $j \notin F$ | Comment |
| :--- | :--- |
| $\mu_j(s)$ (current) | With $B = 0$: hidden features $= 0$; correlation never enters the human's decision, only the loss. |
| $\mathbb{E}[x_j \mid x_F, s]$ | Guan et al. (2026) choice; hidden features count at the human's *wrong* weight. Used by the existing bench (`simlab/correlated.py`). |

Learning stays shown-only: a filled-in value never updates $H_{kj}$.

**The question.** Recorded as "does there exist a solution that is not a global optimum." Exact meaning **not agreed**. The likely reading — *is a stationary $F$ still globally optimal once $x$ is correlated?* (under MSE: true for independent $x$ in Noti et al., false for correlated $x$ in Guan et al.; open for decision regret) — must be confirmed before it is written as the target.

**Already in the repo.** The `/correlated` bench: $n = 3$, $\Sigma$ sliders, Gaussian imputation, every fixed subset vs. explore-then-commit for $K$ actions. It searches one policy family, so it is evidence, not a proof of optimality.

**Notes that assume $\Sigma = I$** (re-check for stage 1):
* [[<G> - Graduation Regret & Crowding-Out]] — "canonical standardized setting $x \sim \mathcal{N}(0, I_n)$".
* [[<G> - Ex-Post Dynamic Teaching]] — states $\mathcal{N}(0, \Sigma)$, then uses $\mathcal{N}(0, I_n)$.
* [[<G> - Kingmaker & Curriculum Allocation]] — per-feature $\sigma_j^2$ only (diagonal).
* [[<G> - Autonomous Test Selection]] — fine for general $\Gamma$; the additive decoupling needs diagonal $\Gamma$.

---

## 4. Stage 2 — lossy observation

$$x \;\longrightarrow\; s = Bx \;\xrightarrow{\;\mathcal{A}\;}\; F_t \;\longrightarrow\; x_{F_t} \;\longrightarrow\; \hat{y}_t,\ H^{(t+1)}$$

* **Within a run:** $B$ given, fixed, deterministic.
* **Across runs:** sweep $B$ — more / less / differently informative — and plot how $\mathcal{A}$'s choices and regret change.
* **Not established:** that $\mathcal{A}$ learns, chooses, or optimizes $B$.

Medical reading: true vitals $x$ → what the patient tells the doctor ($s$) → tests ordered ($F_t$) → results ($x_{F_t}$). No dataset named.

---

## 5. Action items

All open; owners not reliably attributable.

| ID | Action | Depends on |
| :--- | :--- | :--- |
| C1 | Formalize correlated $x$ in the current setup; pick the fill-in rule (§3). | — |
| C2 | Theory for C1; pin down the "global optimum" question. | C1 |
| C3 | Implement + simulate (extend the `/correlated` bench). | C1 |
| C4 | Cross-review derivations and code, AI-drafted parts first. | C2, C3 |
| H1 | Specify stage 2 (fixed $B$, timing of reveal). | stage 1 |
| H2 | Sweep $B$ across environments; figures. | H1 |
| D1 | Put work + notes on GitHub. | — |
| D2 | One Markdown explaining the work and assumptions. | — |

---

## 6. Open questions

1. What exactly is "a solution that is not a global optimum"?
2. Which correlations paper — Guan, Raman & Fang (2026)? Which assumptions carry over?
3. Budget: $\lvert F_t \rvert < c$ or $\le c$?
4. Does $\mathcal{A}$ know $B$ and $\Sigma$, or only see $s$?
5. Which metric compares observation maps?
6. Does correlation break the stationary-policy result?
