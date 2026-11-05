# <C> Sepsis Formulation — PhysioNet/CinC 2019

`<C>` = drafted by Claude; check before trusting. Code: `experiments/sepsis_data.py`, `experiments/sepsis_fit.py`, `simlab/sepsis.py`. Page: `/sepsis`.

[[Problem Description]] with $K = 2$, its $\Sigma$, $B$, $U$ fitted to real ICU data. $H$ stays random.

---

## 1. Data

PhysioNet/CinC Challenge 2019, training sets A and B: 40,336 ICU stays, one row per hour, 8 vitals + 26 labs + demographics, `SepsisLabel` (Sepsis-3, shifted 6 h earlier). CC-BY 4.0.

**One decision per patient.**
* Septic (7.3%): the first hour labelled 1 — 6 h before onset, the last on-time hour.
* Not septic: an hour drawn from the septic decision hours (cut to the stay), so time in the ICU does not separate the classes.
* Each variable: last value at or before that hour.

**Kept:** 7 vitals (EtCO2 never recorded) and the 23 labs measured by the decision hour in $\ge 20\%$ of patients (drops Bilirubin_direct, TroponinI, Fibrinogen).

---

## 2. Mapping

| Theory | Sepsis | How |
| :--- | :--- | :--- |
| $x \in \mathbb{R}^n$, $n = 30$ | 23 labs, then 7 vitals | risk scores (§3) |
| $\Sigma$ | their correlation | EM, Gaussian with missing values |
| $s = Bx$ | the vitals | $B = [\,0 \;\; I_7\,]$ |
| $F$, $\lvert F \rvert \le c$ | labs ordered | vitals are free, always seen |
| $K = 2$ | stay (treat) / go home | |
| $d = u_\text{stay} - u_\text{home}$ | logistic weights | §4 |
| $H^{(0)}$ | random | $p^{(0)} \sim \mathcal{N}(0, s^2 I)$, $s = \text{rms}(d)$ |
| $\psi(m)$ | $0.8^m$ | |

**Why $x$ holds the vitals too.** The vitals carry sepsis signal of their own (vitals-only AUROC 0.65). With $x$ = labs only and $s = Bx$, they would count only through $\mathbb{E}[\text{labs} \mid s]$. With $x$ = (labs, vitals) and $B$ a selection, $s = Bx$ holds exactly and the vitals enter $U$ directly. The labs are still the hidden part, revealed under budget $c$.

**Vitals are learned every round.** They are always seen, so $m_j(t) = t$ for $j \in V$. Lab $j$ is learned only when $j \in F_t$.

---

## 3. The state $x$: risk scores

Raw labs are skewed (WBC skew 16) and U-shaped in risk (temperature, WBC). Per variable, on the training patients only:
1. cut the measured values into 20 quantile bins;
2. assign each bin its smoothed log-odds of sepsis;
3. take normal scores of those values, then EM standardisation, so $x_j \sim \mathcal{N}(0,1)$.

So $x_j$ measures how septic lab $j$ looks, monotone in risk. Revealing lab $j$ reveals $x_j$. Missing values are filled with $\mathbb{E}[x_\text{miss} \mid x_\text{obs}]$ under $\Sigma$.

Raw normal scores instead gave linear AUROC 0.68 and a negative lactate weight.

---

## 4. $U$ from logistic regression

L2 logistic regression of sepsis on the filled-in $x$; penalty chosen by 5-fold CV on AUROC; 80/20 patient split.

$$\log\frac{P(\text{sepsis}\mid x)}{1 - P(\text{sepsis}\mid x)} = b + w^\top x, \qquad d := w, \qquad d_0 := b - \operatorname{logit}\tau.$$

**Threshold.** $\tau = \sigma(b) = 0.058$, the risk of the average patient. This gives $d_0 = 0$, which matches the theory's no-offset setting. It means treating is worth it when a miss costs $(1-\tau)/\tau \approx 16$ false alarms; the challenge's own scoring implies 40.

| Model (held-out 20%) | AUROC |
| :--- | :--- |
| linear, labs + vitals (**this $U$**) | **0.705** |
| linear, vitals only | 0.654 |
| linear, labs only | 0.665 |
| gradient boosting, filled-in $x$ | 0.751 |
| gradient boosting, raw values, gaps as NaN | 0.784 |
| linear, fit A → score B / B → A | 0.725 / 0.698 |

* **The 0.705 → 0.751 gap** is the cost of a linear utility.
* **The 0.751 → 0.784 gap** is what *which labs were ordered* says: informative missingness, which the model drops.

Every weight is positive (risk scores). The largest are Temp 0.33, Lactate 0.25, WBC 0.19, BaseExcess 0.19, BUN 0.19, HR 0.19 and Resp 0.18.

---

## 5. What follows (two actions, $d_0 = 0$)

Fill-in as in [[Problem Description]]: hidden labs get $\mu_L(s) = A s$, with $A = \Sigma_{LV}\Sigma_{VV}^{-1}$. The human's margin is linear in $x$:

$$\hat m = q^\top x, \qquad q_F = p_F, \quad q_{L \setminus F} = 0, \quad q_V = p_V + A_{L\setminus F}^\top\, p_{L \setminus F}.$$

$(d^\top x, q^\top x)$ is jointly normal with mean 0, so the expected regret is exact:

$$\mathbb{E}\,\ell_\text{reg} = \frac{1}{\sqrt{2\pi}}\left(\sqrt{d^\top \Sigma d} - \frac{d^\top \Sigma q}{\sqrt{q^\top \Sigma q}}\right)$$

This is `planning.expected_regret` with $\Sigma \neq I$. It is checked against 400k samples.

**Graduation.** The human orders $\hat F = \arg\max_{\lvert F\rvert \le c} p_F^\top \Gamma_{FF}\, p_F$, with $\Gamma = \Sigma_{LL} - A\Sigma_{VL}$ ([[<G> - Autonomous Test Selection]]). The choice is symptom-invariant, so every training policy here can name its panel before $x_t$ is drawn.

**Second fill-in: from the vitals and the shown labs** (Guan, Raman & Fang). `World.fill = "shown"`.

$$\hat x_L = \mathbb{E}[x_L \mid x_F, s] = As + \Gamma_{LF}\Gamma_{FF}^{-1}(x_F - A_F s).$$

Write $k := \Gamma_{FF}^{-1}(\Gamma p_L)_F$. Then

$$q_F = k, \qquad q_{L\setminus F} = 0, \qquad q_V = p_V + A^\top p_L - A_F^\top k.$$

The matching VoI rule is the variance of the margin given $s$:

$$\operatorname{Var}(\hat m \mid s) = (\Gamma p_L)_F^\top\, \Gamma_{FF}^{-1}\, (\Gamma p_L)_F.$$

**Claim (tested).** With $p = d$, $\hat m = \mathbb{E}[d^\top x \mid x_F, s]$. So $\operatorname{Cov}(d^\top x, \hat m) = \operatorname{Var}(\hat m)$, and the exact regret becomes

$$\frac{\sigma_{d^\top x} - \sigma_{\hat m}}{\sqrt{2\pi}}.$$

That is decreasing in $\operatorname{Var}(\hat m) = \operatorname{Var}(\mathbb{E}[d^\top x \mid s]) + \text{VoI}(F)$. Hence the true weights are optimal for every panel, and the panel valued most is the panel of least regret. Under the vitals-only fill-in neither holds.

---

## 6. Findings so far

1. **$B$ is nearly uninformative about labs.** The vitals explain 0–4% of each lab's variance ($1 - \Gamma_{jj}$), so $\mu_L(s) \approx 0$.
2. **The VoI rule misses the best panel, even with $p = d$.** It ranks labs by own variance and ignores lab–lab correlation, which the $\mu(s)$ fill-in throws away.

   | $c$ | VoI pick ($p = d$) | regret | best by regret | regret |
   | :--- | :--- | :--- | :--- | :--- |
   | 1 | Lactate | 0.134 | BUN | 0.112 |
   | 2 | BaseExcess, Lactate | 0.111 | BaseExcess, BUN | 0.089 |
   | 3 | BaseExcess, pH, Lactate | 0.099 | BaseExcess, pH, BUN | 0.078 |

   For reference: vitals only 0.140, every lab 0, guessing 0.384.

   With the shown labs also filling in, the pick *is* the best: BUN 0.106; BaseExcess, BUN 0.083; BaseExcess, BUN, WBC 0.072. Finding 2 is an artefact of the vitals-only fill-in.
3. **Crowding-out shows up on real data.** Teaching the expert's panel every round is the *worst* policy at graduation (0.19 at $c = 2, 3$): the human then orders labs they were never taught.
4. **A greedy teacher beats self-directed learning, under either fill-in.** The teacher shows the labs whose correction most cuts graduation regret. Graduation regret at $T = 40$:

   | Fill-in | Policy | $c = 1$ | $c = 2$ | $c = 3$ |
   | :--- | :--- | :--- | :--- | :--- |
   | vitals only | teacher | 0.124 | 0.102 | 0.084 |
   | vitals only | own panel | 0.128 | 0.114 | 0.105 |
   | vitals only | expert's panel | 0.171 | 0.189 | 0.187 |
   | vitals + shown labs | teacher | 0.119 | 0.094 | 0.077 |
   | vitals + shown labs | own panel | 0.128 | 0.117 | 0.108 |
   | vitals + shown labs | random panel | 0.149 | 0.113 | 0.088 |
   | vitals + shown labs | expert's panel | 0.181 | 0.183 | 0.174 |

   On held-out real patients the teacher's graduates have the lowest miss-weighted cost among trained humans, in every cell (0.761 at its best: vitals + shown labs, $c = 3$).
5. **With the shown labs filling in, crowding-out stays, and self-direction gets worse.** The expert's panel is still the worst. The human's own panel falls behind even a random panel at $c = 2, 3$: a lab valued wrongly now distorts the fill-in of every correlated lab too.

---

## 7. Assumptions to revisit

* Linear $U$ on risk scores: AUROC 0.705, against 0.784 for the best non-linear model.
* Gaussian $x$: risk scores are discrete (20 bins), and Gaussian only in their margins.
* Missing labs are filled in. Real ordering patterns are informative; that is ignored.
* One snapshot per patient. No trends, even though the challenge winners used them.
* $H^{(0)}$ is random, and the human knows $d_0 = 0$.
* Fill-in from $s$ only (theory) against $\mathbb{E}[x_H \mid x_F, s]$ (Guan et al.). With the second fill-in, and the VoI rule rewritten for it ($(\Gamma p)_F^\top \Gamma_{FF}^{-1} (\Gamma p)_F$), $p = d$ is Bayes-optimal for any panel and the VoI pick is the regret-best one, so finding 2 vanishes. Both are now run (§6); which one the human uses is roadmap item **C1**.
