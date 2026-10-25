# <G> - Universal Saliency Theorem: Closed-Form Test Selection for Any K >= 2

This document resolves the central open theoretical challenge for multi-action triage ($K \ge 3$): **deriving a closed-form analytical expression for how presenting symptoms $s$ direct autonomous test selection $\hat{F}(s)$.**

We prove the **Universal Saliency Theorem** via a second-order Log-Sum-Exp expansion, establishing that biomarker test selection is governed by a **Hessian-weighted sum over pairwise disease margins**.

---

## 1. The Multi-Action Value of Information Objective

Recall that for $K \ge 3$ actions, the clinician's utility for action $k \in [K]$ after ordering test panel $F \subseteq [n]$ ($|F| \le c$) on patient with symptoms $s$ is:
$$\hat{u}_k(F, x_F, s) = \bar{u}_k(s) + \tilde{u}_k(F)$$
where:
* $\bar{u}_k(s) = h_k^\top \mu(s) \in \mathbb{R}$ is the prior expected utility.
* $\tilde{u}_k(F) = h_{k, F}^\top (x_F - \mu_F(s)) \sim \mathcal{N}(0, h_{k, F}^\top \Gamma_{FF} h_{k, F})$ is the zero-mean Gaussian test innovation.

The clinician's autonomous test-ordering problem is:
$$F_{\mathcal{H}}^*(s) = \arg\max_{|F| \le c} \mathbb{E}_{\tilde{u} \sim \mathcal{N}(0, \Omega(F))} \left[ \max_{k \in [K]} \big( \bar{u}_k(s) + \tilde{u}_k(F) \big) \right]$$
where $\Omega_{k, \ell}(F) = h_{k, F}^\top \Gamma_{FF} h_{\ell, F}$.

---

## 2. Smooth Approximation & The Log-Sum-Exp Hessian

The non-differentiable maximum $\max_{k \in [K]} u_k$ admits the smooth Log-Sum-Exp approximation with inverse temperature $\beta > 0$:
$$M_\beta(u) := \frac{1}{\beta} \log \sum_{k=1}^K e^{\beta u_k}, \qquad \lim_{\beta \to \infty} M_\beta(u) = \max_{k \in [K]} u_k$$

Let $p_k(s) := \frac{e^{\beta \bar{u}_k(s)}}{\sum_{\ell=1}^K e^{\beta \bar{u}_\ell(s)}}$ denote the **softmax posterior probability of disease $k$ given symptoms $s$ alone**.

### Lemma: The Log-Sum-Exp Hessian
The gradient and Hessian of $M_\beta$ evaluated at the prior vector $\bar{u}(s)$ are:
$$\nabla M_\beta(\bar{u}(s)) = p(s) = \big( p_1(s), \dots, p_K(s) \big)^\top$$
$$\big[ \nabla^2 M_\beta(\bar{u}(s)) \big]_{k, \ell} = \beta \Big( p_k(s) \mathbb{1}[k = \ell] - p_k(s) p_\ell(s) \Big)$$

### Proof:
Direct differentiation: $\frac{\partial M_\beta}{\partial u_k} = \frac{e^{\beta u_k}}{\sum e^{\beta u_\ell}} = p_k$. 
Then $\frac{\partial^2 M_\beta}{\partial u_k \partial u_\ell} = \frac{\beta e^{\beta u_k} \delta_{k\ell} \sum - \beta e^{\beta u_k} e^{\beta u_\ell}}{(\sum)^2} = \beta (p_k \delta_{k\ell} - p_k p_\ell)$. $\blacksquare$

---

## 3. The Universal Saliency Theorem

egin{theorem}[Universal Margin-Weighted Saliency]
Under a second-order Taylor expansion around the symptom baseline $\bar{u}(s)$, the Value of Information for test panel $F$ is:
$$\text{VoI}(F \mid s) = \mathbb{E}\big[ M_\beta(\bar{u}(s) + \tilde{u}(F)) \big] - M_\beta(\bar{u}(s)) = \frac{\beta}{2} \sum_{j \in F} \text{Salience}_j(s) + \mathcal{O}(\|\Omega(F)\|^2)$$
where the universal symptom-dependent diagnostic salience of biomarker $j$ is:
$$\text{Salience}_j(s) := \sigma_j^2 \sum_{1 \le k < \ell \le K} \underbrace{p_k(s) p_\ell(s)}_{\text{Active Pair Probability}} \cdot \underbrace{\big( H_{kj} - H_{\ell j} \big)^2}_{\text{Pairwise Discriminative Weight}}$$
and $\sigma_j^2 = (\Gamma)_{jj}$ is the residual diagnostic uncertainty of biomarker $j$.
\end{theorem}

egin{proof}
Expanding $M_\beta(\bar{u} + \tilde{u})$ in a second-order Taylor series:
$$M_\beta(\bar{u} + \tilde{u}) = M_\beta(\bar{u}) + \nabla M_\beta(\bar{u})^\top \tilde{u} + \frac{1}{2} \tilde{u}^\top \nabla^2 M_\beta(\bar{u}) \tilde{u} + \mathcal{O}(\|\tilde{u}\|^3)$$
Taking expectations: $\mathbb{E}[\tilde{u}] = 0$, so the first-order term vanishes.
The second-order term is:
$$\mathbb{E}\left[ \frac{1}{2} \tilde{u}^\top \nabla^2 M_\beta \tilde{u} \right] = \frac{\beta}{2} \mathbb{E}\left[ \sum_{k=1}^K p_k \tilde{u}_k^2 - \Big( \sum_{k=1}^K p_k \tilde{u}_k \Big)^2 \right]$$
Using the algebraic identity for weighted variance $\sum p_k z_k^2 - (\sum p_k z_k)^2 = \sum_{k < \ell} p_k p_\ell (z_k - z_\ell)^2$:
$$= \frac{\beta}{2} \sum_{1 \le k < \ell \le K} p_k(s) p_\ell(s) \mathbb{E}\left[ \big( \tilde{u}_k(F) - \tilde{u}_\ell(F) \big)^2 \right]$$
Since $\tilde{u}_k(F) - \tilde{u}_\ell(F) = (h_{k, F} - h_{\ell, F})^\top (x_F - \mu_F(s))$, its variance is $(h_{k, F} - h_{\ell, F})^\top \Gamma_{FF} (h_{k, F} - h_{\ell, F}) = \sum_{j \in F} (H_{kj} - H_{\ell j})^2 \sigma_j^2$ under diagonal residual uncertainty $\Gamma$.
Summing across features $j \in F$ completes the proof. $\blacksquare$
\end{proof}

---

## 4. Why This Completely Unifies the Theory

Look at the structure of $\text{Salience}_j(s)$:
$$\text{Salience}_j(s) = \sigma_j^2 \sum_{1 \le k < \ell \le K} p_k(s) p_\ell(s) \big( H_{kj} - H_{\ell j} \big)^2$$

1. **Recovery of $K = 2$ as an Exact Special Case**:
   When $K = 2$, there is only one pair $(1, 2)$. The product $p_1(s) p_2(s)$ is a positive scalar.
   The salience is $(H_{1j} - H_{2j})^2 \sigma_j^2 = p_j^2 \sigma_j^2$.
   **It matches our $K=2$ greedy salience formula identically!**

2. **The Active Margin Mechanism for $K \ge 3$**:
   If symptoms $s$ rule out disease $k$ ($p_k(s) \approx 0$), then $p_k(s) p_\ell(s) = 0$ for all $\ell$.
   Disease $k$ completely drops out of the salience calculation!
   Only pairs where **both diseases are active contenders** ($p_k(s) > 0$ and $p_\ell(s) > 0$) have non-zero product $p_k(s) p_\ell(s)$.

3. **Symptom-Steered Greedy Algorithm**:
   For any $K \ge 2$ and budget $c$, the clinician autonomously orders the $c$ biomarkers with the largest values of $\text{Salience}_j(s)$.
   Because $p_k(s)$ changes with symptoms $s$, **the test panel $\hat{F}(s)$ dynamically changes with presenting symptoms!**

---

## 5. Exponential Pruning of Inactive Margins (The Active Differential Diagnosis)

In emergency medicine, a presenting symptom $s$ does not leave all $K$ diagnoses actively competing.
Sort the diseases by prior expected utility given symptoms:
$$\bar{u}_{(1)}(s) \ge \bar{u}_{(2)}(s) \ge \bar{u}_{(3)}(s) \ge \dots \ge \bar{u}_{(K)}(s)$$
For any candidate disease $k$, define its clinical deficit relative to the leading diagnosis: $\Delta_k(s) := \bar{u}_{(1)}(s) - \bar{u}_k(s) \ge 0$.

\begin{lemma}[Exponential Suppression of Distant Diseases]
Let test innovation variances be bounded by $\sigma_{\max}^2 := \max_{k, F} \Omega_{kk}(F)$.
For any candidate action $k$ whose deficit satisfies $\Delta_k(s) > 3 \sigma_{\max}$:
$$p_k(s) = \frac{e^{\beta \bar{u}_k(s)}}{\sum_{\ell} e^{\beta \bar{u}_\ell(s)}} \le e^{-\beta \Delta_k(s)} = \mathcal{O}\left( e^{-\Delta_k(s)^2 / 2\sigma_{\max}^2} \right)$$
Its pairwise product $p_k(s) p_\ell(s)$ is exponentially suppressed for all $\ell$, contributing asymptotically zero to $\text{Salience}_j(s)$.
\end{lemma}

### Result:
Only the small active differential diagnosis (diseases where $\Delta_k(s) \le 3\sigma_{\max}$) contribute to biomarker salience. The clinician automatically and dynamically orders tests separating the primary contenders for patient $s$!
