# <G> - Active Margin Theory: How Symptoms Direct Test Selection

This document proves the **Active Margin Theorem**: why and how presenting symptoms $s$ filter the $K$-action space down to the active differential diagnosis, dynamically directing which lab tests are ordered.

---

## 1. The Clinical Differential Diagnosis

In emergency medicine, a patient presenting with symptoms $s$ does not have all $K$ diagnoses actively competing.
Instead, triage symptoms immediately rule out the vast majority of conditions, leaving a small **active differential diagnosis** of top candidate actions.

Let the actions be sorted by prior expected utility:
$$\bar{u}_{(1)}(s) \ge \bar{u}_{(2)}(s) \ge \bar{u}_{(3)}(s) \ge \dots \ge \bar{u}_{(K)}(s)$$
where $\bar{u}_k(s) = h_k^\top \mu(s)$.

### Definition: Contender Gap
For any action $k \in [K]$, define its clinical deficit relative to the leading diagnosis:
$$\Delta_k(s) := \bar{u}_{(1)}(s) - \bar{u}_k(s) \ge 0$$

---

## 2. The Active Margin Theorem

egin{theorem}[Exponential Pruning of Inactive Margins]
Let test innovation variances be bounded by $\sigma_{\max}^2 := \max_{k, F} \Omega_{kk}(F)$.
For any candidate action $k$ whose deficit satisfies $\Delta_k(s) > 3 \sigma_{\max}$:
$$P\big( \hat{u}_k(F) > \hat{u}_{(1)}(F) \big) = P\big( \tilde{u}_k(F) - \tilde{u}_{(1)}(F) > \Delta_k(s) \big) < \Phi(-3) \approx 0.0013$$
The contribution of action $k$ to the Value of Information is bounded by:
$$\mathbb{E}\big[ \max(\hat{u}_k, \hat{u}_{(1)}) - \hat{u}_{(1)} \big] \le \sigma_{\max} \phi\left( \frac{\Delta_k(s)}{\sigma_{\max}} \right) = O\left( e^{-\Delta_k(s)^2 / 2\sigma_{\max}^2} \right)$$
\end{theorem}

egin{proof}
Let $Z = \tilde{u}_k - \tilde{u}_{(1)} \sim \mathcal{N}(0, \sigma_Z^2)$ with $\sigma_Z \le 2\sigma_{\max}$.
Then $\mathbb{E}[\max(0, Z - \Delta_k)] = \sigma_Z \phi(\Delta_k/\sigma_Z) - \Delta_k \Phi(-\Delta_k/\sigma_Z)$.
Using the standard Mills ratio bound $\Phi(-z) < \frac{\phi(z)}{z}$ for $z > 0$, the quantity decays as $e^{-z^2/2}$, proving exponential suppression. $\blacksquare$
\end{proof}

---

## 3. The Active Pairwise Margin Reduction

When symptoms $s$ leave two primary contending diagnoses $k_1(s)$ and $k_2(s)$ while suppressing all others ($\Delta_k(s) \gg \sigma_{\max}$ for $k \notin \{k_1, k_2\}$):

### Corollary (Emergence of the Dynamic Margin Vector)
The multi-action Value of Information collapses to the active pairwise difference vector:
$$p^{\text{active}}(s) := h_{k_1(s)} - h_{k_2(s)} \in \mathbb{R}^n$$
and the autonomous test selection rule is:
$$F_{\mathcal{H}}^*(s) \approx \arg\max_{|F| \le c} \big(p_F^{\text{active}}(s)\big)^\top \Gamma_{FF} \, p_F^{\text{active}}(s)$$

### Fundamental Clinical Insight:
* When a patient presents with **Chest Pain** ($s_{\text{chest}}$), symptoms make $k_1 = \text{Heart}$ and $k_2 = \text{Discharge}$. The active margin is $p^{\text{cardiac}}$, ordering **Troponin**.
* When a patient presents with **Acute Headache** ($s_{\text{head}}$), symptoms make $k_1 = \text{Stroke}$ and $k_2 = \text{Discharge}$. The active margin is $p^{\text{neuro}}$, ordering **Head CT**.

This rigorously resolves the problem: **symptoms $s$ dynamically select which disease margin is active, which dictates which tests are ordered!**
