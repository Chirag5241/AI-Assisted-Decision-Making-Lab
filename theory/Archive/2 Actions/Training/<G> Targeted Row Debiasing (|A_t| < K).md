# <G> Targeted Row Debiasing (|A_t| < K)

This note explores **Action Menu Restriction** (scaffolding) as a pedagogical tool: restricting the menu of actions $A_t \subset [K]$ presented to the human decision-maker.

---

## 1. Motivation: Error Cancellation across Rows

In multi-action choice, a human's relative preference between two actions depends on the difference between row estimates:
$$g_j = h_{1j} - h_{2j}$$
If both actions are shown ($A_t = \{1, 2\}$), both rows update simultaneously according to the learning dynamic:
$$h_{kj}^{(t+1)} - u_{kj} = \psi(m_{kj})(h_{kj}^{(0)} - u_{kj})$$
However, when the human's prior errors are **asymmetric across actions**, updating both rows causes the errors to subtract against each other, potentially **slowing down the sign flip** of $g_j$.

---

## 2. Mathematical Formalization ($K=2$)

Fix a single feature $j$ with true difference $d_j = u_{1j} - u_{2j} > 0$.
Define the initial row errors:
$$e_{1j} := h_{1j}^{(0)} - u_{1j}, \qquad e_{2j} := h_{2j}^{(0)} - u_{2j}$$
The perceived difference after $m_{1j}$ showings of action 1 and $m_{2j}$ showings of action 2 is:
$$g_j(m_{1j}, m_{2j}) = d_j + \psi(m_{1j}) e_{1j} - \psi(m_{2j}) e_{2j}$$

Initially ($m_{1j} = m_{2j} = 0$):
$$g_j(0, 0) = d_j + (e_{1j} - e_{2j})$$
Assume the human starts with a wrong sign: $g_j(0, 0) < 0$, which occurs when:
$$e_{2j} - e_{1j} > d_j$$

---

## 3. Acceleration Theorem

### Proposition: Single-Row Updating Can Strictly Accelerate Sign Flips
Suppose the human overestimates both actions, with $0 < e_{1j} < e_{2j}$ and $e_{2j} - e_{1j} > d_j > 0$.
1. **Updating Both Rows ($A_t = \{1, 2\}$)**:
   $$g_j(m, m) = d_j - \psi(m)(e_{2j} - e_{1j}) > 0 \iff \psi(m) < \frac{d_j}{e_{2j} - e_{1j}}$$
   requiring count:
   $$\tau_j^{\text{both}} := \min\left\{ m \ge 0 : \psi(m) < \frac{d_j}{e_{2j} - e_{1j}} \right\}$$

2. **Updating Only Row 2 ($A_t = \{2\}$)**:
   Action 1 is held fixed at its initial belief ($m_{1j} = 0$), so:
   $$g_j(0, m_2) = d_j + e_{1j} - \psi(m_2) e_{2j} > 0 \iff \psi(m_2) < \frac{d_j + e_{1j}}{e_{2j}}$$
   requiring count:
   $$\tau_j^{\text{row 2}} := \min\left\{ m \ge 0 : \psi(m) < \frac{d_j + e_{1j}}{e_{2j}} \right\}$$

### Proof of Acceleration:
We compare the two thresholds. Notice that:
$$\frac{d_j + e_{1j}}{e_{2j}} - \frac{d_j}{e_{2j} - e_{1j}} = \frac{(d_j + e_{1j})(e_{2j} - e_{1j}) - d_j e_{2j}}{e_{2j}(e_{2j} - e_{1j})} = \frac{e_{1j}(e_{2j} - e_{1j} - d_j)}{e_{2j}(e_{2j} - e_{1j})}$$
Since $e_{1j} > 0$ and $(e_{2j} - e_{1j}) > d_j$, this difference is **strictly positive**:
$$\frac{d_j + e_{1j}}{e_{2j}} > \frac{d_j}{e_{2j} - e_{1j}}$$
Because $\psi(m)$ is non-increasing, crossing a strictly higher threshold requires fewer (or equal) exposures:
$$\tau_j^{\text{row 2}} \le \tau_j^{\text{both}}$$

#### Concrete Numerical Example:
* Let $d_j = 1$, $e_{1j} = 1$, $e_{2j} = 10$. Then $g_j^{(0)} = 1 + 1 - 10 = -8 < 0$.
* Let learning be geometric: $\psi(m) = 0.6^m$.
* **Two-row threshold**: $\psi(m) < \frac{1}{10 - 1} = \frac{1}{9} \approx 0.111 \implies 0.6^m < 0.111 \implies \mathbf{m = 5}$.
* **Single-row threshold**: $\psi(m) < \frac{1 + 1}{10} = 0.200 \implies 0.6^m < 0.200 \implies \mathbf{m = 4}$.
* **Result**: Restricting the menu flips the sign in 4 rounds instead of 5!

---

## 4. Scaffolding Taxonomy: Answering vs. Scaffolding vs. Full Menu

| Action Menu Size | Educational Analogue | Immediate Regret | Learning Mechanism |
| :--- | :--- | :--- | :--- |
| **$|A_t| = 1$ ($A_t = \{y^*(x)\})$** | **Showing the Answer** | **$0$ always** (no mistake possible) | Updates only row $y^*$; prevents error cancellation on non-optimal rows. |
| **$1 < |A_t| < K$** | **Scaffolding / Hinting** | Intermediate | Isolates subsets of confusing actions, focusing human attention on critical pairwise distinctions. |
| **$|A_t| = K$** | **Unassisted / Full Menu** | Highest exposure risk | Updates all rows simultaneously, but subject to cross-row error masking. |

> [!TIP] Pedagogical Implication
> Showing a single option (forcing the correct move) is not just a protective crutch—it is an **active debiasing tool** when misconceptions about other options are currently worse than the misconception about the correct option.
