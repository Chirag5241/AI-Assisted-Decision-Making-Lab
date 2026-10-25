# <G> - Ex-Post Dynamic Teaching: The Bellman Stakes Threshold Policy

This document formalizes the **dynamic runtime teaching policy** when the AI preceptor observes the patient's realized latent state $x^{(t)}$ and triage symptoms $s^{(t)}$ at round $t$ before selecting the training panel $F_t$.

We prove that runtime state observation allows the AI to exploit **zero-regret teaching opportunities** via Sheppard's Arcsine Law, and establish the exact **Bellman Stakes Threshold Policy**.

---

## 1. Runtime State Observation & The Zero-Regret Set

At round $t \le T$, nature realizes patient $x = x^{(t)} \sim \mathcal{N}(0, \Sigma)$.
The true optimal decision is Action 1 iff $D := d^\top x > 0$.
The clinician holds belief vector $p = p(m)$.
If the AI presents feature $j \in [n]$ ($c=1$), the clinician predicts Action 1 iff $p_j x_j > 0$.

### Definition: The Zero-Regret Teaching Set $Z(x)$
A feature $j \in [n]$ belongs to the **zero-regret set** $Z(x)$ if showing feature $j$ leads to the correct clinical decision:
$$Z(x) := \Big\{ j \in [n] : \operatorname{sgn}(p_j x_j) = \operatorname{sgn}(d^\top x) \Big\}$$
Showing any feature $j \in Z(x)$ incurs **exactly zero immediate decision regret** on patient $x$:
$$\ell_{\text{reg}}(x, j) = |d^\top x| \cdot \mathbb{1}[j \notin Z(x)] = 0$$

---

## 2. Sheppard's Arcsine Law for Zero-Regret Teaching

egin{theorem}[Exact Probability of Free Teaching]
Let $x \sim \mathcal{N}(0, I_n)$. For any biomarker $j \in [n]$:
1. If biomarker $j$ is currently **misunderstood** ($p_j d_j < 0$), it provides a zero-regret teaching opportunity with strictly positive probability:
   $$P\big(j \in Z(x)\big) = \frac{1}{2} - \frac{1}{\pi} \arcsin\left( \frac{|d_j|}{\|d\|_2} \right) > 0$$
2. If biomarker $j$ is currently **informative** ($p_j d_j > 0$), it provides a zero-regret decision with probability:
   $$P\big(j \in Z(x)\big) = \frac{1}{2} + \frac{1}{\pi} \arcsin\left( \frac{|d_j|}{\|d\|_2} \right) > \frac{1}{2}$$
3. The zero-regret set is empty ($Z(x) = \emptyset$) with exponentially small probability $2^{1-n}$.
\end{theorem}

### Clinical Significance:
Even if a resident completely misunderstands a critical biomarker ($p_j d_j < 0$), nature naturally produces patient presentations where the sign of that biomarker aligns with the correct diagnosis. The AI can teach the misunderstood biomarker on those patients **with zero risk to patient health**!

---

## 3. The Bellman Stakes Threshold Policy

Let $V(m)$ denote the optimal cost-to-go function from cumulative count vector $m = (m_1, \dots, m_n)$ with remaining training rounds $T - \sum m_j$.
Let $\Delta V_j(m) := \delta V(m + e_j) - \delta V(m)$ be the future curriculum value gained by advancing feature $j$.

egin{theorem}[Optimal Ex-Post Decision Rule]
In round $t$, facing patient with stakes $|D| = |d^\top x|$:
1. **Opportunistic Free Teaching:** If there exists a desired teaching feature $j^* \in Z(x)$, the AI shows feature $j^*$. Immediate regret is zero, and belief error contracts:
   $$j_t = \arg\min_{j \in Z(x)} \delta V(m + e_j)$$
2. **Paid Low-Stakes Teaching:** If all uncalibrated features lie outside $Z(x)$, the AI compares patient stakes $|D|$ against the future curriculum gain:
   - If $|D| < \min_{j \in Z(x)} \delta V(m+e_j) - \min_{k \notin Z(x)} \delta V(m+e_k)$, the AI **pays the low immediate regret $|D|$** to teach target feature $k^* \notin Z(x)$.
   - Otherwise, the AI plays it safe, showing $j \in Z(x)$ to protect the patient.
\end{theorem}

This establishes the formal mathematical threshold for clinical scaffolding: **teach hard lessons when the clinical stakes $|d^\top x|$ are low; shield the patient when stakes are high.**
