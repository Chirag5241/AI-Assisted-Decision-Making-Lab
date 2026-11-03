# <G> - Ex-Post Dynamic Teaching: Zero-Regret Opportunities & Bellman Thresholds Under Independent vs. Correlated Features

This document analyzes the dynamic runtime teaching policy during the training phase ($t \le T$). 

In the ex-post dynamic setting, the AI preceptor observes the arriving patient presentation before deciding which diagnostic test $j_t \in [n]$ to show the clinician ($c = 1$). 

We provide a complete, rigorous separation between:
* **Part I: Independent Biomarkers ($x \sim \mathcal{N}(0, I_n)$)**: Deriving the zero-regret set $Z(x)$, the canonical Sheppard's Arcsine Law, and the Bellman Stakes Threshold rule.
* **Part II: General Correlated Biomarkers ($x \sim \mathcal{N}(0, \Sigma)$)**: Deriving the convolved projection correlation $\rho_j^{\text{corr}}$ and proving that biological collinearity enables safe free teaching of distractors ($\mathbb{P}(k \in Z(x)) > 1/2$).

---

## 1. Dynamic Training Setup & The Zero-Regret Set

At training round $t \le T$:
1. A patient arrives with latent state $x := x^{(t)} \in \mathbb{R}^n$.
2. The ground-truth optimal action is Action 1 if $D := d^\top x > 0$ and Action 2 if $D < 0$, where $d = u_1 - u_2$.
   The clinical stakes of this patient are $|D| = |d^\top x|$.
3. The clinician holds current difference belief vector $p := p^{(t)} = p(m)$, where $m = (m_1, \dots, m_n)$ tracks exposures.
4. The AI preceptor presents a single biomarker $j \in [n]$ to the clinician.
5. The clinician observes $x_j$ and predicts Action 1 if $p_j x_j > 0$ and Action 2 if $p_j x_j < 0$.
6. The immediate clinical regret incurred on patient $x$ is:
   $$\ell_{\text{train}}(x, j) := |d^\top x| \cdot \mathbb{1}\big[\operatorname{sgn}(p_j x_j) \neq \operatorname{sgn}(d^\top x)\big]$$
7. Presenting biomarker $j$ increments its exposure count ($m_j \leftarrow m_j + 1$), contracting belief error:
   $$p_j(m_j + 1) = \big(1 - \psi(m_j + 1)\big) d_j + \psi(m_j + 1) p_j^{(0)}$$

\begin{definition}[The Zero-Regret Set]
For any patient presentation $x$, the **zero-regret teaching set** $Z(x)$ is the set of biomarkers whose observation leads the clinician to the correct clinical diagnosis:
$$Z(x) := \Big\{ j \in [n] : \operatorname{sgn}(p_j x_j) = \operatorname{sgn}(d^\top x) \Big\}$$
\end{definition}

\begin{claim}[Free Teaching Opportunity]
If the AI presents any biomarker $j \in Z(x)$, the clinician's immediate decision regret on patient $x$ is identically zero:
$$\ell_{\text{train}}(x, j) = 0 \quad \forall j \in Z(x)$$
\end{claim}
*Proof:* If $j \in Z(x)$, then $\operatorname{sgn}(p_j x_j) = \operatorname{sgn}(d^\top x)$, so $\mathbb{1}[\operatorname{sgn}(p_j x_j) \neq \operatorname{sgn}(d^\top x)] = 0$. Thus $\ell_{\text{train}}(x, j) = |d^\top x| \cdot 0 = 0$. $\blacksquare$

---

# Part I: Independent Biomarkers ($x \sim \mathcal{N}(0, I_n)$)

In the standardized independent setting, latent vitals follow $x \sim \mathcal{N}(0, I_n)$.

### 1. Probability of Free Teaching (Sheppard's Arcsine Law)

\begin{lemma}[Sheppard's Quadrant Probability, 1899]
For any two zero-mean, jointly Gaussian variables $(U, Y)$ with Pearson correlation $\rho = \frac{\operatorname{Cov}(U, Y)}{\sqrt{\operatorname{Var}[U]\operatorname{Var}[Y]}}$, the probability that they share the same sign is:
$$\mathbb{P}(U \cdot Y > 0) = \frac{1}{2} + \frac{1}{\pi} \arcsin(\rho)$$
\end{lemma}

\begin{proof}
Standardize $U^* = U / \sigma_U$ and $Y^* = Y / \sigma_Y$. The pair $(U^*, Y^*)$ is standard bivariate normal with correlation $\rho$. By the rotational symmetry of standard bivariate Gaussians, the angular opening of the first and third quadrants (where $U^* Y^* > 0$) is $\pi + 2 \arcsin(\rho)$. Dividing by the total $2\pi$ circle yields $\frac{1}{2} + \frac{1}{\pi} \arcsin(\rho)$. $\blacksquare$
\end{proof}

\begin{theorem}[Free Teaching Probability under Independence]
Let $x \sim \mathcal{N}(0, I_n)$. For any biomarker $j \in [n]$ with belief $p_j \neq 0$:
1. The Pearson correlation between the clinician's signal $U_j = p_j x_j$ and the true advantage $Y = d^\top x$ is:
   $$\rho_j = \operatorname{sgn}(p_j) \frac{d_j}{\|d\|_2}$$
2. The probability of biomarker $j$ belonging to the zero-regret set is:
   $$\mathbb{P}\big(j \in Z(x)\big) = \frac{1}{2} + \frac{1}{\pi} \arcsin\left( \operatorname{sgn}(p_j) \frac{d_j}{\|d\|_2} \right)$$
\end{theorem}

\begin{proof}
Since $x \sim \mathcal{N}(0, I_n)$, $U_j = p_j x_j$ and $Y = d^\top x$ are linear combinations of standard normal variables, hence jointly zero-mean Gaussian.
$\operatorname{Var}[U_j] = p_j^2 \operatorname{Var}[x_j] = p_j^2$ and $\operatorname{Var}[Y] = \|d\|_2^2$.
Their covariance is:
$$\operatorname{Cov}(U_j, Y) = \mathbb{E}\left[ (p_j x_j) \left(\sum_{i=1}^n d_i x_i\right) \right] = p_j d_j$$
The correlation is $\rho_j = \frac{p_j d_j}{|p_j| \|d\|_2} = \operatorname{sgn}(p_j) \frac{d_j}{\|d\|_2}$.
Applying Sheppard's Quadrant Probability yields the exact formula. $\blacksquare$
\end{proof}

---

### 2. Clinical Training Regimes Under Independence

1. **Misunderstood Biomarkers ($p_j d_j < 0$):**
   Even when the clinician has an inverted belief on a critical test, the probability of free teaching is **strictly positive**:
   $$\mathbb{P}(j \in Z(x)) = \frac{1}{2} - \frac{1}{\pi} \arcsin\left(\frac{|d_j|}{\|d\|_2}\right) > 0$$
   *Clinical Mechanism:* Natural patient variability occasionally produces presentations where a misunderstood test accidentally points to the right treatment. The preceptor waits for these presentations to debias the misunderstood test at zero risk!
2. **Informative Biomarkers ($p_j d_j > 0$):**
   The probability of free teaching strictly exceeds 50%:
   $$\mathbb{P}(j \in Z(x)) = \frac{1}{2} + \frac{1}{\pi} \arcsin\left(\frac{|d_j|}{\|d\|_2}\right) > \frac{1}{2}$$
3. **Pure Distractor Biomarkers ($d_j = 0$):**
   Since $d_j = 0$, $\rho_j = 0$, so $\arcsin(0) = 0$:
   $$\mathbb{P}(j \in Z(x)) = \frac{1}{2}$$
   An independent distractor is safe to teach exactly 50% of the time.

---

### 3. The Bellman Stakes Threshold Policy (Independent Case)

Let $V_t(m)$ denote the minimal expected total loss from round $t$ to graduation under exposure vector $m$.
Let $\Delta V_j(m) := \mathbb{E}_{x'}[V_{t+1}(m + e_j)] - \mathbb{E}_{x'}[V_{t+1}(m)] \le 0$ denote the curriculum advantage of advancing biomarker $j$.

\begin{theorem}[Optimal Ex-Post Decision Rule under Independence]
For any arriving patient presentation $x$ with stakes $|D| = |d^\top x|$:
1. **Opportunistic Free Teaching**: If there exists an uncalibrated target biomarker $k^* \in Z(x)$, present $k^*$ immediately:
   $$j_t^*(x) = \arg\min_{j \in Z(x)} \Delta V_j(m)$$
   Immediate regret is identically zero ($\ell_{\text{train}} = 0$).
2. **Paid Low-Stakes Teaching**: If the target biomarker $k^* \notin Z(x)$, and $j_{\text{safe}} \in Z(x)$ is the best safe feature:
   Present target biomarker $k^* \notin Z(x)$ if and only if clinical stakes are smaller than future curriculum gains:
   $$|d^\top x| < \Delta V_{j_{\text{safe}}}(m) - \Delta V_{k^*}(m)$$
   Otherwise, present safe feature $j_{\text{safe}} \in Z(x)$ to shield the patient.
\end{theorem}

---

# Part II: General Correlated Biomarkers ($x \sim \mathcal{N}(0, \Sigma), \Sigma \succeq 0$)

In real emergency medicine, patient vitals are correlated via prior covariance $\Sigma \succeq 0$.

### 1. Correlated Free Teaching Probability

\begin{theorem}[Correlated Arcsine Law]
Let latent vitals follow $x \sim \mathcal{N}(0, \Sigma)$. For any biomarker $j \in [n]$ with belief $p_j \neq 0$:
1. The Pearson correlation between the clinician's test observation $U_j = p_j x_j$ and the true advantage $Y = d^\top x$ is:
   $$\rho_j^{\text{corr}} = \operatorname{sgn}(p_j) \frac{(\Sigma d)_j}{\sqrt{\Sigma_{jj}} \sqrt{d^\top \Sigma d}} = \operatorname{sgn}(p_j) \frac{\sum_{i=1}^n \Sigma_{ji} d_i}{\sqrt{\Sigma_{jj}} \sqrt{d^\top \Sigma d}}$$
2. The probability of biomarker $j$ belonging to the zero-regret set is:
   $$\mathbb{P}\big(j \in Z(x)\big) = \frac{1}{2} + \frac{1}{\pi} \arcsin\left( \operatorname{sgn}(p_j) \frac{\sum_{i=1}^n \Sigma_{ji} d_i}{\sqrt{\Sigma_{jj}} \sqrt{d^\top \Sigma d}} \right)$$
\end{theorem}

\begin{proof}
$U_j = p_j x_j$ and $Y = d^\top x$ are zero-mean Gaussians under $x \sim \mathcal{N}(0, \Sigma)$.
$\operatorname{Var}[U_j] = p_j^2 \operatorname{Var}[x_j] = p_j^2 \Sigma_{jj}$, so $\sigma_{U_j} = |p_j| \sqrt{\Sigma_{jj}}$.
$\operatorname{Var}[Y] = d^\top \Sigma d$, so $\sigma_Y = \sqrt{d^\top \Sigma d}$.
Their covariance is:
$$\operatorname{Cov}(U_j, Y) = \mathbb{E}\big[ (p_j x_j)(d^\top x) \big] = p_j \sum_{i=1}^n d_i \, \mathbb{E}[x_j x_i] = p_j \sum_{i=1}^n \Sigma_{ji} d_i = p_j (\Sigma d)_j$$
The correlation is:
$$\rho_j^{\text{corr}} = \frac{p_j (\Sigma d)_j}{|p_j| \sqrt{\Sigma_{jj}} \sqrt{d^\top \Sigma d}} = \operatorname{sgn}(p_j) \frac{(\Sigma d)_j}{\sqrt{\Sigma_{jj}} \sqrt{d^\top \Sigma d}}$$
Applying Sheppard's Quadrant Probability directly yields the result. $\blacksquare$
\end{proof}

---

### 2. The Convolved Free Teaching Regimes

Under correlated vitals, the probability of free teaching depends on the **convolved projection** $(\Sigma d)_j = \sum_{i=1}^n \Sigma_{ji} d_i$. This radically alters the three training regimes:

1. **Correlated Distractors ($\mathbb{P} > 1/2$ Safe Teaching):**
   Suppose biomarker $k$ is a pure distractor ($d_k = 0$).
   * If distractor $k$ is collinear with an informative biomarker $j^*$ ($\Sigma_{k j^*} > 0, d_{j^*} > 0$), then:
     $$(\Sigma d)_k = \Sigma_{k j^*} d_{j^*} > 0$$
   * If the clinician holds a positive belief ($p_k > 0$), then $\rho_k^{\text{corr}} > 0$.
   * Therefore:
     $$\mathbb{P}\big(k \in Z(x)\big) = \frac{1}{2} + \frac{1}{\pi}\arcsin(\rho_k^{\text{corr}}) > \frac{1}{2}$$
   * **Clinical Significance:** Under natural biological correlation, a distractor test **aligns with the correct diagnosis more than 50% of the time**. The preceptor can exploit this proxy alignment to extinguish distractor misconceptions with greater safety!

2. **Adversarial Informative Confounders ($\mathbb{P} < 1/2$):**
   Suppose biomarker $j$ is nominally informative ($d_j > 0$).
   * If biomarker $j$ is negatively correlated with another high-stakes biomarker $\ell$ such that $\Sigma_{j \ell} d_\ell < -\Sigma_{jj} d_j$, then:
     $$(\Sigma d)_j = \Sigma_{jj} d_j + \Sigma_{j \ell} d_\ell < 0$$
   * In this scenario, $\rho_j^{\text{corr}} < 0$, which causes:
     $$\mathbb{P}\big(j \in Z(x)\big) < \frac{1}{2}$$
   * Even though feature $j$ has positive true weight ($d_j > 0$), nature's negative correlation causes it to act as a **clinical confounder**, leading the clinician astray more than half the time. The preceptor must exercise extreme caution when presenting such tests!

---

### 3. The Correlated Bellman Policy

Under general covariance $\Sigma$, the Bellman stakes condition becomes:
$$|d^\top x| < \Delta V_{j_{\text{safe}}}(m) - \Delta V_{k^*}(m)$$
where future value function $V_{t+1}(m)$ incorporates the correlated graduation regret:
$$\mathcal{L}_{\text{grad}}(p(m)) = \frac{\sqrt{d^\top \Sigma d}}{\sqrt{2\pi}} \Big( 1 - \rho_{\text{corr}}\big(\hat{F}(p(m)), p(m)\big) \Big)$$
The preceptor uses $\rho_j^{\text{corr}}$ to evaluate the likelihood of arriving in the zero-regret set at future rounds.

---

# Comparison Matrix: Dynamic Runtime Teaching

| Dimension | Part I: Independent Biomarkers | Part II: General Correlated Biomarkers |
| :--- | :--- | :--- |
| **Biomarker Signal Variance** | $\operatorname{Var}[p_j x_j] = p_j^2$ | $\operatorname{Var}[p_j x_j] = p_j^2 \Sigma_{jj}$ |
| **True Stakes Variance** | $\operatorname{Var}[d^\top x] = \|d\|_2^2$ | $\operatorname{Var}[d^\top x] = d^\top \Sigma d$ |
| **Signal-Outcome Correlation** | $\rho_j = \operatorname{sgn}(p_j) \frac{d_j}{\|d\|_2}$ | $\rho_j^{\text{corr}} = \operatorname{sgn}(p_j) \frac{(\Sigma d)_j}{\sqrt{\Sigma_{jj}} \sqrt{d^\top \Sigma d}}$ |
| **Distractor Safe Teaching Rate** | Exactly $50\%$ ($\mathbb{P} = 1/2$) | Modulated by collinearity: $\mathbb{P} > 1/2$ if $(\Sigma d)_k > 0$ |
| **Informative Biomarker Safety** | Always $> 50\%$ ($\mathbb{P} > 1/2$) | Can drop $< 50\%$ if confounded by negative covariance |
| **Stakes Threshold Metric** | Scaled to standard deviation $\|d\|_2$ | Scaled to general Mahalanobis stakes $\sqrt{d^\top \Sigma d}$ |
