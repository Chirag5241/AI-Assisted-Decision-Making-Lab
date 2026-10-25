# <G> Ex-Post Opportunistic Teaching (c = 1)

This note details the theory of **Ex-Post Opportunistic Teaching**, where the AI assistant observes the realization of the environment state $x^{(t)} \sim \mathcal{N}(0, I_n)$ *before* deciding which single feature $j \in [n]$ to reveal to the human.

---

## 1. Mathematical Formulation

Let $K = 2$, and suppose the AI displays $A_t = \{1, 2\}$ and a single feature $F_t = \{j\}$.
Let $\sigma_j := \operatorname{sgn}(g_j(m_j)) \in \{-1, +1\}$ denote the current sign of the human's belief difference.
Define:
$$z_j := \sigma_j x_j \sim \mathcal{N}(0, 1) \quad (\text{i.i.d.}), \qquad w_j := \sigma_j d_j$$
Then the ground truth decision margin is:
$$D := d^\top x = \sum_{j=1}^n w_j z_j$$
* A feature $j$ is **good** (correct sign) iff $w_j > 0$.
* A feature $j$ is **bad** (inverted sign) iff $w_j < 0$.

When feature $j$ is shown, the human observes $x_j$ and predicts action 1 iff $g_j x_j > 0 \iff z_j > 0$. 
The human's prediction is correct ($y^* = \hat{y}$) iff $\operatorname{sgn}(z_j) = \operatorname{sgn}(D)$.
Thus, showing feature $j$ yields **zero immediate regret** on state $x$ iff $j \in Z(x)$, where:
$$Z(x) := \big\{ j \in [n] : \operatorname{sgn}(z_j) = \operatorname{sgn}(D) \big\}$$

---

## 2. Zero-Regret Availability & Sheppard's Arcsine Law

### Theorem: Availability of Zero-Regret Teaching
Fix any count vector $m$ with non-zero beliefs $\sigma_j = \operatorname{sgn}(g_j(m_j))$:

1. **Empty Set Condition**: $Z(x) = \emptyset$ if and only if all $z_1, \dots, z_n$ share a common sign $s \in \{-1, +1\}$ and $\operatorname{sgn}(D) = -s$.
2. **All Good Features**: If all features are good ($w_j > 0$ for all $j$), then:
   $$\mathbb{P}(Z(x) \neq \emptyset) = 1$$
   *Consequence*: Once all features have the correct sign, an ex-post policy achieves **zero regret on every round**.
3. **All Bad Features**: If all features are bad ($w_j < 0$ for all $j$), then:
   $$\mathbb{P}(Z(x) = \emptyset) = 2^{1-n}$$
   *Consequence*: Even in the worst-case epistemic state where the human misunderstands *every* feature, a zero-regret teaching opportunity exists with probability $1 - 2^{1-n}$. For $n=3$, this is $75\%$; for $n=5$, it is $93.75\%$.
4. **Sheppard's Arcsine Opportunity Law**:
   * Any individual **bad feature** $j$ ($w_j < 0$) belongs to $Z(x)$ (providing a **free teaching opportunity**) with probability:
     $$\mathbb{P}(j \in Z(x)) = \frac{1}{2} - \frac{1}{\pi} \arcsin\left(\frac{|d_j|}{\|d\|}\right)$$
   * Any individual **good feature** $j$ ($w_j > 0$) belongs to $Z(x)$ with probability:
     $$\mathbb{P}(j \in Z(x)) = \frac{1}{2} + \frac{1}{\pi} \arcsin\left(\frac{|d_j|}{\|d\|}\right)$$

#### Proof Sketch of Arcsine Law:
For a bad feature $j$, $\operatorname{sgn}(z_j) = -\operatorname{sgn}(d_j x_j)$, so $j \in Z(x) \iff \operatorname{sgn}(d_j x_j) \neq \operatorname{sgn}(D)$.
The pair $(d_j x_j, D)$ is bivariate normal with covariance $\operatorname{Cov}(d_j x_j, D) = d_j^2$ and correlation $\rho = \frac{|d_j|}{\|d\|}$.
By **Sheppard's (1899) Median Dichotomy Theorem**:
$$\mathbb{P}(\text{opposite signs}) = \frac{\arccos(\rho)}{\pi} = \frac{1}{2} - \frac{1}{\pi} \arcsin(\rho)$$

---

## 3. Bellman Threshold Policy

Let $V(m)$ denote the optimal discounted regret from count vector $m$, and let:
$$\nu_j(m) := \delta V(m + e_j)$$
Define the best continuation values in and outside the zero-regret set:
$$\nu_Z(x, m) := \min_{j \in Z(x)} \nu_j(m), \qquad \nu_{Z^c}(x, m) := \min_{j \notin Z(x)} \nu_j(m)$$

### Theorem: The Optimal Ex-Post Policy
In realized state $x$:
1. **If $Z(x) \neq \emptyset$**:
   * The AI shows $\arg\min_{j \in Z(x)} \nu_j(m)$ at **zero immediate regret** (free teaching at *any* stakes $|D|$),
   * **Unless** the stakes are small:
     $$|D| < \nu_Z(x, m) - \nu_{Z^c}(x, m)$$
     in which case it pays regret $|D|$ to teach $\arg\min_{j \notin Z(x)} \nu_j(m)$ (**paid low-stakes teaching**).
2. **If $Z(x) = \emptyset$**:
   * The AI pays regret $|D|$ and shows $\arg\min_{j \in [n]} \nu_j(m)$.

> [!NOTE] Strategic Stakes Insight
> Intuition might suggest showing "safe" features in high-stakes states and "teaching" in low-stakes states. 
> The Bellman policy reveals a sharper truth: **free teaching happens at any stakes** whenever an unlearned feature falls in $Z(x)$. Stakes only matter when deciding whether to *pay* regret for a feature outside $Z(x)$.

---

## 4. Value Monotonicity under Learning

### Theorem: Monotonicity
For every count vector $m \in \mathbb{N}^n$ and every feature coordinate $i \in [n]$:
$$V(m + e_i) \le V(m)$$

#### Proof Mechanism (Conditional Gaussian Coupling):
* Increments only change immediate loss when crossing the flip threshold $m_i + 1 = \tau_i$, changing $\sigma_i$ from bad to good.
* Decomposing $D = d_i x_i + S_{-i}$, conditioning on $(|D|, Z_{-i}(x))$ shows that the Gaussian density of $S_{-i}$ is strictly larger when $\operatorname{sgn}(d_i x_i) = \operatorname{sgn}(D)$ than when $\operatorname{sgn}(d_i x_i) = -\operatorname{sgn}(D)$.
* Integrating over $|d_i x_i| > 0$ yields a stochastic dominance coupling:
  $$\mathbb{P}(\operatorname{sgn}(d_i x_i) = \operatorname{sgn}(D) \mid D, Z_{-i}(x)) \ge \mathbb{P}(\operatorname{sgn}(-d_i x_i) = \operatorname{sgn}(D) \mid D, Z_{-i}(x))$$
* Thus, flipping feature $i$ strictly enlarges the distribution of $Z(x)$, guaranteeing $V(m + e_i) \le V(m)$ by backwards induction.
