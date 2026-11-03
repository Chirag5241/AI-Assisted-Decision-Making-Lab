# <G> - Graduation Regret & Crowding-Out: Exact Decision Regret Under General Correlated Biomarkers

This document derives the exact closed-form clinical regret incurred by an autonomous clinician after graduation ($t > T$) in the binary decision setting ($K = 2$).

We establish the **Master Theorem** directly for general latent biomarker covariance $\Sigma \succ 0$ and general triage observation matrix $B \in \mathbb{R}^{d \times n}$. We show every intermediate algebraic step without skipping, specialize to the canonical baseline as a corollary, and analyze the clinical mechanisms of the **Crowding-Out Law** for single-test capacity ($c = 1$).

---

### 1. Clinical Formulation: Nature vs. The Clinician

At graduation ($t > T$), a patient arrives at the emergency department with a latent physiological state vector $x \in \mathbb{R}^n$ drawn from a multivariate normal distribution:
$$ x \sim \mathcal{N}(0, \Sigma), \qquad \Sigma \succ 0 $$

#### A. Nature's Ground-Truth Decision
* Nature assigns utility vectors $u_1, u_2 \in \mathbb{R}^n$ to Action 1 and Action 2.
* The true utility difference (the net advantage of Action 1 over Action 2) is:
  $$ D := (u_1 - u_2)^\top x = d^\top x = \sum_{j=1}^n d_j x_j $$
  where $d := u_1 - u_2 \in \mathbb{R}^n$ is the ground-truth advantage vector.
* The truly optimal clinical decision is:
  * Choose **Action 1** if $D > 0$.
  * Choose **Action 2** if $D < 0$.
* The clinical stakes of this patient's presentation are given by the absolute magnitude $|D| = |d^\top x|$.

#### B. Step 1 of Clinical Workflow: Triage Observation ($s = Bx$)
Before any blood tests or imaging are ordered, the clinician observes triage symptoms $s \in \mathbb{R}^d$ ($d \ll n$). Symptoms are a low-dimensional linear projection of the patient's state:
$$ s = Bx, \qquad B \in \mathbb{R}^{d \times n} $$
Because $x$ and $s$ are jointly Gaussian:
$$ 
\begin{pmatrix} x \\ s \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \; \begin{pmatrix} \Sigma & \Sigma B^\top \\ B \Sigma & B \Sigma B^\top \end{pmatrix} \right)
$$
Using the standard Gaussian conditioning theorem, the conditional distribution of latent biomarkers $x$ given presenting symptoms $s$ is:
$$ x \mid s \sim \mathcal{N}\big(\mu(s), \; \Gamma\big) $$
where the conditional mean vector is:
$$ \mu(s) := \mathbb{E}[x \mid s] = \Sigma B^\top (B \Sigma B^\top)^{-1} s $$
Substituting $s = Bx$, we can write $\mu(s)$ directly as a linear transformation of $x$:
$$ \mu(s) = \Sigma B^\top (B \Sigma B^\top)^{-1} B x =: \Pi x $$
Here, we define the **symptom projection matrix** $\Pi \in \mathbb{R}^{n \times n}$:
$$ \Pi := \Sigma B^\top (B \Sigma B^\top)^{-1} B $$
Notice that $\Pi$ is an oblique projection matrix:
$$ \Pi^2 = \Big(\Sigma B^\top (B \Sigma B^\top)^{-1} B\Big) \Big(\Sigma B^\top (B \Sigma B^\top)^{-1} B\Big) = \Sigma B^\top (B \Sigma B^\top)^{-1} (B \Sigma B^\top) (B \Sigma B^\top)^{-1} B = \Pi $$
And the residual conditional covariance matrix is:
$$ \Gamma := \operatorname{Var}[x \mid s] = \Sigma - \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma $$

#### C. Two Critical Algebraic Properties of $\Gamma$ and $\Pi$
Before proceeding, we verify two fundamental orthogonality identities that will simplify all subsequent algebra:

1. **Property 1: $B \Gamma = 0$**  
   Multiplying $\Gamma$ from the left by $B$:
   $$ 
   \begin{aligned}
   B \Gamma &= B \Big( \Sigma - \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma \Big) \\
   &= B \Sigma - (B \Sigma B^\top)(B \Sigma B^\top)^{-1} B \Sigma \\
   &= B \Sigma - I_d (B \Sigma) = 0
   \end{aligned}
   $$
   *Clinical meaning:* The residual variation $(x - \mu(s))$ has zero correlation with symptoms $s$. Symptoms have extracted all linear information they possibly could.

2. **Property 2: $\Pi \Gamma = 0$ and $\Gamma \Pi^\top = 0$**  
   Since $\Pi = \Sigma B^\top (B \Sigma B^\top)^{-1} B$, multiplying by $\Gamma$ gives:
   $$ \Pi \Gamma = \Sigma B^\top (B \Sigma B^\top)^{-1} (B \Gamma) = \Sigma B^\top (B \Sigma B^\top)^{-1} \cdot 0 = 0 $$
   Taking the transpose (and noting that $\Gamma^\top = \Gamma$):
   $$ \Gamma \Pi^\top = (\Pi \Gamma)^\top = 0^\top = 0 $$

3. **Property 3: Factoring $\Gamma$ via $\Pi$**  
   Notice that:
   $$ \Gamma = \Sigma - \Pi \Sigma = (I_n - \Pi) \Sigma $$
   Taking transposes:
   $$ \Gamma = \Sigma - \Sigma \Pi^\top = \Sigma (I_n - \Pi^\top) $$

#### D. Step 2 of Clinical Workflow: Autonomous Test Ordering
From [[2. Test Selection.md]], having observed symptoms $s$, the clinician orders the test panel $F^* \subseteq [n]$ ($|F^*| \le c$) that maximizes perceived Value of Information:
$$ F^*(p) = \arg\max_{|F| \le c} p_F^\top \Gamma_{FF} p_F $$
where $p := p^{(T)} = h_1^{(T)} - h_2^{(T)} \in \mathbb{R}^n$ is the clinician's vector of perceived utility differences at graduation.

#### E. Step 3 of Clinical Workflow: The Post-Test Decision Variable
The lab runs test panel $F^*$ and reveals the true scalar values $x_{F^*}$.
* For measured biomarkers $j \in F^*$, the clinician observes the exact value $x_j$.
* For unmeasured biomarkers $j \notin F^*$, the clinician cannot observe $x_j$, and therefore imputes their conditional expectation $\mu_j(s) = \mathbb{E}[x_j \mid s]$.

The clinician's perceived advantage of Action 1 over Action 2 is:
$$ V(F^*) = \sum_{j \in F^*} p_j x_j + \sum_{j \notin F^*} p_j \mu_j(s) $$
To write this cleanly in vector notation, add and subtract $\sum_{j \in F^*} p_j \mu_j(s)$:
$$ 
\begin{aligned}
V(F^*) &= \sum_{j \in F^*} p_j x_j + \sum_{j \notin F^*} p_j \mu_j(s) + \left( \sum_{j \in F^*} p_j \mu_j(s) - \sum_{j \in F^*} p_j \mu_j(s) \right) \\
&= \left( \sum_{j \in F^*} p_j \mu_j(s) + \sum_{j \notin F^*} p_j \mu_j(s) \right) + \sum_{j \in F^*} p_j \big( x_j - \mu_j(s) \big) \\
&= \sum_{j=1}^n p_j \mu_j(s) + \sum_{j \in F^*} p_j \big( x_j - \mu_j(s) \big) \\
&= p^\top \mu(s) + p_{F^*}^\top \big( x_{F^*} - \mu_{F^*}(s) \big)
\end{aligned}
$$
* The first term $p^\top \mu(s)$ is the **triage baseline advantage** (what the doctor thought after triage).
* The second term $p_{F^*}^\top (x_{F^*} - \mu_{F^*}(s))$ is the **test innovation** (how the lab results updated the doctor's mind).

The clinician chooses **Action 1** if $V(F^*) > 0$ and **Action 2** if $V(F^*) < 0$.

#### F. Clinical Regret Definition
When the clinician's decision $\operatorname{sgn}(V(F^*))$ disagrees with nature's optimal choice $\operatorname{sgn}(D)$, the patient receives the suboptimal treatment and suffers a utility loss equal to $|D| = |d^\top x|$. 
We define the graduation regret on patient $x$ as:
$$ \ell_{\text{grad}}(x; p) := |d^\top x| \cdot \mathbb{1}\big[ (d^\top x) \cdot V(F^*) < 0 \big] $$

---

### 2. Expressing $V(F^*)$ as a Linear Transformation of $x$

To evaluate the expectation of $\ell_{\text{grad}}(x; p)$ over all arriving patients $x \sim \mathcal{N}(0, \Sigma)$, we must express $V(F^*)$ as an explicit linear function of $x$.

Let $E_{F^*} \in \mathbb{R}^{c \times n}$ be the coordinate selection matrix that extracts the subvector of tested features:
$$ x_{F^*} = E_{F^*} x, \qquad \mu_{F^*}(s) = E_{F^*} \mu(s) $$
Then the test innovation vector is:
$$ x_{F^*} - \mu_{F^*}(s) = E_{F^*} \big( x - \mu(s) \big) $$
Recall from Section 1B that $\mu(s) = \Pi x$. Therefore:
$$ x - \mu(s) = x - \Pi x = (I_n - \Pi) x $$
Substituting this into the test innovation term:
$$ x_{F^*} - \mu_{F^*}(s) = E_{F^*} (I_n - \Pi) x $$
Now substitute this back into the full formula for $V(F^*)$:
$$ 
\begin{aligned}
V(F^*) &= p^\top \mu(s) + p_{F^*}^\top \big( x_{F^*} - \mu_{F^*}(s) \big) \\
&= p^\top (\Pi x) + p_{F^*}^\top \Big( E_{F^*} (I_n - \Pi) x \Big) \\
&= \Big( p^\top \Pi \Big) x + \Big( p_{F^*}^\top E_{F^*} (I_n - \Pi) \Big) x \\
&= \Big( \Pi^\top p \Big)^\top x + \Big( (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} \Big)^\top x \\
&= \Big( \Pi^\top p + (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} \Big)^\top x
\end{aligned}
$$
We have established that $V(F^*)$ is a pure linear transformation of the patient state $x$:
$$ V(F^*) = w_V^\top x $$
where the clinician's effective decision weight vector $w_V \in \mathbb{R}^n$ is:
$$ w_V := \underbrace{\Pi^\top p}_{\text{symptom weights}} \;+\; \underbrace{(I_n - \Pi^\top) E_{F^*}^\top p_{F^*}}_{\text{lab test weights}} $$

---

### 3. Joint Distribution of Nature ($D$) and Clinician ($V(F^*)$)

Because $x \sim \mathcal{N}(0, \Sigma)$, any two linear transformations of $x$ are jointly Gaussian:
$$ D = d^\top x, \qquad V(F^*) = w_V^\top x $$
Both variables have mean zero across arriving patients:
$$ \mathbb{E}[D] = d^\top \mathbb{E}[x] = d^\top \cdot 0 = 0 $$
$$ \mathbb{E}[V(F^*)] = w_V^\top \mathbb{E}[x] = w_V^\top \cdot 0 = 0 $$

Therefore, $(D, V(F^*))$ is a jointly zero-mean bivariate normal vector:
$$ 
\begin{pmatrix} D \\ V(F^*) \end{pmatrix} \sim \mathcal{N}\left( \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \; \begin{pmatrix} \sigma_D^2 & \operatorname{Cov}(D, V) \\ \operatorname{Cov}(D, V) & \sigma_V^2 \end{pmatrix} \right)
$$

We now calculate each variance and the covariance with complete algebraic detail.

#### A. Variance of Nature's True Advantage ($\sigma_D^2$)
$$ \sigma_D^2 := \operatorname{Var}[D] = \mathbb{E}[(d^\top x)^2] = d^\top \mathbb{E}[x x^\top] d = d^\top \Sigma d $$

#### B. Variance of Clinician's Decision Variable ($\sigma_V^2$)
By definition:
$$ \sigma_V^2 := \operatorname{Var}[V(F^*)] = w_V^\top \operatorname{Var}[x] w_V = w_V^\top \Sigma w_V $$
Substitute $w_V = \Pi^\top p + (I_n - \Pi^\top) E_{F^*}^\top p_{F^*}$:
$$ 
\begin{aligned}
w_V^\top \Sigma w_V &= \Big( \Pi^\top p + (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} \Big)^\top \Sigma \Big( \Pi^\top p + (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} \Big) \\
&= \underbrace{(\Pi^\top p)^\top \Sigma (\Pi^\top p)}_{\text{Term 1: Symptom Variance}} \;+\; \underbrace{2 (\Pi^\top p)^\top \Sigma (I_n - \Pi^\top) E_{F^*}^\top p_{F^*}}_{\text{Term 2: Cross Term}} \;+\; \underbrace{\Big((I_n - \Pi^\top) E_{F^*}^\top p_{F^*}\Big)^\top \Sigma \Big((I_n - \Pi^\top) E_{F^*}^\top p_{F^*}\Big)}_{\text{Term 3: Test Variance}}
\end{aligned}
$$

Let us simplify each of these three terms individually:

* **Simplifying Term 2 (The Cross Term):**
  Recall from Property 3 that $\Sigma (I_n - \Pi^\top) = \Gamma$. Thus:
  $$ (\Pi^\top p)^\top \Sigma (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} = p^\top \Pi \Big( \Sigma (I_n - \Pi^\top) \Big) E_{F^*}^\top p_{F^*} = p^\top \Big( \Pi \Gamma \Big) E_{F^*}^\top p_{F^*} $$
  From Property 2, $\Pi \Gamma = 0$! Therefore:
  $$ \text{Term 2} = 2 p^\top (0) E_{F^*}^\top p_{F^*} = 0 $$
  The cross-term vanishes completely! Symptoms and test innovations are **statistically orthogonal**.

* **Simplifying Term 1 (Symptom Variance):**
  $$ (\Pi^\top p)^\top \Sigma (\Pi^\top p) = p^\top \Pi \Sigma \Pi^\top p $$
  Recall that $\Pi = \Sigma B^\top (B \Sigma B^\top)^{-1} B$. Substituting $\Pi$:
  $$ 
  \begin{aligned}
  \Pi \Sigma \Pi^\top &= \Big(\Sigma B^\top (B \Sigma B^\top)^{-1} B\Big) \Sigma \Big(B^\top (B \Sigma B^\top)^{-1} B \Sigma\Big) \\
  &= \Sigma B^\top (B \Sigma B^\top)^{-1} \Big(B \Sigma B^\top\Big) (B \Sigma B^\top)^{-1} B \Sigma \\
  &= \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma
  \end{aligned}
  $$
  Notice that $\Sigma - \Gamma = \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma$. Therefore:
  $$ \Pi \Sigma \Pi^\top = \Sigma - \Gamma $$
  Thus:
  $$ \text{Term 1} = p^\top (\Sigma - \Gamma) p $$

* **Simplifying Term 3 (Test Variance):**
  $$ 
  \begin{aligned}
  \Big((I_n - \Pi^\top) E_{F^*}^\top p_{F^*}\Big)^\top \Sigma \Big((I_n - \Pi^\top) E_{F^*}^\top p_{F^*}\Big) &= p_{F^*}^\top E_{F^*} (I_n - \Pi) \Sigma (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} \\
  &= p_{F^*}^\top E_{F^*} (I_n - \Pi) \Gamma E_{F^*}^\top p_{F^*} \qquad \text{(since } \Sigma(I_n - \Pi^\top) = \Gamma) \\
  &= p_{F^*}^\top E_{F^*} (\Gamma - \Pi \Gamma) E_{F^*}^\top p_{F^*} \\
  &= p_{F^*}^\top E_{F^*} (\Gamma - 0) E_{F^*}^\top p_{F^*} \qquad \text{(since } \Pi \Gamma = 0) \\
  &= p_{F^*}^\top \big( E_{F^*} \Gamma E_{F^*}^\top \big) p_{F^*}
  \end{aligned}
  $$
  Notice that $E_{F^*} \Gamma E_{F^*}^\top$ is simply the submatrix of $\Gamma$ restricted to rows and columns in $F^*$, which is $\Gamma_{F^* F^*}$! Therefore:
  $$ \text{Term 3} = p_{F^*}^\top \Gamma_{F^* F^*} p_{F^*} $$

Combining Terms 1, 2, and 3:
$$ \sigma_V^2(F^*) = \underbrace{p^\top (\Sigma - \Gamma) p}_{\text{symptom variance } \sigma_{\text{sym}}^2} \;+\; \underbrace{p_{F^*}^\top \Gamma_{F^* F^*} p_{F^*}}_{\text{test innovation variance } \sigma_{\text{test}}^2(F^*)} $$

#### C. Covariance Between Nature and Clinician ($\operatorname{Cov}(D, V)$)
Now we compute the covariance:
$$ \operatorname{Cov}(D, V(F^*)) = \mathbb{E}[(d^\top x)(w_V^\top x)] = d^\top \mathbb{E}[x x^\top] w_V = d^\top \Sigma w_V $$
Substitute $w_V = \Pi^\top p + (I_n - \Pi^\top) E_{F^*}^\top p_{F^*}$:
$$ 
\begin{aligned}
d^\top \Sigma w_V &= d^\top \Sigma \Big( \Pi^\top p + (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} \Big) \\
&= \underbrace{d^\top \Sigma \Pi^\top p}_{\text{Symptom Covariance}} \;+\; \underbrace{d^\top \Sigma (I_n - \Pi^\top) E_{F^*}^\top p_{F^*}}_{\text{Test Innovation Covariance}}
\end{aligned}
$$

Let us simplify both terms:
* For the first term, we know from earlier that $\Sigma \Pi^\top = \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma = \Sigma - \Gamma$. Therefore:
  $$ d^\top \Sigma \Pi^\top p = d^\top (\Sigma - \Gamma) p $$
  *Clinical meaning:* This measures how well the clinician's beliefs $p$ align with the true disease weights $d$ on the subspace of symptoms that were observed at triage.

* For the second term, we know that $\Sigma (I_n - \Pi^\top) = \Gamma$. Therefore:
  $$ d^\top \Sigma (I_n - \Pi^\top) E_{F^*}^\top p_{F^*} = d^\top \Gamma E_{F^*}^\top p_{F^*} = p_{F^*}^\top E_{F^*} \Gamma d $$
  Notice that $E_{F^*} \Gamma$ is the submatrix containing only the rows of $\Gamma$ indexed by $F^*$, denoted $\Gamma_{F^*, :}$. Therefore:
  $$ p_{F^*}^\top E_{F^*} \Gamma d = p_{F^*}^\top \Gamma_{F^*, :} \, d $$
  *Clinical meaning:* This measures the diagnostic alignment provided by the ordered test panel $F^*$. Crucially, because $\Gamma$ has off-diagonal entries, ordered tests $F^*$ capture signal not just from themselves, but from all unmeasured biomarkers correlated with them through $\Gamma$!

Combining both terms gives the exact covariance:
$$ \operatorname{Cov}(D, V(F^*)) = \underbrace{d^\top (\Sigma - \Gamma) p}_{\text{symptom alignment}} \;+\; \underbrace{p_{F^*}^\top \Gamma_{F^*, :} \, d}_{\text{test alignment}} $$

#### D. The Pearson Clinical Correlation ($\rho$)
The Pearson correlation coefficient between nature's advantage $D$ and clinician's prediction $V(F^*)$ is:
$$ \rho(F^*(p), p) := \frac{\operatorname{Cov}(D, V(F^*))}{\sigma_D \sigma_V(F^*)} = \frac{d^\top (\Sigma - \Gamma) p + p_{F^*}^\top \Gamma_{F^*, :} \, d}{\sqrt{d^\top \Sigma d} \;\; \sqrt{p^\top (\Sigma - \Gamma) p + p_{F^*}^\top \Gamma_{F^* F^*} p_{F^*}}} $$

---

### 4. Derivation of the Master Graduation Regret Theorem

We now evaluate the expected clinical regret:
$$ \mathcal{L}_{\text{grad}}(p) := \mathbb{E}_{x \sim \mathcal{N}(0, \Sigma)}\big[\ell_{\text{grad}}(x; p)\big] = \mathbb{E}\Big[ |D| \cdot \mathbb{1}[D \cdot V(F^*) < 0] \Big] $$

#### Step 1: Standardize to Bivariate Standard Normal
Define standardized variables:
$$ U := \frac{D}{\sigma_D} = \frac{D}{\sqrt{d^\top \Sigma d}} \sim \mathcal{N}(0, 1) $$
$$ Y := \frac{V(F^*)}{\sigma_V(F^*)} = \frac{V(F^*)}{\sqrt{w_V^\top \Sigma w_V}} \sim \mathcal{N}(0, 1) $$
The pair $(U, Y)$ is standard bivariate normal with correlation:
$$ \operatorname{Cov}(U, Y) = \frac{\operatorname{Cov}(D, V(F^*))}{\sigma_D \sigma_V(F^*)} = \rho(F^*(p), p) $$
Since $\sigma_D > 0$ and $\sigma_V > 0$, the signs satisfy:
$$ \operatorname{sgn}(D) = \operatorname{sgn}(U), \qquad \operatorname{sgn}(V(F^*)) = \operatorname{sgn}(Y) \implies \mathbb{1}[D \cdot V(F^*) < 0] = \mathbb{1}[U \cdot Y < 0] $$
And the absolute magnitude is $|D| = \sigma_D |U| = \sqrt{d^\top \Sigma d} \, |U|$.
Substituting these into the regret expectation:
$$ \mathcal{L}_{\text{grad}}(p) = \sqrt{d^\top \Sigma d} \cdot \mathbb{E}\big[ |U| \cdot \mathbb{1}[U \cdot Y < 0] \big] $$

#### Step 2: Evaluating Sheppard's Quadrant Integral
We compute $I := \mathbb{E}[|U| \cdot \mathbb{1}[UY < 0]]$.
By conditional Gaussian properties, we can decompose $Y$ as:
$$ Y = \rho U + \sqrt{1 - \rho^2} Z $$
where $Z \sim \mathcal{N}(0, 1)$ is independent of $U \sim \mathcal{N}(0, 1)$.
Conditioning on $U = u$:
$$ 
\begin{aligned}
\mathbb{P}(U Y < 0 \mid U = u) &= \mathbb{P}(u Y < 0) \\
&= \begin{cases} \mathbb{P}(Y < 0 \mid U = u) & \text{if } u > 0 \\ \mathbb{P}(Y > 0 \mid U = u) & \text{if } u < 0 \end{cases}
\end{aligned}
$$
Since $Y \mid (U = u) \sim \mathcal{N}\big(\rho u, 1 - \rho^2\big)$, for $u > 0$:
$$ \mathbb{P}(Y < 0 \mid U = u) = \Phi\left( \frac{0 - \rho u}{\sqrt{1 - \rho^2}} \right) = \Phi\left( -\frac{\rho u}{\sqrt{1 - \rho^2}} \right) $$
By symmetry across $u > 0$ and $u < 0$:
$$ I = \int_{-\infty}^\infty |u| \cdot \mathbb{P}(u Y < 0 \mid U = u) \phi(u) \, du = 2 \int_0^\infty u \, \Phi\left( -\frac{\rho u}{\sqrt{1 - \rho^2}} \right) \phi(u) \, du $$
where $\phi(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$ is the standard normal PDF.

We evaluate this integral using **integration by parts**:
$$ \int_0^\infty f(u) g'(u) \, du = \Big[ f(u) g(u) \Big]_0^\infty - \int_0^\infty f'(u) g(u) \, du $$
Choose:
* $f(u) = \Phi\left( -\frac{\rho u}{\sqrt{1 - \rho^2}} \right)$
* $g'(u) = u \phi(u) = -\phi'(u) \implies g(u) = -\phi(u)$

1. **The Boundary Term:**
   $$ \Big[ -\phi(u) f(u) \Big]_0^\infty = \lim_{u \to \infty} \big(-\phi(u) f(u)\big) - \Big( -\phi(0) f(0) \Big) $$
   As $u \to \infty$, $\phi(u) \to 0$ exponentially, so the upper limit is 0.
   At $u = 0$, $\phi(0) = \frac{1}{\sqrt{2\pi}}$ and $f(0) = \Phi(0) = \frac{1}{2}$.
   Therefore:
   $$ \text{Boundary Term} = 0 - \left( -\frac{1}{\sqrt{2\pi}} \cdot \frac{1}{2} \right) = \frac{1}{2\sqrt{2\pi}} $$

2. **The Derivative $f'(u)$:**
   Using the chain rule:
   $$ f'(u) = \frac{d}{du} \Phi\left( -\frac{\rho u}{\sqrt{1 - \rho^2}} \right) = \phi\left( -\frac{\rho u}{\sqrt{1 - \rho^2}} \right) \cdot \left( -\frac{\rho}{\sqrt{1 - \rho^2}} \right) $$

3. **The Remaining Integral:**
   $$ 
   \begin{aligned}
   -\int_0^\infty f'(u) g(u) \, du &= -\int_0^\infty \left[ -\frac{\rho}{\sqrt{1 - \rho^2}} \phi\left(-\frac{\rho u}{\sqrt{1 - \rho^2}}\right) \right] \big[ -\phi(u) \big] \, du \\
   &= -\frac{\rho}{\sqrt{1 - \rho^2}} \int_0^\infty \phi\left(\frac{\rho u}{\sqrt{1 - \rho^2}}\right) \phi(u) \, du
   \end{aligned}
   $$
   Expand the product of normal densities $\phi(a) \phi(b) = \frac{1}{2\pi} e^{-\frac{1}{2}(a^2 + b^2)}$:
   $$ \left(\frac{\rho u}{\sqrt{1 - \rho^2}}\right)^2 + u^2 = u^2 \left( \frac{\rho^2}{1 - \rho^2} + 1 \right) = u^2 \left( \frac{\rho^2 + 1 - \rho^2}{1 - \rho^2} \right) = \frac{u^2}{1 - \rho^2} $$
   Therefore:
   $$ \phi\left(\frac{\rho u}{\sqrt{1 - \rho^2}}\right) \phi(u) = \frac{1}{2\pi} \exp\left( -\frac{u^2}{2(1 - \rho^2)} \right) = \frac{\sqrt{1 - \rho^2}}{\sqrt{2\pi}} \cdot \frac{1}{\sqrt{2\pi(1 - \rho^2)}} \exp\left( -\frac{u^2}{2(1 - \rho^2)} \right) $$
   Integrating over $u \in [0, \infty)$ integrates the half-normal distribution with variance $1 - \rho^2$, which evaluates to $\frac{1}{2}$:
   $$ \int_0^\infty \phi\left(\frac{\rho u}{\sqrt{1 - \rho^2}}\right) \phi(u) \, du = \frac{\sqrt{1 - \rho^2}}{\sqrt{2\pi}} \cdot \frac{1}{2} = \frac{\sqrt{1 - \rho^2}}{2\sqrt{2\pi}} $$
   Multiplying by $-\frac{\rho}{\sqrt{1 - \rho^2}}$:
   $$ -\frac{\rho}{\sqrt{1 - \rho^2}} \left( \frac{\sqrt{1 - \rho^2}}{2\sqrt{2\pi}} \right) = -\frac{\rho}{2\sqrt{2\pi}} $$

4. **Summing Boundary Term and Integral:**
   $$ \int_0^\infty u \, \Phi\left( -\frac{\rho u}{\sqrt{1 - \rho^2}} \right) \phi(u) \, du = \frac{1}{2\sqrt{2\pi}} - \frac{\rho}{2\sqrt{2\pi}} = \frac{1 - \rho}{2\sqrt{2\pi}} $$
   Multiplying by 2:
   $$ I = 2 \left( \frac{1 - \rho}{2\sqrt{2\pi}} \right) = \frac{1 - \rho}{\sqrt{2\pi}} $$

#### Step 3: Concluding the Master Regret Formula
Multiplying $I$ by $\sigma_D = \sqrt{d^\top \Sigma d}$ establishes the Master Theorem:

> [!NOTE] Master Theorem: Exact Clinical Graduation Regret
> For any patient population covariance $\Sigma \succ 0$, triage observation matrix $B \in \mathbb{R}^{d \times n}$, and clinician belief vector $p$, the expected graduation regret incurred under autonomous test selection $F^*(p)$ is:
> $$ \mathcal{L}_{\text{grad}}(p) = \frac{\sqrt{d^\top \Sigma d}}{\sqrt{2\pi}} \Big( 1 - \rho(F^*(p), p) \Big) $$
> where the clinical correlation is given exactly by:
> $$ \rho(F^*(p), p) = \frac{d^\top (\Sigma - \Gamma) p + p_{F^*}^\top \Gamma_{F^*, :} \, d}{\sqrt{d^\top \Sigma d} \;\; \sqrt{p^\top (\Sigma - \Gamma) p + p_{F^*}^\top \Gamma_{F^* F^*} p_{F^*}}} $$

---

### 5. The Canonical Baseline Corollary

To see how the Master Theorem specializes to classical uninformative baselines, consider the setting where no triage symptoms are observed ($B = 0$) and biomarkers are independent standard Gaussians ($x \sim \mathcal{N}(0, I_n)$):

> [!TIP] Corollary: Canonical Baseline with Independent Biomarkers
> When $B = 0$ and $\Sigma = I_n$:
> 1. $\Pi = 0$, $\mu(s) = 0$, and residual covariance $\Gamma = I_n$.
> 2. The symptom variance and covariance vanish: $\Sigma - \Gamma = I_n - I_n = 0$.
> 3. The test submatrix is $\Gamma_{F^* F^*} = I_c$, and $\Gamma_{F^*, :} d = d_{F^*}$.
> 4. The correlation collapses directly to:
>    $$ \rho = \frac{\sum_{j \in F^*} d_j p_j}{\|d\|_2 \sqrt{\sum_{j \in F^*} p_j^2}} $$
> 5. Expected graduation regret collapses directly to:
>    $$ \mathcal{L}_{\text{grad}}(p) = \frac{\|d\|_2}{\sqrt{2\pi}} \left( 1 - \frac{\sum_{j \in F^*} d_j p_j}{\|d\|_2 \sqrt{\sum_{j \in F^*} p_j^2}} \right) $$

---

### 6. Single-Test Capacity ($c = 1$) & The Crowding-Out Law

We now focus specifically on the clinical emergency setting where the doctor is constrained to ordering at most a single confirmatory test ($c = 1$).

#### A. Autonomous Test Choice for $c = 1$
When $|F| = 1$, candidate panels are singletons $F = \{j\}$ for $j \in [n]$.
From `Test Selection.md`, the quadratic salience objective for a single feature simplifies to:
$$ p_{\{j\}}^\top \Gamma_{\{j\}\{j\}} p_{\{j\}} = p_j^2 \Gamma_{jj} $$
Thus, the clinician greedily selects the single biomarker:
$$ j^* = \arg\max_{j \in [n]} p_j^2 \Gamma_{jj} = \arg\max_{j \in [n]} |p_j| \sqrt{\Gamma_{jj}} $$

#### B. Exact Correlation for $c = 1$
Define the **residual ground-truth advantage vector** orthogonal to symptoms:
$$ \tilde{d} := \Gamma d \in \mathbb{R}^n $$
Define the scalar symptom baseline quantities:
* Baseline symptom covariance: $C_{\text{sym}} := d^\top (\Sigma - \Gamma) p$
* Baseline symptom variance: $\sigma_{\text{sym}}^2 := p^\top (\Sigma - \Gamma) p$

When single test $j$ is ordered, the clinical correlation is:
$$ \rho(\{j\}, p) = \frac{C_{\text{sym}} + p_j \tilde{d}_j}{\sqrt{d^\top \Sigma d} \;\; \sqrt{\sigma_{\text{sym}}^2 + p_j^2 \Gamma_{jj}}} $$

#### C. The Anatomy of Crowding-Out
Suppose:
* Test $j^*$ is an **informative biomarker**: it carries strong residual diagnostic value orthogonal to symptoms ($\tilde{d}_{j^*} > 0$), and the clinician is calibrated on it ($p_{j^*} \approx d_{j^*}$).
* Test $k$ is a **pure distractor**: it carries zero residual diagnostic value orthogonal to symptoms ($\tilde{d}_k = 0$), but the clinician holds a massive uncalibrated misconception ($|p_k| = M \gg 1$).

We observe two distinct clinical catastrophes:

#### Catastrophe 1: Slot Theft
The clinician autonomously orders distractor $k$ instead of informative test $j^*$ whenever:
$$ p_k^2 \Gamma_{kk} > p_{j^*}^2 \Gamma_{j^* j^*} \iff M > |p_{j^*}| \sqrt{\frac{\Gamma_{j^* j^*}}{\Gamma_{kk}}} $$
Because capacity is limited to $c = 1$, distractor $k$ completely ejects the true life-saving test $j^*$ from the clinical workup.

#### Catastrophe 2: Triage Dilution
When distractor $k$ is ordered, what happens to the clinical correlation?
1. Because $\tilde{d}_k = 0$, the test innovation contributes **zero** to the numerator:
   $$ p_k \tilde{d}_k = M \cdot 0 = 0 $$
2. But in the denominator, the clinician's confidence $M^2$ adds a massive noise variance $M^2 \Gamma_{kk}$:
   $$ \rho(\{k\}, p) = \frac{C_{\text{sym}} + 0}{\sqrt{d^\top \Sigma d} \;\; \sqrt{\sigma_{\text{sym}}^2 + M^2 \Gamma_{kk}}} = \frac{C_{\text{sym}}}{\sqrt{d^\top \Sigma d} \;\; \sqrt{\sigma_{\text{sym}}^2 + M^2 \Gamma_{kk}}} $$
As $M \to \infty$, the denominator explodes, driving clinical correlation to zero:
$$ \lim_{M \to \infty} \rho(\{k\}, p) = 0 $$

> [!WARNING] The Clinical Danger of Triage Dilution
> Notice the crucial insight revealed by this equation:  
> When the clinician orders distractor $k$, the test doesn't just fail to help. The loud noise variance $M^2 \Gamma_{kk}$ actively **washes out and destroys the good diagnostic correlation $C_{\text{sym}}$ already achieved at triage!**  
> The clinician acts on the random noise of test $k$, overruling their own triage assessment and driving graduation regret to the catastrophic uncalibrated ceiling:
> $$ \mathcal{L}_{\text{grad}} \to \frac{\sqrt{d^\top \Sigma d}}{\sqrt{2\pi}} \approx 0.3989 \sqrt{d^\top \Sigma d} $$

This proves mathematically why an AI preceptor cannot simply train clinicians on good biomarkers. The AI must dedicate explicit training exposures to **extinguish loud distractors ($M \to 0$)**, protecting the clinician from self-inflicted crowding-out at graduation.
