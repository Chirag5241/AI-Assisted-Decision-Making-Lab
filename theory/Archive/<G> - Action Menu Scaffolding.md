# <G> - Action Menu Scaffolding: Targeted Row Debiasing for K >= 3

This document formalizes the theory of **Action Menu Scaffolding** ($|A_t| < K$), resolving the open question in `Problem Description.md`:
> *"What happens when the algorithm is able to reduce the subset of actions shown?"*

We prove that restricting the diagnostic menu prevents **cross-row error cancellation**, allowing the AI to debias specific disease rows strictly faster than full-menu exposure.

---

## 1. Action Menu Scaffolding Formulation

During the pedagogical training phase ($t \le T$), in addition to selecting feature panel $F_t$ ($|F_t| \le c$), the AI can restrict the candidate diagnostic menu to a non-empty subset $A_t \subset [K]$ with $|A_t| \le a_{\max} < K$.

The clinician observes $A_t$ and $F_t$ and chooses among the restricted menu:
$$\hat{y}_t = \arg\max_{k \in A_t} \sum_{j \in F_t} H_{kj}^{(t)} x_j$$

### The Row Exposure Rule
Crucially, **only rows $k \in A_t$ are evaluated and updated**:
$$H_{kj}^{(t+1)} - U_{kj} = \psi\big(m_{kj}(t)\big) \big(H_{kj}^{(0)} - U_{kj}\big) \quad \text{for } (k, j) \in A_t \times F_t$$
Rows $k \notin A_t$ receive no credit or blame, so $m_{kj}$ remains unchanged.

---

## 2. The Row Debiasing Acceleration Theorem

egin{theorem}[Menu Restriction Accelerates Critical Row Calibration]
Let $K \ge 3$. Suppose the clinician overestimates the utility of Action 2 relative to Action 1 due to severe positive bias on row 2 ($e_{2j} := H_{2j}^{(0)} - U_{2j} \gg 0$), while row 1 is moderately biased ($e_{1j} > 0$ with $e_{2j} > e_{1j}$).
1. Under **Full Menu Exposure** ($A_t = [K]$), both rows update simultaneously, requiring exposure count:
   $$\tau^{\text{both}} = \min\left\{ m : \psi(m) < \frac{d_j}{e_{2j} - e_{1j}} \right\}$$
2. Under **Restricted Menu Scaffolding** ($A_t = \{2, 3\}$, isolating row 2), row 2 updates while row 1 remains frozen, flipping the error sign at count:
   $$\tau^{\text{scaffold}} = \min\left\{ m : \psi(m) < \frac{d_j + e_{1j}}{e_{2j}} \right\}$$
3. Because $\frac{d_j + e_{1j}}{e_{2j}} > \frac{d_j}{e_{2j} - e_{1j}}$, the threshold is strictly higher, proving:
   $$\tau^{\text{scaffold}} < \tau^{\text{both}}$$
\end{theorem}

### Clinical Significance:
In a complex emergency triage room with 10 differential diagnoses, a junior resident is easily confused by cross-row interference.
By scaffolding the problem—presenting binary or ternary subsets $A_t = \{1, 2\}$—the AI **isolates and extinguishes dangerous row-level misconceptions in fewer training rounds**, protecting patient safety while accelerating human learning.
