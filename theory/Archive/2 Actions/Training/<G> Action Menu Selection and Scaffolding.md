# <G> Action Menu Selection and Scaffolding

This note analyzes the strategic selection of the **Action Menu** $A_t \subseteq [K]$ in the binary action setting ($K = 2$), examining the tradeoffs between forced actions ($|A_t| = 1$) and full menu exposure ($|A_t| = 2$).

---

## 1. Action Menu Modalities ($K = 2$)

At round $t$, nature realizes state $x^{(t)}$ with true optimal action $a = y^*(x) \in \{1, 2\}$ and suboptimal action $\bar{a}$.
The algorithm selects an action menu $A_t \subseteq \{1, 2\}$:

| Action Menu | Operational Mechanism | Immediate Regret | Learning Impact |
| :--- | :--- | :--- | :--- |
| **$A_t = \{a\}$ (Forced Action / Answer)** | Human is presented only with the correct action $a$. | **Identically $0$** | Only row $a$ updates ($m_{aj} \to m_{aj} + 1$). Row $\bar{a}$ is never exposed. |
| **$A_t = \{1, 2\}$ (Full Menu / Choice)** | Human chooses between $1$ and $2$ based on shown features $F_t$. | Dependent on $F_t$ ($0$ if $F_t \in C_t$, $|D|$ otherwise) | Both rows update ($m_{1j}$ and $m_{2j}$ increment symmetrically). |

---

## 2. Why "Showing the Answer" ($A_t = \{a\}$) Fails as a Global Strategy

It is tempting to think that since showing $A_t = \{y^*(x)\}$ guarantees zero immediate regret in every round, $\mathcal{A}$ should simply show $A_t = \{y^*(x)\}$ throughout the entire training phase.

### Theorem: The Starvation of Unselected Actions
Suppose $\mathcal{A}$ sets $A_t = \{y^*(x)\}$ for all $t = 1, \dots, T$.
1. **Asymmetric Exposure**: Because nature draws $x^{(t)}$ symmetrically, each action is optimal approximately $T/2$ times. More crucially, for any specific state-feature combination, row $\bar{a}$ is never exposed during those rounds.
2. **Unassisted Graduation Failure**:
   Under unassisted graduation ($t > T$), the human receives no algorithmic help ($A_{\text{grad}} = \{1, 2\}$ and $F_{\text{grad}} = [n]$).
   The human's prediction is:
   $$\hat{y} = 1 \iff \big(h_1^{(T)} - h_2^{(T)}\big)^\top x > 0$$
   If row 2 was under-exposed, $h_2^{(T)}$ retains its initial misconceptions $h_2^{(0)}$.
   If $h_2^{(0)}$ contains large errors that are unlearned, the human will suffer severe permanent regret at graduation.

---

## 3. When is Action Menu Restriction Strictly Optimal?

Restricting the action menu to a single action is optimal in two specific regimes:

### 1. Asymmetric Row Debiasing (Accelerated Sign Flip)
When the human's prior errors are heavily asymmetric between actions ($e_{kj} \gg e_{\bar{k}j}$):
* Updating both rows causes error cancellation, requiring $\tau_j^{\text{both}} = \min\{m : \psi(m) < \frac{d_j}{e_{2j} - e_{1j}}\}$ exposures.
* Updating only row $k$ crosses the higher threshold $\frac{d_j + e_{1j}}{e_{2j}}$, flipping the sign of $g_j$ in **fewer total exposures**:
  $$\tau_j^{\text{row } k} \le \tau_j^{\text{both}}$$

### 2. High-Stakes Error Shielding
When $x^{(t)}$ is a high-stakes state ($|d^\top x^{(t)}| \gg 0$) and the zero-regret feature set is empty ($C_t = \emptyset$):
* Presenting both actions $A_t = \{1, 2\}$ will cause an inevitable mistake, incurring a massive loss $|d^\top x^{(t)}|$.
* Setting $A_t = \{a\}$ **shields the human from the high stakes**, forcing the correct action and reducing immediate regret to 0, while still advancing exposure on row $a$.

---

## 4. Summary: The Integrated Decision Rule for $(A_t, F_t)$

At round $t$, $\mathcal{A}$ joint-optimizes $(A_t, F_t)$:
1. **Low/Moderate Stakes**: Present the full menu $A_t = \{1, 2\}$ to ensure symmetric debiasing of both rows, choosing $F_t$ according to [[<G> Ex-Post c=1 Strategy]] or [[<G> Ex-Post c>=2 Strategy]].
2. **High Stakes with $C_t = \emptyset$**: Restrict to $A_t = \{a\}$ to shield immediate regret while updating row $a$.
3. **Severe Row Asymmetry**: Select $A_t = \{k\}$ when isolating a specific over-estimated action $k$ achieves a faster sign flip than full menu exposure.
