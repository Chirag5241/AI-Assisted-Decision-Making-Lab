# <G> Ex-Post c=1 Strategy

This note formalizes the **Ex-Post, Single-Feature ($c = 1$)** training strategy for binary actions ($K=2$).

---

## 1. Problem Setting

At each training round $t$, nature draws $x^{(t)} \sim \mathcal{N}(0, I_n)$.
The algorithm $\mathcal{A}$ **observes $x^{(t)}$ before choosing** which single feature $j \in [n]$ to show.
The ground truth decision margin is $D := d^\top x^{(t)}$. Action 1 is optimal iff $D > 0$.

Let $\sigma_j := \operatorname{sgn}(g_j(m_j))$, $z_j := \sigma_j x_j \sim \mathcal{N}(0, 1)$, and $w_j := \sigma_j d_j$.
Showing feature $j$ makes the human predict correctly iff $j \in Z(x)$, where:
$$Z(x) := \big\{ j \in [n] : \operatorname{sgn}(z_j) = \operatorname{sgn}(D) \big\}$$

---

## 2. Structural Theorems

### Theorem 1: Zero-Regret Availability & Arcsine Law
* **Empty Set Probability**: $\mathbb{P}(Z(x) = \emptyset) = 2^{1-n}$ when all beliefs are inverted ($w_j < 0$ for all $j$), and $0$ when all beliefs are correct.
* **Sheppard's Arcsine Law**: For any bad feature $j$ ($w_j < 0$), the probability it lands in $Z(x)$ (a **free zero-regret teaching opportunity**) is:
  $$\mathbb{P}(j \in Z(x)) = \frac{1}{2} - \frac{1}{\pi}\arcsin\left(\frac{|d_j|}{\|d\|}\right)$$

### Theorem 2: Value Monotonicity
The value function satisfies $V(m + e_i) \le V(m)$ for all $i \in [n]$ and all $m \in \mathbb{N}^n$.

---

## 3. Algorithmic Decision Rules for $\mathcal{A}$

Let $\nu_j(m) = \delta V(m + e_j)$, $\nu_Z = \min_{j \in Z(x)} \nu_j(m)$, and $\nu_{Z^c} = \min_{j \notin Z(x)} \nu_j(m)$.

```
                             State x realized: Compute Z(x)
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
                   Z(x) is EMPTY                    Z(x) is NON-EMPTY
                         │                                   │
               Immediate regret is |D|             Zero-regret candidate:
               for ANY feature.                    j* = argmin_{j in Z} \nu_j
                         │                                   │
                         ▼                                   ▼
               Max-Learning Action:                Compare with low stakes:
               Show argmin_{j} \nu_j(m)            Is |D| < \nu_Z - \nu_{Z^c}?
               (Teach most urgent feature)                   /          \
                                                           YES           NO
                                                           /              \
                                                    Paid Teaching:      Free Teaching:
                                                    Show argmin_{Z^c}   Show j* in Z(x)
                                                    (Pay |D| regret)    (0 regret!)
```

### Direct Resolution to Core Questions:
1. **When $Z(x) = \emptyset$ (Zero-regret impossible)**:
   * Showing any feature incurs regret $|D|$.
   * **Rule**: Since immediate loss is fixed, $\mathcal{A}$ chooses $j^* = \arg\min_{j \in [n]} \nu_j(m)$—teaching the feature with the highest future marginal return (the most divergent unlearned feature).
2. **When $Z(x) \ne \emptyset$ (Zero-regret possible)**:
   * **Rule**: $\mathcal{A}$ shows the feature in $Z(x)$ with the smallest $\nu_j(m)$. This provides **free teaching at zero regret at any stakes**.
   * Exception: Paid teaching ($\arg\min_{j \notin Z} \nu_j$) is selected only if stakes $|D|$ are smaller than the future continuation advantage $\nu_Z - \nu_{Z^c}$.
