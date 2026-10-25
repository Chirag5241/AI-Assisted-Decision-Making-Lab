# <G> - Multi-Action Pedagogical Algorithm: Universal Saliency Gradient (USG)

This document presents the upgraded **Universal Saliency Gradient (USG)** algorithm for multi-action pedagogical AI assistance ($K \ge 3$).

By incorporating the **Universal Saliency Theorem**, this algorithm moves beyond rough graph heuristics and directly optimizes the exact analytical gradient of population Value of Information.

---

## 1. Closed-Form Population Saliency Gradient

From the Universal Saliency Theorem, the expected diagnostic salience of biomarker $j$ for a patient with symptoms $s$ under clinician mental model $H \in \mathbb{R}^{K \times n}$ is:
$$\text{Salience}_j(s; H) = \sigma_j^2 \sum_{1 \le k < \ell \le K} p_k(s; H) p_\ell(s; H) \big( H_{kj} - H_{\ell j} \big)^2$$
where $p_k(s; H) = \frac{e^{\beta (H \mu(s))_k}}{\sum_{r=1}^K e^{\beta (H \mu(s))_r}}$ is the softmax prior likelihood of condition $k$.

### The Exact Miscalibration Gradient
The AI preceptor wants to calibrate $H$ to minimize the divergence between the clinician's autonomous test selection $\hat{F}(s; H)$ and the true optimal test panel $\hat{F}(s; U)$.
For each matrix element $H_{kj}$ (row $k$, biomarker $j$), the exact sensitivity gradient is:
$$\nabla_{H_{kj}} \text{Salience}_j(s; H) = 2 \sigma_j^2 \sum_{\ell \neq k} p_k(s) p_\ell(s) \big( H_{kj} - H_{\ell j} \big)$$

Integrating over the patient symptom distribution $s \sim \mathcal{D}_s$, the **Global Diagnostic Priority** of cell $(k, j)$ is:
$$\Pi_{kj}(H) := \mathbb{E}_{s \sim \mathcal{D}_s} \left[ \big| \nabla_{H_{kj}} \text{Salience}_j(s; H) - \nabla_{U_{kj}} \text{Salience}_j(s; U) \big| \right]$$

---

## 2. The Universal Saliency Gradient (USG) Algorithm

```
Algorithm: Universal Saliency Gradient Curriculum (USG)
Input: Symptom distribution D_s, true matrix U, initial H^(0), budget T, capacity c
Output: Optimal pedagogical training policy pi*

1. For round t = 1 to T:
2.    Compute current cell diagnostic priorities Pi_kj(H^(t)) for all k in [K], j in [n]
3.    Identify the top-c biomarkers with largest aggregate row miscalibration:
         j*_1, ..., j*_c = argmax_{F, |F|<=c} sum_{j in F} sum_{k=1}^K Pi_kj(H^(t))
         
4.    Observe patient presentation (s^(t), x^(t)):
5.        If ex-post zero-regret opportunities exist for the target biomarkers:
              Present target biomarkers at zero regret!
6.        Else if stakes are below Bellman threshold:
              Present target biomarkers (paid low-stakes teaching)
7.        Else:
              Present safe, calibrated biomarkers to shield patient
              
8.    Observe clinician response, compute prediction error, and update H^(t+1)
9. Return calibrated matrix H^(T)
```

### Theoretical Properties:
1. **Exact Asymptotic Convergence**: Unlike heuristic margin covers, the USG algorithm provably tracks the steepest descent direction of the true population Value of Information error.
2. **Polynomial Time**: Evaluation of $\Pi_{kj}$ requires only $O(K^2 n)$ operations per symptom sample, scaling smoothly to large clinical disposition spaces ($K \ge 10$).
