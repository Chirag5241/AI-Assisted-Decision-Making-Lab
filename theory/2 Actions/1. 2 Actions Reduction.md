We first look at the case where there are only two actions: $K = 2$. Here, the ground truth utility vectors are $u_1, u_2 \in \mathbb{R}^n$ and the human internal belief vectors are $h_1^{(t)}, h_2^{(t)} \in \mathbb{R}^n$.

##### **Reduction to Difference Vectors:**
We define the ground truth difference vector as $d := u_1 - u_2 \in \mathbb{R}^n \setminus \{0\}$, where we assume $d \neq 0$, as otherwise the two actions are the same. 
At time $t$, the perceived difference vector is $p^{(t)} := h_1^{(t)} - h_2^{(t)} \in \mathbb{R}^n$. 

Then, for $x \in \mathbb{R}^n$, the first action is optimal iff $u_1^Tx > u_2^Tx \iff d^T x > 0$.
For any prediction $\hat{y} \in \{1, 2\}$, the regret suffered is: $\ell_{\text{reg}}(x, \hat{y}) = |d^T x| \cdot \mathbb{1}[\hat{y} \neq y^*(x)]$ . 

##### **Learning Dynamics:**  
Since both actions are shown at each time, the exposure count for both rows is identical: $m_{1j}(t) = m_{2j}(t) =: m_j(t)$. We can thus rewrite the learning dynamic, $h_{kj}^{(t+1)} - u_{kj} = \psi(m_{kj}(t))(h_{kj}^{(0)} - u_{kj})$, by taking the difference at $k = 2$ from $k = 1$ to get that the perceived difference vector evolves deterministically as: $$p_j(m_j) = (1 - \psi(m_j))d_j + \psi(m_j) p_j^{(0)}$$
where $m_j = m_j(t)$ and $p_j(m_j) := p_j^{(t)}$. 

##### **Feature Classification:**
We can now classify each feature $j \in [n]$ into one of the following categories:
1. **Informative**: $\mathcal{H}$ would pick the correct action given only feature $j$: $d_j \neq 0$ and $p_j^{(0)} d_j > 0$.
	* Since $\psi(m_j)$ decreasing in $m_j$ and is positive,  $p_j^{(t)}(m_j) \cdot d_j > 0$ , thus the feature will always remain informative.
2. **Misunderstood**: $\mathcal{H}$ would pick the incorrect action given only feature $j$: $d_j \neq 0$ and $p_j^{(0)}d_j < 0$.
	* Since the learning dynamic updates $p_j^{(t)}$ to contract towards $d_j$, the more it is observed, we can see that $p_j^{(t)} d_j > 0$, when: $$p_j^{(t)} d_j = (1 - \psi(m_j))d_j^2 + \psi(m_j) p_j^{(0)} d_j = (1 - \psi(m_j)) d_j^2 - \psi(m_j) p_j^{(0)} |d_j| > 0$$
		which is true for $\psi(m_j) <  \frac{|d_j|}{|p_j^{(0)}| + |d_j|}$ . Thus after seeing a misunderstood feature enough, it will become informative.
3. **Distraction**: $\mathcal{H}$ would take input from feature $j$, when in reality it is irrelevant in distinguishing between the two actions: $d_j = 0$ and $|p_j^{(0)}| > 0$. 
	* The weight of the importance given to this feature decays as: $p_j^{(t)}(m_j) = \psi(m_j) p_j^{(0)}$






