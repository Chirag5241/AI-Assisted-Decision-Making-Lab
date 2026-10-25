Consider a human $\mathcal{H}$ who needs to make a move $k \in [K]$ from a state $x \in \mathbb{R}^n$. 

The ground truth utility matrix $U \in \mathbb{R}^{K \times n}$ with rows $u_1, ..., u_K$, gives us the utility of each move, $k$, from the state $x$ as $u_k^T x$.  Thus the optimal action to make from $x$ is $y^*(x) = \arg\max_k u_k^T x$. 

At time $t$, the human holds a belief $H^{(t)} \in \mathbb{R}^{K \times n}$ with rows $h_1^{(t)}, ..., h_K^{(t)}$, of the utilities of each move, from which she makes the move $\hat{y}_t = \arg\max_k (h_k^{(t)})^T x$. 

We look at the problem of an algorithm $\mathcal{A}$ assisting $\mathcal{H}$ in learning the true utilities and making the correct move. This consists of two phases: a training phase and graduation phase. 

The training phase occurs for $t = 1, ..., T$. Here,:
- $x^{(t)} \sim \mathcal{D}$ is realized
- $\mathcal{A}$ sees $x^{(t)}$ and picks $F_t \subseteq [n]$ and $A_t \subseteq [K]$ with $|F_t| < c$ and $a_{\min} \leq |A_t| \leq a_{\max}$ to show to $\mathcal{H}$
- $\mathcal{H}$ sees $A_t$ and $F_t$, and makes move $\hat{y}_t = \arg\max_{k \in A_t} \sum_{j \in F_t} h_{jk}^{(t)} x_{jk}$ and observes the decision regret loss: $\ell_{\text{reg}}(x^{(t)}, \hat{y}_t) = \max_k u^T x^{(t)} - u_{\hat{y}_t}^T x^{(t)}$ 
- $\mathcal{H}$ updates her matrix, contracting towards the true utility for the features and actions observed: $h_{kj}^{(t + 1)} - u_{kj} = \psi(m_{kj}(t)) [h_{kj}^{(t)} - u_{kj}]$ for $(k, j) \in A_T \times F_t)$ ,                                            and where $m_{kj}(t) = |\{ s < t : k \in A_s, j \in F_s \}|$ and $\psi: \mathbb{N} \rightarrow [0, 1]$ is non-increasing with $\psi(0) = 1$. 

The graduation phase occurs after time $T$, where for each $t > T$:
- $\mathcal{A}$ picks $A_t, F_t$ to show $\mathcal{H}$
- $x^{(t)} \sim \mathcal{D}$ is realized, actions $A_t$ and corresponding features of $F_t$ are shown to $\mathcal{H}$
- $\mathcal{H}$ makes move $\hat{y}_t = \arg\max_{k \in A_t} \sum_{j \in F_t} h_{jk}^{(t)} x_{jk}$ and observes loss: $\ell_{\text{grad}}(x^{(t)}, \hat{y}_t)$ 
- $\mathcal{H}$ updates her matrix as in the training phase

The goal of the algorithm is to select subsets of actions and features features $(A_t)_{t > 0}, (F_t)_{t > 0}$ minimizing the loss:
$$ \lambda \sum_{t = 1}^{T} \ell_{\text{reg}}(x^{(t)}, \hat{y}_t) + \sum_{t > T} \delta^{t}\ell_{\text{grad}}(x^{(t)}, \hat{y}_t)$$ 
where $\lambda \in [0, 1)$ is small cost incurred ($\lambda \approx 0$) on the training phase, and $\delta \in (0, 1)$ is discount factor.

To start, we consider that $\mathcal{A}$ has full access to $U, H^{(0)}, \psi$ and $\mathcal{D} = \mathcal{N}(0, I_n)$.

**Things to iron out:**
- [ ] best grad loss -- want to have something like decision regret, but also some sort of measure for how close $H^{(t)}$ and $U$ are
	- [ ] or might be worth looking into switching training loss to $||H - U||$ and grad loss to decision regret
	- [ ] don't want AI to be a crutch in the end, human should have learnt most of $U$
- [ ] should graduation phase be assisted or unassisted -- maybe some analysis of one versus another
