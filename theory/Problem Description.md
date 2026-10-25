Consider a human $\mathcal{H}$ who needs to take an action $k \in [K]$ based on an observation: $s \in \mathbb{R}^d$ of a hidden state $x \in \mathbb{R}^n$, according to a prior distribution $\mathcal{D} = \mathcal{N}(0, \Sigma)$. 

The ground truth utility matrix $U \in \mathbb{R}^{K \times n}$ with rows $u_1, ..., u_K$, gives us the utility of each move, $k$, from the state $x$ as $u_k^T x$.  Thus the optimal action to make from $x$ is $y^*(x) = \arg\max_k u_k^T x$. 

The human makes an action based on only an observation she sees, $s = Bx$, where $B \in \mathbb{R}^{d \times n}$ and a limited number of revealed true features $x_F := \{x_j : j \in F\}$ for $F \subseteq [n]$ with $|F| \leq c$. From the observation, she is able to estimate the true features: $x | s \sim \mathcal{N}(\mu(s), \Sigma_{x | s})$ where $\mu(s) = \Sigma B^\top (B \Sigma B^\top)^{-1} s$ and $\Sigma_{x \mid s} = \Sigma - \Sigma B^\top (B \Sigma B^\top)^{-1} B \Sigma$.

At time $t$, the human holds belief matrix  $H^{(t)} \in \mathbb{R}^{K \times n}$ with rows $h_1^{(t)}, ..., h_K^{(t)}$. Thus from the observation $s^{(t)}$ and revealed features $x^{(t)}_F$, she holds the utility for action $j \in [K]$ as: $$\hat{u}_j(F, x_F, s; H^{(t)}) = \sum_{j \in F} H_{kj}^{(t)} x_j + \sum_{j \notin F} H_{kj}^{(t)} \mu_j(s)$$
and takes action $\hat{y}\big(F, x_F, s; H^{(t)}\big) = \arg\max_{k \in [K]} \hat{u}_k(F, x_F, s; H^{(t)})$.

We look at the problem of an algorithm $\mathcal{A}$ assisting $\mathcal{H}$ in learning the true utilities, making the correct move, and selecting the correct set of features to reveal. This consists of two phases: a training phase and graduation phase. 

The training phase occurs for $t = 1, ..., T$. Here,:
- $x^{(t)} \sim \mathcal{N}(0, \Sigma)$ and $s^{(t)} = B x^{(t)}$ are realized 
- $\mathcal{A}$ sees $s^{(t)}$ and picks $F_t \subseteq [n]$ with $|F_t| \leq c$ and to show to $\mathcal{H}$
- $\mathcal{H}$ observes $s^{(t)}$, and $(F_t, x^{(t)}_{F^t})$, makes move $\hat{y}\big(F, x_F, s; H^{(t)}\big)$ and observes the decision regret loss: $\ell_{\text{reg}}(x^{(t)}, \hat{y}_t) = \max_k u^T x^{(t)} - u_{\hat{y}_t}^T x^{(t)}$ 
- $\mathcal{H}$ updates her matrix, contracting towards the true utility for the features and actions observed: $H_{kj}^{(t + 1)} - U_{kj} = \psi(m_{kj}(t)) [H_{kj}^{(t)} - U_{kj}]$ for $(k, j) \in [K] \times F_t$ ,                                            and where $m_{kj}(t) = |\{ s < t : k \in [K], j \in F_s \}|$ and $\psi: \mathbb{N} \rightarrow [0, 1]$ is non-increasing with $\psi(0) = 1$. For unmeasured features, the beliefs remain the same.

The graduation phase occurs after time $T$, where for each $t > T$:
- $\mathcal{H}$ observes $s^{(t)}$ and picks subset $F^{(t)} = F^*_{\mathcal{H}}(s^{(t)}; H^{(T)})$ where $F^{(t)}$ is chosen to maximize their perceived value of information: $$ F_{\mathcal{H}}^*(s; H^{(T)}) = \arg\max_{|F| \le c} \mathbb{E}_{x_F \mid s} \left[ \max_{k \in [K]} \hat{u}_k(F, x_F, s; H^{(t)}) \right]$$
- The true features $(F_t, x^{(t)}_{F^{(t)}})$ are revealed to $\mathcal{H}$. 
- $\mathcal{H}$ makes move $\hat{y}_t = \hat{y}(\hat{F}_t, x_{\hat{F}_t}^{(t)}, s^{(t)}; H^{(T)})$ and observes loss: $\ell_{\text{grad}}(x^{(t)}, \hat{y}_t)$ 

The goal of the algorithm is to select training policy: $\pi = (F^{(1)}, F^{(2)}, ..., F^{(T)})$ minimizing the combined objective: 
$$\mathcal{L}(\pi) := \lambda \sum_{t=1}^T \mathbb{E}\big[\ell_{\text{reg}}(x^{(t)}, \hat{y}_t)\big] + \sum_{t > T} \delta^{(t-T)} \mathbb{E}\big[\ell_{\text{grad}}(x^{(t)}, s^{(t)}; H^{(T)})\big]$$ 
where $\lambda \in [0, 1)$ is small cost incurred ($\lambda \approx 0$) on the training phase, and $\delta \in (0, 1)$ is discount factor.

To start, we consider that $\mathcal{A}$ has full access to $U, H^{(0)}, \psi$ .


To confirm/further:
- [ ] consider lossy H after graduation... if some feature isn't seen in some time window, can be forgotten?
- [ ] what happens algm is able to reduce subset of actions shown
- [ ] moving from gaussian to arbitrary distribution where certain symptoms/underlying conditions more frequent than others -- standardize and turn into gaussian?
- [ ] Should $s = Bx$ where there is one $B$ matrix for each patient, or $B$ is dependent on each patient -- mask for what they disclose to doctor

