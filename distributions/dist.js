// The distributions, their maths, and how they connect. Shared by the explorer (index.html) and the map (map.html).
(function (root) {
  "use strict";

  // ---------- special functions ----------
  const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
    x -= 1;
    let a = LG[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  const gammaFn = (x) => Math.exp(lgamma(x));
  const lchoose = (n, k) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
  const TINY = 1e-300;

  // regularized lower incomplete gamma P(a, x)
  function gammaP(a, x) {
    if (x <= 0) return 0;
    const pre = Math.exp(-x + a * Math.log(x) - lgamma(a));
    if (x < a + 1) {
      let ap = a, s = 1 / a, t = s;
      for (let n = 0; n < 1000; n++) { ap += 1; t *= x / ap; s += t; if (Math.abs(t) < Math.abs(s) * 1e-15) break; }
      return Math.min(1, s * pre);
    }
    let b = x + 1 - a, c = 1 / TINY, d = 1 / b, h = d;
    for (let i = 1; i < 1000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < TINY) d = TINY;
      c = b + an / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return Math.max(0, 1 - pre * h);
  }

  // regularized incomplete beta I_x(a, b)
  function betacf(x, a, b) {
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < TINY) d = TINY;
    d = 1 / d;
    let h = d;
    for (let m = 1; m < 500; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return h;
  }
  function betaI(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(x, a, b) / a : 1 - bt * betacf(1 - x, b, a) / b;
  }

  function erf(x) {
    const s = x < 0 ? -1 : 1;
    x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const Phi = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
  const SQ2PI = Math.sqrt(2 * Math.PI);
  const EULER = 0.5772156649015329;

  // ---------- random numbers (seeded, so a redraw is reproducible) ----------
  function rng(seed) {
    let s = seed >>> 0;
    const u = () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
    };
    let spare = null;
    const n = () => {
      if (spare !== null) { const v = spare; spare = null; return v; }
      const r = Math.sqrt(-2 * Math.log(u())), a = 2 * Math.PI * u();
      spare = r * Math.sin(a);
      return r * Math.cos(a);
    };
    const e = () => -Math.log(u());
    const g = (k) => {   // Gamma(k, 1), Marsaglia and Tsang
      if (k < 1) return g(k + 1) * Math.pow(u(), 1 / k);
      const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
      for (;;) {
        let x, v;
        do { x = n(); v = 1 + c * x; } while (v <= 0);
        v = v * v * v;
        const w = u();
        if (w < 1 - 0.0331 * x * x * x * x || Math.log(w) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
      }
    };
    const pois = (lam) => { let t = e() / lam, k = 0; while (t <= 1) { k++; t += e() / lam; } return k; };
    return { u, n, e, g, pois };
  }

  // ---------- the distributions ----------
  // params: [key, label, min, max, step, default]. x: fixed plot range; y: fixed top of the density axis.
  // Discrete ones give integer x ranges and a pmf; lo(p) is the smallest value with mass.
  const D = [
    // ----- discrete -----
    {
      id: "bernoulli", name: "Bernoulli", kind: "discrete",
      params: [["p", "p (chance of yes)", 0, 1, 0.05, 0.3]],
      x: [0, 1], y: 1,
      story: "One yes/no trial.",
      gen: "1 if a uniform draw lands below p, else 0.",
      formula: "P(1) = p,  P(0) = 1 − p",
      support: () => "{0, 1}",
      lo: () => 0,
      pdf: (k, p) => (k === 0 ? 1 - p.p : k === 1 ? p.p : 0),
      sample: (r, p) => (r.u() < p.p ? 1 : 0),
      mean: (p) => p.p, variance: (p) => p.p * (1 - p.p),
    },
    {
      id: "binomial", name: "Binomial", kind: "discrete",
      params: [["n", "n (trials)", 1, 40, 1, 10], ["p", "p (chance of yes)", 0, 1, 0.05, 0.3]],
      x: [0, 40], y: 0.6,
      story: "Number of yeses in n independent trials.",
      gen: "Run n Bernoulli(p) trials and count the yeses.",
      formula: "P(k) = C(n, k) pᵏ (1 − p)ⁿ⁻ᵏ",
      support: (p) => `{0, …, ${p.n}}`,
      lo: () => 0,
      pdf: (k, p) => {
        if (k < 0 || k > p.n) return 0;
        if (p.p === 0) return k === 0 ? 1 : 0;
        if (p.p === 1) return k === p.n ? 1 : 0;
        return Math.exp(lchoose(p.n, k) + k * Math.log(p.p) + (p.n - k) * Math.log(1 - p.p));
      },
      sample: (r, p) => { let c = 0; for (let i = 0; i < p.n; i++) if (r.u() < p.p) c++; return c; },
      mean: (p) => p.n * p.p, variance: (p) => p.n * p.p * (1 - p.p),
    },
    {
      id: "discUniform", name: "Discrete uniform", kind: "discrete",
      params: [["a", "a (lowest)", 0, 10, 1, 1], ["b", "b (highest)", 0, 20, 1, 6]],
      fix: (p) => { if (p.b < p.a) p.b = p.a; },
      x: [0, 20], y: 0.6,
      story: "Every whole number from a to b equally likely, like a fair die.",
      gen: "a + ⌊(b − a + 1)·U⌋ for a uniform U.",
      formula: "P(k) = 1 / (b − a + 1)",
      support: (p) => `{${p.a}, …, ${p.b}}`,
      lo: (p) => p.a,
      pdf: (k, p) => (k >= p.a && k <= p.b ? 1 / (p.b - p.a + 1) : 0),
      sample: (r, p) => p.a + Math.floor(r.u() * (p.b - p.a + 1)),
      mean: (p) => (p.a + p.b) / 2, variance: (p) => ((p.b - p.a + 1) ** 2 - 1) / 12,
    },
    {
      id: "hypergeom", name: "Hypergeometric", kind: "discrete",
      params: [["N", "N (items)", 10, 100, 1, 50], ["K", "K (successes in it)", 0, 100, 1, 20], ["n", "n (draws)", 1, 30, 1, 10]],
      fix: (p) => { if (p.K > p.N) p.K = p.N; if (p.n > p.N) p.n = p.N; },
      x: [0, 30], y: 0.6,
      story: "Successes in n draws without replacement from N items, K of them successes.",
      gen: "Draw n items one at a time without putting them back; count successes.",
      formula: "P(k) = C(K, k) C(N − K, n − k) / C(N, n)",
      support: (p) => `{${Math.max(0, p.n - (p.N - p.K))}, …, ${Math.min(p.n, p.K)}}`,
      lo: (p) => Math.max(0, p.n - (p.N - p.K)),
      pdf: (k, p) => {
        if (k < Math.max(0, p.n - (p.N - p.K)) || k > Math.min(p.n, p.K)) return 0;
        return Math.exp(lchoose(p.K, k) + lchoose(p.N - p.K, p.n - k) - lchoose(p.N, p.n));
      },
      sample: (r, p) => {
        let K = p.K, N = p.N, c = 0;
        for (let i = 0; i < p.n; i++) { if (r.u() < K / N) { c++; K--; } N--; }
        return c;
      },
      mean: (p) => p.n * p.K / p.N,
      variance: (p) => (p.N > 1 ? p.n * (p.K / p.N) * (1 - p.K / p.N) * (p.N - p.n) / (p.N - 1) : 0),
    },
    {
      id: "geometric", name: "Geometric", kind: "discrete",
      params: [["p", "p (chance of yes)", 0.05, 1, 0.05, 0.3]],
      x: [0, 25], y: 0.6,
      story: "Number of trials until the first yes.",
      gen: "Run Bernoulli(p) trials until the first yes; count them.",
      formula: "P(k) = (1 − p)ᵏ⁻¹ p",
      support: () => "{1, 2, 3, …}",
      lo: () => 1,
      pdf: (k, p) => (k >= 1 ? Math.pow(1 - p.p, k - 1) * p.p : 0),
      sample: (r, p) => { let k = 1; while (r.u() >= p.p) k++; return k; },
      mean: (p) => 1 / p.p, variance: (p) => (1 - p.p) / (p.p * p.p),
    },
    {
      id: "negbin", name: "Negative binomial", kind: "discrete",
      params: [["r", "r (yeses wanted)", 1, 20, 1, 3], ["p", "p (chance of yes)", 0.05, 1, 0.05, 0.4]],
      x: [0, 60], y: 0.4,
      story: "Number of trials until the r-th yes.",
      gen: "Add up r independent Geometric(p) waits.",
      formula: "P(k) = C(k − 1, r − 1) pʳ (1 − p)ᵏ⁻ʳ",
      support: (p) => `{${p.r}, ${p.r + 1}, …}`,
      lo: (p) => p.r,
      pdf: (k, p) => {
        if (k < p.r) return 0;
        if (p.p === 1) return k === p.r ? 1 : 0;
        return Math.exp(lchoose(k - 1, p.r - 1) + p.r * Math.log(p.p) + (k - p.r) * Math.log(1 - p.p));
      },
      sample: (r, p) => { let t = 0; for (let i = 0; i < p.r; i++) { let k = 1; while (r.u() >= p.p) k++; t += k; } return t; },
      mean: (p) => p.r / p.p, variance: (p) => p.r * (1 - p.p) / (p.p * p.p),
    },
    {
      id: "poisson", name: "Poisson", kind: "discrete",
      params: [["lam", "λ (average count)", 0.5, 20, 0.5, 4]],
      x: [0, 40], y: 0.6,
      story: "Count of rare, independent events in a fixed window.",
      gen: "Lay Exponential(λ) gaps end to end; count how many fit in time 1.",
      formula: "P(k) = λᵏ e^(−λ) / k!",
      support: () => "{0, 1, 2, …}",
      lo: () => 0,
      pdf: (k, p) => (k >= 0 ? Math.exp(k * Math.log(p.lam) - p.lam - lgamma(k + 1)) : 0),
      sample: (r, p) => r.pois(p.lam),
      mean: (p) => p.lam, variance: (p) => p.lam,
    },
    {
      id: "betabinom", name: "Beta-binomial", kind: "discrete",
      params: [["n", "n (trials)", 1, 40, 1, 10], ["a", "α", 0.2, 10, 0.1, 2], ["b", "β", 0.2, 10, 0.1, 2]],
      x: [0, 40], y: 0.6,
      story: "A binomial whose chance p is itself uncertain.",
      gen: "Draw p from Beta(α, β), then run n Bernoulli(p) trials.",
      formula: "P(k) = C(n, k) B(k + α, n − k + β) / B(α, β)",
      support: (p) => `{0, …, ${p.n}}`,
      lo: () => 0,
      pdf: (k, p) => {
        if (k < 0 || k > p.n) return 0;
        const lb = (x, y) => lgamma(x) + lgamma(y) - lgamma(x + y);
        return Math.exp(lchoose(p.n, k) + lb(k + p.a, p.n - k + p.b) - lb(p.a, p.b));
      },
      sample: (r, p) => {
        const g1 = r.g(p.a), q = g1 / (g1 + r.g(p.b));
        let c = 0;
        for (let i = 0; i < p.n; i++) if (r.u() < q) c++;
        return c;
      },
      mean: (p) => p.n * p.a / (p.a + p.b),
      variance: (p) => p.n * p.a * p.b * (p.a + p.b + p.n) / ((p.a + p.b) ** 2 * (p.a + p.b + 1)),
    },
    {
      id: "skellam", name: "Skellam", kind: "discrete",
      params: [["m1", "μ₁", 0.5, 10, 0.5, 3], ["m2", "μ₂", 0.5, 10, 0.5, 2]],
      x: [-20, 20], y: 0.4,
      story: "Difference of two independent Poisson counts, such as goals for minus goals against.",
      gen: "Draw Poisson(μ₁) and Poisson(μ₂); subtract.",
      formula: "P(k) = Σⱼ Pois(j + k; μ₁) · Pois(j; μ₂)",
      support: () => "all integers",
      lo: () => -80,
      pdf: (k, p) => {
        const lp = (j, m) => j * Math.log(m) - m - lgamma(j + 1);
        let s = 0;
        for (let j = Math.max(0, -k); j < Math.max(0, -k) + 120; j++) s += Math.exp(lp(j + k, p.m1) + lp(j, p.m2));
        return s;
      },
      sample: (r, p) => r.pois(p.m1) - r.pois(p.m2),
      mean: (p) => p.m1 - p.m2, variance: (p) => p.m1 + p.m2,
    },

    // ----- continuous -----
    {
      id: "uniform", name: "Uniform", kind: "continuous",
      params: [["a", "a (left end)", -5, 5, 0.1, 0], ["b", "b (right end)", -5, 5, 0.1, 1]],
      fix: (p) => { if (p.b <= p.a) p.b = Math.round((p.a + 0.1) * 10) / 10; },
      x: [-6, 6], y: 1.2,
      story: "Any value between a and b equally likely. The raw material every other sampler starts from.",
      gen: "a + (b − a)·U, where U is the computer's uniform random number.",
      formula: "f(x) = 1 / (b − a)",
      support: (p) => `[${p.a}, ${p.b}]`,
      pdf: (x, p) => (x >= p.a && x <= p.b ? 1 / (p.b - p.a) : 0),
      cdf: (x, p) => (x <= p.a ? 0 : x >= p.b ? 1 : (x - p.a) / (p.b - p.a)),
      sample: (r, p) => p.a + (p.b - p.a) * r.u(),
      mean: (p) => (p.a + p.b) / 2, variance: (p) => (p.b - p.a) ** 2 / 12,
    },
    {
      id: "normal", name: "Normal (Gaussian)", kind: "continuous",
      params: [["mu", "μ (centre)", -3, 3, 0.1, 0], ["s", "σ (spread)", 0.5, 3, 0.1, 1]],
      x: [-8, 8], y: 0.85,
      story: "Sums of many small independent effects. The bell curve.",
      gen: "Box–Muller: √(−2 ln U₁)·cos(2πU₂), then scale by σ and shift by μ.",
      formula: "f(x) = e^(−(x − μ)² / 2σ²) / (σ√(2π))",
      support: () => "all real numbers",
      pdf: (x, p) => Math.exp(-0.5 * ((x - p.mu) / p.s) ** 2) / (p.s * SQ2PI),
      cdf: (x, p) => Phi((x - p.mu) / p.s),
      sample: (r, p) => p.mu + p.s * r.n(),
      mean: (p) => p.mu, variance: (p) => p.s * p.s,
    },
    {
      id: "lognormal", name: "Log-normal", kind: "continuous",
      params: [["mu", "μ (of ln X)", -1, 1.5, 0.1, 0], ["s", "σ (of ln X)", 0.2, 1.5, 0.05, 0.5]],
      x: [0, 10], y: 1.2,
      story: "Products of many positive factors: incomes, lab values, file sizes.",
      gen: "e raised to a Normal(μ, σ) draw.",
      formula: "f(x) = e^(−(ln x − μ)² / 2σ²) / (xσ√(2π))",
      support: () => "x > 0",
      pdf: (x, p) => (x > 0 ? Math.exp(-0.5 * ((Math.log(x) - p.mu) / p.s) ** 2) / (x * p.s * SQ2PI) : 0),
      cdf: (x, p) => (x > 0 ? Phi((Math.log(x) - p.mu) / p.s) : 0),
      sample: (r, p) => Math.exp(p.mu + p.s * r.n()),
      mean: (p) => Math.exp(p.mu + p.s * p.s / 2),
      variance: (p) => (Math.exp(p.s * p.s) - 1) * Math.exp(2 * p.mu + p.s * p.s),
    },
    {
      id: "exponential", name: "Exponential", kind: "continuous",
      params: [["lam", "λ (rate)", 0.2, 2, 0.05, 1]],
      x: [0, 8], y: 2,
      story: "Waiting time until the next event when events arrive at a steady rate. Memoryless.",
      gen: "−ln(U) / λ.",
      formula: "f(x) = λ e^(−λx)",
      support: () => "x ≥ 0",
      pdf: (x, p) => (x >= 0 ? p.lam * Math.exp(-p.lam * x) : 0),
      cdf: (x, p) => (x > 0 ? 1 - Math.exp(-p.lam * x) : 0),
      sample: (r, p) => r.e() / p.lam,
      mean: (p) => 1 / p.lam, variance: (p) => 1 / (p.lam * p.lam),
    },
    {
      id: "erlang", name: "Erlang", kind: "continuous",
      params: [["k", "k (events)", 1, 10, 1, 3], ["lam", "λ (rate)", 0.5, 4, 0.1, 1]],
      x: [0, 20], y: 1,
      story: "Waiting time until the k-th event.",
      gen: "Add up k Exponential(λ) waits.",
      formula: "f(x) = λᵏ xᵏ⁻¹ e^(−λx) / (k − 1)!",
      support: () => "x ≥ 0",
      pdf: (x, p) => (x > 0 ? Math.exp(p.k * Math.log(p.lam) + (p.k - 1) * Math.log(x) - p.lam * x - lgamma(p.k)) : (x === 0 && p.k === 1 ? p.lam : 0)),
      cdf: (x, p) => gammaP(p.k, p.lam * x),
      sample: (r, p) => { let t = 0; for (let i = 0; i < p.k; i++) t += r.e(); return t / p.lam; },
      mean: (p) => p.k / p.lam, variance: (p) => p.k / (p.lam * p.lam),
    },
    {
      id: "gamma", name: "Gamma", kind: "continuous",
      params: [["k", "k (shape)", 0.5, 10, 0.1, 2], ["th", "θ (scale)", 0.2, 3, 0.1, 1]],
      x: [0, 20], y: 1,
      story: "Erlang with a shape that need not be whole. Positive, right-skewed amounts.",
      gen: "Marsaglia–Tsang rejection sampler, scaled by θ.",
      formula: "f(x) = xᵏ⁻¹ e^(−x/θ) / (Γ(k) θᵏ)",
      support: () => "x > 0",
      pdf: (x, p) => (x > 0 ? Math.exp((p.k - 1) * Math.log(x) - x / p.th - lgamma(p.k) - p.k * Math.log(p.th)) : 0),
      cdf: (x, p) => gammaP(p.k, x / p.th),
      sample: (r, p) => p.th * r.g(p.k),
      mean: (p) => p.k * p.th, variance: (p) => p.k * p.th * p.th,
    },
    {
      id: "chi2", name: "Chi-squared", kind: "continuous",
      params: [["nu", "ν (degrees of freedom)", 1, 20, 1, 4]],
      x: [0, 40], y: 0.5,
      story: "Sum of ν squared standard normals. Measures total squared error.",
      gen: "Square ν standard normal draws and add them.",
      formula: "f(x) = x^(ν/2 − 1) e^(−x/2) / (2^(ν/2) Γ(ν/2))",
      support: () => "x > 0",
      pdf: (x, p) => (x > 0 ? Math.exp((p.nu / 2 - 1) * Math.log(x) - x / 2 - (p.nu / 2) * Math.LN2 - lgamma(p.nu / 2)) : 0),
      cdf: (x, p) => gammaP(p.nu / 2, x / 2),
      sample: (r, p) => { let s = 0; for (let i = 0; i < p.nu; i++) { const z = r.n(); s += z * z; } return s; },
      mean: (p) => p.nu, variance: (p) => 2 * p.nu,
    },
    {
      id: "chi", name: "Chi", kind: "continuous",
      params: [["nu", "ν (dimensions)", 1, 10, 1, 3]],
      x: [0, 6], y: 0.9,
      story: "Length of a random vector with ν standard normal coordinates.",
      gen: "√(sum of ν squared standard normals).",
      formula: "f(x) = xᵛ⁻¹ e^(−x²/2) / (2^(ν/2 − 1) Γ(ν/2))",
      support: () => "x ≥ 0",
      pdf: (x, p) => (x > 0 ? Math.exp((p.nu - 1) * Math.log(x) - x * x / 2 - (p.nu / 2 - 1) * Math.LN2 - lgamma(p.nu / 2)) : (x === 0 && p.nu === 1 ? Math.sqrt(2 / Math.PI) : 0)),
      cdf: (x, p) => gammaP(p.nu / 2, x * x / 2),
      sample: (r, p) => { let s = 0; for (let i = 0; i < p.nu; i++) { const z = r.n(); s += z * z; } return Math.sqrt(s); },
      mean: (p) => Math.SQRT2 * Math.exp(lgamma((p.nu + 1) / 2) - lgamma(p.nu / 2)),
      variance: (p) => p.nu - 2 * Math.exp(2 * (lgamma((p.nu + 1) / 2) - lgamma(p.nu / 2))),
    },
    {
      id: "halfnormal", name: "Half-normal", kind: "continuous",
      params: [["s", "σ", 0.5, 3, 0.1, 1]],
      x: [0, 8], y: 1.7,
      story: "Size of a normal error, ignoring its sign.",
      gen: "σ·|Z| for a standard normal Z.",
      formula: "f(x) = √2 e^(−x²/2σ²) / (σ√π)",
      support: () => "x ≥ 0",
      pdf: (x, p) => (x >= 0 ? Math.SQRT2 * Math.exp(-x * x / (2 * p.s * p.s)) / (p.s * Math.sqrt(Math.PI)) : 0),
      cdf: (x, p) => (x > 0 ? erf(x / (p.s * Math.SQRT2)) : 0),
      sample: (r, p) => p.s * Math.abs(r.n()),
      mean: (p) => p.s * Math.sqrt(2 / Math.PI), variance: (p) => p.s * p.s * (1 - 2 / Math.PI),
    },
    {
      id: "rayleigh", name: "Rayleigh", kind: "continuous",
      params: [["s", "σ", 0.5, 3, 0.1, 1]],
      x: [0, 8], y: 1.3,
      story: "Distance from the origin of a 2-D normal point, such as wind speed or a missed shot.",
      gen: "σ·√(Z₁² + Z₂²).",
      formula: "f(x) = (x/σ²) e^(−x²/2σ²)",
      support: () => "x ≥ 0",
      pdf: (x, p) => (x >= 0 ? x / (p.s * p.s) * Math.exp(-x * x / (2 * p.s * p.s)) : 0),
      cdf: (x, p) => (x > 0 ? 1 - Math.exp(-x * x / (2 * p.s * p.s)) : 0),
      sample: (r, p) => { const a = r.n(), b = r.n(); return p.s * Math.sqrt(a * a + b * b); },
      mean: (p) => p.s * Math.sqrt(Math.PI / 2), variance: (p) => (4 - Math.PI) / 2 * p.s * p.s,
    },
    {
      id: "maxwell", name: "Maxwell–Boltzmann", kind: "continuous",
      params: [["a", "a (scale)", 0.5, 3, 0.1, 1]],
      x: [0, 10], y: 1.2,
      story: "Speed of a gas particle: length of a 3-D normal velocity.",
      gen: "a·√(Z₁² + Z₂² + Z₃²).",
      formula: "f(x) = √(2/π) x² e^(−x²/2a²) / a³",
      support: () => "x ≥ 0",
      pdf: (x, p) => (x >= 0 ? Math.sqrt(2 / Math.PI) * x * x * Math.exp(-x * x / (2 * p.a * p.a)) / p.a ** 3 : 0),
      cdf: (x, p) => gammaP(1.5, x * x / (2 * p.a * p.a)),
      sample: (r, p) => { const a = r.n(), b = r.n(), c = r.n(); return p.a * Math.sqrt(a * a + b * b + c * c); },
      mean: (p) => 2 * p.a * Math.sqrt(2 / Math.PI), variance: (p) => p.a * p.a * (3 * Math.PI - 8) / Math.PI,
    },
    {
      id: "weibull", name: "Weibull", kind: "continuous",
      params: [["k", "k (shape)", 0.5, 5, 0.1, 1.5], ["lam", "λ (scale)", 0.5, 3, 0.1, 1]],
      x: [0, 6], y: 2,
      story: "Time to failure. k < 1: early failures; k > 1: wear-out.",
      gen: "λ·E^(1/k) for an Exponential(1) draw E.",
      formula: "f(x) = (k/λ)(x/λ)ᵏ⁻¹ e^(−(x/λ)ᵏ)",
      support: () => "x ≥ 0",
      pdf: (x, p) => (x > 0 ? (p.k / p.lam) * Math.pow(x / p.lam, p.k - 1) * Math.exp(-Math.pow(x / p.lam, p.k)) : 0),
      cdf: (x, p) => (x > 0 ? 1 - Math.exp(-Math.pow(x / p.lam, p.k)) : 0),
      sample: (r, p) => p.lam * Math.pow(r.e(), 1 / p.k),
      mean: (p) => p.lam * gammaFn(1 + 1 / p.k),
      variance: (p) => p.lam * p.lam * (gammaFn(1 + 2 / p.k) - gammaFn(1 + 1 / p.k) ** 2),
    },
    {
      id: "beta", name: "Beta", kind: "continuous",
      params: [["a", "α", 0.2, 10, 0.1, 2], ["b", "β", 0.2, 10, 0.1, 5]],
      x: [0, 1], y: 5,
      story: "An unknown proportion or probability, between 0 and 1.",
      gen: "X / (X + Y) with X ~ Gamma(α), Y ~ Gamma(β).",
      formula: "f(x) = x^(α−1) (1 − x)^(β−1) / B(α, β)",
      support: () => "0 < x < 1",
      pdf: (x, p) => (x > 0 && x < 1 ? Math.exp((p.a - 1) * Math.log(x) + (p.b - 1) * Math.log(1 - x) - (lgamma(p.a) + lgamma(p.b) - lgamma(p.a + p.b))) : 0),
      cdf: (x, p) => betaI(x, p.a, p.b),
      sample: (r, p) => { const g1 = r.g(p.a); return g1 / (g1 + r.g(p.b)); },
      mean: (p) => p.a / (p.a + p.b), variance: (p) => p.a * p.b / ((p.a + p.b) ** 2 * (p.a + p.b + 1)),
    },
    {
      id: "arcsine", name: "Arcsine", kind: "continuous",
      params: [],
      x: [0, 1], y: 5,
      story: "Fraction of time a random walk spends ahead. Mass piles up at the ends.",
      gen: "sin²(πU/2).",
      formula: "f(x) = 1 / (π√(x(1 − x)))",
      support: () => "0 < x < 1",
      pdf: (x) => (x > 0 && x < 1 ? 1 / (Math.PI * Math.sqrt(x * (1 - x))) : 0),
      cdf: (x) => (x <= 0 ? 0 : x >= 1 ? 1 : (2 / Math.PI) * Math.asin(Math.sqrt(x))),
      sample: (r) => Math.sin(Math.PI * r.u() / 2) ** 2,
      mean: () => 0.5, variance: () => 0.125,
    },
    {
      id: "powerfn", name: "Power function", kind: "continuous",
      params: [["a", "α (shape)", 0.2, 5, 0.1, 2]],
      x: [0, 1], y: 5,
      story: "Beta(α, 1): the largest of α uniforms when α is whole.",
      gen: "U^(1/α).",
      formula: "f(x) = α x^(α−1)",
      support: () => "0 < x < 1",
      pdf: (x, p) => (x > 0 && x < 1 ? p.a * Math.pow(x, p.a - 1) : 0),
      cdf: (x, p) => (x <= 0 ? 0 : x >= 1 ? 1 : Math.pow(x, p.a)),
      sample: (r, p) => Math.pow(r.u(), 1 / p.a),
      mean: (p) => p.a / (p.a + 1), variance: (p) => p.a / ((p.a + 1) ** 2 * (p.a + 2)),
    },
    {
      id: "triangular", name: "Triangular", kind: "continuous",
      params: [["a", "a (left)", -3, 3, 0.1, 0], ["c", "c (peak)", -3, 3, 0.1, 1], ["b", "b (right)", -3, 3, 0.1, 2]],
      fix: (p) => {
        if (p.b <= p.a) p.b = Math.round((p.a + 0.1) * 10) / 10;
        if (p.c < p.a) p.c = p.a;
        if (p.c > p.b) p.c = p.b;
      },
      x: [-4, 4], y: 2,
      story: "A rough guess with a min, a most-likely value, and a max. The sum of two uniforms.",
      gen: "Invert its CDF at a uniform draw.",
      formula: "f rises linearly from a to c, falls linearly to b",
      support: (p) => `[${p.a}, ${p.b}]`,
      pdf: (x, p) => {
        if (x < p.a || x > p.b) return 0;
        const w = p.b - p.a;
        if (x < p.c) return 2 * (x - p.a) / (w * (p.c - p.a));
        if (x > p.c) return 2 * (p.b - x) / (w * (p.b - p.c));
        return 2 / w;
      },
      cdf: (x, p) => {
        if (x <= p.a) return 0;
        if (x >= p.b) return 1;
        const w = p.b - p.a;
        return x <= p.c ? (x - p.a) ** 2 / (w * (p.c - p.a)) : 1 - (p.b - x) ** 2 / (w * (p.b - p.c));
      },
      sample: (r, p) => {
        const u = r.u(), w = p.b - p.a, fc = (p.c - p.a) / w;
        return u < fc ? p.a + Math.sqrt(u * w * (p.c - p.a)) : p.b - Math.sqrt((1 - u) * w * (p.b - p.c));
      },
      mean: (p) => (p.a + p.b + p.c) / 3,
      variance: (p) => (p.a ** 2 + p.b ** 2 + p.c ** 2 - p.a * p.b - p.a * p.c - p.b * p.c) / 18,
    },
    {
      id: "t", name: "Student's t", kind: "continuous",
      params: [["nu", "ν (degrees of freedom)", 1, 30, 1, 3]],
      x: [-6, 6], y: 0.45,
      story: "A normal whose spread was estimated from few data. Heavier tails than the normal.",
      gen: "Z / √(χ²ᵥ / ν): a standard normal over a scaled chi.",
      formula: "f(x) ∝ (1 + x²/ν)^(−(ν + 1)/2)",
      support: () => "all real numbers",
      pdf: (x, p) => Math.exp(lgamma((p.nu + 1) / 2) - lgamma(p.nu / 2) - 0.5 * Math.log(p.nu * Math.PI) - (p.nu + 1) / 2 * Math.log(1 + x * x / p.nu)),
      cdf: (x, p) => { const tail = 0.5 * betaI(p.nu / (p.nu + x * x), p.nu / 2, 0.5); return x >= 0 ? 1 - tail : tail; },
      sample: (r, p) => r.n() / Math.sqrt(2 * r.g(p.nu / 2) / p.nu),
      mean: (p) => (p.nu > 1 ? 0 : NaN), variance: (p) => (p.nu > 2 ? p.nu / (p.nu - 2) : p.nu > 1 ? Infinity : NaN),
    },
    {
      id: "cauchy", name: "Cauchy", kind: "continuous",
      params: [["x0", "x₀ (centre)", -3, 3, 0.1, 0], ["g", "γ (half-width)", 0.3, 3, 0.1, 1]],
      x: [-8, 8], y: 1.1,
      story: "So heavy-tailed it has no mean: averages of Cauchy draws never settle.",
      gen: "x₀ + γ·Z₁/Z₂, a ratio of two standard normals.",
      formula: "f(x) = 1 / (πγ(1 + ((x − x₀)/γ)²))",
      support: () => "all real numbers",
      pdf: (x, p) => 1 / (Math.PI * p.g * (1 + ((x - p.x0) / p.g) ** 2)),
      cdf: (x, p) => 0.5 + Math.atan((x - p.x0) / p.g) / Math.PI,
      sample: (r, p) => p.x0 + p.g * r.n() / r.n(),
      mean: () => NaN, variance: () => NaN,
    },
    {
      id: "F", name: "F", kind: "continuous",
      params: [["d1", "ν₁", 1, 30, 1, 5], ["d2", "ν₂", 1, 30, 1, 10]],
      x: [0, 6], y: 1.2,
      story: "Ratio of two variance estimates. The ANOVA test statistic.",
      gen: "(χ²ᵥ₁ / ν₁) / (χ²ᵥ₂ / ν₂).",
      formula: "f(x) ∝ x^(ν₁/2 − 1) (1 + ν₁x/ν₂)^(−(ν₁+ν₂)/2)",
      support: () => "x > 0",
      pdf: (x, p) => {
        if (x <= 0) return 0;
        const a = p.d1 / 2, b = p.d2 / 2;
        return Math.exp(a * Math.log(p.d1 / p.d2) + (a - 1) * Math.log(x) - (a + b) * Math.log(1 + p.d1 * x / p.d2) - (lgamma(a) + lgamma(b) - lgamma(a + b)));
      },
      cdf: (x, p) => (x > 0 ? betaI(p.d1 * x / (p.d1 * x + p.d2), p.d1 / 2, p.d2 / 2) : 0),
      sample: (r, p) => (2 * r.g(p.d1 / 2) / p.d1) / (2 * r.g(p.d2 / 2) / p.d2),
      mean: (p) => (p.d2 > 2 ? p.d2 / (p.d2 - 2) : NaN),
      variance: (p) => (p.d2 > 4 ? 2 * p.d2 * p.d2 * (p.d1 + p.d2 - 2) / (p.d1 * (p.d2 - 2) ** 2 * (p.d2 - 4)) : NaN),
    },
    {
      id: "laplace", name: "Laplace", kind: "continuous",
      params: [["mu", "μ (centre)", -3, 3, 0.1, 0], ["b", "b (scale)", 0.5, 3, 0.1, 1]],
      x: [-8, 8], y: 1.05,
      story: "Two exponentials back to back. A sharp peak with heavier tails than the normal.",
      gen: "μ + b·(E₁ − E₂), the difference of two Exponential(1) draws.",
      formula: "f(x) = e^(−|x − μ|/b) / (2b)",
      support: () => "all real numbers",
      pdf: (x, p) => Math.exp(-Math.abs(x - p.mu) / p.b) / (2 * p.b),
      cdf: (x, p) => (x < p.mu ? 0.5 * Math.exp((x - p.mu) / p.b) : 1 - 0.5 * Math.exp(-(x - p.mu) / p.b)),
      sample: (r, p) => p.mu + p.b * (r.e() - r.e()),
      mean: (p) => p.mu, variance: (p) => 2 * p.b * p.b,
    },
    {
      id: "logistic", name: "Logistic", kind: "continuous",
      params: [["mu", "μ (centre)", -3, 3, 0.1, 0], ["s", "s (scale)", 0.3, 3, 0.1, 1]],
      x: [-10, 10], y: 0.85,
      story: "Its CDF is the sigmoid. Behind logistic regression.",
      gen: "μ + s·ln(U / (1 − U)).",
      formula: "f(x) = e^(−z) / (s(1 + e^(−z))²),  z = (x − μ)/s",
      support: () => "all real numbers",
      pdf: (x, p) => { const z = (x - p.mu) / p.s, e = Math.exp(-Math.abs(z)); return e / (p.s * (1 + e) ** 2); },
      cdf: (x, p) => 1 / (1 + Math.exp(-(x - p.mu) / p.s)),
      sample: (r, p) => { const u = r.u(); return p.mu + p.s * Math.log(u / (1 - u)); },
      mean: (p) => p.mu, variance: (p) => p.s * p.s * Math.PI * Math.PI / 3,
    },
    {
      id: "loglogistic", name: "Log-logistic", kind: "continuous",
      params: [["a", "α (scale)", 0.5, 3, 0.1, 1], ["b", "β (shape)", 1, 8, 0.1, 3]],
      x: [0, 6], y: 2,
      story: "e raised to a logistic draw. Survival times that first rise, then fall.",
      gen: "α·(U / (1 − U))^(1/β).",
      formula: "f(x) = (β/α)(x/α)^(β−1) / (1 + (x/α)^β)²",
      support: () => "x > 0",
      pdf: (x, p) => { if (x <= 0) return 0; const t = Math.pow(x / p.a, p.b); return (p.b / p.a) * Math.pow(x / p.a, p.b - 1) / (1 + t) ** 2; },
      cdf: (x, p) => (x > 0 ? 1 / (1 + Math.pow(x / p.a, -p.b)) : 0),
      sample: (r, p) => { const u = r.u(); return p.a * Math.pow(u / (1 - u), 1 / p.b); },
      mean: (p) => (p.b > 1 ? p.a * (Math.PI / p.b) / Math.sin(Math.PI / p.b) : NaN),
      variance: (p) => {
        if (p.b <= 2) return NaN;
        const b = Math.PI / p.b;
        return p.a * p.a * (2 * b / Math.sin(2 * b) - b * b / Math.sin(b) ** 2);
      },
    },
    {
      id: "gumbel", name: "Gumbel", kind: "continuous",
      params: [["mu", "μ (location)", -3, 3, 0.1, 0], ["b", "β (scale)", 0.5, 3, 0.1, 1]],
      x: [-6, 12], y: 0.8,
      story: "The largest of many draws: the biggest flood of the century.",
      gen: "μ − β·ln(E) for an Exponential(1) draw E.",
      formula: "f(x) = (1/β) e^(−(z + e^(−z))),  z = (x − μ)/β",
      support: () => "all real numbers",
      pdf: (x, p) => { const z = (x - p.mu) / p.b; return Math.exp(-(z + Math.exp(-z))) / p.b; },
      cdf: (x, p) => Math.exp(-Math.exp(-(x - p.mu) / p.b)),
      sample: (r, p) => p.mu - p.b * Math.log(r.e()),
      mean: (p) => p.mu + p.b * EULER, variance: (p) => Math.PI * Math.PI * p.b * p.b / 6,
    },
    {
      id: "pareto", name: "Pareto", kind: "continuous",
      params: [["xm", "xₘ (minimum)", 0.5, 3, 0.1, 1], ["a", "α (tail)", 0.5, 5, 0.1, 2]],
      x: [0, 10], y: 4,
      story: "Power-law tail: a few huge values dominate (wealth, city sizes).",
      gen: "xₘ·e^(E/α) for an Exponential(1) draw E.",
      formula: "f(x) = α xₘ^α / x^(α+1)",
      support: (p) => `x ≥ ${p.xm}`,
      pdf: (x, p) => (x >= p.xm ? p.a * Math.pow(p.xm, p.a) / Math.pow(x, p.a + 1) : 0),
      cdf: (x, p) => (x >= p.xm ? 1 - Math.pow(p.xm / x, p.a) : 0),
      sample: (r, p) => p.xm * Math.exp(r.e() / p.a),
      mean: (p) => (p.a > 1 ? p.a * p.xm / (p.a - 1) : Infinity),
      variance: (p) => (p.a > 2 ? p.xm * p.xm * p.a / ((p.a - 1) ** 2 * (p.a - 2)) : Infinity),
    },
    {
      id: "invgamma", name: "Inverse gamma", kind: "continuous",
      params: [["a", "α (shape)", 0.5, 10, 0.1, 3], ["b", "β (scale)", 0.2, 5, 0.1, 1]],
      x: [0, 4], y: 3,
      story: "One over a gamma. The usual prior on an unknown variance.",
      gen: "β / G for G ~ Gamma(α, 1).",
      formula: "f(x) = βᵅ x^(−α−1) e^(−β/x) / Γ(α)",
      support: () => "x > 0",
      pdf: (x, p) => (x > 0 ? Math.exp(p.a * Math.log(p.b) - (p.a + 1) * Math.log(x) - p.b / x - lgamma(p.a)) : 0),
      cdf: (x, p) => (x > 0 ? 1 - gammaP(p.a, p.b / x) : 0),
      sample: (r, p) => p.b / r.g(p.a),
      mean: (p) => (p.a > 1 ? p.b / (p.a - 1) : Infinity),
      variance: (p) => (p.a > 2 ? p.b * p.b / ((p.a - 1) ** 2 * (p.a - 2)) : Infinity),
    },
    {
      id: "levy", name: "Lévy", kind: "continuous",
      params: [["c", "c (scale)", 0.5, 3, 0.1, 1]],
      x: [0, 10], y: 1,
      story: "First time a random walk hits a level. Infinite mean.",
      gen: "c / Z² for a standard normal Z.",
      formula: "f(x) = √(c/2π) e^(−c/2x) / x^(3/2)",
      support: () => "x > 0",
      pdf: (x, p) => (x > 0 ? Math.sqrt(p.c / (2 * Math.PI)) * Math.exp(-p.c / (2 * x)) / Math.pow(x, 1.5) : 0),
      cdf: (x, p) => (x > 0 ? 1 - erf(Math.sqrt(p.c / (2 * x))) : 0),
      sample: (r, p) => { const z = r.n(); return p.c / (z * z); },
      mean: () => Infinity, variance: () => Infinity,
    },
  ];

  // ---------- how they connect ----------
  // [from, to, label, type]. type: transform (do something to draws), special (set a parameter),
  // limit (a parameter runs off to infinity), mix (a parameter is itself random).
  const E = [
    ["uniform", "bernoulli", "1 if U < p", "transform"],
    ["uniform", "discUniform", "⌊nU⌋", "transform"],
    ["uniform", "exponential", "−ln U", "transform"],
    ["uniform", "normal", "Box–Muller", "transform"],
    ["uniform", "triangular", "U₁ + U₂", "transform"],
    ["uniform", "logistic", "ln(U/(1−U))", "transform"],
    ["bernoulli", "binomial", "sum of n", "transform"],
    ["bernoulli", "geometric", "trials to 1st yes", "transform"],
    ["bernoulli", "discUniform", "p = ½", "special"],
    ["binomial", "poisson", "n → ∞, np = λ", "limit"],
    ["binomial", "normal", "n → ∞", "limit"],
    ["hypergeom", "binomial", "N → ∞", "limit"],
    ["betabinom", "binomial", "α + β → ∞", "limit"],
    ["betabinom", "discUniform", "α = β = 1", "special"],
    ["beta", "betabinom", "p ~ Beta", "mix"],
    ["geometric", "negbin", "sum of r", "transform"],
    ["geometric", "exponential", "p → 0, rescaled", "limit"],
    ["negbin", "poisson", "r → ∞", "limit"],
    ["gamma", "negbin", "Poisson with λ ~ Gamma", "mix"],
    ["poisson", "normal", "λ → ∞", "limit"],
    ["poisson", "skellam", "X₁ − X₂", "transform"],
    ["exponential", "poisson", "count in time 1", "transform"],
    ["exponential", "erlang", "sum of k", "transform"],
    ["exponential", "laplace", "X₁ − X₂", "transform"],
    ["exponential", "weibull", "X^(1/k)", "transform"],
    ["exponential", "pareto", "xₘ eˣ", "transform"],
    ["exponential", "gumbel", "−ln X", "transform"],
    ["erlang", "gamma", "k any real", "special"],
    ["gamma", "exponential", "k = 1", "special"],
    ["gamma", "chi2", "k = ν/2, θ = 2", "special"],
    ["gamma", "beta", "X/(X + Y)", "transform"],
    ["gamma", "invgamma", "1/X", "transform"],
    ["gamma", "normal", "k → ∞", "limit"],
    ["chi2", "exponential", "ν = 2", "special"],
    ["chi2", "chi", "√X", "transform"],
    ["chi2", "t", "Z / √(X/ν)", "transform"],
    ["chi2", "F", "(X₁/ν₁)/(X₂/ν₂)", "transform"],
    ["normal", "chi2", "sum of ν squares", "transform"],
    ["normal", "lognormal", "eˣ", "transform"],
    ["normal", "halfnormal", "|X|", "transform"],
    ["normal", "t", "X / √(χ²/ν)", "transform"],
    ["normal", "cauchy", "X₁ / X₂", "transform"],
    ["normal", "levy", "1/X²", "transform"],
    ["chi", "halfnormal", "ν = 1", "special"],
    ["chi", "rayleigh", "ν = 2", "special"],
    ["chi", "maxwell", "ν = 3", "special"],
    ["weibull", "rayleigh", "k = 2", "special"],
    ["t", "normal", "ν → ∞", "limit"],
    ["t", "cauchy", "ν = 1", "special"],
    ["t", "F", "X²", "transform"],
    ["F", "beta", "ν₁X/(ν₂ + ν₁X)", "transform"],
    ["beta", "uniform", "α = β = 1", "special"],
    ["beta", "arcsine", "α = β = ½", "special"],
    ["beta", "powerfn", "β = 1", "special"],
    ["powerfn", "pareto", "1/X", "transform"],
    ["gumbel", "logistic", "X₁ − X₂", "transform"],
    ["logistic", "loglogistic", "eˣ", "transform"],
    ["invgamma", "levy", "α = ½", "special"],
  ].map(([from, to, label, type]) => ({ from, to, label, type }));

  const EDGE_TYPES = {
    transform: { name: "Transform the draws", dash: "" },
    special: { name: "Special case (set a parameter)", dash: "7 4" },
    limit: { name: "Limit (a parameter → ∞)", dash: "2 4" },
    mix: { name: "Mixture (a parameter is random)", dash: "10 3 2 3" },
  };

  const byId = Object.fromEntries(D.map((d) => [d.id, d]));
  const defaults = (d) => Object.fromEntries(d.params.map(([k, , , , , v]) => [k, v]));

  // P(X <= x). Discrete: add the pmf up from the lowest value with mass.
  function cdf(d, x, p) {
    if (d.kind === "continuous") return d.cdf(x, p);
    let s = 0;
    for (let k = d.lo(p); k <= Math.floor(x + 1e-9); k++) s += d.pdf(k, p);
    return Math.min(1, s);
  }

  root.DIST = { list: D, byId, edges: E, edgeTypes: EDGE_TYPES, defaults, cdf, rng };
})(typeof window !== "undefined" ? window : globalThis);
