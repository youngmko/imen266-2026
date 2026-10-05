"""Renewal, renewal-reward and regenerative processes (Chapter 7).

Used by the Ch.7 notebooks (CP9 renewal process / renewal function / limit
theorems / Wald, CP10 renewal reward / age and residual life / inspection
paradox / regenerative and alternating renewal processes) and HW4.
Dependency-light: numpy + scipy; matplotlib only for the two plot helpers.

Conventions
-----------
* A *sampler* draws iid inter-event times: ``sampler(rng, size) -> ndarray``.
  ``sampler_from(dist)`` turns a frozen scipy distribution into one, and
  ``samplers_same_mean(mean)`` gives the standard comparison set (deterministic,
  uniform, Erlang(4), exponential, hyperexponential, lognormal) with equal means.
* ``epochs`` are the renewal epochs S_1 < S_2 < ... (S_0 = 0 is not stored);
  ``N(t) = #{n : S_n <= t}``.
* ``renewal_function(dist, t)`` solves the renewal equation
  m(t) = F(t) + int_0^t m(t-x) dF(x) on a grid; closed forms are provided for
  uniform(0,1) (0 <= t <= 2), the hyperexponential (2025 final Q1) and Erlang(2).

Public API
----------
hyperexp(p, mu1, mu2)                  frozen-style hyperexponential distribution (pdf, cdf, mean, var, rvs)
sampler_from(dist)                     sampler from a frozen scipy distribution
samplers_same_mean(mean)               dict name -> sampler, all with the given mean
simulate_renewal(sampler, T, rng, overshoot=False)   epochs <= T (overshoot=True: also the first epoch > T)
count(epochs, t)                       N(t) for scalar or array t
renewal_function(dist, t, h=None)      m(t) on [0, t_max] by the discretised renewal equation (returns grid, m)
renewal_function_series(dist, t, h, nmax)  m(t) = sum_n F_n(t) by numerical convolution (returns grid, m, n_used)
renewal_function_mc(sampler, ts, n_paths, rng)   Monte-Carlo estimate of m(t) at the times ts
m_uniform01(t), m_hyperexp(t, p, mu1, mu2), m_erlang2(t, lam)   closed-form renewal functions
rate_convergence_plot(samplers, T, rng, ax)   N(t)/t paths for several F (same mean), with the 1/mu line
renewal_reward(sampler, reward_fn, T, rng)    cumulative reward path; returns epochs, rewards, R(t)/t at T
age_residual(epochs, t)                (A(t), Y(t), C(t)) for scalar/array t (epochs must extend beyond t)
age_residual_samples(sampler, T, rng, n_times)   (A, Y, C) at n_times uniform inspection times in [T/2, T]
equilibrium_cdf(dist, x), length_biased_pdf(dist, x)   F_e(x) = int_0^x (1-F)/E[X],  x f(x)/E[X]
inspection_paradox_demo(dist, T, rng)  dict: E[X], E[X^2]/E[X], mean spread seen, mean age/residual, E[X^2]/(2E[X])
alternating_renewal(on_sampler, off_sampler, T, rng)   (switch_times, states, on_fraction) on [0, T]
semi_markov(P, means)                  long-run time fractions pi_i mu_i / sum_j pi_j mu_j of a semi-Markov process
"""
from __future__ import annotations
import math
import numpy as np
from scipy import stats

__all__ = ["hyperexp", "sampler_from", "samplers_same_mean", "simulate_renewal", "count",
           "renewal_function", "renewal_function_series", "renewal_function_mc",
           "m_uniform01", "m_hyperexp", "m_erlang2", "rate_convergence_plot",
           "renewal_reward", "age_residual", "age_residual_samples",
           "equilibrium_cdf", "length_biased_pdf", "inspection_paradox_demo",
           "alternating_renewal", "semi_markov"]


# ----------------------------------------------------------------------------
# a hyperexponential "frozen distribution" (scipy has none): the 2025 final-exam F
class hyperexp:
    """Mixture of two exponentials, F(t) = p(1 - e^{-mu1 t}) + (1-p)(1 - e^{-mu2 t}).

    Mimics the frozen-scipy interface used in this module: pdf, cdf, sf, mean(),
    var(), moment(2), rvs(size, random_state).
    """

    def __init__(self, p: float, mu1: float, mu2: float):
        self.p, self.mu1, self.mu2 = float(p), float(mu1), float(mu2)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.p * self.mu1 * np.exp(-self.mu1 * x)
                        + (1 - self.p) * self.mu2 * np.exp(-self.mu2 * x), 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.p * (1 - np.exp(-self.mu1 * x))
                        + (1 - self.p) * (1 - np.exp(-self.mu2 * x)), 0.0)

    def sf(self, x):
        return 1.0 - self.cdf(x)

    def mean(self):
        return self.p / self.mu1 + (1 - self.p) / self.mu2

    def moment(self, k: int):
        return self.p * math.factorial(k) / self.mu1 ** k + (1 - self.p) * math.factorial(k) / self.mu2 ** k

    def var(self):
        return self.moment(2) - self.mean() ** 2

    def rvs(self, size=1, random_state=None):
        rng = np.random.default_rng(random_state)
        u = rng.random(size)
        return np.where(u < self.p, rng.exponential(1 / self.mu1, size), rng.exponential(1 / self.mu2, size))


# ----------------------------------------------------------------------------
# samplers
def sampler_from(dist):
    """Sampler ``f(rng, size)`` drawing iid values from a frozen scipy distribution."""
    def f(rng, size):
        return np.asarray(dist.rvs(size=size, random_state=rng), dtype=float)
    f.dist = dist
    return f


def samplers_same_mean(mean: float = 1.0, cv_lognormal: float = 1.5):
    """Standard comparison set: six inter-event distributions with the same mean.

    deterministic (cv 0), uniform(0, 2 mean) (cv 0.58), Erlang(4) (cv 0.5),
    exponential (cv 1), hyperexponential (cv 2.35), lognormal (cv 1.5).
    Returns an ordered dict name -> sampler; each sampler has a ``.dist``
    attribute (a frozen scipy distribution, or ``hyperexp``) except the
    deterministic one (``.dist = None``).
    """
    m = float(mean)
    out = {}
    def det(rng, size):
        return np.full(size, m)
    det.dist = None
    out["deterministic"] = det
    out["uniform"] = sampler_from(stats.uniform(0, 2 * m))
    out["Erlang(4)"] = sampler_from(stats.gamma(4, scale=m / 4))
    out["exponential"] = sampler_from(stats.expon(scale=m))
    # hyperexponential: with prob p an exp(mu1), else exp(mu2); p=0.9, means 0.5m and 5.5m -> mean m, cv 2.35
    out["hyperexponential"] = sampler_from(hyperexp(0.9, 1 / (0.5 * m), 1 / (5.5 * m)))
    s2 = math.log(1 + cv_lognormal ** 2)
    out[f"lognormal(cv={cv_lognormal:g})"] = sampler_from(stats.lognorm(s=math.sqrt(s2), scale=m * math.exp(-s2 / 2)))
    return out


# ----------------------------------------------------------------------------
# simulation of a renewal process
def simulate_renewal(sampler, T: float, rng=None, overshoot: bool = False) -> np.ndarray:
    """Renewal epochs S_1 < S_2 < ... <= T (cumulative sums of iid gaps).

    overshoot=True also returns the first epoch beyond T, which age/residual
    computations at times near T need.
    """
    rng = np.random.default_rng(rng)
    T = float(T)
    gaps = sampler(rng, 64)
    mean = max(float(np.mean(gaps)), 1e-12)
    n_guess = int(T / mean + 10 * math.sqrt(T / mean + 1)) + 16
    S = np.cumsum(sampler(rng, n_guess))
    while S[-1] <= T:                                   # (rare) extend
        S = np.concatenate([S, S[-1] + np.cumsum(sampler(rng, n_guess))])
    k = int(np.searchsorted(S, T, side="right"))        # number of epochs <= T
    return S[:k + 1] if overshoot else S[:k]


def count(epochs, t):
    """N(t) = number of epochs <= t, for scalar or array t."""
    epochs = np.asarray(epochs, dtype=float)
    return np.searchsorted(epochs, np.asarray(t, dtype=float), side="right")


# ----------------------------------------------------------------------------
# the renewal function
def renewal_function(dist, t: float, h: float | None = None):
    """Solve m(t) = F(t) + int_0^t m(t-x) dF(x) on the grid 0, h, 2h, ..., t.

    Riemann-Stieltjes discretisation with the trapezoid rule in m (second-order
    accurate): with dF_j = F(jh) - F((j-1)h),
        m_k = F(kh) + sum_{j=1}^{k} dF_j (m_{k-j} + m_{k-j+1}) / 2,
    solved for m_k (the j = 1 term contains m_k itself). Returns (grid, m).
    ``dist`` is a frozen scipy distribution (needs .cdf and .mean()).
    """
    t = float(t)
    if h is None:
        h = min(dist.mean() / 200.0, t / 400.0)
    K = int(round(t / h))
    grid = np.arange(K + 1) * h
    F = dist.cdf(grid)
    dF = np.diff(F)                                     # dF[j-1] = F(jh) - F((j-1)h), j = 1..K
    m = np.zeros(K + 1)
    denom = 1.0 - dF[0] / 2.0 if K > 0 else 1.0
    for k in range(1, K + 1):
        # sum_{j=1}^{k} dF_j (m_{k-j} + m_{k-j+1})/2, with m_k unknown in the j=1 term
        a = m[k - 1::-1][:k]                            # m_{k-j}   for j = 1..k:  m_{k-1}, ..., m_0
        b = m[k:0:-1]                                   # m_{k-j+1} for j = 1..k:  m_k (unknown, still 0), ..., m_1
        s = float(np.dot(dF[:k], (a + b) / 2.0))
        m[k] = (F[k] + s) / denom
    return grid, m


def renewal_function_series(dist, t: float, h: float | None = None, nmax: int = 400, eps: float = 1e-10):
    """m(t) = sum_{n>=1} F_n(t) by numerical convolution of the density on a grid.

    Returns (grid, m, n_used). Stops when F_n(t_max) < eps or n = nmax.
    Second-order accurate for smooth densities; a density with an atom (e.g.
    deterministic gaps) is not supported.
    """
    t = float(t)
    if h is None:
        h = min(dist.mean() / 200.0, t / 400.0)
    K = int(round(t / h))
    grid = np.arange(K + 1) * h
    f = dist.pdf(grid) * h                              # probability mass per cell (midpoint-ish)
    f[0] = dist.cdf(h / 2)                              # mass of the first half-cell
    fn = f.copy()
    m = np.cumsum(fn)                                   # F_1
    n = 1
    while n < nmax:
        fn = np.convolve(fn, f)[:K + 1]
        Fn = np.cumsum(fn)
        m += Fn
        n += 1
        if Fn[-1] < eps:
            break
    return grid, m, n


def renewal_function_mc(sampler, ts, n_paths: int = 2000, rng=None) -> np.ndarray:
    """Monte-Carlo estimate of m(t) = E[N(t)] at the times ``ts`` (array)."""
    rng = np.random.default_rng(rng)
    ts = np.asarray(ts, dtype=float)
    T = float(ts.max())
    acc = np.zeros_like(ts)
    for _ in range(n_paths):
        S = simulate_renewal(sampler, T, rng)
        acc += count(S, ts)
    return acc / n_paths


def m_uniform01(t):
    """Renewal function of uniform(0,1) gaps for 0 <= t <= 2 (closed form; nan beyond)."""
    t = np.asarray(t, dtype=float)
    out = np.where(t <= 1.0, np.exp(t) - 1.0,
                   np.exp(t) - 1.0 + (1.0 - t) * np.exp(t - 1.0))
    return np.where(t <= 2.0, out, np.nan)


def m_hyperexp(t, p: float, mu1: float, mu2: float):
    """Renewal function of F(t) = p(1-e^{-mu1 t}) + (1-p)(1-e^{-mu2 t}) (2025 final Q1).

    m(t) = mu1 mu2 t / c + p(1-p)(mu1-mu2)^2 / c^2 (1 - e^{-c t}),  c = (1-p) mu1 + p mu2.
    """
    t = np.asarray(t, dtype=float)
    c = (1 - p) * mu1 + p * mu2
    return mu1 * mu2 * t / c + p * (1 - p) * (mu1 - mu2) ** 2 / c ** 2 * (1 - np.exp(-c * t))


def m_erlang2(t, lam: float):
    """Renewal function of Erlang(2, lam) gaps: m(t) = lam t / 2 - (1 - e^{-2 lam t}) / 4."""
    t = np.asarray(t, dtype=float)
    return lam * t / 2.0 - (1.0 - np.exp(-2.0 * lam * t)) / 4.0


# ----------------------------------------------------------------------------
# limit theorems: rate convergence
def rate_convergence_plot(samplers: dict, T: float = 2000.0, rng=None, ax=None, mean: float | None = None):
    """Plot N(t)/t against t (log axis) for each sampler, with the 1/mean line.

    ``samplers``: name -> sampler (all with the same mean). Returns the axis.
    """
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(rng)
    ax = ax or plt.gca()
    ts = np.logspace(math.log10(max(T / 1000.0, 1e-3)), math.log10(T), 300)
    mu = mean
    for name, sampler in samplers.items():
        S = simulate_renewal(sampler, T, rng)
        if mu is None:
            mu = float(np.mean(np.diff(np.concatenate([[0.0], S]))))
        ax.plot(ts, count(S, ts) / ts, lw=1.4, label=name)
    if mu:
        ax.axhline(1.0 / mu, color="k", ls="--", lw=1, label=f"1/mu = {1/mu:.3g}")
    ax.set_xscale("log"); ax.set_xlabel("t"); ax.set_ylabel("N(t)/t"); ax.legend(fontsize=8)
    return ax


# ----------------------------------------------------------------------------
# renewal reward
def renewal_reward(sampler, reward_fn, T: float, rng=None):
    """Simulate a renewal reward process on [0, T].

    reward_fn(X, rng) -> array of rewards, one per cycle (may depend on the cycle
    length X; the pairs (X_n, R_n) are then iid). Returns a dict with the cycle
    lengths ``X``, rewards ``R``, epochs ``S``, the cumulative reward ``R_T`` of the
    completed cycles by T, the rate ``R_T / T`` and the ratio of means ``E[R]/E[X]``
    estimated from the same cycles.
    """
    rng = np.random.default_rng(rng)
    S = simulate_renewal(sampler, T, rng)
    X = np.diff(np.concatenate([[0.0], S]))
    R = np.asarray(reward_fn(X, rng), dtype=float)
    R_T = float(R.sum())
    return dict(X=X, R=R, S=S, R_T=R_T, rate=R_T / T,
                ratio_of_means=float(R.mean() / X.mean()) if len(X) else float("nan"),
                mean_of_ratios=float(np.mean(R / X)) if len(X) else float("nan"))


# ----------------------------------------------------------------------------
# age, residual life, spread
def age_residual(epochs, t):
    """(A(t), Y(t), C(t)) = (age, remaining time, spread) at scalar/array t.

    ``epochs`` must contain the first epoch beyond max(t) (use
    simulate_renewal(..., overshoot=True)); otherwise nan is returned there.
    """
    S = np.asarray(epochs, dtype=float)
    t = np.asarray(t, dtype=float)
    n = np.searchsorted(S, t, side="right")             # N(t)
    last = np.where(n > 0, S[np.clip(n - 1, 0, len(S) - 1)], 0.0)
    ok = n < len(S)
    nxt = np.where(ok, S[np.clip(n, 0, len(S) - 1)], np.nan)
    A = t - last
    Y = nxt - t
    return A, Y, A + Y


def age_residual_samples(sampler, T: float, rng=None, n_times: int = 20000):
    """Age, residual life and spread at n_times uniform inspection times in [T/2, T]
    of ONE long renewal path (a time average, as in the theorem)."""
    rng = np.random.default_rng(rng)
    S = simulate_renewal(sampler, T, rng, overshoot=True)
    ts = rng.uniform(T / 2.0, T, n_times)
    return age_residual(S, ts)


def equilibrium_cdf(dist, x):
    """F_e(x) = int_0^x (1 - F(u)) du / E[X]: the limiting distribution of the age and of the residual life."""
    x = np.asarray(x, dtype=float)
    mu = float(dist.mean())
    # int_0^x (1-F) = x - int_0^x F = x - (x F(x) - E[X 1{X<=x}])  (integration by parts) = E[min(X, x)]
    # use E[min(X,x)] = int_0^x (1-F(u)) du computed with the partial expectation via numerical quadrature
    from scipy import integrate
    out = np.empty_like(x)
    for i, xi in np.ndenumerate(x):
        out[i] = integrate.quad(lambda u: 1.0 - dist.cdf(u), 0.0, float(xi), limit=200)[0] / mu
    return out


def length_biased_pdf(dist, x):
    """x f(x) / E[X]: the density of the inter-event interval that covers a random time."""
    x = np.asarray(x, dtype=float)
    return x * dist.pdf(x) / float(dist.mean())


def inspection_paradox_demo(dist, T: float = 20000.0, rng=None, n_times: int = 20000) -> dict:
    """Numbers behind the inspection paradox for a frozen distribution ``dist``.

    Returns E[X], E[X^2]/E[X] (theory), the simulated mean spread C seen at random
    times, the simulated mean age and residual, and E[X^2]/(2E[X]) (theory).
    """
    rng = np.random.default_rng(rng)
    A, Y, C = age_residual_samples(sampler_from(dist), T, rng, n_times)
    mu, m2 = float(dist.mean()), float(dist.var() + dist.mean() ** 2)
    return dict(EX=mu, spread_theory=m2 / mu, spread_sim=float(np.nanmean(C)),
                age_sim=float(np.nanmean(A)), residual_sim=float(np.nanmean(Y)),
                age_theory=m2 / (2 * mu))


# ----------------------------------------------------------------------------
# alternating renewal and semi-Markov
def alternating_renewal(on_sampler, off_sampler, T: float, rng=None, start_on: bool = True):
    """Simulate an on/off process on [0, T].

    Returns (times, states, on_fraction): ``times`` are the switch epochs starting
    at 0 (the last entry is <= T), ``states[k]`` is the state (1 = on, 0 = off)
    holding on [times[k], times[k+1]), and ``on_fraction`` is the fraction of
    [0, T] spent on. The samplers draw the on and off durations (``f(rng, size)``);
    (Z_n, Y_n) pairs are independent across cycles here.
    """
    rng = np.random.default_rng(rng)
    T = float(T)
    z = on_sampler(rng, 64); y = off_sampler(rng, 64)
    cyc = max(float(np.mean(z) + np.mean(y)), 1e-12)
    n = int(T / cyc + 10 * math.sqrt(T / cyc + 1)) + 16
    Z = on_sampler(rng, n); Y = off_sampler(rng, n)
    durations = np.empty(2 * n); durations[0::2] = Z if start_on else Y; durations[1::2] = Y if start_on else Z
    times = np.concatenate([[0.0], np.cumsum(durations)])
    while times[-1] <= T:
        Z = on_sampler(rng, n); Y = off_sampler(rng, n)
        d = np.empty(2 * n); d[0::2] = Z if start_on else Y; d[1::2] = Y if start_on else Z
        times = np.concatenate([times, times[-1] + np.cumsum(d)])
    k = int(np.searchsorted(times, T, side="right"))     # switches at times[0..k-1] are <= T
    times = times[:k]
    states = np.array([(1 if start_on else 0) if i % 2 == 0 else (0 if start_on else 1) for i in range(k)])
    ends = np.concatenate([times[1:], [T]])
    on_time = float(np.sum((ends - times) * states))
    return times, states, on_time / T


def semi_markov(P, means):
    """Long-run fraction of time in each state of a semi-Markov process.

    P: transition matrix of the embedded chain; means[i]: mean sojourn in state i.
    Fractions are pi_i mu_i / sum_j pi_j mu_j with pi the stationary distribution of P
    (regenerative argument: cycles between visits to a fixed state).
    """
    from .dtmc import DTMC
    pi = DTMC(np.asarray(P, dtype=float)).stationary()
    w = pi * np.asarray(means, dtype=float)
    return w / w.sum()
