"""Queueing models and Jackson networks (Chapter 8).

Used by the Ch.8 notebooks (CP11 Markovian queues, CP12 finite capacity /
M/G/1 / variations, CP13 Jackson networks) and HW5. Dependency-light: numpy +
scipy; matplotlib is not required (the notebooks draw with their own setup).

Conventions
-----------
* Rates: ``lam`` = arrival rate, ``mu`` = service rate *per server*,
  ``s`` = number of servers (``np.inf`` allowed), ``K`` = capacity of the
  *system* (in service + waiting). ``rho = lam / (s mu)`` is the traffic
  intensity, ``a = lam / mu`` the offered load.
* Every analytic function returns a plain ``dict`` with (at least) the keys
  ``rho, p0, L, Lq, W, Wq`` and, for finite-capacity models, ``p`` (the whole
  stationary vector), ``pK`` (loss probability) and ``lam_eff = lam (1 - pK)``;
  infinite-capacity models carry ``pn`` (a function n -> p_n). An unstable
  infinite-capacity queue returns ``stable=False`` and infinite L, W.
* Samplers for the simulators are ``sampler(rng, size) -> ndarray`` (as in
  ``imen266.renewal``); a plain number is read as the rate of an exponential
  law (Poisson arrivals / exponential service).

Public API
----------
mm1(lam, mu)                         M/M/1: rho, p_n, L, Lq, W, Wq; sojourn ~ exp(mu - lam)
mms(lam, mu, s)                      M/M/s: p0, Lq (Erlang C), L, W, Wq, P(wait); s = inf -> mminf
mminf(lam, mu)                       M/M/inf: Poisson(lam/mu), L = lam/mu, W = 1/mu
mm1k(lam, mu, K)                     M/M/1/K: p, pK, lam_eff, L, Lq, W, Wq
mmsk(lam, mu, s, K)                  M/M/s/K (s <= K)
mmkk(lam, mu, K)                     Erlang loss system M/M/K/K (= M/G/K/K)
erlang_b(a, K), erlang_c(a, s)       loss / waiting probabilities (stable recursions)
birth_death_stationary(births, deaths)   p_n = p_0 prod births/deaths for a finite chain
truncate(p, S)                       renormalise a reversible chain's weights on a subset S
mm1_sojourn_tail(lam, mu, x)         P(W > x) = exp(-(mu - lam) x); mm1_required_rate(lam, x, eps)
mg1_pk(lam, mean, var)               Pollaczek-Khinchine: rho, Lq, Wq, W, L
bulk_mm1(lam, q, mu)                 M^[X]/M/1 with P(X = i) = q[i-1]: E[X], E[X^2], rho, L, W, Lq, Wq
gg1_approx(lam, mu, ca2, cs2, method) G/G/1 approximations ("kingman", "h1", "h2", "h3" = PLMS handout)
ggm_approx(lam, mu, m, ca2, cs2)     G/G/m approximation of the PLMS handout (alpha_m)
reneging_balking_rates(lam, mu, theta, K)   birth/death rates of the 2025 final-exam queue
jackson_solve(theta, P, mu, s)       traffic equations + product form: lam, rho, per-node L/W, network L/W
optimal_split(lam, p, mu_total)      square-root rule for splitting capacity between parallel M/M/1 queues
pooling_compare(lam, mu)             M/M/2 with PP(2 lam) vs two M/M/1 with PP(lam) each
braess(lam, mu)                      Braess' paradox: W before / after the new link, equilibrium flows
SimQueue(...)                        event-driven simulator: s servers, capacity K, patience (reneging), balking
simulate_queue(...)                  one-call wrapper around SimQueue(...).run(...)
sim_network(theta, P, mu, s, T, rng) event-driven open network (Jackson or not): per-node L, W, customer sojourn
little_check(result)                 L vs lam_eff * W from a simulation result
pasta_demo(lam, mu, T, rng, arrivals)   arrival-average vs time-average distribution of N(t)
"""
from __future__ import annotations
import heapq
import math
from collections import deque
import numpy as np

__all__ = ["mm1", "mms", "mminf", "mm1k", "mmsk", "mmkk", "erlang_b", "erlang_c",
           "birth_death_stationary", "truncate", "mm1_sojourn_tail", "mm1_required_rate",
           "mg1_pk", "bulk_mm1", "gg1_approx", "ggm_approx", "reneging_balking_rates",
           "jackson_solve", "optimal_split", "pooling_compare", "braess",
           "SimQueue", "simulate_queue", "sim_network", "little_check", "pasta_demo"]


# ----------------------------------------------------------------------------
# helpers
def _unstable(rho, **extra):
    d = dict(rho=rho, stable=False, p0=0.0, L=np.inf, Lq=np.inf, W=np.inf, Wq=np.inf)
    d.update(extra)
    return d


def _as_sampler(x):
    """A number -> exponential sampler with that rate; a callable is returned as is."""
    if callable(x):
        return x
    rate = float(x)
    if rate <= 0:
        raise ValueError("a rate must be positive")
    return lambda rng, n: rng.exponential(1.0 / rate, n)


# ----------------------------------------------------------------------------
# infinite-capacity Markovian queues
def mm1(lam: float, mu: float) -> dict:
    """M/M/1: p_n = (1 - rho) rho^n, L = rho/(1-rho), W = 1/(mu - lam), sojourn ~ exp(mu - lam)."""
    rho = lam / mu
    if rho >= 1:
        return _unstable(rho)
    L = rho / (1 - rho)
    return dict(rho=rho, stable=True, p0=1 - rho, L=L, Lq=L - rho, W=1 / (mu - lam),
                Wq=rho / (mu - lam), pn=lambda n: (1 - rho) * rho ** np.asarray(n, dtype=float),
                Pwait=rho)


def mminf(lam: float, mu: float) -> dict:
    """M/M/inf: number in system ~ Poisson(lam/mu); L = lam/mu, W = 1/mu, no queue."""
    from scipy.stats import poisson
    a = lam / mu
    return dict(rho=0.0, stable=True, p0=math.exp(-a), L=a, Lq=0.0, W=1 / mu, Wq=0.0,
                pn=lambda n: poisson.pmf(np.asarray(n), a), Pwait=0.0, a=a)


def erlang_b(a: float, K: int) -> float:
    """Erlang B: P(all K servers busy) in M/M/K/K (also M/G/K/K) with offered load a = lam/mu."""
    B = 1.0
    for k in range(1, int(K) + 1):
        B = a * B / (k + a * B)
    return float(B)


def erlang_c(a: float, s: int) -> float:
    """Erlang C: P(an arrival waits) in M/M/s with offered load a = lam/mu (needs a < s)."""
    s = int(s)
    rho = a / s
    if rho >= 1:
        return 1.0
    B = erlang_b(a, s)
    return float(B / (1 - rho * (1 - B)))


def mms(lam: float, mu: float, s) -> dict:
    """M/M/s (s servers, each rate mu). Returns p0, Lq, L, W, Wq, Pwait (Erlang C), pn(n)."""
    if s is None or (isinstance(s, float) and math.isinf(s)):
        return mminf(lam, mu)
    s = int(s)
    a = lam / mu
    rho = a / s
    if rho >= 1:
        return _unstable(rho, a=a)
    terms = [a ** k / math.factorial(k) for k in range(s)]
    tail = a ** s / math.factorial(s) / (1 - rho)
    p0 = 1.0 / (sum(terms) + tail)
    Pwait = tail * p0                                    # Erlang C
    Lq = Pwait * rho / (1 - rho)
    L = Lq + a
    weights = np.array(terms + [a ** s / math.factorial(s)])

    def pn(n):
        n = np.asarray(n, dtype=int)
        out = np.empty(n.shape, dtype=float)
        flat = n.ravel()
        res = np.array([p0 * (weights[k] if k < s else weights[s] * rho ** (k - s)) for k in flat])
        out.ravel()[:] = res
        return out if out.shape else float(out)

    return dict(rho=rho, stable=True, a=a, p0=p0, L=L, Lq=Lq, W=L / lam, Wq=Lq / lam,
                Pwait=Pwait, pn=pn)


# ----------------------------------------------------------------------------
# finite-capacity Markovian queues
def birth_death_stationary(births, deaths) -> np.ndarray:
    """Stationary distribution of a finite birth-death chain on 0..n:
    births[k] = rate k -> k+1, deaths[k] = rate k+1 -> k (k = 0..n-1); p_k = p_0 prod births/deaths."""
    births = np.asarray(births, dtype=float)
    deaths = np.asarray(deaths, dtype=float)
    if births.shape != deaths.shape:
        raise ValueError("births and deaths must have the same length (n)")
    w = np.concatenate([[1.0], np.cumprod(births / deaths)])
    return w / w.sum()


def truncate(p, S) -> np.ndarray:
    """Truncation theorem: the stationary distribution of a reversible chain restricted to the
    states S (indices into p) is p[S] / p[S].sum()."""
    p = np.asarray(p, dtype=float)
    S = np.asarray(S, dtype=int)
    q = p[S]
    return q / q.sum()


def _finite_summary(lam, mu, s, p):
    K = len(p) - 1
    n = np.arange(K + 1)
    L = float((n * p).sum())
    Lq = float((np.maximum(n - s, 0) * p).sum())
    pK = float(p[-1])
    lam_eff = lam * (1 - pK)
    W = L / lam_eff
    Wq = Lq / lam_eff
    return dict(p=p, p0=float(p[0]), pK=pK, lam_eff=lam_eff, L=L, Lq=Lq, W=W, Wq=Wq,
                busy=L - Lq, stable=True)


def mm1k(lam: float, mu: float, K: int) -> dict:
    """M/M/1/K: p_i = (1-a) a^i / (1 - a^{K+1}) (a = lam/mu; uniform if a = 1), effective rate lam(1-pK)."""
    K = int(K)
    a = lam / mu
    p = birth_death_stationary([lam] * K, [mu] * K)
    d = _finite_summary(lam, mu, 1, p)
    d.update(rho=a, a=a, K=K)
    return d


def mmsk(lam: float, mu: float, s: int, K: int) -> dict:
    """M/M/s/K (s <= K): the M/M/s cut equations truncated at K; Little with lam(1-pK)."""
    s, K = int(s), int(K)
    if s > K:
        raise ValueError("need s <= K")
    deaths = [min(k + 1, s) * mu for k in range(K)]
    p = birth_death_stationary([lam] * K, deaths)
    d = _finite_summary(lam, mu, s, p)
    d.update(rho=lam / (s * mu), a=lam / mu, s=s, K=K)
    return d


def mmkk(lam: float, mu: float, K: int) -> dict:
    """Erlang loss system M/M/K/K: p_i = (a^i/i!) / sum_j a^j/j!; pK = Erlang B (same for M/G/K/K)."""
    d = mmsk(lam, mu, K, K)
    d["erlang_b"] = erlang_b(lam / mu, K)
    return d


def mm1_sojourn_tail(lam: float, mu: float, x) -> np.ndarray:
    """P(sojourn time > x) in a stable M/M/1: exp(-(mu - lam) x)."""
    return np.exp(-(mu - lam) * np.asarray(x, dtype=float))


def mm1_required_rate(lam: float, x: float, eps: float) -> float:
    """Smallest service rate with P(sojourn > x) <= eps in M/M/1 (2024 final Q2): lam + ln(1/eps)/x."""
    return lam + math.log(1.0 / eps) / x


# ----------------------------------------------------------------------------
# non-Markovian single-server results and approximations
def mg1_pk(lam: float, mean: float, var: float) -> dict:
    """Pollaczek-Khinchine: service time with E[S] = mean, Var[S] = var;
    Lq = (lam^2 var + rho^2) / (2 (1 - rho)), Wq = Lq/lam, W = Wq + mean, L = Lq + rho."""
    rho = lam * mean
    if rho >= 1:
        return _unstable(rho)
    ES2 = var + mean ** 2
    Lq = (lam ** 2 * var + rho ** 2) / (2 * (1 - rho))
    Wq = Lq / lam
    return dict(rho=rho, stable=True, p0=1 - rho, Lq=Lq, Wq=Wq, W=Wq + mean, L=Lq + rho,
                ES2=ES2, mean_residual=lam * ES2 / 2)


def bulk_mm1(lam: float, q, mu: float) -> dict:
    """M^[X]/M/1: batches PP(lam), batch size X with P(X = i) = q[i-1] (i = 1..n), individual exp(mu)
    service. L = (rho + (lam/mu) E[X^2]) / (2 (1 - rho)), rho = lam E[X]/mu; Little with rate lam E[X]."""
    q = np.asarray(q, dtype=float)
    if abs(q.sum() - 1) > 1e-9 or (q < 0).any():
        raise ValueError("q must be a probability vector over batch sizes 1..n")
    sizes = np.arange(1, len(q) + 1)
    EX = float((sizes * q).sum())
    EX2 = float((sizes ** 2 * q).sum())
    rho = lam * EX / mu
    if rho >= 1:
        return _unstable(rho, EX=EX, EX2=EX2)
    L = (rho + (lam / mu) * EX2) / (2 * (1 - rho))
    lam_c = lam * EX                                     # customer arrival rate
    return dict(rho=rho, stable=True, EX=EX, EX2=EX2, lam_customers=lam_c, L=L, W=L / lam_c,
                Lq=L - rho, Wq=(L - rho) / lam_c, p0=1 - rho)


def gg1_approx(lam: float, mu: float, ca2: float, cs2: float, method: str = "kingman") -> dict:
    """G/G/1 approximations with squared coefficients of variation ca2 (inter-arrivals), cs2 (service).

    method = "kingman": Wq ~ (rho/(1-rho)) ((ca2+cs2)/2) (1/mu)  (heavy-traffic approximation)
             "h1", "h2", "h3": the three approximations for L of the PLMS handout
             (h3 = Kingman's Lq + (1-ca2) ca2 rho/2 + rho).
    Returns rho, L, Lq, W, Wq (L and W of the "h" methods via Little, L = lam W).
    """
    rho = lam / mu
    if rho >= 1:
        return _unstable(rho)
    if method == "kingman":
        Wq = rho / (1 - rho) * (ca2 + cs2) / 2 / mu
        Lq = lam * Wq
        L = Lq + rho
    elif method == "h1":
        L = (rho ** 2 * (1 + cs2) / (1 + rho ** 2 * cs2)) * ((ca2 + rho ** 2 * cs2) / (2 * (1 - rho))) + rho
        Lq = L - rho
        Wq = Lq / lam
    elif method == "h2":
        L = (rho * (1 + cs2) / (2 - rho + rho * cs2)) * ((rho * (2 - rho) * ca2 + rho ** 2 * cs2) / (2 * (1 - rho))) + rho
        Lq = L - rho
        Wq = Lq / lam
    elif method == "h3":
        L = rho ** 2 * (ca2 + cs2) / (2 * (1 - rho)) + (1 - ca2) * ca2 * rho / 2 + rho
        Lq = L - rho
        Wq = Lq / lam
    else:
        raise ValueError("method must be 'kingman', 'h1', 'h2' or 'h3'")
    return dict(rho=rho, stable=True, L=L, Lq=Lq, W=L / lam, Wq=Wq, method=method)


def ggm_approx(lam: float, mu: float, m: int, ca2: float, cs2: float) -> dict:
    """G/G/m approximation of the PLMS handout: rho = lam/(m mu),
    alpha_m = (rho^m + rho)/2 if rho > 0.7 else rho^((m+1)/2),
    Wq ~ (alpha_m/mu) (1/(1-rho)) ((ca2 + cs2)/(2m)); then Lq = lam Wq, W = Wq + 1/mu, L = lam W."""
    m = int(m)
    rho = lam / (m * mu)
    if rho >= 1:
        return _unstable(rho)
    alpha = (rho ** m + rho) / 2 if rho > 0.7 else rho ** ((m + 1) / 2)
    Wq = alpha / mu / (1 - rho) * (ca2 + cs2) / (2 * m)
    W = Wq + 1 / mu
    return dict(rho=rho, stable=True, alpha_m=alpha, Wq=Wq, Lq=lam * Wq, W=W, L=lam * W)


def reneging_balking_rates(lam: float, mu: float, theta: float, K: int):
    """Birth and death rates of the single-server queue of the 2025 final (and homework08 Q3):
    an arrival that finds i customers balks w.p. i/(i+1) (joins w.p. 1/(i+1)); each waiting
    customer reneges after an exp(theta) patience; service exp(mu). Truncated at K states
    (K large approximates the infinite chain). Returns (births, deaths) for states 0..K."""
    K = int(K)
    births = np.array([lam / (i + 1) for i in range(K)])
    deaths = np.array([mu + (i + 1 - 1) * theta for i in range(K)])   # from state i+1: service + i waiting
    return births, deaths


# ----------------------------------------------------------------------------
# Jackson networks and routing
def jackson_solve(theta, P, mu, s=None) -> dict:
    """Open Jackson network: external Poisson rates theta, routing matrix P (row i: p_ij, exit
    prob 1 - sum_j p_ij), service rates mu, servers s (default 1; np.inf for self-service).

    Returns lam (effective arrival rates = theta (I - P)^{-1}), rho, per-node dicts in ``nodes``
    (each an mms() result with arrival rate lam_i), the vectors L_i, Lq_i, W_i (per visit),
    network L = sum L_i, W = L / sum theta (per external customer), visit ratios v_i = lam_i / sum theta.
    """
    theta = np.asarray(theta, dtype=float)
    P = np.asarray(P, dtype=float)
    mu = np.asarray(mu, dtype=float)
    N = len(theta)
    if P.shape != (N, N) or len(mu) != N:
        raise ValueError("theta, mu must have length N and P must be N x N")
    if (P < 0).any() or (P.sum(axis=1) > 1 + 1e-12).any():
        raise ValueError("P must be substochastic (rows sum to at most 1)")
    s = [1] * N if s is None else list(s)
    lam = theta @ np.linalg.inv(np.eye(N) - P)
    nodes = [mms(lam[i], mu[i], s[i]) for i in range(N)]
    rho = np.array([nd["rho"] for nd in nodes])
    L = np.array([nd["L"] for nd in nodes])
    Lq = np.array([nd["Lq"] for nd in nodes])
    W = np.array([nd["W"] for nd in nodes])
    tot = theta.sum()
    return dict(lam=lam, rho=rho, nodes=nodes, L_i=L, Lq_i=Lq, W_i=W, L=float(L.sum()),
                W=float(L.sum() / tot), visits=lam / tot, stable=bool((rho < 1).all()),
                exit=1 - P.sum(axis=1))


def optimal_split(lam: float, p, mu_total: float) -> dict:
    """PP(lam) split to parallel M/M/1 queues with probabilities p; capacity sum mu_i = mu_total.
    Minimising the mean sojourn W = sum_i p_i / (mu_i - p_i lam) gives the square-root rule
    mu_i* = p_i lam + sqrt(p_i) (mu_total - lam) / sum_j sqrt(p_j)  (2024 final Q1: p = (1/2, 1/2)
    gives mu_1 = mu_2 = mu/2). Returns mu_opt, W_opt and W for the equal split mu_i = mu_total/len(p)."""
    p = np.asarray(p, dtype=float)
    if mu_total <= lam:
        raise ValueError("need mu_total > lam for stability")
    mu_opt = p * lam + np.sqrt(p) * (mu_total - lam) / np.sqrt(p).sum()
    W_opt = float((p / (mu_opt - p * lam)).sum())
    mu_eq = np.full(len(p), mu_total / len(p))
    with np.errstate(divide="ignore"):
        W_eq = float((p / (mu_eq - p * lam)).sum()) if (mu_eq > p * lam).all() else np.inf
    return dict(mu_opt=mu_opt, W_opt=W_opt, mu_equal=mu_eq, W_equal=W_eq)


def pooling_compare(lam: float, mu: float) -> dict:
    """2025 final Q4: PP(2 lam) into one M/M/2 (pooled) vs split 1/2-1/2 into two M/M/1 queues.
    Returns L and W of both systems (W per customer)."""
    pooled = mms(2 * lam, mu, 2)
    single = mm1(lam, mu)
    return dict(L_pooled=pooled["L"], W_pooled=pooled["W"], L_split=2 * single["L"], W_split=single["W"],
                ratio_L=pooled["L"] / (2 * single["L"]) if single["stable"] else np.nan)


def braess(lam: float, mu: float, d_long: float = 2.0, d_short: float = 1.0) -> dict:
    """Braess' paradox of the Ch.8 deck. Demand PP(2 lam); two routes: queue x then a fixed delay
    d_long, or the delay then queue y (both queues exp(mu), single server). A new link of delay
    d_short from x to y creates a third route x -> link -> y. Customers choose the fastest route
    (Wardrop equilibrium: all used routes have the same mean sojourn).

    Returns W_before = 1/(mu - lam) + d_long, W_after, the equilibrium route flows
    (f_upper, f_middle, f_lower) and the regime ("three routes", "middle only", "link unused").
    With d_long = 2, d_short = 1 the three-route equilibrium (x = y = 1, W = 3) exists for
    lam + 1 <= mu <= 2 lam + 1, and the paradox W_after > W_before holds for mu > lam + 1.
    """
    if mu <= lam:
        raise ValueError("need mu > lam (each queue carries flow lam before the link)")
    W_before = 1 / (mu - lam) + d_long
    gap = d_long - d_short                        # = sojourn in each queue when all three routes are used
    f_q = mu - 1 / gap                            # flow through each queue in the three-route equilibrium
    f_m = 2 * f_q - 2 * lam                       # flow on the middle route
    if 0 <= f_m <= 2 * lam:
        flows = (2 * lam - f_q, f_m, 2 * lam - f_q)
        W_after = gap + d_long
        regime = "three routes"
    elif f_m > 2 * lam:                           # everybody takes the new link
        x = 1 / (mu - 2 * lam) if mu > 2 * lam else np.inf
        flows = (0.0, 2 * lam, 0.0)
        W_after = 2 * x + d_short
        regime = "middle only"
    else:                                         # the link is not attractive
        flows = (lam, 0.0, lam)
        W_after = W_before
        regime = "link unused"
    return dict(W_before=W_before, W_after=W_after, flows=flows, regime=regime,
                paradox=bool(W_after > W_before + 1e-12))


# ----------------------------------------------------------------------------
# discrete-event simulation of a single station
class SimQueue:
    """Event-driven simulation of a G/G/s/K queue with optional reneging and balking.

    Parameters
    ----------
    arrival : float or sampler    inter-arrival law (a number = Poisson arrivals with that rate)
    service : float or sampler    service-time law (a number = exp(rate))
    s : int or np.inf             number of servers
    K : int or np.inf             capacity of the system (an arrival that finds K customers is lost)
    patience : None or sampler    patience of a *waiting* customer; it reneges when the wait exceeds it
    balk : None or callable       balk(n) = probability that an arrival finding n customers leaves at once

    ``run(T, rng, warmup=0.0)`` returns a dict: L, Lq (time averages over (warmup, T]),
    W, Wq (means over customers served who arrived after the warm-up), served, lost, lost_balk,
    lost_renege, loss_fraction (lost / arrivals), lam_eff (served per unit time), throughput,
    busy (time-average number of busy servers), p_time (time-average distribution of N(t)),
    p_arrival (distribution of N seen by arrivals, before joining), t, n (the sample path),
    waits, sojourns (arrays), arrivals (count).
    """

    def __init__(self, arrival, service, s=1, K=np.inf, patience=None, balk=None):
        self.arr = _as_sampler(arrival)
        self.svc = _as_sampler(service)
        self.s = s
        self.K = K
        self.pat = patience if (patience is None or callable(patience)) else _as_sampler(patience)
        self.balk = balk

    def run(self, T: float, rng=None, warmup: float = 0.0, chunk: int = 4096) -> dict:
        rng = np.random.default_rng(rng)
        s_inf = (self.s is None) or (isinstance(self.s, float) and math.isinf(self.s))
        s = None if s_inf else int(self.s)
        K_inf = (self.K is None) or (isinstance(self.K, float) and math.isinf(self.K))
        K = None if K_inf else int(self.K)
        # pre-drawn random numbers, refilled in chunks
        pool = {"a": self.arr(rng, chunk), "s": self.svc(rng, chunk),
                "p": self.pat(rng, chunk) if self.pat is not None else None}
        idx = {"a": 0, "s": 0, "p": 0}

        def draw(key):
            if idx[key] >= len(pool[key]):
                src = {"a": self.arr, "s": self.svc, "p": self.pat}[key]
                pool[key] = src(rng, chunk)
                idx[key] = 0
            v = pool[key][idx[key]]
            idx[key] += 1
            return float(v)

        events = []                       # (time, order, kind, payload)
        order = 0
        t_next_arr = draw("a")
        heapq.heappush(events, (t_next_arr, order, 0, None)); order += 1
        n = 0                             # number in system
        busy = 0                          # busy servers
        queue = deque()                   # waiting customers: ids
        arrival_time = {}                 # id -> arrival time
        waiting = set()                   # ids still waiting (for reneging)
        cid = 0
        now = 0.0
        # statistics
        area_n = 0.0; area_q = 0.0; area_busy = 0.0; last = warmup
        ts = [0.0]; ns = [0]
        p_time = {}
        seen = []                         # number in system seen by arrivals (after warm-up)
        waits = []; sojourns = []
        n_arr = 0; n_served = 0; n_balk = 0; n_renege = 0; n_lost_cap = 0
        max_track = 0

        def start_service(j, t):
            nonlocal busy, order
            busy += 1
            w = t - arrival_time[j]
            heapq.heappush(events, (t + draw("s"), order, 1, (j, w))); order += 1

        def accumulate(t):
            nonlocal area_n, area_q, area_busy, last
            if t > last:
                dt = t - last
                area_n += n * dt
                area_q += len(queue) * dt
                area_busy += busy * dt
                p_time[n] = p_time.get(n, 0.0) + dt
                last = t

        while events:
            t, _, kind, payload = heapq.heappop(events)
            if t > T:
                break
            if t >= warmup:
                accumulate(t)
            now = t
            if kind == 0:                                           # arrival
                n_arr += 1
                heapq.heappush(events, (t + draw("a"), order, 0, None)); order += 1
                if t >= warmup:
                    seen.append(n)
                if K is not None and n >= K:
                    n_lost_cap += 1
                    continue
                if self.balk is not None and rng.random() < self.balk(n):
                    n_balk += 1
                    continue
                cid += 1
                arrival_time[cid] = t
                n += 1
                ts.append(t); ns.append(n)
                if s is None or busy < s:
                    start_service(cid, t)
                else:
                    queue.append(cid)
                    waiting.add(cid)
                    if self.pat is not None:
                        heapq.heappush(events, (t + draw("p"), order, 2, cid)); order += 1
            elif kind == 1:                                         # service completion
                j, w = payload
                busy -= 1
                n -= 1
                ts.append(t); ns.append(n)
                if arrival_time[j] >= warmup:
                    n_served += 1
                    waits.append(w)
                    sojourns.append(t - arrival_time[j])
                del arrival_time[j]
                while queue:
                    k = queue.popleft()
                    if k in waiting:
                        waiting.discard(k)
                        start_service(k, t)
                        break
            else:                                                   # patience expires
                j = payload
                if j in waiting:
                    waiting.discard(j)
                    n -= 1
                    n_renege += 1
                    ts.append(t); ns.append(n)
                    del arrival_time[j]
        accumulate(T)                                                # the final stretch up to T
        horizon = max(T - warmup, 1e-12)
        ts.append(T); ns.append(n)
        tot_time = sum(p_time.values()) or 1.0
        p_time_arr = np.zeros(max(p_time) + 1 if p_time else 1)
        for k, v in p_time.items():
            p_time_arr[k] = v / tot_time
        seen = np.asarray(seen, dtype=int)
        p_arr = np.bincount(seen, minlength=len(p_time_arr)) / max(len(seen), 1) if len(seen) else np.zeros(1)
        waits = np.asarray(waits); sojourns = np.asarray(sojourns)
        lost = n_balk + n_renege + n_lost_cap
        n_arr_eff = len(seen)
        return dict(L=area_n / horizon, Lq=area_q / horizon, busy=area_busy / horizon,
                    W=float(sojourns.mean()) if len(sojourns) else np.nan,
                    Wq=float(waits.mean()) if len(waits) else np.nan,
                    served=n_served, arrivals=n_arr_eff, lost=lost, lost_balk=n_balk, lost_renege=n_renege,
                    lost_capacity=n_lost_cap,
                    loss_fraction=lost / max(n_arr, 1), lam_eff=n_served / horizon, throughput=n_served / horizon,
                    p_time=p_time_arr, p_arrival=p_arr, t=np.asarray(ts), n=np.asarray(ns),
                    waits=waits, sojourns=sojourns, T=T, warmup=warmup)


def simulate_queue(arrival, service, T, rng=None, s=1, K=np.inf, patience=None, balk=None, warmup=0.0):
    """One-call wrapper: SimQueue(arrival, service, s, K, patience, balk).run(T, rng, warmup)."""
    return SimQueue(arrival, service, s=s, K=K, patience=patience, balk=balk).run(T, rng, warmup)


def little_check(result: dict) -> dict:
    """Little's law on a simulation result: compare L with lam_eff * W and Lq with lam_eff * Wq."""
    return dict(L=result["L"], lamW=result["lam_eff"] * result["W"],
                Lq=result["Lq"], lamWq=result["lam_eff"] * result["Wq"])


def pasta_demo(lam: float, mu: float, T: float, rng=None, arrivals: str = "poisson", s: int = 1) -> dict:
    """Compare the distribution of N seen by arrivals with the time-average distribution for an
    M/M/s (arrivals = 'poisson') or a D/M/s / E_4/M/s queue of the same rate ('deterministic', 'erlang').
    PASTA: the two agree for Poisson arrivals only."""
    if arrivals == "poisson":
        arr = lam
    elif arrivals == "deterministic":
        arr = lambda r, n: np.full(n, 1.0 / lam)
    elif arrivals == "erlang":
        arr = lambda r, n: r.gamma(4, 1.0 / (4 * lam), n)
    else:
        raise ValueError("arrivals must be 'poisson', 'deterministic' or 'erlang'")
    res = SimQueue(arr, mu, s=s).run(T, rng, warmup=0.05 * T)
    m = max(len(res["p_time"]), len(res["p_arrival"]))
    pt = np.zeros(m); pa = np.zeros(m)
    pt[:len(res["p_time"])] = res["p_time"]; pa[:len(res["p_arrival"])] = res["p_arrival"]
    return dict(p_time=pt, p_arrival=pa, L_time=float((np.arange(m) * pt).sum()),
                L_arrival=float((np.arange(m) * pa).sum()), tv_distance=float(0.5 * np.abs(pt - pa).sum()),
                result=res)


# ----------------------------------------------------------------------------
# discrete-event simulation of an open network
def sim_network(theta, P, mu, s=None, T: float = 10_000.0, rng=None, service=None, warmup: float = 0.0,
                record: bool = False) -> dict:
    """Event-driven simulation of an open network of FCFS stations.

    theta : external Poisson arrival rates; P : routing matrix (exit prob = 1 - row sum);
    mu : service rates; s : servers per station (default 1, np.inf allowed);
    service : optional list of samplers (one per station) replacing the exponential service
    laws -- e.g. deterministic service to show where the product form fails.
    record : also return the joint sample path (``t``, ``n_path`` with one column per station,
    the state after each event) and the per-station arrival / departure epochs
    (``arrivals_at[i]``, ``departures_from[i]``) -- used to check Burke's theorem and the
    independence of the station counts at a fixed time.

    Returns per-station time-average numbers L_i, Lq_i, per-visit mean sojourns W_i, the
    network L = sum L_i, the mean total sojourn of an external customer W, the number of
    customers that left, and the throughput.
    """
    theta = np.asarray(theta, dtype=float)
    P = np.asarray(P, dtype=float)
    mu = np.asarray(mu, dtype=float)
    N = len(theta)
    s = [1] * N if s is None else list(s)
    s_cap = [None if (x is None or (isinstance(x, float) and math.isinf(x))) else int(x) for x in s]
    svc = [(_as_sampler(mu[i]) if service is None or service[i] is None else service[i]) for i in range(N)]
    rng = np.random.default_rng(rng)
    exit_p = 1 - P.sum(axis=1)
    cum = np.cumsum(np.hstack([P, exit_p[:, None]]), axis=1)      # routing cdf, last column = exit

    events = []; order = 0
    for i in range(N):
        if theta[i] > 0:
            heapq.heappush(events, (rng.exponential(1 / theta[i]), order, 0, i, None)); order += 1
    n = np.zeros(N, dtype=int); busy = np.zeros(N, dtype=int)
    queues = [deque() for _ in range(N)]
    entry = {}                   # customer id -> entry time into the network
    visits = {}                  # customer id -> number of station visits
    cid = 0
    area = np.zeros(N); area_q = np.zeros(N); last = warmup
    soj_visit = [[] for _ in range(N)]
    total_soj = []; n_left = 0
    path_t = [0.0]; path_n = [n.copy()]
    arr_at = [[] for _ in range(N)]; dep_from = [[] for _ in range(N)]

    def start(i, c, t_arr, now):
        """customer c (which arrived at station i at t_arr) begins service at time now"""
        nonlocal order
        busy[i] += 1
        heapq.heappush(events, (now + float(svc[i](rng, 1)[0]), order, 1, i, (c, t_arr)))
        order += 1

    def accumulate(t):
        nonlocal last
        if t > last:
            dt = t - last
            area[:] += n * dt
            area_q[:] += np.array([len(q) for q in queues]) * dt
            last = t

    def admit(i, c, t):
        n[i] += 1
        if record:
            arr_at[i].append(t); path_t.append(t); path_n.append(n.copy())
        if s_cap[i] is None or busy[i] < s_cap[i]:
            start(i, c, t, t)
        else:
            queues[i].append((c, t))

    while events:
        t, _, kind, i, payload = heapq.heappop(events)
        if t > T:
            break
        if t >= warmup:
            accumulate(t)
        if kind == 0:                                               # external arrival at station i
            heapq.heappush(events, (t + rng.exponential(1 / theta[i]), order, 0, i, None)); order += 1
            cid += 1
            entry[cid] = t; visits[cid] = 0
            visits[cid] += 1
            admit(i, cid, t)
        else:                                                       # service completion at i
            c, t_in = payload
            busy[i] -= 1; n[i] -= 1
            if record:
                dep_from[i].append(t); path_t.append(t); path_n.append(n.copy())
            if t_in >= warmup:
                soj_visit[i].append(t - t_in)
            if queues[i]:
                c2, t2 = queues[i].popleft()
                start(i, c2, t2, t)
            u = rng.random()
            j = int(np.searchsorted(cum[i], u, side="right"))
            if j >= N:                                              # leaves the network
                if entry[c] >= warmup:
                    total_soj.append(t - entry[c]); n_left += 1
                del entry[c]; del visits[c]
            else:
                visits[c] += 1
                admit(j, c, t)
    accumulate(T)
    horizon = max(T - warmup, 1e-12)
    L_i = area / horizon
    out = dict(L_i=L_i, Lq_i=area_q / horizon, L=float(L_i.sum()),
               W_i=np.array([np.mean(x) if x else np.nan for x in soj_visit]),
               W=float(np.mean(total_soj)) if total_soj else np.nan,
               left=n_left, throughput=n_left / horizon, T=T, warmup=warmup)
    if record:
        out.update(t=np.asarray(path_t), n_path=np.asarray(path_n),
                   arrivals_at=[np.asarray(a) for a in arr_at],
                   departures_from=[np.asarray(d) for d in dep_from])
    return out
