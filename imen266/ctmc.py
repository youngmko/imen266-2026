"""Continuous-time Markov chains (Chapters 5-6).

Used by the Ch.5 notebooks (CP6 exponential/Poisson, CP7 CTMC modeling,
CP8 transient/limiting/cost) and HW3. Dependency-light: numpy + scipy
required; networkx and matplotlib only for ``plot_rate_diagram``.

Conventions
-----------
* ``Q`` is the (infinitesimal) generator: ``Q[i, j] >= 0`` is the rate of
  the transition i -> j for i != j, and ``Q[i, i] = -sum_{j != i} Q[i, j]``
  (rows sum to 0). ``rates[i] = -Q[i, i]`` is the total rate out of i, so
  the sojourn time in i is exp(rates[i]).
* Distributions are *row* vectors: ``p(t) = p(0) @ P(t)`` with
  ``P(t) = expm(Q t)`` (Kolmogorov forward equations ``dP/dt = P(t) Q``).
* The limiting distribution solves ``p Q = 0, sum(p) = 1``.
* States can carry labels (``states=["down", "up"]``); every method that
  takes a state accepts either the label or the integer index.

Public API
----------
CTMC(Q, states)            rates, embedded_chain(), transition_matrix(t),
                           distribution(p0, t), distribution_path(p0, ts, method="expm"|"ode"),
                           stationary(), long_run_cost(c), mean_sojourn(),
                           sample_path(T, x0, rng), occupancy(path), summary(),
                           plot_rate_diagram(), uniformize()
birth_death(lam, mu)       CTMC with birth rates lam[k] (k -> k+1) and death rates mu[k] (k+1 -> k)
bd_stationary(lam, mu)     product-form limiting distribution of a birth-death chain
machine_repair(...)        number of UP machines with c repair persons (CP7 #2, CP8 #2/#9)
erlang_loss(a, K)          Erlang-B blocking probability (telephone switch, CP8 #3)
expm_series(Q, t, ...)     the truncated series sum Q^n t^n / n! and the stopping index N*
expm_limit(Q, t, N)        the (I + Qt/N)^N approximation
poisson_process(rate, T)   arrival epochs of a PP(rate) on [0, T]
thin(times, p)             probabilistic splitting of a point process
merge(*times)              superposition of point processes
"""
from __future__ import annotations
import math
import numpy as np
from scipy.linalg import expm

__all__ = ["CTMC", "birth_death", "bd_stationary", "machine_repair", "erlang_loss",
           "expm_series", "expm_limit", "poisson_process", "thin", "merge"]


class CTMC:
    """A finite continuous-time Markov chain given by its generator ``Q``.

    Parameters
    ----------
    Q : (n, n) array_like
        Generator matrix: nonnegative off-diagonal entries, rows summing to 0.
    states : sequence, optional
        State labels (default 0..n-1).
    """

    def __init__(self, Q, states=None):
        self.Q = np.asarray(Q, dtype=float)
        n = self.Q.shape[0]
        if self.Q.ndim != 2 or self.Q.shape != (n, n):
            raise ValueError("Q must be square")
        off = self.Q - np.diag(np.diag(self.Q))
        if (off < -1e-12).any():
            bad = np.argwhere(off < -1e-12)
            raise ValueError(f"off-diagonal rates must be >= 0 (negative at {bad.tolist()})")
        bad = np.where(~np.isclose(self.Q.sum(axis=1), 0.0, atol=1e-8))[0]
        if len(bad):
            raise ValueError(f"Rows {bad.tolist()} of Q do not sum to 0 "
                             f"(sums {self.Q.sum(axis=1)[bad].round(6).tolist()})")
        self.states = list(states) if states is not None else list(range(n))
        if len(self.states) != n:
            raise ValueError("len(states) must equal the size of Q")
        self.n = n

    # ------------------------------------------------------------------
    # helpers
    def index(self, s) -> int:
        """Integer index of a state given its label (or its index)."""
        if s in self.states:
            return self.states.index(s)
        return int(s)

    def __repr__(self):
        return f"CTMC(n={self.n}, states={self.states})"

    @property
    def rates(self) -> np.ndarray:
        """Total rate out of each state, lambda_i = -Q_ii (sojourn time ~ exp(lambda_i))."""
        return -np.diag(self.Q)

    def mean_sojourn(self) -> np.ndarray:
        """Expected sojourn time in each state, 1 / lambda_i (inf for absorbing states)."""
        lam = self.rates
        with np.errstate(divide="ignore"):
            return np.where(lam > 0, 1.0 / lam, np.inf)

    # ------------------------------------------------------------------
    # embedded (jump) chain
    def embedded_chain(self):
        """The embedded DTMC: P_ij = q_ij / lambda_i (absorbing states get P_ii = 1)."""
        from .dtmc import DTMC
        lam = self.rates
        P = np.zeros_like(self.Q)
        for i in range(self.n):
            if lam[i] > 1e-12:
                P[i] = self.Q[i] / lam[i]
                P[i, i] = 0.0
            else:
                P[i, i] = 1.0
        return DTMC(P, self.states)

    # ------------------------------------------------------------------
    # transient analysis
    def transition_matrix(self, t: float) -> np.ndarray:
        """P(t) = expm(Q t), the matrix of p_ij(t) = P(X(t) = j | X(0) = i)."""
        return expm(self.Q * float(t))

    P = transition_matrix

    def distribution(self, p0, t: float) -> np.ndarray:
        """p(t) = p(0) @ P(t) as a row vector."""
        p0 = np.asarray(p0, dtype=float)
        return p0 @ self.transition_matrix(t)

    def distribution_path(self, p0, ts, method: str = "expm") -> np.ndarray:
        """Rows p(t) for each t in ``ts`` (shape (len(ts), n)).

        method="expm": p(t) = p0 @ expm(Q t) for each t.
        method="ode":  integrate the Kolmogorov forward equations dp/dt = p Q
                       with scipy.integrate.solve_ivp (what one does when expm
                       is unavailable or Q is time-dependent).
        """
        p0 = np.asarray(p0, dtype=float)
        ts = np.asarray(ts, dtype=float)
        if method == "expm":
            return np.array([p0 @ expm(self.Q * t) for t in ts])
        if method == "ode":
            from scipy.integrate import solve_ivp
            Q = self.Q
            t0, t1 = float(ts.min()), float(ts.max())
            sol = solve_ivp(lambda t, p: p @ Q, (min(t0, 0.0), t1), p0,
                            t_eval=ts, rtol=1e-9, atol=1e-11, method="DOP853")
            return sol.y.T
        raise ValueError("method must be 'expm' or 'ode'")

    def uniformize(self, Lam: float | None = None):
        """Uniformization: (Lam, P_tilde) with Q = Lam (P_tilde - I), Lam >= max_i lambda_i.

        Then P(t) = sum_n e^{-Lam t} (Lam t)^n / n! * P_tilde^n: the CTMC is a
        DTMC (matrix P_tilde) observed at the events of a PP(Lam).
        """
        if Lam is None:
            Lam = float(self.rates.max())
        if Lam < self.rates.max() - 1e-12:
            raise ValueError("Lam must be at least max_i lambda_i")
        P_tilde = np.eye(self.n) + self.Q / Lam
        return Lam, P_tilde

    # ------------------------------------------------------------------
    # limiting behavior
    def stationary(self) -> np.ndarray:
        """Solve p Q = 0, sum(p) = 1 by a linear system.

        Assumes the chain is irreducible (unique limiting distribution);
        for reducible chains this returns one stationary distribution.
        """
        A = np.vstack([self.Q.T, np.ones(self.n)])
        b = np.zeros(self.n + 1)
        b[-1] = 1.0
        p, *_ = np.linalg.lstsq(A, b, rcond=None)
        p = np.clip(p, 0, None)
        return p / p.sum()

    def stationary_from_embedded(self) -> np.ndarray:
        """p_j proportional to pi_j / lambda_j, pi = stationary distribution of the jump chain."""
        pi = self.embedded_chain().stationary()
        w = pi / self.rates
        return w / w.sum()

    def long_run_cost(self, c) -> float:
        """Long-run average cost/reward per unit time:  sum_j c_j p_j."""
        c = np.asarray(c, dtype=float)
        return float(self.stationary() @ c)

    # ------------------------------------------------------------------
    # structure
    def reachability(self) -> np.ndarray:
        return self.embedded_chain().reachability()

    def is_irreducible(self) -> bool:
        return self.embedded_chain().is_irreducible()

    def absorbing_states(self) -> list:
        return [self.states[i] for i in range(self.n) if self.rates[i] <= 1e-12]

    def summary(self) -> str:
        emb = self.embedded_chain()
        info = emb.classify()
        lines = [f"{self.n} states; {len(info['classes'])} communicating class(es)"]
        for cls, t in zip(info["classes"], info["type"]):
            labels = [self.states[i] for i in cls]
            lines.append(f"  {labels}: {t}")
        lines.append(f"absorbing: {self.absorbing_states() or 'none'}")
        lines.append(f"sojourn rates lambda_i: {np.round(self.rates, 4).tolist()}")
        lines.append(f"irreducible (finite => ergodic, no aperiodicity needed): {self.is_irreducible()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # simulation
    def sample_path(self, T: float, x0=None, rng=None, method: str = "clocks"):
        """Simulate on [0, T]. Returns (t, x): x[k] is the state on [t[k], t[k+1]),
        t[0] = 0 and the last entry of t is T (the final state is repeated).

        method="clocks": in state i start one exponential clock per possible
        transition (rate q_ij); the first to ring wins (minimum of exponentials).
        method="gillespie": equivalent two-step version: sojourn ~ exp(lambda_i),
        then jump to j w.p. q_ij / lambda_i.
        """
        rng = np.random.default_rng(rng)
        i = rng.integers(self.n) if x0 is None else self.index(x0)
        t, x = [0.0], [i]
        now = 0.0
        while True:
            lam = self.rates[i]
            if lam <= 1e-15:           # absorbing: stays forever
                break
            if method == "clocks":
                js = np.where(self.Q[i] > 1e-15)[0]
                js = js[js != i]
                clocks = rng.exponential(1.0 / self.Q[i, js])
                k = int(np.argmin(clocks))
                dt, j = clocks[k], int(js[k])
            else:
                dt = rng.exponential(1.0 / lam)
                probs = self.Q[i].copy(); probs[i] = 0.0; probs /= lam
                j = int(rng.choice(self.n, p=probs))
            now += dt
            if now >= T:
                break
            t.append(now); x.append(j); i = j
        t.append(float(T)); x.append(x[-1])
        return np.array(t), np.array(x)

    def occupancy(self, path) -> np.ndarray:
        """Time-weighted fraction of [0, T] spent in each state, from sample_path output."""
        t, x = path
        occ = np.zeros(self.n)
        np.add.at(occ, x[:-1], np.diff(t))
        return occ / t[-1]

    def visit_counts(self, path) -> np.ndarray:
        """Number of visits (jumps into each state, plus the initial state) - the embedded chain's view."""
        t, x = path
        return np.bincount(x[:-1], minlength=self.n)

    # ------------------------------------------------------------------
    def plot_rate_diagram(self, ax=None, seed=7, layout="circular", fmt="{:.3g}"):
        """Draw the rate diagram (nodes = states, arrows = transition rates)."""
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(self.states)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.Q[i, j] > 1e-12:
                    G.add_edge(self.states[i], self.states[j], weight=self.Q[i, j])
        if layout == "line":
            pos = {s: (k, 0.0) for k, s in enumerate(self.states)}
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=seed)
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4) if layout != "line" else (1.6 * self.n + 1, 2.2))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900,
                               node_color="#f0f0f0", edgecolors="k")
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle="arc3,rad=0.25",
                               arrowsize=15, min_target_margin=18)
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, label_pos=0.3, font_size=9,
            connectionstyle="arc3,rad=0.25",
            edge_labels={(u, v): fmt.format(d["weight"]) for u, v, d in G.edges(data=True)})
        ax.set_axis_off()
        return ax


# ----------------------------------------------------------------------
# birth-death chains and friends
def birth_death(lam, mu, states=None) -> CTMC:
    """CTMC on 0..n-1 with birth rates lam[k]: k -> k+1 (k = 0..n-2) and
    death rates mu[k]: k+1 -> k (k = 0..n-2). ``len(lam) == len(mu) == n-1``."""
    lam = np.asarray(lam, dtype=float); mu = np.asarray(mu, dtype=float)
    if len(lam) != len(mu):
        raise ValueError("lam and mu must have the same length (n-1)")
    n = len(lam) + 1
    Q = np.zeros((n, n))
    for k in range(n - 1):
        Q[k, k + 1] = lam[k]
        Q[k + 1, k] = mu[k]
    Q -= np.diag(Q.sum(axis=1))
    return CTMC(Q, states)


def bd_stationary(lam, mu) -> np.ndarray:
    """Product form: p_k = p_0 * prod_{i<k} lam[i] / mu[i]  (arc balance lam_k p_k = mu_{k+1} p_{k+1})."""
    lam = np.asarray(lam, dtype=float); mu = np.asarray(mu, dtype=float)
    w = np.concatenate([[1.0], np.cumprod(lam / mu)])
    return w / w.sum()


def machine_repair(n_machines: int, n_crews: int, fail_rate: float, repair_rate: float) -> CTMC:
    """Number of UP machines (states 0..n) with c independent repair persons:
    k -> k-1 at rate k*fail_rate, k -> k+1 at rate min(n-k, c)*repair_rate."""
    n, c = int(n_machines), int(n_crews)
    lam = [min(n - k, c) * repair_rate for k in range(n)]      # births: repairs complete
    mu = [(k + 1) * fail_rate for k in range(n)]              # deaths: failures
    return birth_death(lam, mu, states=list(range(n + 1)))


def erlang_loss(a: float, K: int) -> float:
    """Erlang-B: blocking probability of an M/M/K/K system with offered load a = lam/mu."""
    terms = np.array([a ** k / math.factorial(k) for k in range(K + 1)])
    return float(terms[-1] / terms.sum())


# ----------------------------------------------------------------------
# matrix exponential, as computed in class
def expm_series(Q, t: float = 1.0, N: int | None = None, eps: float = 1e-10):
    """Truncated series  sum_{n=0}^{N} (Qt)^n / n!.

    If N is None, stop at N* = the first N for which the N-th term is smaller
    than eps in max-norm (the stopping rule of the class derivation).
    Returns (P_N, N_used, partial_sums) where partial_sums[k] is the sum up to k.
    """
    A = np.asarray(Q, dtype=float) * float(t)
    n = A.shape[0]
    term = np.eye(n); S = term.copy(); sums = [S.copy()]
    k = 0
    while True:
        k += 1
        term = term @ A / k
        S = S + term
        sums.append(S.copy())
        if N is not None:
            if k >= N:
                break
        elif np.abs(term).max() < eps or k > 500:
            break
    return S, k, sums


def expm_limit(Q, t: float = 1.0, N: int = 1000) -> np.ndarray:
    """(I + Qt/N)^N  ->  expm(Qt) as N -> inf."""
    A = np.asarray(Q, dtype=float) * float(t)
    return np.linalg.matrix_power(np.eye(A.shape[0]) + A / N, int(N))


# ----------------------------------------------------------------------
# Poisson-process helpers (Ch.5)
def poisson_process(rate: float, T: float, rng=None) -> np.ndarray:
    """Arrival epochs S_1 < S_2 < ... <= T of a PP(rate): cumulative sums of iid exp(rate)."""
    rng = np.random.default_rng(rng)
    n_guess = int(rate * T + 10 * math.sqrt(rate * T + 1)) + 10
    gaps = rng.exponential(1.0 / rate, n_guess)
    S = np.cumsum(gaps)
    while S[-1] < T:                                  # (rare) extend
        S = np.concatenate([S, S[-1] + np.cumsum(rng.exponential(1.0 / rate, n_guess))])
    return S[S <= T]


def thin(times, p: float, rng=None):
    """Split a point process: each point is kept w.p. p independently. Returns (kept, dropped)."""
    rng = np.random.default_rng(rng)
    times = np.asarray(times, dtype=float)
    keep = rng.random(len(times)) < p
    return times[keep], times[~keep]


def merge(*times) -> np.ndarray:
    """Superpose point processes (sorted union of their epochs)."""
    return np.sort(np.concatenate([np.asarray(t, dtype=float) for t in times]))
