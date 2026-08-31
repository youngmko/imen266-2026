"""Discrete-time Markov chains (Chapter 4).

Used by the Ch.4 notebooks (CP3 modeling, CP4 transient, CP5 limiting/cost,
PageRank practicum) and HW2. Dependency-light: numpy required; networkx and
matplotlib only for ``plot_transition_diagram``.

Conventions
-----------
* ``P`` is row-stochastic: ``P[i, j] = P(X_{n+1} = j | X_n = i)``.
* Distributions are *row* vectors: ``p_n = p_0 @ P^n``.
* States can carry labels (``states=["down", "up"]``); every method that
  takes a state accepts either the label or the integer index.
"""
from __future__ import annotations
from math import gcd
import numpy as np


class DTMC:
    """A finite discrete-time Markov chain.

    Parameters
    ----------
    P : (n, n) array_like
        Row-stochastic transition probability matrix.
    states : sequence, optional
        State labels (default 0..n-1).
    """

    def __init__(self, P, states=None):
        self.P = np.asarray(P, dtype=float)
        n = self.P.shape[0]
        if self.P.ndim != 2 or self.P.shape != (n, n):
            raise ValueError("P must be square")
        if (self.P < -1e-12).any():
            raise ValueError("P must be nonnegative")
        bad = np.where(~np.isclose(self.P.sum(axis=1), 1.0, atol=1e-8))[0]
        if len(bad):
            raise ValueError(f"Rows {bad.tolist()} of P do not sum to 1 "
                             f"(sums {self.P.sum(axis=1)[bad].round(6).tolist()})")
        self.states = list(states) if states is not None else list(range(n))
        if len(self.states) != n:
            raise ValueError("len(states) must equal the size of P")
        self.n = n

    # ------------------------------------------------------------------
    # helpers
    def index(self, s) -> int:
        """Integer index of a state given its label (or its index)."""
        if s in self.states:
            return self.states.index(s)
        return int(s)

    def __repr__(self):
        return f"DTMC(n={self.n}, states={self.states})"

    # ------------------------------------------------------------------
    # transient analysis
    def n_step(self, n: int) -> np.ndarray:
        """Return the n-step transition matrix P^n."""
        return np.linalg.matrix_power(self.P, int(n))

    def distribution(self, pi0, n: int) -> np.ndarray:
        """Return pi0 @ P^n (row vector of state probabilities at time n)."""
        pi0 = np.asarray(pi0, dtype=float)
        return pi0 @ self.n_step(n)

    def distribution_path(self, pi0, n: int) -> np.ndarray:
        """Rows p_0, p_1, ..., p_n  (shape (n+1, n_states)) for plotting."""
        out = np.empty((n + 1, self.n))
        out[0] = np.asarray(pi0, dtype=float)
        for t in range(n):
            out[t + 1] = out[t] @ self.P
        return out

    def path_probability(self, path, pi0=None) -> float:
        """P(X_0=path[0], X_1=path[1], ...) = pi0[x0] * prod P[x_t, x_{t+1}].

        If ``pi0`` is None the probability is conditional on X_0 = path[0].
        """
        idx = [self.index(s) for s in path]
        p = 1.0 if pi0 is None else float(np.asarray(pi0, dtype=float)[idx[0]])
        for a, b in zip(idx[:-1], idx[1:]):
            p *= self.P[a, b]
        return p

    # ------------------------------------------------------------------
    # limiting behavior
    def stationary(self) -> np.ndarray:
        """Solve pi = pi P, sum(pi) = 1 by a linear system.

        Assumes the chain is irreducible (unique stationary distribution);
        for reducible chains this returns one stationary distribution.
        """
        A = np.vstack([self.P.T - np.eye(self.n), np.ones(self.n)])
        b = np.zeros(self.n + 1)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.clip(pi, 0, None)
        return pi / pi.sum()

    def mean_return_times(self) -> np.ndarray:
        """Expected return time to each state, m_i = 1 / pi_i (ergodic chain)."""
        pi = self.stationary()
        with np.errstate(divide="ignore"):
            return np.where(pi > 0, 1.0 / pi, np.inf)

    def long_run_cost(self, c) -> float:
        """Long-run average cost/reward per period:  sum_i pi_i c_i."""
        c = np.asarray(c, dtype=float)
        return float(self.stationary() @ c)

    def hitting_times(self, targets) -> np.ndarray:
        """Expected number of steps to first reach any state in ``targets``.

        Solves  h_i = 1 + sum_{j not in T} P_ij h_j  for i not in T, h_T = 0.
        Returns an array over all states (0 on the target set).
        """
        if isinstance(targets, (str, int, np.integer)) or not hasattr(targets, "__iter__"):
            targets = [targets]
        T = {self.index(s) for s in targets}
        others = [i for i in range(self.n) if i not in T]
        h = np.zeros(self.n)
        if others:
            Q = self.P[np.ix_(others, others)]
            h[others] = np.linalg.solve(np.eye(len(others)) - Q, np.ones(len(others)))
        return h

    # ------------------------------------------------------------------
    # classification of states
    def reachability(self) -> np.ndarray:
        """Boolean matrix R with R[i, j] = True iff j is accessible from i."""
        A = self.P > 1e-12
        R = A | np.eye(self.n, dtype=bool)
        for _ in range(int(np.ceil(np.log2(max(self.n, 2)))) + 1):
            R = R | ((R.astype(int) @ R.astype(int)) > 0)
        return R

    def communicating_classes(self) -> list:
        """List of communicating classes (each a list of state indices)."""
        R = self.reachability()
        seen, classes = set(), []
        for i in range(self.n):
            if i in seen:
                continue
            cls = [j for j in range(self.n) if R[i, j] and R[j, i]]
            seen.update(cls)
            classes.append(cls)
        return classes

    def classify(self) -> dict:
        """Classes with their type (finite chain: closed <=> recurrent).

        Returns {"classes": [...], "type": ["recurrent"|"transient", ...],
                 "period": [...], "recurrent": set(idx), "transient": set(idx)}
        """
        R = self.reachability()
        classes = self.communicating_classes()
        types, periods = [], []
        rec, tra = set(), set()
        for cls in classes:
            closed = all(not R[i, j] for i in cls for j in range(self.n) if j not in cls)
            types.append("recurrent" if closed else "transient")
            (rec if closed else tra).update(cls)
            periods.append(self._class_period(cls))
        return {"classes": classes, "type": types, "period": periods,
                "recurrent": rec, "transient": tra}

    def _class_period(self, cls) -> int:
        """gcd of cycle lengths inside one communicating class (BFS levels)."""
        cls = list(cls)
        if len(cls) == 1 and self.P[cls[0], cls[0]] <= 1e-12:
            return 0            # no return possible: period undefined
        inside = set(cls)
        level = {cls[0]: 0}
        frontier = [cls[0]]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.where(self.P[u] > 1e-12)[0]:
                    v = int(v)
                    if v in inside and v not in level:
                        level[v] = level[u] + 1
                        nxt.append(v)
            frontier = nxt
        d = 0
        for u in cls:
            for v in np.where(self.P[u] > 1e-12)[0]:
                v = int(v)
                if v in inside:
                    d = gcd(d, level[u] + 1 - level[v])
        return d

    def is_irreducible(self) -> bool:
        return len(self.communicating_classes()) == 1

    def period(self, s=None) -> int:
        """Period of state ``s`` (or of the chain if irreducible and s is None)."""
        info = self.classify()
        if s is None:
            if not self.is_irreducible():
                raise ValueError("chain is reducible: give a state")
            return info["period"][0]
        i = self.index(s)
        for cls, d in zip(info["classes"], info["period"]):
            if i in cls:
                return d
        raise AssertionError("unreachable")

    def is_aperiodic(self) -> bool:
        return all(d == 1 for d in self.classify()["period"])

    def absorbing_states(self) -> list:
        return [self.states[i] for i in range(self.n) if self.P[i, i] >= 1 - 1e-12]

    def is_ergodic(self) -> bool:
        """Irreducible + aperiodic (finite => positive recurrent)."""
        return self.is_irreducible() and self.is_aperiodic()

    def summary(self) -> str:
        info = self.classify()
        lines = [f"{self.n} states; {len(info['classes'])} communicating class(es)"]
        for cls, t, d in zip(info["classes"], info["type"], info["period"]):
            labels = [self.states[i] for i in cls]
            lines.append(f"  {labels}: {t}, period {d if d else 'undefined'}")
        lines.append(f"absorbing: {self.absorbing_states() or 'none'}")
        lines.append(f"ergodic: {self.is_ergodic()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # simulation
    def sample_path(self, n_steps: int, x0=None, rng=None) -> np.ndarray:
        """Simulate a path of length n_steps+1 (indices of states).

        ``x0`` may be a state label, an index, or None (uniform random start).
        """
        rng = np.random.default_rng(rng)
        C = np.cumsum(self.P, axis=1)
        C[:, -1] = 1.0                      # guard against round-off
        u = rng.random(n_steps)
        x = np.empty(n_steps + 1, dtype=int)
        x[0] = rng.integers(self.n) if x0 is None else self.index(x0)
        for t in range(n_steps):
            x[t + 1] = np.searchsorted(C[x[t]], u[t], side="right")
        return x

    def occupancy(self, path) -> np.ndarray:
        """Empirical state frequencies of a simulated path."""
        path = np.asarray(path)
        return np.bincount(path, minlength=self.n) / len(path)

    def empirical_P(self, path) -> np.ndarray:
        """Transition-frequency estimate of P from a path (rows with no
        visits are left as NaN)."""
        path = np.asarray(path)
        N = np.zeros((self.n, self.n))
        np.add.at(N, (path[:-1], path[1:]), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return N / N.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def plot_transition_diagram(self, ax=None, seed=7, layout="circular"):
        """Draw the transition diagram with edge labels (requires networkx)."""
        import matplotlib.pyplot as plt
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(self.states)
        for i in range(self.n):
            for j in range(self.n):
                if self.P[i, j] > 1e-12:
                    G.add_edge(self.states[i], self.states[j], weight=self.P[i, j])
        pos = (nx.circular_layout(G) if layout == "circular"
               else nx.spring_layout(G, seed=seed))
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900,
                               node_color="#f0f0f0", edgecolors="k")
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle="arc3,rad=0.18",
                               arrowsize=15, min_target_margin=18)
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, label_pos=0.3, font_size=9,
            edge_labels={(u, v): f"{d['weight']:.3g}"
                         for u, v, d in G.edges(data=True)})
        ax.set_axis_off()
        return ax
