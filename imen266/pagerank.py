"""PageRank as a Markov chain (Chapter 4 practicum).

Following the class handout (Harchol-Balter, *Performance Modeling and
Design of Computer Systems*, Ch. 10): a random surfer follows one of the
k outgoing links of the current page with probability 1/k; dead ends and
spider traps break ergodicity; "taxation" (damping) restores it.
"""
from __future__ import annotations
import numpy as np


def link_matrix(edges, n: int, dangling: str = "uniform") -> np.ndarray:
    """Row-stochastic surfer matrix from a list of (from, to) page indices.

    Pages with no outgoing links ("dead ends") get a uniform row if
    ``dangling="uniform"``; with ``dangling="keep"`` their row stays all-zero
    (then the matrix is *sub*-stochastic and mass leaks away).
    """
    A = np.zeros((n, n))
    for u, v in edges:
        A[u, v] += 1.0
    out = A.sum(axis=1)
    P = np.zeros_like(A)
    for i in range(n):
        if out[i] > 0:
            P[i] = A[i] / out[i]
        elif dangling == "uniform":
            P[i] = 1.0 / n
    return P


def google_matrix(P: np.ndarray, beta: float = 0.85) -> np.ndarray:
    """G = beta * P + (1 - beta) * (1/n) 11^T  ("tax" (1-beta) redistributed
    equally to every page). For 0 < beta < 1, G is positive => ergodic."""
    n = P.shape[0]
    return beta * P + (1.0 - beta) / n * np.ones((n, n))


def pagerank(P: np.ndarray, beta: float = 0.85, tol: float = 1e-12,
             max_iter: int = 10_000, pi0=None, history: bool = False):
    """PageRank vector by power iteration  pi <- pi G.

    Returns ``pi`` (and the list of iterates if ``history=True``).
    """
    n = P.shape[0]
    G = google_matrix(P, beta)
    pi = np.full(n, 1.0 / n) if pi0 is None else np.asarray(pi0, dtype=float)
    hist = [pi.copy()]
    for _ in range(max_iter):
        new = pi @ G
        hist.append(new.copy())
        if np.abs(new - pi).sum() < tol:
            pi = new
            break
        pi = new
    return (pi, hist) if history else pi
