"""imen266 — course library for IMEN266 Operations Research II (POSTECH).

Modules
-------
style        : shared matplotlib defaults
check        : self-check helper used by in-class TODO notebooks
dtmc         : discrete-time Markov chains (Ch.4) — modeling, transient, classification, limits
pagerank     : PageRank as a DTMC (Ch.4 practicum)
ctmc         : continuous-time Markov chains (Ch.5-6) — exponential/Poisson helpers, CTMC(Q), birth-death, transient P(t), limiting p
renewal      : renewal / regenerative processes (Ch.7) — simulation, renewal function (equation / series / closed forms), limit theorems, renewal reward, age/residual, inspection paradox, alternating renewal, semi-Markov
queueing     : queueing models (Ch.8) — M/M/s, finite capacity, Erlang B/C, truncation, M/G/1, bulk arrivals, G/G/1 approximations, Jackson networks, Braess; event-driven simulators (single station, open network)
reliability  : structure functions & system reliability (Ch.9) [stub, planned]

Notebooks for Ch.1-3 are intentionally self-contained (no import required)
so that they run on Colab with zero setup.  From Ch.4 onward, notebooks
import this package to keep in-class code short.
"""

__version__ = "0.5.0"

from .check import check  # noqa: F401
