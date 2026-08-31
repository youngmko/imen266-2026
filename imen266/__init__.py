"""imen266 — course library for IMEN266 Operations Research II (POSTECH).

Modules
-------
style        : shared matplotlib defaults
check        : self-check helper used by in-class TODO notebooks
dtmc         : discrete-time Markov chains (Ch.4) — modeling, transient, classification, limits
pagerank     : PageRank as a DTMC (Ch.4 practicum)
ctmc         : continuous-time Markov chains (Ch.5-6)   [stub, planned]
renewal      : renewal / regenerative processes (Ch.7)  [stub, planned]
queueing     : Markovian queues & Jackson networks (Ch.8) [stub, planned]
reliability  : structure functions & system reliability (Ch.9) [stub, planned]

Notebooks for Ch.1-3 are intentionally self-contained (no import required)
so that they run on Colab with zero setup.  From Ch.4 onward, notebooks
import this package to keep in-class code short.
"""

__version__ = "0.2.0"

from .check import check  # noqa: F401
