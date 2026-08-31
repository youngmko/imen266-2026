"""Shared matplotlib style for IMEN266 figures."""
import matplotlib as mpl

def apply():
    mpl.rcParams.update({
        "figure.figsize": (7, 4),
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "legend.frameon": False,
    })
