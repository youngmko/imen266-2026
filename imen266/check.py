"""Self-check helper for in-class TODO notebooks.

Design goal: a TODO cell left as `...` (Ellipsis) or None must NOT crash
"Run all"; it should print a friendly "not yet" message instead, and turn
green once the student's answer matches the analytic value.
"""

def check(name, got, expected, tol=None, rel=0.05):
    """Compare a student's value against the expected answer.

    Parameters
    ----------
    name : str
        Label shown in the message.
    got : number or None or Ellipsis
        The student's value (may be unfinished).
    expected : number
        Analytic/reference value.
    tol : float, optional
        Absolute tolerance. If None, uses relative tolerance `rel`
        (default 5%, suitable for Monte-Carlo answers).
    """
    if got is None or got is Ellipsis:
        print(f"[TODO] {name}: not computed yet — fill in the TODO cell above.")
        return False
    try:
        g = float(got)
    except (TypeError, ValueError):
        print(f"[??]   {name}: got a non-numeric value ({got!r}).")
        return False
    if tol is None:
        ok = abs(g - expected) <= rel * max(abs(expected), 1e-12)
    else:
        ok = abs(g - expected) <= tol
    mark = "OK " if ok else "X  "
    print(f"[{mark}] {name}: got {g:.6g}   (reference {expected:.6g})")
    return ok
