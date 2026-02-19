# orc_modeling/core/solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class SolveResult:
    """Result of a 1D root solve."""
    x: float
    f_x: float
    iterations: int
    converged: bool


class SolverError(RuntimeError):
    """Raised when a solve fails in a non-recoverable way."""


def bracket_root(
    f: Callable[[float], float],
    x0: float,
    step: float,
    max_expand: int = 50,
    growth: float = 1.6,
) -> Tuple[float, float]:
    """
    Expand an interval around x0 until f(a) and f(b) have opposite signs.

    Parameters
    ----------
    f : callable
        Scalar function.
    x0 : float
        Initial guess (near expected root).
    step : float
        Initial half-width of bracket (must be > 0).
    max_expand : int
        Maximum expansions.
    growth : float
        Multiplicative growth factor for step each expansion.

    Returns
    -------
    (a, b) : tuple[float, float]
        Bracketing interval with opposite signs.

    Raises
    ------
    SolverError
        If a bracketing interval cannot be found.
    """
    if step <= 0:
        raise ValueError("step must be > 0")

    a = x0 - step
    b = x0 + step
    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        return a, a
    if fb == 0.0:
        return b, b
    if fa * fb < 0:
        return a, b

    s = step
    for _ in range(max_expand):
        s *= growth
        a = x0 - s
        b = x0 + s
        fa = f(a)
        fb = f(b)
        if fa == 0.0:
            return a, a
        if fb == 0.0:
            return b, b
        if fa * fb < 0:
            return a, b

    raise SolverError("Failed to bracket root")


def bisect_root(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    xtol: float = 1e-8,
    ftol: float = 1e-10,
    max_iter: int = 200,
) -> SolveResult:
    """
    Bisection root solver for f(x)=0 on a bracketing interval [a, b].

    Requires f(a) and f(b) to have opposite signs (or one endpoint is exactly a root).

    Returns a SolveResult with convergence info.
    """
    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        return SolveResult(x=a, f_x=fa, iterations=0, converged=True)
    if fb == 0.0:
        return SolveResult(x=b, f_x=fb, iterations=0, converged=True)
    if fa * fb > 0:
        raise SolverError("bisect_root requires a bracketing interval (opposite signs)")

    lo, hi = (a, b) if a < b else (b, a)
    flo, fhi = (fa, fb) if a < b else (fb, fa)

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)

        # Convergence tests
        if abs(fmid) <= ftol or (hi - lo) <= xtol:
            return SolveResult(x=mid, f_x=fmid, iterations=it, converged=True)

        # Keep the sub-interval that brackets the root
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return SolveResult(x=0.5 * (lo + hi), f_x=f(0.5 * (lo + hi)), iterations=max_iter, converged=False)


def find_intersection(
    f: Callable[[float], float],
    g: Callable[[float], float],
    *,
    x0: float,
    step: float,
    xtol: float = 1e-8,
    ftol: float = 1e-10,
    max_expand: int = 50,
    max_iter: int = 200,
) -> SolveResult:
    """
    Solve f(x) = g(x) near x0 by finding a root of h(x) = f(x) - g(x).

    This is the primitive used for "pump curve intersects system curve".
    """
    def h(x: float) -> float:
        return f(x) - g(x)

    a, b = bracket_root(h, x0=x0, step=step, max_expand=max_expand)
    if a == b:  # exact root at endpoint
        return SolveResult(x=a, f_x=h(a), iterations=0, converged=True)

    return bisect_root(h, a, b, xtol=xtol, ftol=ftol, max_iter=max_iter)
