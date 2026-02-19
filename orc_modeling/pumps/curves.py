# orc_modeling/pumps/curves.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from orc_modeling.core.solver import SolverError, bisect_root
from orc_modeling.core.types import PumpCurveRef

try:
    # Optional dependency
    from scipy.interpolate import PchipInterpolator  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    PchipInterpolator = None  # type: ignore
    _HAVE_SCIPY = False


# assuming PumpCurve is defined in this same module
# (if not, adjust the import accordingly)

def curve_model_from_ref(curve_ref: PumpCurveRef, *, N_ref_rpm: float) -> "PumpCurve":
    """
    Convert core.types.PumpCurveRef (static) into pumps.curves.PumpCurve (dynamic model).
    """
    # Head function at reference speed
    def H_ref(q: float) -> float:
        # curve_ref.H_m is a callable: H_m(Q)
        return float(curve_ref.H_m(float(q)))

    eta_ref = None
    if curve_ref.eta is not None:
        def _eta_ref(q: float) -> float:
            return float(curve_ref.eta(float(q)))
        eta_ref = _eta_ref

    return PumpCurve(
        N_ref_rpm=float(N_ref_rpm),
        Q_min_m3_s=float(curve_ref.Q_min_m3_s),
        Q_max_m3_s=float(curve_ref.Q_max_m3_s),
        H_ref=H_ref,
        eta_ref=eta_ref,
        _samples=None,
    )


def _linear_interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Simple piecewise-linear interpolation; xs must be strictly increasing."""
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])

    # binary search
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid

    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    t = (x - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


@dataclass(frozen=True)
class PumpCurve:
    """
    Pump curve at reference speed N_ref_rpm.

    Provides:
      - H(Q, N): head at flow and speed (affinity-scaled from reference)
      - Q_at_head(H, N): inverse (flow at given head and speed) for parallel combining

    Notes:
      - We assume typical centrifugal behavior: in the usable region, H decreases with Q.
      - For inversion, we solve within [Q_min, Q_max] after applying affinity scaling.
    """
    N_ref_rpm: float
    Q_min_m3_s: float
    Q_max_m3_s: float

    # Reference-speed head function H_ref(Q) [m]
    H_ref: Callable[[float], float]

    # Reference efficiency
    eta_ref: Optional[Callable[[float], float]] = None

    # Optional reference samples (used for linear fallback / debugging)
    _samples: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None

    @staticmethod
    def from_points(
        Q_m3_s: Sequence[float],
        H_m: Sequence[float],
        *,
        N_ref_rpm: float,
        eta: Optional[Sequence[float]] = None,
        use_scipy_pchip: bool = True,
    ) -> "PumpCurve":
        if len(Q_m3_s) < 2:
            raise ValueError("Need at least 2 points to build PumpCurve")
        if len(Q_m3_s) != len(H_m):
            raise ValueError("Q and H arrays must be same length")

        Qs = tuple(float(q) for q in Q_m3_s)
        Hs = tuple(float(h) for h in H_m)
        if any(Qs[i] >= Qs[i + 1] for i in range(len(Qs) - 1)):
            raise ValueError("Q points must be strictly increasing")

        qmin, qmax = Qs[0], Qs[-1]

        if use_scipy_pchip and _HAVE_SCIPY:
            interp = PchipInterpolator(Qs, Hs, extrapolate=False)

            def H_ref(q: float) -> float:
                qf = float(q)
                if qf < qmin or qf > qmax:
                    # clamp to avoid NaNs from extrapolate=False
                    qf = min(max(qf, qmin), qmax)
                return float(interp(qf))

        else:
            def H_ref(q: float) -> float:
                return _linear_interp(float(q), Qs, Hs)
            
        eta_ref = None

        if eta is not None:
            if len(eta) != len(Q_m3_s):
                raise ValueError("eta must be same length as Q_m3_s")

            # build interpolator same way you built H_ref
            if use_scipy_pchip and _HAVE_SCIPY:
                eta_interp = PchipInterpolator(Q_m3_s, eta, extrapolate=False)
                eta_ref = lambda Q: float(eta_interp(Q))
            else:
                # simple linear interpolation fallback
                import numpy as np
                eta_ref = lambda Q: float(np.interp(Q, Q_m3_s, eta))

        return PumpCurve(
            N_ref_rpm=float(N_ref_rpm),
            Q_min_m3_s=qmin,
            Q_max_m3_s=qmax,
            H_ref=H_ref,
            eta_ref=eta_ref,
            _samples=(Qs, Hs),
        )
    
    def eta(self, Q_m3_s: float, N_rpm: float) -> float:
            if self.eta_ref is None:
                raise ValueError("No eta data available on this PumpCurve")
            
            # Scale flow back to reference speed
            Q_ref, _ = self._affinity_scale(Q_m3_s, N_rpm)

            return float(self.eta_ref(Q_ref))

    def _affinity_scale(self, Q: float, N_rpm: float) -> Tuple[float, float]:
        """
        Affinity laws from reference curve:
          Q ~ N
          H ~ N^2

        If we want H(Q, N):
          Let r = N/N_ref
          Equivalent reference-flow: Q_ref = Q / r
          Head: H = r^2 * H_ref(Q_ref)
        """
        N = float(N_rpm)
        if N <= 0:
            raise ValueError("N_rpm must be > 0")

        r = N / self.N_ref_rpm
        if r <= 0:
            raise ValueError("Invalid speed ratio")

        Q_ref = float(Q) / r
        H = (r * r) * float(self.H_ref(Q_ref))
        return Q_ref, H

    def H(self, Q_m3_s: float, *, N_rpm: float) -> float:
        """Head [m] at flow Q and speed N."""
        _Qref, H = self._affinity_scale(Q_m3_s, N_rpm)
        return float(H)

    def Q_bounds_at_speed(self, *, N_rpm: float) -> Tuple[float, float]:
        """Scaled flow bounds at speed N (assuming Q scales linearly with N)."""
        r = float(N_rpm) / self.N_ref_rpm
        return (self.Q_min_m3_s * r, self.Q_max_m3_s * r)

    def Q_at_head(self, H_m: float, *, N_rpm: float) -> float:
        """
        Invert the curve at speed N: find Q such that H(Q, N) = H_target.

        Returns:
          Q >= 0 (clamped to [0, Q_max_at_speed] if no solution at positive Q).
        """
        H_target = float(H_m)
        qmin, qmax = self.Q_bounds_at_speed(N_rpm=float(N_rpm))

        # Define f(Q) = H(Q,N) - H_target
        def f(q: float) -> float:
            return self.H(q, N_rpm=float(N_rpm)) - H_target

        # Quick checks:
        f0 = f(0.0)
        fqmax = f(qmax)

        # If even shutoff head is below target -> cannot deliver positive flow at that head
        if f0 < 0:
            return 0.0

        # If at qmax we still exceed target head, then solution would be beyond qmax;
        # clamp to qmax (means "max flow at this speed within curve limits").
        if fqmax > 0:
            return float(qmax)

        # Otherwise bracket on [0, qmax] and bisect
        sol = bisect_root(f, 0.0, qmax)
        if not sol.converged:
            raise SolverError("Q_at_head did not converge")
        return float(sol.x)
