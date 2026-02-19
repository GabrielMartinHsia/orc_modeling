# orc_modeling/pumps/parallel_bank.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from orc_modeling.core.solver import SolverError, bisect_root
from .curves import PumpCurve

__all__ = [
    "ParallelBank",
    "enforce_bep_band",
]

# """
# `parallel_bank.py` (primitive)

# This is the lowest-level pump combining module. Given:

#     * a PumpCurve, and,
#     * a number of identical pumps in parallel

# Evaluate `H_total(Q_total, N)` by splitting flow across pumps. Invert with `Q_at_head(H, N)` as needed.
# """


@dataclass(frozen=True)
class ParallelBank:
    """
    Parallel pump bank with a single shared VFD speed across all VFD pumps.

    All pumps discharge into the same header:
      - common head
      - flows add

    We build an equivalent bank head function H_bank(Q_total) by:
      1) Q_total(H) = n_fixed*Q_fixed(H) + n_vfd*Q_vfd(H)
      2) invert to get H as a function of Q_total
    """
    curve: PumpCurve

    n_fixed: int
    n_vfd: int

    N_fixed_rpm: float
    N_vfd_rpm: Optional[float]  # if n_vfd > 0, must be provided

    def _Q_total_at_head(self, H: float) -> float:
        Qt = 0.0
        if self.n_fixed > 0:
            q1 = self.curve.Q_at_head(H, N_rpm=self.N_fixed_rpm)
            Qt += self.n_fixed * q1
        if self.n_vfd > 0:
            if self.N_vfd_rpm is None:
                raise ValueError("N_vfd_rpm must be provided when n_vfd > 0")
            q2 = self.curve.Q_at_head(H, N_rpm=self.N_vfd_rpm)
            Qt += self.n_vfd * q2
        return float(Qt)

    def head_fn(self) -> Callable[[float], float]:
        """
        Return H_bank(Q_total) [m].

        Raises SolverError if Q_total exceeds bank capacity.
        """
        def H_bank(Q_total: float) -> float:
            Q = float(Q_total)
            if Q < 0:
                raise ValueError("Q_total must be >= 0")

            # Upper head bound: use max of shutoff heads of active sets
            H_hi = 0.0
            if self.n_fixed > 0:
                H_hi = max(H_hi, self.curve.H(0.0, N_rpm=self.N_fixed_rpm))
            if self.n_vfd > 0:
                if self.N_vfd_rpm is None:
                    raise ValueError("N_vfd_rpm must be provided when n_vfd > 0")
                H_hi = max(H_hi, self.curve.H(0.0, N_rpm=self.N_vfd_rpm))

            if H_hi <= 0.0:
                return 0.0

            # At H=0, bank flow is maximum (for typical curves)
            Q_max = self._Q_total_at_head(0.0)
            if Q > Q_max + 1e-12:
                raise SolverError("Requested flow exceeds bank capacity at this speed set")

            # Solve Q_total_at_head(H) - Q = 0 on [0, H_hi]
            def f(H: float) -> float:
                return self._Q_total_at_head(H) - Q

            sol = bisect_root(f, 0.0, H_hi)
            if not sol.converged:
                raise SolverError("Failed to invert parallel bank curve")
            return float(sol.x)

        return H_bank
    
    def flows_per_pump_at_head(self, H: float) -> tuple[Optional[float], Optional[float]]:
        """
        Return (q_fixed_each, q_vfd_each) at common head H.
        Values are None if that subset is inactive.
        """
        q_fixed = None
        q_vfd = None
        if self.n_fixed > 0:
            q_fixed = float(self.curve.Q_at_head(H, N_rpm=self.N_fixed_rpm))
        if self.n_vfd > 0:
            if self.N_vfd_rpm is None:
                raise ValueError("N_vfd_rpm must be provided when n_vfd > 0")
            q_vfd = float(self.curve.Q_at_head(H, N_rpm=self.N_vfd_rpm))
        return q_fixed, q_vfd

#-------------------------------
# BEP Enforcement
#-------------------------------
def enforce_bep_band(
    bank: ParallelBank,
    *,
    Q_total: float,
    Q_bep_ref_m3_s: float,
    N_ref_rpm: float,
    low: float = 0.8,
    high: float = 1.1,
) -> None:
    """
    Raise SolverError if any active subset is outside BEP band.
    """
    if bank.n_fixed + bank.n_vfd <= 0:
        return
    if Q_bep_ref_m3_s is None:
        return

    # Get common header head by inverting the bank
    H_bank = bank.head_fn()
    H_head = float(H_bank(Q_total))

    q_fixed, q_vfd = bank.flows_per_pump_at_head(H_head)

    def _check(q_each: float, N: float) -> None:
        r = float(N) / float(N_ref_rpm)
        Q_bep = float(Q_bep_ref_m3_s) * r
        if Q_bep <= 0:
            return
        ratio = float(q_each) / Q_bep
        if ratio < low or ratio > high:
            raise SolverError("Pump operating outside BEP band")

    if q_fixed is not None and bank.n_fixed > 0:
        _check(q_fixed, bank.N_fixed_rpm)
    if q_vfd is not None and bank.n_vfd > 0 and bank.N_vfd_rpm is not None:
        _check(q_vfd, bank.N_vfd_rpm)
