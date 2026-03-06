from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Union, Sequence
import pint

NumberOrQty = Union[float, int, pint.Quantity]


@dataclass(frozen=True)
class FluidSpec:
    """
    Canonical description of a pure fluid or mixture.

    ids:
        tuple of component names; pure fluids are stored as a 1-tuple
    composition:
        normalized fractions corresponding to ids, or None for pure fluids
    composition_basis:
        "mole" or "mass"
    """
    ids: tuple[str, ...]
    composition: tuple[float, ...] | None = None
    composition_basis: str = "mole"

    @property
    def is_pure(self) -> bool:
        return len(self.ids) == 1

    @property
    def is_mixture(self) -> bool:
        return len(self.ids) > 1


def _normalize_fractions(xs: Sequence[float]) -> tuple[float, ...]:
    vals = tuple(float(x) for x in xs)
    if any(x < 0.0 for x in vals):
        raise ValueError("Fractions must be nonnegative.")
    s = sum(vals)
    if s <= 0.0:
        raise ValueError("Fractions must sum to a positive value.")
    return tuple(x / s for x in vals)


def make_spec(
    fluid_id: str | Sequence[str],
    composition: Sequence[float] | None = None,
    composition_basis: str = "mole",
) -> FluidSpec:
    basis = composition_basis.lower().strip()
    if basis not in {"mass", "mole"}:
        raise ValueError("composition_basis must be 'mass' or 'mole'.")

    if isinstance(fluid_id, str):
        if composition is not None:
            raise ValueError("Pure fluid_id must not be given with composition.")
        return FluidSpec(ids=(fluid_id,), composition=None, composition_basis=basis)

    ids = tuple(str(x) for x in fluid_id)
    if len(ids) < 2:
        raise ValueError("Mixture fluid_id must contain at least two components.")
    if composition is None:
        raise ValueError("Mixture requires composition.")
    if len(composition) != len(ids):
        raise ValueError("composition length must match number of components.")

    return FluidSpec(
        ids=ids,
        composition=_normalize_fractions(composition),
        composition_basis=basis,
    )


def ws_to_zs(ws: Sequence[float], MWs_kg_per_mol: Sequence[float]) -> tuple[float, ...]:
    ws_n = _normalize_fractions(ws)
    ns = [w / max(MW, 1e-30) for w, MW in zip(ws_n, MWs_kg_per_mol)]
    return _normalize_fractions(ns)


def zs_to_ws(zs: Sequence[float], MWs_kg_per_mol: Sequence[float]) -> tuple[float, ...]:
    zs_n = _normalize_fractions(zs)
    ms = [z * MW for z, MW in zip(zs_n, MWs_kg_per_mol)]
    return _normalize_fractions(ms)


class FluidBackend(Protocol):
    """Backend interface in SI floats only."""
    fluid_spec: FluidSpec

    def T_crit(self) -> float: ...
    def p_crit(self) -> float: ...

    def p_sat(self, T_K: float) -> float: ...
    def T_sat(self, P_Pa: float) -> float: ...

    def s_sat_liq(self, P_Pa: float) -> float: ...
    def s_sat_vap(self, P_Pa: float) -> float: ...
    def s_fg(self, P_Pa: float) -> float: ...

    def h_sat_liq(self, P_Pa: float) -> float: ...
    def h_sat_vap(self, P_Pa: float) -> float: ...
    def h_fg(self, P_Pa: float) -> float: ...

    def p_vap(self, T_K: float) -> float: ...

    def s(self, T_K: float, P_Pa: float) -> float: ...
    def h(self, T_K: float, P_Pa: float) -> float: ...
    def rho(self, T_K: float, P_Pa: float) -> float: ...
    def mu(self, T_K: float, P_Pa: float) -> float: ...
    def a(self, T_K: float, P_Pa: float) -> float: ...

    def rho_sat_liq(self, P_Pa: float) -> float: ...
    def rho_sat_vap(self, P_Pa: float) -> float: ...

    def cp(self, T_K: float, P_Pa: float) -> float: ...
    def cv(self, T_K: float, P_Pa: float) -> float: ...


class FluidAPI(Protocol):
    """User-facing API: accepts/returns pint.Quantity by default."""
    fluid_id: str | tuple[str, ...]
    fluid_spec: FluidSpec

    def T_crit(self): ...
    def p_crit(self): ...

    def p_sat(self, T: NumberOrQty): ...
    def T_sat(self, P: NumberOrQty): ...

    def s_sat_liq(self, P: NumberOrQty): ...
    def s_sat_vap(self, P: NumberOrQty): ...
    def s_fg(self, P: NumberOrQty): ...

    def h_sat_liq(self, P: NumberOrQty): ...
    def h_sat_vap(self, P: NumberOrQty): ...
    def h_fg(self, P: NumberOrQty): ...

    def p_vap(self, T: NumberOrQty): ...

    def s(self, T: NumberOrQty, P: NumberOrQty): ...
    def h(self, T: NumberOrQty, P: NumberOrQty): ...
    def rho(self, T: NumberOrQty, P: NumberOrQty): ...
    def mu(self, T: NumberOrQty, P: NumberOrQty): ...
    def a(self, T: NumberOrQty, P: NumberOrQty): ...

    def rho_sat_liq(self, P: NumberOrQty): ...
    def rho_sat_vap(self, P: NumberOrQty): ...

    def cp(self, T: NumberOrQty, P: NumberOrQty): ...
    def cv(self, T: NumberOrQty, P: NumberOrQty): ...