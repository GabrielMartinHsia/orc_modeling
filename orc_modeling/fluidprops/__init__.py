from __future__ import annotations
from dataclasses import dataclass
from typing import Union, Sequence
import pint

from .base import FluidBackend, FluidAPI, FluidSpec, make_spec
from orc_modeling.utilities.units import (
    ureg, Q_,
    to_si, as_qty,
    U_T, U_P, U_S, U_H, U_RHO, U_MU, U_CP, U_CV, U_A
)

NumberOrQty = Union[float, int, pint.Quantity]

__all__ = [
    "ureg", "Q_", "make_fluid", "Fluid",
    "FluidBackend", "FluidAPI", "FluidSpec",
]


@dataclass
class Fluid:
    backend: FluidBackend
    return_quantity: bool = True

    @property
    def fluid_id(self) -> str | tuple[str, ...]:
        return self.backend.fluid_spec.ids[0] if self.backend.fluid_spec.is_pure else self.backend.fluid_spec.ids

    @property
    def fluid_spec(self) -> FluidSpec:
        return self.backend.fluid_spec

    # ---- saturation ----
    def p_sat(self, T: NumberOrQty):
        T_K = to_si(T, U_T)
        return as_qty(self.backend.p_sat(T_K), U_P, self.return_quantity)

    def T_sat(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.T_sat(P_Pa), U_T, self.return_quantity)

    def s_sat_liq(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.s_sat_liq(P_Pa), U_S, self.return_quantity)

    def s_sat_vap(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.s_sat_vap(P_Pa), U_S, self.return_quantity)

    def s_fg(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.s_fg(P_Pa), U_S, self.return_quantity)

    def h_sat_liq(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.h_sat_liq(P_Pa), U_H, self.return_quantity)

    def h_sat_vap(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.h_sat_vap(P_Pa), U_H, self.return_quantity)

    def h_fg(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.h_fg(P_Pa), U_H, self.return_quantity)

    def p_vap(self, T: NumberOrQty):
        T_K = to_si(T, U_T)
        return as_qty(self.backend.p_vap(T_K), U_P, self.return_quantity)

    # ---- point props ----
    def s(self, T: NumberOrQty, P: NumberOrQty):
        T_K = to_si(T, U_T); P_Pa = to_si(P, U_P)
        return as_qty(self.backend.s(T_K, P_Pa), U_S, self.return_quantity)

    def h(self, T: NumberOrQty, P: NumberOrQty):
        T_K = to_si(T, U_T); P_Pa = to_si(P, U_P)
        return as_qty(self.backend.h(T_K, P_Pa), U_H, self.return_quantity)

    def rho(self, T: NumberOrQty, P: NumberOrQty):
        T_K = to_si(T, U_T); P_Pa = to_si(P, U_P)
        return as_qty(self.backend.rho(T_K, P_Pa), U_RHO, self.return_quantity)

    def mu(self, T: NumberOrQty, P: NumberOrQty):
        T_K = to_si(T, U_T); P_Pa = to_si(P, U_P)
        return as_qty(self.backend.mu(T_K, P_Pa), U_MU, self.return_quantity)

    def a(self, T: NumberOrQty, P: NumberOrQty):
        T_K = to_si(T, U_T); P_Pa = to_si(P, U_P)
        return as_qty(self.backend.a(T_K, P_Pa), U_A, self.return_quantity)

    def rho_sat_liq(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.rho_sat_liq(P_Pa), U_RHO, self.return_quantity)

    def rho_sat_vap(self, P: NumberOrQty):
        P_Pa = to_si(P, U_P)
        return as_qty(self.backend.rho_sat_vap(P_Pa), U_RHO, self.return_quantity)

    def cp(self, T: NumberOrQty, P: NumberOrQty):
        T_K = to_si(T, U_T); P_Pa = to_si(P, U_P)
        return as_qty(self.backend.cp(T_K, P_Pa), U_CP, self.return_quantity)

    def cv(self, T: NumberOrQty, P: NumberOrQty):
        T_K = to_si(T, U_T); P_Pa = to_si(P, U_P)
        return as_qty(self.backend.cv(T_K, P_Pa), U_CV, self.return_quantity)

    # ---- critical point ----
    def T_crit(self):
        return as_qty(self.backend.T_crit(), U_T, self.return_quantity)

    def p_crit(self):
        return as_qty(self.backend.p_crit(), U_P, self.return_quantity)


def make_fluid(
    fluid_id: str | Sequence[str],
    backend: str = "thermo",
    composition: Sequence[float] | None = None,
    composition_basis: str = "mole",
    return_quantity: bool = True,
) -> Fluid:
    spec = make_spec(fluid_id, composition=composition, composition_basis=composition_basis)
    backend = backend.lower().strip()

    if backend == "thermo":
        from .thermo_backend import ThermoBackend
        b = ThermoBackend(spec)
    elif backend == "refprop":
        from .refprop_backend import RefpropBackend
        b = RefpropBackend(spec)
    elif backend == "coolprop":
        from .coolprop_backend import CoolPropBackend
        b = CoolPropBackend(spec)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return Fluid(backend=b, return_quantity=return_quantity)