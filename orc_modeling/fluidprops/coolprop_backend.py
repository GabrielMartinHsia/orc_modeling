from __future__ import annotations

from dataclasses import dataclass

try:
    import CoolProp.CoolProp as CP
except Exception as e:
    CP = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

from .base import FluidSpec, zs_to_ws


def _norm_name(name: str) -> str:
    return name.lower().strip().replace("_", " ")


def _same_components(ids: tuple[str, ...], a: str, b: str) -> bool:
    return set(_norm_name(x) for x in ids) == {_norm_name(a), _norm_name(b)}


def _coolprop_pure_adapter(spec: FluidSpec) -> tuple[str, bool] | None:
    if spec.is_pure:
        return spec.ids[0], False
    return None


def _coolprop_incomp_meg_adapter(spec: FluidSpec) -> tuple[str, bool] | None:
    """
    Translate canonical water/ethylene-glycol mixture spec into a CoolProp
    incompressible MEG fluid string.

    Returns:
        (fluid_string, is_incompressible) or None if not applicable
    """
    if not spec.is_mixture:
        return None

    if not _same_components(spec.ids, "water", "ethylene glycol"):
        return None

    ids_n = tuple(_norm_name(x) for x in spec.ids)

    if spec.composition_basis == "mass":
        ws = tuple(spec.composition)
    else:
        mw_map = {
            "water": 0.01801528,
            "ethylene glycol": 0.06206784,
        }
        MWs = [mw_map[x] for x in ids_n]
        ws = zs_to_ws(spec.composition, MWs)

    w_meg = ws[0] if ids_n[0] == "ethylene glycol" else ws[1]

    # CoolProp docs show incompressible mixture strings using bracket fraction syntax.
    return f"INCOMP::MEG[{w_meg:.12g}]", True


def _build_coolprop_target(spec: FluidSpec) -> tuple[str, bool]:
    adapters = (
        _coolprop_pure_adapter,
        _coolprop_incomp_meg_adapter,
    )

    for adapter in adapters:
        out = adapter(spec)
        if out is not None:
            return out

    raise NotImplementedError(f"CoolPropBackend does not support mixture spec {spec.ids!r}.")


@dataclass
class CoolPropBackend:
    fluid_spec: FluidSpec

    def __post_init__(self):
        if CP is None:
            raise ImportError(
                "CoolProp is not available. Install with `conda install -c conda-forge coolprop` "
                "or `pip install CoolProp`."
            ) from _IMPORT_ERROR

        self._f, self._is_incompressible = _build_coolprop_target(self.fluid_spec)

    def _props(self, out: str, in1: str, val1: float, in2: str, val2: float) -> float:
        return float(CP.PropsSI(out, in1, float(val1), in2, float(val2), self._f))

    def _require_not_incompressible(self, name: str):
        if self._is_incompressible:
            raise NotImplementedError(f"{name} is not implemented for CoolProp incompressible mixtures.")

    def T_crit(self) -> float:
        self._require_not_incompressible("T_crit")
        return float(CP.PropsSI("Tcrit", self._f))

    def p_crit(self) -> float:
        self._require_not_incompressible("p_crit")
        return float(CP.PropsSI("pcrit", self._f))

    def p_sat(self, T_K: float) -> float:
        self._require_not_incompressible("p_sat")
        return self._props("P", "T", T_K, "Q", 0.0)

    def T_sat(self, P_Pa: float) -> float:
        self._require_not_incompressible("T_sat")
        return self._props("T", "P", P_Pa, "Q", 0.0)

    def p_vap(self, T_K: float) -> float:
        return self.p_sat(T_K)

    def s_sat_liq(self, P_Pa: float) -> float:
        self._require_not_incompressible("s_sat_liq")
        return self._props("S", "P", P_Pa, "Q", 0.0)

    def s_sat_vap(self, P_Pa: float) -> float:
        self._require_not_incompressible("s_sat_vap")
        return self._props("S", "P", P_Pa, "Q", 1.0)

    def s_fg(self, P_Pa: float) -> float:
        return self.s_sat_vap(P_Pa) - self.s_sat_liq(P_Pa)

    def h_sat_liq(self, P_Pa: float) -> float:
        self._require_not_incompressible("h_sat_liq")
        return self._props("H", "P", P_Pa, "Q", 0.0)

    def h_sat_vap(self, P_Pa: float) -> float:
        self._require_not_incompressible("h_sat_vap")
        return self._props("H", "P", P_Pa, "Q", 1.0)

    def h_fg(self, P_Pa: float) -> float:
        return self.h_sat_vap(P_Pa) - self.h_sat_liq(P_Pa)

    def rho(self, T_K: float, P_Pa: float) -> float:
        return self._props("D", "T", T_K, "P", P_Pa)

    def h(self, T_K: float, P_Pa: float) -> float:
        return self._props("H", "T", T_K, "P", P_Pa)

    def s(self, T_K: float, P_Pa: float) -> float:
        return self._props("S", "T", T_K, "P", P_Pa)

    def mu(self, T_K: float, P_Pa: float) -> float:
        return self._props("V", "T", T_K, "P", P_Pa)

    def cp(self, T_K: float, P_Pa: float) -> float:
        return self._props("C", "T", T_K, "P", P_Pa)

    def cv(self, T_K: float, P_Pa: float) -> float:
        self._require_not_incompressible("cv")
        return self._props("O", "T", T_K, "P", P_Pa)

    def a(self, T_K: float, P_Pa: float) -> float:
        self._require_not_incompressible("a")
        return self._props("A", "T", T_K, "P", P_Pa)

    def rho_sat_liq(self, P_Pa: float) -> float:
        self._require_not_incompressible("rho_sat_liq")
        return self._props("D", "P", P_Pa, "Q", 0.0)

    def rho_sat_vap(self, P_Pa: float) -> float:
        self._require_not_incompressible("rho_sat_vap")
        return self._props("D", "P", P_Pa, "Q", 1.0)