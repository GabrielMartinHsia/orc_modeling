from __future__ import annotations

from dataclasses import dataclass

try:
    import CoolProp.CoolProp as CP
except Exception as e:
    CP = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass
class CoolPropBackend:
    """
    SI-float backend using CoolProp PropsSI (mass-basis SI).

    Inputs/outputs:
      T [K], P [Pa], h [J/kg], s [J/kg/K], rho [kg/m3], mu [Pa*s], cp/cv [J/kg/K], a [m/s]
    """
    fluid_id: str

    def __post_init__(self):
        if CP is None:
            raise ImportError(
                "CoolProp is not available. Install with `conda install -c conda-forge coolprop` "
                "or `pip install CoolProp`."
            ) from _IMPORT_ERROR
        self._f = self.fluid_id

    # ---- critical ----
    def T_crit(self) -> float:
        return float(CP.PropsSI("Tcrit", self._f))

    def p_crit(self) -> float:
        return float(CP.PropsSI("pcrit", self._f))

    # ---- saturation ----
    def p_sat(self, T_K: float) -> float:
        return float(CP.PropsSI("P", "T", float(T_K), "Q", 0.0, self._f))

    def T_sat(self, P_Pa: float) -> float:
        return float(CP.PropsSI("T", "P", float(P_Pa), "Q", 0.0, self._f))

    def p_vap(self, T_K: float) -> float:
        return self.p_sat(T_K)

    def s_sat_liq(self, P_Pa: float) -> float:
        return float(CP.PropsSI("S", "P", float(P_Pa), "Q", 0.0, self._f))

    def s_sat_vap(self, P_Pa: float) -> float:
        return float(CP.PropsSI("S", "P", float(P_Pa), "Q", 1.0, self._f))

    def s_fg(self, P_Pa: float) -> float:
        return self.s_sat_vap(P_Pa) - self.s_sat_liq(P_Pa)

    def h_sat_liq(self, P_Pa: float) -> float:
        return float(CP.PropsSI("H", "P", float(P_Pa), "Q", 0.0, self._f))

    def h_sat_vap(self, P_Pa: float) -> float:
        return float(CP.PropsSI("H", "P", float(P_Pa), "Q", 1.0, self._f))

    def h_fg(self, P_Pa: float) -> float:
        return self.h_sat_vap(P_Pa) - self.h_sat_liq(P_Pa)

    def rho_sat_liq(self, P_Pa: float) -> float:
        return float(CP.PropsSI("D", "P", float(P_Pa), "Q", 0.0, self._f))

    def rho_sat_vap(self, P_Pa: float) -> float:
        return float(CP.PropsSI("D", "P", float(P_Pa), "Q", 1.0, self._f))

    # ---- point props ----
    def s(self, T_K: float, P_Pa: float) -> float:
        return float(CP.PropsSI("S", "T", float(T_K), "P", float(P_Pa), self._f))

    def h(self, T_K: float, P_Pa: float) -> float:
        return float(CP.PropsSI("H", "T", float(T_K), "P", float(P_Pa), self._f))

    def rho(self, T_K: float, P_Pa: float) -> float:
        return float(CP.PropsSI("D", "T", float(T_K), "P", float(P_Pa), self._f))

    def mu(self, T_K: float, P_Pa: float) -> float:
        # Dynamic viscosity: "V" [Pa*s] in PropsSI table :contentReference[oaicite:1]{index=1}
        return float(CP.PropsSI("V", "T", float(T_K), "P", float(P_Pa), self._f))

    def cp(self, T_K: float, P_Pa: float) -> float:
        # Cp mass: "C" [J/kg/K] :contentReference[oaicite:2]{index=2}
        return float(CP.PropsSI("C", "T", float(T_K), "P", float(P_Pa), self._f))

    def cv(self, T_K: float, P_Pa: float) -> float:
        # Cv mass: "O" [J/kg/K] :contentReference[oaicite:3]{index=3}
        return float(CP.PropsSI("O", "T", float(T_K), "P", float(P_Pa), self._f))

    def a(self, T_K: float, P_Pa: float) -> float:
        # Speed of sound: "A" [m/s] :contentReference[oaicite:4]{index=4}
        return float(CP.PropsSI("A", "T", float(T_K), "P", float(P_Pa), self._f))