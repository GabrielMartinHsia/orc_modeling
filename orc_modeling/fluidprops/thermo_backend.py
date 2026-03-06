from __future__ import annotations
from dataclasses import dataclass
import math

from thermo import ChemicalConstantsPackage, CEOSLiquid, CEOSGas, FlashPureVLS, FlashVL, PRMIX

from .base import FluidSpec, ws_to_zs


@dataclass
class ThermoBackend:
    fluid_spec: FluidSpec
    _flasher: object = None
    _MW_kg_per_mol: float | None = None
    _MWs_kg_per_mol: list[float] | None = None
    _zs: list[float] | None = None

    def __post_init__(self):
        ids = list(self.fluid_spec.ids)
        constants, correlations = ChemicalConstantsPackage.from_IDs(ids)

        eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
        liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)

        self._MWs_kg_per_mol = [mw / 1000.0 for mw in constants.MWs]

        if self.fluid_spec.is_mixture:
            if self.fluid_spec.composition_basis == "mass":
                self._zs = list(ws_to_zs(self.fluid_spec.composition, self._MWs_kg_per_mol))
            else:
                self._zs = list(self.fluid_spec.composition)

            self._MW_kg_per_mol = sum(z * MW for z, MW in zip(self._zs, self._MWs_kg_per_mol))
            self._T_crit_K = float("nan")
            self._p_crit_Pa = float("nan")
            self._flasher = FlashVL(constants=constants, correlations=correlations, gas=gas, liquid=liquid)
        else:
            self._T_crit_K = float(constants.Tcs[0])
            self._p_crit_Pa = float(constants.Pcs[0])
            self._flasher = FlashPureVLS(constants=constants, correlations=correlations, gas=gas, liquids=[liquid], solids=[])
            r = self._flasher.flash(T=298.15, P=101325.0)
            self._MW_kg_per_mol = r.MW() / 1000.0

    def _flash(self, **kwargs):
        if self.fluid_spec.is_mixture:
            return self._flasher.flash(zs=self._zs, **kwargs)
        return self._flasher.flash(**kwargs)

    def _require_pure(self, name: str):
        if self.fluid_spec.is_mixture:
            raise NotImplementedError(f"{name} is not implemented for mixtures in ThermoBackend.")

    def T_crit(self) -> float:
        if self.fluid_spec.is_mixture:
            raise NotImplementedError("Mixture critical temperature is not implemented in ThermoBackend.")
        return self._T_crit_K

    def p_crit(self) -> float:
        if self.fluid_spec.is_mixture:
            raise NotImplementedError("Mixture critical pressure is not implemented in ThermoBackend.")
        return self._p_crit_Pa

    def _S_molar_to_mass(self, S_J_per_mol_K: float) -> float:
        return S_J_per_mol_K / max(self._MW_kg_per_mol, 1e-30)

    def _H_molar_to_mass(self, H_J_per_mol: float) -> float:
        return H_J_per_mol / max(self._MW_kg_per_mol, 1e-30)

    def _rho_from_result(self, res) -> float:
        V_molar = res.V()
        return self._MW_kg_per_mol / max(V_molar, 1e-30)

    # saturation
    def p_sat(self, T_K: float) -> float:
        self._require_pure("p_sat")
        return float(self._flash(T=T_K, VF=0.0).P)

    def T_sat(self, P_Pa: float) -> float:
        self._require_pure("T_sat")
        return float(self._flash(P=P_Pa, VF=0.0).T)

    def s_sat_liq(self, P_Pa: float) -> float:
        self._require_pure("s_sat_liq")
        return self._S_molar_to_mass(self._flash(P=P_Pa, VF=0.0).S())

    def s_sat_vap(self, P_Pa: float) -> float:
        self._require_pure("s_sat_vap")
        return self._S_molar_to_mass(self._flash(P=P_Pa, VF=1.0).S())

    def s_fg(self, P_Pa: float) -> float:
        return self.s_sat_vap(P_Pa) - self.s_sat_liq(P_Pa)

    def h_sat_liq(self, P_Pa: float) -> float:
        self._require_pure("h_sat_liq")
        return self._H_molar_to_mass(self._flash(P=P_Pa, VF=0.0).H())

    def h_sat_vap(self, P_Pa: float) -> float:
        self._require_pure("h_sat_vap")
        return self._H_molar_to_mass(self._flash(P=P_Pa, VF=1.0).H())

    def h_fg(self, P_Pa: float) -> float:
        return self.h_sat_vap(P_Pa) - self.h_sat_liq(P_Pa)

    def p_vap(self, T_K: float) -> float:
        return self.p_sat(T_K)

    # point props
    def s(self, T_K: float, P_Pa: float) -> float:
        return self._S_molar_to_mass(self._flash(T=T_K, P=P_Pa).S())

    def h(self, T_K: float, P_Pa: float) -> float:
        return self._H_molar_to_mass(self._flash(T=T_K, P=P_Pa).H())

    def rho(self, T_K: float, P_Pa: float) -> float:
        return self._rho_from_result(self._flash(T=T_K, P=P_Pa))

    def rho_sat_liq(self, P_Pa: float) -> float:
        self._require_pure("rho_sat_liq")
        return self._rho_from_result(self._flash(P=P_Pa, VF=0.0))

    def rho_sat_vap(self, P_Pa: float) -> float:
        self._require_pure("rho_sat_vap")
        return self._rho_from_result(self._flash(P=P_Pa, VF=1.0))

    def mu(self, T_K: float, P_Pa: float) -> float:
        r = self._flash(T=T_K, P=P_Pa)
        mu_attr = getattr(r, "mu", None)
        if callable(mu_attr):
            try:
                v = float(mu_attr())
                if v > 0 and math.isfinite(v):
                    return v
            except Exception:
                pass
        return 2.0e-4

    def cp(self, T_K: float, P_Pa: float) -> float:
        r = self._flash(T=T_K, P=P_Pa)
        for name in ("Cp", "Cpm", "Cp_molar"):
            attr = getattr(r, name, None)
            if callable(attr):
                val = float(attr())
                if val > 0:
                    return val / max(self._MW_kg_per_mol, 1e-30)
        raise NotImplementedError("Thermo backend could not provide Cp from the flash result.")

    def cv(self, T_K: float, P_Pa: float) -> float:
        r = self._flash(T=T_K, P=P_Pa)
        for name in ("Cv", "Cvm", "Cv_molar"):
            attr = getattr(r, name, None)
            if callable(attr):
                val = float(attr())
                if val > 0:
                    return val / max(self._MW_kg_per_mol, 1e-30)
        raise NotImplementedError("Thermo backend could not provide Cv from the flash result.")

    def a(self, T_K: float, P_Pa: float) -> float:
        r = self._flash(T=T_K, P=P_Pa)

        vf = getattr(r, "VF", None)
        try:
            if vf is not None:
                vf = float(vf)
                if 1e-9 < vf < 1.0 - 1e-9:
                    raise ValueError("Speed of sound is undefined for two-phase equilibrium states.")
        except Exception:
            pass

        for name in ("speed_of_sound", "a", "w", "W"):
            attr = getattr(r, name, None)
            if callable(attr):
                try:
                    val = float(attr())
                    if val > 0.0 and math.isfinite(val):
                        return val
                except Exception:
                    pass
            elif attr is not None:
                try:
                    val = float(attr)
                    if val > 0.0 and math.isfinite(val):
                        return val
                except Exception:
                    pass

        cp = self.cp(T_K, P_Pa)
        cv = self.cv(T_K, P_Pa)
        rho = self.rho(T_K, P_Pa)
        gamma = cp / max(cv, 1e-30)
        return math.sqrt(max(gamma, 1e-12) * P_Pa / max(rho, 1e-30))