from __future__ import annotations
from dataclasses import dataclass
import math

from thermo import ChemicalConstantsPackage, CEOSLiquid, CEOSGas, FlashPureVLS, PRMIX


@dataclass
class ThermoBackend:
    fluid_id: str
    _flasher: FlashPureVLS = None
    _MW_kg_per_mol: float = None

    def __post_init__(self):
        constants, correlations = ChemicalConstantsPackage.from_IDs([self.fluid_id])
        eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
        liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
        gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)

        self._T_crit_K = float(constants.Tcs[0])
        self._p_crit_Pa = float(constants.Pcs[0])

        self._flasher = FlashPureVLS(constants=constants, correlations=correlations, gas=gas, liquids=[liquid], solids=[])
        r = self._flasher.flash(T=298.15, P=101325.0)
        self._MW_kg_per_mol = r.MW() / 1000.0  # g/mol → kg/mol

    def T_crit(self) -> float:
        return self._T_crit_K
    
    def p_crit(self) -> float:
        return self._p_crit_Pa

    def _S_molar_to_mass(self, S_J_per_mol_K: float) -> float:
        return S_J_per_mol_K / max(self._MW_kg_per_mol, 1e-30)

    def _H_molar_to_mass(self, H_J_per_mol: float) -> float:
        return H_J_per_mol / max(self._MW_kg_per_mol, 1e-30)

    def _rho_from_result(self, res) -> float:
        V_molar = res.V()  # m^3/mol
        return self._MW_kg_per_mol / max(V_molar, 1e-30)

    # saturation
    def p_sat(self, T_K: float) -> float:
        return float(self._flasher.flash(T=T_K, VF=0.0).P)

    def T_sat(self, P_Pa: float) -> float:
        return float(self._flasher.flash(P=P_Pa, VF=0.0).T)

    def s_sat_liq(self, P_Pa: float) -> float:
        r = self._flasher.flash(P=P_Pa, VF=0.0)
        return self._S_molar_to_mass(r.S())

    def s_sat_vap(self, P_Pa: float) -> float:
        r = self._flasher.flash(P=P_Pa, VF=1.0)
        return self._S_molar_to_mass(r.S())

    def s_fg(self, P_Pa: float) -> float:
        return self.s_sat_vap(P_Pa) - self.s_sat_liq(P_Pa)

    def h_sat_liq(self, P_Pa: float) -> float:
        r = self._flasher.flash(P=P_Pa, VF=0.0)
        return self._H_molar_to_mass(r.H())

    def h_sat_vap(self, P_Pa: float) -> float:
        r = self._flasher.flash(P=P_Pa, VF=1.0)
        return self._H_molar_to_mass(r.H())

    def h_fg(self, P_Pa: float) -> float:
        return self.h_sat_vap(P_Pa) - self.h_sat_liq(P_Pa)

    def p_vap(self, T_K: float) -> float:
        return self.p_sat(T_K)

    # point props
    def s(self, T_K: float, P_Pa: float) -> float:
        r = self._flasher.flash(T=T_K, P=P_Pa)
        return self._S_molar_to_mass(r.S())

    def h(self, T_K: float, P_Pa: float) -> float:
        r = self._flasher.flash(T=T_K, P=P_Pa)
        return self._H_molar_to_mass(r.H())

    def rho(self, T_K: float, P_Pa: float) -> float:
        r = self._flasher.flash(T=T_K, P=P_Pa)
        return self._rho_from_result(r)

    def rho_sat_liq(self, P_Pa: float) -> float:
        return self._rho_from_result(self._flasher.flash(P=P_Pa, VF=0.0))

    def rho_sat_vap(self, P_Pa: float) -> float:
        return self._rho_from_result(self._flasher.flash(P=P_Pa, VF=1.0))

    def mu(self, T_K: float, P_Pa: float) -> float:
        r = self._flasher.flash(T=T_K, P=P_Pa)
        mu_attr = getattr(r, "mu", None)
        if callable(mu_attr):
            try:
                v = float(mu_attr())
                if v > 0 and math.isfinite(v):
                    return v
            except Exception:
                pass
        return 2.0e-4  # fallback Pa*s
    
    def cp(self, T_K: float, P_Pa: float) -> float:
        """
        Mass-basis Cp at (T,P), returned as J/kg/K.

        Thermo flash result Cp() is typically molar-basis J/mol/K, so we convert to mass basis
        by dividing by MW [kg/mol].
        """
        r = self._flasher.flash(T=T_K, P=P_Pa)

        # Try common Thermo attribute names
        for name in ("Cp", "Cpm", "Cp_molar"):
            attr = getattr(r, name, None)
            if callable(attr):
                val = float(attr())
                if val > 0:
                    # assume J/mol/K -> J/kg/K
                    return val / max(self._MW_kg_per_mol, 1e-30)

        raise NotImplementedError(
            "Thermo backend could not provide Cp from the flash result; check Thermo version/model."
        )

    def cv(self, T_K: float, P_Pa: float) -> float:
        """
        Mass-basis Cv at (T,P), returned as J/kg/K.

        Thermo flash result Cv() is typically molar-basis J/mol/K, so we convert to mass basis
        by dividing by MW [kg/mol].
        """
        r = self._flasher.flash(T=T_K, P=P_Pa)

        # Try common Thermo attribute names
        for name in ("Cv", "Cvm", "Cv_molar"):
            attr = getattr(r, name, None)
            if callable(attr):
                val = float(attr())
                if val > 0:
                    # assume J/mol/K -> J/kg/K
                    return val / max(self._MW_kg_per_mol, 1e-30)

        raise NotImplementedError(
            "Thermo backend could not provide Cv from the flash result; check Thermo version/model."
        )
