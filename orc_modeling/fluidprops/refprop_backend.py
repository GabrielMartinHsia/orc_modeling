# orc_modeling/fluidprops/refprop_backend.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Note:
#   Requires:
#     pip install ctREFPROP
#   and REFPROP installed with environment variable RPPREFIX set to the install folder.
#
# See REFPROP-wrappers README for typical setup patterns. :contentReference[oaicite:1]{index=1}


# -----------------------------
# Module-level cache (per process)
# -----------------------------
_RP = None
_UNIT_ENUM_CACHE: Dict[str, int] = {}
_LOADED_FLUIDS: Dict[str, bool] = {}


def _get_rp(root: Optional[str] = None):
    """Load ctREFPROP library (cached)."""
    global _RP
    if _RP is not None:
        return _RP

    try:
        from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
    except Exception as e:
        raise ImportError(
            "ctREFPROP not importable. Install with `pip install ctREFPROP` "
            "and ensure your Python matches your REFPROP DLL architecture (64-bit)."
        ) from e

    rp_root = root or os.environ.get("RPPREFIX", None)
    if not rp_root:
        raise RuntimeError(
            "REFPROP path not set. Set environment variable RPPREFIX to your REFPROP install folder "
            "(the folder containing REFPRP64.DLL on Windows), or pass refprop_root=... to RefpropBackend."
        )

    _RP = REFPROPFunctionLibrary(rp_root)
    # Tell REFPROP where it lives (common pattern in wrapper docs) :contentReference[oaicite:2]{index=2}
    _RP.SETPATHdll(rp_root)
    return _RP


def _get_unit_enum(rp, enum_name: str) -> int:
    """Cache unit enums (like 'MASS BASE SI')."""
    if enum_name in _UNIT_ENUM_CACHE:
        return _UNIT_ENUM_CACHE[enum_name]
    r = rp.GETENUMdll(0, enum_name)
    enum_val = int(r.iEnum)
    _UNIT_ENUM_CACHE[enum_name] = enum_val
    return enum_val


def _canonical_refprop_name(fluid_id: str) -> str:
    """
    Minimal aliasing to make common IDs nicer.
    You can expand this later if your project uses different naming conventions.
    """
    key = fluid_id.strip()
    key_low = key.lower().replace(" ", "").replace("_", "")
    aliases = {
        "co2": "CO2",
        "carbondioxide": "CO2",
        "r744": "CO2",
        "water": "WATER",
        "propane": "PROPANE",
        "nitrogen": "NITROGEN",
        "oxygen": "OXYGEN",
        "isopentane": "ISOPENTANE",
        "ic5": "ISOPENTANE",
        "pentane": "PENTANE",
    }
    return aliases.get(key_low, key)


def _raise_if_error(ierr: int, herr: str, context: str):
    # REFPROP: ierr > 0 typically indicates error; ierr < 0 warnings.
    # We'll treat >0 as fatal; warnings pass through.
    if ierr > 0:
        raise RuntimeError(f"REFPROP error in {context}: ierr={ierr} herr={herr}")


@dataclass
class RefpropBackend:
    """
    SI-float backend using REFPROP via ctREFPROP.

    All inputs/outputs are SI on a MASS basis:
      T [K], P [Pa], h [J/kg], s [J/kg/K], rho [kg/m3], mu [Pa*s], cp/cv [J/kg/K]
    """
    fluid_id: str
    refprop_root: Optional[str] = None
    ref_state: str = "DEF"  # REFPROP reference state (DEF is standard; can change later)

    def __post_init__(self):
        self._rp = _get_rp(self.refprop_root)
        self._fluid = _canonical_refprop_name(self.fluid_id)

        # Prefer mass-basis SI units directly from REFPROPdll. :contentReference[oaicite:3]{index=3}
        self._IU_MASS_SI = _get_unit_enum(self._rp, "MASS BASE SI")

        # Load fluid once per process instance (safe for repeated calls)
        # SETFLUIDS is recommended for repeated evaluation patterns. :contentReference[oaicite:4]{index=4}
        if not _LOADED_FLUIDS.get(self._fluid, False):
            self._rp.SETFLUIDSdll(self._fluid)
            _LOADED_FLUIDS[self._fluid] = True

        # Set reference state (optional). Signature is a bit finicky across versions;
        # if it fails, we leave default behavior.
        self._try_setref(self.ref_state)

        # Pure fluid composition vector:
        self._z = [1.0]

        # Cache critical props (computed lazily)
        self._Tc: Optional[float] = None
        self._Pc: Optional[float] = None

    # -----------------------------
    # Low-level helpers
    # -----------------------------
    def _try_setref(self, ref_state: str) -> None:
        """
        Best-effort reference state setter.
        If REFPROP doesn't like it (or signature mismatch), we silently ignore.
        """
        try:
            # Common legacy usage: SETREFdll(hRef, iRef, z, Tref, Dref, href, sref)
            # Many users just do ("DEF", 1, z, 0,0,0,0). :contentReference[oaicite:5]{index=5}
            self._rp.SETREFdll(ref_state, 1, [1.0], 0, 0, 0, 0)
        except Exception:
            pass

    def _refprop(self, hIn: str, hOut: str, a: float, b: float) -> float:
        """
        Call REFPROPdll using the already-set fluid.
        Returns the first output as float.
        """
        r = self._rp.REFPROPdll("", hIn, hOut, self._IU_MASS_SI, 0, 0, a, b, self._z)
        _raise_if_error(int(r.ierr), str(r.herr), f"REFPROPdll({hIn}->{hOut})")
        return float(r.Output[0])

    def _refprop_multi(self, hIn: str, hOut: str, a: float, b: float, n: int) -> Tuple[float, ...]:
        r = self._rp.REFPROPdll("", hIn, hOut, self._IU_MASS_SI, 0, 0, a, b, self._z)
        _raise_if_error(int(r.ierr), str(r.herr), f"REFPROPdll({hIn}->{hOut})")
        out = tuple(float(r.Output[i]) for i in range(n))
        return out

    # -----------------------------
    # Critical point
    # -----------------------------
    def T_crit(self) -> float:
        if self._Tc is None:
            self._load_crit()
        assert self._Tc is not None
        return self._Tc

    def p_crit(self) -> float:
        if self._Pc is None:
            self._load_crit()
        assert self._Pc is not None
        return self._Pc

    def _load_crit(self) -> None:
        """
        Use CRITPdll if available; apply a small heuristic on pressure units.
        (CRITPdll pressure is often returned in kPa in some legacy conventions.)
        """
        try:
            r = self._rp.CRITPdll(self._z)
            _raise_if_error(int(r.ierr), str(r.herr), "CRITPdll")
            Tc = float(r.Tc)
            Pc = float(r.Pc)

            # Heuristic: if Pc looks like kPa, convert to Pa.
            # Typical Pc for CO2 ~ 7.38e6 Pa; in kPa that's ~ 7380.
            if Pc < 1.0e5:  # 100,000 Pa = 100 kPa
                Pc *= 1000.0

            self._Tc = Tc
            self._Pc = Pc
            return
        except Exception:
            # Fallback: approximate by asking for saturation endpoints not reliable;
            # if CRITPdll isn't available, raise.
            raise RuntimeError("Could not read critical properties from REFPROP (CRITPdll failed).")

    # -----------------------------
    # Saturation
    # -----------------------------
    def p_sat(self, T_K: float) -> float:
        # Use TQ (T, quality) to get saturation P; quality value doesn't matter for P at saturation
        return self._refprop("TQ", "P", T_K, 0.0)

    def T_sat(self, P_Pa: float) -> float:
        return self._refprop("PQ", "T", P_Pa, 0.0)

    def p_vap(self, T_K: float) -> float:
        # Alias
        return self.p_sat(T_K)

    def s_sat_liq(self, P_Pa: float) -> float:
        return self._refprop("PQ", "S", P_Pa, 0.0)

    def s_sat_vap(self, P_Pa: float) -> float:
        return self._refprop("PQ", "S", P_Pa, 1.0)

    def s_fg(self, P_Pa: float) -> float:
        return self.s_sat_vap(P_Pa) - self.s_sat_liq(P_Pa)

    def h_sat_liq(self, P_Pa: float) -> float:
        return self._refprop("PQ", "H", P_Pa, 0.0)

    def h_sat_vap(self, P_Pa: float) -> float:
        return self._refprop("PQ", "H", P_Pa, 1.0)

    def h_fg(self, P_Pa: float) -> float:
        return self.h_sat_vap(P_Pa) - self.h_sat_liq(P_Pa)

    def rho_sat_liq(self, P_Pa: float) -> float:
        return self._refprop("PQ", "D", P_Pa, 0.0)

    def rho_sat_vap(self, P_Pa: float) -> float:
        return self._refprop("PQ", "D", P_Pa, 1.0)

    # -----------------------------
    # Point properties
    # -----------------------------
    def s(self, T_K: float, P_Pa: float) -> float:
        return self._refprop("TP", "S", T_K, P_Pa)

    def h(self, T_K: float, P_Pa: float) -> float:
        return self._refprop("TP", "H", T_K, P_Pa)

    def rho(self, T_K: float, P_Pa: float) -> float:
        return self._refprop("TP", "D", T_K, P_Pa)

    def mu(self, T_K: float, P_Pa: float) -> float:
        # REFPROP output code for dynamic viscosity is commonly "VIS" in high-level interfaces. :contentReference[oaicite:6]{index=6}
        return self._refprop("TP", "VIS", T_K, P_Pa)

    def cp(self, T_K: float, P_Pa: float) -> float:
        return self._refprop("TP", "CP", T_K, P_Pa)

    def cv(self, T_K: float, P_Pa: float) -> float:
        return self._refprop("TP", "CV", T_K, P_Pa)
    
    def a(self, T_K: float, P_Pa: float) -> float:
        # REFPROP'S thermodynamic speed of sound is typically output code "W" [m/s]
        return self._refprop("TP", "W", T_K, P_Pa)
