# orc_modeling/pumps/sizing.py
from __future__ import annotations

import math
from typing import Optional

import pint

from orc_modeling.utilities.units import NumberOrQty, Q_, ureg

# --- constants (physics only) ---
G_STD = 9.80665  # m/s^2
RPM_TO_RAD_S = 2.0 * math.pi / 60.0


# -------------------------
# Unit helper (SI floats or Pint Quantities)
# -------------------------

def si_to_unit(value: NumberOrQty, unit_si, unit_out) -> float:
    """
    Convert:
      - float/int assumed in unit_si
      - or pint.Quantity
    into a float magnitude in unit_out.

    Example:
      si_to_unit(0.05, ureg("m^3/s"), ureg("gallon/minute")) -> gpm (float)
      si_to_unit(5 * ureg("m"), ureg.meter, ureg.ft) -> ft (float)
    """
    if isinstance(value, pint.Quantity):
        return float(value.to(unit_out).magnitude)
    return float(Q_(float(value), unit_si).to(unit_out).magnitude)


# -------------------------
# Power (SI)
# -------------------------

def hydraulic_power_W(*, rho_kg_m3: float, Q_m3_s: float, H_m: float, g_m_s2: float = G_STD) -> float:
    """Hydraulic power: P = rho * g * Q * H  [W]."""
    return float(rho_kg_m3) * float(g_m_s2) * float(Q_m3_s) * float(H_m)


def hydraulic_power_kW(*, rho_kg_m3: float, Q_m3_s: float, H_m: float, g_m_s2: float = G_STD) -> float:
    """Hydraulic power [kW]."""
    return hydraulic_power_W(rho_kg_m3=rho_kg_m3, Q_m3_s=Q_m3_s, H_m=H_m, g_m_s2=g_m_s2) / 1000.0


def shaft_power_W(*, P_hyd_W: float, eta: float) -> float:
    """Shaft power from hydraulic power and efficiency. eta in (0, 1]."""
    e = float(eta)
    if e <= 0.0:
        raise ValueError("eta must be > 0")
    return float(P_hyd_W) / e


# -------------------------
# Specific speed (conventional indices + dimensionless)
# -------------------------

def specific_speed_metric(*, N_rpm: float, Q_m3_s: float, H_m: float) -> float:
    """
    Conventional 'metric form' specific speed:
        Ns = N * sqrt(Q) / H^(3/4)
    N [rpm], Q [m^3/s], H [m]
    (Not dimensionless; engineering index.)
    """
    H = float(H_m)
    if H <= 0.0:
        return float("nan")
    return float(N_rpm) * math.sqrt(max(float(Q_m3_s), 0.0)) / (H ** 0.75)


def specific_speed_us(*, N_rpm: float, Q_gpm: float, H_ft: float) -> float:
    """
    Conventional 'US customary' specific speed:
        Ns = N * sqrt(Q) / H^(3/4)
    N [rpm], Q [gpm], H [ft]
    (Not dimensionless; engineering index.)
    """
    H = float(H_ft)
    if H <= 0.0:
        return float("nan")
    return float(N_rpm) * math.sqrt(max(float(Q_gpm), 0.0)) / (H ** 0.75)


def specific_speed_dimensionless(*, N_rpm: float, Q_m3_s: float, H_m: float, g_m_s2: float = G_STD) -> float:
    """
    True dimensionless specific speed (often written ω_s):
        ω_s = ω * sqrt(Q) / (g*H)^(3/4)
    ω [rad/s], Q [m^3/s], H [m]
    """
    H = float(H_m)
    if H <= 0.0:
        return float("nan")
    omega = float(N_rpm) * RPM_TO_RAD_S
    denom = (float(g_m_s2) * H) ** 0.75
    return omega * math.sqrt(max(float(Q_m3_s), 0.0)) / denom


# -------------------------
# Suction specific speed (NPSH)
# -------------------------

def suction_specific_speed_us(*, N_rpm: float, Q_gpm: float, NPSHr_ft: float) -> float:
    """
    Conventional US suction specific speed:
        Nss = N * sqrt(Q_gpm) / NPSHr_ft^(3/4)
    """
    Hs = float(NPSHr_ft)
    if Hs <= 0.0:
        return float("nan")
    return float(N_rpm) * math.sqrt(max(float(Q_gpm), 0.0)) / (Hs ** 0.75)


def suction_specific_speed_dimensionless(
    *, N_rpm: float, Q_m3_s: float, NPSHr_m: float, g_m_s2: float = G_STD
) -> float:
    """
    True dimensionless suction specific speed:
        ω_ss = ω * sqrt(Q) / (g*NPSHr)^(3/4)
    """
    Hs = float(NPSHr_m)
    if Hs <= 0.0:
        return float("nan")
    omega = float(N_rpm) * RPM_TO_RAD_S
    denom = (float(g_m_s2) * Hs) ** 0.75
    return omega * math.sqrt(max(float(Q_m3_s), 0.0)) / denom


# -------------------------
# Non-dimensional coefficients (require impeller diameter D)
# -------------------------

def flow_coefficient_phi(*, Q_m3_s: float, D_m: float, N_rpm: float) -> float:
    """φ = Q / (ω * D^3)"""
    D = float(D_m)
    if D <= 0.0:
        return float("nan")
    omega = float(N_rpm) * RPM_TO_RAD_S
    return float(Q_m3_s) / (omega * D**3)


def head_coefficient_psi(*, H_m: float, D_m: float, N_rpm: float, g_m_s2: float = G_STD) -> float:
    """ψ = g*H / (ω^2 * D^2)"""
    D = float(D_m)
    if D <= 0.0:
        return float("nan")
    omega = float(N_rpm) * RPM_TO_RAD_S
    return (float(g_m_s2) * float(H_m)) / (omega**2 * D**2)


def power_coefficient_lambda(*, P_W: float, rho_kg_m3: float, D_m: float, N_rpm: float) -> float:
    """λ = P / (rho * ω^3 * D^5)"""
    D = float(D_m)
    if D <= 0.0:
        return float("nan")
    omega = float(N_rpm) * RPM_TO_RAD_S
    return float(P_W) / (float(rho_kg_m3) * omega**3 * D**5)


# -------------------------
# Convenience "from SI" helpers (floats assumed SI or Pint quantities)
# -------------------------

def specific_speed_us_from_si(*, N_rpm: float, Q_m3_s: NumberOrQty, H_m: NumberOrQty) -> float:
    """US specific speed, accepting SI float/Qty for Q and H."""
    Q_gpm = si_to_unit(Q_m3_s, ureg("m^3/s"), ureg("gallon/minute"))
    H_ft = si_to_unit(H_m, ureg.meter, ureg.ft)
    return specific_speed_us(N_rpm=N_rpm, Q_gpm=Q_gpm, H_ft=H_ft)


def suction_specific_speed_us_from_si(*, N_rpm: float, Q_m3_s: NumberOrQty, NPSHr_m: NumberOrQty) -> float:
    """US suction specific speed, accepting SI float/Qty for Q and NPSHr."""
    Q_gpm = si_to_unit(Q_m3_s, ureg("m^3/s"), ureg("gallon/minute"))
    NPSHr_ft = si_to_unit(NPSHr_m, ureg.meter, ureg.ft)
    return suction_specific_speed_us(N_rpm=N_rpm, Q_gpm=Q_gpm, NPSHr_ft=NPSHr_ft)
