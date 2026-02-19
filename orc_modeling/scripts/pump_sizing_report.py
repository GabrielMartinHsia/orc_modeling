from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from orc_modeling.pumps.curves import PumpCurve


@dataclass(frozen=True)
class Scenario:
    ambient_label: str
    Q_m3_s: float          # target volumetric flow
    H_m: float             # required head (system requirement)
    rho_kg_m3: float = 1000.0  # use water-ish default until fluidprops is wired


def hydraulic_power_kW(rho: float, g: float, Q: float, H: float) -> float:
    # P = rho*g*Q*H
    return (rho * g * Q * H) / 1000.0


def specific_speed_metric(N_rpm: float, Q_m3_s: float, H_m: float) -> float:
    # Metric specific speed (common form):
    # Ns = N * sqrt(Q) / H^(3/4)
    # Units: N in rpm, Q in m^3/s, H in m
    if H_m <= 0:
        return float("nan")
    return N_rpm * math.sqrt(max(Q_m3_s, 0.0)) / (H_m ** 0.75)


def impeller_family_from_Ns(Ns: float) -> str:
    # Very rough buckets (tune later to your preferred convention)
    if math.isnan(Ns):
        return "unknown"
    if Ns < 40:
        return "radial"
    if Ns < 120:
        return "mixed"
    return "axial"


def main():
    # --- 1) Define a pump curve from vendor points (example placeholders) ---
    # You should replace these with real points (Q, H, eta) at N_ref_rpm.
    N_ref_rpm = 3600.0
    Q_points = [0.02, 0.04, 0.06, 0.08]   # m^3/s
    H_points = [120, 110, 90, 60]         # m
    eta_points = [0.55, 0.70, 0.72, 0.60] # fraction

    curve = PumpCurve.from_points(
        N_ref_rpm=N_ref_rpm,
        Q_m3_s=Q_points,
        H_m=H_points,
        eta=eta_points,
    )

    # --- 2) Define “ambient conditions” as simple scenarios for now ---
    scenarios: list[Scenario] = [
        Scenario(ambient_label="cold", Q_m3_s=0.06, H_m=95),
        Scenario(ambient_label="hot",  Q_m3_s=0.06, H_m=105),
    ]

    # --- 3) Sweep configurations ---
    pump_counts = [1, 2, 3]
    speeds_rpm = [1800.0, 2400.0, 3000.0, 3600.0]  # or generate

    g = 9.80665

    for sc in scenarios:
        print(f"\n=== Ambient: {sc.ambient_label} | Q={sc.Q_m3_s:.4f} m3/s | H_req={sc.H_m:.1f} m ===")
        print("Npumps  N_rpm  H_avail(m)  margin(m)  P_hyd(kW)  Ns      impeller")

        for n_pumps in pump_counts:
            # In parallel: each pump sees Q / n at same head.
            Q_per = sc.Q_m3_s / n_pumps

            for N in speeds_rpm:
                H_avail = curve.H(Q_m3_s=Q_per, N_rpm=N)
                margin = H_avail - sc.H_m
                P_hyd = hydraulic_power_kW(sc.rho_kg_m3, g, sc.Q_m3_s, sc.H_m)

                Ns = specific_speed_metric(N, Q_per, sc.H_m)
                fam = impeller_family_from_Ns(Ns)

                print(f"{n_pumps:>6d}  {N:>5.0f}  {H_avail:>9.1f}  {margin:>8.1f}  {P_hyd:>9.1f}  {Ns:>6.1f}  {fam}")

if __name__ == "__main__":
    main()
