from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from orc_modeling.utilities.units import Q_
from orc_modeling.pumps.curves import PumpCurve


@dataclass(frozen=True)
class ImportedPumpCurve:
    curve: PumpCurve
    N_ref_rpm: float
    source_path: str


def import_pump_curve_from_excel_points(
    path: str,
    *,
    sheet_name: Optional[str] = None,
    col_flow_gpm: str = "Flow_gpm",
    col_head_ft: str = "Head_ft",
    col_eff: str = "Eff",
    col_power_hp: str = "Power_hp",
    col_npshr_ft: str = "NPSHr_ft",
    col_speed_rpm: str = "Speed_rpm",
    use_scipy_pchip: bool = True,
    N_ref_rpm_override: Optional[float] = None,
) -> ImportedPumpCurve:
    df = pd.read_excel(path, sheet_name=sheet_name)

    required = [col_flow_gpm, col_head_ft, col_speed_rpm]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c!r}. Found: {list(df.columns)}")

    d = df.copy()

    # Drop rows missing essential data
    d = d.dropna(subset=[col_flow_gpm, col_head_ft, col_speed_rpm])

    # Determine reference speed from the file
    speeds = sorted({float(x) for x in d[col_speed_rpm].to_list()})
    if len(speeds) != 1:
        raise ValueError(f"Expected a single reference speed in {col_speed_rpm}, found: {speeds}")
    N_ref_file = float(speeds[0])

    N_ref = float(N_ref_rpm_override) if N_ref_rpm_override is not None else N_ref_file
    if abs(N_ref - N_ref_file) > 1e-6:
        raise ValueError(f"N_ref mismatch: file has {N_ref_file} rpm but override is {N_ref} rpm")

    # Sort by flow
    d = d.sort_values(by=col_flow_gpm)

    # Convert to SI for PumpCurve
    Q_m3_s = [float(Q_(float(x), "gallon/minute").to("m^3/s").magnitude) for x in d[col_flow_gpm].to_list()]
    H_m = [float(Q_(float(x), "ft").to("m").magnitude) for x in d[col_head_ft].to_list()]

    eta = None
    if col_eff in d.columns and d[col_eff].notna().any():
        eta = [float(x) for x in d[col_eff].to_list()]

    power_W = None
    if col_power_hp in d.columns and d[col_power_hp].notna().any():
        power_W = [float(Q_(float(x), "horsepower").to("W").magnitude) for x in d[col_power_hp].to_list()]

    npshr_m = None
    if col_npshr_ft in d.columns and d[col_npshr_ft].notna().any():
        npshr_m = [float(Q_(float(x), "ft").to("m").magnitude) for x in d[col_npshr_ft].to_list()]

    curve = PumpCurve.from_points(
        Q_m3_s,
        H_m,
        N_ref_rpm=N_ref,
        eta=eta,
        power_W=power_W,
        npshr_m=npshr_m,
        use_scipy_pchip=use_scipy_pchip,
    )

    return ImportedPumpCurve(curve=curve, N_ref_rpm=N_ref, source_path=path)