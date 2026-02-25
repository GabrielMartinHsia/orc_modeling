from __future__ import annotations

from orc_modeling.io.pump_curves import import_pump_curve_from_excel_points
from orc_modeling.viz.pumpcurves_plotly import plot_pump_and_bank_us


def main():
    imp = import_pump_curve_from_excel_points(
        # "your_curve.xlsx",
        "Cape2_ORC_FeedPumpCurveData.xlsx",
        sheet_name="Sheet1",
        col_flow_gpm="Flow_gpm",
        col_head_ft="Head_ft",
        col_eff="Eff",
        col_power_hp="Power_hp",
        col_npshr_ft="NPSHr_ft",
        col_speed_rpm="Speed_rpm",
    )

    fig = plot_pump_and_bank_us(
        curve=imp.curve,
        N_fixed_rpm=1780.0,
        n_fixed=1,
        n_vfd=1,
        N_vfd_rpm=1600.0,
        title="Vendor curve + affinity scaling + fixed/VFD parallel bank (US units)",
    )
    fig.show()


if __name__ == "__main__":
    main()