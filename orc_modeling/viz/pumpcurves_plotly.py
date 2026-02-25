from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from orc_modeling.utilities.units import Q_
from orc_modeling.pumps.curves import PumpCurve
from orc_modeling.pumps.banks import ParallelBank


def _Q_to_gpm(Q_m3_s: np.ndarray) -> np.ndarray:
    return np.array([float(Q_(float(q), "m^3/s").to("gallon/minute").magnitude) for q in Q_m3_s], dtype=float)

def _H_to_ft(H_m: np.ndarray) -> np.ndarray:
    return np.array([float(Q_(float(h), "m").to("ft").magnitude) for h in H_m], dtype=float)

def _P_to_hp(P_W: np.ndarray) -> np.ndarray:
    return np.array([float(Q_(float(p), "W").to("horsepower").magnitude) for p in P_W], dtype=float)

def _NPSH_to_ft(N_m: np.ndarray) -> np.ndarray:
    return np.array([float(Q_(float(n), "m").to("ft").magnitude) for n in N_m], dtype=float)


def plot_pump_and_bank_us(
    *,
    curve: PumpCurve,
    N_fixed_rpm: float,
    n_fixed: int,
    n_vfd: int,
    N_vfd_rpm: float,
    title: str = "Pump curves (US units)",
    n_samples: int = 220,
) -> go.Figure:
    """
    Plots:
      - Head vs Flow: single fixed, single vfd, bank (fixed+vfd)
      - Efficiency vs Flow (if available)
      - Power vs Flow (if available)
      - NPSHr vs Flow (if available)
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Head (ft) vs Flow (gpm)", "Efficiency vs Flow (gpm)",
                        "Power (hp) vs Flow (gpm)", "NPSHr (ft) vs Flow (gpm)"),
    )

    def sample_single(N: float):
        qmin, qmax = curve.Q_bounds_at_speed(N_rpm=float(N))
        Q = np.linspace(0.0, float(qmax), n_samples)
        H = np.array([curve.H(float(q), N_rpm=float(N)) for q in Q], dtype=float)
        return Q, H

    # Head: single fixed + single vfd
    Qf, Hf = sample_single(N_fixed_rpm)
    Qv, Hv = sample_single(N_vfd_rpm)

    fig.add_trace(go.Scatter(x=_Q_to_gpm(Qf), y=_H_to_ft(Hf), mode="lines",
                             name=f"1 pump @ {int(N_fixed_rpm)} rpm"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=_Q_to_gpm(Qv), y=_H_to_ft(Hv), mode="lines",
                             name=f"1 pump @ {int(N_vfd_rpm)} rpm"),
                  row=1, col=1)

    # Bank curve: fixed + shared VFD
    bank = ParallelBank(
        curve=curve,
        n_fixed=int(n_fixed),
        n_vfd=int(n_vfd),
        N_fixed_rpm=float(N_fixed_rpm),
        N_vfd_rpm=float(N_vfd_rpm) if n_vfd > 0 else None,
    )
    H_bank = bank.head_fn()

    # Plot range: approx max bank flow at H=0
    _qmin_f, qmax_f = curve.Q_bounds_at_speed(N_rpm=float(N_fixed_rpm))
    _qmin_v, qmax_v = curve.Q_bounds_at_speed(N_rpm=float(N_vfd_rpm))
    Qmax_bank = float(n_fixed) * float(qmax_f) + float(n_vfd) * float(qmax_v)
    Qb = np.linspace(0.0, Qmax_bank, n_samples)
    Hb = np.array([float(H_bank(float(q))) for q in Qb], dtype=float)

    fig.add_trace(go.Scatter(x=_Q_to_gpm(Qb), y=_H_to_ft(Hb), mode="lines",
                             line=dict(dash="dash"),
                             name=f"Bank: {n_fixed} fixed + {n_vfd} VFD"),
                  row=1, col=1)

    # Efficiency (reference-speed map, scaled by Q_ref internally)
    if curve.eta_ref is not None:
        # plot at fixed and vfd as separate traces
        etaf = np.array([curve.eta(float(q), float(N_fixed_rpm)) for q in Qf], dtype=float)
        etav = np.array([curve.eta(float(q), float(N_vfd_rpm)) for q in Qv], dtype=float)
        fig.add_trace(go.Scatter(x=_Q_to_gpm(Qf), y=etaf, mode="lines",
                                 name=f"η @ {int(N_fixed_rpm)} rpm"),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=_Q_to_gpm(Qv), y=etav, mode="lines",
                                 name=f"η @ {int(N_vfd_rpm)} rpm"),
                      row=1, col=2)

    # Power
    if curve.P_ref_W is not None:
        Pf = np.array([curve.power_W(float(q), N_rpm=float(N_fixed_rpm)) for q in Qf], dtype=float)
        Pv = np.array([curve.power_W(float(q), N_rpm=float(N_vfd_rpm)) for q in Qv], dtype=float)
        fig.add_trace(go.Scatter(x=_Q_to_gpm(Qf), y=_P_to_hp(Pf), mode="lines",
                                 name=f"HP @ {int(N_fixed_rpm)} rpm"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=_Q_to_gpm(Qv), y=_P_to_hp(Pv), mode="lines",
                                 name=f"HP @ {int(N_vfd_rpm)} rpm"),
                      row=2, col=1)

    # NPSHr
    if curve.NPSHr_ref_m is not None:
        Nf = np.array([curve.NPSHr_m(float(q), N_rpm=float(N_fixed_rpm)) for q in Qf], dtype=float)
        Nv = np.array([curve.NPSHr_m(float(q), N_rpm=float(N_vfd_rpm)) for q in Qv], dtype=float)
        fig.add_trace(go.Scatter(x=_Q_to_gpm(Qf), y=_NPSH_to_ft(Nf), mode="lines",
                                 name=f"NPSHr @ {int(N_fixed_rpm)} rpm"),
                      row=2, col=2)
        fig.add_trace(go.Scatter(x=_Q_to_gpm(Qv), y=_NPSH_to_ft(Nv), mode="lines",
                                 name=f"NPSHr @ {int(N_vfd_rpm)} rpm"),
                      row=2, col=2)

    fig.update_layout(title=title, hovermode="closest")
    fig.update_xaxes(title_text="Flow (gpm)", row=1, col=1)
    fig.update_xaxes(title_text="Flow (gpm)", row=1, col=2)
    fig.update_xaxes(title_text="Flow (gpm)", row=2, col=1)
    fig.update_xaxes(title_text="Flow (gpm)", row=2, col=2)

    fig.update_yaxes(title_text="Head (ft)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Power (hp)", row=2, col=1)
    fig.update_yaxes(title_text="NPSHr (ft)", row=2, col=2)

    return fig