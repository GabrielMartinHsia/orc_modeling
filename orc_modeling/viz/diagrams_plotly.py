from __future__ import annotations

from typing import Optional, Sequence

from orc_modeling.fluidprops import Fluid
from orc_modeling.utilities.units import ureg, Q_, U_P, U_T, U_H, U_S, to_si
from orc_modeling.viz.diagrams import ProcessPoint, saturation_dome


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
        return go
    except ImportError as e:
        raise ImportError(
            "Plotly is not installed. Install it with:\n"
            "  pip install plotly\n"
            "or add it as an optional dependency for viz."
        ) from e


def _si(v, unit) -> float:
    return to_si(v, unit)


def plot_ph_plotly(
    fluid: Fluid,
    P_min,
    P_max,
    *,
    n: int = 300,
    points: Optional[Sequence[ProcessPoint]] = None,
    title: Optional[str] = None,
    x_unit: str = "kJ/kg",
    y_unit: str = "bar",
    y_log: bool = True,
    dome: bool = True,
):
    go = _require_plotly()

    d = saturation_dome(fluid, P_min, P_max, n=n) if dome else None

    x_scale = (1.0 * ureg("J/kg")).to(x_unit).magnitude
    y_scale = (1.0 * ureg("Pa")).to(y_unit).magnitude

    fig = go.Figure()

    if dome and d is not None:
        fig.add_trace(go.Scatter(
            x=d["hL"] * x_scale, y=d["P"] * y_scale,
            mode="lines", name="sat liq",
            hovertemplate=f"h=%{{x:.3g}} {x_unit}<br>P=%{{y:.3g}} {y_unit}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=d["hV"] * x_scale, y=d["P"] * y_scale,
            mode="lines", name="sat vap",
            hovertemplate=f"h=%{{x:.3g}} {x_unit}<br>P=%{{y:.3g}} {y_unit}<extra></extra>",
        ))

    if points:
        xs, ys, labels = [], [], []
        for pt in points:
            if pt.P is None:
                continue
            P_si = _si(pt.P, U_P)

            if pt.h is not None:
                h_si = _si(pt.h, U_H)
            elif pt.T is not None:
                h_si = _si(fluid.h(pt.T, pt.P), U_H)
            else:
                continue

            xs.append(h_si * x_scale)
            ys.append(P_si * y_scale)
            labels.append(pt.label)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=labels,
            textposition="top right",
            name="points",
            hovertemplate=f"label=%{{text}}<br>h=%{{x:.3g}} {x_unit}<br>P=%{{y:.3g}} {y_unit}<extra></extra>",
        ))

    fig.update_layout(
        title=title or f"P-h: {fluid.fluid_id}",
        xaxis_title=f"h [{x_unit}]",
        yaxis_title=f"P [{y_unit}]",
        template="plotly_white",
    )
    if y_log:
        fig.update_yaxes(type="log")

    return fig


def plot_ts_plotly(
    fluid: Fluid,
    P_min,
    P_max,
    *,
    n: int = 300,
    points: Optional[Sequence[ProcessPoint]] = None,
    title: Optional[str] = None,
    x_unit: str = "kJ/kg/K",
    y_unit: str = "degC",
    dome: bool = True,
):
    go = _require_plotly()

    d = saturation_dome(fluid, P_min, P_max, n=n) if dome else None

    x_scale = (1.0 * ureg("J/kg/kelvin")).to(x_unit).magnitude

    def T_to_unit(T_K: float) -> float:
        return Q_(T_K, "K").to(y_unit).magnitude

    fig = go.Figure()

    if dome and d is not None:
        fig.add_trace(go.Scatter(
            x=d["sL"] * x_scale, y=[T_to_unit(t) for t in d["T"]],
            mode="lines", name="sat liq",
            hovertemplate=f"s=%{{x:.3g}} {x_unit}<br>T=%{{y:.3g}} {y_unit}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=d["sV"] * x_scale, y=[T_to_unit(t) for t in d["T"]],
            mode="lines", name="sat vap",
            hovertemplate=f"s=%{{x:.3g}} {x_unit}<br>T=%{{y:.3g}} {y_unit}<extra></extra>",
        ))

    if points:
        xs, ys, labels = [], [], []
        for pt in points:
            if pt.T is None:
                continue
            T_si = _si(pt.T, U_T)

            if pt.s is not None:
                s_si = _si(pt.s, U_S)
            elif pt.P is not None:
                s_si = _si(fluid.s(pt.T, pt.P), U_S)
            else:
                continue

            xs.append(s_si * x_scale)
            ys.append(T_to_unit(T_si))
            labels.append(pt.label)

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=labels,
            textposition="top right",
            name="points",
            hovertemplate=f"label=%{{text}}<br>s=%{{x:.3g}} {x_unit}<br>T=%{{y:.3g}} {y_unit}<extra></extra>",
        ))

    fig.update_layout(
        title=title or f"T-s: {fluid.fluid_id}",
        xaxis_title=f"s [{x_unit}]",
        yaxis_title=f"T [{y_unit}]",
        template="plotly_white",
    )

    return fig
