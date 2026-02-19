from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

import pint

from orc_modeling.fluidprops import Fluid
from orc_modeling.utilities.units import (
    Q_, ureg,
    to_si,
    U_T, U_P, U_H, U_S,
)

NumberOrQty = Union[float, int, pint.Quantity]


@dataclass(frozen=True)
class ProcessPoint:
    """A labeled thermodynamic point.

    Provide any subset of (T, P, h, s). Plot functions will use what they need.

    Units:
      T: K (or pint Quantity)
      P: Pa (or pint Quantity)
      h: J/kg (or pint Quantity)
      s: J/kg/K (or pint Quantity)
    """
    label: str
    T: Optional[NumberOrQty] = None
    P: Optional[NumberOrQty] = None
    h: Optional[NumberOrQty] = None
    s: Optional[NumberOrQty] = None


def _si(value: NumberOrQty, unit) -> float:
    """Convert float/Quantity to SI float using project's unit conventions."""
    return to_si(value, unit)


def saturation_dome(
    fluid: Fluid,
    P_min: NumberOrQty,
    P_max: NumberOrQty,
    n: int = 200,
) -> dict[str, np.ndarray]:
    """Compute saturation dome arrays.

    Returns a dict of SI-float numpy arrays:
      P, T, hL, hV, sL, sV

    Notes:
      - Uses VF=0 and VF=1 sat states via your backend methods.
      - Pressures are linearly spaced in log(P) to give better resolution over wide ranges.
    """
    Pmin = _si(P_min, U_P)
    Pmax_user = _si(P_max, U_P)
    if Pmin <= 0 or Pmax_user <= 0:
        raise ValueError("Require P_min > 0 and P_max > 0.")
    if Pmax_user <= Pmin:
        raise ValueError("Require P_min < P_max.")

    # Clamp to just below critical to avoid supercritical / numerical issues
    Pc = _si(fluid.p_crit(), U_P)
    Pmax = min(Pmax_user, 0.999 * Pc)

    # If user requested a range entirely above Pc, make it a clear error
    if Pmin >= Pmax:
        raise ValueError(
            f"P_min is at/above clamped P_max near critical. "
            f"Got P_min={Pmin:.3g} Pa, Pc={Pc:.3g} Pa."
        )


    P = np.geomspace(Pmin, Pmax, int(n))

    T = np.empty_like(P)
    hL = np.empty_like(P)
    hV = np.empty_like(P)
    sL = np.empty_like(P)
    sV = np.empty_like(P)

    # Ensure outputs are floats for plotting speed
    # (If your Fluid returns pint by default, these will be quantities; we convert.)
    for i, Pi in enumerate(P):
        Ti = fluid.T_sat(Pi)
        T[i] = _si(Ti, U_T)

        hL[i] = _si(fluid.h_sat_liq(Pi), U_H)
        hV[i] = _si(fluid.h_sat_vap(Pi), U_H)

        sL[i] = _si(fluid.s_sat_liq(Pi), U_S)
        sV[i] = _si(fluid.s_sat_vap(Pi), U_S)

    return {"P": P, "T": T, "hL": hL, "hV": hV, "sL": sL, "sV": sV}


def plot_ph(
    fluid: Fluid,
    P_min: NumberOrQty,
    P_max: NumberOrQty,
    *,
    n: int = 250,
    points: Optional[Sequence[ProcessPoint]] = None,
    ax=None,
    title: Optional[str] = None,
    x_unit: str = "kJ/kg",
    y_unit: str = "bar",
    y_log: bool = True,
    dome: bool = True,
):
    """Plot a P-h diagram with optional saturation dome and process points."""
    if ax is None:
        fig, ax = plt.subplots()

    # Unit scale factors for nicer axes
    x_scale = (1.0 * ureg("J/kg")).to(x_unit).magnitude  # J/kg -> x_unit
    y_scale = (1.0 * ureg("Pa")).to(y_unit).magnitude   # Pa -> y_unit

    if dome:
        d = saturation_dome(fluid, P_min, P_max, n=n)
        ax.plot(d["hL"] * x_scale, d["P"] * y_scale)
        ax.plot(d["hV"] * x_scale, d["P"] * y_scale)

    if points:
        for pt in points:
            if pt.h is None or pt.P is None:
                # If missing, try to compute h from (T,P) if provided
                if pt.T is not None and pt.P is not None and pt.h is None:
                    h_val = fluid.h(pt.T, pt.P)
                    h_si = _si(h_val, U_H)
                    P_si = _si(pt.P, U_P)
                else:
                    continue
            else:
                h_si = _si(pt.h, U_H)
                P_si = _si(pt.P, U_P)

            ax.scatter(h_si * x_scale, P_si * y_scale)
            ax.annotate(pt.label, (h_si * x_scale, P_si * y_scale), xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(f"h [{x_unit}]")
    ax.set_ylabel(f"P [{y_unit}]")
    if y_log:
        ax.set_yscale("log")

    if title is None:
        title = f"P-h: {fluid.fluid_id}"
    ax.set_title(title)

    return ax


def plot_ts(
    fluid: Fluid,
    P_min: NumberOrQty,
    P_max: NumberOrQty,
    *,
    n: int = 250,
    points: Optional[Sequence[ProcessPoint]] = None,
    ax=None,
    title: Optional[str] = None,
    x_unit: str = "kJ/kg/K",
    y_unit: str = "degC",
    dome: bool = True,
):
    """Plot a T-s diagram with optional saturation dome and process points."""
    if ax is None:
        fig, ax = plt.subplots()

    x_scale = (1.0 * ureg("J/kg/kelvin")).to(x_unit).magnitude  # J/kg/K -> x_unit
    # For temperature axis, we’ll convert values directly rather than scale factors
    # because Celsius is affine.
    def T_to_unit(T_K: float) -> float:
        return Q_(T_K, "K").to(y_unit).magnitude

    if dome:
        d = saturation_dome(fluid, P_min, P_max, n=n)
        ax.plot(d["sL"] * x_scale, np.array([T_to_unit(t) for t in d["T"]]))
        ax.plot(d["sV"] * x_scale, np.array([T_to_unit(t) for t in d["T"]]))

    if points:
        for pt in points:
            # Need (s,T) to plot. Compute missing from (T,P) if possible.
            if pt.T is None:
                continue

            T_si = _si(pt.T, U_T)

            if pt.s is None:
                if pt.P is None:
                    continue
                s_val = fluid.s(pt.T, pt.P)
                s_si = _si(s_val, U_S)
            else:
                s_si = _si(pt.s, U_S)

            ax.scatter(s_si * x_scale, T_to_unit(T_si))
            ax.annotate(pt.label, (s_si * x_scale, T_to_unit(T_si)), xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel(f"s [{x_unit}]")
    ax.set_ylabel(f"T [{y_unit}]")

    if title is None:
        title = f"T-s: {fluid.fluid_id}"
    ax.set_title(title)

    return ax

