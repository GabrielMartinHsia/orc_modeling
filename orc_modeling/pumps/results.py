# orc_modeling/pumps/results.py
from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import dataclass

from orc_modeling.core.solver import SolverError
from orc_modeling.core.types import (
    Arrangement,
    ConfigSpec,
    FluidProps,
    PumpGroupResult,
    Scenario,
)

from orc_modeling.utilities.units import to_si, ureg
from . import sizing as sz  # uses your pumps/sizing.py
from .banks import ParallelBank, enforce_bep_band
from .curves import curve_model_from_ref

__all__ = [
    "SubBankSizing",
    "PumpGroupOperatingPoint",
    "OperatingPointResults",
    "compute_operating_point",
]


@dataclass(frozen=True)
class SubBankSizing:
    """Sizing metrics for one sub-bank (fixed-speed or VFD) within a group."""
    n_active: int
    N_rpm: float
    Q_per_pump_m3_s: float

    # Power (group-level at this operating point; same for both sub-banks if same H and Q_total)
    P_hyd_W: float
    P_shaft_W: float

    # Specific speed indices (per pump)
    Ns_metric: float
    Ns_us: float
    Ns_dimless: float

    # Suction specific speed (optional; fill when you have NPSHr)
    NPSHr_m: Optional[float] = None
    Nss_us: Optional[float] = None
    Nss_dimless: Optional[float] = None


@dataclass(frozen=True)
class PumpGroupOperatingPoint:
    base: PumpGroupResult

    Q_total_m3_s: float
    H_group_m: float

    # Representative values (keep for convenience/back-compat)
    Q_per_pump_m3_s: Optional[float] = None
    N_rpm_effective: Optional[float] = None
    P_hyd_W: Optional[float] = None
    P_shaft_W: Optional[float] = None
    Ns_metric: Optional[float] = None
    Ns_us: Optional[float] = None
    Ns_dimless: Optional[float] = None

    # NEW: optional sub-bank metrics
    fixed: Optional["SubBankSizing"] = None
    vfd: Optional["SubBankSizing"] = None

    # Suction (overall or leave None for now)
    NPSHr_m: Optional[float] = None
    Nss_us: Optional[float] = None
    Nss_dimless: Optional[float] = None



@dataclass(frozen=True)
class OperatingPointResults:
    """
    Post-solve pump configuration results at a specific operating point.
    """
    group_results: Tuple[PumpGroupOperatingPoint, ...]
    P_total_W: float
    eta_overall: Optional[float]
    min_npsh_margin_m: Optional[float]



def compute_operating_point(
    config: ConfigSpec,
    active_counts: Tuple[int, ...],
    scenario: Scenario,
    props: FluidProps,
    Q_total: float,
) -> OperatingPointResults:
    """
    Post-solve details at the operating flow Q_total.

    Returns:
      OperatingPointResults

    Current assumptions:
      - Groups are PARALLEL banks (same as configurations.build_pump_head_parallel_banks)
      - Banks are treated as SERIES at the config level (heads add)
      - Efficiency is placeholder (0.7 unless you add an eta model)
      - BEP check enforced if BEP metadata exists on the curve (Q_bep_ref_m3_s)
    """
    rho = to_si(props.rho_kg_m3, ureg("kg/m^3"))
    mu = to_si(props.mu_Pa_s, ureg("Pa*s"))
    Pvap = to_si(props.Pvap_Pa, ureg.Pa)
    g = 9.80665

    groups_out = []
    P_total = 0.0
    H_total = 0.0

    # TODO: replace with real eta model (curve-based / fitted)
    eta_default = 0.7

    for i, (grp, n_active) in enumerate(zip(config.groups, active_counts)):
        pump = grp.pump
        # curve = pump.curve_ref
        curve = curve_model_from_ref(pump.curve_ref, N_ref_rpm=pump.N_ref_rpm)

        if n_active == 0:
            base = PumpGroupResult(
                group_index=i,
                pump_name=pump.name,
                arrangement=grp.arrangement,
                count_installed=grp.count_installed,
                count_active=0,
                control=grp.control,
                N_rpm=grp.N_fixed_rpm,
                eta=None,
                H_group_m=0.0,
                P_group_W=0.0,
            )

            groups_out.append(
                PumpGroupOperatingPoint(
                    base=base,
                    Q_total_m3_s=float(Q_total),
                    H_group_m=0.0,
                    Q_per_pump_m3_s=None,
                    N_rpm_effective=None,
                    P_hyd_W=None,
                    P_shaft_W=None,
                    Ns_metric=None,
                    Ns_us=None,
                    Ns_dimless=None,
                    NPSHr_m=None,
                    Nss_us=None,
                    Nss_dimless=None,
                    fixed=None,
                    vfd=None,
                )
            )

            continue


        if grp.arrangement != Arrangement.PARALLEL:
            raise SolverError("compute_operating_point currently supports PARALLEL groups only")

        # Split active pumps into VFD + fixed (same policy as configurations.py)
        n_vfd_inst = int(getattr(grp, "count_vfd_installed", 0))
        n_vfd_inst = max(0, min(n_vfd_inst, grp.count_installed))

        n_vfd_active = min(n_active, n_vfd_inst)
        n_fixed_active = n_active - n_vfd_active

        N_fixed = grp.N_fixed_rpm
        if n_fixed_active > 0 and N_fixed is None:
            raise SolverError("Fixed-speed pumps active but grp.N_fixed_rpm is None")

        N_vfd = None
        if n_vfd_active > 0:
            N_vfd = scenario.N_vfd_cmd_rpm
            if N_vfd is None:
                if grp.N_max_rpm is not None:
                    N_vfd = grp.N_max_rpm
                elif grp.N_fixed_rpm is not None:
                    N_vfd = grp.N_fixed_rpm
                else:
                    raise SolverError("VFD pumps active but no speed provided")

            if grp.N_min_rpm is not None:
                N_vfd = max(float(N_vfd), float(grp.N_min_rpm))
            if grp.N_max_rpm is not None:
                N_vfd = min(float(N_vfd), float(grp.N_max_rpm))

        bank = ParallelBank(
            curve=curve,
            n_fixed=n_fixed_active,
            n_vfd=n_vfd_active,
            N_fixed_rpm=float(N_fixed) if N_fixed is not None else 0.0,
            N_vfd_rpm=float(N_vfd) if N_vfd is not None else None,
        )

        # Head contributed by this bank at the solved total flow
        H_bank = bank.head_fn()
        H_group = float(H_bank(Q_total))

        # Enforce BEP if curve carries BEP info
        Q_bep_ref = getattr(curve, "Q_bep_ref_m3_s", None)
        if Q_bep_ref is not None:
            enforce_bep_band(
                bank,
                Q_total=float(Q_total),
                Q_bep_ref_m3_s=float(Q_bep_ref),
                N_ref_rpm=float(curve.N_ref_rpm),
                low=0.8,
                high=1.1,
            )

        # Placeholder efficiency; clamp to sane bounds
        eta = max(0.05, min(float(eta_default), 0.95))

        # Shaft power for this bank (toy): rho*g*Q*H / eta
        P_group = (rho * g * float(Q_total) * float(H_group)) / eta

        H_total += float(H_group)
        P_total += float(P_group)

        def _build_subbank_sizing(n_active: int, N_rpm: float) -> SubBankSizing:
            Q_per = float(Q_total) / float(n_active)  # per-pump flow in that sub-bank

            P_hyd = sz.hydraulic_power_W(rho_kg_m3=rho, Q_m3_s=float(Q_total), H_m=float(H_group))
            P_shaft = sz.shaft_power_W(P_hyd_W=P_hyd, eta=eta)

            Ns_metric = sz.specific_speed_metric(N_rpm=float(N_rpm), Q_m3_s=Q_per, H_m=float(H_group))
            Ns_us = sz.specific_speed_us_from_si(N_rpm=float(N_rpm), Q_m3_s=Q_per, H_m=float(H_group))
            Ns_dim = sz.specific_speed_dimensionless(N_rpm=float(N_rpm), Q_m3_s=Q_per, H_m=float(H_group))

            return SubBankSizing(
                n_active=int(n_active),
                N_rpm=float(N_rpm),
                Q_per_pump_m3_s=float(Q_per),
                P_hyd_W=float(P_hyd),
                P_shaft_W=float(P_shaft),
                Ns_metric=float(Ns_metric),
                Ns_us=float(Ns_us),
                Ns_dimless=float(Ns_dim),
                NPSHr_m=None,
                Nss_us=None,
                Nss_dimless=None,
            )


        fixed_metrics: Optional[SubBankSizing] = None
        vfd_metrics: Optional[SubBankSizing] = None

        if n_fixed_active > 0:
            if N_fixed is None:
                raise SolverError("Fixed-speed sub-bank active but N_fixed_rpm is None")
            fixed_metrics = _build_subbank_sizing(n_fixed_active, float(N_fixed))

        if n_vfd_active > 0:
            if N_vfd is None:
                raise SolverError("VFD sub-bank active but N_vfd_rpm is None")
            vfd_metrics = _build_subbank_sizing(n_vfd_active, float(N_vfd))

        base = PumpGroupResult(
            group_index=i,
            pump_name=pump.name,
            arrangement=grp.arrangement,
            count_installed=grp.count_installed,
            count_active=n_active,
            control=grp.control,
            N_rpm=grp.N_fixed_rpm,  # your core type only stores one; keep as-is for now
            eta=eta,
            H_group_m=float(H_group),
            P_group_W=float(P_group),
        )

        # Representative fields: if only one sub-bank is active, use it
        rep = fixed_metrics or vfd_metrics
        rep_Q_per = rep.Q_per_pump_m3_s if rep is not None else None
        rep_N = rep.N_rpm if rep is not None else None
        rep_P_hyd = rep.P_hyd_W if rep is not None else None
        rep_P_shaft = rep.P_shaft_W if rep is not None else None
        rep_Ns_metric = rep.Ns_metric if rep is not None else None
        rep_Ns_us = rep.Ns_us if rep is not None else None
        rep_Ns_dimless = rep.Ns_dimless if rep is not None else None

        # If BOTH are active, don't pretend there is one representative set
        if fixed_metrics is not None and vfd_metrics is not None:
            rep_Q_per = rep_N = rep_P_hyd = rep_P_shaft = None
            rep_Ns_metric = rep_Ns_us = rep_Ns_dimless = None


        groups_out.append(
            PumpGroupOperatingPoint(
                base=base,
                Q_total_m3_s=float(Q_total),
                H_group_m=float(H_group),

                Q_per_pump_m3_s=rep_Q_per,
                N_rpm_effective=rep_N,
                P_hyd_W=rep_P_hyd,
                P_shaft_W=rep_P_shaft,
                Ns_metric=rep_Ns_metric,
                Ns_us=rep_Ns_us,
                Ns_dimless=rep_Ns_dimless,

                fixed=fixed_metrics,
                vfd=vfd_metrics,
            )
        )


    eta_overall = (rho * g * float(Q_total) * float(H_total)) / P_total if P_total > 0 else None
    min_npsh_margin_m = None # not yet computing NPSH margin, return none for now...

    return OperatingPointResults(
        group_results=tuple(groups_out),
        P_total_W=float(P_total),
        eta_overall=float(eta_overall) if eta_overall is not None else None,
        min_npsh_margin_m=min_npsh_margin_m,
    )

