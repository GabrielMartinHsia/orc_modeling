# tests/toy_context.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from orc_modeling.pumps.configurations import _Q_at_head, _H_at_speed  # ok in tests
from orc_modeling.pumps.configurations import build_pump_head_parallel_banks
from orc_modeling.core.solver import SolverError
from orc_modeling.core.evaluate import EvalContext
from orc_modeling.core.types import (
    Arrangement,
    ConfigSpec,
    ControlMode,
    DispatchObjective,
    FluidProps,
    PumpCurveRef,
    PumpGroupResult,
    PumpGroupSpec,
    PumpLimits,
    PumpSpec,
    Scenario,
    SystemSpec,
)


def constant_fluid_props(_: Scenario) -> FluidProps:
    # Just enough for hydraulics/power placeholders
    return FluidProps(rho_kg_m3=1000.0, mu_Pa_s=1e-3, Pvap_Pa=2000.0)


def make_system_head_builder(H_static_m: float, k_m_per_q2: float):
    # H_sys(Q) = H_static + k*Q^2
    def build_system_head(system: SystemSpec, scenario: Scenario, props: FluidProps) -> Callable[[float], float]:
        def H_sys(Q: float) -> float:
            return float(H_static_m + k_m_per_q2 * Q * Q)
        return H_sys
    return build_system_head


def build_pump_head_from_linear_curves( #<-- NO LONGER USED: superceded by `build_pump_head_parallel_banks()` in /pumps/configurations.py
    config: ConfigSpec,
    active_counts: Tuple[int, ...],
    scenario: Scenario,
    props: FluidProps,
) -> Callable[[float], float]:
    """
    Toy: assumes all groups act in SERIES at the config level, and within a group:
      - PARALLEL: head is that of one pump at per-pump flow = Q_total / count_active
      - SERIES: head adds count_active times (rare for identical pumps in one group, but supported)
    """
    groups = config.groups

    #validate length to avoid accidentally testing the wrong configuration
    if len(active_counts) != len(groups):
        raise ValueError(
            f"active_counts length ({len(active_counts)}) "
            f"must match number of groups ({len(groups)})"
        )


    def H_pump(Q_total: float) -> float:
        H_total = 0.0
        for grp, n_active in zip(groups, active_counts):
            if n_active == 0:
                # group contributes no head if off
                continue

            curve = grp.pump.curve_ref

            if grp.arrangement == Arrangement.PARALLEL:
                q_each = Q_total / n_active
                H_total += float(curve.H_m(q_each))
            else:  # SERIES
                # each pump sees same flow; heads add
                H_total += float(n_active * curve.H_m(Q_total))

        return H_total

    return H_pump


def toy_compute_details(
    config: ConfigSpec,
    active_counts: Tuple[int, ...],
    scenario: Scenario,
    props: FluidProps,
    Q_total: float,
):
    """
    Returns:
      (group_results, P_total_W, eta_overall, min_npsh_margin_m)

    Toy power model:
      P = rho*g*Q*H / eta_total   (using group eta if provided, else 0.7)
    """
    rho = props.rho_kg_m3
    g = 9.80665

    groups_out = []
    P_total = 0.0
    H_total = 0.0
    eta_used = 0.7

    for i, (grp, n_active) in enumerate(zip(config.groups, active_counts)):
        curve = grp.pump.curve_ref

        if n_active == 0:
            groups_out.append(
                PumpGroupResult(
                    group_index=i,
                    pump_name=grp.pump.name,
                    arrangement=grp.arrangement,
                    count_installed=grp.count_installed,
                    count_active=0,
                    control=grp.control,
                    N_rpm=grp.N_fixed_rpm,
                    eta=None,
                    H_group_m=0.0,
                    P_group_W=0.0,
                )
            )
            continue

        if grp.arrangement == Arrangement.PARALLEL:
            q_each = Q_total / n_active
            H_group = float(curve.H_m(q_each))
        else:
            H_group = float(n_active * curve.H_m(Q_total))



        # --- BEP CHECK (correct for mixed speeds in parallel) ---
        if curve.Q_bep_ref_m3_s is not None and n_active > 0:
            # Compute the *common* header head for this bank at Q_total using affinity scaling.
            # (Toy assumes one bank; if you later use multiple groups-in-series, we’ll compute per-bank heads.)
            N_ref = grp.pump.N_ref_rpm

            # determine how many active are VFD vs fixed
            n_vfd_inst = int(getattr(grp, "count_vfd_installed", 0))
            n_vfd_active = min(n_active, n_vfd_inst)
            n_fixed_active = n_active - n_vfd_active

            if n_fixed_active > 0 and grp.N_fixed_rpm is None:
                raise SolverError("Fixed-speed pump active but grp.N_fixed_rpm is None")
            if n_vfd_active > 0 and scenario.N_vfd_cmd_rpm is None:
                raise SolverError("VFD pump active but Scenario.N_vfd_cmd_rpm is None")

            N_fixed = float(grp.N_fixed_rpm) if n_fixed_active > 0 else None
            N_vfd = float(scenario.N_vfd_cmd_rpm) if n_vfd_active > 0 else None

            # Upper head bound for inversion
            H_hi = 0.0
            if n_fixed_active > 0:
                H_hi = max(H_hi, _H_at_speed(curve, 0.0, N_rpm=N_fixed, N_ref_rpm=N_ref))
            if n_vfd_active > 0:
                H_hi = max(H_hi, _H_at_speed(curve, 0.0, N_rpm=N_vfd, N_ref_rpm=N_ref))
            if H_hi <= 0.0:
                raise SolverError("Invalid bank shutoff head")

            # Define bank capacity at a head: sum flows at common head
            def Q_bank_at_head(H: float) -> float:
                Qt = 0.0
                if n_fixed_active > 0:
                    q1 = _Q_at_head(curve, H, N_rpm=N_fixed, N_ref_rpm=N_ref)
                    Qt += n_fixed_active * q1
                if n_vfd_active > 0:
                    q2 = _Q_at_head(curve, H, N_rpm=N_vfd, N_ref_rpm=N_ref)
                    Qt += n_vfd_active * q2
                return float(Qt)

            # Solve for the common header head H_head such that Q_bank(H_head)=Q_total
            if Q_total > Q_bank_at_head(0.0) + 1e-12:
                raise SolverError("Requested flow exceeds bank capacity")

            from orc_modeling.core.solver import bisect_root
            solH = bisect_root(lambda H: Q_bank_at_head(H) - Q_total, 0.0, H_hi)
            if not solH.converged:
                raise SolverError("Failed to compute header head for BEP check")

            H_head = float(solH.x)

            # Now compute *per-pump* flows at that common head
            def _check(q_each: float, N: float) -> None:
                r = float(N) / float(N_ref)
                Q_bep = float(curve.Q_bep_ref_m3_s) * r
                if Q_bep <= 0:
                    return
                ratio = q_each / Q_bep
                if ratio < 0.8 or ratio > 1.1:
                    raise SolverError("Pump operating outside BEP band")

            if n_fixed_active > 0:
                q_each_fixed = _Q_at_head(curve, H_head, N_rpm=N_fixed, N_ref_rpm=N_ref)
                _check(q_each_fixed, N_fixed)

            if n_vfd_active > 0:
                q_each_vfd = _Q_at_head(curve, H_head, N_rpm=N_vfd, N_ref_rpm=N_ref)
                _check(q_each_vfd, N_vfd)
        # --- END BEP CHECK ---



        if curve.eta is not None:
            eta_used = float(curve.eta(Q_total))  # toy: use total flow
            eta_used = max(0.05, min(eta_used, 0.95))

        P_group = (rho * g * Q_total * H_group) / eta_used
        H_total += H_group
        P_total += P_group

        groups_out.append(
            PumpGroupResult(
                group_index=i,
                pump_name=grp.pump.name,
                arrangement=grp.arrangement,
                count_installed=grp.count_installed,
                count_active=n_active,
                control=grp.control,
                N_rpm=grp.N_fixed_rpm,
                eta=eta_used,
                H_group_m=H_group,
                P_group_W=P_group,
            )
        )

    # overall eta is rough placeholder
    eta_overall = (rho * g * Q_total * H_total) / P_total if P_total > 0 else None

    return tuple(groups_out), P_total, eta_overall, None


def make_linear_pump(name: str, *, N_ref_rpm: float, H0: float, slope: float, Q_min: float, Q_max: float, Q_bep_ref_m3_s: float | None = None) -> PumpSpec:
    """
    Pump curve: H(Q) = H0 - slope*Q
    (Q in m^3/s, H in m)
    """
    def H(Q: float) -> float:
        return H0 - slope * Q

    curve = PumpCurveRef(H_m=H, eta=None, NPSHr_m=None, Q_min_m3_s=Q_min, Q_max_m3_s=Q_max, Q_bep_ref_m3_s=Q_bep_ref_m3_s)
    return PumpSpec(name=name, curve_ref=curve, N_ref_rpm=N_ref_rpm, limits=PumpLimits(), cost=None)  # cost not needed


def make_toy_context(H_static_m: float = 20.0, k_m_per_q2: float = 2000.0) -> EvalContext:
    return EvalContext(
        fluid_props=constant_fluid_props,
        build_system_head=make_system_head_builder(H_static_m=H_static_m, k_m_per_q2=k_m_per_q2),
        # build_pump_head=build_pump_head_from_linear_curves,
        build_pump_head=build_pump_head_parallel_banks,
        compute_details=toy_compute_details,
    )
