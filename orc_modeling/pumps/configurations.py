# orc_modeling/pumps/configurations.py
from __future__ import annotations

from typing import Callable, Tuple

from orc_modeling.core.types import ConfigSpec, Scenario, FluidProps, Arrangement
from .banks import ParallelBank
from .curves import curve_model_from_ref

"""
This module acts as a composition/assembly layer, and builds a "configuration head function" out of a number of (parallel) banks 
and how they are connected:

    * Groups in parallel (same head, flows add)
    * Groups in series (heads add, flows same)
    * Strictly observe any enable/disable, min/max pump count constraints

(this is where `parallel_bank.py` gets *used*, but not *defined*)
"""



def build_pump_head_parallel_banks(
    config: ConfigSpec,
    active_counts: Tuple[int, ...],
    scenario: Scenario,
    props: FluidProps,
) -> Callable[[float], float]:
    """
    Build H_pump(Q) [m] for a configuration.

    Assumptions (for now):
      - Each group represents ONE parallel pump bank discharging into a common header.
      - Banks (groups) are in SERIES with each other at the config level (heads add across groups).
        (If you only have one bank, this reduces to that bank.)
      - Within a bank, pumps are identical, and some subset can be VFD with a single shared speed.
      - Number of VFD-installed pumps in the group is grp.count_vfd_installed.

    VFD speed:
      - scenario.N_vfd_cmd_rpm is used for ALL VFD pumps (shared)
      - if None and VFD pumps are active, we fall back to grp.N_max_rpm or grp.N_fixed_rpm

    This function is the physically correct "sum flows at head, then invert to H(Q)" approach.
    """
    groups = config.groups
    if len(active_counts) != len(groups):
        raise ValueError("active_counts must align with config.groups")

    # Pre-resolve per-group pump/curve references and speed settings
    bank_specs = []
    for grp, n_active in zip(groups, active_counts):
        if n_active < 0 or n_active > grp.count_installed:
            raise ValueError("active count outside installed range")

        if grp.arrangement != Arrangement.PARALLEL:
            raise ValueError("This builder currently supports PARALLEL banks only")

        pump = grp.pump
        # curve = pump.curve_ref
        curve = curve_model_from_ref(pump.curve_ref, N_ref_rpm=pump.N_ref_rpm)

        n_vfd_inst = int(getattr(grp, "count_vfd_installed", 0))
        n_vfd_inst = max(0, min(n_vfd_inst, grp.count_installed))

        # split active pumps into VFD + fixed (shared VFD speed)
        n_vfd_active = min(n_active, n_vfd_inst)
        n_fixed_active = n_active - n_vfd_active

        N_fixed = grp.N_fixed_rpm
        if n_fixed_active > 0 and N_fixed is None:
            raise ValueError("Fixed-speed pumps active but grp.N_fixed_rpm is None")

        N_vfd = None
        if n_vfd_active > 0:
            N_vfd = scenario.N_vfd_cmd_rpm
            if N_vfd is None:
                # fallback policy for now
                if grp.N_max_rpm is not None:
                    N_vfd = grp.N_max_rpm
                elif grp.N_fixed_rpm is not None:
                    N_vfd = grp.N_fixed_rpm
                else:
                    raise ValueError("VFD pumps active but no N_vfd_cmd_rpm, N_max_rpm, or N_fixed_rpm provided")

            # enforce bounds if provided
            if grp.N_min_rpm is not None:
                N_vfd = max(float(N_vfd), float(grp.N_min_rpm))
            if grp.N_max_rpm is not None:
                N_vfd = min(float(N_vfd), float(grp.N_max_rpm))

        bank_specs.append(
            ParallelBank(
                curve=curve,
                n_fixed=n_fixed_active,
                n_vfd=n_vfd_active,
                N_fixed_rpm=float(N_fixed) if N_fixed is not None else 0.0,  # only used if n_fixed>0
                N_vfd_rpm=float(N_vfd) if N_vfd is not None else None,
            )
        )

        bank_head_fns = [bank.head_fn() for bank in bank_specs]



    def H_pump(Q_total: float) -> float:
        Q_total = float(Q_total)
        if Q_total < 0:
            raise ValueError("Q_total must be >= 0")

        H_total = 0.0

        for bank, H_bank in zip(bank_specs, bank_head_fns):
            if (bank.n_fixed + bank.n_vfd) == 0:
                continue
            H_total += float(H_bank(Q_total))

        return float(H_total)

    return H_pump
