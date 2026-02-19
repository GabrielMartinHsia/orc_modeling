from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from orc_modeling.core.evaluate import evaluate
from orc_modeling.core.types import ConfigSpec, Scenario, SystemSpec
from orc_modeling.core.evaluate import EvalContext


def evaluate_with_speed_sweep(
    ctx: EvalContext,
    config: ConfigSpec,
    scenario: Scenario,
    system: SystemSpec,
    *,
    Q_hint_domain: Optional[Tuple[float, float]] = None,
    n_speed_points: int = 15,
):
    """
    Sweep shared VFD speed between N_min and N_max
    and return best feasible result.
    """

    # Find VFD bounds from config
    grp = config.groups[0]  # assuming single bank for now
    if grp.N_min_rpm is None or grp.N_max_rpm is None:
        return evaluate(ctx, config, scenario, system, Q_hint_domain=Q_hint_domain)

    speeds = np.linspace(grp.N_min_rpm, grp.N_max_rpm, n_speed_points)

    best = None

    for N in speeds:
        scenario.N_vfd_cmd_rpm = float(N)

        r = evaluate(ctx, config, scenario, system, Q_hint_domain=Q_hint_domain)
        if not r.feasible:
            continue

        if best is None:
            best = r
        else:
            if config.dispatch_objective.value == "min_power":
                if r.P_total_W is not None and best.P_total_W is not None:
                    if r.P_total_W < best.P_total_W:
                        best = r
            else:
                if sum(g.count_active for g in r.groups) < sum(g.count_active for g in best.groups):
                    best = r

    return best
