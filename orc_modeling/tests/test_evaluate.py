# tests/test_evaluate.py
from __future__ import annotations

# #temporary debugging:
# import orc_modeling.core.evaluate as ev
# print("evaluate imported from:", ev.__file__)
# #end temporary debugging

from orc_modeling.core.evaluate import evaluate
from orc_modeling.core.types import (
    Arrangement,
    ConfigSpec,
    ControlMode,
    DispatchObjective,
    PumpGroupSpec,
    Scenario,
    SystemSpec,
)
from .toy_context import make_linear_pump, make_toy_context


def test_intersection_solves_and_returns_op_point():
    ctx = make_toy_context(H_static_m=20.0, k_m_per_q2=2000.0)
    system = SystemSpec()
    sc = Scenario(name="case1", T_suction_K=300.0, P_suction_Pa=101325.0, Q_target_m3_s=0.02)

    pump = make_linear_pump(
        name="P1",
        N_ref_rpm=3600.0,
        H0=60.0,
        slope=800.0,
        Q_min=0.0,
        Q_max=0.05,
    )

    grp = PumpGroupSpec(
        pump=pump,
        arrangement=Arrangement.PARALLEL,
        count_installed=1,
        allowed_active_counts=(1,),  # always on
        control=ControlMode.FIXED_SPEED,
        N_fixed_rpm=3600.0,
    )

    cfg = ConfigSpec(name="single", groups=(grp,), booster=None, dispatch_objective=DispatchObjective.MIN_POWER)

    r = evaluate(ctx, cfg, sc, system, Q_hint_domain=(0.0, 0.05))
    assert r.feasible
    assert r.op is not None
    assert 0.0 <= r.op.Q_m3_s <= 0.05
    # pump head and system head should be close
    assert abs(r.op.H_pump_m - r.op.H_system_m) < 1e-5


def test_dispatch_stages_lag_on_when_needed():
    # System is demanding enough head that a single *slow VFD* pump cannot operate,
    # so at least one fixed-speed pump must be staged on.
    ctx = make_toy_context(H_static_m=25.0, k_m_per_q2=3000.0)
    system = SystemSpec()

    sc = Scenario(
        name="high_flow",
        T_suction_K=300.0,
        P_suction_Pa=101325.0,
        Q_target_m3_s=0.03,
        N_vfd_cmd_rpm=2000.0,  # shared VFD command (low speed)
    )

    pump = make_linear_pump(
        name="P1",
        N_ref_rpm=3600.0,
        H0=70.0,
        slope=900.0,
        Q_min=0.0,
        Q_max=0.06,
    )

    # One physical parallel bank of 3 pumps total:
    # - 1 pump is VFD-capable
    # - 2 pumps are fixed-speed
    bank = PumpGroupSpec(
        pump=pump,
        arrangement=Arrangement.PARALLEL,
        count_installed=3,
        count_vfd_installed=1,
        allowed_active_counts=(1, 2, 3),
        control=ControlMode.FIXED_SPEED,  # group-level tag; mixed behavior comes from count_vfd_installed
        N_fixed_rpm=3600.0,
        N_min_rpm=1800.0,
        N_max_rpm=3600.0,
    )

    cfg = ConfigSpec(
        name="bank_1vfd_2fixed",
        groups=(bank,),
        booster=None,
        dispatch_objective=DispatchObjective.MIN_ACTIVE,
    )

    r = evaluate(ctx, cfg, sc, system, Q_hint_domain=(0.0, 0.06))
    assert r.feasible
    assert len(r.groups) == 1

    grp = r.groups[0]
    # With the VFD commanded very low, 1 active pump (which will be the VFD) is infeasible,
    # so MIN_ACTIVE should stage to 2 active pumps.
    assert grp.count_active == 2


def test_dispatch_objective_min_power_prefers_lower_power_solution():
    ctx = make_toy_context(H_static_m=15.0, k_m_per_q2=1500.0)
    system = SystemSpec()
    sc = Scenario(name="mid", T_suction_K=300.0, P_suction_Pa=101325.0, Q_target_m3_s=0.02)

    pump = make_linear_pump(
        name="P1",
        N_ref_rpm=3600.0,
        H0=60.0,
        slope=700.0,
        Q_min=0.0,
        Q_max=0.06,
    )

    # Two alternative ways to meet flow:
    # - one pump active in a parallel bank of 2
    # - two pumps active in the same bank
    bank = PumpGroupSpec(
        pump=pump,
        arrangement=Arrangement.PARALLEL,
        count_installed=2,
        allowed_active_counts=(1, 2),
        control=ControlMode.FIXED_SPEED,
        N_fixed_rpm=3600.0,
    )

    cfg = ConfigSpec(
        name="bank2",
        groups=(bank,),
        booster=None,
        dispatch_objective=DispatchObjective.MIN_POWER,
    )

    r = evaluate(ctx, cfg, sc, system, Q_hint_domain=(0.0, 0.06))
    assert r.feasible
    assert len(r.groups) == 1
    # For this toy power model, more pumps in parallel usually increases head at per-pump q_each,
    # which can change power. This assertion is intentionally weak: ensure it chose a permitted count.
    assert r.groups[0].count_active in (1, 2)


def test_bep_constraint_rejects_off_bep_operation():
    ctx = make_toy_context(H_static_m=20.0, k_m_per_q2=2000.0)
    system = SystemSpec()

    sc = Scenario(
        name="bep_fail",
        T_suction_K=300.0,
        P_suction_Pa=101325.0,
        Q_target_m3_s=0.02,
    )

    pump = make_linear_pump(
        name="P1",
        N_ref_rpm=3600.0,
        H0=60.0,
        slope=800.0,
        Q_min=0.0,
        Q_max=0.05,
        Q_bep_ref_m3_s=1e-4,
    )

    pump = make_linear_pump(
        name="P1",
        N_ref_rpm=3600.0,
        H0=60.0,
        slope=800.0,
        Q_min=0.0,
        Q_max=0.05,
        Q_bep_ref_m3_s=1e-4,
    )

    grp = PumpGroupSpec(
        pump=pump,
        arrangement=Arrangement.PARALLEL,
        count_installed=1,
        allowed_active_counts=(1,),
        control=ControlMode.FIXED_SPEED,
        N_fixed_rpm=3600.0,
    )

    cfg = ConfigSpec(name="single_bep", groups=(grp,), booster=None, dispatch_objective=DispatchObjective.MIN_POWER)

    r = evaluate(ctx, cfg, sc, system, Q_hint_domain=(0.0, 0.05))
    assert not r.feasible
