from orc_modeling.core.types import (
    Arrangement,
    ConfigSpec,
    ControlMode,
    FluidProps,
    PumpGroupSpec,
    PumpLimits,
    PumpSpec,
    PumpCurveRef,
    Scenario,
)
from orc_modeling.pumps.results import compute_operating_point
from orc_modeling.fluidprops import make_fluid
from orc_modeling.utilities.units import Q_


def main():
    # Simple linear head curve at reference speed (static PumpCurveRef)
    H0 = 50.0
    slope = 2000.0  # m per (m^3/s)

    def H_m(Q: float) -> float:
        return float(H0 - slope * float(Q))

    curve_ref = PumpCurveRef(
        H_m=H_m,
        eta=None,
        NPSHr_m=None,
        Q_min_m3_s=0.0,
        Q_max_m3_s=0.1,
        Q_bep_ref_m3_s=0.04,
    )

    pump = PumpSpec(
        name="ToyPump",
        curve_ref=curve_ref,
        N_ref_rpm=3600.0,
        limits=PumpLimits(),
        cost=None,
    )

    group = PumpGroupSpec(
        pump=pump,
        arrangement=Arrangement.PARALLEL,
        count_installed=2,
        control=ControlMode.FIXED_SPEED,
        N_fixed_rpm=3600.0,
    )

    config = ConfigSpec(name="test_config", groups=(group,))


    scenario = Scenario(
        name="test_scenario",
        T_suction_K=300.0,
        P_suction_Pa=2.0e5,
        N_vfd_cmd_rpm=None,
    )

    fluid = make_fluid("isopentane")
    T_suct, P_suct = Q_(90, "degF"), Q_(0.79, "bar")
    rho = fluid.rho(T=T_suct, P=P_suct)
    mu = fluid.mu(T=T_suct, P=P_suct)
    P_vap = fluid.p_vap(T=T_suct)
    
    props = FluidProps(rho_kg_m3=rho, mu_Pa_s=mu, Pvap_Pa=P_vap)

    res = compute_operating_point(
        config=config,
        active_counts=(2,),
        scenario=scenario,
        props=props,
        Q_total=0.04,
    )

    print("=== TEST OUTPUT ===")
    print("Total P (W):", res.P_total_W)
    print("Overall eta:", res.eta_overall)

    g = res.group_results[0]
    print("Pump:", g.base.pump_name)
    print("H_group (m):", g.H_group_m)
    print("P_group (W):", g.base.P_group_W)
    print("Ns_us (rep):", g.Ns_us)
    if g.fixed:
        print("Ns_us (fixed):", g.fixed.Ns_us)
    if g.vfd:
        print("Ns_us (vfd):", g.vfd.Ns_us)


if __name__ == "__main__":
    main()
