# orc_modeling/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# -------------------------
# Core conventions (SI)
# -------------------------
# T: K
# P: Pa
# rho: kg/m^3
# mu: Pa*s
# head: m (of pumped fluid)
# g: m/s^2
# Q (volumetric flow): m^3/s
# mdot: kg/s
# power: W
# NPSH: m


# -------------------------
# Generic callable curve types
# -------------------------
HeadCurve = Callable[[float], float]          # H(Q) -> m
EffCurve = Callable[[float], float]           # eta(Q) -> 0..1
NpshrCurve = Callable[[float], float]         # NPSHr(Q) -> m


class ControlMode(str, Enum):
    FIXED_SPEED = "fixed_speed"
    VARIABLE_SPEED = "variable_speed"


class Arrangement(str, Enum):
    SERIES = "series"
    PARALLEL = "parallel"


class DispatchObjective(str, Enum):
    """
    Objective used to select among multiple feasible dispatch combinations
    (i.e., different numbers of active pumps across groups).

    MIN_POWER
        Choose the combination with lowest total power consumption.

    MIN_ACTIVE
        Choose the combination with the fewest active pumps (useful when
        minimizing equipment wear or switching events).
    """
    MIN_POWER = "min_power"         # typical
    MIN_ACTIVE = "min_active"       # fewer running pumps if multiple solutions


class FailureReason(str, Enum):
    NO_INTERSECTION = "no_intersection"
    PUMP_MIN_FLOW = "pump_below_min_flow"
    PUMP_MAX_FLOW = "pump_above_max_flow"
    PUMP_MAX_SPEED = "pump_exceeds_max_speed"
    PUMP_MIN_SPEED = "pump_below_min_speed"
    PUMP_MAX_POWER = "pump_exceeds_max_power"
    NPSH_INSUFFICIENT = "npsh_insufficient"
    SYSTEM_CONSTRAINT = "system_constraint"
    NUMERICAL_FAILURE = "numerical_failure"


# -------------------------
# Scenario + weighting
# -------------------------
@dataclass(frozen=True)
class Scenario:
    """One operating condition.

    Keep this 'thin': only what's needed to define the operating case.
    Anything derived belongs in evaluate().
    """
    name: str

    # Thermal/pressure boundary conditions for fluid properties & suction conditions
    T_suction_K: float
    P_suction_Pa: float

    # Demand specification (choose one; you can ignore the other)
    Q_target_m3_s: Optional[float] = None
    mdot_target_kg_s: Optional[float] = None

    # If you need discharge pressure constraints, include here (optional)
    P_discharge_Pa: Optional[float] = None

    # Operating weight (e.g., fraction of year or hours). Default=1 for unweighted sweeps.
    weight: float = 1.0

    # Command speed for VFD driven pump(s):
    N_vfd_cmd_rpm: Optional[float] = None


# -------------------------
# Fluid properties interface result (computed elsewhere)
# -------------------------
@dataclass(frozen=True)
class FluidProps:
    """Minimum set of properties needed for hydraulics + NPSH.

    Produced by fluidprops backend; SI floats only.
    """
    rho_kg_m3: float
    mu_Pa_s: float
    Pvap_Pa: float


# -------------------------
# Hydraulic system definition
# -------------------------
@dataclass(frozen=True)
class SuctionSpec:
    """Suction boundary conditions and losses upstream of the pump suction flange."""
    # Elevation difference from suction free surface (or reference) to pump datum.
    # Positive if free surface is above pump (helps NPSH), negative if below.
    dz_static_m: float = 0.0

    # Lumped suction-side loss coefficient expressed as head loss = K * (v^2 / 2g)
    # or you can interpret this as 'equivalent K' for the suction line.
    K_suction: float = 0.0

    # If suction is from a vessel at some pressure different than scenario.P_suction_Pa,
    # you can override here (else use Scenario P_suction_Pa)
    P_source_Pa: Optional[float] = None


@dataclass(frozen=True)
class DischargeSpec:
    """Discharge-side static head and losses downstream of pump discharge flange."""
    dz_static_m: float = 0.0
    K_discharge: float = 0.0


@dataclass(frozen=True)
class SystemSpec:
    """Hydraulic system model inputs.

    Transport/hydraulics will interpret these and build H_sys(Q).
    """
    suction: SuctionSpec = field(default_factory=SuctionSpec)
    discharge: DischargeSpec = field(default_factory=DischargeSpec)

    # Gravity constant used across the project
    g_m_s2: float = 9.80665


# -------------------------
# Pump curve + pump specification
# -------------------------
@dataclass(frozen=True)
class PumpCurveRef:
    """A pump's reference curves at a reference speed.

    The curve functions should be monotonic/sane in their intended domain.
    No fitting logic here—only the callable results.
    """
    H_m: HeadCurve
    eta: Optional[EffCurve] = None
    NPSHr_m: Optional[NpshrCurve] = None

    # Validity domain for Q to avoid extrapolation surprises
    Q_min_m3_s: float = 0.0
    Q_max_m3_s: float = 1.0

    # BEP at reference speed
    Q_bep_ref_m3_s: float | None = None


@dataclass(frozen=True)
class PumpLimits:
    """Operating constraints for a pump/motor combination."""
    Q_min_m3_s: Optional[float] = None
    Q_max_m3_s: Optional[float] = None

    N_min_rpm: Optional[float] = None
    N_max_rpm: Optional[float] = None

    P_max_W: Optional[float] = None

    # NPSH margin requirement: require NPSHa >= NPSHr + margin (in meters)
    NPSH_margin_m: float = 0.0


@dataclass(frozen=True)
class CostModel:
    """Extremely simple cost placeholders (optional).

    Keep it dumb now; you can replace with richer model later.
    """
    capex_usd: float = 0.0
    # If you want to roll up OPEX in scoring:
    electricity_usd_per_kWh: float = 0.0


@dataclass(frozen=True)
class PumpSpec:
    """A pump at reference speed with optional limits/cost."""
    name: str
    curve_ref: PumpCurveRef

    # Reference speed for curve data (e.g., 1800 or 3600 rpm)
    N_ref_rpm: float

    limits: PumpLimits = field(default_factory=PumpLimits)
    cost: CostModel = field(default_factory=CostModel)


# -------------------------
# Configuration specification
# -------------------------
@dataclass(frozen=True)
class PumpGroupSpec:
    """
    A grouping of identical pumps arranged in SERIES or PARALLEL within the system.

    This represents physical pumps installed in the plant (not internal
    impeller "stages" of a multistage pump).

    The group may contain multiple identical pumps. At any operating point,
    some subset of the installed pumps may be active. Lead/lag behavior is
    modeled by specifying which active pump counts are allowed.

    Parameters
    ----------
    pump :
        The PumpSpec describing the pump model used in this group.

    arrangement :
        SERIES or PARALLEL relationship among pumps within this group.

    count_installed :
        Total number of identical pumps physically installed in this group.

    allowed_active_counts :
        Tuple of permitted numbers of active pumps. If None, all values
        from 0 through count_installed are allowed. This enables modeling
        of lead/lag dispatch and staging logic.

    control :
        FIXED_SPEED or VARIABLE_SPEED operation for pumps in this group.

    N_fixed_rpm :
        Required if control == FIXED_SPEED. The operating speed.

    N_min_rpm, N_max_rpm :
        Optional speed bounds for VARIABLE_SPEED operation. If not provided,
        PumpSpec limits are used.
    """
    pump: PumpSpec
    arrangement: Arrangement

    # How many identical pumps are installed in this group
    count_installed: int = 1

    # How many of these identical pumps have a VFD
    count_vfd_installed: int = 0

    # Which numbers of pumps may be active (defaults to all 0..count_installed)
    allowed_active_counts: Optional[Tuple[int, ...]] = None

    control: ControlMode = ControlMode.FIXED_SPEED
    N_fixed_rpm: Optional[float] = None
    N_min_rpm: Optional[float] = None
    N_max_rpm: Optional[float] = None



@dataclass(frozen=True)
class ConfigSpec:
    """
    Full pump train configuration.

    Groups are applied in SERIES order. Within each group,
    pumps may be arranged in SERIES or PARALLEL.

    Dispatch logic determines how many pumps in each group are active
    for a given scenario, subject to allowed_active_counts.

    If allowed_active_counts of PumpGroupSpec is None, all counts from 0 
    through count_installed are allowed. This enables modeling of lead/lag
    dispatch and staging logic.
    """
    name: str
    groups: Tuple[PumpGroupSpec, ...] = ()
    booster: Optional[PumpGroupSpec] = None

    dispatch_objective: DispatchObjective = DispatchObjective.MIN_POWER




# -------------------------
# Evaluation results
# -------------------------
@dataclass(frozen=True)
class OperatingPoint:
    """A solved operating point for the full configuration."""
    Q_m3_s: float
    H_system_m: float
    H_pump_m: float


@dataclass(frozen=True)
class PumpGroupResult:
    """Per-group results (aggregated across pumps in that group)."""
    group_index: int
    pump_name: str
    arrangement: Arrangement

    count_installed: int
    count_active: int

    control: ControlMode
    N_rpm: Optional[float] = None
    eta: Optional[float] = None

    H_group_m: Optional[float] = None
    P_group_W: Optional[float] = None

    NPSHa_m: Optional[float] = None
    NPSHr_m: Optional[float] = None



@dataclass(frozen=True)
class EvalResult:
    config_name: str
    scenario_name: str

    feasible: bool
    failure_reasons: Tuple[FailureReason, ...] = ()

    op: Optional[OperatingPoint] = None
    groups: Tuple[PumpGroupResult, ...] = ()

    # Rollups
    P_total_W: Optional[float] = None
    eta_overall: Optional[float] = None  # optional: hydraulic power / electrical power
    npsh_margin_m: Optional[float] = None  # min margin across relevant groups


@dataclass(frozen=True)
class ScoreResult:
    """Aggregated scoring across scenarios."""
    config_name: str

    feasible: bool
    feasible_fraction: float

    # Weighted sums (if scenario weights represent hours, you can interpret energy directly)
    weighted_energy_Wh: Optional[float] = None
    weighted_cost_usd: Optional[float] = None

    # Keep per-scenario results so failures are diagnosable
    results: Tuple[EvalResult, ...] = ()
