# orc_modeling/core/evaluate.py
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .solver import SolverError, SolveResult, bisect_root, find_intersection
from .types import (
    ConfigSpec,
    DispatchObjective,
    EvalResult,
    FailureReason,
    FluidProps,
    OperatingPoint,
    PumpGroupResult,
    Scenario,
    ScoreResult,
    SystemSpec,
)


# -------------------------
# Injection interfaces
# -------------------------

# Build H_sys(Q) [m] from system + scenario + fluid properties
SystemHeadFnBuilder = Callable[[SystemSpec, Scenario, FluidProps], Callable[[float], float]]

# Build H_pump(Q) [m] from config + active counts + scenario + fluid properties
PumpHeadFnBuilder = Callable[[ConfigSpec, Tuple[int, ...], Scenario, FluidProps], Callable[[float], float]]

# Optional: compute per-group details (power, eta, NPSH margins, etc.) at the solved operating Q
DetailsFn = Callable[
    [ConfigSpec, Tuple[int, ...], Scenario, FluidProps, float],
    Tuple[Tuple[PumpGroupResult, ...], Optional[float], Optional[float], Optional[float]],
]
# returns: (group_results, P_total_W, eta_overall, min_npsh_margin_m)


# -------------------------
# Evaluation context
# -------------------------

@dataclass(frozen=True)
class EvalContext:
    """
    Dependency injection for evaluate().

    This keeps core/evaluate.py independent from specific implementations of:
      - fluid property libraries
      - hydraulics models
      - pump curve models & affinity scaling
      - power/efficiency models

    You provide these callables from your domain modules.
    """
    fluid_props: Callable[[Scenario], FluidProps]
    build_system_head: SystemHeadFnBuilder
    build_pump_head: PumpHeadFnBuilder

    # Optional: richer post-solve calculations (power, per-group speeds, NPSH, etc.)
    compute_details: Optional[DetailsFn] = None


# -------------------------
# Dispatch helpers
# -------------------------

def _allowed_active_counts(count_installed: int, allowed: Optional[Tuple[int, ...]]) -> Tuple[int, ...]:
    if count_installed < 0:
        raise ValueError("count_installed must be >= 0")
    if allowed is None:
        return tuple(range(count_installed + 1))
    # basic sanity
    for k in allowed:
        if k < 0 or k > count_installed:
            raise ValueError(f"allowed_active_counts contains {k}, outside [0, {count_installed}]")
    # preserve user order, but ensure unique
    seen = set()
    out: List[int] = []
    for k in allowed:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return tuple(out)


def _enumerate_dispatches(config: ConfigSpec) -> Iterable[Tuple[int, ...]]:
    """
    Yields tuples of active counts aligned with config.groups order.
    Booster (if present) is not included here; treat booster separately if desired.
    """
    per_group_options: List[Tuple[int, ...]] = []
    for grp in config.groups:
        per_group_options.append(_allowed_active_counts(grp.count_installed, grp.allowed_active_counts))
    yield from product(*per_group_options)


def _pick_initial_guess(config: ConfigSpec, scenario: Scenario, Q_min: float = 0.0, Q_max: float = 1.0) -> Tuple[float, float]:
    """
    Choose a reasonable (x0, step) for intersection solving.

    - x0: prefer scenario.Q_target if provided, else mid of [Q_min, Q_max]
    - step: 10% of span (fallback to small)
    """
    if scenario.Q_target_m3_s is not None:
        x0 = float(scenario.Q_target_m3_s)
    else:
        x0 = 0.5 * (Q_min + Q_max)
    span = max(Q_max - Q_min, 1e-9)
    step = 0.1 * span
    return x0, step


# -------------------------
# Core API
# -------------------------

def solve_intersection(
    H_pump: Callable[[float], float],
    H_sys: Callable[[float], float],
    *,
    Q_domain: Optional[Tuple[float, float]] = None,
    x0: float = 0.0,
    step: float = 0.1,
) -> SolveResult:
    """
    Solve H_pump(Q) = H_sys(Q).

    If Q_domain is provided, attempt to bracket and solve within that interval.
    Otherwise, fall back to generic root bracketing around (x0, step).

    Debug instrumentation:
      - prints f(qmin), f(qmax) when Q_domain is provided
      - prints where bisect_root and SolverError are imported from
      - prints the SolverError message if bisect_root raises unexpectedly
    """
    # import inspect  # stdlib; safe for temporary debug

    def h(q: float) -> float:
        return float(H_pump(q) - H_sys(q))

    if Q_domain is not None:
        qmin, qmax = Q_domain
        fmin = h(qmin)
        fmax = h(qmax)

        # # debug
        # print(f"[solve_intersection] qmin={qmin}, fmin={fmin}, qmax={qmax}, fmax={fmax}")

        # # sanity checks on imports (helps detect shadowed modules)
        # print("[debug] bisect_root from:", inspect.getsourcefile(bisect_root))
        # print("[debug] SolverError from:", inspect.getsourcefile(SolverError))

        if fmin == 0.0:
            return SolveResult(x=qmin, f_x=0.0, iterations=0, converged=True)
        if fmax == 0.0:
            return SolveResult(x=qmax, f_x=0.0, iterations=0, converged=True)

        if fmin * fmax < 0.0:
            try:
                return bisect_root(h, qmin, qmax)
            except SolverError as e:
                # print("[debug] bisect_root raised SolverError:", repr(e))
                # print("[debug] f(qmin), f(qmax) =", fmin, fmax)
                raise

        raise SolverError("No sign change in provided Q_domain")

    # no domain provided: use generic intersection finder
    return find_intersection(H_pump, H_sys, x0=x0, step=step)



def evaluate(
    ctx: EvalContext,
    config: ConfigSpec,
    scenario: Scenario,
    system: SystemSpec,
    *,
    Q_hint_domain: Optional[Tuple[float, float]] = None,
) -> EvalResult:
    """
    Evaluate one (config, scenario, system).

    Workflow:
      1) compute fluid properties for scenario
      2) enumerate dispatch combinations (active pump counts per group)
      3) for each dispatch:
         - build H_sys(Q) and H_pump(Q)
         - solve intersection H_pump(Q) = H_sys(Q)
         - optionally compute details (power, NPSH, etc.)
      4) select best feasible dispatch by config.dispatch_objective

    Parameters
    ----------
    ctx:
        EvalContext providing injected builders.
    config:
        Pump train configuration (with lead/lag flexibility expressed via allowed_active_counts).
    scenario:
        Operating case.
    system:
        System hydraulic spec.
    Q_hint_domain:
        Optional (Q_min, Q_max) to help choose a reasonable initial guess/step.

    Returns
    -------
    EvalResult
    """
    props = ctx.fluid_props(scenario)
    H_sys = ctx.build_system_head(system, scenario, props)

    # Domain hint only affects solver initial guess/step
    Q_min, Q_max = Q_hint_domain if Q_hint_domain is not None else (0.0, 1.0)
    x0, step = _pick_initial_guess(config, scenario, Q_min=Q_min, Q_max=Q_max)

    best: Optional[EvalResult] = None
    best_metric: Optional[float] = None  # lower is better for MIN_POWER, MIN_ACTIVE

    any_solved = False
    any_feasible = False
    last_failure: FailureReason = FailureReason.NO_INTERSECTION

    for active_counts in _enumerate_dispatches(config):
        H_pump = ctx.build_pump_head(config, active_counts, scenario, props)

        # try:
        #     sol: SolveResult = find_intersection(H_pump, H_sys, x0=x0, step=step)
        # except SolverError:
        #     last_failure = FailureReason.NO_INTERSECTION
        #     continue
        try:
            sol = solve_intersection(H_pump, H_sys, Q_domain=Q_hint_domain, x0=x0, step=step)
        except SolverError:
            last_failure = FailureReason.NO_INTERSECTION
            continue


        any_solved = True
        if not sol.converged:
            last_failure = FailureReason.NUMERICAL_FAILURE
            continue

        Q_star = sol.x
        op = OperatingPoint(Q_m3_s=Q_star, H_system_m=float(H_sys(Q_star)), H_pump_m=float(H_pump(Q_star)))

        # Default: treat an intersection as feasible (until details layer enforces constraints)
        feasible = True
        failure_reasons: Tuple[FailureReason, ...] = ()

        groups: Tuple[PumpGroupResult, ...] = ()
        P_total_W: Optional[float] = None
        eta_overall: Optional[float] = None
        npsh_margin_m: Optional[float] = None

        if ctx.compute_details is not None:
            try:
                groups, P_total_W, eta_overall, npsh_margin_m = ctx.compute_details(
                    config, active_counts, scenario, props, Q_star
                )
                # If DetailsFn wants to mark infeasible, it can do so by
                # encoding FailureReason(s) in group results or by raising; for now,
                # keep it simple: infeasibility should be reported by raising SolverError
                # OR by returning P_total_W=None plus a sentinel group result.
            except SolverError:
                feasible = False
                failure_reasons = (FailureReason.SYSTEM_CONSTRAINT,)
                last_failure = FailureReason.SYSTEM_CONSTRAINT
            except Exception:
                feasible = False
                failure_reasons = (FailureReason.NUMERICAL_FAILURE,)
                last_failure = FailureReason.NUMERICAL_FAILURE

            # except Exception as e:
            #     raise

        if not feasible:
            any_feasible = any_feasible or False
            continue

        any_feasible = True

        result = EvalResult(
            config_name=config.name,
            scenario_name=scenario.name,
            feasible=True,
            failure_reasons=(),
            op=op,
            groups=groups,
            P_total_W=P_total_W,
            eta_overall=eta_overall,
            npsh_margin_m=npsh_margin_m,
        )

        # Select among feasible dispatches
        if config.dispatch_objective == DispatchObjective.MIN_ACTIVE:
            metric = float(sum(active_counts))
        else:  # MIN_POWER default
            # If power isn't computed yet, fall back to MIN_ACTIVE as a deterministic tie-breaker
            metric = float(P_total_W) if P_total_W is not None else float(sum(active_counts))

        if best is None or metric < (best_metric if best_metric is not None else float("inf")):
            best = result
            best_metric = metric

    if best is not None:
        return best

    # No feasible dispatch found — report the best-known failure mode
    if not any_solved:
        reasons = (FailureReason.NO_INTERSECTION,)
    else:
        reasons = (last_failure,)

    return EvalResult(
        config_name=config.name,
        scenario_name=scenario.name,
        feasible=False,
        failure_reasons=reasons,
        op=None,
        groups=(),
        P_total_W=None,
        eta_overall=None,
        npsh_margin_m=None,
    )


def score(
    ctx: EvalContext,
    config: ConfigSpec,
    scenarios: Sequence[Scenario],
    system: SystemSpec,
    *,
    Q_hint_domain: Optional[Tuple[float, float]] = None,
) -> ScoreResult:
    """
    Aggregate evaluate() across a set of scenarios.

    - feasible_fraction: fraction of scenario weight that is feasible
    - weighted_energy_Wh: if power is solved, sum(P[W] * weight[h]) in Wh
      (Interpretation of Scenario.weight is matter of discretion; common choice is hours.)

    Returns ScoreResult including per-scenario EvalResult for diagnostics.
    """
    results: List[EvalResult] = []
    total_weight = 0.0
    feasible_weight = 0.0

    energy_Wh = 0.0
    have_power = True

    for sc in scenarios:
        r = evaluate(ctx, config, sc, system, Q_hint_domain=Q_hint_domain)
        results.append(r)

        w = float(sc.weight)
        total_weight += w

        if r.feasible:
            feasible_weight += w
            if r.P_total_W is not None:
                energy_Wh += float(r.P_total_W) * w  # W * h = Wh, if weight is hours
            else:
                have_power = False

    feasible_fraction = (feasible_weight / total_weight) if total_weight > 0 else 0.0
    feasible = feasible_fraction >= 1.0 - 1e-12  # strict by default

    return ScoreResult(
        config_name=config.name,
        feasible=feasible,
        feasible_fraction=feasible_fraction,
        weighted_energy_Wh=energy_Wh if have_power else None,
        weighted_cost_usd=None,  # compute later when we have cost models + tariff logic
        results=tuple(results),
    )
