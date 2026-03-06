from __future__ import annotations

import math

import pytest

from orc_modeling.fluidprops import make_fluid, Q_, ureg


def _assert_positive_quantity(x, unit: str):
    v = x.to(unit).magnitude
    assert math.isfinite(v)
    assert v > 0.0


def _assert_finite_quantity(x, unit: str):
    v = x.to(unit).magnitude
    assert math.isfinite(v)


def _backend_available(name: str, *args, **kwargs) -> bool:
    try:
        make_fluid(*args, backend=name, **kwargs)
        return True
    except Exception:
        return False


HAS_THERMO = _backend_available("thermo", "CO2")
HAS_COOLPROP = _backend_available("coolprop", "CO2")
HAS_REFPROP = _backend_available("refprop", "CO2")


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("thermo", marks=pytest.mark.skipif(not HAS_THERMO, reason="thermo not available")),
        pytest.param("coolprop", marks=pytest.mark.skipif(not HAS_COOLPROP, reason="coolprop not available")),
        pytest.param("refprop", marks=pytest.mark.skipif(not HAS_REFPROP, reason="refprop not available")),
    ],
)
def test_pure_co2_smoke(backend: str):
    fluid = make_fluid("CO2", backend=backend)

    T = Q_(310.0, ureg.kelvin)
    P = Q_(10.0e6, ureg.pascal)

    rho = fluid.rho(T, P)
    h = fluid.h(T, P)
    s = fluid.s(T, P)
    mu = fluid.mu(T, P)
    cp = fluid.cp(T, P)

    _assert_positive_quantity(rho, "kg/m^3")
    _assert_finite_quantity(h, "J/kg")
    _assert_finite_quantity(s, "J/kg/K")
    _assert_positive_quantity(mu, "Pa*s")
    _assert_positive_quantity(cp, "J/kg/K")

    # Speed of sound is available on all three for single-phase pure CO2
    a = fluid.a(T, P)
    _assert_positive_quantity(a, "m/s")


@pytest.mark.skipif(not HAS_COOLPROP, reason="coolprop not available")
def test_coolprop_water_eg_mass_mixture_smoke():
    fluid = make_fluid(
        ["water", "ethylene glycol"],
        backend="coolprop",
        composition=[0.55, 0.45],
        composition_basis="mass",
    )

    T = Q_(298.15, ureg.kelvin)
    P = Q_(300e3, ureg.pascal)

    rho = fluid.rho(T, P)
    cp = fluid.cp(T, P)
    mu = fluid.mu(T, P)
    h = fluid.h(T, P)

    _assert_positive_quantity(rho, "kg/m^3")
    _assert_positive_quantity(cp, "J/kg/K")
    _assert_positive_quantity(mu, "Pa*s")
    _assert_finite_quantity(h, "J/kg")


@pytest.mark.skipif(not HAS_COOLPROP, reason="coolprop not available")
def test_coolprop_water_eg_blocks_incompressible_only_props():
    fluid = make_fluid(
        ["water", "ethylene glycol"],
        backend="coolprop",
        composition=[0.55, 0.45],
        composition_basis="mass",
    )

    T = Q_(298.15, ureg.kelvin)
    P = Q_(300e3, ureg.pascal)

    with pytest.raises(NotImplementedError):
        fluid.cv(T, P)

    with pytest.raises(NotImplementedError):
        fluid.a(T, P)

    with pytest.raises(NotImplementedError):
        fluid.T_crit()


@pytest.mark.skipif(not HAS_THERMO, reason="thermo not available")
def test_thermo_water_eg_mass_mixture_smoke():
    fluid = make_fluid(
        ["water", "ethylene glycol"],
        backend="thermo",
        composition=[0.55, 0.45],
        composition_basis="mass",
    )

    T = Q_(298.15, ureg.kelvin)
    P = Q_(300e3, ureg.pascal)

    rho = fluid.rho(T, P)
    h = fluid.h(T, P)
    s = fluid.s(T, P)
    mu = fluid.mu(T, P)
    cp = fluid.cp(T, P)

    _assert_positive_quantity(rho, "kg/m^3")
    _assert_finite_quantity(h, "J/kg")
    _assert_finite_quantity(s, "J/kg/K")
    _assert_positive_quantity(mu, "Pa*s")
    _assert_positive_quantity(cp, "J/kg/K")


@pytest.mark.skipif(not HAS_THERMO, reason="thermo not available")
def test_thermo_mixture_pure_only_methods_raise():
    fluid = make_fluid(
        ["water", "ethylene glycol"],
        backend="thermo",
        composition=[0.55, 0.45],
        composition_basis="mass",
    )

    with pytest.raises(NotImplementedError):
        fluid.T_crit()

    with pytest.raises(NotImplementedError):
        fluid.p_sat(Q_(298.15, ureg.kelvin))


@pytest.mark.skipif(not HAS_REFPROP, reason="refprop not available")
def test_refprop_co2_n2_mole_mixture_smoke():
    fluid = make_fluid(
        ["CO2", "NITROGEN"],
        backend="refprop",
        composition=[0.98, 0.02],
        composition_basis="mole",
    )

    T = Q_(310.0, ureg.kelvin)
    P = Q_(10.0e6, ureg.pascal)

    rho = fluid.rho(T, P)
    h = fluid.h(T, P)
    s = fluid.s(T, P)
    mu = fluid.mu(T, P)
    cp = fluid.cp(T, P)
    cv = fluid.cv(T, P)
    a = fluid.a(T, P)

    _assert_positive_quantity(rho, "kg/m^3")
    _assert_finite_quantity(h, "J/kg")
    _assert_finite_quantity(s, "J/kg/K")
    _assert_positive_quantity(mu, "Pa*s")
    _assert_positive_quantity(cp, "J/kg/K")
    _assert_positive_quantity(cv, "J/kg/K")
    _assert_positive_quantity(a, "m/s")


@pytest.mark.skipif(not HAS_REFPROP, reason="refprop not available")
def test_refprop_mixture_pure_only_methods_raise():
    fluid = make_fluid(
        ["CO2", "NITROGEN"],
        backend="refprop",
        composition=[0.98, 0.02],
        composition_basis="mole",
    )

    with pytest.raises(NotImplementedError):
        fluid.T_crit()

    with pytest.raises(NotImplementedError):
        fluid.p_sat(Q_(290.0, ureg.kelvin))