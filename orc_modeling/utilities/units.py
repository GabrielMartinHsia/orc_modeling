from __future__ import annotations
from typing import Union
import pint

'''
SI units internally, with optional pint at boundaries
'''

# Create one registry for the whole project
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
ureg.default_format = "~P"
ureg.formatter.use_unicode = True

NumberOrQty = Union[float, int, pint.Quantity]

# Canonical units for this project (mass-basis SI)
U_T = ureg.kelvin
U_P = ureg.pascal
U_S = ureg.joule / ureg.kilogram / ureg.kelvin
U_H = ureg.joule / ureg.kilogram
U_RHO = ureg.kilogram / (ureg.meter ** 3)
U_MU = ureg.pascal * ureg.second
U_CP = ureg.joule / (ureg.kilogram * ureg.kelvin)
U_CV = ureg.joule / (ureg.kilogram * ureg.kelvin)
U_A = ureg.meter / ureg.second

# Clean boundary conversions: 
# ...if a user passes a float --> assume SI (canonical)
def to_si(value: NumberOrQty, unit) -> float:
    """Accept float (assumed already SI) or pint.Quantity; return SI float."""
    if isinstance(value, pint.Quantity):
        return float(value.to(unit).magnitude)
    return float(value)

#...and if a user passes a Quantity --> convert to SI (canonical)
def as_qty(value_si: float, unit, return_quantity: bool) -> NumberOrQty:
    """Return Quantity (preferred) or raw float SI depending on flag."""
    if return_quantity:
        return Q_(value_si, unit)
    return float(value_si)

def ensure_qty(value: NumberOrQty, unit):
    """Return value as Quantity in given unit."""
    if isinstance(value, pint.Quantity):
        return value.to(unit)
    return Q_(value, unit)

def to_unit(value: NumberOrQty, unit) -> float:
    """Return float magnitude in the requested unit."""
    return float(ensure_qty(value, unit).magnitude)

def si_to_unit(value_si: NumberOrQty, unit_si, unit_out) -> float:
    # floats assumed in unit_si
    if isinstance(value_si, pint.Quantity):
        return float(value_si.to(unit_out).magnitude)
    return float(Q_(value_si, unit_si).to(unit_out).magnitude)


