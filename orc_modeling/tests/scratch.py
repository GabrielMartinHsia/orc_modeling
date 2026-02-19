#orc_modeling/tests/scratch

from orc_modeling.fluidprops import make_fluid
from orc_modeling.utilities.units import Q_
import matplotlib.pyplot as plt
import numpy as np

ic5 = make_fluid("isopentane", backend="thermo")
# print(ic5.h_sat_liq(1.2))
P_min = Q_(0.01, "bar")
P_max = Q_(35, "bar")

P_range = np.linspace(P_min.magnitude, P_max.magnitude, 100)

def pressure_enthalpy_curve_liq(pressures, fluid):
    H_vector = []
    for p in pressures:
        h = fluid.h_sat_liq(p)
        H_vector.append(h.magnitude)
    return H_vector

H_range = pressure_enthalpy_curve_liq(P_range, ic5)    
plt.plot(H_range, P_range)