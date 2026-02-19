from orc_modeling.fluidprops import make_fluid, Q_
from orc_modeling.viz import plot_ph, plot_ts, ProcessPoint
import matplotlib.pyplot as plt

f = make_fluid("isopentane", backend="thermo", return_quantity=True)

Pc = f.p_crit()          # pint Quantity (Pa) by default
Pmin = Q_(0.05, "bar")   # low enough to show the wide dome base
Pmax = Pc                # request "to critical" (viz will clamp to ~0.999*Pc)

pts = [
    ProcessPoint("1", T=Q_(31.9, "degC"), P=Q_(1.25, "bar")),
    ProcessPoint("2", T=Q_(10.24, "degC"), P=Q_(0.636, "bar")),
]

plot_ph(f, Pmin, Q_(2.5, "bar"), points=pts, n=400, y_log=False)
plot_ts(f, Pmin, Q_(2.5, "bar"), points=pts, n=400)

plt.show()


