from orc_modeling.fluidprops import make_fluid, Q_
from orc_modeling.viz import ProcessPoint
from orc_modeling.viz.diagrams_plotly import plot_ph_plotly, plot_ts_plotly

f = make_fluid("Isopentane", backend="thermo", return_quantity=True)
Pmin = Q_(0.05, "bar")
Pmax = f.p_crit()

design_case = [
    ProcessPoint("F105", T=Q_(86.6, "degC"), P=Q_(1.441, "bar")),      #Turbine exit
    ProcessPoint("F119", T=Q_(86.6, "degC"), P=Q_(1.418, "bar")),      #Recuperator inlet
    ProcessPoint("F106", T=Q_(51.1, "degC"), P=Q_(1.376, "bar")),      #Recuperator exit
    ProcessPoint("F107", T=Q_(51.1, "degC"), P=Q_(1.360, "bar")),      #ACC inlet
    ProcessPoint("F112", T=Q_(31.9, "degC"), P=Q_(1.325, "bar")),      #ACC exit
    ProcessPoint("F100", T=Q_(31.9, "degC"), P=Q_(1.275, "bar")),      #Suction strainer inlet
    ProcessPoint("F111", T=Q_(31.9, "degC"), P=Q_(1.248, "bar")),      #Feed pump inlet
    ProcessPoint("F101", T=Q_(33.9, "degC"), P=Q_(32.011, "bar")),     #Feed pump discharge
    ProcessPoint("F108", T=Q_(33.9, "degC"), P=Q_(31.909, "bar")),      #Liq feed at recuperator inlet
    ProcessPoint("F102", T=Q_(62.1, "degC"), P=Q_(31.211, "bar")),       #Liq feed at recup exit
    ProcessPoint("F117", T=Q_(62.1, "degC"), P=Q_(31.068, "bar")),       #Liq feed at preheater inlet (note: two feed streams have converged)
    ProcessPoint("F118", T=Q_(174.3, "degC"), P=Q_(29.516, "bar")),       #Liq feed at preheater exit
    ProcessPoint("F103", T=Q_(174.3, "degC"), P=Q_(29.442, "bar")),       #Liq feed at evap inlet (note: four parallel streams)
    ProcessPoint("F104", T=Q_(179.1, "degC"), P=Q_(29.349, "bar")),       #Vap feed at evap exit (note: four parallel streams)
    ProcessPoint("F109", T=Q_(179.1, "degC"), P=Q_(29.250, "bar")),       #Vap feed at turbine inlet (note: two parallel streams per turbine)
]

for p in design_case:
    rho = f.rho(p.T, p.P)
    h = f.h(p.T, p.P)
    print(f"{p.label}: {rho:6.1f~P} | {h:6.2f~P}")

fig = plot_ts_plotly(f, Pmin, Pmax, points=design_case)
fig.show()

fig2 = plot_ph_plotly(f, Pmin, Pmax, points=design_case, y_log=False)
fig2.show()