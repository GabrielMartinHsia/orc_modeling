from orc_modeling.fluidprops import make_fluid, Q_

c3 = make_fluid("propane")

T_cnd = Q_(110, "degF")
p_sat = c3.p_sat(T=T_cnd)
rho_sat = c3.rho_sat_liq(P=p_sat)

mdot = Q_(407, "kg/s")
V_dot_m3_s = mdot / rho_sat
V_dot_gpm = V_dot_m3_s.to("gal/min")

print(f"c3 density at {T_cnd}: {rho_sat:6.2f~P}"
      f"c3 mass flow:   {mdot:6.2f~P}\t|\t",
      f"c3 Q m3/s:   {V_dot_m3_s:6.2f~P}\t|\t",
      f"c3 Q gal/min:   {V_dot_gpm:6.2f~P}\t|\t",
)

