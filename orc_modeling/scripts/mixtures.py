from orc_modeling.fluidprops import make_fluid, Q_, ureg

wg = make_fluid(
    ["water", "ethylene glycol"],
    backend="refprop",
    composition=[0.55, 0.45],
    composition_basis="mass",
)

T,P = Q_(122, "degF"), Q_(75.4, "psi")

cp = wg.cp(T=T, P=P)
rho = wg.rho(T=T, P=P)
print(f"backend={wg.backend}")
print(f"cp: {cp}")
print(f"rho: {rho}")