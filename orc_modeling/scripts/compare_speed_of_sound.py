import numpy as np
from orc_modeling.fluidprops import make_fluid, Q_, ureg

def safe_eval(fluid, T, P):
    try:
        val = fluid.a(Q_(T, ureg.K), Q_(P, ureg.Pa))
        return float(val.to("m/s").magnitude)
    except Exception as e:
        return f"ERR: {type(e).__name__}"

def main():
    fluid_name = "CO2"

    backends = {}
    for b in ["thermo", "coolprop", "refprop"]:
        try:
            backends[b] = make_fluid(fluid_name, backend=b)
            print(f"{b}: OK")
        except Exception as e:
            print(f"{b}: not available ({e})")

    if not backends:
        print("No backends available.")
        return

    # Critical region grid
    T_vals = np.linspace(300, 320, 6)        # around Tc = 304 K
    P_vals = np.linspace(7e6, 12e6, 6)       # around Pc = 7.38 MPa

    print("\nComparing speed of sound [m/s]\n")

    for T in T_vals:
        for P in P_vals:
            results = {}
            for name, fluid in backends.items():
                results[name] = safe_eval(fluid, T, P)

            # Choose reference priority: refprop > coolprop > thermo
            ref = None
            for candidate in ["refprop", "coolprop", "thermo"]:
                if candidate in results and isinstance(results[candidate], float):
                    ref = results[candidate]
                    break

            print(f"T={T:.1f} K  P={P/1e6:.2f} MPa")

            for name, val in results.items():
                if isinstance(val, float) and isinstance(ref, float):
                    rel_err = abs(val - ref) / ref * 100
                    print(f"  {name:9s}: {val:8.2f} m/s  (Δ={rel_err:6.2f}%)")
                else:
                    print(f"  {name:9s}: {val}")

            print("-" * 60)


if __name__ == "__main__":
    main()