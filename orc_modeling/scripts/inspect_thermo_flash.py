from thermo import ChemicalConstantsPackage, CEOSLiquid, CEOSGas, FlashPureVLS, PRMIX

def main():
    fluid_id = "Water"
    constants, correlations = ChemicalConstantsPackage.from_IDs([fluid_id])
    eos_kwargs = dict(Tcs=constants.Tcs, Pcs=constants.Pcs, omegas=constants.omegas)
    liquid = CEOSLiquid(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)
    gas = CEOSGas(PRMIX, HeatCapacityGases=correlations.HeatCapacityGases, eos_kwargs=eos_kwargs)

    flasher = FlashPureVLS(constants=constants, correlations=correlations, gas=gas, liquids=[liquid], solids=[])
    r = flasher.flash(T=300.0, P=101325.0)

    interesting_prefixes = ("H", "S", "Cp", "Cv", "MW", "V", "mu")
    interesting = sorted({k for k in dir(r) if k.startswith(interesting_prefixes)})

    print("Result type:", type(r))
    print("Interesting dir() entries:")
    for k in interesting:
        print(" ", k)

if __name__ == "__main__":
    main()
