# Fluid Property Backends

This project supports multiple thermodynamic property libraries through a
common interface. The goal is to allow ORC modeling code to use fluid
properties without depending on a specific backend.

Current backends:

- `thermo` (Caleb Bell's thermo library)
- `CoolProp`
- `REFPROP`

All backends implement the same property API.

---

# Architecture Overview
User code --> Fluid(unit-aware wrapper) --> FluidBackend(SI-only interface) --> Backend implementation (`thermo_backend.py`, `coolprop_backend.py`, `refprop_backend.py`)


The separation ensures:

- **User-facing API handles units**
- **Backends operate only in SI floats**
- **Property engines are interchangeable**

---

# FluidSpec (Canonical Fluid Description)

Mixtures and pure fluids are described using a `FluidSpec`.
`FluidSpec(
    ids=("water", "ethylene glycol"),
    composition=(0.55, 0.45),
    composition_basis="mass"
)`


Key features:

- Supports **pure fluids** and **mixtures**
- Composition may be **mass** or **mole basis**
- Fractions are normalized automatically

Pure fluid example:
`FluidSpec(ids=("CO2",))`


---

# Backend Translation

Each backend translates `FluidSpec` into its native representation.

| Backend | Native Format |
|-------|------|
| thermo | component list + mole fractions |
| REFPROP | fluid string + mole fractions |
| CoolProp | fluid string |

Example translation:
["water", "ethylene glycol"] --> CoolProp:"INCOMP::MEG[0.45]"


Adapters inside each backend perform this translation.

---

# Supported Properties

All backends implement the same core interface.

State properties:
h(T,P)
s(T,P)
rho(T,P)
mu(T,P)
cp(T,P)
cv(T,P)
a(T,P) # speed of sound


Saturation properties (pure fluids only):
p_sat(T)
T_sat(P)

h_sat_liq(P)
h_sat_vap(P)

s_sat_liq(P)
s_sat_vap(P)

rho_sat_liq(P)
rho_sat_vap(P)

Critical properties:
T_crit()
p_crit()


---

# Units

User-facing functions accept and return `pint.Quantity`.

Example:
T = Q_(120, "degF")
P = Q_(75, "psi")

rho = fluid.rho(T, P)


Backends internally operate on **SI floats only**.

---

# Mixture Support

Mixtures are supported for:

| Backend | Mixtures |
|-------|------|
| thermo | ✓ |
| REFPROP | ✓ |
| CoolProp | limited |

CoolProp currently supports: water + ethylene glycol


through the incompressible fluid model.

---

# Testing

Smoke tests are located in: orc_modeling/tests/test_mixtures.py


These verify:

- pure fluid behavior
- mixture initialization
- property evaluation
- backend availability

---

# Design Goals

1. **Backend interchangeability**
2. **Strict SI backend interface**
3. **Unit-safe user API**
4. **Minimal coupling to property libraries**
5. **Extendable mixture support**

---

# Future Improvements

Possible extensions:

- mixture critical properties
- generalized CoolProp mixture support
- caching repeated property calls
- vectorized property evaluation

