# models.py

import numpy as np

def cost_function(rho_cat, delta, eps, A_cell, c_cat):
    """
    Catalyst Cost Function
    ----------------------
    Parameters:
      rho_cat (g/cm³): Catalyst density
      delta   (cm)   : Layer thickness
      eps     (dimensionless): Porosity
      A_cell  (cm²)  : Cell active area
      c_cat   ($/g)  : Catalyst cost per gram

    Returns:
      cost ($): The cost contribution for that catalyst layer
    """
    # mass per area = rho_cat * delta * (1 - eps)
    # cost = mass/area * area * c_cat
    # => cost_layer = [rho_cat*delta*(1-eps) * A_cell * c_cat]
    mass_per_area = rho_cat * delta * (1.0 - eps)
    return mass_per_area * A_cell * c_cat


def eta_total(
    j,         # (A/cm²) operating current density
    j0,        # (A/cm²_active) exchange current density (active area basis)
    S_cat,     # (cm²_active/g or cm²_active/cm³) specific surface area
    epsilon,   # (dimensionless) porosity
    delta,     # (cm) thickness
    a, b,      # (V) Tafel constants
    R, T,      # R (J/(mol*K)), T (K)
    n, F,      # n (dimensionless), F (C/mol)
    C_bulk,    # (mol/cm³) bulk concentration
    D,         # (cm²/s) diffusivity
    tau        # (dimensionless) tortuosity
):
    """
    Total Overpotential Model (Simple Tafel + Simple Concentration)
    ----------------------------------------------------------------
    Activation Overpotential (Tafel form): η_act = a + b * ln(j / j0_geo)
      j0_geo = j0 * S_cat * (1-epsilon)*delta

    Concentration Overpotential (approx):
      η_conc = (R*T/(n*F)) * ln( 1 - j/j_lim )
      j_lim = (n*F * (epsilon/tau)*D * C_bulk) / delta

    Summation: η_total = η_act + η_conc

    Returns:
      eta (V) total
    """
    # 1) Tafel Activation
    j0_geo = j0 * S_cat * (1 - epsilon) * delta
    if j0_geo <= 1e-15:  # avoid log(0)
        return 1e6  # large penalty

    # a + b ln( j / j0_geo )
    eta_act = a + b * np.log(j / j0_geo) if (j>0 and j0_geo>0) else 1e6

    # 2) Concentration Overpotential
    # j_lim = nF * (epsilon/tau*D) * C_bulk / delta
    D_eff = (epsilon/tau)*D
    j_lim = (n*F*D_eff*C_bulk)/delta
    if j_lim <= j or j_lim<=0:
        eta_conc = 1e6
    else:
        fac = (R*T)/(n*F)
        # (R*T/(nF)) ln(1 - j / j_lim)
        part = 1 - (j/j_lim)
        if part <= 1e-15:
            eta_conc = 1e6
        else:
            eta_conc = fac * np.log(part)

    return eta_act + eta_conc
