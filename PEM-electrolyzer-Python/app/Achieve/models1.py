# utils/models.py

import numpy as np

def cost_function(L, delta, epsilon, A_cell, c_cat, rho_cat):
    """
    Cost Function
    -------------
    Parameters:
    - L        (g/cm^2)    : Catalyst loading [not always used explicitly if derived from δ, (1-ε), etc.]
    - delta    (cm)        : Catalyst layer thickness
    - epsilon  (dimensionless) : Porosity
    - A_cell   (cm^2)      : Cell active area
    - c_cat    ($/g)       : Cost of catalyst per gram
    - rho_cat  (g/cm^3)    : Catalyst density

    Returns:
    - cost ($) : Catalyst cost for the entire cell area
      C = rho_cat * delta * (1 - epsilon) * A_cell * c_cat
    """
    return rho_cat * delta * (1 - epsilon) * A_cell * c_cat


def eta_activation(j, j0, S_cat, epsilon, delta, a, b):
    """
    Activation Overpotential (Tafel form)
    -------------------------------------
    η_act = a + b * log( j / j0_geo )
    where j0_geo = j0 * S_cat * (1 - epsilon) * delta

    Parameters:
    - j       (A/cm^2)    : Operating current density per geometric area
    - j0      (A/cm^2_active) : Exchange current density per active surface area
    - S_cat   (cm^2_active/cm^3 or cm^2_active/g, etc.): Specific surface area
    - epsilon (dimensionless) : Porosity
    - delta   (cm)        : Layer thickness
    - a       (V)         : Tafel coefficient (constant term)
    - b       (V)         : Tafel slope (associated with log term)

    Returns:
    - η_act (V)
    """
    # Exchange current density per geometric area
    j0_geo = j0 * S_cat * (1 - epsilon) * delta

    # Avoid taking log of non-positive numbers
    if j0_geo <= 0 or j <= 0:
        return 1e6  # large penalty

    return a + b * np.log(j / j0_geo)


def eta_concentration(j, R, T, n, F, C_bulk, D, epsilon, tau, delta):
    """
    Concentration Overpotential
    ---------------------------
    η_conc = (R*T/(n*F)) * ln(1 - j/j_lim)
    where j_lim = (n*F*D_eff*C_bulk)/delta
          D_eff = (epsilon/tau) * D

    Parameters:
    - j       (A/cm^2)    : Operating current density (geometric)
    - R       (J/(mol*K)) : Gas constant
    - T       (K)         : Temperature
    - n       (dimensionless) : Number of electrons per reaction
    - F       (C/mol)     : Faraday constant
    - C_bulk  (mol/cm^3)  : Bulk concentration of reactant
    - D       (cm^2/s)    : Diffusivity in free medium
    - epsilon (dimensionless) : Porosity
    - tau     (dimensionless) : Tortuosity
    - delta   (cm)        : Layer thickness

    Returns:
    - η_conc (V)
    """
    D_eff = (epsilon / tau) * D  # effective diffusivity (cm^2/s)
    # limiting current density (A/cm^2)
    j_lim = (n * F * D_eff * C_bulk) / delta

    # check validity
    if j_lim <= j or j_lim <= 0 or j <= 0:
        return 1e6  # large penalty or infeasible

    # RT/(nF) ~ typical factor in log term
    factor = (R * T) / (n * F)

    return factor * np.log(1.0 - j / j_lim)


def eta_total(
    j,        # (A/cm^2) current density (geometric)
    j0,       # (A/cm^2_active) exchange current density (per active area)
    S_cat,    # (cm^2_active/cm^3 or cm^2_active/g) specific surface area
    epsilon,  # (dimensionless) porosity
    delta,    # (cm) layer thickness
    a, b,     # Tafel coefficients for activation: a (V), b (V)
    R, T,     # R (J/(mol*K)), T (K)
    n, F,     # n (dimensionless), F (C/mol)
    C_bulk,   # (mol/cm^3) bulk reactant concentration
    D,        # (cm^2/s) diffusivity
    tau       # (dimensionless) tortuosity
):
    """
    Total Overpotential
    -------------------
    η_total = η_act + η_conc

    Where:
    η_act = a + b * ln( j / [j0 * S_cat * (1-epsilon)*delta] )
    η_conc = (R*T/(n*F)) * ln(1 - j / j_lim),
       j_lim = n*F*(epsilon/tau*D)*C_bulk / delta

    Parameters:
    - j, j0, S_cat, epsilon, delta, a, b
    - R, T, n, F, C_bulk, D, tau
    (units described in the docstrings above)

    Returns:
    - η_total (V) : sum of activation and concentration overpotentials
    """
    act = eta_activation(j, j0, S_cat, epsilon, delta, a, b)
    conc = eta_concentration(j, R, T, n, F, C_bulk, D, epsilon, tau, delta)
    return act + conc
