import numpy as np

def cost_function(L, delta, epsilon, A_cell, c_cat, rho_cat):
    """
    Cost function for a single catalyst layer:
    C = rho_cat * delta * (1 - epsilon) * A_cell * c_cat
    """
    return rho_cat * delta * (1 - epsilon) * A_cell * c_cat

def eta_activation(J, j0, L, S_cat, R, T, alpha, n, F):
    """
    eta_act = (R*T/(alpha*n*F)) * ln(J/(j0*L*S_cat))
    """
    val = J/(j0 * L * S_cat)
    if val <= 0:
        return 1e6  # large penalty
    return (R * T / (alpha * n * F)) * np.log(val)

def eta_concentration(J, delta, epsilon, tau, C_bulk, D, R, T, n, F):
    """
    eta_conc = (R*T/(n*F))*ln(C_bulk/(C_bulk - J*delta*n*F/D_eff))
    D_eff = D*(epsilon/tau)
    """
    D_eff = D * (epsilon / tau)
    denom = C_bulk - (J * delta * n * F / D_eff)
    if denom <= 0 or denom >= C_bulk:
        return 1e6
    return (R*T/(n*F))*np.log(C_bulk/denom)

def eta_total_layer(L, delta, epsilon, J, j0, S_cat, R, T, alpha, n, F, C_bulk, D, tau):
    """
    Total overpotential for a single layer = eta_act + eta_conc
    """
    eta_act = eta_activation(J, j0, L, S_cat, R, T, alpha, n, F)
    eta_conc = eta_concentration(J, delta, epsilon, tau, C_bulk, D, R, T, n, F)
    return eta_act + eta_conc
