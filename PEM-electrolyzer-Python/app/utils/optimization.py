import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_termination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from utils.models import cost_function, eta_total_layer

class PEMProblem(ElementwiseProblem):
    def __init__(self, 
                 A_cell, R, T, alpha, n, F, C_bulk, D, tau, 
                 eta_max, J_min,
                 # Anode properties
                 rho_cat_a, c_cat_a, j0_a, S_cat_a,
                 L_max_a, delta_max_a,
                 # Cathode properties
                 rho_cat_c, c_cat_c, j0_c, S_cat_c,
                 L_max_c, delta_max_c
                 ):
        # 6 decision variables: L_anode, delta_anode, epsilon_anode, L_cathode, delta_cathode, epsilon_cathode
        super().__init__(
            n_var=6, 
            n_obj=2, 
            n_constr=2, 
            xl=[1e-6, 1e-6, 0.01, 1e-6, 1e-6, 0.01], 
            xu=[L_max_a, delta_max_a, 0.99, L_max_c, delta_max_c, 0.99]
        )

        self.A_cell = A_cell
        self.R = R
        self.T = T
        self.alpha = alpha
        self.n = n
        self.F = F
        self.C_bulk = C_bulk
        self.D = D
        self.tau = tau
        self.eta_max = eta_max
        self.J_min = J_min

        # Catalyst properties for anode
        self.rho_cat_a = rho_cat_a
        self.c_cat_a = c_cat_a
        self.j0_a = j0_a
        self.S_cat_a = S_cat_a

        # Catalyst properties for cathode
        self.rho_cat_c = rho_cat_c
        self.c_cat_c = c_cat_c
        self.j0_c = j0_c
        self.S_cat_c = S_cat_c

        # Assume J = J_min for demonstration
        self.J = J_min

    def _evaluate(self, x, out, *args, **kwargs):
        L_a, delta_a, epsilon_a, L_c, delta_c, epsilon_c = x

        # Compute cost for each layer
        C_a = cost_function(L_a, delta_a, epsilon_a, self.A_cell, self.c_cat_a, self.rho_cat_a)
        C_c = cost_function(L_c, delta_c, epsilon_c, self.A_cell, self.c_cat_c, self.rho_cat_c)
        C_total = C_a + C_c

        # Compute overpotential for each layer
        eta_a = eta_total_layer(L_a, delta_a, epsilon_a, self.J, self.j0_a, self.S_cat_a, 
                                self.R, self.T, self.alpha, self.n, self.F, self.C_bulk, self.D, self.tau)
        eta_c = eta_total_layer(L_c, delta_c, epsilon_c, self.J, self.j0_c, self.S_cat_c, 
                                self.R, self.T, self.alpha, self.n, self.F, self.C_bulk, self.D, self.tau)
        
        eta_total = eta_a + eta_c

        # Objectives: Minimize cost (C_total) and minimize overpotential (eta_total)
        f1 = C_total
        f2 = eta_total

        # Constraints:
        # eta_total <= eta_max  -> eta_total - eta_max <= 0
        g1 = eta_total - self.eta_max

        # J >= J_min -> J_min - J <= 0
        # J is fixed to J_min, so this is always 0
        g2 = self.J_min - self.J

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

def run_optimization(
    # Operating and structural parameters
    A_cell, R, T, alpha, n, F, C_bulk, D, tau, 
    eta_max, J_min,
    # Anode catalyst properties & bounds
    rho_cat_a, c_cat_a, j0_a, S_cat_a,
    L_max_a, delta_max_a,
    # Cathode catalyst properties & bounds
    rho_cat_c, c_cat_c, j0_c, S_cat_c,
    L_max_c, delta_max_c,
    method='NSGA2', pop_size=40, n_gen=50
):
    if method == 'NSGA2':
        algorithm = NSGA2(pop_size=pop_size)
    elif method == 'NSGA3':
        algorithm = NSGA3(pop_size=pop_size)
    else:
        algorithm = NSGA2(pop_size=pop_size)

    problem = PEMProblem(A_cell, R, T, alpha, n, F, C_bulk, D, tau, eta_max, J_min,
                         rho_cat_a, c_cat_a, j0_a, S_cat_a,
                         L_max_a, delta_max_a,
                         rho_cat_c, c_cat_c, j0_c, S_cat_c,
                         L_max_c, delta_max_c)

    termination = get_termination("n_gen", n_gen)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   pf=False,
                   verbose=False)

    return res
