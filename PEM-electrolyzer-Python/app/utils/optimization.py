# utils/optimization.py

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
# from pymoo.algorithms.moo.moga import MOGA
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from utils.models import cost_function, eta_total_layer

class PEMProblem(ElementwiseProblem):
    """
    Multiobjective problem with separate anode and cathode layers.
    Decision variables (n_var=6):
      1) L_a (g/cm^2)
      2) delta_a (cm)
      3) epsilon_a (dimensionless)
      4) L_c (g/cm^2)
      5) delta_c (cm)
      6) epsilon_c (dimensionless)
    """
    def __init__(self,
                 A_cell, R, T, alpha, n, F, C_bulk, D, tau,
                 eta_max, J_min,
                 # Anode properties
                 rho_cat_a, c_cat_a, j0_a, S_cat_a,
                 L_max_a, delta_max_a,
                 # Cathode properties
                 rho_cat_c, c_cat_c, j0_c, S_cat_c,
                 L_max_c, delta_max_c):
        
        # 6 decision variables, 2 objectives, 2 constraints
        super().__init__(
            n_var=6,
            n_obj=2,
            n_constr=2,
            xl=[1e-6, 1e-6, 0.01, 1e-6, 1e-6, 0.01],
            xu=[L_max_a, delta_max_a, 0.99, L_max_c, delta_max_c, 0.99]
        )
        
        # Store problem parameters
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
        
        # Anode
        self.rho_cat_a = rho_cat_a
        self.c_cat_a = c_cat_a
        self.j0_a = j0_a
        self.S_cat_a = S_cat_a
        
        # Cathode
        self.rho_cat_c = rho_cat_c
        self.c_cat_c = c_cat_c
        self.j0_c = j0_c
        self.S_cat_c = S_cat_c
        
        # We fix J = J_min for demonstration (feasibility constraint).
        self.J = J_min

    def _evaluate(self, x, out, *args, **kwargs):
        L_a, delta_a, eps_a, L_c, delta_c, eps_c = x

        # Cost of anode and cathode layers
        C_a = cost_function(L_a, delta_a, eps_a, self.A_cell, self.c_cat_a, self.rho_cat_a)
        C_c = cost_function(L_c, delta_c, eps_c, self.A_cell, self.c_cat_c, self.rho_cat_c)
        C_total = C_a + C_c

        # Overpotential of anode and cathode layers
        eta_a = eta_total_layer(L_a, delta_a, eps_a, self.J, self.j0_a, self.S_cat_a,
                                self.R, self.T, self.alpha, self.n, self.F, self.C_bulk, self.D, self.tau)
        eta_c = eta_total_layer(L_c, delta_c, eps_c, self.J, self.j0_c, self.S_cat_c,
                                self.R, self.T, self.alpha, self.n, self.F, self.C_bulk, self.D, self.tau)
        eta_total = eta_a + eta_c

        # Objectives:
        # f1 = cost, f2 = overpotential
        f1 = C_total
        f2 = eta_total

        # Constraints:
        # g1: eta_total <= eta_max -> (eta_total - eta_max <= 0)
        g1 = eta_total - self.eta_max
        # g2: J >= J_min -> (J_min - J <= 0), but J is fixed to J_min => g2 = 0
        g2 = self.J_min - self.J

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

def run_optimization(A_cell, R, T, alpha, n, F, C_bulk, D, tau,
                     eta_max, J_min,
                     rho_cat_a, c_cat_a, j0_a, S_cat_a,
                     L_max_a, delta_max_a,
                     rho_cat_c, c_cat_c, j0_c, S_cat_c,
                     L_max_c, delta_max_c,
                     method='NSGA2', pop_size=40, n_gen=50):
    
    ref_dirs = get_reference_directions("energy", n_obj=2, n_points=12)
    # Available multiobjective algorithms
    alg_dict = {
        "NSGA2": NSGA2(pop_size=pop_size),
        "NSGA3": NSGA3(pop_size=pop_size),
        "SPEA2": SPEA2(pop_size=pop_size),
        "MOEAD": MOEAD(pop_size=pop_size, n_neighbors=15, decomposition="pbi"),
        # "MOGA": MOGA(pop_size=pop_size)
    }

    # Select the chosen algorithm or default to NSGA2
    if method not in alg_dict:
        algorithm = NSGA2(pop_size=pop_size)
    else:
        algorithm = alg_dict[method]

    problem = PEMProblem(
        A_cell, R, T, alpha, n, F, C_bulk, D, tau, eta_max, J_min,
        rho_cat_a, c_cat_a, j0_a, S_cat_a, L_max_a, delta_max_a,
        rho_cat_c, c_cat_c, j0_c, S_cat_c, L_max_c, delta_max_c
    )

    termination = get_termination("n_gen", n_gen)
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
        pf=False,
        verbose=False
    )

    return res
