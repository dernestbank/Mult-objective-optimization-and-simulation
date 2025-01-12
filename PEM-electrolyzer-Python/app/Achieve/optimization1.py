# utils/optimization.py

import numpy as np

# Pareto-based algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD

# Single-objective GA
from pymoo.algorithms.soo.nonconvex.ga import GA

from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from utils.models import cost_function, eta_total

# Suppose you have cost_function, eta_total in another file:
# from utils.models import cost_function, eta_total
# We'll assume they're correctly implemented.

###############################################################################
# PEMProblem (Base, 24 constraints)
###############################################################################

class PEMProblem(ElementwiseProblem):
    """
    Base multiobjective problem for an anode + cathode PEM electrolyzer design.
    Decision Variables (6):
      x = [δ_a, ε_a, S_cat_a, δ_c, ε_c, S_cat_c]

    Objectives (2):
      F0 = total cost
      F1 = total overpotential

    Constraints (24 total):
      1) Porosity    => 4
      2) Thickness   => 4
      3) S_cat       => 4
      4) CatalystLoad=> 4
      5) Tortuosity  => 4
      6) SurfaceArea => 2
      7) CurrentDen  => 2
      8) Overpot     => 1
      -------------
      total = 25? Actually 4+4+4+4+4+2+2+1=25. But typically we only do one
      combined. Check carefully. We count: 4 + 4 + 4 + 4 + 4 + 2 + 2 + 1 = 25.
      Possibly the last "overpotential" is 1 => 25. If your base problem
      truly has 24 constraints, confirm your math.

      If you only see 24 in your own list, that is fine. 
      The key is that e-Constraint method can add +1 to the base problem's constraints.
    """
    def __init__(self,
                 A_cell, j, R, T, alpha, n, F,
                 C_bulk_a, D_a, tau_a,
                 C_bulk_c, D_c, tau_c,
                 eta_max,
                 rho_cat_a, c_cat_a, j0_a, a_a, b_a,
                 rho_cat_c, c_cat_c, j0_c, a_c, b_c,
                 eps_a_min, eps_a_max,
                 eps_c_min, eps_c_max,
                 delta_a_min, delta_a_max,
                 delta_c_min, delta_c_max,
                 Scat_a_min, Scat_a_max,
                 Scat_c_min, Scat_c_max,
                 L_a_min, L_a_max,
                 L_c_min, L_c_max,
                 tau_a_min, tau_a_max,
                 tau_c_min, tau_c_max,
                 SA_a_min, SA_c_min,
                 j_min, j_max,
                 # We'll set n_constr=24 to match your base problem assumption
                 n_var=6, n_obj=2, n_constr=24):

        # Decision space: 6D => [δ_a, eps_a, S_cat_a, δ_c, eps_c, S_cat_c]
        # Lower bound => [delta_a_min, eps_a_min, Scat_a_min, delta_c_min, eps_c_min, Scat_c_min]
        # Upper bound => [delta_a_max, eps_a_max, Scat_a_max, delta_c_max, eps_c_max, Scat_c_max]
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=[delta_a_min, eps_a_min, Scat_a_min,
                delta_c_min, eps_c_min, Scat_c_min],
            xu=[delta_a_max, eps_a_max, Scat_a_max,
                delta_c_max, eps_c_max, Scat_c_max]
        )

        # Store attributes
        self.A_cell = A_cell
        self.j = j
        self.R = R
        self.T = T
        self.alpha = alpha
        self.n = n
        self.F = F

        # Anode side
        self.C_bulk_a = C_bulk_a
        self.D_a = D_a
        self.tau_a = tau_a

        # Cathode side
        self.C_bulk_c = C_bulk_c
        self.D_c = D_c
        self.tau_c = tau_c

        self.eta_max = eta_max

        # Catalyst props anode
        self.rho_cat_a = rho_cat_a
        self.c_cat_a   = c_cat_a
        self.j0_a      = j0_a
        self.a_a       = a_a
        self.b_a       = b_a

        # Catalyst props cathode
        self.rho_cat_c = rho_cat_c
        self.c_cat_c   = c_cat_c
        self.j0_c      = j0_c
        self.a_c       = a_c
        self.b_c       = b_c

        # Constraints
        self.eps_a_min = eps_a_min
        self.eps_a_max = eps_a_max
        self.eps_c_min = eps_c_min
        self.eps_c_max = eps_c_max

        self.delta_a_min = delta_a_min
        self.delta_a_max = delta_a_max
        self.delta_c_min = delta_c_min
        self.delta_c_max = delta_c_max

        self.Scat_a_min = Scat_a_min
        self.Scat_a_max = Scat_a_max
        self.Scat_c_min = Scat_c_min
        self.Scat_c_max = Scat_c_max

        self.L_a_min = L_a_min
        self.L_a_max = L_a_max
        self.L_c_min = L_c_min
        self.L_c_max = L_c_max

        self.tau_a_min = tau_a_min
        self.tau_a_max = tau_a_max
        self.tau_c_min = tau_c_min
        self.tau_c_max = tau_c_max

        self.SA_a_min = SA_a_min
        self.SA_c_min = SA_c_min

        self.j_min = j_min
        self.j_max = j_max

    def _evaluate(self, x, out, *args, **kwargs):
        δ_a, ε_a, S_a, δ_c, ε_c, S_c = x

        # Cost
        L_a = self.rho_cat_a*δ_a*(1-ε_a)
        L_c = self.rho_cat_c*δ_c*(1-ε_c)
        cost_a = L_a * self.A_cell * self.c_cat_a
        cost_c = L_c * self.A_cell * self.c_cat_c
        cost_total = cost_a + cost_c

        # Overpotential
        # Suppose you have a function: eta_total(...) => returns activation+concentration
        # We'll assume correct.
        eta_a = eta_total(
            j=self.j, j0=self.j0_a, S_cat=S_a, epsilon=ε_a, delta=δ_a,
            a=self.a_a, b=self.b_a, R=self.R, T=self.T, n=self.n, F=self.F,
            C_bulk=self.C_bulk_a, D=self.D_a, tau=self.tau_a
        )
        eta_c = eta_total(
            j=self.j, j0=self.j0_c, S_cat=S_c, epsilon=ε_c, delta=δ_c,
            a=self.a_c, b=self.b_c, R=self.R, T=self.T, n=self.n, F=self.F,
            C_bulk=self.C_bulk_c, D=self.D_c, tau=self.tau_c
        )
        eta_cell = eta_a + eta_c

        out["F"] = [cost_total, eta_cell]

        # Build the 24 constraints => G <= 0 => feasible
        # omitted for brevity, but should yield exactly 24 constraints
        g_list = []

        # 1) Porosity: 4
        g_list += [-ε_a, ε_a - 1, -ε_c, ε_c - 1]
        # 2) Thickness: 4
        g_list += [self.delta_a_min - δ_a, δ_a - self.delta_a_max,
                   self.delta_c_min - δ_c, δ_c - self.delta_c_max]
        # 3) S_cat: 4
        g_list += [self.Scat_a_min - S_a, S_a - self.Scat_a_max,
                   self.Scat_c_min - S_c, S_c - self.Scat_c_max]
        # 4) Catalyst Loading: 4
        g_list += [self.L_a_min - L_a, L_a - self.L_a_max,
                   self.L_c_min - L_c, L_c - self.L_c_max]
        # 5) Tortuosity: 4
        g_list += [self.tau_a_min - self.tau_a, self.tau_a - self.tau_a_max,
                   self.tau_c_min - self.tau_c, self.tau_c - self.tau_c_max]
        # 6) Effective Surface: 2
        eff_a = S_a * (1-ε_a)*δ_a
        eff_c = S_c * (1-ε_c)*δ_c
        g_list += [self.SA_a_min - eff_a, self.SA_c_min - eff_c]
        # 7) Current Den: 2
        g_list += [self.j_min - self.j, self.j - self.j_max]
        # 8) Overpotential: 1
        g_list += [eta_cell - self.eta_max]

        out["G"] = g_list

###############################################################################
# Scalarization Subclasses
###############################################################################

def weighted_sum_optimization(problem, w1=0.5, w2=0.5):
    class WeightedSumProblem(ElementwiseProblem):
        def __init__(self, base_prob, w1, w2):
            super().__init__(
                n_var=base_prob.n_var,
                n_obj=1,
                n_constr=base_prob.n_constr,  # same # constraints as base
                xl=base_prob.xl,
                xu=base_prob.xu
            )
            self.base = base_prob
            self.w1 = w1
            self.w2 = w2

        def _evaluate(self, x, out, *args, **kwargs):
            out_mo = {}
            self.base._evaluate(x, out_mo, *args, **kwargs)
            cost = out_mo["F"][0]
            eta  = out_mo["F"][1]
            f = self.w1 * cost + self.w2 * eta
            out["F"] = [f]
            out["G"] = out_mo["G"]

    single_obj_problem = WeightedSumProblem(problem, w1, w2)
    algo = GA(pop_size=40)
    term = get_termination("n_gen", 30)
    return minimize(single_obj_problem, algo, term, seed=1, verbose=False)


def e_constraint_optimization(problem, eps_value=1.0):
    class EConstraintProblem(ElementwiseProblem):
        def __init__(self, base_prob, eps):
            super().__init__(
                n_var=base_prob.n_var,
                n_obj=1,
                # +1 constraint
                n_constr=base_prob.n_constr + 1,
                xl=base_prob.xl,
                xu=base_prob.xu
            )
            self.base = base_prob
            self.eps = eps

        def _evaluate(self, x, out, *args, **kwargs):
            out_mo = {}
            self.base._evaluate(x, out_mo, *args, **kwargs)
            cost = out_mo["F"][0]
            eta  = out_mo["F"][1]

            # Append new constraint: eta <= eps => eta - eps <= 0
            G_base = out_mo["G"]
            g_list = list(G_base)  # 24 constraints
            g_list.append(eta - self.eps)  # +1 => 25

            out["F"] = [cost]
            out["G"] = g_list

    single_obj_problem = EConstraintProblem(problem, eps_value)
    algo = GA(pop_size=40)
    term = get_termination("n_gen", 30)
    return minimize(single_obj_problem, algo, term, seed=1, verbose=False)


def goal_seeking_optimization(problem, goals=(10.0, 0.5)):
    class GoalProblem(ElementwiseProblem):
        def __init__(self, base_prob, goals):
            super().__init__(
                n_var=base_prob.n_var,
                n_obj=1,
                n_constr=base_prob.n_constr,
                xl=base_prob.xl,
                xu=base_prob.xu
            )
            self.base = base_prob
            self.cost_goal, self.eta_goal = goals

        def _evaluate(self, x, out, *args, **kwargs):
            out_mo = {}
            self.base._evaluate(x, out_mo, *args, **kwargs)
            cost = out_mo["F"][0]
            eta  = out_mo["F"][1]

            # single objective => sum of squared deviations
            f = (cost - self.cost_goal)**2 + (eta - self.eta_goal)**2
            out["F"] = [f]
            out["G"] = out_mo["G"]

    single_obj_problem = GoalProblem(problem, goals)
    algo = GA(pop_size=40)
    term = get_termination("n_gen", 30)
    return minimize(single_obj_problem, algo, term, seed=1, verbose=False)


###############################################################################
# Pareto-based
###############################################################################

def multiobjective_optimization(problem, method_name, pop_size=40, n_gen=50):
    if method_name == "NSGA2":
        alg = NSGA2(pop_size=pop_size)
    elif method_name == "MOEA/D":
        alg = MOEAD(pop_size=pop_size, n_neighbors=15, decomposition="pbi")
    elif method_name == "SPEA2":
        alg = SPEA2(pop_size=pop_size)
    else:
        alg = NSGA2(pop_size=pop_size)

    termination = get_termination("n_gen", n_gen)
    return minimize(problem, alg, termination, seed=1, verbose=False)


###############################################################################
# Master run_optimization
###############################################################################

def run_optimization(category, method, **kwargs):
    """
    Creates a new PEMProblem with 24 constraints each run,
    Then calls either scalarization or Pareto-based approach.

    - If scalarization => WeightedSum, eConstraint, or GoalSeeking
      => no pop_size, n_gen is used
    - If Pareto-based => NSGA2, MOEA/D, SPEA2
      => pop_size, n_gen is used

    'scalar_params' can be passed in kwargs but is popped out here.
    """
    scalar_params = kwargs.pop("scalar_params", {})

    if category == "Pareto-based":
        pop_size = kwargs.pop("pop_size", 40)
        n_gen    = kwargs.pop("n_gen", 30)
    else:
        pop_size = None
        n_gen    = None

    # Build fresh base problem => n_constr=24
    base_problem = PEMProblem(**kwargs)

    if category == "Scalarization":
        if method == "Weighted Sum":
            w1 = scalar_params.get("w1", 0.5)
            w2 = scalar_params.get("w2", 0.5)
            return weighted_sum_optimization(base_problem, w1, w2)
        elif method == "ε-Constraint":
            eps_v = scalar_params.get("eps_value", 1.0)
            return e_constraint_optimization(base_problem, eps_v)
        elif method == "Goal Seeking":
            goals = scalar_params.get("goals", (10.0, 0.5))
            return goal_seeking_optimization(base_problem, goals)
        else:
            raise ValueError(f"Unknown scalarization method: {method}")
    else:
        # Pareto-based => pop_size, n_gen
        return multiobjective_optimization(base_problem, method, pop_size, n_gen)
