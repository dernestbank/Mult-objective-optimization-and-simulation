# optimization.py
import numpy as np
from pymoo.core.problem import Problem
from membrane_model import PEMModel

class PEMOptimizationProblem(Problem):
    def __init__(self, model: PEMModel):
        # Define decision variable bounds.
        # x[0] = t (thickness) in [50e-6, 300e-6] m,
        # x[1] = j (current density) in [0.5, 3.0] A/cm².
        # Note: Our mechanical constraint (t >= 158e-6) should render the lower bound effectively 158e-6.
        xl = np.array([50e-6, 0.5])
        xu = np.array([300e-6, 3.0])
        super().__init__(n_var=2,
                         n_obj=4,
                         n_constr=1,
                         xl=xl,
                         xu=xu)
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate the objectives and constraints for a set of solutions X.
        X is a 2D array where each row is a candidate solution [t, j].
        """
        F = np.zeros((X.shape[0], 4))
        G = np.zeros((X.shape[0], 1))
        
        for i in range(X.shape[0]):
            x = X[i, :]
            F[i, :] = self.model.evaluate_objectives(x)
            # Our constraint: t - t_mech_min >= 0. pymoo expects constraint violation values <= 0.
            # Thus, we return g(x) = t_mech_min - t, so feasibility means g(x) <= 0.
            G[i, 0] = self.model.t_mech_min - x[0]
        
        out["F"] = F
        out["G"] = G

def run_optimization(algorithm_name="NSGA2", seed=1, pop_size=100, n_gen=100):
    """
    Run the multiobjective optimization using pymoo.
    
    Parameters:
      algorithm_name: string indicating the algorithm (currently supports "NSGA2")
      seed: random seed
      pop_size: population size
      n_gen: number of generations
      
    Returns:
      res: optimization result object from pymoo.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination
    
    # Instantiate the PEM model and problem.
    model = PEMModel()
    problem = PEMOptimizationProblem(model=model)
    
    # Select algorithm. (For now, we only implement NSGA2; others can be added similarly.)
    if algorithm_name.upper() == "NSGA2":
        algorithm = NSGA2(pop_size=pop_size, random_state=seed)
    else:
        # Default to NSGA2 if the algorithm is not recognized.
        algorithm = NSGA2(pop_size=pop_size, random_state=seed)
    
    termination = get_termination("n_gen", n_gen)
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True,
                   verbose=True)
    
    return res
