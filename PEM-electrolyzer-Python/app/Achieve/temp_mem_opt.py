# utils/membrane_optimization.py
import numpy as np
from pymoo.core.problem import Problem
from temp_2mem import MembraneModel

class MembraneOptimizationProblem(Problem):
    def __init__(self, model: MembraneModel, xl=None, xu=None):
        # Decision variables: x[0]=t (m), x[1]=i (A/cm²)
        # Default bounds (can be overridden via the streamlit interface):
        if xl is None:
            xl = np.array([50e-6, 0.5])
        if xu is None:
            xu = np.array([300e-6, 6.0])
        super().__init__(n_var=2, n_obj=2, n_constr=1, xl=xl, xu=xu)
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 2))
        G = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            xi = X[i, :]
            F[i, :] = self.model.evaluate_objectives(xi)
            G[i, 0] = self.model.evaluate_constraints(xi)[0]  # lifetime constraint
        out["F"] = F
        out["G"] = G

# (Optional) We could add scalarization transformations for Weighted Sum or Goal Seeking
# For simplicity, here we stick with the Pareto-based problem.
