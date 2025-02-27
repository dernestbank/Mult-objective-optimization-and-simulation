# pages/2_mem_layer.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.membrane_optimization import MembraneOptimizationProblem
from utils.temp_2mem import MembraneModel

# Import pymoo methods
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2

st.title("PEM Membrane Design Optimization")

st.sidebar.header("Select Optimization Method")
method_category = st.sidebar.selectbox("Optimization Category", ["Scalarization", "Pareto-based"])

if method_category == "Scalarization":
    method_name = st.sidebar.selectbox("Scalarization Method", ["Weighted Sum", "Goal Seeking"])
else:
    method_name = st.sidebar.selectbox("Pareto-based Method", ["NSGA2", "MOEA/D", "SPEA2"])

st.sidebar.header("Decision Variable Boundaries")
t_lb = st.sidebar.number_input("Lower bound for membrane thickness (m)", value=50e-6, format="%.6e")
t_ub = st.sidebar.number_input("Upper bound for membrane thickness (m)", value=300e-6, format="%.6e")
i_lb = st.sidebar.number_input("Lower bound for current density (A/cm²)", value=0.5)
i_ub = st.sidebar.number_input("Upper bound for current density (A/cm²)", value=6.0)

bounds = {"t_lb": t_lb, "t_ub": t_ub, "i_lb": i_lb, "i_ub": i_ub}

st.sidebar.header("Lifetime Constraint")
L_min = st.sidebar.number_input("Minimum Lifetime (hours)", value=12000.0)

st.sidebar.header("Membrane Model Parameters")
c_ionomer = st.sidebar.number_input("Cost of ionomer ($/kg)", value=300.0)
rho = st.sidebar.number_input("Membrane density (kg/m³)", value=2000.0)
c_manuf = st.sidebar.number_input("Manufacturing cost ($/m²)", value=20.0)
# Lifetime parameters
L_base = st.sidebar.number_input("Baseline lifetime (hours)", value=10000.0)
alpha_L = st.sidebar.number_input("Lifetime gain per m (h/m)", value=1e7)
beta_L = st.sidebar.number_input("Lifetime loss per (A/cm²) (h per A/cm²)", value=5000.0)

# Efficiency parameters
HHV = st.sidebar.number_input("Higher Heating Value (J/kg)", value=142e6)
MW_H2 = st.sidebar.number_input("Molecular Weight of H2 (kg/mol)", value=0.002016)
eta_F = st.sidebar.number_input("Faradaic efficiency", value=0.95)
V_oc = st.sidebar.number_input("Open Circuit Voltage (V)", value=1.23)
k1 = st.sidebar.number_input("Activation parameter k1 (V/(A/cm²))", value=0.005)
k2 = st.sidebar.number_input("Ohmic parameter k2 (V·m)", value=0.1)

# Create the membrane model instance using provided parameters.
model_params = {
    "HHV": HHV,
    "MW_H2": MW_H2,
    "eta_F": eta_F,
    "V_oc": V_oc,
    "k1": k1,
    "k2": k2,
    "c_ionomer": c_ionomer,
    "rho": rho,
    "c_manuf": c_manuf,
    "L_base": L_base,
    "alpha_L": alpha_L,
    "beta_L": beta_L,
    "L_min": L_min
}
model = MembraneModel(**model_params)

# Define decision variable bounds for the optimization problem.
xl = np.array([bounds["t_lb"], bounds["i_lb"]])
xu = np.array([bounds["t_ub"], bounds["i_ub"]])

# Set up the multiobjective problem.
problem = MembraneOptimizationProblem(model=model, xl=xl, xu=xu)

# Choose algorithm based on selection.
if method_category == "Pareto-based":
    if method_name == "NSGA2":
        algorithm = NSGA2(pop_size=100, seed=1)
    elif method_name == "MOEA/D":
        algorithm = MOEAD(pop_size=100, seed=1)
    elif method_name == "SPEA2":
        algorithm = SPEA2(pop_size=100, seed=1)
    else:
        algorithm = NSGA2(pop_size=100, seed=1)
else:
    # For scalarization methods, one could set up a weighted sum or goal-seeking version.
    # For now, we default to NSGA2 on the scalarized objective.
    # (Scalarization code can be added as needed.)
    algorithm = NSGA2(pop_size=100, seed=1)

termination = get_termination("n_gen", 100)

if st.button("Run Optimization"):
    st.write("Running optimization... please wait.")
    from pymoo.optimize import minimize
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   verbose=True)
    
    st.write("Optimization Completed!")
    X = res.X
    F = res.F

    st.subheader("Optimal Decision Variables")
    st.write("Each solution: [membrane thickness (m), current density (A/cm²)]")
    st.write(X)

    st.subheader("Objective Values")
    st.write("Columns: [ -Energy Efficiency, Cost ($/m²) ]")
    st.write(F)

    # For Pareto-based methods, plot the Pareto front.
    if method_category == "Pareto-based":
        f_eff = -F[:, 0]  # Efficiency is maximized (so negative was minimized)
        f_cost = F[:, 1]
        fig, ax = plt.subplots()
        sc = ax.scatter(f_cost, f_eff, c=-model.evaluate_objectives(X[0])[0], cmap="viridis")
        ax.set_xlabel("Cost ($/m²)")
        ax.set_ylabel("Energy Efficiency")
        st.pyplot(fig)
