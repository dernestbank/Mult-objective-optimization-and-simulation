import streamlit as st
import numpy as np
from utils.optimization import run_optimization

st.set_page_config(page_title="Sensitivity Analysis", layout="wide")

st.title("Sensitivity Analysis")

st.write("This page allows you to perform sensitivity analysis by varying one parameter and observing the impact on the optimization results.")

st.sidebar.header("Sensitivity Analysis Settings")
parameter_to_vary = st.sidebar.selectbox("Select parameter to vary", ["C_bulk", "D", "tau", "eta_max", "J_min"])

param_min = st.sidebar.number_input(f"Minimum {parameter_to_vary}", value=0.05)
param_max = st.sidebar.number_input(f"Maximum {parameter_to_vary}", value=0.15)
steps = st.sidebar.number_input("Number of Steps", value=5, min_value=2)
run_sens = st.sidebar.button("Run Sensitivity Analysis")

# Fixed parameters for demonstration:
A_cell=50.0
R=8.314
T=353.0
alpha=0.5
n=2
F=96485.0
C_bulk=0.1
D=1e-2
tau=1.2
eta_max=2.0
J_min=0.0001
rho_cat_a=11.66
c_cat_a=100.0
j0_a=1e-2
S_cat_a=100000.0
L_max_a=0.01
delta_max_a=0.001
rho_cat_c=21.45
c_cat_c=60.0
j0_c=1e-2
S_cat_c=100000.0
L_max_c=0.01
delta_max_c=0.001
method="NSGA2"
pop_size=20
n_gen=20

if run_sens:
    st.write(f"Varying {parameter_to_vary} from {param_min} to {param_max} in {steps} steps.")
    param_values = np.linspace(param_min, param_max, steps)
    results = []

    for val in param_values:
        # Adjust parameter
        if parameter_to_vary == "C_bulk":
            C_bulk = val
        elif parameter_to_vary == "D":
            D = val
        elif parameter_to_vary == "tau":
            tau = val
        elif parameter_to_vary == "eta_max":
            eta_max = val
        elif parameter_to_vary == "J_min":
            J_min = val

        res = run_optimization(A_cell, R, T, alpha, n, F, C_bulk, D, tau,
                               eta_max, J_min,
                               rho_cat_a, c_cat_a, j0_a, S_cat_a,
                               L_max_a, delta_max_a,
                               rho_cat_c, c_cat_c, j0_c, S_cat_c,
                               L_max_c, delta_max_c,
                               method=method, pop_size=pop_size, n_gen=n_gen)

        if res.F is not None and len(res.F) > 0:
            # For simplicity, let's record the minimal cost found and minimal overpotential
            min_cost = np.min(res.F[:,0])
            min_eta = np.min(res.F[:,1])
            results.append((val, min_cost, min_eta))
        else:
            # If no solutions found, record None
            results.append((val, None, None))

    # Display results
    st.write("Sensitivity Analysis Results:")
    st.write(f"Parameter: {parameter_to_vary}")
    st.write("Value, Min Cost, Min Overpotential")
    for r in results:
        st.write(r)

    # Plot results if feasible
    valid_results = [r for r in results if r[1] is not None]
    if len(valid_results) > 0:
        vals = [r[0] for r in valid_results]
        costs = [r[1] for r in valid_results]
        etas = [r[2] for r in valid_results]

        st.subheader("Sensitivity Plots")
        st.line_chart({"Cost": costs}, x=vals)
        st.line_chart({"Overpotential": etas}, x=vals)
    else:
        st.warning("No feasible solutions found in the given parameter range.")
