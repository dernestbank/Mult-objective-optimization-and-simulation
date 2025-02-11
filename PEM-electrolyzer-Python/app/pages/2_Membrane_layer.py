# streamlit_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.membrane_optimization import run_optimization

st.title("PEM Electrolyzer Membrane Design Optimization")

st.sidebar.header("User Input Parameters")

# Select optimization algorithm (for now, only NSGA2 is implemented)
algorithm_option = st.sidebar.selectbox("Choose Optimization Algorithm", options=["NSGA2"])

pop_size = st.sidebar.slider("Population Size", min_value=50, max_value=300, value=100, step=10)
n_gen = st.sidebar.slider("Number of Generations", min_value=10, max_value=200, value=100, step=10)
seed = st.sidebar.number_input("Random Seed", value=1, step=1)

st.sidebar.header("Decision Variable Bounds")
t_lb = st.sidebar.number_input("Lower bound for membrane thickness (m)", value=50e-6, format="%.6e")
t_ub = st.sidebar.number_input("Upper bound for membrane thickness (m)", value=300e-6, format="%.6e")
j_lb = st.sidebar.number_input("Lower bound for current density (A/cm²)", value=0.5)
j_ub = st.sidebar.number_input("Upper bound for current density (A/cm²)", value=3.0)

st.sidebar.header("Constraint Adjustments")
# For demonstration, display the mechanical minimum thickness (this is computed in model.py)
# In a more advanced version, you might let the user adjust ΔP, r, SF, etc.
st.write("Mechanical durability constraint (membrane thickness must be ≥ 158 µm).")

if st.button("Run Optimization"):
    st.write("Running optimization...")
    # Note: Our current optimization problem (in optimization.py) uses fixed bounds.
    # In a full implementation, these bounds would be passed dynamically.
    res = run_optimization(algorithm_name=algorithm_option, seed=seed, pop_size=pop_size, n_gen=n_gen)
    st.write("Optimization Completed!")
    
    # Retrieve objective function values
    F = res.F
    # Since f1 and f2 were defined as negatives to enable maximization, we plot:
    efficiency = -F[:, 0]
    lifetime = -F[:, 1]
    cost = F[:, 2]
    env_impact = F[:, 3]
    
    st.subheader("Pareto Front: Efficiency vs. Cost")
    fig, ax = plt.subplots()
    sc = ax.scatter(cost, efficiency, c=lifetime, cmap="viridis")
    ax.set_xlabel("Capital Cost ($/m²)")
    ax.set_ylabel("Energy Efficiency")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Lifetime (hours)")
    st.pyplot(fig)
    
    st.subheader("Optimization Result (Decision Variables)")
    st.write("Membrane thickness (m) and current density (A/cm²):")
    st.write(res.X)
    
    st.subheader("Objective Values")
    st.write("Columns: [-Efficiency, -Lifetime, Capital Cost, Environmental Impact]")
    st.write(F)
    
    st.subheader("Pareto Front: Lifetime vs. Environmental Impact")
    fig2, ax2 = plt.subplots()
    ax2.scatter(lifetime, env_impact, c=cost, cmap="plasma")
    ax2.set_xlabel("Lifetime (hours)")
    ax2.set_ylabel("Environmental Impact (kg CO₂-eq/m²)")
    st.pyplot(fig2)
