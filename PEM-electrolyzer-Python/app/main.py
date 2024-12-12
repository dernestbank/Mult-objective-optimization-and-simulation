import streamlit as st
import numpy as np
import plotly.express as px
from utils.optimization import run_optimization

st.set_page_config(page_title="PEM Electrolyser Multi-Catalyst Optimization", layout="wide")
st.title("Multiobjective Optimization with Separate Anode and Cathode Catalysts")

st.sidebar.header("Operating Parameters")
A_cell = st.sidebar.number_input("Cell Active Area (cm²)", value=50.0)
R = 8.314  # J/(mol*K)
T = st.sidebar.number_input("Temperature (K)", value=353.0)
alpha = st.sidebar.number_input("Charge Transfer Coefficient (alpha)", value=0.5)
n = st.sidebar.number_input("Number of Electrons (n)", value=1)
F = 96485.0  # C/mol
C_bulk = st.sidebar.number_input("Bulk Concentration (mol/cm³)", value=10.0)
D = st.sidebar.number_input("Diffusivity (cm²/s)", value=1e-3)
tau = st.sidebar.number_input("Tortuosity (tau)", value=1.5)
eta_max = st.sidebar.number_input("Max Overpotential (V)", value=1.0)
J_min = st.sidebar.number_input("Min Current Density (A/cm²)", value=0.001)

st.sidebar.header("Anode Catalyst Selection")
anode_catalyst = st.sidebar.selectbox("Anode Catalyst", 
                                      ["IrO2", "RuO2", "Custom"])
if anode_catalyst == "IrO2":
    rho_cat_a = 11.66  # g/cm³ approx IrO2
    c_cat_a = 100.0    # $/g (example)
    j0_a = 1e-4        # A/cm²_active (example)
    S_cat_a = 50000.0   # cm²_active/g (example)
elif anode_catalyst == "RuO2":
    rho_cat_a = 6.97
    c_cat_a = 80.0
    j0_a = 2e-4
    S_cat_a = 40000.0
else:
    rho_cat_a = st.sidebar.number_input("Anode Catalyst Density (g/cm³)", value=10.0)
    c_cat_a = st.sidebar.number_input("Anode Catalyst Cost ($/g)", value=50.0)
    j0_a = st.sidebar.number_input("Anode Exchange Current Density (A/cm²_active)", value=1e-4)
    S_cat_a = st.sidebar.number_input("Anode Specific Surface Area (cm²_active/g)", value=50000.0)

L_max_a = st.sidebar.number_input("Max Anode Loading (g/cm²)", value=0.01)
delta_max_a = st.sidebar.number_input("Max Anode Thickness (cm)", value=0.001)

st.sidebar.header("Cathode Catalyst Selection")
cathode_catalyst = st.sidebar.selectbox("Cathode Catalyst", 
                                        ["Pt", "Pt-Ru", "Custom"])
if cathode_catalyst == "Pt":
    rho_cat_c = 21.45
    c_cat_c = 60.0
    j0_c = 1e-5
    S_cat_c = 50000.0
elif cathode_catalyst == "Pt-Ru":
    rho_cat_c = 16.0
    c_cat_c = 55.0
    j0_c = 1.5e-5
    S_cat_c = 45000.0
else:
    rho_cat_c = st.sidebar.number_input("Cathode Catalyst Density (g/cm³)", value=21.45)
    c_cat_c = st.sidebar.number_input("Cathode Catalyst Cost ($/g)", value=60.0)
    j0_c = st.sidebar.number_input("Cathode Exchange Current Density (A/cm²_active)", value=1e-5)
    S_cat_c = st.sidebar.number_input("Cathode Specific Surface Area (cm²_active/g)", value=50000.0)

L_max_c = st.sidebar.number_input("Max Cathode Loading (g/cm²)", value=0.01)
delta_max_c = st.sidebar.number_input("Max Cathode Thickness (cm)", value=0.001)

st.sidebar.header("Optimization Settings")
method = st.sidebar.selectbox("Optimization Method", ["NSGA2", "NSGA3"])
pop_size = st.sidebar.number_input("Population Size", value=40, min_value=10)
n_gen = st.sidebar.number_input("Number of Generations", value=50, min_value=10)

if st.button("Run Optimization"):
    with st.spinner("Optimizing... Please wait..."):
        res = run_optimization(
            A_cell, R, T, alpha, n, F, C_bulk, D, tau,
            eta_max, J_min,
            rho_cat_a, c_cat_a, j0_a, S_cat_a,
            L_max_a, delta_max_a,
            rho_cat_c, c_cat_c, j0_c, S_cat_c,
            L_max_c, delta_max_c,
            method=method, pop_size=pop_size, n_gen=n_gen
        )
    if res.F is None:
        st.error("No feasible solutions found. Try adjusting parameters or constraints.")
    else:
        st.success("Optimization Complete!")
        F = res.F
        X = res.X

        fig = px.scatter(x=F[:,0], y=F[:,1], 
                         title="Pareto Front: Cost vs Overpotential",
                         labels={"x":"Cost (C)", "y":"Overpotential (eta_total)"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sample Pareto-Optimal Solutions:")
        for i in range(min(len(X), 5)):
            L_a, delta_a, eps_a, L_c, delta_c, eps_c = X[i]
            st.write(f"Solution {i+1}:")
            st.write(f"  Anode: L={L_a:.6f} g/cm², δ={delta_a:.6f} cm, ε={eps_a:.4f}")
            st.write(f"  Cathode: L={L_c:.6f} g/cm², δ={delta_c:.6f} cm, ε={eps_c:.4f}")
            st.write(f"  Cost={F[i,0]:.4f}, η_total={F[i,1]:.4f}")
