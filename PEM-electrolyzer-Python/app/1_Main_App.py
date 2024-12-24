# 1_Main_App.py

import streamlit as st
import numpy as np
import plotly.express as px

from utils.optimization import run_optimization
from utils.visualization import (
    create_dataframe,
    scatter_matrix_plot,
    parallel_coordinates_plot,
    heatmap_plot,
    glyph_plot
)

st.set_page_config(page_title="PEM Electrolyser Optimization", layout="wide")

st.title("PEM Electrolyser Multiobjective Optimization")

st.write("Welcome to the main page of the multi-page application. "
         "Use the sidebar to set parameters and run the optimization. "
         "Navigate to the 'Sensitivity Analysis' page for parameter sensitivity studies.")

# ------------- SIDEBAR INPUTS -------------
st.sidebar.header("Operating Parameters")
A_cell = st.sidebar.number_input("Cell Active Area (cm²)", value=50.0)
R = 8.314  # J/(mol*K)
T = st.sidebar.number_input("Temperature (K)", value=353.0)
alpha = st.sidebar.number_input("Charge Transfer Coefficient (alpha)", value=0.5)
n = st.sidebar.number_input("Number of Electrons (n)", value=2)
F = 96485.0  # C/mol

C_bulk = st.sidebar.number_input("Bulk Concentration (mol/cm³)", value=0.1)
D = st.sidebar.number_input("Diffusivity (cm²/s)", value=1e-2)
tau = st.sidebar.number_input("Tortuosity (tau)", value=1.2)
eta_max = st.sidebar.number_input("Max Overpotential (V)", value=2.0)
J_min = st.sidebar.number_input("Min Current Density (A/cm²)", value=0.0001)

st.sidebar.header("Anode Catalyst Selection")
anode_catalyst = st.sidebar.selectbox("Anode Catalyst", ["IrO2", "RuO2", "Custom"])
if anode_catalyst == "IrO2":
    rho_cat_a = 11.66
    c_cat_a = 100.0
    j0_a = 1e-2
    S_cat_a = 100000.0
elif anode_catalyst == "RuO2":
    rho_cat_a = 6.97
    c_cat_a = 80.0
    j0_a = 1e-2
    S_cat_a = 100000.0
else:
    rho_cat_a = st.sidebar.number_input("Anode Catalyst Density (g/cm³)", value=10.0)
    c_cat_a = st.sidebar.number_input("Anode Catalyst Cost ($/g)", value=50.0)
    j0_a = st.sidebar.number_input("Anode Exchange Current Density (A/cm²_active)", value=1e-2)
    S_cat_a = st.sidebar.number_input("Anode Specific Surface Area (cm²_active/g)", value=100000.0)

st.sidebar.header("Anode Layer Constraints")
L_max_a = st.sidebar.slider("Max Anode Loading (g/cm²)", min_value=0.0001, max_value=0.02, value=0.01, step=0.0001)
delta_max_a = st.sidebar.slider("Max Anode Thickness (cm)", min_value=0.0001, max_value=0.005, value=0.001, step=0.0001)

st.sidebar.header("Cathode Catalyst Selection")
cathode_catalyst = st.sidebar.selectbox("Cathode Catalyst", ["Pt", "Pt-Ru", "Custom"])
if cathode_catalyst == "Pt":
    rho_cat_c = 21.45
    c_cat_c = 60.0
    j0_c = 1e-2
    S_cat_c = 100000.0
elif cathode_catalyst == "Pt-Ru":
    rho_cat_c = 16.0
    c_cat_c = 55.0
    j0_c = 1e-2
    S_cat_c = 100000.0
else:
    rho_cat_c = st.sidebar.number_input("Cathode Catalyst Density (g/cm³)", value=21.45)
    c_cat_c = st.sidebar.number_input("Cathode Catalyst Cost ($/g)", value=60.0)
    j0_c = st.sidebar.number_input("Cathode Exchange Current Density (A/cm²_active)", value=1e-2)
    S_cat_c = st.sidebar.number_input("Cathode Specific Surface Area (cm²active/g)", value=100000.0)

st.sidebar.header("Cathode Layer Constraints")
L_max_c = st.sidebar.slider("Max Cathode Loading (g/cm²)", min_value=0.0001, max_value=0.02, value=0.01, step=0.0001)
delta_max_c = st.sidebar.slider("Max Cathode Thickness (cm)", min_value=0.0001, max_value=0.005, value=0.001, step=0.0001)

st.sidebar.header("Methods I: Multiobjective Optimization")
method_options = ["NSGA2", "NSGA3", "SPEA2", "MOEAD", "MOGA"]
method = st.sidebar.selectbox("Multiobjective Optimization Method", method_options)
pop_size = st.sidebar.number_input("Population Size", value=40, min_value=10)
n_gen = st.sidebar.number_input("Number of Generations", value=50, min_value=10)

if st.sidebar.button("Run Optimization"):
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
    if res.F is None or len(res.F) == 0:
        st.error("No feasible solutions found. Try adjusting parameters or constraints.")
    else:
        st.success("Optimization Complete!")
        F = res.F
        X = res.X

        # Create a combined DataFrame for advanced visualization
        var_names = ["L_a", "δ_a", "ε_a", "L_c", "δ_c", "ε_c"]
        obj_names = ["Cost", "Overpotential"]
        df = create_dataframe(X, F, var_names=var_names, obj_names=obj_names)

        # Display the Pareto front in a scatter plot
        fig = px.scatter(
            x=F[:, 0], y=F[:, 1],
            title="Pareto Front: Cost vs Overpotential",
            labels={"x": "Cost (C)", "y": "Overpotential (η_total)"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Visualization Options
        st.subheader("Methods II: Visualization")
        viz_method = st.selectbox(
            "Select Visualization Method",
            ["None", "Scatter Matrix", "Parallel Coordinates", "Heatmap", "Glyph Plot"]
        )

        if viz_method == "Scatter Matrix":
            dimensions = st.multiselect("Select dimensions", df.columns, default=df.columns.tolist())
            color = st.selectbox("Select a column for color", [None] + df.columns.tolist())
            fig = scatter_matrix_plot(df, dimensions=dimensions, color=color)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_method == "Parallel Coordinates":
            dimensions = st.multiselect("Select dimensions", df.columns, default=df.columns.tolist())
            color = st.selectbox("Select a column for color", [None] + df.columns.tolist())
            fig = parallel_coordinates_plot(df, dimensions=dimensions, color=color)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_method == "Heatmap":
            x_var = st.selectbox("X variable", df.columns)
            y_var = st.selectbox("Y variable", df.columns)
            z_var = st.selectbox("Z variable (values)", df.columns)
            fig = heatmap_plot(df, x_var, y_var, z_var)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_method == "Glyph Plot":
            x = st.selectbox("X variable", df.columns)
            y = st.selectbox("Y variable", df.columns)
            size_col = st.selectbox("Size column", [None] + df.columns.tolist())
            color_col = st.selectbox("Color column", [None] + df.columns.tolist())
            fig = glyph_plot(df, x, y, size_col=size_col, color_col=color_col)
            st.plotly_chart(fig, use_container_width=True)
