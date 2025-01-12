# 1_Main_App.py

import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime

from utils.optimization import run_optimization

st.set_page_config(page_title="PEM Electrolyser Optimization", layout="wide")
st.title("PEM Electrolyser: Scalarization vs. Pareto-based Optimization")

RESULTS_DIR = "results_logs"
os.makedirs(RESULTS_DIR, exist_ok=True)

###############################################################################
# HELPER VISUAL & METRIC FUNCTIONS
###############################################################################

def create_dataframe(X, F, var_names, obj_names):
    df_vars = pd.DataFrame(X, columns=var_names)
    df_objs = pd.DataFrame(F, columns=obj_names)
    return pd.concat([df_vars, df_objs], axis=1)

def compute_hypervolume(F, ref_point=(1e5,1e5)):
    from pymoo.indicators import Hypervolume
    if F is None or len(F)==0:
        return None
    hv = Hypervolume(ref_point=ref_point)
    return hv.do(F)

def compute_c_metric(F, F2=None):
    from pymoo.indicators import CMetric
    if F is None or len(F)==0:
        return None
    if F2 is None:
        F2 = F
    cm = CMetric().do(F, F2)
    return cm

def compute_euclidean_distance(F, reference=(0,0)):
    if F is None or len(F)==0:
        return None
    dists = np.linalg.norm(F - np.array(reference), axis=1)
    return np.mean(dists)

###############################################################################
# 1) Sidebar: Method Category
###############################################################################
st.sidebar.header("Select Method Category")
method_category = st.sidebar.selectbox("Optimization Category", ["Scalarization","Pareto-based"])

if method_category=="Scalarization":
    method_name = st.sidebar.selectbox("Scalarization Method", ["Weighted Sum","ε-Constraint","Goal Seeking"])
else:
    method_name = st.sidebar.selectbox("Pareto-based Method", ["NSGA2","MOEA/D","SPEA2"])

###############################################################################
# 2) Scalarization Parameters
###############################################################################
scalar_params = {}
if method_category == "Scalarization":
    if method_name == "Weighted Sum":
        w1 = st.sidebar.slider("Weight for Cost (w1)", 0.0,1.0,0.5)
        w2 = 1.0 - w1
        scalar_params["w1"] = w1
        scalar_params["w2"] = w2
    elif method_name=="ε-Constraint":
        eps_val = st.sidebar.number_input("Overpotential Constraint (ε)", value=1.0)
        scalar_params["eps_value"] = eps_val
    elif method_name=="Goal Seeking":
        c_goal = st.sidebar.number_input("Cost Goal (C)", value=10.0)
        eta_goal = st.sidebar.number_input("Overpotential Goal (V)", value=0.5)
        scalar_params["goals"] = (c_goal, eta_goal)

###############################################################################
# 3) Operating & Catalyst/Constraints
###############################################################################
st.sidebar.header("Operating & Constraint Parameters")

A_cell = st.sidebar.number_input("Cell Active Area (cm²)", value=50.0)
j = st.sidebar.number_input("Operating Current Density j (A/cm²)", value=0.05)
R=8.314
T= st.sidebar.number_input("Temperature (K)", value=353.0)
alpha = st.sidebar.number_input("alpha", value=0.5)
n_e   = st.sidebar.number_input("n (electrons)", value=2)
F_const = 96485.0

C_bulk_a = st.sidebar.number_input("Anode Bulk Conc (mol/cm³)", value=0.01)
D_a      = st.sidebar.number_input("Anode Diffusivity (cm²/s)", value=1e-2)
tau_a    = st.sidebar.number_input("Anode Tortuosity", value=1.2)

C_bulk_c = st.sidebar.number_input("Cathode Bulk Conc (mol/cm³)", value=0.01)
D_c      = st.sidebar.number_input("Cathode Diffusivity (cm²/s)", value=1e-2)
tau_c    = st.sidebar.number_input("Cathode Tortuosity", value=1.2)

eta_max = st.sidebar.number_input("Max Overpotential (V)", value=2.0)

# Catalyst: Anode
rho_cat_a = st.sidebar.number_input("Anode Catalyst Density (g/cm³)", value=11.66)
c_cat_a   = st.sidebar.number_input("Anode Catalyst Cost ($/g)", value=100.0)
j0_a      = st.sidebar.number_input("Anode j0 (A/cm²_active)", value=1e-3)
a_a       = st.sidebar.number_input("Anode Tafel a (V)", value=0.1)
b_a       = st.sidebar.number_input("Anode Tafel b (V)", value=0.05)

# Catalyst: Cathode
rho_cat_c = st.sidebar.number_input("Cathode Catalyst Density (g/cm³)", value=21.45)
c_cat_c   = st.sidebar.number_input("Cathode Catalyst Cost ($/g)", value=60.0)
j0_c      = st.sidebar.number_input("Cathode j0 (A/cm²_active)", value=1e-3)
a_c       = st.sidebar.number_input("Cathode Tafel a (V)", value=0.08)
b_c       = st.sidebar.number_input("Cathode Tafel b (V)", value=0.04)


###############################################################################
# 4) Constraint Sliders
###############################################################################
st.sidebar.header("Constraint Bounds")

eps_a_min, eps_a_max = st.sidebar.slider("Anode Porosity Range", 0.0,1.0,(0.01,0.99))
eps_c_min, eps_c_max = st.sidebar.slider("Cathode Porosity Range",0.0,1.0,(0.01,0.99))

delta_a_min, delta_a_max = st.sidebar.slider("Anode Thickness δ_a (cm)", 1e-4,0.05,(1e-4,0.01))
delta_c_min, delta_c_max = st.sidebar.slider("Cathode Thickness δ_c (cm)",1e-4,0.05,(1e-4,0.01))

Scat_a_min, Scat_a_max = st.sidebar.slider("Anode S_cat Range", 1e2,1e6,(1e3,1e5))
Scat_c_min, Scat_c_max = st.sidebar.slider("Cathode S_cat Range",1e2,1e6,(1e3,1e5))

L_a_min, L_a_max = st.sidebar.slider("Anode Catalyst Loading L_a",0.0,0.05,(0.001,0.02))
L_c_min, L_c_max = st.sidebar.slider("Cathode Catalyst Loading L_c",0.0,0.05,(0.001,0.02))

tau_a_min, tau_a_max = st.sidebar.slider("Anode Tortuosity Range",0.5,5.0,(1.0,3.0))
tau_c_min, tau_c_max = st.sidebar.slider("Cathode Tortuosity Range",0.5,5.0,(1.0,3.0))

SA_a_min = st.sidebar.number_input("Anode Min Effective Surface (cm²)", value=100.0)
SA_c_min = st.sidebar.number_input("Cathode Min Effective Surface (cm²)", value=100.0)

j_min, j_max = st.sidebar.slider("Operating Current Density j Range",0.0,1.0,(0.0,0.1))

###############################################################################
# 5) Algorithm Settings (For Pareto-based)
###############################################################################
st.sidebar.header("Algorithm Settings")
pop_size = st.sidebar.number_input("Population Size", value=40, min_value=10)
n_gen    = st.sidebar.number_input("Number of Generations", value=30, min_value=10)

###############################################################################
# RUN Optimization
###############################################################################
if st.button("Run Optimization"):
    with st.spinner("Running optimization..."):
        run_kwargs = dict(
            category=method_category,
            method=method_name,
            A_cell=A_cell,
            j=j,
            R=R,
            T=T,
            alpha=alpha,
            n=n_e,
            F=F_const,
            C_bulk_a=C_bulk_a,
            D_a=D_a,
            tau_a=tau_a,
            C_bulk_c=C_bulk_c,
            D_c=D_c,
            tau_c=tau_c,
            eta_max=eta_max,
            rho_cat_a=rho_cat_a,
            c_cat_a=c_cat_a,
            j0_a=j0_a,
            a_a=a_a,
            b_a=b_a,
            rho_cat_c=rho_cat_c,
            c_cat_c=c_cat_c,
            j0_c=j0_c,
            a_c=a_c,
            b_c=b_c,
            eps_a_min=eps_a_min, eps_a_max=eps_a_max,
            eps_c_min=eps_c_min, eps_c_max=eps_c_max,
            delta_a_min=delta_a_min, delta_a_max=delta_a_max,
            delta_c_min=delta_c_min, delta_c_max=delta_c_max,
            Scat_a_min=Scat_a_min, Scat_a_max=Scat_a_max,
            Scat_c_min=Scat_c_min, Scat_c_max=Scat_c_max,
            L_a_min=L_a_min, L_a_max=L_a_max,
            L_c_min=L_c_min, L_c_max=L_c_max,
            tau_a_min=tau_a_min, tau_a_max=tau_a_max,
            tau_c_min=tau_c_min, tau_c_max=tau_c_max,
            SA_a_min=SA_a_min, SA_c_min=SA_c_min,
            j_min=j_min, j_max=j_max,
            scalar_params=scalar_params
        )

        # For multi-objective => pass pop_size & n_gen
        if method_category=="Pareto-based":
            try:
                res = run_optimization(
                    **run_kwargs,
                    pop_size=pop_size,
                    n_gen=n_gen
                )
            except Exception as e:
                st.error(f"An unexpected error occurred (Pareto-based): {e}")
                st.stop()
        else:
            # Scalarization => do not pass pop_size/n_gen
            try:
                res = run_optimization(**run_kwargs)
            except Exception as e:
                st.error(f"An unexpected error occurred (Scalarization): {e}")
                st.stop()

    if res.X is None or res.F is None:
        st.error("No feasible solutions or solver failure.")
    else:
        st.success("Optimization Complete!")

        if method_category=="Scalarization":
            # single-objective => 1 best solution
            st.write("**Best Single-Objective Solution**")
            var_names = ["delta_a","eps_a","Scat_a","delta_c","eps_c","Scat_c"]
            for var,val in zip(var_names, res.X):
                st.write(f" - **{var}**: {val:.6f}")
            st.write("Objective Value:", res.F[0])
        else:
            # Pareto-based => multiple solutions in res.X, res.F
            F = res.F
            X = res.X
            if F.ndim==1 or len(F)==1:
                st.write("Single solution found. Possibly everything is dominated or infeasible.")
            else:
                var_names = ["delta_a","eps_a","Scat_a","delta_c","eps_c","Scat_c"]
                obj_names = ["Cost","Overpotential"]
                df = create_dataframe(X,F,var_names,obj_names)

                st.subheader("Pareto Front (Objectives)")
                fig_obj = px.scatter(df, x="Cost", y="Overpotential", 
                                     hover_data=df.columns,
                                     title=f"{method_name} Pareto Front")
                st.plotly_chart(fig_obj, use_container_width=True)

                st.subheader("Decision Variables (Parallel Coordinates)")
                fig_par = px.parallel_coordinates(
                    df, color="Cost", labels={col:col for col in df.columns},
                    title="Decision Space"
                )
                st.plotly_chart(fig_par, use_container_width=True)

                # Performance metrics
                st.subheader("Performance Metrics")
                metric_sel = st.selectbox("Pick a metric:", ["None","Hypervolume","C-Metric","Euclidean Distance"])
                if metric_sel=="Hypervolume":
                    refpt = (1.1*max(F[:,0]), 1.1*max(F[:,1]))
                    hv_val = compute_hypervolume(F, ref_point=refpt)
                    st.write(f"Hypervolume ~ {hv_val:.4f}")
                elif metric_sel=="C-Metric":
                    c_val = compute_c_metric(F,F2=F)
                    st.write(f"C-Metric self coverage = {c_val:.4f}")
                elif metric_sel=="Euclidean Distance":
                    dist_val = compute_euclidean_distance(F,(0,0))
                    st.write(f"Average distance to (0,0) = {dist_val:.4f}")

                st.subheader("All Pareto Solutions")
                st.dataframe(df)

                # Show top 3 solutions by sum of objectives
                sums = F[:,0]+F[:,1]
                idx_sorted = np.argsort(sums)
                top3 = idx_sorted[:3]
                st.markdown("### Top 3 Solutions (lowest Cost+Overpotential)")
                for rank, idx in enumerate(top3):
                    cost_val, overp_val = F[idx]
                    st.markdown(f"**Rank {rank+1}** => Cost={cost_val:.4f}, Overpot={overp_val:.4f}")
                    st.json({var_names[i]: X[idx][i] for i in range(len(var_names))})

###############################################################################
# Save, Load, Clear
###############################################################################
st.sidebar.header("Results I/O")

if "results_data" not in st.session_state:
    st.session_state["results_data"] = None

def save_results_to_disk(data, fname=None):
    if fname is None:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"results_{now_str}.json"
    path = os.path.join(RESULTS_DIR,fname)
    with open(path,"w") as f:
        json.dump(data,f,indent=2)
    return fname

def load_results_from_disk(fname):
    path = os.path.join(RESULTS_DIR,fname)
    if not os.path.exists(path):
        return None
    with open(path,"r") as f:
        return json.load(f)

def clear_results_on_disk(pattern="results_*.json"):
    import glob
    flist = glob.glob(os.path.join(RESULTS_DIR, pattern))
    for file in flist:
        os.remove(file)

c1,c2,c3 = st.sidebar.columns(3)
with c1:
    if st.button("Save Results"):
        if "res" not in locals():
            st.sidebar.warning("No results to save. Please run optimization first.")
        else:
            # minimal approach => store the final solutions
            results_dict = {
                "params": run_kwargs,
                "X": res.X.tolist() if isinstance(res.X, np.ndarray) else None,
                "F": res.F.tolist() if isinstance(res.F, np.ndarray) else None
            }
            fname = save_results_to_disk(results_dict)
            st.session_state["results_data"] = results_dict
            st.sidebar.success(f"Saved results => {fname}")

with c2:
    if st.button("Load Results"):
        files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
        if not files:
            st.sidebar.warning("No .json results files found.")
        else:
            latest = max(files,key=lambda x: os.path.getctime(os.path.join(RESULTS_DIR,x)))
            loaded = load_results_from_disk(latest)
            if loaded is not None:
                st.session_state["results_data"] = loaded
                st.sidebar.info(f"Loaded => {latest}")
            else:
                st.sidebar.warning(f"Failed loading => {latest}")

with c3:
    if st.button("Clear Results"):
        clear_results_on_disk()
        st.session_state["results_data"] = None
        st.sidebar.warning("Cleared all results .json files.")

if st.session_state["results_data"] is not None:
    st.sidebar.markdown("---")
    st.sidebar.write("### Loaded Results Data:")
    st.sidebar.json(st.session_state["results_data"])
