import streamlit as st

st.set_page_config(
    page_title="Documentation",
    page_icon="📚",
    layout="wide",
)

# --- HEADER ---
st.title("Documentation")
st.write("Learn how to use the PEM Electrolyzer Optimization Playground")

# Table of Contents
st.sidebar.header("Contents")
page = st.sidebar.radio("Go to", 
    ["Getting Started", 
     "Catalyst Layer Design", 
     "Membrane Layer Design",
     "Optimization Methods",
     "Parameters Guide",
     "Results Interpretation"])

if page == "Getting Started":
    st.header("Getting Started")
    st.write("""
    Welcome to the PEM Electrolyzer Optimization Playground! This tool helps you explore and optimize 
    the design of PEM electrolyzers. Here's how to get started:
    
    1. **Navigate the Interface:**
       - Use the sidebar to access different features
       - Switch between pages using the navigation menu
       - Adjust parameters using the sliders and input fields
    
    2. **Choose Your Focus:**
       - Catalyst Layer Design: Optimize catalyst loading and distribution
       - Membrane Layer Design: Configure membrane properties
       
    3. **Run Optimizations:**
       - Select optimization method (Scalarization or Pareto-based)
       - Set constraints and objectives
       - Analyze results through visualizations
    """)

elif page == "Catalyst Layer Design":
    st.header("Catalyst Layer Design")
    st.write("""
    ### Key Parameters
    
    1. **Layer Thickness (δ)**
       - Units: cm
       - Impact: Affects both activation and mass transport losses
       - Trade-off: Thicker layers provide more active sites but increase transport resistance
    
    2. **Porosity (ε)**
       - Range: 0-1
       - Impact: Balances active sites and mass transport
       - Trade-off: Higher porosity improves transport but reduces active sites
    
    3. **Specific Surface Area (Scat)**
       - Units: cm²/g
       - Impact: Determines active site density
       - Optimization: Higher values generally improve performance
    """)

elif page == "Optimization Methods":
    st.header("Optimization Methods")
    st.write("""
    ### Available Methods
    
    1. **Scalarization**
       - Weighted Sum: Combine objectives with user-defined weights
       - Goal Seeking: Target specific performance goals
    
    2. **Pareto-based**
       - NSGA-II: Popular multi-objective genetic algorithm
       - MOEA/D: Decomposition-based approach
       - SPEA2: Strength Pareto Evolutionary Algorithm
    """)

elif page == "Parameters Guide":
    st.header("Parameters Guide")
    
    with st.expander("Operating Parameters"):
        st.write("""
        - **Current Density (j)**: A/cm²
        - **Temperature (T)**: K
        - **Cell Area (A_cell)**: cm²
        """)
    
    with st.expander("Material Properties"):
        st.write("""
        - **Catalyst Density**: g/cm³
        - **Catalyst Cost**: $/g
        - **Exchange Current Density**: A/cm²
        """)
    
    with st.expander("Transport Properties"):
        st.write("""
        - **Diffusivity**: cm²/s
        - **Bulk Concentration**: mol/cm³
        - **Tortuosity**: dimensionless
        """)

elif page == "Results Interpretation":
    st.header("Results Interpretation")
    st.write("""
    ### Understanding the Output
    
    1. **Cost Function**
       - Units: $
       - Interpretation: Total catalyst cost
       - Key factors: Loading, material cost
    
    2. **Overpotential**
       - Units: V
       - Components: Activation + Concentration
       - Impact on efficiency
    
    3. **Pareto Front**
       - Trade-off between cost and performance
       - How to select preferred solutions
       - Practical considerations
    """)

# Footer
st.write("---")
st.write("""
💡 **Tip:** Use the sidebar navigation to jump between sections.

Need more help? Contact the S2D2 Lab team or visit our [GitHub repository](https://github.com/yourusername/PEM-electrolyzer-Python).
""")