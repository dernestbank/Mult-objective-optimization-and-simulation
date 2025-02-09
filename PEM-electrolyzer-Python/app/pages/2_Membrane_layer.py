import streamlit as st


st.set_page_config(
    page_title="PEM Membrane Layer Design",
    page_icon="⚡",
    layout="wide",
)

# --- HEADER ---
col1, col2 = st.columns([1,1])
with col1:
    st.write("S2D2 Lab | Penn State")
st.title("Membrane Layer Design")

# Remove the function and simplify the page
st.write("""
## Membrane Layer Configuration

This section allows you to configure and optimize the membrane layer parameters 
of your PEM electrolyzer.
""")

# Add your membrane layer design content here