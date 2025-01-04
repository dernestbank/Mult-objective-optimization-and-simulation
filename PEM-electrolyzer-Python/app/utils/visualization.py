# utils/visualization.py

import pandas as pd
import plotly.express as px

def create_dataframe(X, F, var_names=None, obj_names=None):
    """
    Create a combined DataFrame of decision variables (X) and objectives (F).
    var_names: list of decision variable names
    obj_names: list of objective names
    """
    N, D = X.shape
    M = F.shape[1]

    if var_names is None:
        var_names = [f"Var_{i+1}" for i in range(D)]
    if obj_names is None:
        obj_names = [f"Obj_{j+1}" for j in range(M)]

    df = pd.DataFrame(X, columns=var_names)
    for j, name in enumerate(obj_names):
        df[name] = F[:, j]
    return df

def scatter_matrix_plot(df, dimensions=None, color=None):
    """
    Plotly scatter matrix.
    dimensions: columns to include
    color: column name for coloring points
    """
    if dimensions is None:
        dimensions = df.columns
    fig = px.scatter_matrix(df, dimensions=dimensions, color=color)
    return fig

def parallel_coordinates_plot(df, dimensions=None, color=None):
    """
    Plotly parallel coordinates plot.
    """
    if dimensions is None:
        dimensions = df.columns
    fig = px.parallel_coordinates(df, dimensions=dimensions, color=color)
    return fig

def heatmap_plot(df, x_var, y_var, z_var):
    """
    Heatmap using pivot_table to aggregate z_var over x_var,y_var
    """
    pivot_df = df.pivot_table(index=y_var, columns=x_var, values=z_var, aggfunc='mean')
    fig = px.imshow(pivot_df, aspect='auto', color_continuous_scale='Viridis', 
                    labels=dict(color=z_var), title=f"Heatmap of {z_var}")
    return fig

def glyph_plot(df, x, y, size_col=None, color_col=None):
    """
    Scatter plot as a 'glyph' plot with size and color encodings.
    """
    fig = px.scatter(df, x=x, y=y, size=size_col, color=color_col, 
                     title="Glyph Plot", hover_data=df.columns)
    return fig
