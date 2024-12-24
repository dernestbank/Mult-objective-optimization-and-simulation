import pandas as pd
import plotly.express as px

def create_dataframe(X, F, var_names=None, obj_names=None):
    """
    Create a dataframe with decision variables and objectives.
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
    Create a scatter matrix plot using Plotly.
    dimensions: list of columns to include in the scatter matrix
    color: column name for coloring points
    """
    if dimensions is None:
        dimensions = df.columns
    fig = px.scatter_matrix(df, dimensions=dimensions, color=color)
    return fig

def parallel_coordinates_plot(df, dimensions=None, color=None):
    """
    Create a parallel coordinates plot.
    dimensions: list of columns to include
    color: column name for coloring lines
    """
    if dimensions is None:
        dimensions = df.columns
    fig = px.parallel_coordinates(df, dimensions=dimensions, color=color)
    return fig

def heatmap_plot(df, x_var, y_var, z_var):
    """
    Create a heatmap from a dataframe.
    x_var: column for x axis
    y_var: column for y axis
    z_var: column for heatmap values (aggregated)
    """
    pivot_df = df.pivot_table(index=y_var, columns=x_var, values=z_var, aggfunc='mean')
    fig = px.imshow(pivot_df, aspect='auto', color_continuous_scale='Viridis', 
                    labels=dict(color=z_var), title=f"Heatmap of {z_var}")
    return fig

def glyph_plot(df, x, y, size_col=None, color_col=None):
    """
    A glyph plot can be approximated by using a scatter plot with symbol/size encoding.
    x, y: columns for axes
    size_col: column for sizing glyphs
    color_col: column for coloring glyphs
    """
    fig = px.scatter(df, x=x, y=y, size=size_col, color=color_col, 
                     title="Glyph Plot", hover_data=df.columns)
    return fig
