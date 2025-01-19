import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots  # Make sure to import make_subplots

# Univariate analysis: Histogram, Boxplot, KDE, Violin Plot (3x3 matrix)
def univariate_analysis(data, column=None):
    # Default to the first column if no column is provided
    column = column or data.columns[0]

    # Check if the column exists in the dataset
    if column not in data.columns:
        print(f"Column {column} not found in the dataset.")
        return

    # Create figure with subplots
    fig = make_subplots(rows=3, cols=3, subplot_titles=[
        f"Histogram for {column}",
        f"Boxplot for {column}",
        f"KDE Plot for {column}",
        f"Violin Plot for {column}"
    ])
    
    # Histogram
    fig.add_trace(go.Histogram(x=data[column], nbinsx=30, name=f"Histogram {column}"),
                  row=1, col=1)

    # Boxplot
    fig.add_trace(go.Box(y=data[column], name=f"Boxplot {column}", boxmean='sd'),
                  row=1, col=2)

    # KDE Plot (using Kernel Density Estimation)
    fig.add_trace(go.Histogram(x=data[column], histnorm='density', nbinsx=30, name=f"KDE {column}"),
                  row=1, col=3)

    # Violin Plot
    fig.add_trace(go.Violin(y=data[column], box_visible=True, line_color='orange', name=f"Violin {column}"),
                  row=2, col=1)

    fig.update_layout(height=900, width=900, title_text=f"Univariate Analysis for {column}",
                      showlegend=False)

    fig.show()

# Bivariate analysis: Scatter Plot, Line Plot, Bubble Chart, Hexbin Plot (3x3 matrix)
def bivariate_analysis(data, x_column=None, y_column=None):
    # Default to the first and second columns if no columns are provided
    x_column = x_column or data.columns[0]
    y_column = y_column or data.columns[1]
    
    # Check if the columns exist in the dataset
    if x_column not in data.columns or y_column not in data.columns:
        print(f"Columns {x_column} or {y_column} not found in the dataset.")
        return

    # Create figure with subplots
    fig = make_subplots(rows=3, cols=3, subplot_titles=[
        f"Scatter Plot: {x_column} vs {y_column}",
        f"Line Plot: {x_column} vs {y_column}",
        f"Bubble Chart: {x_column} vs {y_column}",
        f"Hexbin Plot: {x_column} vs {y_column}"
    ])

    # Scatter Plot
    fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='markers', name="Scatter"),
                  row=1, col=1)

    # Line Plot
    fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='lines', name="Line"),
                  row=1, col=2)

    # Bubble Chart (using size from a third column if available)
    if len(data.columns) > 2:
        fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='markers',
                                 marker=dict(size=data[data.columns[2]] * 10, color='green', opacity=0.6),
                                 name="Bubble Chart"),
                      row=1, col=3)
    else:
        # If no third column, use a constant size for bubbles
        fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='markers',
                                 marker=dict(size=10, color='green', opacity=0.6),
                                 name="Bubble Chart"),
                      row=1, col=3)

    # Hexbin Plot (using scatter with color scale)
    fig.add_trace(go.Scatter(x=data[x_column], y=data[y_column], mode='markers',
                             marker=dict(color=np.sqrt(data[x_column]**2 + data[y_column]**2),
                                         colorscale='Blues', opacity=0.5, size=7),
                             name="Hexbin"),
                  row=2, col=1)

    fig.update_layout(height=900, width=900, title_text=f"Bivariate Analysis: {x_column} vs {y_column}",
                      showlegend=True)

    fig.show()

# Multivariate analysis: Pairplot, Correlation Heatmap, 3D Scatter Plot, Facet Grid (3x3 matrix)
def multivariate_analysis(data, columns=None):
    # Default to the first 5 columns if no specific columns are provided
    columns = columns or data.columns[:5]
    
    # Check that enough columns are available for multivariate analysis
    if len(columns) < 2:
        print("Not enough columns for bivariate analysis.")
        return

    # Pairplot (use Plotly Express)
    fig_pairplot = px.scatter_matrix(data[columns])
    fig_pairplot.update_layout(title=f"Pairplot for {', '.join(columns)}")
    fig_pairplot.show()

    # Correlation Heatmap
    correlation_matrix = data[columns].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns,
                                            y=correlation_matrix.columns, colorscale='Viridis'))
    fig_heatmap.update_layout(title="Correlation Heatmap", xaxis_title="Variables", yaxis_title="Variables")
    fig_heatmap.show()

    # 3D Scatter Plot (if there are at least 3 columns)
    if len(columns) >= 3:
        fig_3d = go.Figure(data=[go.Scatter3d(x=data[columns[0]], y=data[columns[1]], z=data[columns[2]],
                                             mode='markers', marker=dict(size=5, color='purple'))])
        fig_3d.update_layout(title=f"3D Scatter Plot: {columns[0]} vs {columns[1]} vs {columns[2]}",
                             scene=dict(xaxis_title=columns[0], yaxis_title=columns[1], zaxis_title=columns[2]))
        fig_3d.show()

    # Facet Grid (using Plotly Express)
    if 'category' in data.columns:  # Adjust as needed for your data
        fig_facet = px.scatter(data_frame=data, x=columns[0], y=columns[1], color='category')
        fig_facet.update_layout(title=f"Facet Grid: {columns[0]} vs {columns[1]}")
        fig_facet.show()
