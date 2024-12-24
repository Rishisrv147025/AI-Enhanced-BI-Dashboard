import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Univariate analysis: Histogram and Boxplot
def univariate_analysis(data):
    univariate_col = data.columns[0]  # Select the first column by default
    fig = px.histogram(data, x=univariate_col, title=f"Histogram for {univariate_col}")
    fig.show()

    fig_box = px.box(data, y=univariate_col, title=f"Boxplot for {univariate_col}")
    fig_box.show()

# Bivariate analysis: Scatter plot
def bivariate_analysis(data):
    bivariate_x = data.columns[0]
    bivariate_y = data.columns[1]
    fig_scatter = px.scatter(data, x=bivariate_x, y=bivariate_y, title=f"Scatter plot: {bivariate_x} vs {bivariate_y}")
    fig_scatter.show()

# Multivariate analysis: Pairplot and Correlation Heatmap
def multivariate_analysis(data):
    selected_cols = data.columns[:5]  # Select the first 5 columns for multivariate analysis
    sns.pairplot(data[selected_cols])
    plt.show()

    correlation_matrix = data[selected_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.show()
