# List of available operations for querying and chart lists
def get_operations_list():
    return [
        "max", "min", "sum", "difference", "top", "mean", "average",
        "groupby", "filter", "sort", "median", "count", "pivot",
        "join", "merge", "null", "range", "date", "trend", "normalize",
        "bin", "aggregate", "unique", "standardize"
    ]

def chart_list():
    return["histogram", "scatter", "lineplot", "bar", "boxplot", "heatmap", "piechart", 
    "areachart", "violin", "sunburst", "treemap", "funnel", "density_heatmap", "density_contour", "clustered_column"]
