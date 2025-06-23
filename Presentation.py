import dash
from dash import html, dcc, Input, Output, State, ctx, callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from Business import GraphBuilder, MapId

# Build the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Amos House Price Dashboard"

app.layout = html.Div([
    # Sidebar toggle button
    html.Button("\u2630", className="button", id="toggle"),

    # Sidebar
    html.Div([
        html.H4("Select Plot", style={"marginBottom": "20px"}),
        dcc.Dropdown(
            id="plot-dropdown",
            options=[
                {"label": "📊 House Sale Price Histogram", "value": "hist"},
                {"label": "📈 PCA vs SalePrice", "value": "pca"},
                {"label": "📉 Learning Curve - Linear", "value": "lc-linear"},
                {"label": "🌲 Learning Curve - Tree", "value": "lc-tree"},
                {"label": "🌳 Learning Curve - Forest", "value": "lc-forest"},
                {"label": "🚀 Learning Curve - Gradient", "value": "lc-gradient"},
                {"label": "🟠 Scatter: Linear", "value": "scatter-linear"},
                {"label": "🔵 Scatter: Tree", "value": "scatter-tree"},
                {"label": "🟢 Scatter: Forest", "value": "scatter-forest"},
                {"label": "🟣 Scatter: Gradient", "value": "scatter-gradient"},
                {"label": "📉 Residuals: Linear", "value": "resid-linear"},
                {"label": "📉 Residuals: Tree", "value": "resid-tree"},
                {"label": "📉 Residuals: Forest", "value": "resid-forest"},
                {"label": "📉 Residuals: Gradient", "value": "resid-gradient"},
                {"label": "🔥 Feature Importances: Linear", "value": "feat-linear"},
                {"label": "🔥 Feature Importances: Tree", "value": "feat-tree"},
                {"label": "🔥 Feature Importances: Forest", "value": "feat-forest"},
                {"label": "🔥 Feature Importances: Gradient", "value": "feat-gradient"},
            ],
            placeholder="Select a visualization...",
            style={"marginBottom": "30px"}
        ),

        html.H4("Map ID for Submission"),
        dcc.Input(id="input-label", type="text", placeholder="Enter submission label...", style={"marginBottom": "10px", "width": "100%"}),
        html.Button("Download Submission", id="submit-btn", className="train-model-button"),
        html.Div(id="download-text", style={"marginTop": "15px", "fontSize": "14px", "color": "green"})

    ], className="side-bar", id="sidebar"),

    # Main panel
    html.Div([
        html.Div("\ud83d\udcf1 Swipe left/right to view full plot", style={"fontSize": "14px", "color": "#888"}),

        dcc.Loading(
            id="loading-graph",
            type="circle",
            children=html.Div(
                dcc.Graph(id="main-graph"),
                className="responsive-plot-container"
            )
        )
    ], style={"marginLeft": "240px", "padding": "20px"})
])

# Sidebar toggle callback
@callback(
    Output("sidebar", "className"),
    [Input("toggle", "n_clicks")],
    [State("sidebar", "className")]
)
def toggle_sidebar(n, current):
    if not n:
        raise PreventUpdate
    if "active" in current:
        return "side-bar"
    return "side-bar active"

# Main graph callback
@callback(
    Output("main-graph", "figure"),
    [Input("plot-dropdown", "value")]
)
def update_graph(plot_type):
    if not plot_type:
        raise PreventUpdate

    gb = GraphBuilder()

    if plot_type == "hist":
        return gb.house_price_hist()
    elif plot_type == "pca":
        return gb.pca_plot()
    elif plot_type == "lc-linear":
        return gb.learning_curve_linear()
    elif plot_type == "lc-tree":
        return gb.learning_curve_tree()
    elif plot_type == "lc-forest":
        return gb.learning_curve_forest()
    elif plot_type == "lc-gradient":
        return gb.learning_curve_gradient()
    elif plot_type == "scatter-linear":
        return gb.scatter_plot("linear")
    elif plot_type == "scatter-tree":
        return gb.scatter_plot("tree")
    elif plot_type == "scatter-forest":
        return gb.scatter_plot("forest")
    elif plot_type == "scatter-gradient":
        return gb.scatter_plot("gradient")
    elif plot_type == "resid-linear":
        return gb.residual_plot()
    elif plot_type == "resid-tree":
        _, fig = gb.residual_tree_plot("tree")
        return fig
    elif plot_type == "resid-forest":
        _, fig = gb.residual_tree_plot("forest")
        return fig
    elif plot_type == "resid-gradient":
        _, fig = gb.residual_tree_plot("gradient")
        return fig
    elif plot_type == "feat-linear":
        return gb.feature_importance("linear")
    elif plot_type == "feat-tree":
        return gb.feature_importance("tree")
    elif plot_type == "feat-forest":
        return gb.feature_importance("forest")
    elif plot_type == "feat-gradient":
        return gb.feature_importance("gradient")

    raise PreventUpdate

# Submission ID mapping
@callback(
    Output("download-text", "children"),
    [Input("submit-btn", "n_clicks")],
    [State("input-label", "value")]
)
def submit_id(n, label):
    if not n:
        raise PreventUpdate
    if not label:
        return "Please enter a label."
    try:
        MapId().get_id(label)
        return f"✅ '{label}_submission.csv' downloaded successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run_server(debug=True)
