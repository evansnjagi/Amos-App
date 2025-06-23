import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx
from Business import GraphBuilder, MapId
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout definition
def create_layout():
    return html.Div([
        html.Button("Menu", className="button", id="toggle-sidebar"),

        html.Div([
            html.H4("Amos House Price App", style={"textAlign": "center"}),
            html.Hr(),
            dbc.Nav([
                dbc.Button("Home", id="btn-home", className="mb-2 w-100", color="primary"),
                dbc.Button("Learning Curves", id="btn-lc", className="mb-2 w-100", color="primary"),
                dbc.Button("Feature Importance", id="btn-fi", className="mb-2 w-100", color="primary"),
                dbc.Button("Predictions", id="btn-predictions", className="mb-2 w-100", color="primary"),
                dbc.Button("Residuals", id="btn-residual", className="mb-2 w-100", color="primary"),
                dbc.Button("Know More", id="btn-about", className="mb-2 w-100", color="primary")
            ], vertical=True),
            html.Br(),
            html.Div([
                dbc.Input(id="id-label", placeholder="Enter label for submission file", type="text"),
                dbc.Button("Generate Submission", id="download-button", color="success", className="mt-2 w-100"),
                dcc.Download(id="download-component"),
                html.Div(id="download-message", className="text-success mt-2")
            ])
        ], className="side-bar", id="sidebar"),

        html.Div([
            html.Div(id="main-content"),
            dcc.Loading(
                id="loading-output",
                type="circle",
                children=html.Div(id="plots-container"),
                fullscreen=False
            )
        ], className="main-content")
    ])

# Set app layout
app.layout = create_layout

# Sidebar toggle functionality
@app.callback(
    Output("sidebar", "className"),
    Input("toggle-sidebar", "n_clicks"),
    State("sidebar", "className")
)
def toggle_sidebar(n, className):
    if n:
        if "active" in className:
            return "side-bar"
        else:
            return "side-bar active"
    return className

# Routing logic
@app.callback(
    Output("plots-container", "children"),
    [Input("btn-home", "n_clicks"),
     Input("btn-lc", "n_clicks"),
     Input("btn-fi", "n_clicks"),
     Input("btn-predictions", "n_clicks"),
     Input("btn-residual", "n_clicks"),
     Input("btn-about", "n_clicks")]
)
def render_content(home, lc, fi, predictions, residual, about):
    triggered = ctx.triggered_id

    if triggered == "btn-home":
        return html.Div([
            html.H5("Quick EDA", className="text-center mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(figure=GraphBuilder().house_price_hist()), width=12)]),
            dbc.Row([dbc.Col(dcc.Graph(figure=GraphBuilder().pca_plot()), width=12)])
        ])

    elif triggered == "btn-lc":
        return html.Div([
            html.H5("Training and Validation Learning Curves", className="text-center mb-4"),
            dbc.Tabs([
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_linear()), label="Linear Model"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_tree()), label="Decision Tree"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_forest()), label="Random Forest"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_gradient()), label="Gradient Boosting")
            ])
        ])

    elif triggered == "btn-fi":
        return html.Div([
            html.H5("Feature Importance & Performance Analysis", className="text-center mb-4"),
            dbc.Tabs([
                dbc.Tab(dcc.Graph(figure=GraphBuilder().feature_importance("linear")), label="Linear"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().feature_importance("tree")), label="Decision Tree"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().feature_importance("forest")), label="Random Forest"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().feature_importance("gradient")), label="Gradient Boosting")
            ])
        ])

    elif triggered == "btn-predictions":
        return html.Div([
            html.H5("Prediction Scatter Plots", className="text-center mb-4"),
            dbc.Tabs([
                dbc.Tab(dcc.Graph(figure=GraphBuilder().scatter_plot("linear")), label="Linear"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().scatter_plot("tree")), label="Decision Tree"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().scatter_plot("forest")), label="Random Forest"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().scatter_plot("gradient")), label="Gradient Boosting")
            ])
        ])

    elif triggered == "btn-residual":
        return html.Div([
            html.H5("Residual Plots", className="text-center mb-4"),
            dbc.Tabs([
                dbc.Tab(dcc.Graph(figure=GraphBuilder().residual_plot()), label="Linear"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().residual_tree_plot("tree")[1]), label="Decision Tree"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().residual_tree_plot("forest")[1]), label="Random Forest"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().residual_tree_plot("gradient")[1]), label="Gradient Boosting")
            ])
        ])

    elif triggered == "btn-about":
        return html.Div([
            html.H4("About This Project", className="text-center"),
            html.P("This web application was built as part of the Kaggle competition 'House Prices - Advanced Regression Techniques'.",
                   className="lead text-center"),
            html.P("It demonstrates various regression models and evaluates them using learning curves, feature importance, scatter plots, and residual analysis.",
                   className="text-center"),
            html.P("Visit the Kaggle competition here:", className="text-center mb-1"),
            html.Div(html.A("Kaggle: House Prices - Advanced Regression Techniques",
                            href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
                            target="_blank"), className="text-center"),
            html.P("View project on GitHub:", className="text-center mt-3"),
            html.Div(html.A("GitHub Repository",
                            href="https://github.com/evansnjagi/Amos-App",
                            target="_blank"), className="text-center")
        ])

    return html.Div("Select a tab to view its content.")

# Handle submission download
@app.callback(
    [Output("download-component", "data"),
     Output("download-message", "children")],
    [Input("download-button", "n_clicks")],
    [State("id-label", "value")]
)
def download_submission(n, label):
    if not n or not label:
        raise PreventUpdate
    df = MapId().get_id(label)
    return dcc.send_data_frame(df.to_csv, filename=f"{label}_submission.csv"), "Your CSV file is ready. Click the download button again to save."
