import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, ctx, callback_context
from Business import GraphBuilder, MapId
import pandas as pd
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Layout definition
def create_layout():
    return html.Div([
        # Sidebar toggle button
        html.Button("☰", className="button", id="toggle-sidebar"),

        # Sidebar
        html.Div([
            html.H4("📊 Amos House Price App", style={"textAlign": "center"}),
            html.Hr(),
            dbc.Nav([
                dbc.NavLink("🏠 Home", href="#", id="link-home"),
                dbc.NavLink("📈 Learning Curves", href="#", id="link-lc"),
                dbc.NavLink("📉 Feature Importance", href="#", id="link-fi"),
                dbc.NavLink("📋 Know More", href="#", id="link-about")
            ], vertical=True, pills=True),
            html.Br(),
            dbc.Button("Download Submission", id="download-button", className="train-model-button"),
            dcc.Download(id="download-component")
        ], className="side-bar", id="sidebar"),

        # Main content
        html.Div([
            html.Div(id="main-content"),
            dcc.Loading(
                id="loading-output",
                type="circle",
                children=html.Div(id="plots-container")
            )
        ], style={"marginLeft": "230px", "padding": "20px"})
    ])

# Set app layout
app.layout = create_layout

# Sidebar toggle functionality
@app.callback(
    Output("sidebar", "className"),
    [Input("toggle-sidebar", "n_clicks")],
    [State("sidebar", "className")]
)
def toggle_sidebar(n, className):
    if n:
        if "active" in className:
            return "side-bar"
        else:
            return "side-bar active"
    return className

# Routing buttons to plots
@app.callback(
    Output("plots-container", "children"),
    [Input("link-home", "n_clicks"),
     Input("link-lc", "n_clicks"),
     Input("link-fi", "n_clicks"),
     Input("link-about", "n_clicks")]
)
def render_content(home, lc, fi, about):
    triggered = ctx.triggered_id

    if triggered == "link-home":
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=GraphBuilder().house_price_hist()), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=GraphBuilder().pca_plot()), width=12)
            ])
        ])

    elif triggered == "link-lc":
        return html.Div([
            html.H5("Training and Validation Learning Curves", className="text-center mb-4"),
            dbc.Tabs([
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_linear()), label="Linear Model"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_tree()), label="Decision Tree"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_forest()), label="Random Forest"),
                dbc.Tab(dcc.Graph(figure=GraphBuilder().learning_curve_gradient()), label="Gradient Boosting")
            ])
        ])

    elif triggered == "link-fi":
        return html.Div([
            html.H5("Feature Importance & Performance Analysis", className="text-center mb-4"),
            dbc.Tabs([
                dbc.Tab([
                    dcc.Graph(figure=GraphBuilder().feature_importance("linear")),
                    dcc.Graph(figure=GraphBuilder().scatter_plot("linear")),
                    dcc.Graph(figure=GraphBuilder().residual_plot())
                ], label="Linear"),
                dbc.Tab([
                    dcc.Graph(figure=GraphBuilder().feature_importance("tree")),
                    dcc.Graph(figure=GraphBuilder().scatter_plot("tree")),
                    dcc.Graph(figure=GraphBuilder().residual_tree_plot("tree")[1])
                ], label="Decision Tree"),
                dbc.Tab([
                    dcc.Graph(figure=GraphBuilder().feature_importance("forest")),
                    dcc.Graph(figure=GraphBuilder().scatter_plot("forest")),
                    dcc.Graph(figure=GraphBuilder().residual_tree_plot("forest")[1])
                ], label="Random Forest"),
                dbc.Tab([
                    dcc.Graph(figure=GraphBuilder().feature_importance("gradient")),
                    dcc.Graph(figure=GraphBuilder().scatter_plot("gradient")),
                    dcc.Graph(figure=GraphBuilder().residual_tree_plot("gradient")[1])
                ], label="Gradient Boosting")
            ])
        ])

    elif triggered == "link-about":
        return html.Div([
            html.H4("🏗️ About This Project", className="text-center"),
            html.Hr(),
            html.P(
                "This web application was developed as part of the "
                "Kaggle competition: 'House Prices - Advanced Regression Techniques'.",
                className="lead text-center"
            ),
            html.P(
                "The competition challenges participants to predict house sale prices "
                "based on various housing features such as quality, location, year built, and more.",
                className="text-center"
            ),
            html.P(
                "This app demonstrates an end-to-end data science workflow: "
                "data preprocessing, feature engineering, model training (Linear, Decision Tree, Random Forest, Gradient Boosting), "
                "and model evaluation using learning curves, feature importances, scatter plots, and residual distributions.",
                className="text-center"
            ),
            html.P(
                [
                    "🔗 View the original competition here: ",
                    html.A("Kaggle Competition Link",
                           href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques",
                           target="_blank", style={"fontWeight": "bold", "textDecoration": "underline"})
                ],
                className="text-center"
            ),
            html.P(
                [
                    "🔗 View the GitHub project source code: ",
                    html.A("GitHub Repository",
                           href="https://github.com/evansnjagi/Amos-App",
                           target="_blank", style={"fontWeight": "bold", "textDecoration": "underline"})
                ],
                className="text-center"
            ),
            html.P("The user interface is fully responsive and designed for mobile and desktop viewing.",
                   className="text-center")
        ])

    return html.Div("Select a tab to view its content.")

# Handle submission download
@app.callback(
    Output("download-component", "data"),
    [Input("download-button", "n_clicks")]
)
def download_submission(n):
    if not n:
        raise PreventUpdate
    df = MapId().get_id("linear")
    return dcc.send_data_frame(df.to_csv, filename="submission.csv")

if __name__ == "__main__":
    app.run_server(debug=True)
