import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from Business import GraphBuilder, MapId

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for Render

app.title = "Amos House App"

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Button("\u2630", className="btn btn-outline-secondary", id="sidebar-toggle", n_clicks=0),
            html.Div([
                html.H2("AmosApp", className="text-center my-3"),
                dbc.Button("House Price Histogram", id="hist-btn", color="primary", className="mb-2 w-100"),
                dbc.Button("PCA Plot", id="pca-btn", color="secondary", className="mb-2 w-100"),
                dbc.Button("Learning Curve", id="lc-btn", color="success", className="mb-2 w-100"),
                dbc.Button("Feature Importance", id="fi-btn", color="info", className="mb-2 w-100"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": "Linear", "value": "linear"},
                        {"label": "Tree", "value": "tree"},
                        {"label": "Forest", "value": "forest"},
                        {"label": "Gradient", "value": "gradient"},
                    ],
                    value="linear",
                    placeholder="Select model type...",
                    className="mb-2"
                ),
                dbc.Input(id="label-input", type="text", placeholder="Submission Label", className="mb-2"),
                dbc.Button("Generate Predictions", id="predict-btn", color="dark", className="mb-2 w-100"),
                dbc.Button("Download Submission", id="download-btn", color="danger", className="mb-2 w-100"),
                dcc.Download(id="download-link")
            ], id="sidebar", className="d-grid")
        ], width=12, md=3),

        dbc.Col([
            html.H4("Amos House Price Dashboard", className="text-center mt-3"),
            dcc.Loading(
                id="plot-loading",
                type="circle",
                children=html.Div(dcc.Graph(id="plot-area", config={"responsive": True}), className="p-2")
            ),
            dbc.Toast(
                "Prediction file saved. Ready to download!",
                id="toast",
                header="Success",
                is_open=False,
                dismissable=True,
                icon="success",
                duration=4000,
                style={"position": "fixed", "top": 10, "right": 10, "width": 350}
            )
        ], width=12, md=9)
    ])
], fluid=True)

# Store predictions temporarily
predicted_submissions = {}

@app.callback(
    Output("plot-area", "figure"),
    Input("hist-btn", "n_clicks"),
    Input("pca-btn", "n_clicks"),
    Input("lc-btn", "n_clicks"),
    Input("fi-btn", "n_clicks"),
    State("model-dropdown", "value")
)
def update_plot(n1, n2, n3, n4, model_type):
    button_id = ctx.triggered_id
    gb = GraphBuilder()
    if button_id == "hist-btn":
        return gb.house_price_hist()
    elif button_id == "pca-btn":
        return gb.pca_plot()
    elif button_id == "lc-btn":
        if model_type == "linear":
            return gb.learning_curve_linear()
        elif model_type == "tree":
            return gb.learning_curve_tree()
        elif model_type == "forest":
            return gb.learning_curve_forest()
        else:
            return gb.learning_curve_gradient()
    elif button_id == "fi-btn":
        return gb.feature_importance(model_type)
    return dash.no_update

@app.callback(
    Output("label-input", "value"),
    Output("toast", "is_open"),
    Input("predict-btn", "n_clicks"),
    State("label-input", "value"),
    State("model-dropdown", "value"),
    prevent_initial_call=True
)
def store_predictions(n_clicks, label, model_type):
    if not label:
        return "", False
    df = MapId().get_id(label)
    predicted_submissions[label] = df
    return label, True

@app.callback(
    Output("download-link", "data"),
    Input("download-btn", "n_clicks"),
    State("label-input", "value"),
    prevent_initial_call=True
)
def download_csv(n_clicks, label):
    if label in predicted_submissions:
        df = predicted_submissions[label]
        return dcc.send_data_frame(df.to_csv, f"{label}_submission.csv")
    return dash.no_update

if __name__ == "__main__":
    app.run_server(debug=True)
