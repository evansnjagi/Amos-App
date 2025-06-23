# importing libraries
import dash
from dash import dcc, html, Input, Output, State, ctx
import pandas as pd

from Business import GraphBuilder, ModelBuilder, MapId
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

# building the app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    # Download component for CSV file
    dcc.Download(id="download-submission"),

    # Toggle sidebar button
    html.Button("\u2630", className="button", id="sidebar-toggle"),

    # Sidebar layout
    html.Div([
        html.Label("Quick EDA for Training"),
        dcc.Dropdown(
            options=[
                {"label": "Independent Variable", "value": "SalePrice"},
                {"label": "Decomposed Features", "value": "pca"}
            ],
            value="SalePrice",
            id="eda-dropdown",
            placeholder="Select Variable",
            style={"margin-top": "10px", "margin-bottom": "40px"}
        ),
        html.Label("Model Selection"),
        dcc.Dropdown(
            options=[
                {"label": "Gradient-Boosting Model", "value": "GradientBoostingRegressor"},
                {"label": "Linear Regression Model", "value": "LinearRegression"},
                {"label": "Decision-Tree Model", "value": "DecisionTreeRegressor"},
                {"label": "Random-Forest Model", "value": "RandomForestRegressor"}
            ],
            value="GradientBoostingRegressor",
            id="model-dropdown",
            placeholder="Select Model",
            style={"margin-top": "10px", "margin-bottom": "40px"}
        ),
        html.Button("Train Model Here", n_clicks=0, className="train-model-button", id="train-model-button", style={"margin-bottom": "40px"}),
        html.Button("Learning Curve Plot", n_clicks=0, className="train-model-button", id="get-lc", style={"margin-bottom": "40px"}),
        html.Label("Evaluation Plots"),
        dcc.Dropdown(
            options=[
                {"label": "Scatter Plot", "value": "scatter plot"},
                {"label": "Residual Plot", "value": "residual plot"},
                {"label": "Feature Importances", "value": "feature importance"}
            ],
            id="model-plots",
            placeholder="scatter plot",
            style={"margin-top": "10px", "margin-bottom": "40px"}
        ),
        html.Button("Submission csv", n_clicks=0, className="train-model-button", id="submission-button", style={"margin-bottom": "10px"}),
        html.H4("\u00a9\ufe0f2025 kenthedatascientist", style={"color": "blue"})
    ], className="side-bar", id="side-bar"),

    # Main Content
    html.Div([
        html.H1("Amos House Prices Modelling", style={"color": "#856b6b", "textAlign": "center"}),
        html.Div(id="eda-displayer"),
        html.Div(id="model-display"),
        html.Div([html.Div(id="lc-display"), html.Div(id="eval-plot")]),
        html.Div(id="submission-display")
    ], id="main-content", style={"margin-left": "220px", "padding": "20px"})
])

# === Callbacks === #

# Sidebar Toggle Callback
@app.callback(
    Output("side-bar", "className"),
    Input("sidebar-toggle", "n_clicks"),
    prevent_initial_call=True
)
def toggle_sidebar(n_clicks):
    if n_clicks and n_clicks % 2 == 1:
        return "side-bar active"
    return "side-bar"

# Callback for EDA plots
@app.callback(
    Output("eda-displayer", "children"),
    Input("eda-dropdown", "value")
)
def get_graph(plot_type):
    fig = GraphBuilder().house_price_hist() if plot_type == "SalePrice" else GraphBuilder().pca_plot()
    return dcc.Graph(figure=fig)

# Train model callback
@app.callback(
    Output("model-display", "children"),
    State("model-dropdown", "value"),
    Input("train-model-button", "n_clicks")
)
def train_model(model_type, n_clicks):
    if n_clicks == 0:
        return html.Div()

    model_mapping = {
        "LinearRegression": ModelBuilder().linear_model,
        "DecisionTreeRegressor": ModelBuilder().tree_model,
        "RandomForestRegressor": ModelBuilder().forest_model,
        "GradientBoostingRegressor": ModelBuilder().gradient_model
    }

    if model_type in model_mapping:
        model, X_train, y_train = model_mapping[model_type]()
        pred = model.predict(X_train)
        mae = mean_absolute_error(y_train, pred)
        mape = mean_absolute_percentage_error(y_train, pred)
        cod = r2_score(y_train, pred)

        return html.Div([
            html.H4(f"{model_type} Model Evaluation Metrics", style={"textAlign": "center"}),
            html.Table([
                html.Tr([html.Th("Metric"), html.Th("Value")]),
                html.Tr([html.Td("MAE (Mean Absolute Error)"), html.Td(f"${mae:,.2f}")]),
                html.Tr([html.Td("MAPE (Mean Absolute % Error)"), html.Td(f"{mape:.2%}")]),
                html.Tr([html.Td("R² Score"), html.Td(f"{cod:.4f}")])
            ], className="table")
        ], className="metrics")

    return html.Div(html.H3("Selected model not implemented yet!"))

# Learning curve callback
@app.callback(
    Output("lc-display", "children"),
    State("model-dropdown", "value"),
    Input("get-lc", "n_clicks")
)
def learning_curve(model_type, n_clicks):
    if n_clicks == 0:
        return html.Div()

    curve_funcs = {
        "LinearRegression": GraphBuilder().learning_curve_linear,
        "DecisionTreeRegressor": GraphBuilder().learning_curve_tree,
        "RandomForestRegressor": GraphBuilder().learning_curve_forest,
        "GradientBoostingRegressor": GraphBuilder().learning_curve_gradient
    }

    if model_type in curve_funcs:
        fig = curve_funcs[model_type]()
        return dcc.Graph(figure=fig)
    return html.Div()

# Evaluation plot callback
@app.callback(
    Output("eval-plot", "children"),
    State("model-dropdown", "value"),
    Input("model-plots", "value"),
    Input("train-model-button", "n_clicks")
)
def eval_display(model_type, plot_type, n_clicks):
    if n_clicks == 0 or model_type is None or plot_type is None:
        return html.Div()

    if plot_type == "scatter plot":
        fig = GraphBuilder().scatter_plot(plot_type=model_type.split("Regressor")[0].lower())
        return dcc.Graph(figure=fig)
    elif plot_type == "residual plot":
        if model_type == "LinearRegression":
            fig = GraphBuilder().residual_plot()
            return dcc.Graph(figure=fig)
        else:
            text, fig = GraphBuilder().residual_tree_plot(model_type=model_type.split("Regressor")[0].lower())
            return html.Div([
                html.Pre(text, className="text-table", style={"margin-top": "40px"}),
                dcc.Graph(figure=fig)
            ])
    elif plot_type == "feature importance":
        fig = GraphBuilder().feature_importance(model_type=model_type.split("Regressor")[0].lower())
        return dcc.Graph(figure=fig)

    return html.Div()

# Submission & CSV download
@app.callback(
    Output("submission-display", "children"),
    Output("download-submission", "data"),
    Input("model-dropdown", "value"),
    Input("submission-button", "n_clicks"),
    prevent_initial_call=True
)
def idmapping_display(model_type, n_clicks):
    if n_clicks == 0:
        return html.Div(), None

    label_map = {
        "LinearRegression": "LinearRegressionModel",
        "DecisionTreeRegressor": "DecisionTreeModel",
        "RandomForestRegressor": "RandomForestModel",
        "GradientBoostingRegressor": "GradientBoostingModel"
    }

    label = label_map.get(model_type)
    if not label:
        return html.Div(), None

    ids = MapId().get_id(label)

    # File for download
    csv_string = ids.to_csv(index=False)
    download_data = dict(content=csv_string, filename=f"{label}_submission.csv")

    info_points = [
        "The dataset used in this project was obtained from a Kaggle competition",
        "The competition is about advanced regression techniques",
        "The dataset has house/property features with a target variable: SalePrice",
        "The dataset is about Amos house prices, which we understand is a modified Kaggle dataset",
        "For the success of this project, the dataset (CSV) is embedded in our training files",
        "These four models are trained in real time. Please be patient, especially for tree-based models",
        "PCA was used to decompose the training feature matrix which had 80 features",
        "The learning curve takes ~2 minutes as it performs 5-fold cross-validation",
        "After selecting a model, click the learning curve button again to collapse any previous model plot",
        "This shallow AI model is a great start. We aim to enhance it in the future by integrating a MongoDB database"
    ]

    table_rows = [html.Tr([html.Th("Point"), html.Th("Description")])]
    for i, point in enumerate(info_points, 1):
        table_rows.append(html.Tr([html.Td(f"{i}.", style={"vertical-align": "top"}), html.Td(point)]))

    display_div = html.Div([
        html.H4(f"The IDs are mapped for {model_type}. Download will begin.", style={"textAlign": "center"}),
        html.H4("Note the following about this project:", style={"textAlign": "center", "marginTop": "20px", "marginBottom": "20px"}),
        html.Table(table_rows, className="text-table"),
        html.H3("Thank You! 💕")
    ], className="metrics")

    return display_div, download_data

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
