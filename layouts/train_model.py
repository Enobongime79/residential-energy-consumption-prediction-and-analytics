# Train model layout
# energy_dash_app/layouts/train_model.py

from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

layout = html.Div([
    html.H1("🔧 Train Forecasting Model", className="page-title"),

    html.Div([
        html.Label("Sequence Length (hours)"),
        dcc.Slider(id='seq-len-slider', min=24, max=336, step=24, value=168,
                   marks={i: str(i) for i in range(24, 337, 48)}),
        html.Br(),

        html.Label("Prediction Horizon (hours)"),
        dcc.Slider(id='horizon-slider', min=1, max=48, step=1, value=24,
                   marks={i: str(i) for i in range(0, 49, 8)}),
        html.Br(),

        html.Label("Batch Size"),
        dcc.Dropdown(
            id='batch-size',
            options=[{'label': str(i), 'value': i} for i in [16, 32, 64, 128]],
            value=64,
            style={"color": "black"}
        ),
        html.Br(),

        html.Label("Epochs"),
        dcc.Slider(id='epochs-slider', min=10, max=200, step=10, value=50,
                   marks={i: str(i) for i in range(10, 201, 30)}),
        html.Br(),

        html.Button("🚀 Start Training", id="train-button", n_clicks=0),
        html.Div(id="training-status", style={"marginTop": "20px"}),

        html.Hr(),
        html.Div(id="training-results")
    ], style={"padding": "20px"})
])
