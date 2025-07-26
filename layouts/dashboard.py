# energy_dash_app/layouts/dashboard.py

from dash import html, dcc, callback, Input, Output
import plotly.graph_objs as go
import pandas as pd
import json

# Dynamic callback that reacts to uploaded data
@callback(
    Output("dashboard-content", "children"),
    Input("uploaded-data-store", "data")
)
def render_dashboard(json_data):
    if json_data is None:
        return html.Div("📂 Upload a CSV file to see dashboard data.")

    df = pd.read_json(json_data, convert_dates=True)
    df.index = pd.to_datetime(df.index)

    return html.Div([
        generate_metrics(df),
        html.H3("Consumption Pattern"),
        plot_patterns(df)
    ])

# Generate metric cards (Avg, Max, Min Power)
def generate_metrics(df):
    return html.Div([
        html.Div([
            html.H4("Average Power (kW)"),
            html.H2(f"{df['Global_active_power'].mean():.2f}"),
        ], className="metric-card"),

        html.Div([
            html.H4("Max Power (kW)"),
            html.H2(f"{df['Global_active_power'].max():.2f}"),
        ], className="metric-card"),

        html.Div([
            html.H4("Min Power (kW)"),
            html.H2(f"{df['Global_active_power'].min():.2f}"),
        ], className="metric-card"),
    ], className="metrics-container")

# Plot hourly consumption pattern
def plot_patterns(df):
    df_hourly = df['Global_active_power'].resample('1H').mean().fillna(method='ffill')
    hourly_avg = df_hourly.groupby(df_hourly.index.hour).mean()

    bar = go.Bar(
        x=hourly_avg.index,
        y=hourly_avg.values,
        marker_color='skyblue',
        name="Hourly Avg"
    )

    layout = go.Layout(
        title="Hourly Average Energy Consumption",
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        xaxis=dict(title='Hour of Day'),
        yaxis=dict(title='Avg Power (kW)')
    )

    return dcc.Graph(figure=go.Figure(data=[bar], layout=layout))

# Static layout — placeholder for dynamic content
layout = html.Div([
    html.H1("📊 Dashboard", className="page-title"),
    html.Div(id="dashboard-content")  # <-- Populated dynamically
])
