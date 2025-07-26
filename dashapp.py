# energy_dash_app/app.py

from dash import Dash, dcc, html, Input, Output, State
import pandas as pd
import base64
import io

from layouts import dashboard, train_model, predictions, anomaly, reports
from callbacks import training_callbacks  # Callback side-effects are auto-registered

# External CSS for beautiful dark UI
external_stylesheets = ["assets/custom.css"]

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server  # for deployment

app.title = "⚡ Energy Forecasting & Analytics"

# 🔧 MAIN LAYOUT
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            '📁 Drag and Drop or ',
            html.A('Select a CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'marginBottom': '20px',
            'color': 'white',
            'backgroundColor': '#1e1e1e'
        },
        multiple=False
    ),

    dcc.Store(id="uploaded-data-store", storage_type="memory"),

    dcc.Tabs(id="tabs", value='tab-dashboard', children=[
        dcc.Tab(label='📊 Dashboard', value='tab-dashboard'),
        dcc.Tab(label='🔧 Train Model', value='tab-train'),
        dcc.Tab(label='🔮 Predictions', value='tab-predict'),
        dcc.Tab(label='🚨 Anomaly Detection', value='tab-anomaly'),
        dcc.Tab(label='📋 Reports', value='tab-reports')
    ], className="tab-style"),

    html.Div(id='tab-content')
])

# 🔄 ROUTING CALLBACK
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-dashboard':
        return dashboard.layout
    elif tab == 'tab-train':
        return train_model.layout
    elif tab == 'tab-predict':
        return predictions.layout
    elif tab == 'tab-anomaly':
        return anomaly.layout
    elif tab == 'tab-reports':
        return reports.layout
    return html.Div("404 Tab not found")

# 📁 FILE UPLOAD HANDLER
@app.callback(
    Output('uploaded-data-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def store_uploaded_file(contents, filename):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), parse_dates=True, index_col='time')
        df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce').fillna(0)
        df = df[df['Global_active_power'] >= 0]
        return df.to_json(date_format='iso')
    except Exception as e:
        print("Error parsing CSV:", e)
        return None

if __name__ == "__main__":
    app.run(debug=True)
