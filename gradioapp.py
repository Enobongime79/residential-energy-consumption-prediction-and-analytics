import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
import base64
import traceback
import pickle # For serializing/deserializing the predictor

# Assuming model_utils.py is in the same directory and contains the EnergyPredictor class.
# If not, you will need to ensure it's accessible or copy its content here.
from model_utils import EnergyPredictor

# --- Helper Functions (adapted for Gradio) ---

def create_download_link_gradio(df: pd.DataFrame, filename: str, link_text: str):
    """Generates a base64 encoded link for downloading a dataframe as CSV for Gradio."""
    if df is None or df.empty:
        return "" # Return empty string if no data to download
    csv_string = df.to_csv(index=True)
    b64_encoded = base64.b64encode(csv_string.encode()).decode()
    return f'<a href="data:text/csv;base64,{b64_encoded}" download="{filename}">{link_text}</a>'

def validate_dataframe(df):
    """Validate the uploaded dataframe meets minimum requirements and preprocesses it."""
    if df is None or df.empty:
        return None, "DataFrame is empty"
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None, "DataFrame index must be convertible to datetime."

    # Ensure required columns exist
    if 'Global_active_power' not in df.columns:
        return None, "Required column 'Global_active_power' not found."
    
    if len(df) < 24:  # Minimum 24 hours of data
        return None, "Insufficient data: Need at least 24 hours of data for meaningful analysis."
    
    # Convert all columns to numeric, coercing errors and filling NaNs
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True) # Fallback to mean if still NaNs

    # Handle negative values in 'Global_active_power'
    if df['Global_active_power'].isnull().any():
        return None, "Error: 'Global_active_power' column contains null values after processing."
    if (df['Global_active_power'] < 0).any():
        df.loc[df['Global_active_power'] < 0, 'Global_active_power'] = 0
        return df, "Warning: Negative 'Global_active_power' values were set to 0. Data validation successful."
            
    return df, "Data validation successful."

# --- Plotting Functions (ported directly from Streamlit app.py, with dark theme enforcement) ---

def plot_training_history(history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model MAE'))

    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss', legendgroup='loss'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss', legendgroup='loss'), row=1, col=1)

    fig.add_trace(go.Scatter(y=history.history['mean_absolute_error'], mode='lines', name='Training MAE', legendgroup='mae'), row=1, col=2)
    fig.add_trace(go.Scatter(y=history.history['val_mean_absolute_error'], mode='lines', name='Validation MAE', legendgroup='mae'), row=1, col=2)

    fig.update_layout(height=400, showlegend=True, title_text="Training History",
                      plot_bgcolor='#23272f', # Dark background for plots
                      paper_bgcolor='#23272f', # Dark background for the paper area
                      font_color='#f8f8f8') # White font color for titles, labels, etc.
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    return fig

def plot_predictions(actual, predicted, dates=None, title="Actual vs Predicted"):
    fig = go.Figure()
    if dates is not None:
        fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual Power', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted Power', line=dict(color='red', dash='dot')))
        fig.update_xaxes(title_text="Date")
    else:
        fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual Power', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted Power', line=dict(color='red', dash='dot')))
    
    fig.update_layout(title_text=title, yaxis_title="Global Active Power (kW)", height=400,
                      plot_bgcolor='#23272f',  # Dark background for plots
                      paper_bgcolor='#23272f', # Dark background for the paper area
                      font_color='#f8f8f8')     # White font color for titles, labels, etc.
    return fig

def plot_feature_importance(df):
    if 'Global_active_power' in df.columns:
        numeric_df = df.select_dtypes(include=np.number)
        correlations = numeric_df.corr()['Global_active_power'].sort_values(ascending=False)
        correlations = correlations[correlations.index != 'Global_active_power']
        
        if not correlations.empty:
            fig = px.bar(correlations, 
                         x=correlations.index, 
                         y=correlations.values, 
                         title='Feature Correlation with Global Active Power',
                         labels={'x': 'Feature', 'y': 'Correlation'},
                         color_discrete_sequence=px.colors.qualitative.Plotly)
            fig.update_layout(xaxis_tickangle=-45, height=450,
                              plot_bgcolor='#23272f',  # Dark background for plots
                              paper_bgcolor='#23272f', # Dark background for the paper area
                              font_color='#f8f8f8')     # White font color for titles, labels, etc.
            return fig
        else:
            return None # No numeric features to correlate
    return None

def plot_consumption_patterns(df):
    if 'Global_active_power' not in df.columns:
        return None

    # Ensure resampling can occur (index must be DatetimeIndex)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None # Cannot plot if index is not datetime

    df_hourly = df['Global_active_power'].resample('1h').mean().fillna(method='ffill').fillna(method='bfill')

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Hourly Average Consumption', 'Daily Average Consumption',
            'Monthly Average Consumption', 'Sub-meter Consumption Breakdown'
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "domain"}]],
        vertical_spacing=0.18,  # more space between rows
        horizontal_spacing=0.12 # more space between columns
    )

    # Hourly Average
    hourly_avg = df_hourly.groupby(df_hourly.index.hour).mean()
    fig.add_trace(go.Bar(
        x=hourly_avg.index, y=hourly_avg.values, name='Hourly',
        marker_color='#4e79a7', showlegend=False
    ), row=1, col=1)
    fig.update_xaxes(title_text="Hour", row=1, col=1, tickmode='array', tickvals=list(range(0,24,3)))
    fig.update_yaxes(title_text="Avg Power (kW)", row=1, col=1)

    # Daily Average
    daily_avg = df_hourly.groupby(df_hourly.index.dayofweek).mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig.add_trace(go.Bar(
        x=[day_names[i] for i in daily_avg.index], y=daily_avg.values, name='Daily',
        marker_color='#f28e2b', showlegend=False
    ), row=1, col=2)
    fig.update_xaxes(title_text="Day", row=1, col=2)
    fig.update_yaxes(title_text="Avg Power (kW)", row=1, col=2)

    # Monthly Average
    monthly_avg = df_hourly.groupby(df_hourly.index.month).mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.add_trace(go.Bar(
        x=[month_names[i-1] for i in monthly_avg.index], y=monthly_avg.values, name='Monthly',
        marker_color='#e15759', showlegend=False
    ), row=2, col=1)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Avg Power (kW)", row=2, col=1)

    # Sub-meter Breakdown
    sub_meter_cols = [col for col in df.columns if 'Sub_metering_' in col]
    if sub_meter_cols and 'Global_active_power' in df.columns:
        df['Other_consumption'] = df['Global_active_power'] - df[sub_meter_cols].sum(axis=1)
        df['Other_consumption'] = df['Other_consumption'].clip(lower=0)
        pie_data = df[sub_meter_cols + ['Other_consumption']].sum()
        pie_data = pie_data[pie_data > 0]
        if not pie_data.empty:
            fig.add_trace(go.Pie(
                labels=pie_data.index, values=pie_data.values, name='Sub-meters',
                showlegend=False,
                textinfo='percent',
                pull=[0.05]*len(pie_data),
                marker=dict(colors=px.colors.qualitative.Pastel),
                textfont=dict(size=16, color='#23272f')
            ), row=2, col=2)
    else:
        fig.add_annotation(
            text="No sub-meter data available",
            xref="paper", yref="paper", x=0.75, y=0.25, showarrow=False,
            font=dict(color="#f8f8f8")
        )

    fig.update_layout(
        height=700, width=1000,
        title_text="Energy Consumption Patterns",
        plot_bgcolor='#23272f',
        paper_bgcolor='#23272f',
        font_color='#f8f8f8',
        margin=dict(t=60, l=50, r=50, b=50)
    )
    return fig

def plot_anomalies(df, anomalies, scores):
    fig = go.Figure()

    # Original data
    fig.add_trace(go.Scatter(x=df.index, y=df['Global_active_power'], mode='lines', name='Global Active Power', line=dict(color='blue')))

    # Anomalies
    anomaly_dates = df.index[anomalies]
    anomaly_values = df['Global_active_power'][anomalies]
    anomaly_hover_text = [f"Date: {d.strftime('%Y-%m-%d %H:%M')}<br>Power: {p:.2f}<br>Anomaly Score: {s:.2f}" 
                          for d, p, s in zip(anomaly_dates, anomaly_values, scores[anomalies])]

    if not anomaly_dates.empty:
        fig.add_trace(go.Scatter(x=anomaly_dates, y=anomaly_values, mode='markers', name='Anomaly',
                                 marker=dict(color='red', size=8, symbol='x'),
                                 hovertext=anomaly_hover_text, hoverinfo='text'))
    
    fig.update_layout(title_text='Anomaly Detection in Global Active Power',
                      xaxis_title='Date',
                      yaxis_title='Global Active Power (kW)',
                      showlegend=True,
                      height=500,
                      plot_bgcolor='#23272f',
                      paper_bgcolor='#23272f',
                      font_color='#f8f8f8'
                      )
    return fig

# --- Gradio UI Logic ---

# Initial empty states for Gradio's gr.State components
# DataFrames are stored as JSON strings in gr.State to avoid serialization issues with complex objects
initial_df_json = pd.DataFrame().to_json(date_format='iso', orient='split')
initial_predictor_serialized = None # Will store base64 encoded pickle of predictor
initial_training_status = False
initial_evaluation_results_json = json.dumps({})

# Custom CSS for Gradio theme and elements
gradio_css = """
body {
    background-color: #1a1a1a; /* Dark background for the whole app */
    color: #f8f8f8; /* Light text color */
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    background-color: #1a1a1a;
}
.gr-text-input {
    background-color: #2e2e2e !important;
    color: #f8f8f8 !important;
    border-color: #444 !important;
}
.gr-button {
    background-color: #2F80ED !important;
    color: white !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    border: none !important;
    transition: background-color 0.3s ease;
}
.gr-button:hover {
    background-color: #236bc4 !important;
}
.gr-box {
    background-color: #000000; /* Black background for sections */
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    padding: 20px;
    margin-bottom: 20px;
}
h1, h2, h3, h4 {
    color: #2F80ED; /* Primary color for headers */
    text-align: center;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    margin-bottom: 15px;
}
.main-header {
    font-size: 3.5em;
    margin-bottom: 30px;
}
.metric-box {
    background-color: #000000;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    text-align: center;
    color: #FFFFFF;
    flex: 1; /* For columns */
    margin: 10px;
}
.metric-box h4 {
    color: #FFFFFF;
    font-size: 1.2em;
    margin-bottom: 5px;
}
.metric-box h2 {
    color: #2F80ED; /* Use primary color for metric values */
    font-size: 2.5em;
    margin-top: 0;
}
.insight-box {
    background-color: #000000;
    border-left: 5px solid #2F80ED;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 5px;
    font-size: 1.1em;
    color: #FFFFFF;
}
.gr-tab-item {
    color: #CCCCCC !important;
}
.gr-tab-item.selected {
    color: #2F80ED !important;
    border-bottom: 2px solid #2F80ED !important;
}
.gr-dataframe {
    background-color: #000000 !important;
    color: #f8f8f8 !important;
    border: 1px solid #333 !important;
}
.gr-dataframe thead th {
    background-color: #23272f !important;
    color: white !important;
}
.gr-dataframe tbody td {
    background-color: #000000 !important;
    color: white !important;
}
.gr-plot {
    border-radius: 10px;
    overflow: hidden; /* Ensures plot background respects border-radius */
}
.file_upload_input {
    background-color: #23272f !important;
    color: #f8f8f8 !important;
    border: 1px dashed #444 !important;
    border-radius: 5px !important;
    text-align: center !important;
    padding: 20px !important;
    cursor: pointer !important;
}
.file_upload_input p {
    color: #f8f8f8 !important;
}
.file_upload_input a {
    color: #2F80ED !important;
    text-decoration: underline !important;
}
.gr-slider-input, .gr-dropdown-input {
    background-color: #23272f !important;
    color: #f8f8f8 !important;
    border-color: #444 !important;
}
/* Style for dropdown options, if needed to match theme */
.gr-dropdown-input .dropdown-select .options {
    background-color: #2e2e2e !important;
}
.gr-dropdown-input .dropdown-select .options .option {
    color: #f8f8f8 !important;
}
.gr-dropdown-input .dropdown-select .options .option.selected {
    background-color: #2F80ED !important;
    color: black !important;
}
.error-message {
    background-color: #4d0000; /* Darker red background */
    border-left: 5px solid #ff4d4d; /* Brighter red border */
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 5px;
    font-size: 1.1em;
    color: #ffe0e0; /* Lighter red text */
}
.success-message {
    background-color: #004d00; /* Darker green background */
    border-left: 5px solid #4dff4d; /* Brighter green border */
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 5px;
    font-size: 1.1em;
    color: #e0ffe0; /* Lighter green text */
}
.warning-message {
    background-color: #4d4d00; /* Darker yellow background */
    border-left: 5px solid #ffff4d; /* Brighter yellow border */
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 5px;
    font-size: 1.1em;
    color: #ffffe0; /* Lighter yellow text */
}

"""

with gr.Blocks(css=gradio_css, title="⚡ Energy Consumption Forecasting & Analytics") as demo:
    # State variables to persist data and model across interactions
    # DataFrames are serialized to JSON strings when stored in gr.State
    processed_df_state = gr.State(value=initial_df_json)
    predictor_state = gr.State(value=initial_predictor_serialized)
    training_status_state = gr.State(value=initial_training_status)
    evaluation_results_state = gr.State(value=initial_evaluation_results_json)
    
    gr.HTML('<h1 class="main-header">⚡ Energy Consumption Forecasting & Analytics</h1>')

    with gr.Row():
        with gr.Column(scale=1): # Sidebar column
            gr.Markdown("## 🛠️ Model Parameters")
            seq_len_slider = gr.Slider(minimum=24, maximum=336, step=12, value=168, label="Sequence Length (hours)", info="How many past hours the model observes")
            future_period_predict_slider = gr.Slider(minimum=1, maximum=48, step=1, value=24, label="Prediction Horizon (hours)", info="How far into the future the model predicts")
            batch_size_dropdown = gr.Dropdown(choices=[16, 32, 64, 128], value=64, label="Batch Size", allow_custom_value=False)
            epochs_slider = gr.Slider(minimum=10, maximum=200, step=10, value=50, label="Epochs")

            gr.Markdown("## 📁 Data Upload")
            file_upload = gr.File(label="Upload CSV file", file_types=[".csv"])
            upload_message_md = gr.Markdown()
            sidebar_dataset_info_md = gr.Markdown()

            # --- Data Processing Function ---
            def process_uploaded_file(file):
                """
                Handles file upload, reads CSV, validates, and preprocesses data.
                Returns JSON string of processed DataFrame, and messages for UI.
                """
                if file is None:
                    return initial_df_json, gr.Markdown('<div class="error-message">Please upload a CSV file.</div>'), ""

                try:
                    # Read the CSV file from the Gradio File object's path
                    df = pd.read_csv(file.name, sep=',', index_col='time', parse_dates=True)
                except Exception as e:
                    return initial_df_json, gr.Markdown(f'<div class="error-message">Error reading CSV file: {str(e)}<br>Please ensure your CSV is comma-separated, has a "time" column as index, and "Global_active_power" column.</div>'), ""

                # Validate and preprocess the dataframe
                processed_df, validation_msg = validate_dataframe(df)

                if processed_df is None: # Validation failed
                    return initial_df_json, gr.Markdown(f'<div class="error-message">❌ {validation_msg}</div>'), ""
                
                # Prepare dataset info for sidebar
                dataset_info = f"**Rows:** {processed_df.shape[0]}<br>**Columns:** {processed_df.shape[1]}"
                if isinstance(processed_df.index, pd.DatetimeIndex) and not processed_df.empty:
                    dataset_info += f"<br>**Date Range:** {processed_df.index.min().strftime('%Y-%m-%d %H:%M:%S')} to {processed_df.index.max().strftime('%Y-%m-%d %H:%M:%S')}"
                dataset_info += f"<br>---<br>**Columns:** {', '.join(processed_df.columns.tolist())}"

                # Return processed df as JSON and success message
                return processed_df.to_json(date_format='iso', orient='split'), \
                       gr.Markdown(f'<div class="success-message">CSV file uploaded and processed successfully! {validation_msg if "Warning" in validation_msg else ""}</div>'), \
                       gr.Markdown(dataset_info)
            
            # Connect the file_upload component to the processing function
            file_upload.upload(
                process_uploaded_file,
                inputs=[file_upload],
                outputs=[processed_df_state, upload_message_md, sidebar_dataset_info_md]
            )

        with gr.Column(scale=3): # Main content column
            with gr.Tabs() as main_tabs:
                # --- Dashboard Tab ---
                with gr.TabItem("📊 Dashboard", id="dashboard_tab"):
                    gr.Markdown('<h1 class="main-header">📊 Energy Consumption Dashboard</h1>')
                    gr.Markdown("Explore your energy data at a glance. View key metrics, trends, and patterns to understand your consumption habits.")
                    
                    with gr.Row():
                        avg_power_html = gr.HTML('<div class="metric-box"><h4>Average Power (kW)</h4><h2>--</h2></div>')
                        max_power_html = gr.HTML('<div class="metric-box"><h4>Max Power (kW)</h4><h2>--</h2></div>')
                        min_power_html = gr.HTML('<div class="metric-box"><h4>Min Power (kW)</h4><h2>--</h2></div>')
                    
                    gr.Markdown("### Dataset Overview")
                    dataset_info_dashboard = gr.Markdown("No data uploaded.")
                    gr.Markdown("### Sample Data")
                    sample_data_df = gr.DataFrame(headers=[], datatype="str", col_count=(1, "dynamic"), row_count=(0, "dynamic"),
                                                  wrap=True) # wrap text in cells
                    gr.Markdown("### Consumption Patterns")
                    consumption_patterns_plot = gr.Plot()

                    def update_dashboard_ui(json_df):
                        """Updates dashboard UI elements based on the processed DataFrame."""
                        if json_df == initial_df_json:
                            return [
                                gr.HTML('<div class="metric-box"><h4>Average Power (kW)</h4><h2>--</h2></div>'),
                                gr.HTML('<div class="metric-box"><h4>Max Power (kW)</h4><h2>--</h2></div>'),
                                gr.HTML('<div class="metric-box"><h4>Min Power (kW)</h4><h2>--</h2></div>'),
                                "No data uploaded.",
                                pd.DataFrame(),
                                None
                            ]
                        
                        df = pd.read_json(json_df, orient='split')
                        if 'Global_active_power' not in df.columns:
                            return [
                                gr.HTML('<div class="metric-box"><h4>Average Power (kW)</h4><h2>--</h2></div>'),
                                gr.HTML('<div class="metric-box"><h4>Max Power (kW)</h4><h2>--</h2></div>'),
                                gr.HTML('<div class="metric-box"><h4>Min Power (kW)</h4><h2>--</h2></div>'),
                                "Missing 'Global_active_power' column.",
                                pd.DataFrame(),
                                None
                            ]
                        
                        avg_power = f"{df['Global_active_power'].mean():.2f}"
                        max_power = f"{df['Global_active_power'].max():.2f}"
                        min_power = f"{df['Global_active_power'].min():.2f}"

                        # Format DataFrame dtypes for display
                        dtypes_df = df.dtypes.rename('DataType').reset_index().rename(columns={'index': 'Column Name'})
                        dtypes_markdown = dtypes_df.to_markdown(index=False)

                        dashboard_info = f"**Number of entries:** {len(df)}<br>**Data Frequency:** Assumed Hourly<br>**Missing Values (after processing):** {df.isnull().sum().sum()} (should be 0)<br>**Columns and Data Types:**<br>{dtypes_markdown}"
                        
                        sample_data = df.head().reset_index() # Reset index to show 'time' as a column in Gradio DataFrame
                        
                        fig = plot_consumption_patterns(df)
                        
                        return [
                            gr.HTML(f'<div class="metric-box"><h4>Average Power (kW)</h4><h2>{avg_power}</h2></div>'),
                            gr.HTML(f'<div class="metric-box"><h4>Max Power (kW)</h4><h2>{max_power}</h2></div>'),
                            gr.HTML(f'<div class="metric-box"><h4>Min Power (kW)</h4><h2>{min_power}</h2></div>'),
                            gr.Markdown(dashboard_info),
                            sample_data,
                            fig
                        ]

                    # Callback for Dashboard tab updates when processed_df_state changes
                    processed_df_state.change(
                        update_dashboard_ui,
                        inputs=[processed_df_state],
                        outputs=[avg_power_html, max_power_html, min_power_html, dataset_info_dashboard, sample_data_df, consumption_patterns_plot]
                    )


                # --- Model Training Tab ---
                with gr.TabItem("🔧 Model Training", id="training_tab"):
                    gr.Markdown('<h1 class="main-header">🔧 Train Forecasting Model</h1>')
                    gr.Markdown("Train a forecasting model on your uploaded data. Adjust parameters and start training to build a custom energy predictor.")

                    gr.Markdown("### Model Training Configuration")
                    config_md = gr.Markdown("---") # Placeholder to show current slider config
                    
                    gr.Markdown("### Feature Correlation Overview")
                    feature_importance_plot = gr.Plot()

                    gr.Markdown("---")
                    
                    with gr.Row():
                        with gr.Column():
                            train_button = gr.Button("🚀 Start Training Model")
                            training_progress = gr.Progress(label="Training Progress", visible=False)
                            training_status_message = gr.Markdown()
                            training_output_message = gr.Markdown()
                            training_error_message = gr.Markdown()
                            
                            gr.Markdown("### Training Metrics")
                            training_metrics_output = gr.JSON()
                            validation_metrics_output = gr.JSON()
                            
                            gr.Markdown("### Training History Plots")
                            training_history_plot = gr.Plot()

                        with gr.Column():
                            load_model_button = gr.Button("📂 Load Saved Model")
                            loading_progress = gr.Progress(label="Loading Model Progress", visible=False)
                            loading_status_message = gr.Markdown()
                            loading_error_message = gr.Markdown()
                            gr.Markdown("### Loaded Model Evaluation (on current data)")
                            loaded_train_metrics_output = gr.JSON()
                            loaded_validation_metrics_output = gr.JSON()

                    # Initial update of training config text
                    @gr.on(
                        [
                            seq_len_slider.change,
                            future_period_predict_slider.change,
                            batch_size_dropdown.change,
                            epochs_slider.change
                        ],
                        inputs=[seq_len_slider, future_period_predict_slider, batch_size_dropdown, epochs_slider],
                        outputs=config_md
                    )
                    def update_training_config(seq_len, future_period_predict, batch_size, epochs):
                        return gr.Markdown(f"""
                        - **Sequence Length:** {seq_len} hours (how many past hours the model observes)
                        - **Prediction Horizon:** {future_period_predict} hours (how far into the future the model predicts)
                        - **Batch Size:** {batch_size}
                        - **Epochs:** {epochs}
                        """)
                    
                    # Function to run model training
                    def run_training(json_df, seq_len, future_period_predict, batch_size, epochs, progress=gr.Progress()):
                        if json_df == initial_df_json:
                            return False, initial_predictor_serialized, initial_evaluation_results_json, None, None, \
                                   "", "", gr.Markdown('<div class="error-message">Please upload your energy consumption data first.</div>'), \
                                   gr.Progress(0), "0%"
                        
                        df = pd.read_json(json_df, orient='split')
                        if 'Global_active_power' not in df.columns:
                            return False, initial_predictor_serialized, initial_evaluation_results_json, None, None, \
                                   "", "", gr.Markdown('<div class="error-message">Cannot train model: "Global_active_power" column not found.</div>'), \
                                   gr.Progress(0), "0%"

                        try:
                            progress(0.05, desc="Initializing model...")
                            predictor = EnergyPredictor(
                                seq_len=seq_len,
                                future_period_predict=future_period_predict,
                                batch_size=batch_size,
                                epochs=epochs
                            )
                            progress(0.1, desc="Preprocessing data...")
                            df_processed = predictor.preprocess_data(df.copy())
                            progress(0.3, desc="Preparing training sequences...")
                            # The prepare_data method is called internally by predictor.train in current model_utils.py
                            # No need for explicit train_x, train_y, etc. here unless model_utils.py changes
                            progress(0.5, desc="Training model... This may take a while.")
                            history = predictor.train(df_processed) # Assuming train() handles sequences internally
                            progress(0.8, desc="Evaluating model performance...")
                            evaluation_results = predictor.evaluate_model(df.copy())
                            
                            progress(0.9, desc="Saving model...")
                            os.makedirs('models', exist_ok=True) # Ensure models directory exists
                            predictor.save_complete_model("models/energy_model")
                            
                            progress(1.0, desc="Training completed and model saved!")

                            # Serialize predictor to be stored in gr.State
                            predictor_serialized = base64.b64encode(pickle.dumps(predictor)).decode('utf-8')
                            
                            feature_corr_fig = plot_feature_importance(df)
                            if feature_corr_fig is None:
                                feature_corr_fig = go.Figure(layout=go.Layout(title="No numeric features for correlation"))

                            return True, predictor_serialized, json.dumps(evaluation_results), \
                                   plot_training_history(history), feature_corr_fig, \
                                   gr.Markdown('<div class="success-message">Model training completed and saved successfully!</div>'), \
                                   gr.Markdown(json.dumps(evaluation_results['train_metrics'], indent=2)), \
                                   gr.Markdown(json.dumps(evaluation_results['validation_metrics'], indent=2)), \
                                   gr.Progress(1.0), "100%"
                            
                        except Exception as e:
                            import traceback
                            return False, initial_predictor_serialized, initial_evaluation_results_json, None, None, \
                                   "", gr.Markdown(f'<div class="error-message">An error occurred during training: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>'), \
                                   {}, {}, \
                                   gr.Progress(0.0), "0%"

                    train_button.click(
                        run_training,
                        inputs=[
                            processed_df_state,
                            seq_len_slider,
                            future_period_predict_slider,
                            batch_size_dropdown,
                            epochs_slider
                        ],
                        outputs=[
                            training_status_state,
                            predictor_state,
                            evaluation_results_state,
                            training_history_plot,
                            feature_importance_plot,
                            training_status_message,
                            training_output_message,
                            validation_metrics_output, # Streamlit had validation_metrics_output, let's map this to the second JSON output
                            training_progress,
                            training_progress # Gradio progress updates both value and label
                        ]
                    )

                    # Function to load saved model
                    def load_model(json_df, progress=gr.Progress()):
                        progress(0.2, desc="Checking model directory...")
                        os.makedirs('models', exist_ok=True) # Ensure this directory exists
                        
                        progress(0.4, desc="Initializing predictor...")
                        predictor = EnergyPredictor()
                        
                        try:
                            progress(0.6, desc="Loading model files...")
                            predictor.load_complete_model("models/energy_model")
                            predictor_serialized = base64.b64encode(pickle.dumps(predictor)).decode('utf-8')
                            
                            evaluation_results_dict = {}
                            if json_df != initial_df_json: # Only evaluate if data is uploaded
                                progress(0.8, desc="Evaluating loaded model...")
                                df = pd.read_json(json_df, orient='split')
                                evaluation_results_dict = predictor.evaluate_model(df.copy())
                            else:
                                evaluation_results_dict['message'] = "Upload data to evaluate the loaded model on your dataset."
                            
                            progress(1.0, desc="Model loaded successfully!")

                            return True, predictor_serialized, json.dumps(evaluation_results_dict), \
                                   gr.Markdown('<div class="success-message">Model loaded successfully!</div>'), "", \
                                   evaluation_results_dict.get('train_metrics', {}), \
                                   evaluation_results_dict.get('validation_metrics', {})
                            
                        except FileNotFoundError:
                            return False, initial_predictor_serialized, initial_evaluation_results_json, \
                                   "", gr.Markdown('<div class="warning-message">No saved model found. Please train a model first or check the model path ("models/energy_model").</div>'), \
                                   {}, {}
                        except Exception as e:
                            import traceback
                            return False, initial_predictor_serialized, initial_evaluation_results_json, \
                                   "", gr.Markdown(f'<div class="error-message">An error occurred while loading the model: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>'), \
                                   {}, {}
                    
                    load_model_button.click(
                        load_model,
                        inputs=[processed_df_state],
                        outputs=[
                            training_status_state,
                            predictor_state,
                            evaluation_results_state,
                            loading_status_message,
                            loading_error_message,
                            loaded_train_metrics_output,
                            loaded_validation_metrics_output
                        ]
                    )
                    
                    # Update feature importance plot when processed_df_state changes or tab is selected
                    # The feature importance is based on the data, not model training history
                    @gr.on(
                        [
                            main_tabs.select, # When tab is selected
                            processed_df_state.change # When new data is uploaded
                        ],
                        inputs=[processed_df_state],
                        outputs=[feature_importance_plot]
                    )
                    def update_feature_importance_on_data_change(json_df):
                        if json_df == initial_df_json:
                            return None
                        df = pd.read_json(json_df, orient='split')
                        fig = plot_feature_importance(df)
                        if fig is None:
                            return go.Figure(layout=go.Layout(title="No numeric features for correlation or data not suitable."))
                        return fig


                # --- Predictions Tab ---
                with gr.TabItem("🔮 Predictions", id="predictions_tab"):
                    gr.Markdown('<h1 class="main-header">🔮 Future Energy Predictions</h1>')
                    gr.Markdown("Generate future energy consumption forecasts using your trained model. Visualize and download predictions for planning.")
                    
                    prediction_hours_slider = gr.Slider(minimum=1, maximum=168, step=1, value=24, label="Hours to forecast into the future")
                    generate_predictions_button = gr.Button("📈 Generate Predictions")
                    prediction_status_message = gr.Markdown()
                    predictions_plot = gr.Plot()
                    predicted_data_table = gr.DataFrame(headers=[], datatype="str", col_count=(1, "dynamic"), row_count=(0, "dynamic"), wrap=True)
                    prediction_download_link = gr.HTML()
                    prediction_insights_md = gr.Markdown()

                    def generate_predictions_gradio(json_df, serialized_predictor, training_status, prediction_hours, progress=gr.Progress()):
                        if not training_status or serialized_predictor is None:
                            return None, pd.DataFrame(), gr.HTML(""), gr.Markdown('<div class="warning-message">Please train or load a model first on the "Model Training" page.</div>'), gr.Markdown("")
                        if json_df == initial_df_json:
                            return None, pd.DataFrame(), gr.HTML(""), gr.Markdown('<div class="error-message">No data uploaded for predictions. Please upload data via the sidebar.</div>'), gr.Markdown("")
                        
                        df = pd.read_json(json_df, orient='split')
                        
                        try:
                            progress(0.1, desc="Deserializing model...")
                            predictor = pickle.loads(base64.b64decode(serialized_predictor))
                            
                            progress(0.3, desc="Preparing data for prediction...")
                            last_timestamp = df.index.max()
                            
                            progress(0.6, desc="Generating predictions...")
                            future_predictions_values = predictor.predict_future(df, steps=prediction_hours)
                            
                            future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, prediction_hours + 1)]
                            predictions_df = pd.DataFrame({
                                'Predicted_Power_kW': future_predictions_values,
                                'Timestamp': future_timestamps
                            }).set_index('Timestamp')
                            
                            cost_per_kwh = 0.25 # Example rate in USD
                            predictions_df['Estimated_Cost_USD'] = predictions_df['Predicted_Power_kW'] * cost_per_kwh
                            
                            progress(0.8, desc="Preparing visualizations...")
                            
                            # Combine actual and predicted for plotting
                            combined_df = df['Global_active_power'].tail(predictor.SEQ_LEN).to_frame()
                            combined_df['Type'] = 'Actual'
                            
                            predictions_plot_df = predictions_df.copy()
                            predictions_plot_df.rename(columns={'Predicted_Power_kW': 'Global_active_power'}, inplace=True)
                            predictions_plot_df['Type'] = 'Predicted'
                            
                            full_plot_df = pd.concat([combined_df, predictions_plot_df])
                            
                            fig = px.line(full_plot_df, x=full_plot_df.index, y='Global_active_power', color='Type',
                                          title=f'Energy Consumption: Historical and {prediction_hours}-hour Forecast',
                                          labels={'Global_active_power': 'Power (kW)', 'index': 'Date/Time'},
                                          color_discrete_map={'Actual': 'blue', 'Predicted': 'red'})
                            fig.update_layout(hovermode="x unified", height=500, plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
                            
                            download_link = create_download_link_gradio(predictions_df, "energy_predictions.csv", "Download Predictions as CSV")

                            avg_pred = predictions_df['Predicted_Power_kW'].mean()
                            max_pred = predictions_df['Predicted_Power_kW'].max()
                            min_pred = predictions_df['Predicted_Power_kW'].min()
                            
                            insights_html = f"""
                            <div class="insight-box">
                            - **Average Predicted Power:** {avg_pred:.2f} kW
                            - **Peak Predicted Power:** {max_pred:.2f} kW
                            - **Lowest Predicted Power:** {min_pred:.2f} kW
                            - **Estimated Cost Range:** ${min_pred * cost_per_kwh:.2f} - ${max_pred * cost_per_kwh:.2f} per hour
                            </div>
                            """
                            progress(1.0, desc="Predictions completed!")
                            return fig, predictions_df.reset_index(), gr.HTML(download_link), gr.Markdown('<div class="success-message">Predictions generated successfully!</div>'), gr.HTML(insights_html)
                        
                        except Exception as e:
                            import traceback
                            return None, pd.DataFrame(), gr.HTML(""), gr.Markdown(f'<div class="error-message">An error occurred during prediction: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>'), gr.Markdown("")
                    
                    generate_predictions_button.click(
                        generate_predictions_gradio,
                        inputs=[
                            processed_df_state,
                            predictor_state,
                            training_status_state,
                            prediction_hours_slider
                        ],
                        outputs=[
                            predictions_plot,
                            predicted_data_table,
                            prediction_download_link,
                            prediction_status_message,
                            prediction_insights_md
                        ]
                    )

                # --- Analytics Tab ---
                with gr.TabItem("📈 Analytics", id="analytics_tab"):
                    gr.Markdown('<h1 class="main-header">📈 Energy Usage Analytics</h1>')
                    gr.Markdown("Dive deeper into your energy usage. Discover patterns, peak times, and efficiency insights to optimize consumption.")
                    
                    insights_output_md = gr.HTML('<div class="insight-box">No insights available. Upload data and/or train model first.</div>')
                    seasonality_plot = gr.Plot()

                    def get_analytics(json_df, serialized_predictor, training_status):
                        if json_df == initial_df_json:
                            return gr.HTML('<div class="error-message">Please upload your energy consumption data first.</div>'), None
                        
                        df = pd.read_json(json_df, orient='split')
                        
                        insights_html_output = ""
                        plot_fig = None
                        
                        try:
                            # Always show general consumption patterns if data is there
                            plot_fig = plot_consumption_patterns(df)
                            if plot_fig is None:
                                plot_fig = go.Figure(layout=go.Layout(title="Cannot plot consumption patterns. 'Global_active_power' column not found or insufficient data."))

                            predictor = None
                            if training_status and serialized_predictor:
                                predictor = pickle.loads(base64.b64decode(serialized_predictor))

                            if predictor is not None and training_status:
                                insights = predictor.generate_insights(df.copy())
                                
                                insights_html_output += "<h3>Key Usage Insights</h3>"

                                peak_hour = insights['peak_usage']['peak_hour']
                                peak_consumption = insights['peak_usage']['peak_consumption']
                                peak_recommendation = insights['peak_usage']['recommendation']
                                insights_html_output += f"""
                                <div class="insight-box">
                                <h4>Peak Usage Analysis</h4>
                                Your highest average energy consumption occurs around **{peak_hour}:00** with an average of **{peak_consumption:.2f} kW**.
                                <br><b>Recommendation:</b> {peak_recommendation}.
                                </div>
                                """

                                weekend_avg = insights['usage_pattern']['weekend_avg']
                                weekday_avg = insights['usage_pattern']['weekday_avg']
                                diff_pct = insights['usage_pattern']['difference_pct']
                                insights_html_output += f"""
                                <div class="insight-box">
                                <h4>Usage Patterns: Weekday vs. Weekend</h4>
                                - **Average Weekday Consumption:** {weekday_avg:.2f} kW
                                - **Average Weekend Consumption:** {weekend_avg:.2f} kW
                                - **Weekend vs. Weekday Difference:** {diff_pct:.2f}% {'higher' if diff_pct > 0 else 'lower'}
                                </div>
                                """

                                if insights['efficiency_scores']:
                                    insights_html_output += "<h3>Appliance Efficiency (Sub-meters)</h3>"
                                    for sub_meter, data in insights['efficiency_scores'].items():
                                        insights_html_output += f"""
                                        <div class="insight-box">
                                        **{sub_meter}:**
                                        - Consumes **{data['consumption_percentage']:.2f}%** of total energy.
                                        - Total kWh: **{data['total_kwh']:.2f}**
                                        - **Efficiency Rating:** <span style="color:{'green' if data['efficiency_rating'] == 'Excellent' else 'orange' if data['efficiency_rating'] == 'Good' else 'red'}">{data['efficiency_rating']}</span>
                                        </div>
                                        """
                                else:
                                    insights_html_output += '<div class="insight-box">No sub-metering data available for efficiency analysis.</div>'
                            else:
                                insights_html_output += '<div class="warning-message">Train or load a model to unlock advanced insights like peak usage, patterns, and appliance efficiency.</div>'
                            
                            return gr.HTML(insights_html_output), plot_fig
                        
                        except Exception as e:
                            import traceback
                            return gr.Markdown(f'<div class="error-message">Error generating analytics insights: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>'), plot_fig


                    # This callback will be triggered when the tab changes to Analytics
                    main_tabs.select(
                        get_analytics,
                        inputs=[processed_df_state, predictor_state, training_status_state],
                        outputs=[insights_output_md, seasonality_plot]
                    )
                    # Also update if data or model changes while on tab (e.g., user uploads new data from sidebar)
                    processed_df_state.change(get_analytics, inputs=[processed_df_state, predictor_state, training_status_state], outputs=[insights_output_md, seasonality_plot])
                    predictor_state.change(get_analytics, inputs=[processed_df_state, predictor_state, training_status_state], outputs=[insights_output_md, seasonality_plot])
                    training_status_state.change(get_analytics, inputs=[processed_df_state, predictor_state, training_status_state], outputs=[insights_output_md, seasonality_plot])


                # --- Anomaly Detection Tab ---
                with gr.TabItem("🚨 Anomaly Detection", id="anomaly_tab"):
                    gr.Markdown('<h1 class="main-header">🚨 Anomaly Detection</h1>')
                    gr.Markdown("Detect unusual spikes or drops in your energy data. Identify potential issues or outliers for further investigation.")
                    
                    anomaly_threshold_slider = gr.Slider(minimum=0.1, maximum=0.9, step=0.05, value=0.6, label="Sensitivity Threshold (lower for more anomalies)")
                    detect_anomalies_button = gr.Button("🔍 Detect Anomalies")
                    anomaly_status_message = gr.Markdown()
                    anomaly_summary_output = gr.Markdown()
                    anomalies_plot = gr.Plot()
                    anomaly_report_table = gr.DataFrame(headers=[], datatype="str", col_count=(1, "dynamic"), row_count=(0, "dynamic"), wrap=True)
                    anomaly_download_link = gr.HTML()

                    def run_anomaly_detection(json_df, serialized_predictor, training_status, threshold, progress=gr.Progress()):
                        if not training_status or serialized_predictor is None:
                            return "", None, pd.DataFrame(), gr.HTML(""), gr.Markdown('<div class="warning-message">Please train or load a model first on the "Model Training" page to enable anomaly detection.</div>')
                        if json_df == initial_df_json:
                            return "", None, pd.DataFrame(), gr.HTML(""), gr.Markdown('<div class="error-message">No data uploaded for anomaly detection. Please upload data via the sidebar.</div>')
                        
                        df = pd.read_json(json_df, orient='split')
                        
                        try:
                            progress(0.2, desc="Deserializing model...")
                            predictor = pickle.loads(base64.b64decode(serialized_predictor))
                            
                            progress(0.5, desc="Detecting anomalies...")
                            # Temporarily set the threshold for the detector if it's configurable via slider
                            predictor.anomaly_threshold = threshold
                            anomalies_mask, anomaly_scores_raw = predictor.detect_anomalies(df)
                            
                            if anomalies_mask is not None and np.any(anomalies_mask):
                                num_anomalies = np.sum(anomalies_mask)
                                
                                total_data_points = len(df) - predictor.SEQ_LEN
                                anomaly_rate = (num_anomalies / total_data_points) * 100 if total_data_points > 0 else 0
                                
                                stats_md = f"""
                                <div class="insight-box">
                                - **Total Anomalies Detected:** {num_anomalies}
                                - **Anomaly Rate:** {anomaly_rate:.2f}%
                                - **Max Anomaly Score:** {np.max(anomaly_scores_raw[anomalies_mask]):.2f}
                                - **Anomaly Threshold:** {threshold:.2f}
                                </div>
                                """
                                
                                # Anomalies are detected on the data *after* sequence creation
                                plot_df_for_anomalies = df.iloc[predictor.SEQ_LEN:].copy()
                                
                                # Create a series for scores, aligning by index for plotting
                                # This ensures scores are correctly matched to the plotted data points
                                aligned_scores = pd.Series(0.0, index=plot_df_for_anomalies.index)
                                if np.any(anomalies_mask):
                                    anomalous_data_for_alignment = plot_df_for_anomalies[anomalies_mask]
                                    anomalous_scores_for_alignment = anomaly_scores_raw[anomalies_mask]
                                    for idx, score in zip(anomalous_data_for_alignment.index, anomalous_scores_for_alignment):
                                        if idx in aligned_scores.index:
                                            aligned_scores.loc[idx] = score

                                fig = plot_anomalies(
                                    plot_df_for_anomalies, 
                                    anomalies_mask, 
                                    aligned_scores
                                )

                                anomalous_report_df = plot_df_for_anomalies[anomalies_mask].copy()
                                anomalous_report_df['Anomaly_Score'] = anomaly_scores_raw[anomalies_mask]
                                anomalous_report_df = anomalous_report_df.sort_values('Anomaly_Score', ascending=False).reset_index() # Reset index for Gradio DataFrame
                                
                                download_link = create_download_link_gradio(anomalous_report_df, "energy_anomalies.csv", "Download Anomalies as CSV")
                                
                                progress(1.0, desc="Anomaly detection completed!")
                                return gr.Markdown(stats_md), fig, anomalous_report_df, gr.HTML(download_link), \
                                       gr.Markdown('<div class="success-message">Anomalies detected successfully!</div>')
                            else:
                                progress(1.0, desc="No anomalies detected!")
                                return "", None, pd.DataFrame(), gr.HTML(""), gr.Markdown('<div class="insight-box">No anomalies detected in the data for the given threshold.</div>')

                        except Exception as e:
                            import traceback
                            return "", None, pd.DataFrame(), gr.HTML(""), gr.Markdown(f'<div class="error-message">An error occurred during anomaly detection: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>')

                    detect_anomalies_button.click(
                        run_anomaly_detection,
                        inputs=[
                            processed_df_state,
                            predictor_state,
                            training_status_state,
                            anomaly_threshold_slider
                        ],
                        outputs=[
                            anomaly_summary_output,
                            anomalies_plot,
                            anomaly_report_table,
                            anomaly_download_link,
                            anomaly_status_message
                        ]
                    )

                # --- Reports Tab ---
                with gr.TabItem("📋 Reports", id="reports_tab"):
                    gr.Markdown('<h1 class="main-header">📋 Generate Reports</h1>')
                    gr.Markdown("Generate and download detailed reports on your energy consumption, predictions, efficiency, and anomalies.")
                    
                    report_type_dropdown = gr.Dropdown(
                        choices=[
                            "Consumption Summary Report",
                            "Prediction Report",
                            "Efficiency Analysis Report",
                            "Anomaly Report"
                        ],
                        value="Consumption Summary Report",
                        label="Select Report Type"
                    )
                    
                    with gr.Row():
                        start_date_input = gr.Date(label="Start Date")
                        end_date_input = gr.Date(label="End Date")
                    
                    generate_report_button = gr.Button("📄 Generate Report")
                    report_output_md = gr.Markdown()
                    report_dataframe = gr.DataFrame(headers=[], datatype="str", col_count=(1, "dynamic"), row_count=(0, "dynamic"), wrap=True)
                    report_plot = gr.Plot()
                    report_download_html = gr.HTML()

                    # Function to update date inputs based on uploaded data range
                    def get_min_max_dates_for_report(json_df):
                        if json_df == initial_df_json:
                            return None, None
                        df = pd.read_json(json_df, orient='split')
                        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                            return df.index.min().date(), df.index.max().date()
                        return None, None
                    
                    # Callback to update date inputs when processed_df_state changes
                    processed_df_state.change(
                        get_min_max_dates_for_report,
                        inputs=[processed_df_state],
                        outputs=[start_date_input, end_date_input]
                    )

                    def generate_report_gradio(report_type, start_date, end_date, json_df, serialized_predictor, training_status, progress=gr.Progress()):
                        if json_df == initial_df_json:
                            return gr.Markdown('<div class="error-message">Please upload your energy consumption data first.</div>'), pd.DataFrame(), None, gr.HTML("")
                        
                        df_original = pd.read_json(json_df, orient='split')
                        
                        # Filter by date range for the report
                        try:
                            start_dt = pd.to_datetime(start_date)
                            end_dt = pd.to_datetime(end_date) + timedelta(days=1,microseconds=-1) # Include full end day
                            filtered_df = df_original[(df_original.index >= start_dt) & (df_original.index <= end_dt)]
                        except Exception as e:
                            return gr.Markdown(f'<div class="error-message">Invalid date range: {e}</div>'), pd.DataFrame(), None, gr.HTML("")

                        if filtered_df.empty:
                            return gr.Markdown('<div class="warning-message">No data available for the selected date range. Please adjust the dates.</div>'), pd.DataFrame(), None, gr.HTML("")

                        report_md = f"## {report_type} ({start_date} to {end_date})"
                        report_df_output = pd.DataFrame()
                        report_plot_fig = None
                        download_link_html = gr.HTML("")

                        if report_type == "Consumption Summary Report":
                            progress(0.5, desc="Generating Consumption Summary...")
                            total_consumption_kwh = filtered_df['Global_active_power'].sum() / 1000
                            avg_consumption_kw = filtered_df['Global_active_power'].mean()
                            peak_consumption_kw = filtered_df['Global_active_power'].max()
                            min_consumption_kw = filtered_df['Global_active_power'].min()
                            
                            cost_per_kwh = 0.25 # USD
                            estimated_cost = total_consumption_kwh * cost_per_kwh

                            report_md += f"""
                            <div class="insight-box">
                            - **Total Energy Consumption:** {total_consumption_kwh:.2f} kWh
                            - **Average Hourly Consumption:** {avg_consumption_kw:.2f} kW
                            - **Peak Hourly Consumption:** {peak_consumption_kw:.2f} kW
                            - **Minimum Hourly Consumption:** {min_consumption_kw:.2f} kW
                            - **Estimated Cost:** ${estimated_cost:.2f}
                            </div>
                            """
                            report_plot_fig = px.line(filtered_df, x=filtered_df.index, y='Global_active_power', 
                                                       title='Consumption Trend Over Selected Period',
                                                       labels={'Global_active_power': 'Power (kW)', 'index': 'Date/Time'})
                            report_plot_fig.update_layout(plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
                            report_df_output = filtered_df.reset_index() # For download, user might want raw filtered data
                            download_link_html = create_download_link_gradio(report_df_output, "consumption_summary_report.csv", "Download Raw Data for Report")

                        elif report_type == "Prediction Report":
                            if not training_status or serialized_predictor is None:
                                report_md += gr.Markdown('<div class="warning-message">Please train or load a model first to generate prediction reports.</div>')
                            else:
                                progress(0.3, desc="Generating predictions for report...")
                                try:
                                    predictor = pickle.loads(base64.b64decode(serialized_predictor))
                                    # To generate predictions for a report period, we need to predict from the end of the *original* loaded data
                                    last_timestamp_original_df = df_original.index.max()
                                    # Calculate enough steps to cover up to the end_dt of the report
                                    # Ensure we predict at least FUTURE_PERIOD_PREDICT steps
                                    time_diff_hours = int((end_dt - last_timestamp_original_df).total_seconds() / 3600)
                                    total_future_steps = max(time_diff_hours, predictor.FUTURE_PERIOD_PREDICT) + 24 # Add a buffer of 24 hours

                                    if total_future_steps < 1: total_future_steps = 24 # Ensure at least 24 hours predicted

                                    future_predictions_values = predictor.predict_future(df_original, steps=total_future_steps)
                                    future_timestamps = [last_timestamp_original_df + timedelta(hours=i) for i in range(1, total_future_steps + 1)]
                                    
                                    predictions_report_df_full = pd.DataFrame({
                                        'Predicted_Power_kW': future_predictions_values,
                                        'Timestamp': future_timestamps
                                    }).set_index('Timestamp')

                                    # Filter predictions for the report's date range
                                    predictions_report_df = predictions_report_df_full[(predictions_report_df_full.index >= start_dt) & (predictions_report_df_full.index <= end_dt)]

                                    if not predictions_report_df.empty:
                                        report_df_output = predictions_report_df.reset_index()
                                        report_plot_fig = px.line(predictions_report_df, x=predictions_report_df.index, y='Predicted_Power_kW',
                                                                title='Predicted Consumption Over Selected Period',
                                                                labels={'Predicted_Power_kW': 'Power (kW)', 'index': 'Date/Time'})
                                        report_plot_fig.update_layout(plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
                                        download_link_html = create_download_link_gradio(report_df_output, "prediction_report.csv", "Download Prediction Report CSV")
                                    else:
                                        report_md += gr.Markdown('<div class="insight-box">No predictions available for the selected date range. Try extending the forecast horizon or adjusting dates.</div>')
                                except Exception as e:
                                    report_md += gr.Markdown(f'<div class="error-message">Error generating prediction report: {e}<br><pre>{traceback.format_exc()}</pre></div>')

                        elif report_type == "Efficiency Analysis Report":
                            if not training_status or serialized_predictor is None:
                                report_md += gr.Markdown('<div class="warning-message">Please train or load a model first to generate efficiency analysis reports.</div>')
                            else:
                                progress(0.5, desc="Generating Efficiency Analysis...")
                                try:
                                    predictor = pickle.loads(base64.b64decode(serialized_predictor))
                                    insights = predictor.generate_insights(filtered_df.copy())
                                    efficiency_scores = insights.get('efficiency_scores', {})
                                    
                                    if efficiency_scores:
                                        efficiency_data = pd.DataFrame.from_dict(efficiency_scores, orient='index')
                                        efficiency_data.index.name = 'Sub_Meter'
                                        report_df_output = efficiency_data.reset_index()

                                        labels = [f"{k} ({v['efficiency_rating']})" for k,v in efficiency_scores.items()]
                                        values = [v['total_kwh'] for v in efficiency_scores.items()]
                                        
                                        sub_meter_cols_filtered = [col for col in filtered_df.columns if 'Sub_metering_' in col]
                                        if 'Global_active_power' in filtered_df.columns and sub_meter_cols_filtered:
                                            other_consumption = filtered_df['Global_active_power'].sum() - filtered_df[sub_meter_cols_filtered].sum().sum()
                                            if other_consumption > 0:
                                                labels.append('Other Consumption')
                                                values.append(other_consumption)

                                        if values:
                                            report_plot_fig = px.pie(names=labels, values=values, title='Energy Consumption Distribution by Area',
                                                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                                            report_plot_fig.update_layout(plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
                                        else:
                                            report_md += gr.Markdown('<div class="insight-box">No sub-metering data to display efficiency breakdown.</div>')
                                        download_link_html = create_download_link_gradio(report_df_output, "efficiency_report.csv", "Download Efficiency Report CSV")
                                    else:
                                        report_md += gr.Markdown('<div class="insight-box">No sub-metering data available for efficiency analysis in the selected period.</div>')
                                except Exception as e:
                                    report_md += gr.Markdown(f'<div class="error-message">Error generating efficiency analysis report: {e}<br><pre>{traceback.format_exc()}</pre></div>')

                        elif report_type == "Anomaly Report":
                            if not training_status or serialized_predictor is None:
                                report_md += gr.Markdown('<div class="warning-message">Please train or load a model first to generate anomaly reports.</div>')
                            else:
                                progress(0.5, desc="Generating Anomaly Report...")
                                try:
                                    predictor = pickle.loads(base64.b64decode(serialized_predictor))
                                    # Detect anomalies on the *full* original dataset, then filter for the report's date range
                                    anomalies_mask_full, anomaly_scores_full = predictor.detect_anomalies(df_original)

                                    if anomalies_mask_full is not None and np.any(anomalies_mask_full):
                                        anomalous_df_with_scores = df_original.iloc[predictor.SEQ_LEN:].copy()
                                        anomalous_df_with_scores['Anomaly_Flag'] = anomalies_mask_full
                                        anomalous_df_with_scores['Anomaly_Score'] = anomaly_scores_full
                                        
                                        # Filter anomalies for the report's date range
                                        anomalous_report_df = anomalous_df_with_scores[anomalous_df_with_scores['Anomaly_Flag'] & 
                                                                                (anomalous_df_with_scores.index >= start_dt) & 
                                                                                (anomalous_df_with_scores.index <= end_dt)]
                                        
                                        if not anomalous_report_df.empty:
                                            report_md += f"""<div class="insight-box">**Total Anomalies in Report Period:** {len(anomalous_report_df)}</div>"""
                                            report_df_output = anomalous_report_df.drop('Anomaly_Flag', axis=1).sort_values('Anomaly_Score', ascending=False).reset_index()
                                            
                                            # For plotting, ensure we use the filtered_df from the overall report date range
                                            # And then apply the anomaly mask/scores relevant to that filtered view
                                            plot_df_for_anomalies = filtered_df.copy() # Use the already date-filtered data
                                            # Re-create mask and scores aligned to this smaller plot_df_for_anomalies
                                            plot_anomalies_mask = plot_df_for_anomalies.index.isin(anomalous_report_df.index)
                                            
                                            aligned_scores_for_plot = pd.Series(0.0, index=plot_df_for_anomalies.index)
                                            for idx, score in anomalous_report_df.set_index('index')['Anomaly_Score'].items(): # Set index back for lookup
                                                if idx in aligned_scores_for_plot.index:
                                                    aligned_scores_for_plot.loc[idx] = score

                                            report_plot_fig = plot_anomalies(
                                                plot_df_for_anomalies, 
                                                plot_anomalies_mask, 
                                                aligned_scores_for_plot
                                            )
                                            download_link_html = create_download_link_gradio(report_df_output, "anomaly_report.csv", "Download Anomaly Report CSV")
                                        else:
                                            report_md += gr.Markdown('<div class="insight-box">No anomalies detected in the selected date range.</div>')
                                    else:
                                        report_md += gr.Markdown('<div class="insight-box">No anomalies detected in the entire dataset with current model settings.</div>')
                                except Exception as e:
                                    report_md += gr.Markdown(f'<div class="error-message">Error generating anomaly report: {e}<br><pre>{traceback.format_exc()}</pre></div>')
                        
                        progress(1.0, desc="Report generated!")
                        return gr.Markdown(report_md), report_df_output, report_plot_fig, gr.HTML(download_link_html)

                    generate_report_button.click(
                        generate_report_gradio,
                        inputs=[
                            report_type_dropdown,
                            start_date_input,
                            end_date_input,
                            processed_df_state,
                            predictor_state,
                            training_status_state
                        ],
                        outputs=[
                            report_output_md,
                            report_dataframe,
                            report_plot,
                            report_download_html
                        ]
                    )

demo.launch()

