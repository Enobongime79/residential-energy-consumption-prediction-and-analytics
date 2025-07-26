import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
from io import BytesIO
import base64
import traceback

# Assuming model_utils.py is in the same directory
from model_utils import EnergyPredictor # This import is crucial for the forecasting functionality

st.set_page_config(
    page_title="⚡ Energy Consumption Forecasting & Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        color: #2F80ED;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #000000; /* Black background for metrics */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Slightly stronger shadow for contrast */
        text-align: center;
        margin-bottom: 20px;
        color: #FFFFFF !important; /* White text for metrics */
    }
    /* Ensure the values within stMetric are also white */
    .stMetric > div > div > div > div > div > div {
        color: #FFFFFF !important;
    }
    .insight-box {
        background-color: #000000; /* Changed to black */
        border-left: 5px solid #2F80ED;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        font-size: 1.1em;
        color: #FFFFFF; /* Changed to white text */
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.2rem;
    }
    h2 {
        color: #FFFFFF; /* Ensure headers are visible on dark backgrounds */
    }
    .report-section {
        background-color: #000000; /* Changed to black */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        color: #FFFFFF; /* Changed to white text */
    }
    .error-message {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        font-size: 1.1em;
        color: #000000; /* Ensure error messages are readable */
    }
    </style>
    """, unsafe_allow_html=True)

def reset_session_state():
    """Reset session state variables to their initial values.
    This function is useful for clearing the application state, for example,
    when a new data file is uploaded or to start fresh.
    """
    st.session_state.predictor = None
    st.session_state.training_completed = False
    st.session_state.df = None
    st.session_state.evaluation_results = {}

def validate_dataframe(df):
    """Validate the uploaded dataframe meets minimum requirements for analysis and model training.
    Checks for:
    - Non-empty DataFrame
    - Presence of 'Global_active_power' column
    - Datetime index
    - Sufficient data points (at least 24 hours)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"

    if 'Global_active_power' not in df.columns:
        return False, "Required column 'Global_active_power' not found"

    # Ensure index is DatetimeIndex before checking its length or min/max
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return False, "DataFrame index must be convertible to datetime"

    if len(df) < 24:   # Minimum 24 hours of data
        return False, "Insufficient data: need at least 24 hours of data"

    return True, "Data validation successful"

# --- Session State Initialization ---
# Streamlit uses session state to preserve variable values across reruns.
# This ensures that uploaded data, trained models, and results persist
# as the user interacts with the application.
if 'predictor' not in st.session_state:
    st.session_state.predictor = None # Stores the trained EnergyPredictor model instance
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False # Flag to indicate if model training has been successfully completed
if 'df' not in st.session_state:
    st.session_state.df = None # Stores the uploaded and preprocessed DataFrame
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {} # Stores performance metrics from model evaluation
if 'error_message' not in st.session_state:
    st.session_state.error_message = None # Stores error messages to be displayed prominently

# Display error message if exists
if st.session_state.error_message:
    st.markdown(f'<div class="error-message">{st.session_state.error_message}</div>', unsafe_allow_html=True)
    st.session_state.error_message = None  # Clear error message after displaying

# --- Helper Functions for Plotting and Downloading ---

def create_download_link(df, filename, link_text):
    """Generates a downloadable link for a DataFrame as a CSV file."""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def plot_training_history(history):
    """
    Plots the training and validation loss and Mean Absolute Error (MAE) from the model's history object.
    Helps visualize the model's learning progress and detect overfitting/underfitting.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model MAE'))

    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss', legendgroup='loss'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss', legendgroup='loss'), row=1, col=1)

    # Debug: Check if 'mean_absolute_error' key exists, if not, try 'mae' or 'MeanAbsoluteError'
    mae_key = 'mean_absolute_error'
    if mae_key not in history.history:
        # Common alternative keys for MAE in Keras history
        if 'mae' in history.history:
            mae_key = 'mae'
        elif 'MeanAbsoluteError' in history.history: # For newer TensorFlow/Keras versions with default metric names
            mae_key = 'MeanAbsoluteError'
        else:
            st.warning("MAE metric not found in training history. Displaying only Loss plots.")
            # Remove the second subplot if MAE data is unavailable
            fig = make_subplots(rows=1, cols=1, subplot_titles=('Model Loss'))
            fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Training Loss', legendgroup='loss'), row=1, col=1)
            fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss', legendgroup='loss'), row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_layout(height=400, showlegend=True, title_text="Training History",
                              plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
            return fig


    fig.add_trace(go.Scatter(y=history.history[mae_key], mode='lines', name='Training MAE', legendgroup='mae'), row=1, col=2)
    fig.add_trace(go.Scatter(y=history.history[f'val_{mae_key}'], mode='lines', name='Validation MAE', legendgroup='mae'), row=1, col=2)

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
    """
    Plots actual and predicted energy consumption values over time.
    Provides a visual comparison of the model's forecasting accuracy.
    """
    fig = go.Figure()
    if dates is not None:
        fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual Power', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted Power', line=dict(color='red', dash='dot')))
        fig.update_xaxes(title_text="Date")
    else:
        fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual Power', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted Power', line=dict(color='red', dash='dot')))

    fig.update_layout(title_text=title, yaxis_title="Global Active Power (kW)", height=400,
                      plot_bgcolor='#23272f',   # Dark background for plots
                      paper_bgcolor='#23272f', # Dark background for the paper area
                      font_color='#f8f8f8')     # White font color for titles, labels, etc.
    return fig

def plot_feature_importance(df):
    """
    Analyzes and plots the correlation of various features with 'Global_active_power'.
    This helps identify which factors might influence energy consumption the most.
    """
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
                              plot_bgcolor='#23272f',   # Dark background for plots
                              paper_bgcolor='#23272f', # Dark background for the paper area
                              font_color='#f8f8f8')     # White font color for titles, labels, etc.
            return fig
        else:
            return None # No numeric features to correlate
    return None

def plot_consumption_patterns(df):
    """
    Visualizes daily, hourly, and monthly energy consumption patterns,
    and provides a breakdown of consumption by sub-meters.

    Sub-metering helps in understanding energy usage by specific categories or appliances.
    Based on common household energy datasets (like the one this structure implies, e.g., UCI's Individual household electric power consumption Data Set):
    - **Sub_metering_1**: Typically refers to energy consumption for the kitchen area (e.g., dishwasher, oven, microwave).
    - **Sub_metering_2**: Typically refers to energy consumption for laundry (e.g., washing machine, tumble dryer) and possibly electric water heater.
    - **Sub_metering_3**: Typically refers to energy consumption for electric heating, air conditioning, and a water heater if not covered by sub_metering_2.
    - **Other_consumption**: Represents the remaining energy consumption not captured by the specific sub-meters, which includes lighting, electronics (TVs, computers), and other miscellaneous plug loads throughout the house.
    """
    if 'Global_active_power' not in df.columns:
        return None

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
    # Identifies columns related to sub-metering (e.g., 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3')
    sub_meter_cols = [col for col in df.columns if 'Sub_metering_' in col]
    if sub_meter_cols and 'Global_active_power' in df.columns:
        # Calculate 'Other_consumption' by subtracting known sub-metered consumption
        # from the total global active power. This accounts for all other electrical loads.
        df['Other_consumption'] = df['Global_active_power'] - df[sub_meter_cols].sum(axis=1)
        df['Other_consumption'] = df['Other_consumption'].clip(lower=0) # Ensure consumption is not negative

        pie_data = df[sub_meter_cols + ['Other_consumption']].sum()
        pie_data = pie_data[pie_data > 0] # Filter out sub-meters with zero total consumption
        if not pie_data.empty:
            # Bug fix: `labels` parameter expects a list of labels, not `pie_data.index` directly if it's a Pandas index.
            # However, Plotly usually handles this correctly. The main issue for `textfont` might be that it's a global setting for all text.
            # For better readability, ensure contrast or adjust `textinfo` as done below.
            fig.add_trace(go.Pie(
                labels=pie_data.index, values=pie_data.values, name='Sub-meters',
                showlegend=False,
                textinfo='percent', # Changed to only show percentage to reduce clutter
                pull=[0.05]*len(pie_data), # Pull out slices slightly for emphasis
                marker=dict(colors=px.colors.qualitative.Pastel),
                textfont=dict(size=16, color='#23272f') # Set text color to dark for better contrast on pastel slices
            ), row=2, col=2)
        else:
            fig.add_annotation(
                text="No positive sub-meter data to display in breakdown.",
                xref="paper", yref="paper", x=0.75, y=0.25, showarrow=False,
                font=dict(color="#f8f8f8")
            )
    else:
        fig.add_annotation(
            text="No sub-meter data available or 'Global_active_power' missing.",
            xref="paper", yref="paper", x=0.75, y=0.25, showarrow=False,
            font=dict(color="#f8f8f8") # Ensure annotation text is visible on dark background
        )

    fig.update_layout(
        height=700, width=1000, # Increased width for better spacing
        title_text="Energy Consumption Patterns",
        plot_bgcolor='#23272f', # Dark background for plots
        paper_bgcolor='#23272f', # Dark background for the paper area
        font_color='#f8f8f8', # White font color for titles, labels, etc.
        margin=dict(t=60, l=50, r=50, b=50) # Increased margins
    )
    return fig

def plot_anomalies(df, anomalies, scores):
    """
    Plots energy consumption data and highlights detected anomalies.
    Anomalies are data points that deviate significantly from the expected pattern.
    """
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
                      plot_bgcolor='#23272f',   # Dark background for anomaly plot
                      paper_bgcolor='#23272f', # Dark background for anomaly plot
                      font_color='#f8f8f8'     # White font color for anomaly plot
                      )
    return fig

# --- Sidebar Navigation and Controls ---
st.sidebar.markdown("# ⚡ Energy Analytics")

page = st.sidebar.selectbox("Navigate to:", [
    "📊 Dashboard",
    "🔧 Model Training",
    "🔮 Predictions",
    "📈 Analytics",
    "🚨 Anomaly Detection",
    "📋 Reports"
])

st.sidebar.subheader("🛠️ Model Parameters")
seq_len = st.sidebar.slider("Sequence Length (hours)", 24, 336, 168) # Up to 2 weeks
future_period_predict = st.sidebar.slider("Prediction Horizon (hours)", 1, 48, 24)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64, 128], value=64)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)

st.sidebar.subheader("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# --- Data Upload and Preprocessing Logic (Modified) ---
if uploaded_file is not None:
    try:
        # Reset file pointer to the beginning each time, crucial for multiple uploads or reruns
        uploaded_file.seek(0)

        # Read the CSV with fixed format (based on kaggle_data_1h.csv)
        try:
            df = pd.read_csv(uploaded_file, sep=',', index_col='time', parse_dates=True)
            # Ensure the index is a DatetimeIndex after parsing
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index could not be parsed as datetime.")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.error("Please ensure your CSV file matches the required format:")
            st.error("1. Must have a 'time' column (which will become the index) that can be parsed as dates.")
            st.error("2. Must use comma (,) as the separator.")
            st.error("3. Must have 'Global_active_power' column.")
            st.session_state.df = None
            st.stop() # Stop execution to prevent further errors with a bad DataFrame

        # Validate minimum data requirements
        # Re-using the `validate_dataframe` function is a good practice here.
        is_valid, validation_msg = validate_dataframe(df.copy()) # Pass a copy to avoid modifying original df during validation
        if not is_valid:
            st.error(f"❌ Data validation failed: {validation_msg}")
            st.session_state.df = None
            st.stop()

        # Convert all columns to numeric, coercing errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                # Fill NaNs using ffill, then bfill to handle gaps
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(method='bfill', inplace=True)
                # If still NaNs (e.g., at the very beginning of the series or all NaNs), fill with mean
                if df[col].isnull().any():
                    st.warning(f"Column '{col}' still has NaNs after ffill/bfill. Filling remaining NaNs with mean (or 0 if mean is NaN).")
                    # Debug: If the entire column is NaN, df[col].mean() will be NaN. Handle this.
                    if pd.isna(df[col].mean()):
                         df[col].fillna(0, inplace=True) # Fill with 0 if mean is NaN (e.g., all values were non-numeric)
                    else:
                        df[col].fillna(df[col].mean(), inplace=True)


        # Validate Global_active_power values again after processing
        if df['Global_active_power'].isnull().any():
            st.error("Error: 'Global_active_power' column contains null values after all processing. Please check your data.")
            st.session_state.df = None
            st.stop()
        elif (df['Global_active_power'] < 0).any():
            st.warning("Warning: 'Global_active_power' column contains negative values. These will be set to 0.")
            df.loc[df['Global_active_power'] < 0, 'Global_active_power'] = 0

        # Assign the processed DataFrame to session state
        st.session_state.df = df
        st.success("CSV file uploaded and processed successfully!")

    except Exception as e:
        st.error(f"❌ An unexpected error occurred while loading or processing the file: {str(e)}")
        st.error("Detailed error information:")
        st.code(traceback.format_exc())
        st.session_state.df = None
        st.session_state.error_message = "File processing failed. Please check the file format and content."


# Display dataset info in sidebar
if st.session_state.df is not None:
    st.sidebar.subheader("Dataset Info")
    df = st.session_state.df # Get the processed DF from session state
    st.sidebar.write(f"**Rows:** {df.shape[0]}")
    st.sidebar.write(f"**Columns:** {df.shape[1]}")

    # Display date range only if index is DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
        st.sidebar.info(f"**Date Range:** {df.index.min().strftime('%Y-%m-%d %H:%M:%S')} to {df.index.max().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.sidebar.warning("Date Range could not be determined: DataFrame index is not a proper datetime index or DataFrame is empty.")

    st.sidebar.write("---")
    st.sidebar.write("**Columns:**")
    st.sidebar.write(df.columns.tolist())


# --- Main Page Logic ---

if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">📊 Energy Consumption Dashboard</h1>', unsafe_allow_html=True)
    st.info("Explore your energy data at a glance. View key metrics, trends, and patterns to understand your consumption habits.")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("### Key Metrics")
        col1, col2, col3 = st.columns(3)

        if 'Global_active_power' in df.columns:
            with col1:
                st.metric("Average Power (kW)", f"{df['Global_active_power'].mean():.2f}")
            with col2:
                st.metric("Max Power (kW)", f"{df['Global_active_power'].max():.2f}")
            with col3:
                st.metric("Min Power (kW)", f"{df['Global_active_power'].min():.2f}")

            st.write("### Dataset Overview")
            st.write(f"**Number of entries:** {len(df)}")
            # The data frequency is determined by how `preprocess_data` in `model_utils.py` resamples.
            # It's good to state the assumption or confirm it based on the `model_utils`.
            st.write(f"**Data Frequency:** This dashboard assumes the data is processed to hourly frequency by the model utilities.")
            st.write(f"**Missing Values (after processing):** {df.isnull().sum().sum()} (should be 0 if processing was fully successful)")
            st.write(f"**Columns and Data Types:**")
            st.dataframe(df.dtypes.rename('DataType').reset_index().rename(columns={'index': 'Column Name'}))

            st.write("### Sample Data")
            st.dataframe(df.head())

            st.write("### Consumption Patterns")
            consumption_patterns_fig = plot_consumption_patterns(df)
            if consumption_patterns_fig:
                st.plotly_chart(consumption_patterns_fig, use_container_width=True)
            else:
                st.warning("Cannot plot consumption patterns: 'Global_active_power' column not found or insufficient data.")

        else:
            st.warning("Please upload a CSV file containing 'Global_active_power' column to view the dashboard.")
    else:
        st.info("Please upload your energy consumption data via the sidebar to get started!")

elif page == "🔧 Model Training":
    st.markdown('<h1 class="main-header">🔧 Train Forecasting Model</h1>', unsafe_allow_html=True)
    st.info("Train a forecasting model on your uploaded data. Adjust parameters and start training to build a custom energy predictor.")

    if st.session_state.df is None:
        st.info("Please upload your energy consumption data first in the sidebar.")
    else:
        st.write("### Model Training Configuration")
        st.markdown(f"""
        - **Sequence Length:** {seq_len} hours (how many past hours the model observes to make a prediction)
        - **Prediction Horizon:** {future_period_predict} hours (how far into the future the model predicts energy consumption)
        - **Batch Size:** {batch_size} (number of samples processed before the model's internal parameters are updated)
        - **Epochs:** {epochs} (number of complete passes through the entire training dataset)
        """)

        st.write("### Feature Correlation Overview")
        feature_corr_fig = plot_feature_importance(st.session_state.df)
        if feature_corr_fig:
            st.plotly_chart(feature_corr_fig, use_container_width=True)
        else:
            st.info("Cannot plot feature correlations. Ensure 'Global_active_power' and other numeric columns are present in your uploaded data.")

        st.write("---")

        col_train, col_load = st.columns(2)

        with col_train:
            if st.button("🚀 Start Training Model", use_container_width=True):
                if 'Global_active_power' not in st.session_state.df.columns:
                    st.error("Cannot train model: 'Global_active_power' column not found in your data.")
                else:
                    # Create a progress bar and status text for user feedback during training
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Initialize predictor with chosen parameters
                        status_text.text("Initializing model...")
                        predictor = EnergyPredictor(
                            seq_len=seq_len,
                            future_period_predict=future_period_predict,
                            batch_size=batch_size,
                            epochs=epochs
                        )
                        progress_bar.progress(10)

                        # Preprocess data using the predictor's method
                        status_text.text("Preprocessing data for training (resampling, scaling, feature engineering)...")
                        # Debug: ensure preprocess_data handles the DataFrame correctly (e.g., returns a new df, doesn't modify in place unexpectedly)
                        df_processed = predictor.preprocess_data(st.session_state.df.copy()) # Ensure to pass a copy

                        progress_bar.progress(30)

                        # Prepare sequences (X, y for training and validation)
                        status_text.text("Preparing training and validation sequences...")
                        # Debug: `prepare_data` is called inside `train` method in `model_utils.py` based on `EnergyPredictor` class structure.
                        # Calling it here might be redundant or indicative of a misunderstanding of `model_utils.py`.
                        # Let's assume `predictor.train` handles data preparation internally for simplicity based on the current structure.
                        # If `prepare_data` is a separate prerequisite step, it should be explicitly called.
                        # Given the `predictor.train(df_processed)` call below, `train_x, train_y, valid_x, valid_y = predictor.prepare_data(df_processed)`
                        # is likely not needed *before* calling `train` if `train` itself orchestrates this.
                        # For now, I'll assume `train` encapsulates this, or remove this specific line if it's meant to be internal to `train`.
                        # Based on `model_utils.py` snippets, `train` takes the `df` directly, implying it does its own internal prep.
                        # So, removing the explicit prepare_data call here.
                        # train_x, train_y, valid_x, valid_y = predictor.prepare_data(df_processed) # Removed as train() should handle this
                        progress_bar.progress(50)

                        # Train model
                        status_text.text("Training model... This may take a while depending on data size and epochs.")
                        history = predictor.train(df_processed) # Pass the preprocessed DataFrame
                        progress_bar.progress(80)

                        # Store results in session state
                        st.session_state.predictor = predictor
                        st.session_state.training_completed = True

                        # Evaluate model performance
                        status_text.text("Evaluating model performance on training and validation sets...")
                        evaluation_results = predictor.evaluate_model(st.session_state.df.copy()) # Evaluate on the original data for consistency
                        st.session_state.evaluation_results = evaluation_results
                        progress_bar.progress(90)

                        # --- ADDED: Save the trained model ---
                        status_text.text("Saving trained model and scaler for future use...")
                        # Ensure 'models' directory exists before saving
                        os.makedirs('models', exist_ok=True)
                        st.session_state.predictor.save_complete_model("models/energy_model")
                        # --- END ADDED ---

                        progress_bar.progress(100)
                        status_text.text("Training completed and model saved!")
                        st.success("Model training completed and saved successfully!")

                        # Display results
                        st.write("### Training Metrics")
                        st.json(evaluation_results['train_metrics'])
                        st.write("**Validation Set:**")
                        st.json(evaluation_results['validation_metrics'])

                        st.write("### Training History Plots")
                        # Debug: ensure `history` object from `predictor.train` is compatible with `plot_training_history`
                        # Checked `plot_training_history` and `lstm.py` (assuming `history.history` structure) - looks okay.
                        st.plotly_chart(plot_training_history(history), use_container_width=True)

                        st.balloons() # Visual confirmation of success

                    except Exception as e:
                        st.error(f"An error occurred during training: {str(e)}")
                        st.error("Detailed error information:")
                        st.code(traceback.format_exc())
                        st.session_state.training_completed = False
                        st.session_state.predictor = None # Clear predictor on error
                        st.session_state.error_message = "Model training failed. Please check your data and parameters."
                    finally:
                        # Clean up progress indicators regardless of success or failure
                        progress_bar.empty()
                        status_text.empty()

        with col_load:
            if st.button("📂 Load Saved Model", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Ensure 'models' directory exists
                    status_text.text("Checking model directory...")
                    os.makedirs('models', exist_ok=True) # Ensure this directory exists before trying to load
                    progress_bar.progress(20)

                    # Initialize predictor (without training parameters, as they will be loaded)
                    status_text.text("Initializing predictor for loading...")
                    predictor = EnergyPredictor() # Instantiate the class
                    progress_bar.progress(40)

                    # Load model
                    status_text.text("Loading model files (model, scaler, metadata)...")
                    # Debug: `load_complete_model` needs to be implemented correctly in `model_utils.py`
                    predictor.load_complete_model("models/energy_model")
                    st.session_state.predictor = predictor
                    st.session_state.training_completed = True
                    progress_bar.progress(80)

                    # Evaluate model if data is available
                    if st.session_state.df is not None:
                        status_text.text("Evaluating loaded model on current data...")
                        evaluation_results = predictor.evaluate_model(st.session_state.df.copy())
                        st.session_state.evaluation_results = evaluation_results
                        progress_bar.progress(100)

                        st.success("Model loaded and evaluated successfully!")
                        st.write("### Loaded Model Evaluation (on current data)")
                        st.write("**Training Set:**")
                        st.json(evaluation_results['train_metrics'])
                        st.write("**Validation Set:**")
                        st.json(evaluation_results['validation_metrics'])
                    else:
                        st.success("Model loaded successfully!")
                        st.info("Upload data to evaluate the loaded model on your dataset.")

                except FileNotFoundError:
                    st.warning("No saved model found at 'models/energy_model'. Please train a model first or verify the path.")
                    st.session_state.training_completed = False
                    st.session_state.predictor = None
                except Exception as e:
                    st.error(f"An error occurred while loading the model: {str(e)}")
                    st.error("Detailed error information:")
                    st.code(traceback.format_exc())
                    st.session_state.training_completed = False
                    st.session_state.predictor = None # Clear predictor on error
                    st.session_state.error_message = "Model loading failed. Ensure model files are present and valid."
                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()


elif page == "🔮 Predictions":
    st.markdown('<h1 class="main-header">🔮 Future Energy Predictions</h1>', unsafe_allow_html=True)
    st.info("Generate future energy consumption forecasts using your trained model. Visualize and download predictions for planning.")

    if not st.session_state.training_completed or st.session_state.predictor is None:
        st.info("Please train or load a model first on the 'Model Training' page before generating predictions.")
    elif st.session_state.df is None:
        st.info("Please upload your energy consumption data first in the sidebar to provide context for predictions.")
    else:
        st.write("### Generate Future Forecasts")
        # Ensure prediction_hours doesn't exceed a reasonable limit (e.g., 1 week = 168 hours)
        prediction_hours = st.slider("Hours to forecast into the future", 1, 168, st.session_state.predictor.FUTURE_PERIOD_PREDICT)

        if st.button("📈 Generate Predictions", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                df = st.session_state.df.copy() # Work with a copy to avoid unexpected modifications
                progress_bar.progress(10)

                # Get last known timestamp from the uploaded data
                last_timestamp = df.index.max()
                status_text.text("Preparing data for prediction...")
                progress_bar.progress(30)

                # Generate predictions using the loaded predictor
                status_text.text("Generating predictions using the trained model...")
                # Debug: Ensure `predict_future` method exists in `EnergyPredictor` and works as expected.
                # It should return an array of predicted values.
                future_predictions_values = st.session_state.predictor.predict_future(df, steps=prediction_hours)
                progress_bar.progress(60)

                # Create future timestamps based on the last known timestamp and prediction horizon
                future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, prediction_hours + 1)]

                # Create a DataFrame for predictions
                predictions_df = pd.DataFrame({
                    'Predicted_Power_kW': future_predictions_values.flatten(), # Flatten if predict_future returns 2D array
                    'Timestamp': future_timestamps
                }).set_index('Timestamp')

                # Estimate cost (assuming a fixed rate, e.g., 0.25 USD/kWh)
                cost_per_kwh = 0.25 # Example rate in USD - This can be made configurable
                predictions_df['Estimated_Cost_USD'] = predictions_df['Predicted_Power_kW'] * cost_per_kwh

                status_text.text("Preparing visualizations and insights...")
                progress_bar.progress(80)

                st.success(f"Generated {prediction_hours} hours of future predictions.")

                st.write("### Future Forecast")
                # Combine actual historical data (last SEQ_LEN) and predicted data for a single plot
                # This provides context for the predictions.
                # Debug: Ensure `Global_active_power` column exists in `df`
                if 'Global_active_power' in df.columns:
                    # Get the most recent `SEQ_LEN` hours of actual data
                    combined_df_actual = df['Global_active_power'].tail(st.session_state.predictor.SEQ_LEN).to_frame()
                    combined_df_actual['Type'] = 'Actual'

                    predictions_plot_df = predictions_df.copy()
                    predictions_plot_df.rename(columns={'Predicted_Power_kW': 'Global_active_power'}, inplace=True)
                    predictions_plot_df['Type'] = 'Predicted'

                    # Concatenate actual and predicted data for plotting
                    full_plot_df = pd.concat([combined_df_actual, predictions_plot_df])

                    fig = px.line(full_plot_df, x=full_plot_df.index, y='Global_active_power', color='Type',
                                  title=f'Energy Consumption: Historical and {prediction_hours}-hour Forecast',
                                  labels={'Global_active_power': 'Power (kW)', 'index': 'Date/Time'},
                                  color_discrete_map={'Actual': 'blue', 'Predicted': 'red'})
                    fig.update_layout(hovermode="x unified", height=500,
                                      plot_bgcolor='#23272f',   # Dark background for predictions plot
                                      paper_bgcolor='#23272f', # Dark background for predictions plot
                                      font_color='#f8f8f8'     # White font color for predictions plot
                                      )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Cannot plot predictions: 'Global_active_power' column not found in original data.")
                
                st.write("### Detailed Predictions")
                st.dataframe(predictions_df)
                st.markdown(create_download_link(predictions_df, "energy_predictions.csv", "Download Predictions as CSV"), unsafe_allow_html=True)

                st.write("### Prediction Insights")
                avg_pred = predictions_df['Predicted_Power_kW'].mean()
                max_pred = predictions_df['Predicted_Power_kW'].max()
                min_pred = predictions_df['Predicted_Power_kW'].min()

                st.markdown(f"""
                <div class="insight-box">
                - Over the next **{prediction_hours} hours**, the average predicted energy consumption is **{avg_pred:.2f} kW**.
                - The peak predicted consumption is **{max_pred:.2f} kW**, and the lowest is **{min_pred:.2f} kW**.
                - Based on the current forecast and an estimated rate of ${cost_per_kwh}/kWh, the total estimated cost for the next {prediction_hours} hours is **${predictions_df['Estimated_Cost_USD'].sum():.2f}**.
                - Proactive measures during peak prediction hours can help reduce costs.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction generation: {str(e)}")
                st.error("Detailed error information:")
                st.code(traceback.format_exc())
                st.session_state.error_message = "Prediction generation failed. Ensure your model is trained and data is correct."
            finally:
                progress_bar.empty()
                status_text.empty()

elif page == "📈 Analytics":
    st.markdown('<h1 class="main-header">📈 Energy Consumption Analytics</h1>', unsafe_allow_html=True)
    st.info("Dive deeper into your energy data. Analyze trends, seasonality, and correlations to gain actionable insights.")

    if st.session_state.df is None:
        st.info("Please upload your energy consumption data first in the sidebar.")
    else:
        df = st.session_state.df.copy()

        st.write("### Data Trends Over Time")
        if 'Global_active_power' in df.columns:
            fig_trend = px.line(df, x=df.index, y='Global_active_power', title='Global Active Power Over Time',
                                labels={'Global_active_power': 'Power (kW)', 'index': 'Date/Time'})
            fig_trend.update_layout(hovermode="x unified", height=400,
                                    plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("Cannot plot trend: 'Global_active_power' column not found.")


        st.write("### Daily and Weekly Seasonality")
        # To show seasonality, data needs to be resampled to daily/weekly averages
        df_daily_avg = df['Global_active_power'].resample('D').mean().to_frame()
        df_weekly_avg = df['Global_active_power'].resample('W').mean().to_frame()

        fig_seasonality = make_subplots(rows=1, cols=2, subplot_titles=('Daily Average Power', 'Weekly Average Power'))

        fig_seasonality.add_trace(go.Scatter(x=df_daily_avg.index, y=df_daily_avg['Global_active_power'], mode='lines', name='Daily Avg', line=dict(color='lightgreen')), row=1, col=1)
        fig_seasonality.add_trace(go.Scatter(x=df_weekly_avg.index, y=df_weekly_avg['Global_active_power'], mode='lines', name='Weekly Avg', line=dict(color='orange')), row=1, col=2)

        fig_seasonality.update_layout(height=400, showlegend=False, title_text="Energy Consumption Seasonality",
                                      plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
        fig_seasonality.update_xaxes(title_text="Date", row=1, col=1)
        fig_seasonality.update_yaxes(title_text="Avg Power (kW)", row=1, col=1)
        fig_seasonality.update_xaxes(title_text="Date", row=1, col=2)
        fig_seasonality.update_yaxes(title_text="Avg Power (kW)", row=1, col=2)
        st.plotly_chart(fig_seasonality, use_container_width=True)

        st.write("### Feature Relationships")
        # Scatter plots for key features vs. Global_active_power
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'Global_active_power' in numeric_cols:
            numeric_cols.remove('Global_active_power') # Exclude target from being plotted against itself

        if len(numeric_cols) > 0:
            selected_feature = st.selectbox("Select a feature to visualize its relationship with Global Active Power:", numeric_cols)
            fig_scatter = px.scatter(df, x=selected_feature, y='Global_active_power',
                                     title=f'Global Active Power vs. {selected_feature}',
                                     labels={'Global_active_power': 'Power (kW)'})
            fig_scatter.update_layout(height=500,
                                      plot_bgcolor='#23272f', paper_bgcolor='#23272f', font_color='#f8f8f8')
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No other numeric features available to plot relationships.")


elif page == "🚨 Anomaly Detection":
    st.markdown('<h1 class="main-header">🚨 Anomaly Detection</h1>', unsafe_allow_html=True)
    st.info("Identify unusual spikes or drops in energy consumption that might indicate anomalies, faults, or unusual behavior.")

    if not st.session_state.training_completed or st.session_state.predictor is None:
        st.info("Please train or load a model first on the 'Model Training' page to enable anomaly detection.")
    elif st.session_state.df is None:
        st.info("Please upload your energy consumption data first in the sidebar.")
    else:
        df = st.session_state.df.copy()
        st.write("### Anomaly Detection Settings")
        # Allow user to adjust the anomaly threshold if the predictor supports it
        current_threshold = st.session_state.predictor.anomaly_threshold if hasattr(st.session_state.predictor, 'anomaly_threshold') else 0.5
        anomaly_threshold_input = st.slider("Anomaly Threshold (lower = more sensitive)", 0.01, 1.0, float(current_threshold), 0.01)
        # Update predictor's threshold if changed
        if hasattr(st.session_state.predictor, 'set_anomaly_threshold'):
             st.session_state.predictor.set_anomaly_threshold(anomaly_threshold_input)
        else:
            st.warning("Anomaly threshold adjustment is not supported by the current model configuration.")

        if st.button("🔍 Detect Anomalies", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                status_text.text("Detecting anomalies...")
                # Debug: Ensure `detect_anomalies` method exists in `EnergyPredictor` and returns (anomalies_boolean_array, scores_array).
                anomalies, anomaly_scores = st.session_state.predictor.detect_anomalies(df)
                progress_bar.progress(100)
                status_text.text("Anomaly detection completed!")
                st.success("Anomalies detected successfully!")

                num_anomalies = anomalies.sum()
                st.write(f"**Total Anomalies Detected:** {num_anomalies}")

                if num_anomalies > 0:
                    st.write("### Anomaly Visualization")
                    anomaly_fig = plot_anomalies(df, anomalies, anomaly_scores)
                    st.plotly_chart(anomaly_fig, use_container_width=True)

                    st.write("### Detailed Anomaly Report")
                    # Create a DataFrame for anomalies
                    anomaly_df = df[anomalies].copy()
                    anomaly_df['Anomaly_Score'] = anomaly_scores[anomalies]
                    st.dataframe(anomaly_df.sort_values('Anomaly_Score', ascending=False))
                    st.markdown(create_download_link(anomaly_df, "detected_anomalies.csv", "Download Anomaly Report as CSV"), unsafe_allow_html=True)

                    st.markdown("""
                    <div class="insight-box">
                    **Anomaly Insights:**
                    - Review the detected anomalies to understand unusual consumption patterns.
                    - High anomaly scores indicate a greater deviation from the model's expected behavior.
                    - Investigate these periods for potential equipment malfunctions, unusual usage, or data errors.
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.info("No anomalies detected in the dataset with the current threshold.")
                    st.markdown("""
                    <div class="insight-box">
                    **Anomaly Insights:**
                    - No anomalies found with the current settings. If you suspect anomalies, try lowering the 'Anomaly Threshold' slider to make the detection more sensitive.
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during anomaly detection: {str(e)}")
                st.error("Detailed error information:")
                st.code(traceback.format_exc())
                st.session_state.error_message = "Anomaly detection failed."
            finally:
                progress_bar.empty()
                status_text.empty()

elif page == "📋 Reports":
    st.markdown('<h1 class="main-header">📋 Comprehensive Reports</h1>', unsafe_allow_html=True)
    st.info("Generate and download comprehensive reports including model performance, predictions, and anomaly insights.")

    if st.session_state.df is None:
        st.info("Please upload your energy consumption data first in the sidebar.")
    elif not st.session_state.training_completed or st.session_state.predictor is None:
        st.info("Please train or load a model first on the 'Model Training' page to generate a full report.")
    else:
        st.write("### Generate Full Report")
        st.write("Click below to generate a detailed summary of your energy analytics.")

        if st.button("📝 Generate Report", use_container_width=True):
            report_status = st.empty()
            report_status.text("Generating report...")
            report_content = BytesIO() # Use BytesIO to build the report in memory

            report_content.write(b"<h1>Energy Consumption Analysis Report</h1>\n")
            report_content.write(f"<p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n".encode('utf-8'))
            report_content.write(f"<p>Data Date Range: {st.session_state.df.index.min().strftime('%Y-%m-%d')} to {st.session_state.df.index.max().strftime('%Y-%m-%d')}</p>\n".encode('utf-8'))
            report_content.write("<hr>\n".encode('utf-8'))

            # --- Dataset Overview ---
            report_content.write("<h2>1. Dataset Overview</h2>\n")
            report_content.write(f"<p>Number of entries: {len(st.session_state.df)}</p>\n")
            report_content.write(f"<p>Number of columns: {st.session_state.df.shape[1]}</p>\n")
            report_content.write("<h3>Sample Data:</h3>\n")
            report_content.write(st.session_state.df.head().to_html())
            report_content.write("<br>\n")

            # --- Key Metrics ---
            if 'Global_active_power' in st.session_state.df.columns:
                report_content.write("<h2>2. Key Energy Metrics</h2>\n")
                report_content.write(f"<p>Average Global Active Power: {st.session_state.df['Global_active_power'].mean():.2f} kW</p>\n")
                report_content.write(f"<p>Maximum Global Active Power: {st.session_state.df['Global_active_power'].max():.2f} kW</p>\n")
                report_content.write(f"<p>Minimum Global Active Power: {st.session_state.df['Global_active_power'].min():.2f} kW</p>\n")
                report_content.write("<br>\n")

            # --- Model Performance ---
            report_content.write("<h2>3. Model Performance</h2>\n")
            if st.session_state.evaluation_results:
                report_content.write("<h3>Training Set Metrics:</h3>\n")
                report_content.write(f"<pre>{json.dumps(st.session_state.evaluation_results.get('train_metrics', {}), indent=2)}</pre>\n")
                report_content.write("<h3>Validation Set Metrics:</h3>\n")
                report_content.write(f"<pre>{json.dumps(st.session_state.evaluation_results.get('validation_metrics', {}), indent=2)}</pre>\n")
            else:
                report_content.write("<p>No evaluation results available. Please train or load a model.</p>\n")
            report_content.write("<br>\n")

            # --- Predictions (if available) ---
            report_content.write("<h2>4. Future Energy Predictions</h2>\n")
            # To get predictions, we need to run the prediction logic again.
            try:
                prediction_hours_report = st.session_state.predictor.FUTURE_PERIOD_PREDICT
                future_predictions_values = st.session_state.predictor.predict_future(st.session_state.df.copy(), steps=prediction_hours_report)
                last_timestamp = st.session_state.df.index.max()
                future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, prediction_hours_report + 1)]
                predictions_df_report = pd.DataFrame({
                    'Predicted_Power_kW': future_predictions_values.flatten(),
                    'Timestamp': future_timestamps
                }).set_index('Timestamp')
                cost_per_kwh = 0.25
                predictions_df_report['Estimated_Cost_USD'] = predictions_df_report['Predicted_Power_kW'] * cost_per_kwh

                report_content.write(f"<p>Generated {prediction_hours_report} hours of future predictions.</p>\n")
                report_content.write("<h3>Predicted Values (First 5):</h3>\n")
                report_content.write(predictions_df_report.head().to_html())
                report_content.write(f"<p>Total estimated cost for the forecast period: <strong>${predictions_df_report['Estimated_Cost_USD'].sum():.2f}</strong></p>\n")
            except Exception as e:
                report_content.write(f"<p>Could not generate predictions for report: {e}</p>\n")
            report_content.write("<br>\n")


            # --- Anomaly Detection (if applicable) ---
            report_content.write("<h2>5. Anomaly Detection Summary</h2>\n")
            try:
                anomalies_report, anomaly_scores_report = st.session_state.predictor.detect_anomalies(st.session_state.df.copy())
                num_anomalies_report = anomalies_report.sum()
                report_content.write(f"<p>Total Anomalies Detected: {num_anomalies_report}</p>\n")
                if num_anomalies_report > 0:
                    anomaly_df_report = st.session_state.df[anomalies_report].copy()
                    anomaly_df_report['Anomaly_Score'] = anomaly_scores_report[anomalies_report]
                    report_content.write("<h3>Detected Anomalies (First 5, by Score):</h3>\n")
                    report_content.write(anomaly_df_report.sort_values('Anomaly_Score', ascending=False).head().to_html())
                else:
                    report_content.write("<p>No anomalies detected in the dataset.</p>\n")
            except Exception as e:
                report_content.write(f"<p>Could not perform anomaly detection for report: {e}</p>\n")
            report_content.write("<br>\n")


            # Finalize report content
            report_content.seek(0)
            report_html = report_content.read().decode('utf-8')

            st.download_button(
                label="Download Full HTML Report",
                data=report_html,
                file_name="energy_analysis_report.html",
                mime="text/html",
                use_container_width=True
            )
            report_status.success("Report generated and ready for download!")
            st.info("The report includes key metrics, model performance, and insights based on your data.")