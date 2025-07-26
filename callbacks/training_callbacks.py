# Training callbacks
# training_callbacks.py

from dash import Input, Output, State, callback
import pandas as pd
from model_utils import EnergyPredictor

@callback(
    Output("training-status", "children"),
    Output("training-results", "children"),
    Input("train-button", "n_clicks"),
    State("uploaded-data-store", "data"),
    State("seq-len-slider", "value"),
    State("horizon-slider", "value"),
    State("batch-size", "value"),
    State("epochs-slider", "value"),
    prevent_initial_call=True
)
def train_model(n_clicks, json_data, seq_len, horizon, batch_size, epochs):
    if json_data is None:
        return "❌ Please upload a dataset first.", None

    try:
        df = pd.read_json(json_data, convert_dates=True)
        df.index = pd.to_datetime(df.index)

        predictor = EnergyPredictor(
            seq_len=seq_len,
            future_period_predict=horizon,
            batch_size=batch_size,
            epochs=epochs
        )

        df_processed = predictor.preprocess_data(df.copy())
        predictor.train(df_processed)
        evaluation = predictor.evaluate_model(df.copy())

        # Optional: save the model to reuse
        predictor.save_complete_model("models/energy_model")

        return "✅ Model trained and saved successfully!", [
            html.H4("Training Metrics"),
            html.Pre(str(evaluation['train_metrics'])),
            html.H4("Validation Metrics"),
            html.Pre(str(evaluation['validation_metrics']))
        ]

    except Exception as e:
        return f"❌ Error during training: {str(e)}", None
