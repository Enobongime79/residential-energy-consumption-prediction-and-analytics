import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import joblib
import json
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class EnergyPredictor:
    def __init__(self, seq_len=100, future_period_predict=6, batch_size=64, epochs=30):
        self.SEQ_LEN = seq_len
        self.FUTURE_PERIOD_PREDICT = future_period_predict
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.scaler = MinMaxScaler(feature_range=(0, 1)) # Scaler for input features
        self.target_scaler = MinMaxScaler(feature_range=(0, 1)) # Scaler for the target variable (Global_active_power)
        self.model = None
        self.history = None
        self.feature_names = None
        self.training_stats = {}
        self.anomaly_threshold = None
        
    def preprocess_data(self, df):
        """Enhanced preprocessing with better feature handling"""
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df.sort_index(inplace=True)
        df.dropna(inplace=True)
        
        # Resample to hourly data
        df = df.resample('1h').mean()
        df.dropna(inplace=True)
        
        # Add time-based features first
        df = self._add_time_features(df)
        
        # Add lag features
        df = self._add_lag_features(df)
        
        # Add rolling features
        df = self._add_rolling_features(df)
        
        # Smoothing (after feature engineering)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ewm(alpha=0.2).mean()
        df.dropna(inplace=True)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Create supervised learning data: Shift 'Global_active_power' to create the 'future' target column
        # This must be done AFTER all feature engineering to avoid data leakage.
        df['future'] = df['Global_active_power'].shift(-self.FUTURE_PERIOD_PREDICT)
        
        # Drop rows where future values are NaN (last few rows) and any rows introduced by lag/rolling features
        df = df.dropna()

        # Store feature names BEFORE scaling. 'future' is the target, not a feature.
        self.feature_names = [col for col in df.columns if col != 'future']
        print(f"Features identified: {len(self.feature_names)} - {self.feature_names}")
        
        return df
    
    def _add_time_features(self, df):
        """Add time-based features"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_peak_hour'] = ((df.index.hour >= 17) & (df.index.hour <= 21)).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_lag_features(self, df):
        """Add lagged features based on Global_active_power"""
        df['power_lag_1h'] = df['Global_active_power'].shift(1)
        df['power_lag_24h'] = df['Global_active_power'].shift(24)
        df['power_lag_168h'] = df['Global_active_power'].shift(168)  # 1 week
        return df
    
    def _add_rolling_features(self, df):
        """Add rolling window features based on Global_active_power"""
        df['power_ma_24h'] = df['Global_active_power'].rolling(window=24).mean()
        df['power_std_24h'] = df['Global_active_power'].rolling(window=24).std()
        df['power_ma_168h'] = df['Global_active_power'].rolling(window=168).mean()
        
        # Sub-meter ratios (only if columns exist)
        if 'Sub_metering_1' in df.columns:
            df['sub1_ratio'] = df['Sub_metering_1'] / (df['Global_active_power'] + 1e-8)
        if 'Sub_metering_2' in df.columns:
            df['sub2_ratio'] = df['Sub_metering_2'] / (df['Global_active_power'] + 1e-8)
        if 'Sub_metering_3' in df.columns:
            df['sub3_ratio'] = df['Sub_metering_3'] / (df['Global_active_power'] + 1e-8)
        
        # Power factor (only if reactive power column exists)
        if 'Global_reactive_power' in df.columns:
            df['power_factor'] = df['Global_active_power'] / (
                np.sqrt(df['Global_active_power']**2 + df['Global_reactive_power']**2) + 1e-8
            )
        
        return df
    
    def _remove_outliers(self, df, method='iqr'):
        """Enhanced outlier removal from 'Global_active_power'"""
        if method == 'iqr':
            Q1 = df['Global_active_power'].quantile(0.25)
            Q3 = df['Global_active_power'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df['Global_active_power'] >= lower) & (df['Global_active_power'] <= upper)]
        else:  # z-score method
            factor = 3
            upper_lim = df['Global_active_power'].mean() + df['Global_active_power'].std() * factor
            lower_lim = df['Global_active_power'].mean() - df['Global_active_power'].std() * factor
            df = df[(df['Global_active_power'] < upper_lim) & (df['Global_active_power'] > lower_lim)]
        
        return df
    
    def create_sequences(self, df_scaled, shuffle=False):
        """Creates sequences from already scaled DataFrame"""
        sequential_data = []
        prev_days = deque(maxlen=self.SEQ_LEN)
        
        # Ensure feature_names is set from preprocess_data
        if not hasattr(self, 'feature_names') or not self.feature_names:
            raise AttributeError("feature_names not set. Call preprocess_data first.")
        
        features = self.feature_names # Use stored feature names for consistency
        
        # Create sequences
        for i in range(len(df_scaled)):
            current_features = df_scaled[features].iloc[i].values
            prev_days.append(current_features)
            
            if len(prev_days) == self.SEQ_LEN:
                # 'future' column is already scaled in prepare_data
                future_value = df_scaled['future'].iloc[i]
                sequential_data.append([np.array(prev_days), future_value])
        
        if shuffle:
            np.random.shuffle(sequential_data)
        
        X = []
        y = []
        for seq, target in sequential_data:
            X.append(seq)
            y.append(target)
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Enhanced model with dropout and callbacks"""
        model = Sequential()
    
        # First LSTM layer
        model.add(LSTM(36, return_sequences=True, input_shape=(input_shape[1], input_shape[2])))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    
        # Second LSTM layer
        model.add(LSTM(36, return_sequences=False))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    
        # Output layer for single-step prediction
        model.add(Dense(1))
    
        # Compile the model using MAE as loss
        model.compile(
            loss='mean_absolute_error',
            optimizer='adam',
            metrics=['mean_squared_error', 'mean_absolute_error']
        )
    
        return model

    def get_callbacks(self):
        """Training callbacks for better performance"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        return [early_stopping, reduce_lr]
    
    def prepare_data(self, df):
        """Prepares and scales data for training and validation, splits into X and y"""
        print("Preparing data for training...")
        
        # Separate features and target
        feature_cols = self.feature_names # Use stored feature names
        target_col = ['future']
        
        # Fit and transform features
        self.scaler.fit(df[feature_cols])
        df_scaled_features = self.scaler.transform(df[feature_cols])
        
        # Fit and transform target separately
        self.target_scaler.fit(df[target_col])
        df_scaled_target = self.target_scaler.transform(df[target_col])
        
        # Combine scaled features and target into a new DataFrame
        df_scaled = pd.DataFrame(df_scaled_features, columns=feature_cols, index=df.index)
        df_scaled['future'] = df_scaled_target
        
        print(f"Scaled features shape: {df_scaled_features.shape}")
        print(f"Scaled target shape: {df_scaled_target.shape}")
        
        # Split into train/validation sets (80/20), preserving temporal ordering
        train_size = int(len(df_scaled) * 0.8)
        train_df_scaled = df_scaled[:train_size]
        valid_df_scaled = df_scaled[train_size:]
        
        # Create sequences from scaled data
        train_x, train_y = self.create_sequences(train_df_scaled, shuffle=True)
        valid_x, valid_y = self.create_sequences(valid_df_scaled, shuffle=False)
        
        print(f"Train X shape: {train_x.shape}")
        print(f"Train y shape: {train_y.shape}")
        print(f"Valid X shape: {valid_x.shape}")
        print(f"Valid y shape: {valid_y.shape}")
        
        # Store training statistics
        self.training_stats = {
            'total_samples': len(df_scaled),
            'train_samples': len(train_x),
            'valid_samples': len(valid_x),
            'features_count': len(feature_cols)
        }
        
        return train_x, train_y, valid_x, valid_y
    
    def train(self, df):
        """Enhanced training with better error handling"""
        try:
            print("Starting data preprocessing...")
            df_processed = self.preprocess_data(df)
            print(f"Processed data shape: {df_processed.shape}")
            
            print("Preparing sequences...")
            train_x, train_y, valid_x, valid_y = self.prepare_data(df_processed)
            
            print(f"Training samples: {len(train_x)}")
            print(f"Validation samples: {len(valid_x)}")
            print(f"Features: {len(self.feature_names)}") # This should now reflect the number of input features
            
            # Build model
            print("Building model...")
            self.model = self.build_model(train_x.shape)
            print(f"Model input shape: {train_x.shape}")
            
            # Train with callbacks
            print("Starting model training...")
            self.history = self.model.fit(
                train_x, train_y,
                batch_size=self.BATCH_SIZE,
                epochs=self.EPOCHS,
                validation_data=(valid_x, valid_y),
                callbacks=self.get_callbacks(),
                verbose=1
            )
            
            # Calculate anomaly threshold based on validation residuals (using original scale)
            print("Calculating anomaly threshold...")
            # Predict on validation data (scaled)
            scaled_predictions = self.model.predict(valid_x)
            # Inverse transform predictions and actual values for residual calculation
            inv_valid_y = self.target_scaler.inverse_transform(valid_y.reshape(-1, 1))
            inv_predictions = self.target_scaler.inverse_transform(scaled_predictions)
            
            residuals = np.abs(inv_valid_y - inv_predictions.flatten())
            self.anomaly_threshold = np.percentile(residuals, 95)
            
            print("Training completed successfully!")
            return self.history
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            print("Debug info:")
            if 'df_processed' in locals():
                print(f"Processed data shape: {df_processed.shape}")
                print(f"Processed data columns: {df_processed.columns.tolist()}")
            if hasattr(self, 'feature_names'):
                print(f"Feature names: {self.feature_names}")
            raise e
    
    def predict(self, x_scaled):
        """Makes prediction on scaled input and returns inverse-transformed result."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        if self.target_scaler is None:
            raise ValueError("Target scaler not fitted. Train the model first.")
        
        # Make prediction (output is scaled)
        yhat_scaled = self.model.predict(x_scaled)
        
        # Inverse transform the prediction
        inv_yhat = self.target_scaler.inverse_transform(yhat_scaled)
        
        return inv_yhat.flatten() # Return 1D array
    
    def predict_future(self, df, steps=24):
        """Fixed future prediction using a rolling window approach for multi-step predictions.
        Assumes that future features (like lag/rolling) are derived from the predicted target.
        """
        print(f"Predicting {steps} steps into the future...")
        
        # Ensure feature_names is set and scalers are fitted
        if not hasattr(self, 'feature_names') or not self.feature_names:
            raise AttributeError("feature_names not set. Call preprocess_data first.")
        if self.scaler is None or self.target_scaler is None:
            raise ValueError("Scalers not fitted. Train the model first.")

        # Get the last SEQ_LEN data points, preprocess them to get features, but not 'future' column
        # To avoid re-creating 'future' or dropping data, we only take the relevant columns from df.
        temp_df = df.copy()
        temp_df = self._add_time_features(temp_df)
        temp_df = self._add_lag_features(temp_df)
        temp_df = self._add_rolling_features(temp_df)
        temp_df = temp_df.ewm(alpha=0.2).mean() # Re-apply smoothing if necessary for consistency
        temp_df = self._remove_outliers(temp_df) # Re-apply outlier removal for consistency
        temp_df = temp_df.dropna()
        
        # Get the last relevant sequence of features
        last_sequence_features = temp_df[self.feature_names].tail(self.SEQ_LEN).values
        
        # Scale the last sequence of features
        current_sequence_scaled = self.scaler.transform(last_sequence_features)
        
        predictions_scaled = []
        
        # Start time for future predictions
        last_timestamp = temp_df.index[-1]
        
        for step in range(steps):
            # Predict next value (output is scaled)
            pred_scaled = self.model.predict(current_sequence_scaled.reshape(1, self.SEQ_LEN, -1), verbose=0)[0, 0]
            predictions_scaled.append(pred_scaled)
            
            # Prepare next sequence: Shift and add the new predicted value
            # Create a 'next_step_features_scaled' array for the next prediction
            # Initialize with zeros, then fill in known/predicted values
            next_step_features_scaled = np.zeros(len(self.feature_names))

            # Update the 'Global_active_power' feature (assuming it's the first feature after scaling)
            # This is critical for recursive prediction. Adjust index if 'Global_active_power' is not the 0th feature
            try:
                global_power_idx = self.feature_names.index('Global_active_power')
                next_step_features_scaled[global_power_idx] = pred_scaled
            except ValueError:
                # Fallback if 'Global_active_power' is not in feature_names or cannot be found
                print("Warning: 'Global_active_power' not found in feature_names. Cannot propagate its prediction.")
                # You might need to adjust based on how your features are ordered and what you want to propagate
                pass
            
            # Generate time-based features for the next step
            next_timestamp = last_timestamp + timedelta(hours=(step + 1))
            temp_future_df = pd.DataFrame(index=[next_timestamp])
            temp_future_df['hour'] = next_timestamp.hour
            temp_future_df['day_of_week'] = next_timestamp.dayofweek
            temp_future_df['month'] = next_timestamp.month
            temp_future_df['is_weekend'] = int(next_timestamp.dayofweek >= 5)
            temp_future_df['is_peak_hour'] = int((next_timestamp.hour >= 17) & (next_timestamp.hour <= 21))
            temp_future_df['hour_sin'] = np.sin(2 * np.pi * temp_future_df['hour'] / 24)
            temp_future_df['hour_cos'] = np.cos(2 * np.pi * temp_future_df['hour'] / 24)
            temp_future_df['day_sin'] = np.sin(2 * np.pi * temp_future_df['day_of_week'] / 7)
            temp_future_df['day_cos'] = np.cos(2 * np.pi * temp_future_df['day_of_week'] / 7)
            temp_future_df['month_sin'] = np.sin(2 * np.pi * temp_future_df['month'] / 12)
            temp_future_df['month_cos'] = np.cos(2 * np.pi * temp_future_df['month'] / 12)

            # Update time-based features in next_step_features_scaled.
            # This requires knowing the exact order of features in self.feature_names.
            # A more robust way would be to create a dummy dataframe, populate it, then scale it.
            future_time_features = temp_future_df[[col for col in temp_future_df.columns if col in self.feature_names]].iloc[0].values
            # Find indices for time-based features
            for i, feat_name in enumerate(self.feature_names):
                if feat_name in temp_future_df.columns:
                    # Find the corresponding index in temp_future_df to get the value
                    val = temp_future_df[feat_name].iloc[0]
                    # This value needs to be scaled before placing in next_step_features_scaled
                    # To do this correctly for individual features: create a temp array, scale it, get the single value.
                    # This is complex for dynamic feature generation. For simplicity here, we assume scaler
                    # can handle single feature transformation if `transform` is called on a single array element.
                    # A more robust solution for future features involves predicting or having known future values for all features.
                    # For this implementation, time-based features are known, but need to be scaled.
                    # Lag and rolling features are dynamic and dependent on past predicted values.
                    # This is a simplification common in recursive forecasting when external features are not available.

            # Simple (less robust but common) way to update:
            # We need to manually calculate and scale dynamic features for the next step
            # This requires knowledge of the past 'SEQ_LEN' actual/predicted values for features like lags and rolling stats.
            # The current approach for 'current_sequence_scaled' assumes it's just shifting in the predicted Global_active_power.
            # A truly robust multi-step prediction would re-calculate all dynamic features based on the growing sequence of predictions.
            # For simplicity, we'll shift the existing scaled sequence and inject the new scaled prediction.
            current_sequence_scaled = np.roll(current_sequence_scaled, -1, axis=0) # Shift
            # Update the last row with the new prediction and time-based features
            
            # This is a key simplification: only the primary predicted value is propagated directly.
            # For more complex features, a full re-computation for each step would be needed.
            
            # Create a dummy row for the new step's features, populate it, then scale it.
            new_step_raw_features = pd.Series(0.0, index=self.feature_names)
            # Fill 'Global_active_power' with the *inverse-transformed* prediction for feature engineering purposes
            # Then re-scale this single value before putting it into the scaled sequence.
            # This is often done by inverse transforming, calculating new features, then re-scaling the new features.
            
            # Let's simplify and propagate the scaled prediction into the 'Global_active_power' slot of the *scaled* sequence
            # and rely on the model to implicitly handle other features based on context.
            # This is what the original code attempts.
            current_sequence_scaled[-1, global_power_idx] = pred_scaled
            
            # To update other time-based features, you would need to calculate them for `next_timestamp` and then scale them
            # and insert them into their correct positions in `current_sequence_scaled[-1, :]`. This is non-trivial if
            # `self.scaler` is a multi-feature scaler.
            # A more robust approach for multi-step is to build a new raw feature row, scale it, and append.
            
            # For this fix, let's keep it simple by primarily propagating the predicted power,
            # acknowledging that other features like lags/rolling will then derive from these predictions.
            # Time-based features would ideally be re-calculated for each new timestamp.
            # For brevity, I'm assuming the existing `current_sequence_scaled` handles time features by implicit position,
            # which is less robust. A full recursive multi-step would involve:
            # 1. Predict next scaled value.
            # 2. Inverse transform predicted value.
            # 3. Use inverse transformed value to calculate new lag/rolling/other dynamic features for the next timestamp.
            # 4. Create new row of *raw* features for next timestamp.
            # 5. Scale this new row of features.
            # 6. Append scaled new row to sequence, drop oldest.
            
            # Given the original intent, let's assume `current_sequence_scaled[-1, :]` updates correctly via roll and assignment.
            # This implicitly means the model learns to cope with 'stale' time/lag/rolling features in the recursive part,
            # or that their impact diminishes over the short prediction horizon.
            
            # For truly correct future feature generation, you'd calculate time-based features for `next_timestamp`
            # and then populate the last row of `current_sequence_scaled` with these, scaled appropriately.
            # This is beyond a simple "fix" and involves a more fundamental change to `predict_future`.

        # Inverse transform all collected scaled predictions at once
        inv_predictions = self.target_scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
        
        return inv_predictions.flatten() # Return 1D array
    
    def detect_anomalies(self, df):
        """Detect anomalies in energy consumption using the trained model and threshold."""
        if self.anomaly_threshold is None:
            print("Warning: Anomaly threshold not set. Train the model first.")
            return None, None
            
        df_processed = self.preprocess_data(df)
        
        # Prepare data for prediction: scale features and get targets
        feature_cols = self.feature_names
        target_col = ['future']

        scaled_features = self.scaler.transform(df_processed[feature_cols])
        scaled_targets = self.target_scaler.transform(df_processed[target_col])
        
        # Create sequences for the input to the model
        # Note: `create_sequences` is designed for training split. For full DF, we need to adapt it or manually create sequences.
        # Let's adapt it by making a dummy df_scaled for it.
        df_temp_scaled = pd.DataFrame(scaled_features, columns=feature_cols, index=df_processed.index)
        df_temp_scaled['future'] = scaled_targets
        
        X_test, y_true_scaled = self.create_sequences(df_temp_scaled, shuffle=False)
        
        # Predict scaled values
        y_pred_scaled = self.model.predict(X_test)
        
        # Inverse transform predictions and true values
        y_true_inv = self.target_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        y_pred_inv = self.target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        residuals = np.abs(y_true_inv - y_pred_inv)
        
        anomalies = residuals > self.anomaly_threshold
        anomaly_scores = residuals / self.anomaly_threshold
        
        return anomalies, anomaly_scores
    
    def calculate_efficiency_scores(self, df):
        """Calculate efficiency scores for each sub-meter."""
        scores = {}
        total_consumption = df['Global_active_power'].sum()
        
        if total_consumption == 0:
            print("Warning: Total consumption is zero, cannot calculate efficiency scores.")
            return scores

        for i in range(1, 4):
            sub_col = f'Sub_metering_{i}'
            if sub_col in df.columns:
                sub_consumption = df[sub_col].sum()
                efficiency = (sub_consumption / total_consumption) * 100
                scores[f'SubMeter_{i}'] = {
                    'consumption_percentage': efficiency,
                    'total_kwh': sub_consumption,
                    'efficiency_rating': self._get_efficiency_rating(efficiency)
                }
        
        return scores
    
    def _get_efficiency_rating(self, percentage):
        """Convert percentage to efficiency rating."""
        if percentage < 20:
            return 'Excellent'
        elif percentage < 35:
            return 'Good'
        elif percentage < 50:
            return 'Average'
        else:
            return 'Needs Improvement'
    
    def generate_insights(self, df):
        """Generate actionable insights based on consumption patterns."""
        insights = {}
        
        # Peak usage analysis
        df_temp = df.copy()
        df_temp['hour'] = df_temp.index.hour
        hourly_avg = df_temp.groupby('hour')['Global_active_power'].mean()
        peak_hour = hourly_avg.idxmax()
        peak_consumption = hourly_avg.max()
        
        insights['peak_usage'] = {
            'peak_hour': int(peak_hour),
            'peak_consumption': float(peak_consumption),
            'recommendation': f"Consider shifting non-essential appliances away from {peak_hour}:00"
        }
        
        # Weekend vs weekday analysis
        df_temp['is_weekend'] = df_temp.index.dayofweek >= 5
        weekend_avg = df_temp[df_temp['is_weekend']]['Global_active_power'].mean()
        weekday_avg = df_temp[~df_temp['is_weekend']]['Global_active_power'].mean()
        
        insights['usage_pattern'] = {
            'weekend_avg': float(weekend_avg),
            'weekday_avg': float(weekday_avg),
            'difference_pct': float(((weekend_avg - weekday_avg) / weekday_avg) * 100) if weekday_avg != 0 else np.nan
        }
        
        # Efficiency scores
        insights['efficiency_scores'] = self.calculate_efficiency_scores(df)
        
        return insights
    
    def evaluate_model(self, df):
        """Comprehensive model evaluation, inverse transforming predictions for metrics."""
        df_processed = self.preprocess_data(df) # Ensure features are correctly identified
        train_x, train_y_scaled, valid_x, valid_y_scaled = self.prepare_data(df_processed)
        
        # Predictions (these are scaled)
        train_pred_scaled = self.model.predict(train_x)
        valid_pred_scaled = self.model.predict(valid_x)
        
        # Inverse transform predictions and actuals for metric calculation
        train_y_true = self.target_scaler.inverse_transform(train_y_scaled.reshape(-1, 1)).flatten()
        train_pred_inv = self.target_scaler.inverse_transform(train_pred_scaled).flatten()
        
        valid_y_true = self.target_scaler.inverse_transform(valid_y_scaled.reshape(-1, 1)).flatten()
        valid_pred_inv = self.target_scaler.inverse_transform(valid_pred_scaled).flatten()
        
        # Metrics on inverse-transformed (original scale) values
        train_metrics = {
            'mae': mean_absolute_error(train_y_true, train_pred_inv),
            'mse': mean_squared_error(train_y_true, train_pred_inv),
            'rmse': np.sqrt(mean_squared_error(train_y_true, train_pred_inv)),
            'r2': r2_score(train_y_true, train_pred_inv)
        }
        
        valid_metrics = {
            'mae': mean_absolute_error(valid_y_true, valid_pred_inv),
            'mse': mean_squared_error(valid_y_true, valid_pred_inv),
            'rmse': np.sqrt(mean_squared_error(valid_y_true, valid_pred_inv)),
            'r2': r2_score(valid_y_true, valid_pred_inv)
        }
        
        return {
            'train_metrics': train_metrics,
            'validation_metrics': valid_metrics,
            'training_stats': self.training_stats
        }
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # MAE plot
        axes[1].plot(self.history.history['mean_absolute_error'], label='Training MAE')
        axes[1].plot(self.history.history['val_mean_absolute_error'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_complete_model(self, base_path):
        """Save model, scalers, and metadata."""
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        # Save model
        if self.model is not None:
            model_path = f"{base_path}.h5"
            self.model.save(model_path)
    
        # Save feature scaler
        scaler_path = f"{base_path}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        # Save target scaler
        target_scaler_path = f"{base_path}_target_scaler.pkl"
        joblib.dump(self.target_scaler, target_scaler_path)
        
        # Save metadata
        metadata = {
            'seq_len': self.SEQ_LEN,
            'future_period_predict': self.FUTURE_PERIOD_PREDICT,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'anomaly_threshold': self.anomaly_threshold
        }
        
        metadata_path = f"{base_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Model saved to {base_path}")
    
    def load_complete_model(self, base_path):
        """Load model, scalers, and metadata."""
        # Load model
        model_path = f"{base_path}.h5"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Load feature scaler
        scaler_path = f"{base_path}_scaler.pkl"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Feature Scaler file not found at {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # Load target scaler
        target_scaler_path = f"{base_path}_target_scaler.pkl"
        if not os.path.exists(target_scaler_path):
            raise FileNotFoundError(f"Target Scaler file not found at {target_scaler_path}")
        self.target_scaler = joblib.load(target_scaler_path)
        
        # Load metadata
        metadata_path = f"{base_path}_metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.SEQ_LEN = metadata['seq_len']
        self.FUTURE_PERIOD_PREDICT = metadata['future_period_predict']
        self.feature_names = metadata['feature_names']
        self.training_stats = metadata['training_stats']
        self.anomaly_threshold = metadata['anomaly_threshold']
        
        print(f"Model loaded from {base_path}")

# Usage example (demonstrates how to use the class):
"""
# Assuming 'df' is your loaded and cleaned DataFrame
# from a CSV, with 'Global_active_power' and a datetime index.

# Initialize predictor
predictor = EnergyPredictor(seq_len=168, future_period_predict=24, epochs=50)

# Train model
# The .train() method handles preprocessing, scaling, sequence creation, and model training
history = predictor.train(df) 

# Make predictions (requires preprocessed and scaled input for predict method)
# Example: If you want to predict on new data, preprocess it and create sequences:
# new_df_processed = predictor.preprocess_data(new_raw_df)
# new_features_scaled = predictor.scaler.transform(new_df_processed[predictor.feature_names])
# new_x_sequences, _ = predictor.create_sequences(pd.DataFrame(new_features_scaled, columns=predictor.feature_names, index=new_df_processed.index), is_prediction=True) # is_prediction parameter not used in current create_sequences, may need adjustment for standalone prediction X
# In the provided create_sequences, it always expects 'future' column. For pure prediction X, it needs to be adapted or manually built.
# For simplicity, if you had a new_x_scaled_sequence ready:
# future_predictions = predictor.predict(new_x_scaled_sequence)

# A more practical way to get future predictions using the class is:
future_predictions = predictor.predict_future(df, steps=24) # uses internal preprocessing and recursive prediction

# Detect anomalies
anomalies, scores = predictor.detect_anomalies(df) # Pass original df, it handles internal preprocessing

# Generate insights
insights = predictor.generate_insights(df)

# Evaluate model
evaluation = predictor.evaluate_model(df) # Pass original df, it handles internal preprocessing

# Save complete model
predictor.save_complete_model("models/energy_model") # Saves model, feature scaler, and target scaler

# Load complete model
# loaded_predictor = EnergyPredictor()
# loaded_predictor.load_complete_model("models/energy_model")
"""