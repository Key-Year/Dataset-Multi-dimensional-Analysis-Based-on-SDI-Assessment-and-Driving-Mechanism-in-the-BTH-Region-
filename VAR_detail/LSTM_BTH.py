"""
LSTM (Long Short-Term Memory) Neural Network Model for BTH Region Forecasting

This script implements an LSTM model to forecast 21 variables across 13 cities
in the Beijing-Tianjin-Hebei (BTH) region for the period 2023-2030, based on
historical data from 2000-2022.

The model includes:
1. Data preprocessing and normalization
2. Sequence construction for time series
3. LSTM network architecture
4. Model training with early stopping
5. Out-of-sample forecasting
6. Model validation metrics (R², MAE, RMSE)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import torch
        import torch.nn as nn
        TENSORFLOW_AVAILABLE = False
        PYTORCH_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        PYTORCH_AVAILABLE = False
        print("Warning: Neither TensorFlow nor PyTorch available. Using mock implementations.")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
BASE_PATH = Path("/cpfs01/projects-HDD/cfff-6f37069eba2e_HDD/xzh_23211020204/KeyYear/BTH")
DATA_PATH = BASE_PATH / "data_VAR_summary.xlsx"
OUTPUT_DIR = BASE_PATH / "lstm_model_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CITIES = [
    "Beijing", "Tianjin", "Shijiazhuang", "Tangshan", "Qinhuangdao",
    "Handan", "Xingtai", "Baoding", "Zhangjiakou", "Chengde",
    "Cangzhou", "Langfang", "Hengshui"
]

ATTRIBUTES = [
    "AW", "Beds", "Bus", "DEU", "DWU", "ERV", "ESV", "GCBA", "GDP",
    "GSA", "HETeach", "LFE", "PD", "PIV", "PRA", "Phys", "SIV",
    "TIV", "Teach", "UR", "URate"
]

YEARS_TRAIN = np.arange(2000, 2023)
YEARS_VALID = np.arange(2017, 2023)  # Validation period
YEARS_FORECAST = np.arange(2023, 2031)

# LSTM hyperparameters
LOOKBACK = 5  # Number of time steps to look back
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load and preprocess the BTH region data.
    
    Returns:
        DataFrame with columns: city, year, and all attribute columns
    """
    df = pd.read_excel(file_path)
    
    # Normalize column names
    df.columns = df.columns.str.strip()
    
    # Find city and year columns
    city_col = None
    year_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'city' in col_lower or 'cites' in col_lower:
            city_col = col
        if 'year' in col_lower:
            year_col = col
    
    if city_col is None or year_col is None:
        raise ValueError("Could not find city or year column in data")
    
    # Ensure year is numeric
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    df = df.dropna(subset=[year_col])
    
    # Filter to training period
    df = df[(df[year_col] >= YEARS_TRAIN[0]) & (df[year_col] <= YEARS_TRAIN[-1])]
    
    return df, city_col, year_col


def prepare_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Time series data (n_timesteps, n_features)
        lookback: Number of time steps to look back
    
    Returns:
        (X, y) where X has shape (n_samples, lookback, n_features)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: Tuple[int, int], n_outputs: int) -> object:
    """
    Build LSTM model architecture.
    
    Args:
        input_shape: (lookback, n_features)
        n_outputs: Number of output features
    
    Returns:
        Compiled Keras model
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dense(n_outputs)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Optional[object]:
    """
    Train LSTM model.
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    n_outputs = y_train.shape[1]
    
    model = build_lstm_model(input_shape, n_outputs)
    if model is None:
        return None
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model


def forecast_lstm(model: object, last_sequence: np.ndarray, 
                 steps: int, scaler: object) -> np.ndarray:
    """
    Generate multi-step forecasts using LSTM model.
    
    Args:
        model: Trained LSTM model
        last_sequence: Last lookback sequence (lookback, n_features)
        steps: Number of steps to forecast
        scaler: Scaler for inverse transformation
    
    Returns:
        Forecasted values (steps, n_features)
    """
    if model is None:
        # Mock forecast
        return np.random.randn(steps, last_sequence.shape[1]) * 0.1
    
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for _ in range(steps):
        # Reshape for model input
        X_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict next step
        next_pred = model.predict(X_input, verbose=0)
        forecasts.append(next_pred[0])
        
        # Update sequence (shift and append)
        current_sequence = np.vstack([current_sequence[1:], next_pred])
    
    forecasts = np.array(forecasts)
    
    # Inverse transform
    if scaler is not None:
        forecasts = scaler.inverse_transform(forecasts)
    
    return forecasts


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate validation metrics.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan}
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    r2 = r2_score(y_true_clean, y_pred_clean)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    return {"r2": r2, "mae": mae, "rmse": rmse}


def process_attribute_lstm(df: pd.DataFrame, attribute: str,
                           city_col: str, year_col: str) -> Dict:
    """
    Process a single attribute through LSTM pipeline.
    """
    # Aggregate across cities for this attribute
    attr_data = df.groupby(year_col)[attribute].mean().sort_index()
    
    if len(attr_data) < LOOKBACK + 1:
        return {"error": "Insufficient data"}
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(attr_data.values.reshape(-1, 1))
    
    # Create sequences
    X, y = prepare_sequences(data_scaled, LOOKBACK)
    
    if len(X) < 10:
        return {"error": "Insufficient sequences"}
    
    # Split train/validation
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    model = train_lstm_model(X_train, y_train, X_val, y_val)
    
    if model is None:
        return {"error": "Model training failed"}
    
    # Validation metrics
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_orig = scaler.inverse_transform(y_val)
    y_val_pred_orig = scaler.inverse_transform(y_val_pred)
    
    metrics = calculate_metrics(y_val_orig.flatten(), y_val_pred_orig.flatten())
    
    # Forecast
    last_sequence = data_scaled[-LOOKBACK:]
    forecast_scaled = forecast_lstm(model, last_sequence, len(YEARS_FORECAST), scaler)
    forecast = forecast_scaled.flatten()
    
    # Simple prediction intervals (based on validation RMSE)
    rmse = metrics.get("rmse", np.std(y_val_orig - y_val_pred_orig))
    z_score = 1.96
    forecast_lower = forecast - z_score * rmse
    forecast_upper = forecast + z_score * rmse
    
    return {
        "attribute": attribute,
        "metrics": metrics,
        "forecast": forecast,
        "forecast_lower": forecast_lower,
        "forecast_upper": forecast_upper,
        "model": model,
        "scaler": scaler
    }


def main():
    """
    Main execution function for LSTM model.
    """
    print("=" * 60)
    print("LSTM Model for BTH Region Forecasting")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df, city_col, year_col = load_data(DATA_PATH)
    print(f"   Loaded {len(df)} records from {df[year_col].min():.0f} to {df[year_col].max():.0f}")
    
    # Process each attribute
    print("\n2. Training LSTM models for each attribute...")
    results = {}
    all_metrics = []
    
    for attr in ATTRIBUTES:
        print(f"   Processing {attr}...")
        result = process_attribute_lstm(df, attr, city_col, year_col)
        
        if "error" in result:
            print(f"     Warning: {result['error']}")
            continue
        
        results[attr] = result
        metrics = result["metrics"].copy()
        metrics["attribute"] = attr
        all_metrics.append(metrics)
    
    # Save validation metrics
    print("\n3. Saving validation metrics...")
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = OUTPUT_DIR / "lstm_validation_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"   Saved: {metrics_path}")
        
        # Print summary
        avg_r2 = metrics_df["r2"].mean()
        avg_mae = metrics_df["mae"].mean()
        avg_rmse = metrics_df["rmse"].mean()
        print(f"   Average R²: {avg_r2:.4f}")
        print(f"   Average MAE: {avg_mae:.4f}")
        print(f"   Average RMSE: {avg_rmse:.4f}")
    
    # Generate forecasts for all city-attribute combinations
    print("\n4. Generating forecasts for all cities...")
    forecast_results = []
    
    for city in CITIES:
        city_df = df[df[city_col] == city]
        for attr in ATTRIBUTES:
            if attr not in results:
                continue
            
            # Get historical data for this city-attribute
            hist_data = city_df.groupby(year_col)[attr].first().sort_index()
            
            # Get forecast (aggregated, then adjust for city-specific level)
            base_forecast = results[attr]["forecast"]
            base_lower = results[attr]["forecast_lower"]
            base_upper = results[attr]["forecast_upper"]
            
            # Adjust for city-specific level
            city_mean = hist_data.mean() if len(hist_data) > 0 else np.nan
            overall_mean = df.groupby(year_col)[attr].mean().mean()
            if not np.isnan(city_mean) and not np.isnan(overall_mean) and overall_mean != 0:
                adjustment = city_mean / overall_mean
                base_forecast = base_forecast * adjustment
                base_lower = base_lower * adjustment
                base_upper = base_upper * adjustment
            
            for year_idx, year in enumerate(YEARS_FORECAST):
                if year_idx < len(base_forecast):
                    forecast_results.append({
                        "city": city,
                        "attribute": attr,
                        "year": year,
                        "forecast": base_forecast[year_idx],
                        "pi_lower": base_lower[year_idx],
                        "pi_upper": base_upper[year_idx],
                    })
    
    # Save forecasts
    if forecast_results:
        forecast_df = pd.DataFrame(forecast_results)
        forecast_path = OUTPUT_DIR / "lstm_forecasts_2023_2030.csv"
        forecast_df.to_csv(forecast_path, index=False)
        print(f"   Saved forecasts: {forecast_path}")
    
    print("\n" + "=" * 60)
    print("LSTM model execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

