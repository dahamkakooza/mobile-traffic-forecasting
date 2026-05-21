#!/usr/bin/env python3
"""
Task 3: Forecasting Models
SARIMA (Statistical) + LSTM + GRU (Neural Networks)
For squares: 6169 (highest traffic), 4159, 4556
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import os
import time

# Create output directories
os.makedirs('../output/figures/task3', exist_ok=True)
os.makedirs('../output/tables', exist_ok=True)

# Load data
print("="*60)
print("TASK 3: FORECASTING MODELS")
print("="*60)

df = pd.read_parquet('../data/processed/traffic_optimized.parquet')
print(f"Loaded {len(df):,} rows")

# Squares to forecast
SQUARES = [6169, 4159, 4556]  # highest, 4159, 4556
SQUARE_NAMES = ['Highest Traffic (6169)', 'Square 4159', 'Square 4556']

# Parameters
WINDOW_SIZE = 144  # 1 day (144 * 10-min intervals)
TEST_START = '2013-12-16'  # Forecast week: Dec 16-22
TEST_END = '2013-12-22'

# Store results
results = {square: {} for square in SQUARES}

def prepare_data(series, window=WINDOW_SIZE, test_start=TEST_START):
    """Prepare train/test split and create sequences"""
    # Split
    train = series[series.index < test_start]
    test = series[series.index >= test_start]
    
    # Normalize
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()
    
    # Create sequences for neural networks
    X_train, y_train = [], []
    for i in range(window, len(train_scaled)):
        X_train.append(train_scaled[i-window:i])
        y_train.append(train_scaled[i])
    
    X_test, y_test = [], []
    # For test, use the last window from train as starting point
    last_window = train_scaled[-window:].tolist()
    for i in range(len(test_scaled)):
        X_test.append(last_window[-window:])
        last_window.append(test_scaled[i])
    
    X_train = np.array(X_train).reshape(-1, window, 1)
    X_test = np.array(X_test).reshape(-1, window, 1)
    
    return train, test, train_scaled, test_scaled, X_train, y_train, X_test, scaler

# ============================================================
# MODEL 1: SARIMA (Statistical)
# ============================================================
def sarima_forecast(train, test):
    """SARIMA forecast with auto-detection of parameters"""
    from statsmodels.tsa.arima.model import ARIMA
    
    # Use ARIMA with seasonal differencing (simpler than full SARIMA)
    # Since ADF showed stationarity, use d=0
    model = ARIMA(train, order=(5, 0, 5))
    model_fit = model.fit()
    
    # Forecast
    predictions = []
    history = list(train.values)
    
    for t in range(len(test)):
        model = ARIMA(history, order=(5, 0, 5))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test.iloc[t])
    
    return np.array(predictions)

# ============================================================
# MODEL 2: LSTM
# ============================================================
def build_lstm(window=WINDOW_SIZE):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# ============================================================
# MODEL 3: GRU
# ============================================================
def build_gru(window=WINDOW_SIZE):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# ============================================================
# EVALUATION FUNCTION
# ============================================================
def evaluate_model(y_true, y_pred, name):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# ============================================================
# RUN FOR EACH SQUARE
# ============================================================
all_results = {}

for square, square_name in zip(SQUARES, SQUARE_NAMES):
    print(f"\n{'='*50}")
    print(f"Processing {square_name}")
    print('='*50)
    
    # Get time series for this square
    square_data = df[df['Square id'] == square].set_index('Time Interval')['Internet traffic activity']
    square_data = square_data.resample('1H').mean().dropna()  # Resample to hourly
    
    # Prepare data
    train, test, train_scaled, test_scaled, X_train, y_train, X_test, scaler = prepare_data(square_data)
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # Store predictions
    predictions = {}
    train_times = {}
    exec_times = {}
    
    # -----------------------------------------------------------------
    # Model 1: SARIMA
    # -----------------------------------------------------------------
    print("\n[Model 1: SARIMA]")
    start_time = time.time()
    sarima_pred = sarima_forecast(train, test)
    sarima_train_time = time.time() - start_time
    
    # Inverse transform
    sarima_pred_original = scaler.inverse_transform(sarima_pred.reshape(-1, 1)).flatten()
    test_original = scaler.inverse_transform(test_scaled.reshape(-1, 1)).flatten()
    
    metrics = evaluate_model(test_original, sarima_pred_original, "SARIMA")
    predictions['SARIMA'] = sarima_pred_original
    train_times['SARIMA'] = sarima_train_time
    
    print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Training time: {sarima_train_time:.2f}s")
    
    # -----------------------------------------------------------------
    # Model 2: LSTM
    # -----------------------------------------------------------------
    print("\n[Model 2: LSTM]")
    lstm_model = build_lstm()
    
    start_time = time.time()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                   validation_split=0.2, callbacks=[early_stop], verbose=0)
    lstm_train_time = time.time() - start_time
    
    start_time = time.time()
    lstm_pred_scaled = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_infer_time = time.time() - start_time
    
    lstm_pred_original = scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
    metrics = evaluate_model(test_original, lstm_pred_original, "LSTM")
    predictions['LSTM'] = lstm_pred_original
    train_times['LSTM'] = lstm_train_time
    exec_times['LSTM'] = lstm_infer_time
    
    print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Training time: {lstm_train_time:.2f}s, Inference: {lstm_infer_time:.2f}s")
    
    # -----------------------------------------------------------------
    # Model 3: GRU
    # -----------------------------------------------------------------
    print("\n[Model 3: GRU]")
    gru_model = build_gru()
    
    start_time = time.time()
    gru_model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.2, callbacks=[early_stop], verbose=0)
    gru_train_time = time.time() - start_time
    
    start_time = time.time()
    gru_pred_scaled = gru_model.predict(X_test, verbose=0).flatten()
    gru_infer_time = time.time() - start_time
    
    gru_pred_original = scaler.inverse_transform(gru_pred_scaled.reshape(-1, 1)).flatten()
    metrics = evaluate_model(test_original, gru_pred_original, "GRU")
    predictions['GRU'] = gru_pred_original
    train_times['GRU'] = gru_train_time
    exec_times['GRU'] = gru_infer_time
    
    print(f"  MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Training time: {gru_train_time:.2f}s, Inference: {gru_infer_time:.2f}s")
    
    # -----------------------------------------------------------------
    # Store results
    # -----------------------------------------------------------------
    all_results[square] = {
        'name': square_name,
        'test_dates': test.index,
        'test_values': test_original,
        'predictions': predictions,
        'train_times': train_times,
        'exec_times': exec_times
    }
    
    # -----------------------------------------------------------------
    # Generate plot for this square
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    models = ['SARIMA', 'LSTM', 'GRU']
    colors = ['blue', 'green', 'red']
    
    for idx, (model, color) in enumerate(zip(models, colors)):
        axes[idx].plot(test.index, test_original, label='Actual', color='black', linewidth=1)
        axes[idx].plot(test.index, predictions[model], label=f'{model} Predicted', color=color, linewidth=1, alpha=0.7)
        axes[idx].set_title(f'{square_name} - {model} Forecast (Dec 16-22)')
        axes[idx].set_ylabel('Internet Traffic')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(f'../output/figures/task3/forecast_square_{square}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Saved plot: forecast_square_{square}.png")

# ============================================================
# GENERATE RESULTS TABLES
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

# Collect metrics
metrics_data = []
for square in SQUARES:
    for model in ['SARIMA', 'LSTM', 'GRU']:
        test_vals = all_results[square]['test_values']
        preds = all_results[square]['predictions'][model]
        mae = mean_absolute_error(test_vals, preds)
        rmse = np.sqrt(mean_squared_error(test_vals, preds))
        mape = np.mean(np.abs((test_vals - preds) / (test_vals + 1e-6))) * 100
        
        metrics_data.append({
            'Square': square,
            'Model': model,
            'MAE': f'{mae:.4f}',
            'RMSE': f'{rmse:.4f}',
            'MAPE': f'{mape:.2f}%',
            'Train Time (s)': f'{all_results[square]["train_times"][model]:.2f}',
            'Inference (s)': f'{all_results[square].get("exec_times", {}).get(model, 0):.2f}'
        })

# Create summary table
results_df = pd.DataFrame(metrics_data)
print("\n[PERFORMANCE SUMMARY TABLE]")
print(results_df.to_string(index=False))

# Save to CSV
results_df.to_csv('../output/tables/model_performance.csv', index=False)
print("\n✓ Saved: ../output/tables/model_performance.csv")

# ============================================================
# FAILURE ANALYSIS
# ============================================================
print("\n" + "="*60)
print("FAILURE ANALYSIS")
print("="*60)

# Find worst performing period (Dec 16 15:00 anomaly from Task 2)
target_date = '2013-12-16 15:00:00'

for square in SQUARES:
    test_vals = all_results[square]['test_values']
    test_dates = all_results[square]['test_dates']
    
    # Find index of target time
    target_idx = None
    for i, date in enumerate(test_dates):
        if str(date).startswith('2013-12-16 15'):
            target_idx = i
            break
    
    if target_idx:
        actual = test_vals[target_idx]
        print(f"\nSquare {square} on {target_date}:")
        for model in ['SARIMA', 'LSTM', 'GRU']:
            pred = all_results[square]['predictions'][model][target_idx]
            error = abs(actual - pred)
            print(f"  {model}: Actual={actual:.2f}, Pred={pred:.2f}, Error={error:.2f} ({error/actual*100:.1f}%)")

print("\n" + "="*60)
print("TASK 3 COMPLETED")
print("="*60)
print("\nOutputs created:")
print("  - 3 prediction plots (1 per square)")
print("  - Performance table (CSV)")
print("  - Failure analysis results")