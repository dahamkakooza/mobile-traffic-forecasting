#!/usr/bin/env python3
"""
Task 3: Forecasting Models (No TensorFlow - Using XGBoost)
Models: SARIMA, Holt-Winters, XGBoost
For squares: 6169 (highest traffic), 4159, 4556
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

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

# Squares to forecast (from Task 2 results)
SQUARES = [6169, 4159, 4556]
SQUARE_NAMES = ['Highest Traffic (6169)', 'Square 4159', 'Square 4556']

TEST_START = '2013-12-16'
TEST_END = '2013-12-22'

def create_features(df, window=24):
    """Create lag features for XGBoost"""
    df = df.copy()
    for i in range(1, window + 1):
        df[f'lag_{i}'] = df['traffic'].shift(i)
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['day'] = df.index.day
    df['month'] = df.index.month
    return df.dropna()

def evaluate(y_true, y_pred):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return mae, rmse, mape

# Store all results
all_results = {}

for square, square_name in zip(SQUARES, SQUARE_NAMES):
    print(f"\n{'='*50}")
    print(f"Processing {square_name}")
    print('='*50)
    
    # Get time series for this square (hourly aggregation)
    square_data = df[df['Square id'] == square].set_index('Time Interval')['Internet traffic activity']
    square_data = square_data.resample('1H').mean().dropna()
    
    # Split into train and test
    train = square_data[square_data.index < TEST_START]
    test = square_data[(square_data.index >= TEST_START) & (square_data.index <= TEST_END)]
    
    print(f"Train: {len(train)} hours, Test: {len(test)} hours")
    print(f"Train period: {train.index[0]} to {train.index[-1]}")
    print(f"Test period: {test.index[0]} to {test.index[-1]}")
    
    # -----------------------------------------------------------------
    # Model 1: ARIMA (Statistical)
    # -----------------------------------------------------------------
    print("\n[Model 1: ARIMA]")
    start_time = time.time()
    
    try:
        model = ARIMA(train, order=(5, 1, 5))
        model_fit = model.fit()
        arima_pred = model_fit.forecast(steps=len(test))
        arima_time = time.time() - start_time
        
        arima_mae, arima_rmse, arima_mape = evaluate(test.values, arima_pred)
        print(f"  MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}, MAPE: {arima_mape:.1f}%")
        print(f"  Time: {arima_time:.2f}s")
    except Exception as e:
        print(f"  ARIMA failed: {e}")
        arima_pred = np.zeros(len(test))
        arima_time = 0
        arima_mae, arima_rmse, arima_mape = 0, 0, 0
    
    # -----------------------------------------------------------------
    # Model 2: Holt-Winters Exponential Smoothing
    # -----------------------------------------------------------------
    print("\n[Model 2: Holt-Winters]")
    start_time = time.time()
    
    try:
        hw_model = ExponentialSmoothing(train, seasonal_periods=24, trend='add', seasonal='add')
        hw_fit = hw_model.fit()
        hw_pred = hw_fit.forecast(steps=len(test))
        hw_time = time.time() - start_time
        
        hw_mae, hw_rmse, hw_mape = evaluate(test.values, hw_pred)
        print(f"  MAE: {hw_mae:.2f}, RMSE: {hw_rmse:.2f}, MAPE: {hw_mape:.1f}%")
        print(f"  Time: {hw_time:.2f}s")
    except Exception as e:
        print(f"  Holt-Winters failed: {e}")
        hw_pred = np.zeros(len(test))
        hw_time = 0
        hw_mae, hw_rmse, hw_mape = 0, 0, 0
    
    # -----------------------------------------------------------------
    # Model 3: XGBoost (Gradient Boosting)
    # -----------------------------------------------------------------
    print("\n[Model 3: XGBoost]")
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    
    train_df = pd.DataFrame({'traffic': train_scaled}, index=train.index)
    train_feat = create_features(train_df)
    
    if len(train_feat) > 0:
        X_train = train_feat.drop('traffic', axis=1)
        y_train = train_feat['traffic']
        
        start_time = time.time()
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train, y_train)
        xgb_train_time = time.time() - start_time
        
        start_time = time.time()
        xgb_pred_scaled = []
        history = train_scaled.tolist()
        
        for i in range(len(test)):
            features = []
            for lag in range(1, 25):
                if len(history) >= lag:
                    features.append(history[-lag])
                else:
                    features.append(0)
            
            test_time = test.index[i]
            features.append(test_time.hour)
            features.append(test_time.dayofweek)
            features.append(test_time.day)
            features.append(test_time.month)
            
            pred = xgb_model.predict([features])[0]
            xgb_pred_scaled.append(pred)
            history.append(pred)
        
        xgb_infer_time = time.time() - start_time
        xgb_pred = scaler.inverse_transform(np.array(xgb_pred_scaled).reshape(-1, 1)).flatten()
        xgb_mae, xgb_rmse, xgb_mape = evaluate(test.values, xgb_pred)
        
        print(f"  MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}, MAPE: {xgb_mape:.1f}%")
        print(f"  Train time: {xgb_train_time:.2f}s, Inference: {xgb_infer_time:.2f}s")
    else:
        print("  XGBoost failed: insufficient training data")
        xgb_pred = np.zeros(len(test))
        xgb_mae, xgb_rmse, xgb_mape = 0, 0, 0
        xgb_train_time, xgb_infer_time = 0, 0
    
    # Store results
    all_results[square] = {
        'name': square_name,
        'test': test,
        'test_values': test.values,
        'test_dates': test.index,
        'arima_pred': arima_pred,
        'hw_pred': hw_pred,
        'xgb_pred': xgb_pred,
        'metrics': {
            'ARIMA': (arima_mae, arima_rmse, arima_mape),
            'Holt-Winters': (hw_mae, hw_rmse, hw_mape),
            'XGBoost': (xgb_mae, xgb_rmse, xgb_mape)
        },
        'times': {
            'ARIMA': arima_time,
            'Holt-Winters': hw_time,
            'XGBoost_train': xgb_train_time,
            'XGBoost_infer': xgb_infer_time
        }
    }
    
    # Generate plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    models_info = [
        ('ARIMA', all_results[square]['arima_pred'], 'blue'),
        ('Holt-Winters', all_results[square]['hw_pred'], 'green'),
        ('XGBoost', all_results[square]['xgb_pred'], 'red')
    ]
    
    for idx, (model_name, predictions, color) in enumerate(models_info):
        axes[idx].plot(test.index, test.values, label='Actual', color='black', linewidth=1.5)
        axes[idx].plot(test.index, predictions, label=f'{model_name} Predicted', color=color, linewidth=1.5, alpha=0.8)
        axes[idx].set_title(f'{square_name} - {model_name} Forecast (Dec 16-22, 2013)', fontsize=12)
        axes[idx].set_ylabel('Internet Traffic Activity')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)
        
        mae, rmse, mape = all_results[square]['metrics'][model_name]
        axes[idx].text(0.02, 0.95, f'MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.1f}%',
                       transform=axes[idx].transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[2].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig(f'../output/figures/task3/forecast_square_{square}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Saved plot: forecast_square_{square}.png")

# Results summary table
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

results_list = []
for square in SQUARES:
    for model in ['ARIMA', 'Holt-Winters', 'XGBoost']:
        mae, rmse, mape = all_results[square]['metrics'][model]
        time_val = all_results[square]['times'].get(model, 0)
        if model == 'XGBoost':
            time_val = all_results[square]['times'].get('XGBoost_train', 0)
        
        results_list.append({
            'Square': square,
            'Square Name': all_results[square]['name'],
            'Model': model,
            'MAE': f'{mae:.2f}',
            'RMSE': f'{rmse:.2f}',
            'MAPE': f'{mape:.1f}%',
            'Time (seconds)': f'{time_val:.2f}'
        })

results_df = pd.DataFrame(results_list)
print("\n" + results_df.to_string(index=False))
results_df.to_csv('../output/tables/model_performance.csv', index=False)
print("\n✓ Saved performance table: ../output/tables/model_performance.csv")

# ============================================================
# FAILURE ANALYSIS (Fixed version)
# ============================================================
print("\n" + "="*60)
print("FAILURE ANALYSIS")
print("="*60)

# Look for the anomaly at Dec 16 15:00 (identified in Task 2)
for square in SQUARES:
    test_dates = all_results[square]['test_dates']
    test_values = all_results[square]['test_values']
    
    # Find the index for Dec 16 around 15:00
    target_idx = None
    for i, date in enumerate(test_dates):
        if date.strftime('%Y-%m-%d %H') == '2013-12-16 15':
            target_idx = i
            break
    
    if target_idx is not None:
        actual = test_values[target_idx]
        print(f"\nSquare {square} ({all_results[square]['name']}) at {test_dates[target_idx]}:")
        print(f"  Actual traffic: {actual:.2f}")
        
        # Get predictions (convert to numpy array if needed)
        arima_pred = all_results[square]['arima_pred']
        hw_pred = all_results[square]['hw_pred']
        xgb_pred = all_results[square]['xgb_pred']
        
        # Handle if predictions are Series or array
        if hasattr(arima_pred, 'iloc'):
            arima_val = arima_pred.iloc[target_idx]
        else:
            arima_val = arima_pred[target_idx] if len(arima_pred) > target_idx else 0
            
        if hasattr(hw_pred, 'iloc'):
            hw_val = hw_pred.iloc[target_idx]
        else:
            hw_val = hw_pred[target_idx] if len(hw_pred) > target_idx else 0
            
        if hasattr(xgb_pred, 'iloc'):
            xgb_val = xgb_pred.iloc[target_idx]
        else:
            xgb_val = xgb_pred[target_idx] if len(xgb_pred) > target_idx else 0
        
        for model_name, pred_val in [('ARIMA', arima_val), ('Holt-Winters', hw_val), ('XGBoost', xgb_val)]:
            error = abs(actual - pred_val)
            pct_error = (error / actual) * 100 if actual > 0 else 0
            print(f"  {model_name}: Predicted={pred_val:.2f}, Error={error:.2f} ({pct_error:.1f}%)")
    else:
        print(f"\nSquare {square}: No data point found for 2013-12-16 15:00")

print("\n" + "="*60)
print("TASK 3 COMPLETED SUCCESSFULLY")
print("="*60)
print("\nOutput files created:")
print("  - 3 prediction plots: ../output/figures/task3/forecast_square_*.png")
print("  - Performance table: ../output/tables/model_performance.csv")