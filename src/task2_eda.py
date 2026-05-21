#!/usr/bin/env python3
"""
Task 2: Exploratory Data Analysis
All required plots and analyses for mobile network traffic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the processed data
print("="*60)
print("TASK 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\nLoading processed data...")
df = pd.read_parquet('../data/processed/traffic_optimized.parquet')
print(f"Loaded {len(df):,} rows")
print(f"Time range: {df['Time Interval'].min()} to {df['Time Interval'].max()}")

# Create output directory for figures
import os
os.makedirs('../output/figures', exist_ok=True)

# -------------------------------------------------------------------
# I. Probability Density Function of Total Traffic per Square
# -------------------------------------------------------------------
print("\n[I. Computing PDF of total traffic per square]")

total_per_square = df.groupby('Square id')['Internet traffic activity'].sum().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with KDE
axes[0].hist(total_per_square['Internet traffic activity'], bins=50, density=True, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Total Internet Traffic (2 months)')
axes[0].set_ylabel('Density')
axes[0].set_title('PDF of Total Traffic per Square')
axes[0].grid(True, alpha=0.3)

# Log-scale to better visualize distribution
axes[1].hist(np.log1p(total_per_square['Internet traffic activity']), bins=50, density=True, alpha=0.7, edgecolor='black', color='orange')
axes[1].set_xlabel('Log(Total Internet Traffic + 1)')
axes[1].set_ylabel('Density')
axes[1].set_title('PDF (Log Scale)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/figures/01_pdf_traffic_per_square.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_pdf_traffic_per_square.png")

# Statistics
print(f"  Mean traffic: {total_per_square['Internet traffic activity'].mean():,.2f}")
print(f"  Median traffic: {total_per_square['Internet traffic activity'].median():,.2f}")
print(f"  Skewness: {total_per_square['Internet traffic activity'].skew():.2f}")

# -------------------------------------------------------------------
# II. Time Series for Three Specific Areas (First Two Weeks)
# -------------------------------------------------------------------
print("\n[II. Plotting time series for 3 specific areas]")

areas = [6169, 4159, 4556]  # highest, 4159, 4556
area_names = ['Highest Traffic (6169)', 'Square 4159', 'Square 4556']

fig, axes = plt.subplots(3, 1, figsize=(15, 10))

for idx, (area, name) in enumerate(zip(areas, area_names)):
    # Filter for the specific area and first two weeks
    area_data = df[df['Square id'] == area].copy()
    two_weeks = area_data[area_data['Time Interval'] <= '2013-11-14']
    
    axes[idx].plot(two_weeks['Time Interval'], two_weeks['Internet traffic activity'], linewidth=1)
    axes[idx].set_title(f'{name}', fontsize=12)
    axes[idx].set_ylabel('Internet Traffic Activity')
    axes[idx].grid(True, alpha=0.3)
    
    # Add statistics
    axes[idx].text(0.02, 0.95, f'Mean: {two_weeks["Internet traffic activity"].mean():.2f}', 
                   transform=axes[idx].transAxes, fontsize=9, verticalalignment='top')

axes[2].set_xlabel('Time Interval')
plt.tight_layout()
plt.savefig('../output/figures/02_time_series_three_areas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_time_series_three_areas.png")

# -------------------------------------------------------------------
# III. Stationarity Analysis (Rolling Statistics + ADF Test)
# -------------------------------------------------------------------
print("\n[III. Stationarity Analysis]")

# Use Square 6169 (highest traffic) for analysis
series = df[df['Square id'] == 6169].set_index('Time Interval')['Internet traffic activity']
series = series.resample('1H').mean().dropna()  # Resample to hourly for faster analysis

# Rolling statistics
rolling_mean = series.rolling(window=24).mean()
rolling_std = series.rolling(window=24).std()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(series, label='Original', alpha=0.7)
axes[0].plot(rolling_mean, label='24-hour Rolling Mean', color='red', linewidth=2)
axes[0].set_title('Original Series with Rolling Mean')
axes[0].set_ylabel('Internet Traffic Activity')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(rolling_std, label='24-hour Rolling Std', color='green', linewidth=2)
axes[1].set_title('Rolling Standard Deviation')
axes[1].set_ylabel('Standard Deviation')
axes[1].set_xlabel('Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/figures/03_rolling_statistics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_rolling_statistics.png")

# ADF Test
print("\n  Augmented Dickey-Fuller Test:")
adf_result = adfuller(series.dropna())
print(f"    ADF Statistic: {adf_result[0]:.4f}")
print(f"    p-value: {adf_result[1]:.4f}")
print(f"    Critical values:")
for key, value in adf_result[4].items():
    print(f"      {key}: {value:.4f}")
    
if adf_result[1] < 0.05:
    print("    ✓ Series is STATIONARY (reject H0)")
else:
    print("    ✗ Series is NON-STATIONARY (fail to reject H0)")

# -------------------------------------------------------------------
# IV. Time Series Decomposition
# -------------------------------------------------------------------
print("\n[IV. Time Series Decomposition]")

# Use 10-minute data for decomposition (need at least 2 periods)
decomp_series = df[df['Square id'] == 6169].set_index('Time Interval')['Internet traffic activity']
decomp_series = decomp_series[:144*3]  # 3 days for faster computation

try:
    decomposition = seasonal_decompose(decomp_series, model='additive', period=144)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    axes[0].plot(decomposition.observed)
    axes[0].set_title('Observed')
    axes[0].set_ylabel('Traffic')
    
    axes[1].plot(decomposition.trend)
    axes[1].set_title('Trend')
    axes[1].set_ylabel('Traffic')
    
    axes[2].plot(decomposition.seasonal)
    axes[2].set_title('Seasonal (Daily Pattern)')
    axes[2].set_ylabel('Traffic')
    
    axes[3].plot(decomposition.resid)
    axes[3].set_title('Residual')
    axes[3].set_ylabel('Traffic')
    axes[3].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('../output/figures/04_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: 04_decomposition.png")
except Exception as e:
    print(f"  Decomposition skipped: {e}")

# -------------------------------------------------------------------
# V. Autocorrelation (ACF) and Partial Autocorrelation (PACF)
# -------------------------------------------------------------------
print("\n[V. ACF and PACF Plots]")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot ACF
plot_acf(series.dropna(), lags=48, ax=axes[0], alpha=0.05)
axes[0].set_title('Autocorrelation Function (ACF) - 48 lags')
axes[0].set_xlabel('Lags')
axes[0].set_ylabel('Autocorrelation')
axes[0].grid(True, alpha=0.3)

# Plot PACF
plot_pacf(series.dropna(), lags=48, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF) - 48 lags')
axes[1].set_xlabel('Lags')
axes[1].set_ylabel('Partial Autocorrelation')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/figures/05_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_acf_pacf.png")

# -------------------------------------------------------------------
# VI. Spatial Analysis (Heatmap of Traffic Intensity)
# -------------------------------------------------------------------
print("\n[VI. Spatial Analysis - Heatmap]")

# Create 100x100 grid
grid_size = 100
traffic_grid = np.zeros((grid_size, grid_size))

# Map square ids to grid coordinates (assuming row-major order)
total_by_square = df.groupby('Square id')['Internet traffic activity'].sum().reset_index()
for _, row in total_by_square.iterrows():
    square_id = int(row['Square id'])
    if square_id < grid_size * grid_size:
        row_idx = square_id // grid_size
        col_idx = square_id % grid_size
        traffic_grid[row_idx, col_idx] = row['Internet traffic activity']

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(traffic_grid, cmap='hot', interpolation='nearest', aspect='auto')
ax.set_title('Spatial Distribution of Mobile Network Traffic\n(100×100 Grid)', fontsize=14)
ax.set_xlabel('Column Index')
ax.set_ylabel('Row Index')
plt.colorbar(im, ax=ax, label='Total Internet Traffic (2 months)')
plt.tight_layout()
plt.savefig('../output/figures/06_spatial_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_spatial_heatmap.png")

# Identify hotspots
hotspots = total_by_square.nlargest(10, 'Internet traffic activity')
print("\n  Top 10 traffic hotspots:")
for _, row in hotspots.iterrows():
    print(f"    Square {int(row['Square id'])}: {row['Internet traffic activity']:,.2f} units")

# -------------------------------------------------------------------
# VII. Anomaly Detection
# -------------------------------------------------------------------
print("\n[VII. Anomaly Detection]")

# Use IQR method on hourly aggregated data
hourly_series = series.copy()
Q1 = hourly_series.quantile(0.25)
Q3 = hourly_series.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies = hourly_series[(hourly_series < lower_bound) | (hourly_series > upper_bound)]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(hourly_series.index, hourly_series.values, label='Hourly Traffic', alpha=0.7)
ax.scatter(anomalies.index, anomalies.values, color='red', s=50, label=f'Anomalies ({len(anomalies)})', zorder=5)
ax.axhline(y=upper_bound, color='orange', linestyle='--', label='Upper Bound')
ax.axhline(y=lower_bound, color='orange', linestyle='--', label='Lower Bound')
ax.set_title('Anomaly Detection in Mobile Network Traffic (IQR Method)')
ax.set_xlabel('Time')
ax.set_ylabel('Internet Traffic Activity')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../output/figures/07_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 07_anomaly_detection.png")

print(f"\n  Detected {len(anomalies)} anomalous hours")
if len(anomalies) > 0:
    print(f"  Largest anomaly: {anomalies.max():.2f} at {anomalies.idxmax()}")

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
print("\n" + "="*60)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("="*60)
print("\nOutput figures saved to: ../output/figures/")
print("  01_pdf_traffic_per_square.png")
print("  02_time_series_three_areas.png")
print("  03_rolling_statistics.png")
print("  04_decomposition.png")
print("  05_acf_pacf.png")
print("  06_spatial_heatmap.png")
print("  07_anomaly_detection.png")