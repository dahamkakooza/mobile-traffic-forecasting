#!/usr/bin/env python3
"""
Task 2: Exploratory Data Analysis
All 7 required analyses + supplementary figures
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load processed data
print("="*60)
print("TASK 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

df = pd.read_parquet('../data/processed/traffic_optimized.parquet')
print(f"Loaded {len(df):,} rows")
print(f"Time range: {df['Time Interval'].min()} to {df['Time Interval'].max()}")

# Create output directory
import os
os.makedirs('../output/figures', exist_ok=True)

# Get highest traffic square
total_per_square = df.groupby('Square id')['Internet traffic activity'].sum().reset_index()
total_per_square = total_per_square.sort_values('Internet traffic activity', ascending=False)
highest_square = int(total_per_square.iloc[0]['Square id'])
print(f"Highest traffic square: {highest_square}")

# ============================================================
# Figure 1: PDF of Total Traffic per Square
# ============================================================
print("\n[I. Generating PDF plot]")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
axes[0].hist(total_per_square['Internet traffic activity'], bins=50, density=True, 
             alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Total Internet Traffic (2 months)')
axes[0].set_ylabel('Density')
axes[0].set_title('PDF of Total Traffic per Square')
axes[0].grid(True, alpha=0.3)

# Log scale
axes[1].hist(np.log1p(total_per_square['Internet traffic activity']), bins=50, 
             density=True, alpha=0.7, edgecolor='black', color='orange')
axes[1].set_xlabel('Log(Total Internet Traffic + 1)')
axes[1].set_ylabel('Density')
axes[1].set_title('PDF (Log Scale)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/figures/01_pdf_traffic_per_square.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_pdf_traffic_per_square.png")

# ============================================================
# Figure 2: Time Series for Three Areas (First 14 Available Days)
# ============================================================
print("\n[II. Generating time series plot]")

areas = [highest_square, 4159, 4556]
area_names = [f'Highest Traffic ({highest_square})', 'Square 4159', 'Square 4556']

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, (area, name) in enumerate(zip(areas, area_names)):
    area_data = df[df['Square id'] == area].copy()
    area_data = area_data.sort_values('Time Interval')
    
    # Take first 14 days of available data
    unique_days = area_data['Time Interval'].dt.date.unique()
    first_14_days = unique_days[:14]
    two_weeks = area_data[area_data['Time Interval'].dt.date.isin(first_14_days)]
    
    print(f"  {name}: {len(two_weeks)} rows, traffic max={two_weeks['Internet traffic activity'].max():.2f}")
    
    axes[idx].plot(two_weeks['Time Interval'], two_weeks['Internet traffic activity'], linewidth=1.5)
    axes[idx].set_title(name, fontsize=12)
    axes[idx].set_ylabel('Internet Traffic Activity')
    axes[idx].grid(True, alpha=0.3)

axes[2].set_xlabel('Time Interval')
plt.tight_layout()
plt.savefig('../output/figures/02_time_series_three_areas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_time_series_three_areas.png")

# ============================================================
# Figure 3: Rolling Statistics + ADF Test
# ============================================================
print("\n[III. Generating stationarity plots]")

series = df[df['Square id'] == highest_square].set_index('Time Interval')['Internet traffic activity']
series = series.resample('H').mean().dropna()

rolling_mean = series.rolling(window=24).mean()
rolling_std = series.rolling(window=24).std()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(series, label='Original', alpha=0.7)
axes[0].plot(rolling_mean, label='24-hour Rolling Mean', color='red', linewidth=2)
axes[0].set_title('Original Series with Rolling Mean')
axes[0].set_ylabel('Internet Traffic')
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

adf_result = adfuller(series.dropna())
print(f"  ADF Statistic: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.4f}")
print(f"  Critical values: {adf_result[4]}")
print("  ✓ Saved: 03_rolling_statistics.png")

# ============================================================
# Figure 4: Decomposition
# ============================================================
print("\n[IV. Generating decomposition plot]")

decomp_series = df[df['Square id'] == highest_square].set_index('Time Interval')['Internet traffic activity']
decomp_series = decomp_series[:144*3]

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

# ============================================================
# Figure 5: ACF and PACF
# ============================================================
print("\n[V. Generating ACF/PACF plots]")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(series.dropna(), lags=72, ax=axes[0], alpha=0.05)
axes[0].set_title('Autocorrelation Function (ACF) - 72 lags')
axes[0].set_xlabel('Lags (hours)')
axes[0].grid(True, alpha=0.3)

plot_pacf(series.dropna(), lags=72, ax=axes[1], alpha=0.05, method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF) - 72 lags')
axes[1].set_xlabel('Lags (hours)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../output/figures/05_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_acf_pacf.png")

# ============================================================
# Figure 6: Spatial Heatmap
# ============================================================
print("\n[VI. Generating spatial heatmap]")

grid_size = 100
traffic_grid = np.zeros((grid_size, grid_size))

for _, row in total_per_square.iterrows():
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

print("\nTop 10 Traffic Hotspots:")
print(total_per_square.head(10).to_string(index=False))

# ============================================================
# Figure 7: Anomaly Detection
# ============================================================
print("\n[VII. Generating anomaly detection plot]")

hourly_series = series.copy()
Q1 = hourly_series.quantile(0.25)
Q3 = hourly_series.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies = hourly_series[(hourly_series < lower_bound) | (hourly_series > upper_bound)]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(hourly_series.index, hourly_series.values, label='Hourly Traffic', alpha=0.7)
ax.scatter(anomalies.index, anomalies.values, color='red', s=50, 
           label=f'Anomalies ({len(anomalies)})', zorder=5)
ax.axhline(y=upper_bound, color='orange', linestyle='--', label='Upper Bound')
ax.axhline(y=lower_bound, color='orange', linestyle='--', label='Lower Bound')
ax.set_title('Anomaly Detection in Mobile Network Traffic')
ax.set_xlabel('Time')
ax.set_ylabel('Internet Traffic')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../output/figures/07_anomaly_detection.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 07_anomaly_detection.png")
print(f"  Detected {len(anomalies)} anomalies")
print(f"  Largest anomaly: {anomalies.max():.2f} at {anomalies.idxmax()}")

# ============================================================
# Supplementary Figures (Bonus)
# ============================================================
print("\n[Generating supplementary figures]")

# Figure 8: Hourly Profile
hourly_profile = df.copy()
hourly_profile['Hour'] = hourly_profile['Time Interval'].dt.hour
hourly_avg = hourly_profile.groupby('Hour')['Internet traffic activity'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Average Traffic', fontsize=12)
ax.set_title('Average Hourly Traffic Pattern', fontsize=14)
ax.set_xticks(range(0, 24, 2))
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../output/figures/08_hourly_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 08_hourly_profile.png")

# Figure 9: Weekday vs Weekend
df_temp = df.copy()
df_temp['Hour'] = df_temp['Time Interval'].dt.hour
df_temp['IsWeekend'] = df_temp['Time Interval'].dt.dayofweek >= 5

weekday_profile = df_temp[~df_temp['IsWeekend']].groupby('Hour')['Internet traffic activity'].mean()
weekend_profile = df_temp[df_temp['IsWeekend']].groupby('Hour')['Internet traffic activity'].mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(weekday_profile.index, weekday_profile.values, label='Weekday', linewidth=2, color='blue')
ax.plot(weekend_profile.index, weekend_profile.values, label='Weekend', linewidth=2, color='red', linestyle='--')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Average Traffic', fontsize=12)
ax.set_title('Weekday vs Weekend Traffic Patterns', fontsize=14)
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../output/figures/09_weekday_vs_weekend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 09_weekday_vs_weekend.png")

# Figure 10: Top 5 Squares Time Series
top_5_squares = total_per_square.head(5)['Square id'].tolist()
fig, ax = plt.subplots(figsize=(14, 8))

for square_id in top_5_squares:
    square_data = df[df['Square id'] == square_id].set_index('Time Interval')['Internet traffic activity']
    square_data = square_data.resample('D').sum()
    ax.plot(square_data.index, square_data.values, label=f'Square {square_id}', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Daily Total Internet Traffic', fontsize=12)
ax.set_title('Daily Traffic Patterns - Top 5 Squares', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../output/figures/10_top5_squares_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 10_top5_squares_timeseries.png")

# ============================================================
# Summary
# ============================================================
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
print("  08_hourly_profile.png (bonus)")
print("  09_weekday_vs_weekend.png (bonus)")
print("  10_top5_squares_timeseries.png (bonus)")
