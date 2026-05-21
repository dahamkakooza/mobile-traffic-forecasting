#!/usr/bin/env python3
"""
Task 1: Complete Data Handling for ALL Files
Handles both November (old format) and December (new format)
"""

import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm

def process_november_old_format(file_path):
    """Process November files with old format (datetime, CellID, countrycode, internet)"""
    df = pd.read_csv(file_path)
    
    # Filter for Italy only
    if 'countrycode' in df.columns:
        df = df[df['countrycode'] == 0]
    
    # Extract needed columns
    result = pd.DataFrame({
        'Square id': df['CellID'].astype('uint16'),
        'Internet traffic activity': df['internet'].fillna(0).astype('float32'),
        'Time Interval': pd.to_datetime(df['datetime'])
    })
    
    return result

def process_december_new_format(file_path):
    """Process December files with new format (SquareID, Timestamp, InternetTraffic)"""
    df = pd.read_csv(file_path)
    
    # Convert timestamp (milliseconds) to datetime
    # Note: The timestamps are in milliseconds since epoch
    result = pd.DataFrame({
        'Square id': df['SquareID'].astype('uint16'),
        'Internet traffic activity': df['InternetTraffic'].astype('float32'),
        'Time Interval': pd.to_datetime(df['Timestamp'], unit='ms')
    })
    
    return result

def load_all_data():
    """Load and combine all data files"""
    print("="*60)
    print("TASK 1: DATA HANDLING & MEMORY MANAGEMENT")
    print("="*60)
    
    # Find all November files (old format)
    nov_files = sorted(glob.glob("../data/raw/sms-call-internet-mi-2013-11-*.csv"))
    nov_files = [f for f in nov_files if "_parsed" not in f]
    
    # Find all December files (new format)
    dec_files = sorted(glob.glob("../data/raw/sms-call-internet-mi-2013-12-*.csv"))
    dec_files = [f for f in dec_files if "_parsed" not in f]
    
    print(f"Found {len(nov_files)} November files")
    print(f"Found {len(dec_files)} December files")
    
    all_dfs = []
    
    # Process November files (old format)
    print("\nProcessing November files...")
    for file in tqdm(nov_files, desc="November"):
        try:
            df = process_november_old_format(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not process {os.path.basename(file)}: {e}")
    
    # Process December files (new format)
    print("\nProcessing December files...")
    for file in tqdm(dec_files, desc="December"):
        try:
            df = process_december_new_format(file)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not process {os.path.basename(file)}: {e}")
    
    if not all_dfs:
        print("ERROR: No data loaded!")
        return None
    
    # Combine all data
    print("\nCombining all data...")
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by time
    full_df = full_df.sort_values('Time Interval')
    full_df = full_df.reset_index(drop=True)
    
    # Remove any duplicates
    full_df = full_df.drop_duplicates(subset=['Time Interval', 'Square id'])
    
    # Data quality report
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    print(f"Total rows: {len(full_df):,}")
    print(f"Unique squares: {full_df['Square id'].nunique():,}")
    print(f"Time range: {full_df['Time Interval'].min()} to {full_df['Time Interval'].max()}")
    print(f"Total days: {(full_df['Time Interval'].max() - full_df['Time Interval'].min()).days + 1}")
    
    # Check for missing values
    missing = full_df.isnull().sum()
    print(f"\nMissing values:")
    print(f"  Square id: {missing['Square id']}")
    print(f"  Traffic: {missing['Internet traffic activity']}")
    
    # Memory usage
    mem_usage = full_df.memory_usage(deep=True).sum() / (1024**3)
    print(f"\nMemory usage: {mem_usage:.2f} GB")
    
    # Optimize memory further
    full_df['Square id'] = full_df['Square id'].astype('category')
    mem_after = full_df.memory_usage(deep=True).sum() / (1024**3)
    print(f"Memory after category optimization: {mem_after:.2f} GB")
    print(f"Memory saved: {mem_usage - mem_after:.2f} GB ({(1 - mem_after/mem_usage)*100:.1f}% reduction)")
    
    return full_df

def compute_statistics(df):
    """Compute required statistics for Task 2"""
    print("\n" + "="*50)
    print("COMPUTING STATISTICS")
    print("="*50)
    
    # Total traffic per square
    total_per_square = df.groupby('Square id')['Internet traffic activity'].sum().reset_index()
    total_per_square = total_per_square.sort_values('Internet traffic activity', ascending=False)
    total_per_square.columns = ['Square id', 'Total traffic (2 months)']
    
    # Top squares
    print("\nTop 5 squares by total traffic:")
    print(total_per_square.head(5).to_string(index=False))
    
    highest_square = int(total_per_square.iloc[0]['Square id'])
    print(f"\n✓ Square with highest total traffic: {highest_square}")
    
    # Check required squares
    print("\nRequired squares for Task 2:")
    for sq in [4159, 4556]:
        val = total_per_square[total_per_square['Square id'] == sq]['Total traffic (2 months)'].values
        if len(val) > 0:
            print(f"  ✓ Square {sq}: {val[0]:,.2f} units")
        else:
            print(f"  ✗ Square {sq}: NOT FOUND in dataset")
    
    # Save to CSV
    os.makedirs("../data/processed", exist_ok=True)
    total_per_square.to_csv("../data/processed/total_per_square.csv", index=False)
    print(f"\nSaved total_per_square.csv")
    
    return total_per_square, highest_square

def save_data(df):
    """Save processed data"""
    os.makedirs("../data/processed", exist_ok=True)
    
    # Save full dataset as Parquet
    output_path = "../data/processed/traffic_optimized.parquet"
    df.to_parquet(output_path, index=False, compression='snappy')
    
    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"\n✓ Saved full dataset: {output_path}")
    print(f"  File size: {file_size:.2f} GB")
    
    # Also save a sample for quick testing (1% of data)
    sample_df = df.sample(frac=0.01, random_state=42)
    sample_path = "../data/processed/traffic_sample.parquet"
    sample_df.to_parquet(sample_path, index=False)
    print(f"✓ Saved sample (1%): {sample_path}")

def main():
    # Load all data
    df = load_all_data()
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Compute statistics
    totals, highest_square = compute_statistics(df)
    
    # Save processed data
    save_data(df)
    
    # Verify data for Task 2
    print("\n" + "="*50)
    print("VERIFICATION FOR TASK 2")
    print("="*50)
    
    # Check if we have data for the required squares
    squares_needed = [highest_square, 4159, 4556]
    squares_available = set(df['Square id'].unique())
    
    print(f"Square with highest traffic ({highest_square}): {'✓' if highest_square in squares_available else '✗'}")
    print(f"Square 4159: {'✓' if 4159 in squares_available else '✗'}")
    print(f"Square 4556: {'✓' if 4556 in squares_available else '✗'}")
    
    # Show sample data for first required square
    sample_data = df[df['Square id'] == highest_square].head(10)
    if len(sample_data) > 0:
        print(f"\nSample data for Square {highest_square}:")
        print(sample_data.to_string(index=False))
    
    print("\n" + "="*50)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("="*50)
    print("\nOutput files ready for Task 2:")
    print("  • data/processed/traffic_optimized.parquet")
    print("  • data/processed/total_per_square.csv")
    print("  • data/processed/traffic_sample.parquet")
    
    return df

if __name__ == "__main__":
    df = main()