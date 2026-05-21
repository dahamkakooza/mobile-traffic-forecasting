import pandas as pd
import glob

# Check a few different files
files_to_check = [
    "../data/raw/sms-call-internet-mi-2013-11-01.csv",
    "../data/raw/sms-call-internet-mi-2013-11-06.csv", 
    "../data/raw/sms-call-internet-mi-2013-12-01.csv",
    "../data/raw/sms-call-internet-mi-2013-12-25.csv"
]

for file in files_to_check:
    try:
        print(f"\n{'='*50}")
        print(f"File: {file.split('/')[-1]}")
        print('='*50)
        
        # Read first row
        df = pd.read_csv(file, nrows=1)
        print(f"Columns: {df.columns.tolist()}")
        print(f"First row data:")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")
            
    except Exception as e:
        print(f"Error reading {file}: {e}")