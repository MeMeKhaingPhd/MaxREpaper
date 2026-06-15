
import pandas as pd
import numpy as np
import os
import json
#configuration for workload of data centers
INPUT_FILENAME = 'C:\\Users\\Me Me Khaing\\Downloads\\borg_traces_data.csv\\borg_traces_data.csv' 
OUTPUT_FILENAME = 'C:\\Users\\Me Me Khaing\\Downloads\\borg_traces_data.csv\\preprocessed_workload_hourly.csv'


# First 1. we load the Data

print(f"--- Step 1: Loading data from {INPUT_FILENAME} ---")

if not os.path.exists(INPUT_FILENAME):
    print(f"\nERROR: Input file not found: {INPUT_FILENAME}")
else:
    # this is entire CSV loading. We need to parse columns after loading.
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Successfully loaded {len(df):,} rows.")

    # 2. Parseing the JSON-like String Columns 
    print("\n--- Step 2: Parsing CPU usage from string columns ---")

    # Defining a safe parsing function
    def parse_cpu_usage(s):
        try:
            # this one is to clean the string for valid JSON and parse
            s_clean = s.replace("'", '"')
            data = json.loads(s_clean)
            return data.get('cpus', 0.0)
        except (json.JSONDecodeError, TypeError, AttributeError):
            # If parsing fails, return NaN #just for notes
            return np.nan

    # Applying this function to the average usage column
    df['cpu_usage'] = df['average_usage'].apply(parse_cpu_usage)
    
    # Also parse start_time, coercing errors to NaN
    df['start_time_numeric'] = pd.to_numeric(df['start_time'], errors='coerce')

    # 3. Clean the Data and Handle Timestamps 
    
    print("\nStep 3: Cleaning data and converting timestamps ")

    # Drop rows where parsing failed for either CPU or start_time
    rows_before_drop = len(df)
    df.dropna(subset=['cpu_usage', 'start_time_numeric'], inplace=True)
    print(f"Dropped {rows_before_drop - len(df):,} rows with corrupted/missing data.")
    
    # The timestamps in this trace are in MICROSECONDS ('us'), not nanoseconds.
    print("Converting numeric timestamps to datetime objects using MICROSECOND ('us') unit...")
    df['timestamp'] = pd.to_datetime(df['start_time_numeric'], unit='us')
    
    # Set the new correct timestamp as the index for resampling
    df.set_index('timestamp', inplace=True)
    
    # Diagnostic: Check the now-correct time range
    min_time = df.index.min()
    max_time = df.index.max()
    print(f"--> Data time range is now correctly identified: from {min_time} to {max_time}")


    # 4. Resample to Hourly Workload 
   
    print("\nStep 4: Resampling to hourly workload ")
    
    # Sum the CPU usage of all tasks that started within a given hour
    workload_hourly = df[['cpu_usage']].resample('h').sum()
    
    # Fill any hours with no activity with 0
    workload_hourly.fillna(0, inplace=True)
    
    workload_hourly.rename(columns={'cpu_usage': 'total_cpu_workload'}, inplace=True)
    print(f"Resampling complete. Created a time series with {len(workload_hourly):,} hourly steps.")

    # 5. Save the Preprocessed Data 

    print(f"\n Step 5: Saving the clean data to {OUTPUT_FILENAME} ")
    
    workload_hourly.reset_index(inplace=True)
    workload_hourly.to_csv(OUTPUT_FILENAME, index=False)
    
    print("\nPreprocessing complete!")
    print(f"Clean data saved. You can now use '{OUTPUT_FILENAME}' in your GP modeling script.")
    print("\nFinal Data Head:")
    print(workload_hourly.head())
    print(f"\nTotal rows in final output file: {len(workload_hourly)}")