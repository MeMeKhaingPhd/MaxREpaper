import pandas as pd
import numpy as np
import os

# The name of your raw wind data file
INPUT_FILENAME = 'C:\\Users\\Me Me Khaing\\Downloads\Shrisha Sir\\1942108_39.00_-82.90_2014wind.csv'  

# The name of the final, clean file this script will create
OUTPUT_FILENAME = 'C:\\Users\\Me Me Khaing\\Downloads\\Shrisha Sir\\preprocessed_wind_hourly.csv'

# The names of the important columns in YOUR CSV file, based on your image
TIMESTAMP_COLS = ['Year', 'Month', 'Day', 'Hour', 'Minute']
WIND_SPEED_COL = 'wind speed at 100m (m/s)' # The column with wind speed values


#  model a hypothetical wind turbine's power output.
CUT_IN_SPEED = 3.5  # m/s, the speed at which the turbine starts generating power
RATED_SPEED = 14.0  # m/s, the speed at which the turbine reaches max power output
CUT_OUT_SPEED = 25.0 # m/s, the speed at which the turbine shuts down for safety
RATED_POWER_MW = 2.0 # MW, the maximum power output of our hypothetical turbine# =============================================================================
# 1. Load the Data 

print(f"--- Step 1: Loading data from {INPUT_FILENAME} ---")

if not os.path.exists(INPUT_FILENAME):
    print(f"\nERROR: Input file not found: {INPUT_FILENAME}")
else:
    df = pd.read_csv(INPUT_FILENAME)
    # The image shows the power column is 'power - DI' and has 'N/A'.
    # We will ignore it and calculate our own.
    print(f"Successfully loaded {len(df):,} rows.")

    
    # Create Timestamp and Clean Data 
    
    print("\n--- Step 2: Creating timestamp and cleaning data ---")
    
    # Create a single timestamp column from the separate date/time parts
    df['timestamp'] = pd.to_datetime(df[TIMESTAMP_COLS])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    # Convert wind speed to a numeric type, coercing any errors
    df[WIND_SPEED_COL] = pd.to_numeric(df[WIND_SPEED_COL], errors='coerce')
    df.dropna(subset=[WIND_SPEED_COL], inplace=True)
    print("Timestamp created and non-numeric wind speeds removed.")

    
    #  Calculate Wind Power from Wind Speed ---
   
    print("\n--- Step 3: Calculating wind power (MW) from wind speed (m/s) ---")
    
    def calculate_power(wind_speed):
        """Calculates wind turbine power output based on a standard power curve."""
        if wind_speed < CUT_IN_SPEED or wind_speed > CUT_OUT_SPEED:
            # If wind is too slow or too fast, output is 0
            return 0.0
        elif wind_speed >= RATED_SPEED:
            # If wind is at or above rated speed, output is max power
            return RATED_POWER_MW
        else:
            # Between cut-in and rated speed, power follows a cubic relationship
            # We use a formula to scale it smoothly to the rated power.
            power = RATED_POWER_MW * ((wind_speed**3 - CUT_IN_SPEED**3) / (RATED_SPEED**3 - CUT_IN_SPEED**3))
            return power

    # Apply this function to every row to create our new 'wind_power_mw' column
    df['wind_power_mw'] = df[WIND_SPEED_COL].apply(calculate_power)
    print("New 'wind_power_mw' column created successfully.")

    
    # Resample to Hourly Frequency 
    
    print("\nStep 4: Resampling data to a consistent hourly frequency ")
    
    # Resample the calculated power to hourly. We take the average power over the hour.
    wind_hourly = df[['wind_power_mw']].resample('h').mean()
    
    # Interpolate to fill any gaps
    wind_hourly.interpolate(method='linear', inplace=True)
    wind_hourly.fillna(method='bfill', inplace=True)
    wind_hourly.fillna(method='ffill', inplace=True)
    
    print(f"Resampling complete. Created a time series with {len(wind_hourly):,} hourly steps.")
    
    # Save the Preprocessed Data 
    
    print(f"\n--- Step 5: Saving the clean data to {OUTPUT_FILENAME} ---")

    wind_hourly.reset_index(inplace=True)
    wind_hourly.to_csv(OUTPUT_FILENAME, index=False)

    print("\nPreprocessing complete!")
    print(f"Clean wind power data saved. You can now merge this with your workload data.")
    print("\nFinal Data Head:")
    print(wind_hourly.head())