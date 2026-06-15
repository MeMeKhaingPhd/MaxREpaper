import pandas as pd
import numpy as np
import os

# =============================================================================
# --- Configuration ---
# =============================================================================
# !!! STEP 1: UPDATE THIS VARIABLE TO MATCH YOUR FILE !!!
INPUT_FILENAME = 'C:\\Users\\Me Me Khaing\\Downloads\Shrisha Sir\\Solar.csv' 
# The name of the final, clean file this script will create
OUTPUT_FILENAME = 'C:\\Users\\Me Me Khaing\\Downloads\Shrisha Sir\\preprocessed_solar_hourly.csv'

# =============================================================================
# --- 1. Load the Data ---
# =============================================================================
print(f"--- Step 1: Loading data from {INPUT_FILENAME} ---")

if not os.path.exists(INPUT_FILENAME):
    print(f"\nERROR: Input file not found: {INPUT_FILENAME}")
    print("Please place your solar data CSV in the same directory and update the INPUT_FILENAME.")
else:
    # Load the data. This format is usually clean enough to load directly.
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Successfully loaded {len(df):,} rows.")

    # =============================================================================
    # --- 2. Create a Single Timestamp Column ---
    # =============================================================================
    print("\n--- Step 2: Combining Year, Month, Day, Hour, Minute into a single timestamp ---")
    
    # Create a dictionary of the time columns to pass to pd.to_datetime
    # This is a robust way to build a timestamp from separate columns.
    time_cols = {
        'year': df['Year'],
        'month': df['Month'],
        'day': df['Day'],
        'hour': df['Hour'],
        'minute': df['Minute']
    }
    
    # pd.to_datetime can intelligently build a single timestamp from these columns
    df['timestamp'] = pd.to_datetime(time_cols)
    
    print("Timestamp column created successfully.")
    
    # Set the new timestamp as the index for time-series operations
    df.set_index('timestamp', inplace=True)
    
    # Ensure the data is sorted by time, which is crucial for resampling
    df.sort_index(inplace=True)

    # =============================================================================
    # --- 3. Select and Clean the Target Variable (GHI) ---
    # =============================================================================
    print("\n--- Step 3: Selecting and cleaning the GHI (Global Horizontal Irradiance) data ---")
    
    # We are interested in the 'GHI' column as our measure of solar energy potential
    # Create a new dataframe with just this column for clarity.
    solar_df = df[['GHI']].copy()

    # Solar power cannot be negative. Set any negative values to 0.
    solar_df.loc[solar_df['GHI'] < 0, 'GHI'] = 0
    print("Set any negative GHI values to 0.")

    # Check for missing values
    print(f"Missing GHI values before cleaning: {solar_df['GHI'].isnull().sum()}")
    
    # Interpolate to fill any small gaps (NaN values)
    solar_df['GHI'].interpolate(method='linear', inplace=True)
    print("Filled missing values using linear interpolation.")

    # =============================================================================
    # --- 4. Resample to Hourly Frequency ---
    # =============================================================================
    print("\n--- Step 4: Resampling data to a consistent hourly frequency ---")
    
    # Resample the GHI data to hourly. The standard method is to take the
    # average GHI over the hour.
    solar_hourly = solar_df.resample('h').mean()
    
    # After resampling, there might still be gaps if a whole hour was missing.
    # We can fill these again to ensure a continuous time series.
    solar_hourly.interpolate(method='linear', inplace=True)
    solar_hourly.fillna(method='bfill', inplace=True) # Backfill for any at the start
    solar_hourly.fillna(method='ffill', inplace=True) # Forward-fill for any at the end
    
    print(f"Resampling complete. Created a time series with {len(solar_hourly):,} hourly steps.")
    
    # Rename the column for consistency with our final GP model
    solar_hourly.rename(columns={'GHI': 'solar_mwh'}, inplace=True)

    # =============================================================================
    # --- 5. Save the Preprocessed Data ---
    # =S ============================================================================
    print(f"\n--- Step 5: Saving the clean data to {OUTPUT_FILENAME} ---")

    # The index currently has the timestamp. Reset it to make it a regular column.
    solar_hourly.reset_index(inplace=True)

    # Save the final, clean dataframe to a new CSV file.
    solar_hourly.to_csv(OUTPUT_FILENAME, index=False)

    print("\nPreprocessing complete!")
    print(f"Clean solar data saved. You are now ready to merge this file with your workload data.")
    print("\nFinal Data Head:")
    print(solar_hourly.head())