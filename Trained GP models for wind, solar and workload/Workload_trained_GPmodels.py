
# Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gpflow
import tensorflow as tf
gpflow.config.set_default_float(np.float64)
from gpflow.utilities import print_summary
import os
import pickle
# Configuration for a SINGLE Model Run (Workload) 
# pre-configured to run for your Workload data.
INPUT_FILE = 'preprocessed_workload_hourly.csv'
TARGET_COLUMN_NAME = 'total_cpu_workload' 
# The name for the saved model file
MODEL_FILENAME = f"trained_gp_model_{TARGET_COLUMN_NAME}.pkl"
# =============================================================================

# Check if the file exists
if not os.path.exists(INPUT_FILE):
    print(f"\nFATAL ERROR: The required input file '{INPUT_FILE}' was not found.")
    print("Please make sure you have successfully run the solar preprocessing script first.")
else:
    # --- Load the single dataset ---
    print(f"\n--- Loading and Preparing Data for: {TARGET_COLUMN_NAME} ---")
    df = pd.read_csv(INPUT_FILE, parse_dates=['timestamp'])
    df = df.iloc[::4]
    print(f"Downsampled to {len(df)} rows to reduce memory usage.")
    print(f"Successfully loaded {len(df)} hourly rows from {INPUT_FILE}.")

    # --- Prepare data for the GP model ---
    start_time = df['timestamp'].min()
    time_diff_seconds = (df['timestamp'] - start_time).dt.total_seconds()
    T_observed = (time_diff_seconds.to_numpy() / 3600.0).reshape(-1, 1).astype(np.float64)
    Y_observed = df[TARGET_COLUMN_NAME].to_numpy().reshape(-1, 1).astype(np.float64)
    # --- Normalize the Y data ---
    print("Normalizing Y data...")
    Y_mean = np.mean(Y_observed)
    Y_std = np.std(Y_observed)
    # Add a small epsilon to std dev to prevent division by zero if all values are the same (e.g., all 0 at night)
    if Y_std == 0: Y_std = 1.0 
    Y_normalized = (Y_observed - Y_mean) / Y_std

    # =============================================================================
    # --- Step 2: Define Kernel and Create the GP Model ---
    # =============================================================================
    print("\n--- Step 2: Defining kernel and creating the GP model ---")
    
    print("Using a Solar Kernel (Daily Cycle + RBF)...")
    # This kernel is tailored for solar data, which has a strong 24-hour cycle.
    kernel = (
        gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=5.0) +
        gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=1.0), period=24.0) # Daily period
    )

    # Create the single GPflow model
    # Define M, the number of "inducing points" for the sparse model
    M   = 1000 
# Select M inducing points, spread evenly across the time range
    inducing_variable = np.linspace(T_observed.min(), T_observed.max(), M).reshape(-1, 1).astype(np.float64)

# Create the EFFICIENT Sparse GP model
    gp_model = gpflow.models.SGPR(
    data=(T_observed, Y_normalized), 
    kernel=kernel, 
    inducing_variable=inducing_variable
)

    # =============================================================================
    # --- Step 3: Optimize the GP Model ---
    # =============================================================================
    print(f"\n--- Step 3: Optimizing the GP model for {TARGET_COLUMN_NAME} ---")
    optimizer = gpflow.optimizers.Scipy()
    optimizer.minimize(gp_model.training_loss, gp_model.trainable_variables, options=dict(maxiter=1000))
    print_summary(gp_model)

    # =============================================================================
    # --- Step 4: Save the Trained Model and Stats ---
    # =============================================================================
    print(f"\n--- Step 4: Saving the trained model and stats ---")

    model_and_stats = {
        'model': gp_model,
        'y_mean': Y_mean,
        'y_std': Y_std
    }
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model_and_stats, f)
    print(f"Model and stats successfully saved to '{MODEL_FILENAME}'")

    # =============================================================================
    # --- Step 5: Visualize the Final Result ---
    # =============================================================================
    print("\n--- Step 5: Generating Final Plot ---")
    def plot_gp(gp_model, y_mean, y_std, title):
        X_observed_np = gp_model.data[0].numpy()
        Y_observed_norm = gp_model.data[1]
        Y_observed_unnorm_np = (Y_observed_norm * y_std + y_mean).numpy()
        plot_time = np.arange(X_observed_np.min(), X_observed_np.max() + 48, 0.5, dtype=np.float64).reshape(-1, 1)
        mean_norm, var_norm = gp_model.predict_y(plot_time)
        mean_norm_np = mean_norm.numpy()
        var_norm_np = var_norm.numpy()
        mean_unnorm = mean_norm_np * y_std + y_mean
        std_dev_unnorm = np.sqrt(var_norm_np) * y_std

        plt.figure(figsize=(15, 6))
        plt.title(title, fontsize=16)
        plt.plot(X_observed_np, Y_observed_unnorm_np, 'kx', mew=2, label='Observed Data')
        plt.plot(plot_time, mean_unnorm, 'b-', lw=2, label='GP Mean Prediction')
        plt.fill_between(plot_time.flatten(),
                         (mean_unnorm - 1.96 * std_dev_unnorm).flatten(),
                         (mean_unnorm + 1.96 * std_dev_unnorm).flatten(),
                         color='blue', alpha=0.2, label='95% Confidence Interval')
        plt.xlabel("Time (hours from start)")
        plt.ylabel("Value (Original Scale - Workload)")
        plt.legend()
        plt.grid(True)
        # plt.show() 
        plt.savefig(f'plot_{TARGET_COLUMN_NAME}.png') # <-- Add this line
        print(f"Plot saved to plot_{TARGET_COLUMN_NAME}.png")

    # Call the plotting function with a dynamic title
    plot_gp(gp_model, Y_mean, Y_std, f"GP Model Fit for {TARGET_COLUMN_NAME}")