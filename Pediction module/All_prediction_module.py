# predictionmodule.py

# This is a reusable module for making predictions.
# Reinforcement Learning environment will import and use this class.

import pickle
import gpflow
import numpy as np
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Set backend for server use
import matplotlib.pyplot as plt
import pandas as pd

# Settiing the default float type for all GPflow operations to ensure consistency.
gpflow.config.set_default_float(np.float64)


class PredictionModule:
    """
    A class that loads a trained GPflow model and provides a simple
    interface to make future predictions.
    """

    def __init__(self, model_path: str):
        """
        Initializes the prediction module by loading a trained model from a .pkl file.
        """
        print(f"Initializing prediction module with model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FATAL ERROR: The required model file was not found at '{model_path}'")

        with open(model_path, 'rb') as f:
            predictor_package = pickle.load(f)

        self.gp_model = predictor_package['model']
        self.y_mean = predictor_package['y_mean']
        self.y_std = predictor_package['y_std']

        print(f" -> Module initialized successfully.")

    def predict(self, start_hour: int, horizon: int = 24):
        """
        Makes a prediction for the next `horizon` hours starting from `start_hour`.
        """
        future_time_points = np.arange(start_hour, start_hour + horizon, dtype=np.float64).reshape(-1, 1)
        mu_normalized, sigma_normalized = self.gp_model.predict_f(future_time_points, full_cov=True)

        mu_normalized_np = mu_normalized.numpy()
        sigma_normalized_np = np.squeeze(sigma_normalized.numpy())

        mu_real_scale = mu_normalized_np * self.y_std + self.y_mean
        sigma_real_scale = sigma_normalized_np * (self.y_std ** 2)

        # Clip predictions to be non-negative, as negative power/workload is not physical.
        # This is likely the cause of the flat line at 0 if the model predicts negative values.
        mu_real_scale = mu_real_scale.clip(min=0)

        return mu_real_scale, sigma_real_scale

#  Demonstration Block 

if __name__ == '__main__':
    print("\n" + "="*70)
    print("--- Running Demonstration ---")
    print("="*70 + "\n")

    # Define paths to trained models 

    SOLAR_MODEL_PATH = 'trained_gp_model_solar_mwh.pkl'
    WIND_MODEL_PATH = 'trained_gp_model_wind_power_mw.pkl'
    WORKLOAD_MODEL_PATH = 'trained_gp_model_total_cpu_workload.pkl'

    # Initialize all prediction modules
    try:
        solar_predictor = PredictionModule(model_path=SOLAR_MODEL_PATH)
        wind_predictor = PredictionModule(model_path=WIND_MODEL_PATH)
        workload_predictor = PredictionModule(model_path=WORKLOAD_MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        exit() # Stop if any model is missing

    # -Generate the 24-hour forecasts 
    current_simulation_time = 200
    planning_horizon = 24

    print(f"\nGenerating a {planning_horizon}-hour forecast starting from hour {current_simulation_time} ")
    print("NOTE: If predictions are flat, the model may be extrapolating far outside its training data range.")


    solar_mean, solar_cov = solar_predictor.predict(start_hour=current_simulation_time, horizon=planning_horizon)
    wind_mean, wind_cov = wind_predictor.predict(start_hour=current_simulation_time, horizon=planning_horizon)
    workload_mean, workload_cov = workload_predictor.predict(start_hour=current_simulation_time, horizon=planning_horizon)

    # Process results for plotting and saving 
    # Create the time axis for the plot
    hours = np.arange(current_simulation_time, current_simulation_time + planning_horizon)

    # Calculate standard deviation (uncertainty) from the covariance matrix
    solar_std = np.sqrt(np.diag(solar_cov))
    wind_std = np.sqrt(np.diag(wind_cov))
    workload_std = np.sqrt(np.diag(workload_cov))

    #  Save the raw data to a CSV file with CORRECTED units 
    df_results = pd.DataFrame({
        'Hour': hours,
        'Solar_Mean_MWh': solar_mean.flatten(),
        'Solar_StdDev_MWh': solar_std,
        'Wind_Mean_MW': wind_mean.flatten(),
        'Wind_StdDev_MW': wind_std,
        'Workload_Mean_CPU': workload_mean.flatten(),
        'Workload_StdDev_CPU': workload_std,
    })

    csv_filename = 'combined_24hr_forecast.csv'
    df_results.to_csv(csv_filename, index=False, float_format='%.2f')
    print(f"\nForecast data has been saved to '{csv_filename}'")
    print("CSV Content Preview:")
    print(df_results.head())

    # Create the combined visualization 
    # To plot different units together, we create subplots for clarity.
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 12), sharex=True)
    
    # Plot 1 is for Renewable Generation (Solar and Wind)
    ax1.set_title(f'24-Hour Renewable Energy Forecast (Starting at Hour {current_simulation_time})', fontsize=16, weight='bold')
    
    # Plot Solar (MWh)
    color_solar = 'orange'
    ax1.plot(hours, solar_mean, 'o-', color=color_solar, label='Solar Generation (MWh)')
    ax1.fill_between(hours, (solar_mean.flatten() - 1.96 * solar_std), (solar_mean.flatten() + 1.96 * solar_std), color=color_solar, alpha=0.2, label='Solar 95% Confidence')
    ax1.set_ylabel('Energy (MWh)', fontsize=12, color=color_solar)
    ax1.tick_params(axis='y', labelcolor=color_solar)
    
    # Plot Wind (MW) on a secondary y-axis for clarity
    ax1b = ax1.twinx()
    color_wind = 'dodgerblue'
    ax1b.plot(hours, wind_mean, 's-', color=color_wind, label='Wind Generation (MW)')
    ax1b.fill_between(hours, (wind_mean.flatten() - 1.96 * wind_std), (wind_mean.flatten() + 1.96 * wind_std), color=color_wind, alpha=0.2, label='Wind 95% Confidence')
    ax1b.set_ylabel('Power (MW)', fontsize=12, color=color_wind)
    ax1b.tick_params(axis='y', labelcolor=color_wind)
    
    # Combine legends for the first plot
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc='upper left')

    # plot 2 is for Data Center Workload 
    ax2.set_title(f'24-Hour Data Center Workload Forecast', fontsize=16, weight='bold')
    color_workload = 'black'
    ax2.plot(hours, workload_mean, 'd--', color=color_workload, label='Data Center Workload')
    ax2.fill_between(hours, (workload_mean.flatten() - 1.96 * workload_std), (workload_mean.flatten() + 1.96 * workload_std), color='gray', alpha=0.2, label='Workload 95% Confidence')
    ax2.set_ylabel('CPU Workload (Arbitrary Units)', fontsize=12)
    ax2.legend(loc='upper left')

    # Formatting the plot
    ax2.set_xlabel('Time (Hour of Simulation)', fontsize=12)
    ax2.set_xticks(hours)
    ax2.tick_params(axis='x', rotation=45)
    
    fig.tight_layout(pad=3.0) # Adjust layout to prevent titles from overlapping
    plot_filename = 'combined_forecast_plot.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nCombined forecast plot has been saved to '{plot_filename}'")

    print("\n\n" + "="*70)
    print("--- Demonstration Completed Successfully ---")
    print("="*70)