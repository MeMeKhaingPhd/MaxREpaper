# stability_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from workload_schedule_for_stability import WorkloadSchedulingEnv1_stability
from stable_baselines3 import SAC

# Configuration 


# Use BEST trained agent for this analysis 
TRAINED_MODEL_PATH = "models_multiple/run_1/sac_scheduler_final.zip" 

# A very long simulation is required for stability analysis 
SIMULATION_DURATION_HOURS = 8760 # One full year

# Output files for the results
RESULTS_CSV_PATH = "stability_analysis_results.csv"
PLOT_FILENAME = "stability_analysis_plot.png"


#  Run the Long-Term Simulation 
if __name__ == '__main__':
    # Create the Environment
    print("--- Creating Environment for Stability Analysis ---")
    env = WorkloadSchedulingEnv1_stability(
        solar_model_path='trained_gp_model_solar_mwh.pkl',
        wind_model_path='trained_gp_model_wind_power_mw.pkl',
        workload_model_path='trained_gp_model_total_cpu_workload.pkl',
        simulation_duration_hours=SIMULATION_DURATION_HOURS 
    )

    # Load the Trained Agent 
    print(f"--- Loading Trained Agent from {TRAINED_MODEL_PATH} ---")
    try:
        model = SAC.load(TRAINED_MODEL_PATH, env=env)
    except FileNotFoundError:
        print(f"FATAL ERROR: Trained model not found at '{TRAINED_MODEL_PATH}'")
        exit()

    # Run the Full Year Episode 
    print(f"--- Running {SIMULATION_DURATION_HOURS}-hour simulation... This will take a while. ---")
    obs, info = env.reset()
    done = False
    results_list = []
    
    for step in range(SIMULATION_DURATION_HOURS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        info['step'] = step
        results_list.append(info)

        if (step + 1) % 1000 == 0: # Print a progress update every 1000 steps
            print(f"  ... completed step {step + 1} of {SIMULATION_DURATION_HOURS}")

    env.close()
    
    # Convert results to a DataFrame and save
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\n--- Simulation Complete. Detailed results saved to {RESULTS_CSV_PATH} ---")

    # Generate the Stability Plots 
    print(f"--- Generating stability plot and saving to {PLOT_FILENAME} ---")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot 1: Battery State of Charge (SoC)
    ax1.plot(df_results['step'], df_results['battery_soc_mwh'], color='dodgerblue', linewidth=1)
    ax1.set_title('Long-Term System Stability Analysis (1 Year)', fontsize=16, weight='bold')
    ax1.set_ylabel('Battery SoC (MWh)')
    # Add lines for min/max capacity to show it stays within bounds
    ax1.axhline(y=0, color='r', linestyle='--', label='Min Capacity (0 MWh)')
    ax1.axhline(y=env.battery_capacity_mwh, color='r', linestyle='--', label=f'Max Capacity ({env.battery_capacity_mwh} MWh)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Pending Workload Queue
    ax2.plot(df_results['step'], df_results['pending_workload_mwh'], color='green', linewidth=1)
    ax2.set_ylabel('Pending Workload (MWh)')
    ax2.set_xlabel('Time (Hour of Simulation)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=300)
    print("--- Plot generated successfully. ---")