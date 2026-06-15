import os
import gymnasium as gym
from workloadscheduling_env1 import WorkloadSchedulingEnv1
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration for Multiple Runs 
# Define paths to your trained forecasting models
SOLAR_MODEL_PATH = 'trained_gp_model_solar_mwh.pkl'
WIND_MODEL_PATH = 'trained_gp_model_wind_power_mw.pkl'
WORKLOAD_MODEL_PATH = 'trained_gp_model_total_cpu_workload.pkl'

# times to train the agent 
NUM_TRAINING_RUNS = 3 

# Create a parent directory for all runs
parent_log_dir = "logs_multiple/"
parent_model_dir = "models_multiple/"
os.makedirs(parent_log_dir, exist_ok=True)
os.makedirs(parent_model_dir, exist_ok=True)

# Training parameters
TOTAL_TIMESTEPS = 50_000
SAVE_FREQ = 10_000

# Training Loop


for i in range(NUM_TRAINING_RUNS):
    run_num = i + 1
    print("\n" + "="*80)
    print(f"--- STARTING TRAINING RUN {run_num} of {NUM_TRAINING_RUNS} ---")
    print("="*80)

    # Create unique directories for this specific run
    log_dir = os.path.join(parent_log_dir, f"run_{run_num}")
    model_dir = os.path.join(parent_model_dir, f"run_{run_num}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create the Environment
    print(f"Run {run_num}: Creating the environment...")
    try:
        env = WorkloadSchedulingEnv1(
            solar_model_path=SOLAR_MODEL_PATH,
            wind_model_path=WIND_MODEL_PATH,
            workload_model_path=WORKLOAD_MODEL_PATH,
            simulation_duration_hours=720
        )
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Could not create environment. {e}")
        exit()
    print("Environment created successfully.")

    # Set up the SAC Agent 
    # `seed` is crucial for reproducibility
    print(f"Run {run_num}: Setting up the SAC agent...")
    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        seed=run_num # Use a different seed for each run
    )

    # Set up Callbacks
    checkpoint_callback = CheckpointCallback(
      save_freq=SAVE_FREQ,
      save_path=model_dir,
      name_prefix="sac_scheduler_model"
    )

    # Train the Agent
    print(f"\n--- Run {run_num}: Starting Agent Training ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        log_interval=1
    )

    # --- Save the Final Model for this run ---
    final_model_path = os.path.join(model_dir, "sac_scheduler_final.zip")
    model.save(final_model_path)
    print(f"--- Run {run_num} Complete. Final model saved to: {final_model_path} ---")

    # Close the environment
    env.close()

print("\nAll training runs complete!")