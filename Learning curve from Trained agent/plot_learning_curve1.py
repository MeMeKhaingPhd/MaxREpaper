# plot_learning_curve.py (using TensorBoard files, with correct Timestep X-Axis)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import re

# 1. Configuration
PARENT_LOG_DIR = "logs_multiple/"
REWARD_TAG = 'rollout/ep_rew_mean'
SMOOTHING_WINDOW = 50
# 2. Function to Extract Data from a TensorBoard File 
def extract_tensorboard_data(event_file_path, tag):
    """Extracts scalar data from a specific TensorBoard event file."""
    try:
        ea = event_accumulator.EventAccumulator(event_file_path,
            size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        
        if tag not in ea.Tags()['scalars']:
            print(f"Warning: Tag '{tag}' not found in {event_file_path}.")
            return None, None

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except Exception as e:
        print(f"Error loading {event_file_path}: {e}")
        return None, None
# 3. Load and Process Data from All Runs 
all_rewards = []
all_steps = [] # <<< NEW LINE: Create a list to store the timesteps for each run
min_len = float('inf')

run_dirs = [os.path.join(PARENT_LOG_DIR, d) for d in os.listdir(PARENT_LOG_DIR) if os.path.isdir(os.path.join(PARENT_LOG_DIR, d))]

for log_dir in run_dirs:
    sac_subdirs = [d for d in os.listdir(log_dir) if d.startswith('SAC_') and os.path.isdir(os.path.join(log_dir, d))]
    
    if not sac_subdirs:
        print(f"Warning: No 'SAC_X' subdirectory found in {log_dir}")
        continue

    sac_subdirs.sort(key=lambda x: int(re.search(r'SAC_(\d+)', x).group(1)), reverse=True)
    latest_sac_dir_name = sac_subdirs[0]
    latest_sac_dir_path = os.path.join(log_dir, latest_sac_dir_name)
    
    print(f"Found latest log directory for {os.path.basename(log_dir)}: {latest_sac_dir_name}")

    event_file_path = None
    for item in os.listdir(latest_sac_dir_path):
        if "events.out.tfevents" in item:
            event_file_path = os.path.join(latest_sac_dir_path, item)
            break
            
    if event_file_path:
        steps, rewards = extract_tensorboard_data(event_file_path, REWARD_TAG)
        if rewards:
            all_rewards.append(rewards)
            all_steps.append(steps) # <<< NEW LINE: Add the extracted steps to our list
            if len(rewards) < min_len:
                min_len = len(rewards)
    else:
        print(f"Warning: No TensorBoard event file found in {latest_sac_dir_path}")

if not all_rewards:
    print("\nFATAL ERROR: No reward data could be loaded. Please check your log directories and the REWARD_TAG.")
else:
    # Trim all reward AND step arrays to the length of the shortest run
    trimmed_rewards = [rewards[:min_len] for rewards in all_rewards]
    rewards_array = np.array(trimmed_rewards)
    
    trimmed_steps = [steps[:min_len] for steps in all_steps] # <<< NEW LINE: Trim the steps arrays too
    steps_array = np.array(trimmed_steps) # <<< NEW LINE: Create a numpy array of steps

    #  4. Calculate Smoothed Statistics
    df_rewards=pd.DataFrame(rewards.array.T)
    smoothed_df=df_rewards.rolling(SMOOTHING_WINDOW, min_periods=1).mean()
    mean_rewards=smoothed_df.mean(axis=1)
    std_rewards=smoothed_df.std(axis=1)
    upper_bound=mean_rewards+std_rewards
    lower_bound=mean_rewards-std_rewards

# 5. Generate the Plot (with BOLD and BIGGER Fonts)
print("\n--- Generating Final Plot with Enhanced Fonts ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8)) # Adjusted figsize for a more standard aspect ratio

# Calculate the average timesteps for the x-axis
x_axis = np.mean(steps_array, axis=0)

# Plot the mean line
ax.plot(x_axis, mean_rewards, color='blue', label='Mean Episode Reward', linewidth=2.5)#Made line thicker

# Plot the confidence interval
ax.fill_between(x_axis, lower_bound, upper_bound, color='blue', alpha=0.2, label='Standard Deviation')

# FONT CUSTOMIZATION
# Set the title with a larger, bold font
ax.set_title(f'SAC Agent Learning Curve ({len(run_dirs)} Runs)', fontsize=22, fontweight='bold')

# Set the x-axis label with a larger, bold font
ax.set_xlabel('Training Timesteps', fontsize=18, fontweight='bold')

# Set the y-axis label with a larger, bold font
ax.set_ylabel('Mean Episode Reward', fontsize=18, fontweight='bold')

# Set the legend with a larger font
ax.legend(fontsize=14)

# Set the tick labels (the numbers on the axes) to be larger
ax.tick_params(axis='both', which='major', labelsize=14)

# Add grid lines for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.7)

# Ensure everything fits without overlapping
plt.tight_layout()

# Save the figure with a new name
plt.savefig("learning_curve_bold_fonts.png", dpi=300)
plt.show()

print("\nLearning curve plot with bold and bigger fonts saved to 'learning_curve_bold_fonts.png'")