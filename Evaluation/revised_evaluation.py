import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

# Import your custom environment and the RL model
from workloadscheduling_env1 import WorkloadSchedulingEnv1
from stable_baselines3 import SAC

# Define Baseline Policies 


def random_policy(observation, action_space):
    """A policy that takes a completely random action."""
    return action_space.sample()

def always_on_policy(observation):
    """A naive policy that always takes the maximum possible action."""
    return np.array([1.0], dtype=np.float32)

def mpc_policy(observation, env_params):
    """
    Revised MPC policy to match WorkloadSchedulingEnv1.
    Goal: Decide what fraction of work to run to maximize RE usage and minimize grid cost.
    """
    # 1. Unpack parameters
    horizon = env_params['horizon']
    battery_capacity = env_params['battery_capacity']
    battery_efficiency = env_params['efficiency']
    max_charge_rate = env_params['max_charge_rate']
    grid_price = env_params['grid_price']
    
    # 2. Unpack observations (Matching your environment's keys)
    current_battery_mwh = observation['battery_soc'][0]
    pending_workload = observation['pending_workload_mwh'][0]
    # Combine solar and wind for total RE forecast
    re_forecast = observation['solar_mean'] + observation['wind_mean']
    
    # Assumption for MPC: future workload arrives at a steady rate 
    # (since the environment doesn't provide future workload arrival forecasts)
    future_arrival_estimate = pending_workload / 12.0 

    def objective_function(fractions):
        """Calculates negative reward based on environment logic"""
        temp_battery = current_battery_mwh
        temp_pending = pending_workload
        total_reward = 0
        
        for t in range(horizon):
            # 1. Work to run
            work_to_run = temp_pending * fractions[t]
            available_re = re_forecast[t]
            
            # 2. Battery/Grid logic
            net_energy = available_re - work_to_run
            grid_draw = 0
            
            if net_energy > 0: # Surplus
                space = battery_capacity - temp_battery
                charge = min(net_energy, max_charge_rate, space / battery_efficiency)
                temp_battery += charge * battery_efficiency
            else: # Deficit
                deficit = abs(net_energy)
                draw_from_batt = min(deficit, max_charge_rate, temp_battery)
                temp_battery -= draw_from_batt
                grid_draw = deficit - draw_from_batt
            
            # 3. Reward Calculation (matching env formula)
            work_completion_reward = work_to_run * 50.0
            direct_re_reward = min(work_to_run, available_re)
            grid_cost_penalty = grid_draw * grid_price
            
            total_reward += (work_completion_reward + direct_re_reward - grid_cost_penalty)
            
            # 4. State Update for next step in horizon
            temp_pending = (temp_pending - work_to_run) + future_arrival_estimate

        return -total_reward # Minimize negative reward

    # Fractions must be between 0.0 and 1.0
    bounds = Bounds([0.0] * horizon, [1.0] * horizon)
    initial_guess = np.full(horizon, 0.5)
    
    result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds)
    
    return np.array([result.x[0]], dtype=np.float32)
#  Main Evaluation Loop Function 

def run_evaluation_episode(policy_name, env, policy_function=None, model=None, num_steps=720):
    """Runs a full evaluation episode for a given policy and returns the results."""
    print(f"\n--- Evaluating Policy: {policy_name} ---")
    obs, info = env.reset()
    results_list = []
    
    for step in range(num_steps):
        start_obs = obs.copy()
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        elif policy_function is not None:
            # Handle the extra action_space argument for Random policy
            if policy_name == "Random Agent":
                action = policy_function(obs, env.action_space)
            else:
                action = policy_function(obs)
        else:
            raise ValueError("Must provide either a policy_function or a model.")
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_info = info.copy()
        step_info.update({
            'step': step,
            'action': action.item(),
            'reward': reward,
            'battery_soc': start_obs.get('battery_soc', [0.0])[0]
        })
        results_list.append(step_info)
        if terminated or truncated: break
            
    return pd.DataFrame(results_list)

# Main Execution Block 

if __name__ == '__main__':
    #  Configuration 
    SOLAR_MODEL_PATH = 'trained_gp_model_solar_mwh.pkl'
    WIND_MODEL_PATH = 'trained_gp_model_wind_power_mw.pkl'
    WORKLOAD_MODEL_PATH = 'trained_gp_model_total_cpu_workload.pkl'
    TRAINED_MODEL_PATH = "models_multiple/run_1/sac_scheduler_final.zip"
    EVALUATION_STEPS = 720

    #  Create Environment
    print("--- Creating Environment for Evaluation ---")
    eval_env = WorkloadSchedulingEnv1(
        solar_model_path=SOLAR_MODEL_PATH,
        wind_model_path=WIND_MODEL_PATH,
        workload_model_path=WORKLOAD_MODEL_PATH,
        simulation_duration_hours=EVALUATION_STEPS 
    )

    # MPC Parameters Matching WorkloadSchedulingEnv1
    env_params = {
        'horizon': 24,
        'max_charge_rate': 25.0,
        'battery_capacity': 100.0,
        'efficiency': 0.90,
        'grid_price': 150.0
    }
    
    #  Evaluate All Policies 
    all_results = {}
    all_results["Random Agent"] = run_evaluation_episode("Random Agent", eval_env, policy_function=random_policy, num_steps=EVALUATION_STEPS)
    all_results["Always-On Agent"] = run_evaluation_episode("Always-On Agent", eval_env, policy_function=always_on_policy, num_steps=EVALUATION_STEPS)
    
    mpc_policy_func = lambda obs: mpc_policy(obs, env_params)
    all_results["MPC Baseline"] = run_evaluation_episode("MPC Baseline", eval_env, policy_function=mpc_policy_func, num_steps=EVALUATION_STEPS)

    if os.path.exists(TRAINED_MODEL_PATH):
        trained_sac_model = SAC.load(TRAINED_MODEL_PATH, env=eval_env)
        all_results["GP-Augmented SAC Agent"] = run_evaluation_episode("GP-Augmented SAC Agent", eval_env, model=trained_sac_model, num_steps=EVALUATION_STEPS)
    else:
        print(f"WARNING: Trained model not found at '{TRAINED_MODEL_PATH}'. Skipping SAC evaluation.")
    
    eval_env.close()

    # Summary Table
    print("\n" + "="*80)
    summary_data = []
    for name, df in all_results.items():
        summary_data.append({
            "Policy": name,
            "Total Reward": df['reward'].sum(),
            "Total Grid Cost ($)": df.get('grid_cost', 0).sum(),
            "Total RE Usage (MWh)": df.get('direct_re_usage', 0).sum()
        })
    df_summary = pd.DataFrame(summary_data).set_index("Policy")
    print(df_summary.to_string(float_format="%.2f"))
    df_summary.to_csv("performance_summary_with_mpc.csv")

    # --- 5. Plotting ---
    if "GP-Augmented SAC Agent" in all_results:
        sac_df = all_results["GP-Augmented SAC Agent"]
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        timesteps = sac_df['step']

        axes[0].plot(timesteps, sac_df['reward'], label='Hourly Reward', color='blue')
        axes[0].set_title('SAC Agent Rewards'); axes[0].grid(True)
        
        # Scaling battery SoC to 0-1 range for the plot if needed
        axes[1].plot(timesteps, sac_df['battery_soc'] / 100.0, label='Battery SoC', color='green')
        axes[1].set_title('Battery State of Charge'); axes[1].set_ylim(0, 1.1); axes[1].grid(True)
        
        axes[2].plot(timesteps, sac_df['action'], label='Action (Work Fraction)', color='purple')
        axes[2].set_title('Agent Actions (Fraction of Workload)'); axes[2].set_ylim(0, 1.1); axes[2].grid(True)
        
        axes[3].plot(timesteps, sac_df['grid_cost'], label='Grid Cost', color='red')
        axes[3].set_title('Grid Cost ($)'); axes[3].grid(True)
        
        plt.tight_layout()
        plt.savefig('qualitative_policy_plot_with_mpc.png')
        print("\nPlot saved to qualitative_policy_plot_with_mpc.png")

    print("\nEvaluation Finished.")