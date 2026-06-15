# workload_scheduling_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections
import matplotlib
matplotlib.use('Agg')

# We import the class 
from All_prediction_module import PredictionModule

class WorkloadSchedulingEnv1(gym.Env):
    """
    A custom RL environment for scheduling data center workload to maximize
    the use of renewable energy.

    The agent's goal is to decide what fraction of a 'deferrable' workload
    to run each hour, based on forecasts for renewable energy.

    - State (Observation):Dictionary with forecasts, battery level,
      and the amount of uncompleted workload.
    - Action: A continuous value representing the fraction of the
      available deferrable workload to schedule in the current hour.
    - Reward: A combination of maximizing renewable energy usage and
      penalizing uncompleted work at the end of the day.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 solar_model_path: str,
                 wind_model_path: str,
                 # The workload model now represents the *total available* workload
                 workload_model_path: str, 
                 grid_price_per_mwh: float = 150.0,
                 simulation_duration_hours: int = 720, # ~1 month episodes
                 planning_horizon: int = 24,
                 battery_capacity_mwh: float = 100.0,
                 battery_max_charge_rate_mw: float = 25.0,
                 battery_efficiency: float = 0.9,
                 work_penalty_factor: float = 200.0): # Penalty for not finishing work
        """
        Initializes the Environment.

        Args:
            work_penalty_factor: A multiplier for the penalty assessed for each
                                 MWh of work left unfinished at the end of a 24-hour cycle.
                                 This should be higher than the grid price to incentivize finishing work.
        """
        super().__init__()

        # Initialize Prediction Modules 
        print("Initializing forecasting modules...")
        self.solar_predictor = PredictionModule(solar_model_path)
        self.wind_predictor = PredictionModule(wind_model_path)
        self.workload_predictor = PredictionModule(workload_model_path)
        print("All modules initialized.")

        # Store Environment Parameters
        self.grid_price_per_mwh = grid_price_per_mwh
        self.total_timesteps = simulation_duration_hours
        self.planning_horizon = planning_horizon
        self.battery_capacity_mwh = battery_capacity_mwh
        self.battery_max_charge_rate_mw = battery_max_charge_rate_mw
        self.battery_efficiency = battery_efficiency
        self.work_penalty_factor = work_penalty_factor

        # Define Action Space 
        # A single continuous action: what fraction (0% to 100%) of the available
        # deferrable workload to run this hour.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Define Observation Space
        self.observation_space = spaces.Dict({
            "current_hour_of_day": spaces.Box(low=0, high=23, shape=(1,), dtype=np.int32),
            "battery_soc": spaces.Box(low=0, high=self.battery_capacity_mwh, shape=(1,), dtype=np.float32),
            #  How much work is pending in the queue
            "pending_workload_mwh": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),

            # Forecasts remain the same
            "solar_mean": spaces.Box(low=0, high=np.inf, shape=(self.planning_horizon,), dtype=np.float32),
            "wind_mean": spaces.Box(low=0, high=np.inf, shape=(self.planning_horizon,), dtype=np.float32),
            # We only provide the mean forecast for the agent to act on
        })
        
        # Internal state of the environment
        self._current_timestep = 0
        self._battery_soc_mwh = 0.0
        # A queue to hold the total available workload for the day
        self._daily_workload_queue = collections.deque(maxlen=24)

    def _get_observation(self):
        """Generates the observation for the current timestep."""
        solar_mean, _ = self.solar_predictor.predict(self._current_timestep, self.planning_horizon)
        wind_mean, _ = self.wind_predictor.predict(self._current_timestep, self.planning_horizon)
        
        pending_workload = sum(self._daily_workload_queue)

        return {
            "current_hour_of_day": np.array([self._current_timestep % 24], dtype=np.int32),
            "battery_soc": np.array([self._battery_soc_mwh], dtype=np.float32),
            "pending_workload_mwh": np.array([pending_workload], dtype=np.float32),
            "solar_mean": solar_mean.flatten().astype(np.float32),
            "wind_mean": wind_mean.flatten().astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        
        self._current_timestep = 0
        self._battery_soc_mwh = self.battery_capacity_mwh / 2.0
        self._daily_workload_queue.clear()
        
        # Pre-populate the queue with the first 24 hours of available workload
        workload_mean, _ = self.workload_predictor.predict(0, self.planning_horizon)
        for w in workload_mean.flatten():
            self._daily_workload_queue.append(max(0, w))

        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: np.ndarray):
        """Executes one timestep in the environment."""
        
    def step(self, action: np.ndarray):
        """Executes one timestep in the environment."""
        
        # Determine "True" Renewable Energy for the hour 
        jitter = 1e-6
        true_solar_mean, true_solar_cov = self.solar_predictor.predict(self._current_timestep, 1)
        true_solar_cov_2d = np.atleast_2d(true_solar_cov) + np.eye(1) * jitter
        true_solar = max(0, self.np_random.multivariate_normal(true_solar_mean.flatten(), true_solar_cov_2d)[0])

        true_wind_mean, true_wind_cov = self.wind_predictor.predict(self._current_timestep, 1)
        true_wind_cov_2d = np.atleast_2d(true_wind_cov) + np.eye(1) * jitter
        true_wind = max(0, self.np_random.multivariate_normal(true_wind_mean.flatten(), true_wind_cov_2d)[0])
        
        total_renewable_generation = true_solar + true_wind

        # Determine Workload to Schedule based on Agent's Action
        work_to_run_this_hour = 0
        if len(self._daily_workload_queue) > 0:
            available_work_mwh = self._daily_workload_queue.popleft()
            fraction_to_run = action[0]
            work_to_run_this_hour = available_work_mwh * fraction_to_run
            unfinished_work = available_work_mwh - work_to_run_this_hour
            if unfinished_work > 1e-5:
                self._daily_workload_queue.appendleft(unfinished_work)

        #  Determine Net Energy and Battery Usage 
        # THIS IS THE SECTION WHERE THE MISSING VARIABLES ARE DEFINED
        net_energy_mwh = total_renewable_generation - work_to_run_this_hour
        grid_draw_mwh = 0

        if net_energy_mwh > 0: # Surplus
            space_left_mwh = self.battery_capacity_mwh - self._battery_soc_mwh
            energy_to_charge = min(net_energy_mwh, self.battery_max_charge_rate_mw)
            actual_charge = min(energy_to_charge, space_left_mwh) * self.battery_efficiency
            self._battery_soc_mwh += actual_charge
        else: # Deficit
            deficit_mwh = abs(net_energy_mwh)
            energy_from_battery = min(deficit_mwh, self.battery_max_charge_rate_mw, self._battery_soc_mwh)
            self._battery_soc_mwh -= energy_from_battery
            
            # If the battery can't cover the full deficit, we must draw from the grid
            grid_draw_mwh = deficit_mwh - energy_from_battery

        # Calculate the REVISED Reward 
        # Now the variables used here are all defined from the sections above.
        
        # Define the reward components
        WORK_COMPLETION_BONUS = 50.0 
        
        # POSITIVE reward for completing work.
        work_completion_reward = work_to_run_this_hour * WORK_COMPLETION_BONUS

        # POSITIVE reward for using renewables.
        direct_re_usage_reward = min(work_to_run_this_hour, total_renewable_generation)

        # NEGATIVE reward for grid cost.
        grid_cost_penalty = grid_draw_mwh * self.grid_price_per_mwh
        
        # The new reward formula
        reward = work_completion_reward + direct_re_usage_reward - grid_cost_penalty
        
        # Add New Workload and Apply End-of-Day Penalty 
        if (self._current_timestep + 1) % 24 == 0:
            remaining_work = sum(self._daily_workload_queue)
            work_penalty = remaining_work * self.work_penalty_factor
            reward -= work_penalty 
            self._daily_workload_queue.clear()
        
        next_hour_workload_mean, _ = self.workload_predictor.predict(self._current_timestep + self.planning_horizon, 1)
        self._daily_workload_queue.append(max(0, next_hour_workload_mean.flatten()[0]))
        
        #  Update Timestep and Termination 
        self._current_timestep += 1
        terminated = False
        truncated = self._current_timestep >= self.total_timesteps
        observation = self._get_observation()
        info = {
            "grid_cost": grid_cost_penalty, # Note: grid_cost_penalty is the cost value
            "grid_draw_mwh": grid_draw_mwh, 
            'battery_soc': self.battery_level, #this is new I added 
            "direct_re_usage": direct_re_usage_reward # Note: direct_re_usage_reward is the MWh value
        }
        
        return observation, reward, terminated, truncated, info
        
        # Add the next hour's available workload to the queue
        next_hour_workload_mean, _ = self.workload_predictor.predict(self._current_timestep + self.planning_horizon, 1)
        self._daily_workload_queue.append(max(0, next_hour_workload_mean.flatten()[0]))
        
        # Update Timestep and Termination 
        self._current_timestep += 1
        terminated = False
        truncated = self._current_timestep >= self.total_timesteps
        observation = self._get_observation()
        # Add the new penalty to the info dict for debugging
        info = {
            "grid_cost": grid_cost, 
            "grid_draw_mwh": grid_draw_mwh, 
            "direct_re_usage": direct_re_usage,
            "holding_penalty": holding_penalty # Good to log this
        }
        
        return observation, reward, terminated, truncated, info

    def close(self):
        print("Closing workload scheduling environment.")