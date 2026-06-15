# MaxREpaper
# Research Title: MaxRE: Stable Reinforcement-Learning-Based Scheduling for Sustainable Data Centers

###  What is this research work?
This research aims to improve the scheduling workload of data centers by utilizing renewable energy sources, particularly wind and solar, and implementing an AI agent. Using the uncertainty of weather forecasts in scheduling workloads in a data center can cause unexpected situations like power failures, system overloads, etc. Our work aimed to address the instabilities of data centers’ scheduling workloads using the framework named MaxRE, which combined GP uncertainty and deep learning algorithms like Soft Actor-Critic agents. This work reduces operational expenses, ensures timely processing of energy-intensive tasks during peak renewable availability, and schedules them effectively.
Existing works have been done only in a deterministic way, even though their work has applied machine learning and deep learning approaches. In our work, we address the uncertainty of weather estimates, we have trained the agent to balance continuous usage with systematic performance in data center scheduling, and we also prove with theoretical guarantees that lacked in prior work.
## Benefits & Goals
- **Sustainability:** Optimizes workload timing to coincide with peak solar and wind energy production.
- **Accuracy:** Utilizes Gaussian Process regression to handle uncertainty in weather-dependent energy sources.
- **Efficiency:** Reduces reliance on the traditional power grid, lowering operational costs and carbon footprint.
- **Autonomous Decision Making:** Utilizes a Soft Actor-Critic (SAC) agent to handle complex scheduling environments.

###  How to Run This Code
To get this research work running on your local system, follow these steps:

1. **Install Requirements:** 
   - Install Python (at leastPython 3.13.3 and above)
   - Install the 'requests' library: `pip install requests` like Required Libraries: pandas, numpy, scikit-learn, stable-baselines3, pickle
2.***ngrok:** Used for secure tunneling to monitor Reinforcement Learning progress via TensorBoard remotely.
3. **Run** Open your system and run the python file as I listed one by one
1. **Data Preprocessing**
---Clean and prepare the raw datasets for processing.---
-Run preprocess_solar.py
-Run preprocess_workload.py
-Run wind-power-raw.py
2. **Gaussian Process Model Training**
---Train the forecasting models to generate .pkl files.---
-Run workload_trained_GPmodel.py
-Run wind_trained_GPmodel.py
-Run solar_trained_GPmodel.py
Output: Generates trained-gp-model-solar-mw.pkl, trained-gp-model-wind-power-mw.pkl, etc.
3. **Prediction Module**
---Combine all models to generate the 24-hour forecast data.---
-Run All_prediction_module.py
Output: combined-24hr-forecast.csv
4.**Reinforcement Learning (RL) Setup**
---Initialize the environment and train the scheduling agent.---
Environment: Run workload_scheduling-env1.py 
Agent Training: Run multiple_train.py
Algorithm: Soft Actor-Critic (SAC).
Output: Training logs and the final model: sac-scheduler-final.zip.
***Since Reinforcement Learning (SAC) can take a long time to train, ngrok was used to monitor the agent's learning progress (reward curves, loss, etc.) in real-time through TensorBoard.***
**To monitor training via ngrok:**
1. Run multiple_train.py
2. To start tensorboard please open a new terminal and launch the tensorboard like below
   tensorboard --logdir=./tboard_logs/ --port=6006
  - and open new terminal again to create public url like below
    ngrok http 6006
    **Accessing the Dashboard***
Open the Forwarding URL provided by ngrok (e.g., https://xxxx-xxxx.ngrok-free.app).
You will see live updates of the following metrics:
ep_rew_mean (Reward): This should increase over time as the agent learns to use renewable energy more efficiently.
loss: Monitor the stability of the neural network's learning process. 
5. **Advanced Training** 
For training across multiple logs or models, use:
-Run multiple_train.py

