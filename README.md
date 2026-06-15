# MaxREpaper
# Research Title: MaxRE: Stable Reinforcement-Learning-Based Scheduling for Sustainable Data Centers

###  What is this research work?
This research aims to improve the scheduling workload of data centers by utilizing renewable energy sources, particularly wind and solar, and implementing an AI agent. Using the uncertainty of weather forecasts in scheduling workloads in a data center can cause unexpected situations like power failures, system overloads, etc. Our work aimed to address the instabilities of data centers’ scheduling workloads using the framework named MaxRE, which combined GP uncertainty and deep learning algorithms like Soft Actor-Critic agents. This work reduces operational expenses, ensures timely processing of energy-intensive tasks during peak renewable availability, and schedules them effectively.
Existing works have been done only in a deterministic way, even though their work has applied machine learning and deep learning approaches. In our work, we address the uncertainty of weather estimates, we have trained the agent to balance continuous usage with systematic performance in data center scheduling, and we also prove with theoretical guarantees that lacked in prior work.
###  How to Run This Code
To get this research work running on your local system, follow these steps:

1. **Install Requirements:** 
   - Install Python (at leastPython 3.13.3 and above)
   - Install the 'requests' library: `pip install requests`
2. **Clone the Repo:** `git clone []`
3. **Run the App:** Open your terminal and type `python main.py`

### ⚙️ How it Works (The Process)
1. **Data Collection:** The script connects to the OpenWeather API to fetch current weather strings.
2. **Logic Engine:** The code checks the 'precipitation' percentage. If it is over 50%, it triggers a 'Rain Alert'.
3. **Display:** The final result is displayed as a clean desktop notification for the user.
