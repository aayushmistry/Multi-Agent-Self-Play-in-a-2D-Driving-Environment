# 🧠 Multi-Agent Driving Simulator with PPO

> A modular, reinforcement-learning-ready simulator where multiple agents navigate a grid world with traffic lights, lane constraints, and realistic driving goals — all trained with a shared PPO policy.

---

## ❓ What is this project?

This is a **2D grid-based driving simulator** built using `gymnasium` and `pygame`, where **multiple autonomous agents** learn to:

- Navigate roads & intersections
- Respect traffic lights
- Avoid collisions
- Reach a shared destination

Agents are trained using **Proximal Policy Optimization (PPO)** via `stable-baselines3`.  
The project demonstrates key concepts in **multi-agent reinforcement learning (MARL)**.

---

## 🚗 Features

✅ Multi-agent control with shared PPO policy  
🟩 Traffic light logic (red/green cycling)  
📉 Success & collision rate tracking per episode  
🔄 Reward shaping: lane-following, progress, red-light penalty, collisions  
📊 CSV logging + automatic training visualization  
🧠 Built on Stable-Baselines3 + custom `gymnasium` environment  

---

## 🧩 Environment Overview

- Grid-based 10x10 map  
- Lane = 1, Intersection = 2  
- Discrete action space  
- Agent observations: `[x, y, speed]`  
- Rewards based on:
  - ✅ Staying on valid lanes
  - ✅ Obeying traffic lights
  - ✅ Progress toward goal
  - ✅ Avoiding collisions
  - ✅ Goal completion bonus

---

## 🤔 Why this project?

🔬 To research **shared-policy reinforcement learning** in a simplified urban driving context

🎓 To explore MARL concepts like:
- ✅ Reward shaping
- ✅ Vectorized environments
- ✅ Logging & evaluation metrics

🛠️ To build a **scalable & modular base** for future driving experiments:
- Decentralized control  
- Multi-goal learning  
- Curriculum learning  
- Inter-agent communication  

---

## 🧱 Project Structure

multi-agent-driving-simulator/</br>
├── envs/</br>
│   ├── multi_agent_env.py # Custom Gym environment</br>
├── algorithms/</br> 
│   ├──train_ppo.py # PPO training loop</br> 
├── configs/</br>
│   ├──ppo_config.yaml# YAML hyperparameters</br>
├── logs/ # Training CSV logs</br> 
├── assets/</br>
│   ├──10_agents.png</br>
│   ├──Flowchart.pdf # Visuals & flowcharts</br>
├── main.py # Entrypoint</br> 
├── requirements.txt</br> 
└── README.md</br>

---

# 🧭 Execution Flow

This section explains how your code components interact when training starts.</br>
main.py  
│  
├── calls → train() from algorithms/train_ppo.py  
│   │  
│   ├── calls → make_env()  
│   │   └── returns → Monitor-wrapped MultiAgentDrivingEnv from envs/multi_agent_env.py  
│   │  
│   ├── Initializes PPO agent (Stable-Baselines3)  
│   ├── Starts training loop for total_episodes  
│   │  
│   ├── For each episode:  
│   │   ├── Interacts with environment using PPO policy  
│   │   ├── Collects reward, collisions, success info  
│   │   └── Logs to → logs/training_log_1.csv  
│   │  
│   └── After training:  
│       └── Saves reward/success plot to → assets/training_plot.png

---

# 🚀 How to Run the Code

This guide explains how to set up and run the multi-agent driving simulator with PPO training.

---

### ✅ Prerequisites

Make sure you have:

- Python 3.8+
- `pip` installed
- `git` installed (for cloning the repo)

---

### 📦 Step 1: Clone the Repository

</pre>```bash
git clone https://github.com/aayushmistry/Multi-Agent-Self-Play-in-a-2D-Driving-Environment.git
```</pre>


### 📥 Step 2: Install dependencies

</pre>```pip install -r requirements.txt```</pre>


### ⚙️ Step 3: Run training

</pre>```python main.py```</pre>

---

# 📊 Sample Output

After training, you’ll get:

- 🗂️ CSV log: `logs/training_log_1.csv`  
  Contains episode-wise metrics like total reward, success rate, and collisions.

- 📈 Plot: `assets/training_plot.png`  
  A graph showing reward improvement and agent success rate over time.

---

### 📈 Training Curve Example

> This plot shows the smoothed total reward and success rate during PPO training with 10 agents.

[🔗 View Training_Curve](results/10_agents/10_agents.png)


---

### 🧾 Sample CSV Output (`training_log_1.csv`)

| Episode | TotalReward | SuccessCount | SuccessRate | CollisionCount | CollisionRate |
|---------|-------------|--------------|--------------|----------------|----------------|
| 0       | 15.25       | 2            | 0.20         | 1              | 0.10           |
| 1       | 22.75       | 4            | 0.40         | 0              | 0.00           |
| ...     | ...         | ...          | ...          | ...            | ...            |
| 499     | 87.95       | 7            | 0.70         | 2              | 0.20           |

---

## 📊 Experiment Showcase: 50 vs 100 Agents (Without Scripts)

As part of scalability testing, PPO training was also run using **50** and **100** agents.  
These results are uploaded here **only as performance visualizations**, and **training scripts are intentionally not included** for these experiments.

---

| Agents     | Smoothed Reward | Success Rate | Collision Rate | Notes                              |
|------------|------------------|---------------|----------------|-------------------------------------|
| 50 Agents  | ⚠️ Unstable     | 🔸 Very Low   | 🔼 Medium       | Agents begin interfering frequently |
| 100 Agents | ❌ Volatile     | 🔴 Near Zero  | 🔺 Extremely High | Policy collapses under crowding    |

---

### 📉 PPO Reward Progress (Visualization Only)

#### 🔹 50 Agents  
[🔗 View 50-Agent Plot](results/50_agents/50_agents.png)


#### 🔹 100 Agents  
[🔗 View 100-Agent Plot](results/100_agents/100_agents.png)


---

📝 **Note:**  
These experiments were conducted to analyze multi-agent self-play behavior under increasing population sizes.  
They help highlight challenges in coordination, reward sparsity, and PPO’s generalization limits.

📂 Only results are shown. No code/scripts are provided for these specific runs to keep the repository focused on the base setup.


## ✅ These metrics are useful for analyzing:
- Agent learning progress
- Generalization with different agent counts (10, 50, 100)
- Policy effectiveness and reward shaping

---

## 📄 Documentation

- [🚧 Challenges & Bottleneck Analysis](CHALLENGES.md)


## 🔭 Future Enhancements

🧩 Model checkpointing (save & resume PPO agents)  
🎓 Curriculum training with dynamic difficulty levels  
🗺️ Dynamic map generation & multiple environment scenarios  
🎯 Multi-goal or hierarchical task setups  
🪄 Real-time debugging overlays for policy behavior

