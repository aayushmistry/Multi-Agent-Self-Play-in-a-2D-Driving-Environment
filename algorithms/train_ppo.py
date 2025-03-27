import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.multi_agent_env import MultiAgentDrivingEnv


def make_env():
    return Monitor(MultiAgentDrivingEnv(num_agents=10, grid_size=(10, 10)))


def train():
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=5e-5,
        n_steps=4096,
        batch_size=256,
        n_epochs=20,
        gamma=0.995,
        gae_lambda=0.90,
        clip_range=0.2,
        ent_coef=0.005,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    log_file = open('logs/training_log_1.csv', 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow(["Episode", "TotalReward", "SuccessCount", "SuccessRate", "CollisionCount", "CollisionRate"])

    rewards_list = []
    success_rates = []
    collision_rates = []

    print("Training PPO agent...")

    for ep in range(500):
        model.learn(total_timesteps=4096, reset_num_timesteps=False)
        obs = env.reset()
        done = False
        total_reward = 0
        total_successes = 0
        success_per_agent = []
        collisions_per_agent = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            total_successes += info[0]["success_count"]
            success_per_agent = info[0]["individual_successes"]
            collisions_per_agent = info[0]["collisions"]

        success_rate = np.mean(success_per_agent)
        collision_rate = np.mean(collisions_per_agent)
        collision_count = np.count_nonzero(collisions_per_agent)

        writer.writerow([
            ep,
            float(total_reward),
            total_successes,
            round(success_rate, 3),
            collision_count,
            round(collision_rate, 3)
        ])

        rewards_list.append(float(total_reward))
        success_rates.append(success_rate)
        collision_rates.append(collision_rate)

    log_file.close()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Smoothed Reward", color='tab:blue')
    smoothed = uniform_filter1d(rewards_list, size=10)
    ax1.plot(smoothed, label="Smoothed Reward", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Success Rate", color='tab:green')
    ax2.plot(success_rates, label="Success Rate", color='tab:green', alpha=0.6)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    plt.title("Training Progress: Reward and Success Rate")
    plt.savefig("assets/training_plot.png")
    plt.show()
