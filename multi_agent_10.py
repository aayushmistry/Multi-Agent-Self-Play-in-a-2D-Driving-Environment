import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import uniform_filter1d
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import csv

class MultiAgentDrivingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, num_agents=10, grid_size=(10, 10)):
        super(MultiAgentDrivingEnv, self).__init__()
        self.num_agents = num_agents
        self.grid_size = grid_size

        self.lane_map = np.array([
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 2, 2, 2, 2, 0, 1, 0],
            [0, 0, 1, 1, 2, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 2, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 2, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        ])

        self.traffic_lights = {
            (3, 3): {"state": "red", "timer": 0},
            (4, 4): {"state": "green", "timer": 0}
        }
        self.traffic_cycle = 5

        self.action_space = spaces.MultiDiscrete([7] * num_agents)
        self.observation_space = spaces.Box(low=0, high=max(grid_size), shape=(num_agents, 3), dtype=np.float32)

        self.agents = np.zeros((num_agents, 3))
        self.screen_size = 500
        self.cell_size = self.screen_size // max(grid_size)
        self.screen = None
        self.clock = None
        self.max_steps = 150
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agents[:, 0] = np.random.randint(0, self.grid_size[0], size=self.num_agents)
        self.agents[:, 1] = np.random.randint(0, self.grid_size[1], size=self.num_agents)
        self.agents[:, 2] = 1
        self.current_step = 0
        for light in self.traffic_lights.values():
            light["state"] = "red"
            light["timer"] = 0
        return self.agents, {}

    def step(self, actions):
        self.current_step += 1
        rewards = np.zeros(self.num_agents)
        self.collided = np.zeros(self.num_agents, dtype=bool)

        for pos, light in self.traffic_lights.items():
            light["timer"] += 1
            if light["timer"] >= self.traffic_cycle:
                light["state"] = "green" if light["state"] == "red" else "red"
                light["timer"] = 0

        move_left = actions == 2
        move_right = actions == 3
        move_up = actions == 4
        move_down = actions == 5

        new_positions = self.agents[:, :2].copy()
        new_positions[:, 0] -= move_left
        new_positions[:, 0] += move_right
        new_positions[:, 1] -= move_up
        new_positions[:, 1] += move_down
        new_positions = np.clip(new_positions, 0, self.grid_size[0] - 1)

        position_counts = {}
        for i in range(self.num_agents):
            key = tuple(new_positions[i])
            position_counts[key] = position_counts.get(key, 0) + 1

        goal = np.array([self.grid_size[0] - 1, self.grid_size[1] - 1])
        prev_distances = np.linalg.norm(self.agents[:, :2] - goal, axis=1)
        new_distances = np.linalg.norm(new_positions - goal, axis=1)

        for i in range(self.num_agents):
            x, y = int(new_positions[i, 0]), int(new_positions[i, 1])
            key = (x, y)

            if key in self.traffic_lights and self.traffic_lights[key]["state"] == "red":
                rewards[i] -= 0.5
                continue

            cell_type = self.lane_map[y, x]
            if cell_type == 1:
                rewards[i] += 0.6
            elif cell_type == 2:
                rewards[i] += 0.4
            else:
                rewards[i] -= 1.0

            if position_counts[key] > 1:
                rewards[i] -= 3.0
                self.collided[i] = True
            else:
                self.agents[i, :2] = new_positions[i]

            progress = prev_distances[i] - new_distances[i]
            rewards[i] += 0.2 * progress

            if np.linalg.norm(new_positions[i] - goal) < 2:
                rewards[i] += 1.0

            rewards[i] += 0.05

        reached_goal = np.all(self.agents[:, :2] == goal, axis=1)
        success_count = np.count_nonzero(reached_goal)
        rewards[reached_goal] += 100.0

        for idx, reached in enumerate(reached_goal):
            if reached:
                print(f"Agent {idx} reached the goal.")

        done = np.any(reached_goal) or (self.current_step >= self.max_steps)
        info = {
            "success_count": success_count,
            "individual_successes": reached_goal.tolist(),
            "collisions": self.collided.tolist()
        }
        return self.agents, np.sum(rewards), done, False, info

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))

        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                if self.lane_map[y, x] == 1:
                    color = (200, 200, 200)
                elif self.lane_map[y, x] == 2:
                    color = (255, 200, 0)
                else:
                    color = (50, 50, 50)
                pygame.draw.rect(self.screen, color,
                                 (x * self.cell_size, y * self.cell_size,
                                  self.cell_size, self.cell_size))

        for pos, light in self.traffic_lights.items():
            lx, ly = pos
            color = (255, 0, 0) if light["state"] == "red" else (0, 255, 0)
            pygame.draw.circle(self.screen, color,
                               (lx * self.cell_size + self.cell_size // 2,
                                ly * self.cell_size + self.cell_size // 2),
                               self.cell_size // 4)

        for idx, agent in enumerate(self.agents):
            ax, ay = int(agent[0] * self.cell_size), int(agent[1] * self.cell_size)
            color = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (100, 100, 255), (255, 100, 100),
                (100, 255, 100), (150, 150, 0)
            ][idx % 10]
            pygame.draw.rect(self.screen, color,
                             (ax, ay, self.cell_size, self.cell_size))

        pygame.display.flip()

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None


def make_env():
    return Monitor(MultiAgentDrivingEnv(num_agents=10, grid_size=(10, 10)))
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

log_file = open('training_log_1.csv', 'w', newline='')
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
plt.show()


