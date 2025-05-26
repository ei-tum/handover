# single_user_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import random

class SingleUserEnv(gym.Env):
    def __init__(self, snr_file_paths):
        super().__init__()
        self.snr_file_paths = snr_file_paths  # Liste von 20 Dateien
        self.snr_data = self._load_random_dataset() # Initialisiere mit einer zufälligen Datei
        self.num_steps = self.snr_data.shape[0] # Anzahl der Schritte in der Datei
        self.current_step = 0

        self.action_space = spaces.Discrete(4)  # 4 BS
        self.observation_space = spaces.Box(low=-12, high=25, shape=(4,), dtype=np.float32)

    def _load_random_dataset(self):
        selected_path = random.choice(self.snr_file_paths)
        with open(selected_path, "r") as f:
            data = [list(map(float, line.strip().split())) for line in f]
        return np.array(data)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snr_data = self._load_random_dataset()
        self.num_steps = self.snr_data.shape[0]
        self.current_step = 0
        return self.snr_data[self.current_step], {}

    def step(self, action):
        obs = self.snr_data[self.current_step]
        max_snr = np.max(obs)
        chosen_snr = obs[action]

        # Quadratisch differenzierte Belohnung mit Bonus bei richtiger Wahl
        if np.isclose(chosen_snr, max_snr, atol=0.01):
            reward = 1.0
        else:
            reward = -((max_snr - chosen_snr) / 23.0) **2

        self.current_step += 1
        done = self.current_step >= self.num_steps
        next_obs = self.snr_data[self.current_step] if not done else np.zeros_like(obs)
        return next_obs, reward, done, False, {}

    def close(self):
        pass
