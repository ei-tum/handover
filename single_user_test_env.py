# test_single_user_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import matplotlib.pyplot as plt

class SingleUserTestEnv(gym.Env):
    def __init__(self, snr_file_path, user_id=None):
        super().__init__()
        self.user_id = user_id 
        self.snr_file_path = snr_file_path
        self.snr_data = self._load_data()
        self.num_steps = self.snr_data.shape[0]
        self.current_step = 0

        self.observation_space = spaces.Box(low=-12, high=25, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.reward_history = []
        self.total_correct_counter = 0
        self.previous_action = -1

    def _load_data(self):
        with open(self.snr_file_path, "r") as f:
            return np.array([list(map(float, line.strip().split())) for line in f])

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.total_correct_counter = 0
        self.reward_history = []
        self.previous_action = -1
        return self.snr_data[self.current_step], {}

    def step(self, action):
        snr = self.snr_data[self.current_step]
        max_snr = np.max(snr)
        chosen_snr = snr[action]

        if np.isclose(chosen_snr, max_snr, atol=0.01):
            reward = 1.0
            self.total_correct_counter += 1
        else:
            reward = -((max_snr - chosen_snr) / 23.0) ** 2

        self.reward_history.append(reward)
        self.previous_action = action
        self.current_step += 1

        done = self.current_step >= self.num_steps
        next_obs = self.snr_data[self.current_step] if not done else np.zeros_like(snr)
        return next_obs, reward, done, False, {}

    def render(self, model=None):
        if self.current_step == 0:
            print(f"\n[User {self.user_id}] (render skipped: noch kein Schritt ausgeführt)\n")
            return

        print(f"\n[User {self.user_id}] --- Zeitschritt {self.current_step} ---")
        snr = self.snr_data[self.current_step - 1]
        action = self.previous_action
        best_bs = np.where(snr == np.max(snr))[0]
        status = "✅" if action in best_bs else "❌"

        snr_str = str(np.round(snr, 2).tolist())
        print(f"SNRs (Observation): {snr_str}")
        print(f"Gewählte BS: {action} | Status: {status}")

        if model is not None:
            with torch.no_grad():
                obs_tensor = torch.tensor(snr, dtype=torch.float32).unsqueeze(0)
                dist = model.policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.detach().cpu().numpy().flatten()
                prob_vec = [f"{p:.2f}" for p in probs]
                print(f"Policy-Wahrscheinlichkeiten: [{', '.join(prob_vec)}]")

        if self.current_step == self.num_steps:
            accuracy = self.total_correct_counter / self.num_steps * 100
            print(f"\n✅ Gesamt korrekt (User {self.user_id}): {self.total_correct_counter} von {self.num_steps} ({accuracy:.2f}%)\n")


    def plot_reward_history(self):
        if not self.reward_history:
            print("Keine Belohnungshistorie vorhanden.")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(self.reward_history, label="Reward pro Schritt")
        plt.xlabel("Zeitschritt")
        plt.ylabel("Reward")
        plt.title("Reward-Verlauf der Testepisode")
        plt.grid(True)
        plt.legend()
        plt.show()

    def close(self):
        pass
