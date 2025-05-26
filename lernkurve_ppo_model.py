import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_evalcurve(npz_path: str):
    """
    Lädt die evaluations.npz-Datei von EvalCallback, plottet die Lernkurve.

    Parameters:
    - npz_path: Pfad zur evaluations.npz-Datei (z. B. "./ppo_best_model/evaluations.npz")
    """

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Die Datei '{npz_path}' wurde nicht gefunden.")

    # Daten laden
    data = np.load(npz_path)
    timesteps = data["timesteps"]             
    results = data["results"]                  
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, marker='o', label="Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     alpha=0.2, label="±1 Std. Dev.")
    plt.title("Lernkurve des RL-Agenten (EvalCallback)")
    plt.xlabel("Trainingsschritte.")
    plt.ylabel("Durchschnittlicher Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



#  Anwendung
plot_evalcurve(
    
    #npz_path="./evaluations_1.npz",
    npz_path="./ppo_marl_logs/evaluations.npz"
    
)
