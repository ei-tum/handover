# test_shared_policy.py

import os
import glob
from stable_baselines3 import PPO
from single_user_test_env import SingleUserTestEnv

# 🔧 Pfade
MODEL_PATH = "./ppo_marl_logs/best_model.zip" #Der Path zum besten Modell
#MODEL_PATH = "./ppo_marl_shared_policy_final" # Der Pfad zum finalen Modell, ich habe dies aber wieder gelöscht.
TEST_DIR = "./515000000_test"
NUM_USERS = 15
PLOT = False  # Setze auf True, um die Plotsfunktion zu aktivieren

#  Modell laden
model = PPO.load(MODEL_PATH)

#  Testdateien sammeln (eine Datei je Nutzer)
test_files = sorted(glob.glob(os.path.join(TEST_DIR, "snr_values-seed_515000000-mac_30-mic_20_user_*_0.txt")))
assert len(test_files) == NUM_USERS, f"Erwarte {NUM_USERS} Testdateien, gefunden: {len(test_files)}"

#  15 Testumgebungen initialisieren
envs = [
    SingleUserTestEnv(path, user_id=i)
    for i, path in enumerate(test_files)]

observations = [env.reset()[0] for env in envs]
dones = [False] * NUM_USERS

print("\n Starte Auswertung aller 15 Nutzer...\n")

while not all(dones):
    for i, env in enumerate(envs):
        if not dones[i]:
            action, _ = model.predict(observations[i], deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            observations[i] = obs
            dones[i] = done
            env.render(model)

# Ergebnis anzeigen
gesamt_korrekt = sum(env.total_correct_counter for env in envs)
gesamt_schritte = sum(env.num_steps for env in envs)
accuracy = gesamt_korrekt / gesamt_schritte * 100

print(f"\n✅ Gesamt-Accuracy über alle Nutzer: {gesamt_korrekt} von {gesamt_schritte} korrekt ({accuracy:.2f}%)")

#  Optional: Reward-Plots anzeigen
if PLOT: 
    for i, env in enumerate(envs):
        print(f"\n Nutzer {i} — Reward-Verlauf:")
        env.plot_reward_history()
