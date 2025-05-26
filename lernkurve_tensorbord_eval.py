import pandas as pd
import matplotlib.pyplot as plt

# Aus dem Tensorboard gespeicherten CSV-Datei einlesen (Pfad ggf. anpassen)
df = pd.read_csv("PPO_1.csv")

# Plotten
plt.figure(figsize=(10, 6))
plt.plot(df["Step"], df["Value"], marker="", label="Rollout Mean Reward", color="green")
plt.xlabel("Trainingssschritte in Mio")
plt.ylabel("Eval Mean Reward")
plt.title("Lernkurve (Evaluation)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


pd.read_csv("PPO_1.csv")