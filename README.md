# Handover-Simulation mit Reinforcement Learning

# Das Projekt ist gerade in der Bearbeitung

Dieses Repository enthält eine Python-Simulation des Handover-Prozesses in einem Mobilfunknetz mit 15 Benutzern und 4 Basisstationen.
Dabei wird Reinforcement Learning mit der PPO-Algorithmus-Implementierung aus Stable-Baselines3 eingesetzt.

# Projektübersicht
Benutzer: 15
Basisstationen: 4
Modell: PPO (Proximal Policy Optimization) – Stable-Baselines3
Trainingsdaten: SNR-Werte der 15 Benutzer

Genauigkeit des trainierten Model nach einer Million Steps beträgt: 99,66 %

# Installation
Repository klonen:
bash:
git clone https://github.com/ei-tum/handover.git

bash:
cd handover

Abhängigkeiten installieren:
pip install -r requirements.txt
# Modell testen
Das trainierte Modell kann direkt über die Datei test_marl_ppo.py getestet werden:

bash:
python test_marl_ppo.py

# Modell Trainieren
um das Model zu trainieren, kannst du entweder direkt mit dem Befehl trainiern
bash:
python train_marl_ppo.py

oder du kannst in der Datei train_marl_ppo.py Änderungen vornehmen, wie du möchtest und dann 
bash
python train_marl_ppo.py 
ausführen

