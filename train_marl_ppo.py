import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from single_user_env import SingleUserEnv
from single_user_test_env import SingleUserTestEnv

#  Konfiguration
NUM_USERS = 15
TOTAL_TIMESTEPS = 1000_000
DATA_DIR_TRAIN = "./516000000_train"
DATA_DIR_TEST = "./515000000_test"
LOG_DIR = "./ppo_marl_logs/"
os.makedirs(LOG_DIR, exist_ok=True)

#  Trainingsdateien sammeln
user_file_paths = {
    user_id: sorted(glob.glob(os.path.join(DATA_DIR_TRAIN, f"snr_values-seed_516000000-mac_30-mic_20_user_{user_id}_*.txt")))
    for user_id in range(NUM_USERS)
}

#  Trainingsumgebungen (VecEnv, shared policy)
def make_env(user_id):
    def _init():
        return Monitor(SingleUserEnv(user_file_paths[user_id]))
    return _init

train_env = DummyVecEnv([make_env(i) for i in range(NUM_USERS)])

#  Testumgebung (1 File pro Nutzer)
test_user_paths = {
    user_id: sorted(glob.glob(os.path.join(DATA_DIR_TEST, f"snr_values-seed_515000000-mac_30-mic_20_user_{user_id}_*.txt")))[0]
    for user_id in range(NUM_USERS)
}

def make_test_env(user_id):
    def _init():
        return Monitor(SingleUserTestEnv(test_user_paths[user_id]))
    return _init

test_env = DummyVecEnv([make_test_env(i) for i in range(NUM_USERS)])

#  PPO-Konfiguration
"""policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[64, 64]),
)
"""
# Diesr Modell wurde mit Defualt-Parametern trainiert, die in der Regel gut funktionieren.

model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    #learning_rate=3e-4,
    #n_steps=2048,
    #batch_size=512,
    #gamma=0.99,
    #ent_coef=0.02,
    #clip_range=0.2,
    #policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_tensorboard_log",
)

#  Callbacks
eval_callback = EvalCallback(
    test_env,
    best_model_save_path=LOG_DIR,
    log_path=LOG_DIR,
    eval_freq=5000,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=os.path.join(LOG_DIR, "checkpoints"),
    name_prefix="ppo_marl"
)

callback = CallbackList([eval_callback, checkpoint_callback])

#  Training starten
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

# Speichern des letzten Modells.
#Eigentlich wird bei evalcallback das beste Modell gespeichert, aber hier speichern wir das finale Modell.
model.save("ppo_marl_shared_policy_final")
print("✅ Modell gespeichert unter: ppo_marl_shared_policy_final")
