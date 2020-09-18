from keychestenv import KeyChestEnvironmentRandom, KeyChestGymEnv, KeyChestEnvironment
from keychestenv_gui import jupyter_gui
from keychestenv_gofa import features_for_obs, max_reward, hardcoded_policy_step
from matplotlib import pyplot as plt
from helpers import get_env_performance
import numpy as np
from tqdm import tqdm
from time import time

reward = {'step': -.1, 'food_collected': 3, 'key_collected': 4, 'chest_opened': 5}

env = KeyChestGymEnv(engine_constructor=KeyChestEnvironmentRandom,
                     width=10, height=10, initial_health=15, food_efficiency=15,
                     reward_dict=reward)

print("Steps per second:", get_env_performance(env, 3))
