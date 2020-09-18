import gym

from keychestenv import KeyChestEnvironmentRandom, KeyChestGymEnv, KeyChestEnvironment
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

reward = {'step': -.1, 'food_collected': 3, 'key_collected': 4, 'chest_opened': 5}

def fixed_seed_constructor(seed=42, **kwargs):
    np.random.seed(seed)
    return KeyChestEnvironmentRandom(**kwargs)

env = KeyChestGymEnv(engine_constructor=fixed_seed_constructor,#KeyChestEnvironmentRandom,
                     width=10, height=10, initial_health=15, food_efficiency=15,
                     reward_dict=reward, flatten_observation=True)

#env = gym.make('CartPole-v1')

model = DQN(MlpPolicy, env, verbose=1)
try:
    model = DQN.load("keychest")
except Exception as e:
    print(f"Loading failed {e}")
model.set_env(env)
model.learn(total_timesteps=25000)
model.save("keychest")

del model # remove to demonstrate saving and loading

model = DQN.load("keychest")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

