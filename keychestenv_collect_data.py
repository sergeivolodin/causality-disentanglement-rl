from keychestenv import KeyChestGymEnv, KeyChestEnvironmentRandom
from gym import Wrapper
import argparse
from tqdm import tqdm
from uuid import uuid1
import pickle


parser = argparse.ArgumentParser("Collect data from the environment and save it")
parser.add_argument('--n_episodes', type=int, default=500)

class EnvDataCollector(Wrapper):
    """Collects data from the environment."""

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.rollouts = []
        self.current_rollout = []
        super(EnvDataCollector, self).__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.current_rollout.append({'observation': obs, 'reward': rew, 'done': done,
                                     'info': info})
        return (obs, rew, done, info)

    def reset(self, **kwargs):
        if self.current_rollout:
            self.rollouts.append(self.current_rollout)
            self.current_rollout = []
        obs = self.env.reset(**kwargs)
        self.current_rollout.append({'observation': obs})
        return obs

    @property
    def raw_data(self):
        return self.rollouts


if __name__ == '__main__':
    args = parser.parse_args()
    fn_out = f"episodes-{args.n_episodes}-{str(uuid1())}.pkl"

    reward = {'step': -.1, 'food_collected': 3, 'key_collected': 4, 'chest_opened': 5}

    env = KeyChestGymEnv(engine_constructor=KeyChestEnvironmentRandom,
                         width=10, height=10, initial_health=15, food_efficiency=15,
                         reward_dict=reward)

    env = EnvDataCollector(env)

    for i in tqdm(range(args.n_episodes)):
        done = False
        env.reset()
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())

    pickle.dump(env.raw_data, open(fn_out, 'wb'))