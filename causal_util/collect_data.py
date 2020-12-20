import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from causal_util import load_env
from encoder.observation_encoder import KerasEncoderWrapper
import argparse
from tqdm import tqdm
from uuid import uuid1
import pickle
import gin
from gym import Wrapper


def compute_reward_to_go(rewards_episode, gamma=0.95):
    """Compute discounted rewards from current point.

    Args:
        rewards_episode: iterable with rewards from a single episode
        gamma: discount factor

    Returns:
        iterablee of the same length as rewards_episode with rewards-to-go

    Property:
        Expectation_episode[reward-to-go] = value function
    """

    prev_rtg = 0
    reward_to_go = []
    for r in rewards_episode[::-1]:
        rtg = r + gamma * prev_rtg
        reward_to_go.append(rtg)
        prev_rtg = rtg
    return reward_to_go[::-1]

class EnvDataCollector(Wrapper):
    """Collects data from the environment."""

    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata
        self.clear()
        super(EnvDataCollector, self).__init__(env)

    def clear(self):
        self.rollouts = []
        self.current_rollout = []
        self.steps = 0


    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.steps += 1
        self.current_rollout.append({'observation': obs, 'reward': rew, 'done': done,
                                     'info': info, 'action': action})
        return (obs, rew, done, info)

    def flush(self):
        if self.current_rollout:
            self.rollouts.append(self.current_rollout)
            self.current_rollout = []

    def reset(self, **kwargs):
        self.steps += 1
        self.flush()
        obs = self.env.reset(**kwargs)
        self.current_rollout.append({'observation': obs})
        return obs

    @property
    def raw_data(self):
        return self.rollouts


parser = argparse.ArgumentParser("Collect data from the environment and save it")
parser.add_argument('--n_episodes', type=int, default=1000)
parser.add_argument('--config', type=str, default="keychest/config/5x5.gin")
parser.add_argument('--wrap_keras_encoder', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()
    config_descr = '_'.join(args.config.split('/'))
    fn_out = f"episodes-{args.n_episodes}-config-{config_descr}-{str(uuid1())}.pkl"

    gin.parse_config_file(args.config)
    gin.bind_parameter("observation_encoder.KerasEncoder.model_callable", None)
    env = load_env()
    if args.wrap_keras_encoder:
        env = KerasEncoderWrapper(env)
    print(f"Created environment {env}")
    env = EnvDataCollector(env)

    for i in tqdm(range(args.n_episodes)):
        done = False
        env.reset()
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())

    pickle.dump(env.raw_data, open(fn_out, 'wb'))
    print("Output: ", fn_out)
