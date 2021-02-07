import argparse

import gin
import gym
from time import time
import causal_util
from sparse_causal_model_learner_rl.sacred_gin_tune.sacred_wrapper import load_config_files

parser = argparse.ArgumentParser("Measure performance in steps per second")
parser.add_argument("--time_seconds", type=float, default=3.0)
parser.add_argument("--config", type=str, default=None)


def get_env_performance(env, time_for_test=3.):
    """Get performance of the environment on a random uniform policy."""
    env.reset()
    steps = 1
    episodes = 1

    time_start = time()
    while time() - time_start <= time_for_test:
        _, _, done, _ = env.step(env.action_space.sample())
        steps += 1
        if done:
            env.reset()
            episodes += 1
            steps += 1

    return steps / time_for_test, episodes / time_for_test


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config:
        load_config_files([args.config])
    env = causal_util.load_env()
    print(f"Created {env}")

    steps_per_second, episodes_per_second = get_env_performance(env, args.time_seconds)
    print("Steps per second:", steps_per_second)
    print("Episodes per second:", episodes_per_second)
    print("Mean steps in episode:", steps_per_second / episodes_per_second)
