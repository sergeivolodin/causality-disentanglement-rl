import argparse

import gin
import gym
from time import time
import causal_util

parser = argparse.ArgumentParser("Measure performance in steps per second")
parser.add_argument("--time_seconds", type=float, default=3.0)
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--env", type=str, default="KeyChest-v0")

def get_env_performance(env, time_for_test=3.):
    """Get performance of the environment on a random uniform policy."""
    done = False
    env.reset()
    steps = 0

    time_start = time()
    while time() - time_start <= time_for_test:
        _, _, done, _ = env.step(env.action_space.sample())
        steps += 1
        if done:
            env.reset()
            steps += 1

    return steps / time_for_test


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config:
        gin.parse_config_file(args.config)
    env = causal_util.load_env(args.env)
    print(f"Created {args.env}: {env}")

    steps_per_second = get_env_performance(env, args.time_seconds)
    print("Steps per second:", steps_per_second)
