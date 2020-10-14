from util.helpers import get_env_performance
import argparse
import gin
import gym

parser = argparse.ArgumentParser("Measure performance in steps per second")
parser.add_argument("--time_seconds", type=float, default=3.0)
parser.add_argument("--config", type=str, default="keychest/config/5x5.gin")
parser.add_argument("--env", type=str, default="KeyChest-v0")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.config:
        gin.parse_config_file(args.config)
    env = gym.make(args.env)
    print(f"Created {args.env}: {env}")

    steps_per_second = get_env_performance(env, args.time_seconds)
    print("Steps per second:", steps_per_second)
