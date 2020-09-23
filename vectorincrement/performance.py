import vectorincrement
from helpers import get_env_performance
import numpy as np
from tqdm import tqdm
from time import time
import argparse
import gin

parser = argparse.ArgumentParser("Measure performance in steps per second")
parser.add_argument("--time_seconds", type=float, default=3.0)
parser.add_argument("--config", type=str, default="config/5x5.gin")


if __name__ == "__main__":
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    env = KeyChestGymEnv()

    steps_per_second = get_env_performance(env, args.time_seconds)
    print("Steps per second:", steps_per_second)
