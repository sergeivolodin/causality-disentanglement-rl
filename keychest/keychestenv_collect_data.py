from keychestenv import KeyChestGymEnv
from helpers import EnvDataCollector
import argparse
from tqdm import tqdm
from uuid import uuid1
import pickle
import gin
import os


parser = argparse.ArgumentParser("Collect data from the environment and save it")
parser.add_argument('--n_episodes', type=int, default=1000)
parser.add_argument('--config', type=str, default="config/5x5.gin")




if __name__ == '__main__':
    args = parser.parse_args()
    fn_out = f"episodes-{args.n_episodes}-config-{os.path.basename(args.config)}-{str(uuid1())}.pkl"

    gin.parse_config_file(args.config)
    env = KeyChestGymEnv()
    env = EnvDataCollector(env)

    for i in tqdm(range(args.n_episodes)):
        done = False
        env.reset()
        while not done:
            _, _, done, _ = env.step(env.action_space.sample())

    pickle.dump(env.raw_data, open(fn_out, 'wb'))
    print("Output: ", fn_out)
