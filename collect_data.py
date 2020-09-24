import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import keychest
from helpers import EnvDataCollector, load_env
from vectorincrement.observation_encoder import KerasEncoderWrapper
import vectorincrement
import argparse
from tqdm import tqdm
from uuid import uuid1
import pickle
import gin
import os
import gym


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
