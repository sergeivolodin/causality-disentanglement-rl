import argparse
import os
from uuid import uuid1

import gin
from gym.wrappers import Monitor
# from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as DQNMlpPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv
from tqdm import tqdm
from causal_util.helpers import find_gin_parameter

from causal_util import load_env
from encoder.observation_encoder import KerasEncoderWrapper
from encoder.observation_encoder_sb import KerasEncoderVecWrapper
from datetime import datetime


parser = argparse.ArgumentParser(description="Train/evaluate the model")
parser.add_argument('--train_steps', type=int, default=250000)
parser.add_argument('--eval_episodes', type=int, default=100)
parser.add_argument('--train', required=False, action='store_true')
parser.add_argument('--evaluate', required=False, action='store_true')
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--n_env', type=int, default=8)
parser.add_argument('--trainer', type=str, default="PPO")


if __name__ == '__main__':
    args = parser.parse_args()

    # how to train the policy?
    trainers = {'PPO': PPO2, 'DQN': DQN}
    policies = {'PPO': MlpPolicy, 'DQN': DQNMlpPolicy}
    vectorize = {'PPO': True, 'DQN': False}

    do_vectorize = vectorize[args.trainer]
    TrainerClass = trainers[args.trainer]
    PolicyClass = policies[args.trainer]

    config_basename = "none"
    wrap_keras_encoder = False
    if args.config:
        gin.parse_config_file(args.config)
        config_basename = os.path.basename(args.config)[:-4]

        idx, lst = find_gin_parameter(KerasEncoderWrapper, "load_env.wrappers")
        if idx is not None:
            del lst[idx]
            wrap_keras_encoder = True
        gin.bind_parameter("load_env.wrappers", lst)

    def make_env():
        return load_env()

    checkpoint_fn = f"env-config-{args.trainer}-{config_basename}"

    if do_vectorize:
        env = DummyVecEnv([make_env for _ in range(args.n_env)])
        if wrap_keras_encoder:
            env = KerasEncoderVecWrapper(env)
    else:
        env = make_env()
        if wrap_keras_encoder:
            env = KerasEncoderWrapper(env)

    print("Checkpoint path", checkpoint_fn)
    tb_path = f"./tb_{checkpoint_fn}_{datetime.now().strftime('%Y%M%d_%H%M%S')}/"
    print("TensorBoard path", tb_path)

    model = TrainerClass(PolicyClass, env, verbose=1, tensorboard_log=tb_path)
    try:
        model = TrainerClass.load(checkpoint_fn)
        model.tensorboard_log = tb_path
    except Exception as e:
        print(f"Loading failed {e}")
    model.set_env(env)

    if args.train:
        model.learn(total_timesteps=args.train_steps)
        model.save(checkpoint_fn)

    if args.evaluate:
        directory = "video-" + checkpoint_fn + '-' + str(uuid1())
        env = make_env()
        if wrap_keras_encoder:
            env = KerasEncoderWrapper(env)
        env = Monitor(env, directory=directory, force=True, video_callable=lambda v: True,
                      resume=True, write_upon_reset=False)
        with tqdm(total=args.eval_episodes) as pbar:
            for i in range(args.eval_episodes):
                R = 0
                obs = env.reset()
                done = False
                while not done:
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    env.render(mode='rgb_array')
                    R += rewards
                pbar.update(1)
                pbar.set_postfix(reward=R)
        print(f"Your videos are in {directory}")
        videos = sorted([x for x in os.listdir(directory) if x.endswith('.mp4')])
        list_fn = f"list_file_{directory}.txt"
        with open(list_fn, 'w') as f:
            for video in videos:
                f.write(f"file {directory}/{video}\n")
        os.system(f"ffmpeg -f concat -safe 0 -i {list_fn} -c copy {directory}.mp4")
        os.unlink(list_fn)
        print(f"Video is in {directory}.mp4")
