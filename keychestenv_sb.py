import gym

from keychestenv import KeyChestEnvironmentRandom, KeyChestGymEnv, KeyChestEnvironment
import numpy as np

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.vec_env import VecEnv
import argparse
from gym.wrappers import Monitor
from uuid import uuid1
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description="Train/evaluate the model")
parser.add_argument('--train_steps', type=int, default=250000)
parser.add_argument('--eval_episodes', type=int, default=100)
parser.add_argument('--train', required=False, action='store_true')
parser.add_argument('--variable_seed', required=False, action='store_true')
parser.add_argument('--evaluate', required=False, action='store_true')

reward = {'step': -.1, 'food_collected': 3, 'key_collected': 4, 'chest_opened': 5}

def fixed_seed_constructor(seed=42, **kwargs):
    np.random.seed(seed)
    return KeyChestEnvironmentRandom(**kwargs)

def variable_seed_constructor(**kwargs):
    return KeyChestEnvironmentRandom(**kwargs)

def make_env(variable_seed=False):

    if variable_seed:
        constructor = variable_seed_constructor
    else:
        constructor = fixed_seed_constructor


    env = KeyChestGymEnv(engine_constructor=constructor,#KeyChestEnvironmentRandom,
                         width=10, height=10, initial_health=15, food_efficiency=15,
                         reward_dict=reward, flatten_observation=True)
    return env

if __name__ == '__main__':
    args = parser.parse_args()

    checkpoint_fn = "keychest"
    if args.variable_seed:
        checkpoint_fn += "-variable-seed"
    env = make_env(variable_seed=args.variable_seed)
    #env = DummyVecEnv([make_env for _ in range(8)])

    #env = gym.make('CartPole-v1')

    print("Checkpoint path", checkpoint_fn)

    model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=f"./tb_{checkpoint_fn}/")
    try:
        model = DQN.load(checkpoint_fn)
    except Exception as e:
        print(f"Loading failed {e}")
    model.set_env(env)

    if args.train:
        model.learn(total_timesteps=args.train_steps)
        model.save(checkpoint_fn)

    if args.evaluate:
        directory = "video-" + checkpoint_fn + '-' + str(uuid1())
        env = Monitor(env, directory=directory, force=True, video_callable=lambda v: True,
                      resume=True, write_upon_reset=False)
        for i in tqdm(range(args.eval_episodes)):
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render(mode='rgb_array')
        print(f"Your videos are in {directory}")
        videos = sorted([x for x in os.listdir(directory) if x.endswith('.mp4')])
        list_fn = f"list_file_{directory}.txt"
        with open(list_fn, 'w') as f:
            for video in videos:
                f.write(f"file {directory}/{video}\n")
        os.system(f"ffmpeg -f concat -safe 0 -i {list_fn} -c copy {directory}.mp4")
        os.unlink(list_fn)