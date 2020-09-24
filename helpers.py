from time import time
import gym
import gin
import numpy as np
from gym import Wrapper
from gym.wrappers import TimeLimit
import gin
from vectorincrement.observation_encoder import ObservationScaleWrapper


@gin.configurable
def load_env(env_name, time_limit=None, obs_scaler=None, wrappers=None, **kwargs):
    """Load an environment, configurable via gin."""
    print(f"Make environment {env_name} {wrappers} {kwargs}")
    env = gym.make(env_name, **kwargs)
    if time_limit:
        env = TimeLimit(env, time_limit)
    if obs_scaler:
        env = ObservationScaleWrapper(env, obs_scaler)
    if wrappers is None:
        wrappers = []
    for wrapper in wrappers[::-1]:
        env = wrapper(env)
    return env

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
                                     'info': info, 'action': action})
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


def args_to_dict(**kwargs):
    """Convert kwargs to a dictionary."""
    return dict(kwargs)


@gin.configurable
def reward_as_dict(**kwargs):
    return args_to_dict(**kwargs)

@gin.configurable
class np_random_seed(object):
    """Run stuff with a fixed random seed."""
    # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator
    def __init__(self, seed=42):
        self.seed = seed
        self.st0 = None

    def __enter__(self):
        self.st0 = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, type_, value, traceback):
        np.random.set_state(self.st0)

@gin.configurable
def with_fixed_seed(fcn, seed=42, **kwargs):
    """Call function with np.random.seed fixed."""

    with np_fixed_seed(seed=seed):
        # computing the function
        result = fcn(**kwargs)

    return result


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
