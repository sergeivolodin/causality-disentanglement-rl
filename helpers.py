from time import time
import gym
import gin
import numpy as np


def args_to_dict(**kwargs):
    """Convert kwargs to a dictionary."""
    return dict(kwargs)


@gin.configurable
def reward_as_dict(**kwargs):
    return args_to_dict(**kwargs)

@gin.configurable
def with_fixed_seed(fcn, seed=42, **kwargs):
    """Call function with np.random.seed fixed."""
    # https://stackoverflow.com/questions/32172054/how-can-i-retrieve-the-current-seed-of-numpys-random-number-generator

    # get the initial state of the RNG
    st0 = np.random.get_state()

    # setting the seed
    np.random.seed(seed)

    # computing the function
    result = fcn(**kwargs)

    # set the state back to what it was originally
    np.random.set_state(st0)

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
