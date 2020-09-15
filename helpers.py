from time import time
import gym

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
