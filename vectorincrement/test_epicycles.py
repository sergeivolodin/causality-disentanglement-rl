from vectorincrement.epicycles import RocketEpicycleEnvironment
from scipy.spatial.distance import cdist
import numpy as np

def rollout(env, n_steps=100):
    done = False
    obss = []
    obss.append(env.reset())
    for i in range(n_steps):
        obs, _, _, _ = env.step(env.action_space.sample())
        obss.append(obs)
    return obss

def test_rocket_env():
    env1 = RocketEpicycleEnvironment(epicycles=False)
    env2 = RocketEpicycleEnvironment(epicycles=True)
    
    for env in [env1, env2]:
        for _ in range(10):
            A = np.array(rollout(env))
            A = A.reshape(A.shape[0], -1)
            dst = cdist(A, A) > 0
            n = range(dst.shape[0])
            dst[n, n] = True
            assert np.all(dst), (env)
