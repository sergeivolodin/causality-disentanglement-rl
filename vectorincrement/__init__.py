from vectorincrement.vectorincrementenv import VectorIncrementEnvironment
import gym
import gin


gym.envs.register(
     id='VectorIncrement-v0',
     entry_point=VectorIncrementEnvironment,
)

@gin.configurable
def load_env(env_name, wrappers=None, **kwargs):
    """Load an environment, configurable via gin."""
    print(f"Make environment {env_name} {wrappers} {kwargs}")
    env = gym.make(env_name, **kwargs)
    if wrappers is None:
        wrappers = []
    for wrapper in wrappers[::-1]:
        env = wrapper(env)
    return env
