import gym

from vectorincrement.vectorincrementenv import VectorIncrementEnvironment

gym.envs.register(
    id='VectorIncrement-v0',
    entry_point=VectorIncrementEnvironment,
)
