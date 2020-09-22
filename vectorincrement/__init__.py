from vectorincrement.vectorincrement import VectorIncrementEnvironment
import gym

gym.envs.register(
     id='VectorIncrement-v0',
     entry_point=VectorIncrementEnvironment,
)

