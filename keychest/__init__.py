import gym

from keychest.keychestenv import KeyChestGymEnv

gym.envs.register(
    id='KeyChest-v0',
    entry_point=KeyChestGymEnv,
)
