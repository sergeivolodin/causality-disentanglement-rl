from keychest.keychestenv import KeyChestGymEnv
import gym

gym.envs.register(
     id='KeyChest-v0',
     entry_point=KeyChestGymEnv,
)

