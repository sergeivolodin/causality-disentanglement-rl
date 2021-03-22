import gin
import gym
import numpy as np


@gin.configurable
class GridWorldNavigationEnv(gym.Env):
    """Grid-world with 2D navigation"""
    def __init__(self, H=10, W=10):
        self.H = H
        self.W = W

        self.x = 0
        self.y = 0

        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(self.H, self.W))

        # actions: x+1, x-1, y+1, y-1
        self.action_space = gym.spaces.Discrete(4)
        self.reset()

    def render(self, mode='rgb_array'):
        obs = np.zeros((self.H, self.W, 3))
        obs[self.x, self.y, 0] = 1.0
        return obs * 255

    def observation(self):
        z = np.zeros((self.H, self.W), dtype=np.float32)
        z[self.x, self.y] = 1.0
        return z

    def reset(self):
        self.x = np.random.choice(self.H)
        self.y = np.random.choice(self.W)
        return self.observation()

    def step(self, action):
        action_map = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
        dx, dy = action_map[action]

        self.x += dx
        self.y += dy

        if self.x < 0:
            self.x = 0
        if self.x > self.H - 1:
            self.x = self.H - 1
        if self.y < 0:
            self.y = 0
        if self.y > self.W - 1:
            self.y = self.W - 1

        rew = 0.0
        done = False
        info = {}
        return self.observation(), rew, done, info