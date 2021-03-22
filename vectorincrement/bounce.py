import gin
import gym
import numpy as np


@gin.configurable
class BounceEnv(gym.Env):
    """A dot bounding off a flat surface, with changing gravity."""
    def __init__(self, xlim=(0, 10), ylim=(0, 10)):
        # x -- horizontal coordinate [0, 10]
        # y -- vertical coordinate [0, 10]
        # format: x, y, vx, vy, gx, gy
        self.xlim = xlim
        self.ylim = ylim
        self.state = np.zeros(6, dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=np.float32(-np.inf),
                                                high=np.float32(np.inf),
                                                shape=(6,))

        # actions: noop, vx+dvx, vy+dvy, gx+dgx, gy+dgy, vx-dvx, vy-dvy, gx-dgx, gy-dgy
        self.action_space = gym.spaces.Discrete(9)

        self.dvx = 0.1
        self.dvy = 0.1
        self.dgx = 0.1
        self.dgy = 0.1

        # time between simulations
        self.dt = 0.1
        self.t = 0

    def reset(self):
        self.t = 0

        self.state = np.zeros(6)
        self.state[0] = 0
        self.state[1] = np.random.rand() * 5

        self.state[0] = np.clip(self.state[0], *self.xlim)
        self.state[1] = np.clip(self.state[1], *self.ylim)

        # vel
        self.state[2] = np.random.randn() * 0.1
        self.state[3] = np.random.randn() * 0.1

        # gravity
        self.state[4] = 0
        self.state[5] = -0.1  # down

        return self.state

    def render(self, mode='rgb_array', resolution=100):
        obs = np.zeros((resolution, resolution, 3), dtype=np.float32)
        x, y = self.state[0], self.state[1]
        x = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * resolution
        y = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * resolution

        x = int(x)
        y = int(y)

        x, y = resolution - 1 - y, x

        if x < 0:
            x = 0
        if x >= resolution - 1:
            x = resolution - 1
        if y < 0:
            y = 0
        if y >= resolution - 1:
            y = resolution - 1

        obs[x, y, 1] = 1.0

        return obs * 255.

    def step(self, action):
        dvx, dvy, dgx, dgy = 0, 0, 0, 0
        if action == 1: # vx + 1
            dvx = self.dvx
        elif action == 2: # vy + 1
            dvy = self.dvy
        elif action == 3: # gx + 1
            dgx = self.dgx
        elif action == 4: # gy + 1
            dgy = self.dgy
        elif action == 5: # vx - 1
            dvx = - self.dvx
        elif action == 6: # vy - 1
            dvy = - self.dvy
        elif action == 7: # gx - 1
            dgx = - self.dgx
        elif action == 8:
            dgy = - self.dgy

        # current state
        x, y, vx, vy, gx, gy = self.state

        # next state
        nx, ny, nvx, nvy, ngx, ngy = self.state

        # kinematics
        nx = x + vx * self.dt
        ny = y + vy * self.dt

        if ny <= 0:  # reflection
            ny = - ny
            vy = - vy

        while nx < self.xlim[0]:
            nx += self.xlim[1] - self.xlim[0]
        while nx > self.xlim[1]:
            nx -= self.xlim[1] - self.xlim[0]

        while ny < self.ylim[0]:
            ny += self.ylim[1] - self.ylim[0]
        while ny > self.ylim[1]:
            ny -= self.ylim[1] - self.ylim[0]

        nvx = vx + gx * self.dt + dvx
        nvy = vy + gy * self.dt + dvy

        ngx = gx + dgx
        ngy = gy + dgy

        newstate = np.array([nx, ny, nvx, nvy, ngx, ngy], dtype=np.float32)
        self.state = newstate
        self.t += self.dt

        rew = 0.0
        done = False
        info = {}

        return newstate, rew, done, info