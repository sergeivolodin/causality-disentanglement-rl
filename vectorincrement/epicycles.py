import gin
import gym
import cv2
import numpy as np


@gin.configurable
class RocketEpicycleEnvironment(gym.Env):
    """
    Rocket goes around the sun, the actions adjust the orbit radius.
    Another planet goes around the sun as well.
    Observations are from p.o.v. of the rocket, giving epicycles
    """
    
    def __init__(self, epicycles=True, S=10, h=20, w=20):
        self.epicycles = epicycles  # if False, sun-centric
        self.r_rocket_min = 10.
        self.r_rocket_max = 20.
        self.S = S
        self.h = h
        self.w = w
        
        self.dt = 0.1
        
        self.dr_rocket = 4.
        
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(shape=(h, w, 3), high=np.inf, low=-np.inf)
        
    def __repr__(self):
        return f"<RocketEpicycleEnvironment epicycles={self.epicycles}>"
        
    def reset(self):
        self.r_rocket = self.r_rocket_min + np.random.rand() * (self.r_rocket_max - self.r_rocket_min)
        self.r_planet = 25 #p.random.rand() * 10 + 25
        
        self.phi_planet = np.random.rand() * 2 * np.pi
        self.phi_rocket = np.random.rand() * 2 * np.pi
        
        self.angvel_planet = 0.5 # np.random.rand()
        self.angvel_rocket = 0.8 #np.random.rand()
        
        # maximal radius
        self.rmax = max(self.r_rocket_max, self.r_planet) + 0.5
        
        return self.state()
        
    def step(self, action):
        if action == 1:
            self.r_rocket += self.dr_rocket
        elif action == 2:
            self.r_rocket -= self.dr_rocket
        self.r_rocket = min(max(self.r_rocket, self.r_rocket_min), self.r_rocket_max)
        
        self.phi_planet += self.angvel_planet * self.dt
        self.phi_rocket += self.angvel_rocket * self.dt
        
        rew = 0.0
        done = False
        info = {}
        return self.state(), rew, done, info
        
    def render(mode='rgb_array'):
        return self._observation()
    
    def state(self):
        return self._observation()
        
    def _observation(self):
        # sun coordinate system
        
        h, w, S = self.h, self.w, self.S
        
        xsun, ysun = 0, 0
        xplanet, yplanet = self.r_planet * np.cos(self.phi_planet), self.r_planet * np.sin(self.phi_planet)
        xrocket, yrocket = self.r_rocket * np.cos(self.phi_rocket), self.r_rocket * np.sin(self.phi_rocket)
        
        R = int(np.mean([h, w]) * S * 0.05) #* 10
        
        
        if self.epicycles:
            xsun -= xrocket
            ysun -= yrocket
            
            xplanet -= xrocket
            yplanet -= yrocket
            
#             xmin -= xrocket
#             xmax -= xrocket
#             ymin -= yrocket
#             ymax -= yrocket
            
            xrocket = 0
            yrocket = 0
#             print("epic")
            
            rmax = self.r_rocket_max + self.r_planet + 0.5
            xmin, xmax = -rmax - R, rmax + R
            ymin, ymax = -rmax - R, rmax + R
            
        else:
            xmin, xmax = -self.rmax - R, self.rmax + R
            ymin, ymax = -self.rmax - R, self.rmax + R

            
#         print(xplanet, yplanet, xsun, ysun, xrocket, yrocket)
    
        def to_img_coord(x, y):
            return int(round(h * S * (x - xmin) / (xmax - xmin))),\
                   int(round(w * S * (y - ymin) / (ymax - ymin)))
        
        img1 = np.zeros((h * S, w * S, 3), np.float32)
        img2 = np.zeros((h * S, w * S, 3), np.float32)
        img3 = np.zeros((h * S, w * S, 3), np.float32)
        
        T = -1
        
        # sun
        img1 = cv2.circle(img1, to_img_coord(xsun,    ysun), R, (1, 0, 0), T)
        
        # rocket
        img2 = cv2.circle(img2, to_img_coord(xrocket, yrocket), R, (0, 1, 0), T)
        
        # planet
        img3 = cv2.circle(img3, to_img_coord(xplanet, yplanet), R, (0, 0, 1), T)
        
        img = img1 + img2 + img3
        img_rs = cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)
        return img_rs
