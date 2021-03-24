import gin
import gym
import numpy as np
from matplotlib import pyplot as plt
import argparse
import seaborn as sns
import vectorincrement

        
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Show matrix")
    parser.add_argument("--config", type=str, default=None, required=True)
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    env = gym.make('SparseMatrix-v0')
    plt.figure()
    plt.subplot(1, 2, 1)
    sns.heatmap(np.abs(env.A), vmin=0)
    plt.subplot(1, 2, 2)
    sns.heatmap(np.abs(env.Aa), vmin=0)
    plt.show()
