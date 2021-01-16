import gin
import numpy as np


@gin.configurable
def episode_reward(episode_rewards, **kwargs):
    """Compute mean episode reward."""
    return np.mean(episode_rewards.detach().cpu().numpy()) if len(episode_rewards) else None