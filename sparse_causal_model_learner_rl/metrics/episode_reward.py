import gin


@gin.configurable
def episode_reward(done_y, episode_sum_rewards, **kwargs):
    """Compute mean episode reward."""
    mask = (done_y == done_y.max())
    if not mask.sum().item():
        return None
    return episode_sum_rewards[mask].mean()