import numpy as np
import gym
import gin
from causal_util.collect_data import EnvDataCollector


def get_multi_step_rl_context(collector, n_steps_forward=3, return_intermediate=False):
    """Obtain data for multi-step prediction.

    Args:
        collector: EnvDataCollector object with data
        n_steps_forward: number of steps to predict ahead
        return_intermediate: give non-start and non-end (in n_steps_forward batches) observations

    Returns: a dictionary with keys
        obs_1 ... obs_{n_steps_forward}: numpy arrays with observations corresponding to these offsets
          if return_intermediate is False, only obs_1 and obs_{n_steps_forward} are given
        act_1 ... act_{n_steps_forward-1}: actions taken at steps obs_1 ...
        rew_1 ... rew_{n_steps_forward-1}: rewards given for corresponding actions above
        done_1 ... done_{n_steps_forward-1}: done associated with rewards above
    """

    assert isinstance(collector,
                      EnvDataCollector), f"collector must be an instance of EnvDataCollector, got {collector}"
    assert isinstance(n_steps_forward, int), f"give integer n_steps_forward: {n_steps_forward}"
    assert isinstance(return_intermediate,
                      bool), f"give bool return_intermediate: {return_intermediate}"
    assert n_steps_forward >= 2, f"n_steps_forward must be at least 2, which means 1-step prediction: {n_steps_forward}"

    # the resulting dictionary
    result = {}

    for s in range(1, n_steps_forward + 1):
        result[f'obs_{s}'] = []
    for s in range(1, n_steps_forward):
        result[f'act_{s}'] = []
        result[f'rew_{s}'] = []
        result[f'done_{s}'] = []

    for episode in collector.raw_data:
        episode_len = len(episode)
        actions = [x['action'] for x in episode[1:]]
        dones = [x['done'] for x in episode[1:]]
        rews = [x['reward'] for x in episode[1:]]
        observations = [x['observation'] for x in episode]
        for s in range(1, n_steps_forward):
            curr_slice_obs = slice(s - 1, -(n_steps_forward - s))
            curr_slice = slice_with_0_end(s - 1, -(n_steps_forward - s) + 1)
            if s == 1 or return_intermediate:
                result[f'obs_{s}'].extend(observations[curr_slice_obs])
                result[f'done_{s}'].extend(dones[curr_slice])
            result[f'act_{s}'].extend(actions[curr_slice])
            result[f'rew_{s}'].extend(rews[curr_slice])
        result[f'obs_{n_steps_forward}'].extend(observations[n_steps_forward - 1:])

    result = {x: np.array(y) for x, y in result.items()}

    return result

def slice_with_0_end(s, end):
    if end == 0:
        return slice(s, None)
    else:
        return slice(s, end)
