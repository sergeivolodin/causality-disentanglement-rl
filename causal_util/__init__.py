from gym.wrappers import TimeLimit
import gin
import gym
# noinspection PyUnresolvedReferences
import vectorincrement  # noqa # pylint: disable=unused-import
# noinspection PyUnresolvedReferences
import keychest  # noqa # pylint: disable=unused-import
from causal_util.weight_restorer import WeightRestorer


@gin.configurable
def load_env(env_name, time_limit=None, obs_scaler=None, wrappers=None,
             wrappers_prescale=None,
             **kwargs):
    """Load an environment, configurable via gin."""
    print(f"Make environment {env_name} {wrappers} {kwargs}")
    env = gym.make(env_name, **kwargs)
    if time_limit:
        env = TimeLimit(env, time_limit)

    if wrappers_prescale is None:
        wrappers_prescale = []
    for wrapper in wrappers_prescale[::-1]:
        env = wrapper(env)

    if obs_scaler:
        from encoder.observation_encoder import ObservationScaleWrapper
        env = ObservationScaleWrapper(env, obs_scaler)

    if wrappers is None:
        wrappers = []
    for wrapper in wrappers[::-1]:
        env = wrapper(env)
    return env