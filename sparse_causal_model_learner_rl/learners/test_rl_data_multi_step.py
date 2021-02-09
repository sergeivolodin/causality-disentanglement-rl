import numpy as np
import gym
import gin
from stable_baselines.common.env_checker import check_env
from causal_util.collect_data import EnvDataCollector
from .rl_data_multi_step import get_multi_step_rl_context

gin.enter_interactive_mode()

def episode(env):
    """Roll out one episode with random actions."""
    done = False
    env.reset()
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())


@gin.configurable
class IdEnv(gym.Env):
    """Give ID in observations."""

    def __init__(self, max_steps=100, ):
        super(IdEnv, self).__init__()
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(low=0, high=self.max_steps + 1,
                                                shape=(1,))
        self.action_space = gym.spaces.Discrete(2)
        self.reset()

    def obs(self):
        return np.array([self.n_steps], dtype=np.float32)

    def reset(self):
        self.n_steps = 0
        return self.obs()

    def step(self, action):
        self.n_steps += 1
        done = self.n_steps >= self.max_steps
        rew = float(np.random.choice(2))
        info = {}
        return self.obs(), rew, done, info

def test_id_env():
    env = IdEnv(max_steps=6)
    check_env(env)
    assert env.reset()[0] == 0
    collector = EnvDataCollector(env)

    episode(collector)
    collector.flush()
    assert len(collector.raw_data[0]) == 7

def test_multi_step_manual():
    env = IdEnv(max_steps=6)

    collector = EnvDataCollector(env)

    episode(collector)
    collector.flush()

    actions = [x['action'] for x in collector.raw_data[0][1:]]
    dones = [x['done'] for x in collector.raw_data[0][1:]]
    rews = [x['reward'] for x in collector.raw_data[0][1:]]
    observations = [x['observation'] for x in collector.raw_data[0]]
    assert len(actions) == len(dones) == len(rews) == len(observations) - 1
    assert all([not x for x in dones[:-1]])
    assert dones[-1]

    context = get_multi_step_rl_context(collector, n_steps_forward=3, return_intermediate=True)

    assert np.allclose(context['obs_1'], np.array([0, 1, 2, 3, 4]).reshape(-1, 1))
    assert np.allclose(context['obs_2'], np.array([1, 2, 3, 4, 5]).reshape(-1, 1))
    assert np.allclose(context['obs_3'], np.array([2, 3, 4, 5, 6]).reshape(-1, 1))
    assert np.allclose(context['act_1'], actions[:-1])
    assert np.allclose(context['act_2'], actions[1:])
    assert np.allclose(context['rew_1'], rews[:-1])
    assert np.allclose(context['rew_2'], rews[1:])
    assert np.allclose(context['done_1'], np.array([False] * 5))
    assert np.allclose(context['done_2'], np.array([False] * 4 + [True]))

def check_context(steps, context, observations, actions, rews):
    def soft_assert_eq(a, b, msg):
        if a != b:
            print(f"Error: {msg}, {a} != {b}")
            assert False

    for s in range(1, steps + 1):
        curr_obs = context.get(f'obs_{s}', [])

        if steps == s:
            obs_slice = slice(s-1, len(observations))
        else:
            obs_slice = slice(s-1, -(steps-s))
        obs_target = observations[obs_slice]

        print(s-1, steps-s, len(obs_target))

        soft_assert_eq(len(curr_obs), len(observations) - steps + 1 , f"Wrong obs_{s} len")
        soft_assert_eq(len(obs_target), len(observations) - steps + 1 , f"Wrong obs_{s} target len")


        for item, item_true in zip(curr_obs, observations[s - 1:]):
            assert np.allclose(item, item_true), f"Wrong obs {item} {item_true}"

    for s in range(1, steps):
        curr_act = context.get(f'act_{s}', [])
        curr_rew = context.get(f'rew_{s}', [])
        curr_done = context.get(f'done_{s}', [])

        if -(steps - s - 1) == 0:
            slice1 = slice(s - 1, None)
        else:
            slice1 = slice(s - 1, -(steps - s - 1))
        rew_target = rews[slice1]
        act_target = actions[slice1]

        soft_assert_eq(len(rew_target), len(observations) - steps + 1, f"Wrong rew_{s} len")
        soft_assert_eq(len(act_target), len(observations) - steps + 1, f"Wrong act_{s} len")

        soft_assert_eq(len(curr_act), len(observations) - steps + 1, f"Wrong act_{s} len")
        soft_assert_eq(len(curr_rew), len(observations) - steps + 1, f"Wrong rew_{s} len")
        soft_assert_eq(len(curr_done), len(observations) - steps + 1, f"Wrong done_{s} len")

        for item, item_true in zip(curr_act, actions[s - 1:]):
            assert item == item_true, f"Wrong act {item} {item_true} {s} {curr_act} {actions}"

        for item, item_true in zip(curr_rew, rews[s - 1:]):
            assert item == item_true, f"Wrong rew {item} {item_true}"

        if s < steps - 1:
            assert all([not d for d in curr_done]), f"Wrong done {s} {curr_done}"
        else:
            assert all([not d for d in curr_done[:-1]]), f"Wrong [:-1] done {s} {curr_done}"
            assert len(curr_done) and curr_done[-1], f"Wrong last done {s} {curr_done}"

def test_auto_multistep(env_steps=100, multistep=50):
    env = IdEnv(max_steps=env_steps)

    collector = EnvDataCollector(env)

    episode(collector)
    collector.flush()

    actions = [x['action'] for x in collector.raw_data[0][1:]]
    dones = [x['done'] for x in collector.raw_data[0][1:]]
    rews = [x['reward'] for x in collector.raw_data[0][1:]]
    observations = [x['observation'] for x in collector.raw_data[0]]
    assert len(actions) == len(dones) == len(rews) == len(observations) - 1
    assert all([not x for x in dones[:-1]])
    assert dones[-1]

    context = get_multi_step_rl_context(collector, n_steps_forward=multistep, return_intermediate=True)

    check_context(multistep, context, observations, actions, rews)