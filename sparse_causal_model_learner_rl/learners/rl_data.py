import ray
import logging

import gin
import gym
import numpy as np
from tqdm import tqdm
from tqdm.auto import tqdm

from causal_util import load_env
from causal_util.collect_data import EnvDataCollector, compute_reward_to_go
from causal_util.helpers import one_hot_encode


class RLContext():
    """Collect data from an RL environment on a random policy."""
    def __init__(self, config, gin_config_str=None):
        if gin_config_str:
            gin.parse_config(gin_config_str)
        self.config = config
        self.env = self.create_env()
        self.collector = EnvDataCollector(self.env)
        self.vf_gamma = self.config.get('vf_gamma', 1.0)

        # Discrete action -> one-hot encoding
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.to_onehot = True
            self.action_shape = (self.env.action_space.n,)
        else:
            self.to_onehot = False
            self.action_shape = self.env.action_space.shape

        self.additional_feature_keys = self.config.get('additional_feature_keys', [])

    def create_env(self):
        """Create the RL environment."""
        """Create an environment according to config."""
        if 'env_config_file' in self.config:
            gin.parse_config_file(self.config['env_config_file'])
        return load_env()

    def sample_action(self, obs):
        # TODO: run a policy with curiosity reward instead of the random policy

        return self.collector.action_space.sample()

    def collect_steps(self, do_tqdm=False):

        # removing old data
        self.collector.clear()

        # collecting data
        n_steps = self.config['env_steps']
        with tqdm(total=n_steps, disable=not do_tqdm) as pbar:
            while self.collector.steps < n_steps:
                done = False
                obs = self.collector.reset()
                pbar.update(1)
                while not done:
                    obs, _, done, _ = self.collector.step(self.sample_action(obs))
                    pbar.update(1)
        self.collector.flush()

    def collect_get_context(self):
        self.collect_steps()
        return self.get_context()

    def get_context(self):
        # x: pre time-step, y: post time-step

        # observations, actions, rewards-to-go, total rewards
        obs_x, obs_y, obs, act_x, reward_to_go, episode_sum_rewards = [], [], [], [], [], []
        done_y, rew_y = [], []

        for episode in self.collector.raw_data:
            rew = []
            is_multistep = len(episode) > 1
            for i, step in enumerate(episode):
                remaining_left = i
                remaining_right = len(episode) - 1 - i
                is_first = remaining_left == 0
                is_last = remaining_right == 0

                obs.append(step['observation'])

                if is_multistep and not is_first:
                    action = step['action']
                    if self.to_onehot:
                        action = one_hot_encode(self.action_shape[0], action)

                    rew_y.append(step['reward'])
                    done_y.append(1. * is_last)

                    obs_y.append(step['observation'])
                    act_x.append(action)
                    rew.append(step['reward'])

                if is_multistep and not is_last:
                    obs_x.append(step['observation'])

            rew_to_go_episode = compute_reward_to_go(rew, gamma=self.vf_gamma)
            episode_sum_rewards.extend(list(np.cumsum(rew)))
            reward_to_go.extend(rew_to_go_episode)

        # for value function prediction
        assert len(reward_to_go) == len(obs_x)

        assert len(episode_sum_rewards) == len(obs_y)

        # for modelling
        assert len(obs_x) == len(act_x)

        # for reconstruction
        assert len(obs_x) == len(obs_y)

        obs_x = np.array(obs_x)
        obs_y = np.array(obs_y)
        obs = np.array(obs)
        act_x = np.array(act_x)
        reward_to_go = np.array(reward_to_go)
        done_y = np.array(done_y)
        rew_y = np.array(rew_y)
        episode_sum_rewards = np.array(episode_sum_rewards)

        context = {'obs_x': obs_x, 'obs_y': obs_y, 'action_x': act_x,
                   'rew_y': rew_y, 'done_y': done_y,
                   'obs': obs,
                   'reward_to_go': reward_to_go,
                   'episode_sum_rewards': episode_sum_rewards}

        return context


@gin.configurable
class ExperienceReplayBuffer():
    """Collect data from RL contexts, and store it. Then, sample batches."""
    def __init__(self, config,
                 collectors):
        self.config = config
        self.collectors = collectors
        self.n_collectors = len(self.collectors)
        self.buffer_steps = self.config.get('buffer_steps', 1000000)
        self.shuffle_together = self.config.get('shuffle_together', [])
        self.minibatch_size = self.config.get('minibatch_size', 5000)

        self.next_episode_refs = []
        self.future_episode_size = self.config.get('future_episode_size', 10)

        self.reset()

    def reset(self):
        self.buffer = {}
        self.buffer_n = {}
        self.buffer_write_idx = {}

    def collect(self, min_batches=1):
        while len(self.next_episode_refs) < max(min_batches, self.future_episode_size):
            remote_id = np.random.choice(self.n_collectors)
            ref = self.collectors[remote_id].collect_get_context.remote()
            self.next_episode_refs.append(ref)
            logging.warning(f"Scheduling episode on {remote_id}")

        self.next_episode_refs = self.collect_from(self.next_episode_refs, min_batches=min_batches)

    def collect_from(self, futures, min_batches=1):
        """Collect from futures. Returns list of unprocessed elements."""
        f_ready, f_remaining = ray.wait(futures, num_returns=min_batches)
        for f in f_ready:
            logging.warning("Collecting episode")
            self.observe(ray.get(f))
        return f_remaining

    def check_keys(self):
        assert self.buffer, "Buffer is empty"
        left_keys = set(self.buffer.keys())
        for group in self.shuffle_together:
            left_keys.difference_update(group)

            group_lens = [self.buffer_n[key] for key in group]
            group_len = group_lens[0]
            assert [group_len == l for l in
                    group_lens], f"group lens must be the same {group} {group_lens}"
            assert group_len <= self.buffer_steps, f"Too many steps" \
                                                   f" {group} {group_len} {self.buffer_steps}"
        assert not left_keys, f"Some keys were not used: {left_keys} {self.shuffle_together}"

    def sample_batch(self):
        self.check_keys()
        pre_context_return = {}
        for group in self.shuffle_together:
            group_len = self.buffer_n[group[0]]
            if group_len > self.minibatch_size:
                idxes_return = np.random.choice(a=group_len,
                                                size=self.minibatch_size, replace=False)
            else:
                idxes_return = slice(-1)

            for key in group:
                pre_context_return[key] = self.buffer[key][idxes_return]

        return pre_context_return

    def observe(self, pre_context):
        """Write data to buffer."""
        for key in pre_context.keys():

            assert isinstance(pre_context[key], np.ndarray), f"Inputs must be numpy " \
                                                             f"arrays {key}" \
                                                             f" {type(pre_context[key])}"
            if key not in self.buffer:
                self.buffer[key] = np.zeros((self.buffer_steps, *pre_context[key].shape[1:]),
                                            dtype=np.float32)
                self.buffer_write_idx[key] = 0
                self.buffer_n[key] = 0
            else:
                idx_start = self.buffer_write_idx[key]
                L = len(pre_context[key])
                idx_end = idx_start + L

                # buffer wrap
                if idx_end >= self.buffer_steps:
                    # buffer is fully taken
                    self.buffer_write_idx[key] += L
                    self.buffer_write_idx[key] %= self.buffer_steps

                    part_1_L = self.buffer_steps - idx_start
                    part_2_L = L - part_1_L

                    self.buffer[key][idx_start:self.buffer_steps] = pre_context[key][:part_1_L]
                    self.buffer[key][:part_2_L] = pre_context[key][part_1_L:]
                    self.buffer_n[key] = self.buffer_steps
                else:
                    self.buffer[key][idx_start:idx_end] = pre_context[key]
                    self.buffer_write_idx[key] += L
                    self.buffer_n[key] += L

                self.buffer_n[key] = min(self.buffer_steps, self.buffer_n[key])

@gin.configurable
class ParallelContextCollector():
    def __init__(self, config):
        RLContextRemote = ray.remote(RLContext)
        ExperienceReplayBufferRemote = ray.remote(ExperienceReplayBuffer)

        self.config = config
        self.n_collectors = self.config.get('n_collectors', 3)
        self.remote_rl_contexts = [RLContextRemote.remote(config=self.config,
                                                          gin_config_str=gin.config_str())
                                   for _ in range(self.n_collectors)]

        self.replay_buffer = ExperienceReplayBufferRemote.remote(
            config=self.config, collectors=self.remote_rl_contexts)

        self.next_batch_refs = set()
        self.future_batch_size = self.config.get('future_batch_size', 10)
        self.collect_initial()

    def collect_initial(self):
        ray.get(self.replay_buffer.collect.remote(
            min_batches=self.config.get('collect_initial_batches', self.future_batch_size)))

    def collect_get_context(self):
        # storing episodes into memory of the buffer
        # will collect at least one batch of data
        self.replay_buffer.collect.remote()

        # requesting shuffled batches
        while len(self.next_batch_refs) < self.future_batch_size:
            self.next_batch_refs.add(self.replay_buffer.sample_batch.remote())

        ready_refs, non_ready_refs = ray.wait(list(self.next_batch_refs), num_returns=1)
        ready_ref = ready_refs[0]
        self.next_batch_refs.remove(ready_ref)
        return ray.get(ready_ref)