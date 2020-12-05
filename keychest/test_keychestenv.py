from time import time

import gym
import numpy as np
import pytest

from keychest.keychestenv import KeyChestGymEnv, KeyChestEnvironmentRandom, KeyChestEnvironmentFixedMap


def test_hardcoded_env_behavior():
    def random_reward():
        return np.random.rand() * 5 - 3

    mymap = """

P<>
l@ 
B  



"""

    mymap2 = ["P<>", "l@ ", "B  "]

    mymap3 = np.array([['P', '<', '>'],
                       ['l', '@', ' '],
                       ['B', ' ', ' ']],
                      dtype='<U1')

    maps = [mymap, mymap2, mymap3]

    for map_ in maps:
        reward = {'step': -1, 'food_collected': random_reward(), 'key_collected': random_reward(),
                  'chest_opened': random_reward()}

        env = KeyChestGymEnv(engine_constructor=KeyChestEnvironmentFixedMap,
                             map_array=map_, initial_health=9, food_efficiency=3,
                             reward_dict=reward)
        obs = env.reset()

        obs_hardcoded_match = np.array([['@', '@', '@', '@', '@'],
                                        ['@', '@', '@', '@', ' '],
                                        [' ', ' ', ' ', ' ', ' '],
                                        [' ', ' ', ' ', ' ', ' '],
                                        ['#', '#', '#', '#', '#'],
                                        ['#', 'P', '<', '>', '#'],
                                        ['#', 'l', '@', ' ', '#'],
                                        ['#', 'B', ' ', ' ', '#'],
                                        ['#', '#', '#', '#', '#']], dtype='<U1')

        def assert_obs_equals(env, obs):
            print(env.render(mode='np_array').shape, obs.shape)
            print(np.where(env.render(mode='np_array') != obs))
            print(env.render(mode='np_array'))
            assert all((env.render(mode='np_array') == obs).flatten()), "Wrong observation"

        assert env.engine.shape == (3, 3)
        assert env.engine.health == 9
        assert env.engine.keys == 0
        assert env.engine.player_position == (0, 0)
        assert env.engine.lamp_state == 0
        assert_obs_equals(env, obs_hardcoded_match)

        assert isinstance(env.render('rgb_array'), np.ndarray)

        obs, rew, done, info = env.step_string('up')
        assert env.engine.player_position == (0, 0)
        assert env.engine.health == 8
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('right')
        assert env.engine.health == 7
        assert env.engine.keys == 0
        assert env.engine.player_position == (0, 1)
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('right')
        assert env.engine.health == 6
        assert env.engine.keys == 1
        assert env.engine.player_position == (0, 2)
        assert rew == -1 + reward['key_collected']
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('right')
        assert env.engine.health == 5
        assert env.engine.keys == 0
        assert env.engine.player_position == (0, 2)
        assert rew == -1 + reward['chest_opened']
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('down')
        assert env.engine.health == 4
        assert env.engine.keys == 0
        assert env.engine.player_position == (1, 2)
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('left')
        assert env.engine.health == 3
        assert env.engine.keys == 0
        assert env.engine.player_position == (1, 1)
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('left')
        assert env.engine.health == 2 + env.engine.food_efficiency
        assert env.engine.keys == 0
        assert env.engine.player_position == (1, 0)
        assert rew == -1 + reward['food_collected']
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('down')
        assert env.engine.health == 1 + env.engine.food_efficiency
        assert env.engine.keys == 0
        assert env.engine.player_position == (2, 0)
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('down')
        assert env.engine.health == 3
        assert env.engine.keys == 0
        assert env.engine.player_position == (2, 0)
        assert rew == -1
        assert env.engine.lamp_state == 1
        assert done == False

        obs, rew, done, info = env.step_string('right')
        assert env.engine.health == 2
        assert env.engine.keys == 0
        assert env.engine.player_position == (2, 1)
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('right')
        assert env.engine.health == 1
        assert env.engine.keys == 0
        assert env.engine.player_position == (2, 2)
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == False

        obs, rew, done, info = env.step_string('right')
        assert env.engine.health == 0
        assert env.engine.keys == 0
        assert env.engine.player_position == (2, 2)
        assert rew == -1
        assert env.engine.lamp_state == 0
        assert done == True


def test_env_create():
    env = KeyChestGymEnv(engine_constructor=KeyChestEnvironmentRandom,
                         initial_health=15, food_efficiency=10)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env


def test_rollouts(do_print=False, time_for_test=3):
    """Do rollouts and see if the environment crashes."""
    time_start = time()

    while True:
        if time() - time_start > time_for_test:
            break

        # obtaining random params
        width = np.random.choice(np.arange(1, 20))
        height = np.random.choice(np.arange(1, 20))
        n_keys = np.random.choice(np.arange(1, 20))
        n_chests = np.random.choice(np.arange(1, 20))
        n_food = np.random.choice(np.arange(1, 20))
        initial_health = np.random.choice(np.arange(1, 20))
        food_efficiency = np.random.choice(np.arange(1, 20))

        wh = width * height
        n_objects = 2 + n_keys + n_chests + n_food

        params = dict(width=width, height=height, n_keys=n_keys, n_chests=n_chests, n_food=n_food,
                      initial_health=initial_health, food_efficiency=food_efficiency)

        if do_print:
            print("Obtained params", params)

        if n_objects > wh:
            with pytest.raises(AssertionError) as excinfo:
                # creating environment
                KeyChestGymEnv(engine_constructor=KeyChestEnvironmentRandom,
                               **params)
            assert str(excinfo.value).startswith('Too small width*height')
            continue
        else:
            env = KeyChestGymEnv(engine_constructor=KeyChestEnvironmentRandom,
                                 **params)

        assert isinstance(env, KeyChestGymEnv)

        # doing episodes
        for episode in range(20):
            obs = env.reset()
            img = env.render(mode='rgb_array')
            assert img.shape[2] == 3
            done = False
            steps = 0

            while not done:
                act = env.action_space.sample()
                obs, rew, done, info = env.step(act)
                img = env.render(mode='rgb_array')
                assert img.shape[2] == 3
                steps += 1


def test_wrong_action():
    env = KeyChestGymEnv(engine_constructor=KeyChestEnvironmentRandom,
                         initial_health=15, food_efficiency=10)
    with pytest.raises(KeyError) as excinfo:
        env.step(222)
    assert str(excinfo.value) == '222'
