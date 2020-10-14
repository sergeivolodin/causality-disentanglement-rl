import math
from copy import deepcopy

import gin
import gym
import numpy as np


class DelayedExecutor(object):
    """Execute stored commands after a pre-defined number of steps."""

    def __init__(self):
        self.queue = []
        self.current_step = 0

    def push(self, delay, function):
        """Add a function to execute."""
        assert isinstance(delay, int)
        assert delay >= 0
        assert callable(function)
        self.queue.append((self.current_step + delay, function))

    def step(self):
        """One step, will execute/increment step counter."""
        new_queue = []

        for target_step, fcn in self.queue:
            if target_step <= self.current_step:
                fcn()
            else:
                new_queue.append((target_step, fcn))

        self.queue = new_queue
        self.current_step += 1

    def __repr__(self):
        return f"<DE nextstep={self.current_step} queuelen={len(self.queue)}>"


def inrange(x, m, M):
    """x in [m, M]?"""
    return m <= x <= M


def keychest_obs3d_to_obs2d(obs):
    """Convert 3d observation into a 2d text observation."""
    obs_3d = obs
    assert isinstance(obs_3d, np.ndarray)
    assert len(obs_3d.shape) == 3 and obs_3d.shape[-1] == len(KeyChestEnvironment.SYMBOLS)
    shape = obs_3d.shape[:2]

    obs = np.full(fill_value=KeyChestEnvironment.SYMBOLS['empty'], shape=shape, dtype='<U1')

    for i, obj in enumerate(KeyChestEnvironment.OBJECTS):
        mask = obs_3d[:, :, i] > 0
        symbol = KeyChestEnvironment.SYMBOLS[obj]
        obs[mask] = symbol

    return obs


def keychest_obs2d_to_image(obs, scale=15):
    """Convert 2d observation given by KeyChest environment into a 2D image."""
    assert len(obs.shape) == 2
    out_arr = np.zeros((obs.shape[0], obs.shape[1], 3))
    for symbol in KeyChestEnvironment.SYMBOL_LIST:
        mask = obs == symbol
        out_arr[mask] = KeyChestEnvironment.SYMBOLS_TO_COLORS[symbol]
    out_arr /= 255.

    out_arr = np.repeat(out_arr, scale, axis=1)
    out_arr = np.swapaxes(np.repeat(np.swapaxes(out_arr, 0, 1), scale, axis=1), 0, 1)
    return np.array(out_arr * 255, dtype=np.uint8)


@gin.configurable
class KeyChestEnvironment(object):
    # class constants
    OBJECTS = ['empty', 'keys_collected', 'health', 'wall', 'key', 'chest', 'food', 'lamp_on', 'lamp_off', 'player']
    SYMBOLS = {'wall': '#', 'player': 'P', 'key': '<', 'chest': '>', 'food': '@',
               'lamp_on': 'L', 'lamp_off': 'l', 'empty': ' ', 'health': '@', 'keys_collected': '<'}
    INNER_OBJECTS = ['empty', 'wall', 'key', 'chest', 'food', 'lamp_on', 'lamp_off', 'player']
    COLORS = {'empty': (191, 191, 191), 'wall': (0, 0, 0), 'key': (0, 0, 255), 'chest': (255, 255, 0),
              'lamp_on': (255, 255, 255),
              'lamp_off': (94, 94, 94), 'food': (0, 255, 0), 'player': (255, 0, 0), 'health': (0, 255, 0),
              'keys_collected': (0, 0, 255)}
    ACTIONS = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}
    ACTION_NAMES = {(1, 0): "down", (-1, 0): "up", (0, 1): "right", (0, -1): "left"}

    def __init__(self, labyrinth_maps, initial_health, food_efficiency,
                 food_rows=None, keys_rows=None, callback=None,
                 flatten_observation=False):
        """Environment with keys and chests."""
        self.initial_maps = deepcopy(labyrinth_maps)
        self.maps = deepcopy(labyrinth_maps)
        self.executor = DelayedExecutor()
        self.delay = 1
        self.shape = self.maps['empty'].shape
        assert set(self.maps.keys()) == set(KeyChestEnvironment.OBJECTS)
        self.keys = 0
        self.food_efficiency = food_efficiency
        self.food_rows = food_rows
        self.keys_rows = keys_rows
        self.initial_health = initial_health
        self.health = initial_health
        self.width = self.shape[1]
        self.height = self.shape[0]
        self.first_render = True
        self.callback_ = callback
        self.enabled = True
        self.flatten_observation = flatten_observation

        # to see if everything fits
        self.render()

        self.moves = 0
        self.history = []

        self.callback(dict(event='initialized'))

    def callback(self, *args, **kwargs):
        if self.callback_:
            self.callback_(*args, moves=self.moves, **kwargs)
        else:
            pass
        self.history.append(dict(args=args, kwargs=kwargs, moves=self.moves))

    @property
    def player_position(self):
        return self.locate_single('player')

    def items_at_position(self, pos):
        """Which items are at a particular position?"""
        return [x for x in self.OBJECTS if self.maps[x][pos[0], pos[1]]]

    def item_at_position(self, pos):
        """Got one item at a position."""
        items = self.items_at_position(pos)
        to_remove = ['empty', 'player']
        for item in to_remove:
            if item in items[:]:
                items.remove(item)
        assert len(items) <= 1, f"Must have <= items only at {pos}, got {items}"
        if not items:
            return 'empty'
        return items[0]

    @property
    def _observation(self):
        sx, sy = self.shape

        if self.first_render:
            maxfood = self.initial_health + self.food_efficiency * np.sum(self.initial_maps['food'])
            maxkeys = np.sum(self.initial_maps['key'])
            food_rows = round(math.ceil(1. * maxfood / (2 + self.width)))
            keys_rows = round(math.ceil(1. * maxkeys / (2 + self.width)))

            if self.food_rows is None:
                self.food_rows = food_rows
            else:
                assert self.food_rows >= food_rows

            if self.keys_rows is None:
                self.keys_rows = keys_rows
            else:
                assert self.keys_rows >= keys_rows

            self.first_render = False

        dy1 = 1
        dy2 = 1
        dy = dy1 + dy2
        dx1 = 1 + self.food_rows + self.keys_rows
        dx2 = 1
        dx = dx1 + dx2

        def fill_n(arr, offset_x, value, symbol=True):
            """Fill cells of arr starting from row offset_x with value in unary counting."""
            value_left = value
            width = arr.shape[1]
            current_row = offset_x
            while value_left:
                add_this_iter = min(width, value_left)
                arr[current_row, :add_this_iter] = [symbol] * add_this_iter
                current_row += 1
                value_left -= add_this_iter

        shape = (sx + dx, sy + dy)
        out = {obj: np.full(fill_value=False, shape=shape, dtype=np.bool)
               for obj in self.OBJECTS}

        fill_n(out['health'], 0, self.health)
        fill_n(out['keys_collected'], self.food_rows, self.keys)

        out['wall'][dx1 - 1, :] = True
        out['wall'][-1, :] = True
        out['wall'][dx1 - 1:, 0] = True
        out['wall'][dx1 - 1:, -1] = True
        for obj in self.OBJECTS:
            mask = self.maps[obj] > 0
            out[obj][dx1:-dx2, dy1:-dy2][mask] = True

        # format: C x H x W
        result = 1. * np.array([out[obj] for obj in self.OBJECTS])

        # format: W x H x C
        result = np.swapaxes(result, 0, 2)

        # format: H x W x C
        result = np.swapaxes(result, 0, 1)

        result = np.array(result, dtype=np.float32)

        return result

    @property
    def observation(self):
        result = self._observation

        if self.flatten_observation:
            result = result.flatten()

        return result

    def move_object(self, obj, old_pos, new_pos):
        self.delete_object(obj, old_pos)
        self.add_object(obj, new_pos)

    def delete_object(self, obj, old_pos):
        assert self.maps[obj][old_pos[0], old_pos[1]]
        self.maps[obj][old_pos[0], old_pos[1]] = False

    def add_object(self, obj, new_pos):
        assert not self.maps[obj][new_pos[0], new_pos[1]]
        self.maps[obj][new_pos[0], new_pos[1]] = True

    def step(self, action):
        info = {'action': action, 'event': 'regular_move'}

        next_position = np.array(self.player_position) + np.array(self.ACTIONS[action])

        def clip(x, m, M):
            if x < m:
                return m
            elif x > M:
                return M
            return x

        if not inrange(next_position[0], 0, self.height - 1):
            next_position[0] = clip(next_position[0], 0, self.height - 1)
        elif not inrange(next_position[1], 0, self.width - 1):
            next_position[1] = clip(next_position[1], 0, self.width - 1)

        if self.enabled:
            self.move_object('player', self.player_position, next_position)

        def change_callback(self=self, next_position=next_position, info=info):
            # otherwise we are moving
            item = self.item_at_position(next_position)

            if item == 'lamp_on':
                self.delete_object('lamp_on', next_position)
                self.add_object('lamp_off', next_position)

                info['event'] = 'lamp_turned_off'
            elif item == 'lamp_off':
                self.delete_object('lamp_off', next_position)
                self.add_object('lamp_on', next_position)

                info['event'] = 'lamp_turned_on'
            elif item == 'food':
                self.delete_object('food', next_position)
                self.health += self.food_efficiency

                info['event'] = 'food_collected'
            elif item == 'key':
                self.delete_object('key', next_position)
                self.keys += 1

                info['event'] = 'key_collected'
            elif item == 'chest':
                if self.keys > 0:
                    self.keys -= 1
                    self.delete_object('chest', next_position)
                    info['event'] = 'chest_opened'
                else:
                    info['event'] = 'not_enough_keys'
            else:
                info['event'] = 'regular_move'

            return info

        def move_effect(self=self, change_callback=change_callback, action=action):
            info = change_callback()
            self.callback(info)

            if self.health <= 0:
                info = {'action': action, 'event': 'dead'}
                if not self.enabled:
                    info['event'] = 'already_dead'
                self.enabled = False
                self.callback(info)

        self.executor.push(self.delay, move_effect)
        if self.health:
            self.health -= 1
        self.executor.step()
        self.moves += 1
        return self.observation

    @property
    def obs_2d(self):
        """Get 2d observation with stmbols."""
        return keychest_obs3d_to_obs2d(self._observation)

    def render(self, mode='np_array'):
        obs = self.obs_2d

        if mode == 'np_array':
            return obs
        elif mode == 'str':
            return '\n'.join([''.join(x) for x in obs])
        elif mode == 'rgb_array':
            return keychest_obs2d_to_image(obs)

        return obs

    def locate(self, object_type):
        return KeyChestEnvironment._locate(self.maps, object_type)

    def locate_single(self, object_type):
        return KeyChestEnvironment._locate_single(self.maps, object_type)

    @property
    def lamp_state(self):
        obs = self._observation
        lamp_on = np.sum(self.maps['lamp_on'])
        lamp_off = np.sum(self.maps['lamp_off'])
        if lamp_on:
            return 1
        elif lamp_off:
            return 0
        else:
            return -1

    @staticmethod
    def _locate(maps, object_type):
        """Where are objects of a given type on the map?"""
        assert object_type in KeyChestEnvironment.OBJECTS, \
            f"Wrong object {object_type}, have {maps.keys()}"
        return list(zip(*np.where(maps[object_type])))

    @staticmethod
    def _locate_single(maps, object_type):
        """Where is the single object?"""
        w = KeyChestEnvironment._locate(maps, object_type)
        assert len(w) >= 1, f"No {object_type} found"
        assert len(w) <= 1, f"More than one {object_type} found"
        return w[0]


def compute_attrs(cls=KeyChestEnvironment):
    """Set additional attributes."""
    cls.SYMBOLS_TO_OBJECTS = {y: x for x, y in cls.SYMBOLS.items()}
    cls.SYMBOL_LIST = [cls.SYMBOLS.get(x) for x in cls.OBJECTS]
    cls.SYMBOLS_TO_COLORS = {x: cls.COLORS[cls.SYMBOLS_TO_OBJECTS[x]] for x in cls.SYMBOL_LIST}
    cls.ACTIONS_REVERSE = {y: x for x, y in cls.ACTIONS.items()}
    cls.ACTION_NAME_REVERSE = {y: x for x, y in cls.ACTION_NAMES.items()}


compute_attrs(KeyChestEnvironment)


@gin.configurable
class KeyChestEnvironmentRandom(KeyChestEnvironment):
    """Generate a random map for the KeyChest environment."""

    def __init__(self, width=10, height=10, n_keys=2, n_chests=2, n_food=2, **kwargs):
        objects_to_fill = ['player', 'lamp_off']
        objects_to_fill += ['key'] * n_keys
        objects_to_fill += ['chest'] * n_chests
        objects_to_fill += ['food'] * n_food
        shape = (height, width)
        wh = width * height
        assert wh >= len(objects_to_fill), f"Too small width*height {wh} < {len(objects_to_fill)}"

        positions = []
        for i in range(height):
            for j in range(width):
                positions.append((i, j))

        pos_select = np.random.choice(range(len(positions)), len(objects_to_fill), replace=False)

        maps = {k: np.zeros(shape, dtype=np.bool) for k in self.OBJECTS}
        maps['empty'][:, :] = True

        for pos, obj in zip(pos_select, objects_to_fill):
            m = maps[obj]
            p = positions[pos]
            m[p[0], p[1]] = True

        super(KeyChestEnvironmentRandom, self).__init__(labyrinth_maps=maps, **kwargs)


@gin.configurable
class KeyChestEnvironmentFixedMap(KeyChestEnvironment):
    """Create an environment from a fixed map."""

    def __init__(self, map_array, **kwargs):
        # parsing a string
        if isinstance(map_array, str):
            lines = map_array.splitlines()
            lines = [x for x in lines if x]
            arr = [list(x) for x in lines]
        elif isinstance(map_array, list):
            # list of strings
            if all([isinstance(x, str) for x in map_array]):
                map_array = [list(x) for x in map_array]
            elif all([isinstance(x, list) for x in map_array]) and all(
                    [isinstance(y, str) for x in map_array for y in x]):
                pass
            else:
                raise ValueError("Wrong map")

            arr = map_array
        elif isinstance(map_array, np.ndarray) and map_array.dtype == '<U1':
            arr = map_array
        else:
            raise ValueError("Wrong map")

        arr = np.array(arr, dtype='<U1')
        # checking that all symbols are in the list of symbols
        for sym in arr.flatten():
            assert sym in self.SYMBOLS.values(), f"Unknown symbol {sym}"

        shape = arr.shape
        maps = {k: np.zeros(shape, dtype=np.bool) for k in self.OBJECTS}
        maps['empty'][:, :] = True

        # going over objects in the inner map
        for obj in self.INNER_OBJECTS:
            matched_symbols = [y for x, y in self.SYMBOLS.items() if x == obj]
            assert len(matched_symbols) == 1, f"Too many matched symbols {matched_symbols}"
            symbol = matched_symbols[0]
            mask = (arr == symbol)
            maps[obj][mask] = True

        super(KeyChestEnvironmentFixedMap, self).__init__(labyrinth_maps=maps, **kwargs)


@gin.configurable
class KeyChestGymEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    """Exporting KeyChest envrironment to Gym."""

    def __init__(self, engine_constructor, reward_dict=None, **kwargs):
        kwargs['callback'] = self.callback
        self.engine_kwargs = kwargs
        default_reward_dict = {'step': 0}
        if reward_dict:
            for key in default_reward_dict.keys():
                assert key in reward_dict
            self.reward_dict = reward_dict
        else:
            self.reward_dict = default_reward_dict
        self.engine_constructor = engine_constructor
        self.engine = None
        self.reset()
        self.observation_space = gym.spaces.Box(high=np.float32(1.0), low=np.float32(0.0), dtype=np.float32,
                                                shape=self.engine.observation.shape)
        self.action_space = gym.spaces.Discrete(len(self.engine.ACTIONS))
        # self.spec =

    def reset(self):
        """Create a new engine and return the observation."""
        if self.engine:
            del self.engine
        self.done = False
        self.reward = 0.0
        self.info = {}
        self.engine = self.engine_constructor(**self.engine_kwargs)
        return self.engine.observation

    def callback(self, event, moves):
        ev = event['event']
        event['moves'] = moves
        self.info = event
        if ev == 'dead':
            self.done = True
        if ev in self.reward_dict:
            self.reward += self.reward_dict[ev]

    def step(self, action):
        obs = self.engine.step(action)
        rew = self.reward
        done = self.done
        info = self.info
        rew += self.reward_dict['step']
        result = (obs, rew, done, info)
        self.reward = 0.0
        return result

    def step_string(self, action_str):
        dxdy = self.engine.ACTION_NAME_REVERSE[action_str]
        act = self.engine.ACTIONS_REVERSE[dxdy]
        return self.step(act)

    def render(self, mode='rgb_array'):
        frame = self.engine.render(mode)
        return frame
