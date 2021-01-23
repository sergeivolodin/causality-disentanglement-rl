from keychest.keychestenv import fill_n
import numpy as np


def dict_to_arr(d):
    """Dictionary to a list with keys sorted."""
    return np.array([d[k] for k in sorted(d.keys())], dtype=np.float32)

def arr_to_dict(arr, keys):
    """Array back to dictionary."""
    assert len(arr) == len(keys)
    return {x: y for x, y in zip(sorted(keys), arr)}


def obs_features_handcoded(engine, obs=None):
    """Get features for an observation."""

    if obs is None:
        obs = engine._observation

    def get_map(key):
        idx = engine.OBJECTS.index(key)
        return obs[:, :, idx]

    result = {}
    result['health'] = np.sum(get_map('health'))
    result['keys'] = np.sum(get_map('keys_collected'))
    lamp_on = np.sum(get_map('lamp_on'))
    lamp_off = np.sum(get_map('lamp_off'))
    result['lamp_status'] = 1 if lamp_on else (0 if lamp_off else -1)

    n_keys = engine.n_keys_init
    n_chests = engine.n_chests_init
    n_food = engine.n_food_init

    objects_to_out = ['key', 'chest', 'food', 'button', 'player', 'lamp_on', 'lamp_off']
    objects_cnt = {'key': n_keys, 'chest': n_chests, 'food': n_food}

    def register(key, x, y, idx=None):
        """Add a variable to the result."""
        if x is None:
            x = -1
        if y is None:
            y = -1
        name_prefix = key + "__"
        if idx is not None:
            name_prefix = "%s%02d__" % (name_prefix, idx)

        result[name_prefix + "x"] = x
        result[name_prefix + "y"] = y

    def arr_get_val(arr, idx):
        """Get coordinates in array, or -1, -1."""
        if idx < len(arr):
            x, y = arr[idx]
        else:
            x, y = -1, -1
        return x, y

    for key in objects_to_out:
        m = get_map(key)
        where = sorted(list(zip(*np.where(m))))
        #         print(key, where)
        assert len(where) <= 1 or key in objects_cnt

        if key in objects_cnt:
            for i in range(objects_cnt[key]):
                x, y = arr_get_val(where, i)
                register(key, x, y, i)
        else:
            x, y = arr_get_val(where, 0)
            register(key, x, y, None)

    return result


def reconstruct_image_from_features(engine, dct):
    """Reconstruct the observation from features."""

    sx, sy = engine.shape
    dy1 = 1
    dy2 = 1
    dy = dy1 + dy2
    dx1 = 1 + engine.food_rows + engine.keys_rows
    dx2 = 1
    dx = dx1 + dx2

    shape = (sx + dx, sy + dy)

    obj = engine.OBJECTS
    obs_reconstruct = np.full(fill_value=False, shape=(*shape, len(engine.OBJECTS)), dtype=np.bool)

    fill_n(obs_reconstruct[:, :, obj.index('health')], 0, int(dct['health']))
    fill_n(obs_reconstruct[:, :, obj.index('keys_collected')], engine.food_rows, int(dct['keys']))

    walls = obs_reconstruct[:, :, obj.index('wall')]
    walls[dx1 - 1, :] = True
    walls[-1, :] = True
    walls[dx1 - 1:, 0] = True
    walls[dx1 - 1:, -1] = True

    objects_to_out = ['key', 'chest', 'food', 'button', 'player', 'lamp_on', 'lamp_off']

    for o in objects_to_out:
        prefixes = set(map(lambda x: x[:-1], filter(lambda x: x.startswith(o + '__'), dct.keys())))
        for p in prefixes:
            name = p.split('__')[0]
            out = obs_reconstruct[:, :, obj.index(name)]
            x, y = dct[p + 'x'], dct[p + 'y']
            if x > 0 and y > 0:
                out[int(x), int(y)] = True

    empty = obs_reconstruct[:, :, 0]
    empty[dx1:-1, 1:-1] = True

    return obs_reconstruct