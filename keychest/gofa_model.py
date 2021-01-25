import numpy as np


def extract_objects(f_t):
    """Get objects coordinates as a list."""
    objs_k = ['key', 'chest', 'food']
    objs = {}
    for obj in objs_k:
        if obj not in objs:
            objs[obj] = []

        keys = f_t.keys()
        keys = filter(lambda x: x.startswith(obj + '_'), keys)
        keys = map(lambda x: x[:-1], keys)
        keys = set(keys)
        #         print(list(keys))
        for k in keys:
            objs[obj].append((f_t[k + 'x'], f_t[k + 'y']))
    return objs


def manual_model_features(f_t, a_t, engine):
    """Do one step with features."""
    assert isinstance(f_t, dict)
    assert isinstance(a_t, np.ndarray)
    assert a_t.shape == (4,)

    f_t = dict(f_t)
    f_t = {x: round(y) for x, y in f_t.items()}

    f_t['health'] -= 1

    objs = extract_objects(f_t)

    def obj_coord(pref):
        return (f_t[pref + '__x'], f_t[pref + '__y'])

    player = obj_coord('player')

    def add_obj(k, x, y):
        f_t[k + "__x"] = x
        f_t[k + "__y"] = y

    def player_at(k):
        return any([player == o for o in objs[k]])

    def obj_id(k):
        return objs[k].index(player)

    def rm_obj(k):
        if f"{k}__x" in f_t:
            f_t[f"{k}__x"] = -1
            f_t[f"{k}__y"] = -1
        else:
            idx = obj_id(k)
            f_t["%s__%02d__x" % (k, idx)] = -1
            f_t["%s__%02d__y" % (k, idx)] = -1

    if player_at('food'):
        f_t['health'] += engine.food_efficiency
        rm_obj('food')

    if player_at('key'):
        f_t['keys'] += 1
        rm_obj('key')

    if player_at('chest') and f_t['keys'] > 0:
        rm_obj('chest')
        f_t['keys'] -= 1

    if player == obj_coord('button') and obj_coord('lamp_on')[0] != -1:
        add_obj('lamp_off', *obj_coord('lamp_on'))
        rm_obj('lamp_on')
        f_t['lamp_status'] = 0
    elif player == obj_coord('button') and obj_coord('lamp_off')[0] != -1:
        add_obj('lamp_on', *obj_coord('lamp_off'))
        rm_obj('lamp_off')
        f_t['lamp_status'] = 1

    def clip(x, m, M):
        if x > M:
            return M
        elif x < m:
            return m
        return x

    xmin, ymin = engine.food_rows + engine.keys_rows + 1, 1
    xmax, ymax = engine.height + engine.food_rows + engine.keys_rows, engine.width

    dx, dy = engine.ACTIONS[a_t.argmax()]
    #     print(dx, dy, a_t)
    f_t['player__x'] = clip(f_t['player__x'] + dx, xmin, xmax)
    f_t['player__y'] = clip(f_t['player__y'] + dy, ymin, ymax)

    #     f_t['player__y'] += dy
    #     print(dx, dy)

    return f_t