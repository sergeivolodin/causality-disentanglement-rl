from keychestenv import KeyChestEnvironmentRandom, KeyChestGymEnv, KeyChestEnvironment
from keychestenv_gui import jupyter_gui
from matplotlib import pyplot as plt
import numpy as np

def features_for_obs(obs):
    """Features for an observation."""
    
    objects = KeyChestEnvironment.OBJECTS
    
    def get_map(obj):
        idx = objects.index(obj)
        return obs[:, :, idx]
    
    def get_where(obj):
        m = get_map(obj)
        return list(zip(*np.where(m)))
    
    def get_where1(obj):
        r = get_where(obj)
        assert len(r) == 1
        return r[0]
    
    ppos = get_where1('player')
    
    def is_present(obj):
        r = get_where(obj)
        return len(r) > 0
    
    def items_at_player():
        player_location = ppos
        result = []
        for i, obj in enumerate(objects):
            if obs[player_location[0], player_location[1], i]:
                result.append(obj)
        return result
    
    assert obs.shape[2] == len(objects)
    result = {}
    
    
    
    result['player_position_x'] = ppos[0]
    result['player_position_y'] = ppos[1]
    if is_present('lamp_on'):
        result['lamp_state'] = 1
    elif is_present('lamp_off'):
        result['lamp_state'] = 0
    else:
        result['lamp_state'] = -1
    
    items = items_at_player()
    
    result['at_food'] = 'food' in items
    result['at_key'] = 'key' in items
    result['at_chest'] = 'chest' in items
    result['at_button'] = 'lamp_on' in items or 'lamp_off' in items
    result['health'] = np.sum(get_map('health'))
    result['keys_collected'] = np.sum(get_map('keys_collected'))
        
    return result


def max_reward(env):
    """Return reward upper bound."""
    obs = env.reset()
    
    objects = KeyChestEnvironment.OBJECTS
    
    def get_map(obs, obj):
        return obs[:, :, objects.index(obj)]
    
    max_reward_ = 0
    rd = env.reward_dict

    n_keys = np.sum(get_map(obs, 'key'))
    n_chests = np.sum(get_map(obs, 'chest'))
    n_food = np.sum(get_map(obs, 'food'))

    if 'key_collected' in rd:
        max_reward_ += n_keys * rd['key_collected']
    if 'food_collected' in rd:
        max_reward_ += n_food * rd['food_collected']
    if 'chest_opened' in rd:
        max_reward_ += min(n_keys, n_chests) * rd['chest_opened']

    max_reward_ += (n_food * env.engine.food_efficiency + env.engine.initial_health) * rd['step']
        
    return max_reward_

def hardcoded_policy_step(env, do_print=False):
    """Get a step by a hardcoded policy."""
    obs = env.engine.observation
    objects = env.engine.OBJECTS
    
    def get_map(obs, obj):
        return obs[:, :, objects.index(obj)]

    def get_objects(obs, obj):
        return list(zip(*np.where(get_map(obs, obj))))

    def closest_object(obs, obj):
        ppos = get_objects(obs, 'player')[0]
        objects = get_objects(obs, obj)
        if objects:
            distances = np.linalg.norm(np.array(objects) - np.array(ppos), axis=1, ord=1)
            closest_idx = np.argmin(distances)
        else:
            distances = []
            closest_idx = -1
        
        result = {'distances': distances, 'ppos': ppos, 'objects': objects,
                  'closest_idx': closest_idx,
                  'n': len(objects)}
        
        if closest_idx >= 0:
            result['smallest_distance'] = distances[closest_idx]
            result['closest_object'] = objects[closest_idx]
        return result


    health = features_for_obs(obs)['health']
    keys = features_for_obs(obs)['keys_collected']

    button_pos = list(zip(*np.where(get_map(obs, 'lamp_on') + get_map(obs, 'lamp_off'))))[0]

    key_info = closest_object(obs, 'key')
    chest_info = closest_object(obs, 'chest')
    food_info = closest_object(obs, 'food')
    ppos = key_info['ppos']
    
    if do_print:
        print("Health", health, "Keys", keys)

    def dist_to(v):
        return np.linalg.norm(np.array(ppos) - v, ord=1)

    # have keys -> going for a chest
    if keys > 0 and chest_info['n']:
        target = chest_info['closest_object']
        if do_print:
            print('Going to the chest!', target)
    elif key_info['n']:
        target = key_info['closest_object']
        if do_print:
            print('Going for the key!', target)
    else:
        target = button_pos
        if do_print:
            print("Going for the button", target)

    if do_print:
        print("Dist", dist_to(target))
        
    # overriding target if there is food
    
    def health_alert():
        if health < 3:
            return True
        if health < dist_to(target) * 2:
            return True
        return False
    
    if health_alert() and food_info['n']:
        target = food_info['closest_object']
        if do_print:
            print('Going for food, hungry')

    dx, dy = np.array(target) - ppos

    to_sample_from = []
    if dx > 0:
        to_sample_from.append('down')
    elif dx < 0:
        to_sample_from.append('up')
    if dy > 0:
        to_sample_from.append('right')
    elif dy < 0:
        to_sample_from.append('left')

    if not to_sample_from:
        action = env.action_space.sample()
    else:
        action = np.random.choice(to_sample_from)
        dxdy = env.engine.ACTION_NAME_REVERSE[action]
        action = env.engine.ACTIONS_REVERSE[dxdy]
    return action
