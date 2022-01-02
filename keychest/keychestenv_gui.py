import argparse
from time import sleep

import cv2
import gin
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt

from keychest.keychestenv import KeyChestGymEnv
from keychest.keychestenv_gofa import hardcoded_policy_step
import gin

parser = argparse.ArgumentParser("Play the KeyChest environment in a GUI manually")
parser.add_argument("--config", type=str, default="config/5x5.gin")
parser.add_argument("--solver", action='store_true')
parser.add_argument("--scale", type=float, help="Scale for images", default=3)


@gin.configurable
def show_rendered(scale=3, text=''):
    image = np.array(env.render(mode='rgb_array'), dtype=np.float32) / 255.
    old_shape = np.array(image.shape)[:2][::-1]
    new_shape = old_shape * scale
    new_shape = tuple([int(x) for x in new_shape])
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Key Chest Environment", image)

def gui_for_env_gofa(env):
    """cv2 GUI for the KeyChest environment."""
    env.reset()
    show_rendered()

    while True:
        k = cv2.waitKey(0)

        if k == ord('x'):
            break

        action = hardcoded_policy_step(env)
        obs, rew, done, info = env.step(action)
        text = f"reward={rew}"
        if done:
            text = "done " + text
            sleep(0.2)
            env.reset()
            show_rendered()
            continue

        show_rendered(text=text)
        sleep(0.1)


    cv2.destroyAllWindows()

def gui_for_env(env):
    """cv2 GUI for the KeyChest environment."""
    env.reset()
    show_rendered()

    while True:
        k = cv2.waitKey(1)

        if k == -1:
            sleep(0.1)
            continue

        key_map = {ord('w'): 'up', ord('s'): 'down', ord('a'): 'left', ord('d'): 'right'}

        if k == ord('x'):
            break
        elif k == ord('r'):
            env.reset()
            show_rendered()
            continue

        if k not in key_map: continue

        obs, rew, done, info = env.step_string(key_map[k])
        text = f"reward={rew}"
        if done:
            text = "done " + text

        show_rendered(text=text)

    cv2.destroyAllWindows()


def jupyter_gui(env):
    """GUI for jupyter notebook and input()."""
    env.reset()
    plt.imshow(env.render('rgb_array'))
    plt.show()
    R = 0

    while True:
        key = input()
        if key == 'x':
            break
        if key == 'r':
            env.reset()
            clear_output()
            plt.imshow(env.render('rgb_array'))
            plt.show()
            R = 0
            continue
        clear_output()
        mapping = {'w': 'up', 'a': 'left', 's': 'down', 'd': 'right'}
        try:
            key = mapping[key]
            dxdy = env.engine.ACTION_NAME_REVERSE[key]
            act = env.engine.ACTIONS_REVERSE[dxdy]
        except:
            print("Wrong action")
        obs, rew, done, info = env.step(act)
        R += rew
        plt.show()
        plt.imshow(env.render('rgb_array'))
        plt.show()
        print(key, dxdy, act, rew, done, info, R)
        if done:
            print("TOTAL REWARD", R)
            env.reset()


if __name__ == '__main__':
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    env = KeyChestGymEnv(flatten_observation=False)
    print("Observation shape:", env.reset().shape)
    gin.bind_parameter("show_rendered.scale", args.scale)
    if args.solver:
        gui_for_env_gofa(env)
    else:
        gui_for_env(env)
