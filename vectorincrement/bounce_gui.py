import argparse
from time import sleep

import cv2
import gin
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt

from vectorincrement.bounce import BounceEnv
import gin

parser = argparse.ArgumentParser("Play the Bounce environment in a GUI manually")
parser.add_argument("--config", type=str, default=None)
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
    cv2.imshow("Bounce Environment", image)


def gui_for_env(env):
    """cv2 GUI for the Bounce environment."""
    env.reset()
    show_rendered()

    while True:
        k = cv2.waitKey(1)

        if k == -1:
            k = 48
            sleep(0.01)
            # continue

        key_map = {48: 0, 49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, 55: 7, 56: 8}

        if k == ord('x'):
            break
        elif k == ord('r'):
            env.reset()
            show_rendered()
            continue

        if k not in key_map: continue

        x, y, vx, vy, gx, gy = env.state

        print("vx=%.2f vy=%.2f gx=%.2f gy=%.2f" % (vx, vy, gx, gy))

        obs, rew, done, info = env.step(key_map[k])
        text = f"reward={rew}"
        if done:
            text = "done " + text

        show_rendered(text=text)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
        gin.parse_config_file(args.config)
    env = BounceEnv()
    gui_for_env(env)
