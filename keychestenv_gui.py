from keychestenv import KeyChestEnvironmentRandom, KeyChestGymEnv
from IPython.display import clear_output
from time import sleep
from matplotlib import pyplot as plt
import cv2
import numpy as np

def gui_for_env(env):
    """cv2 GUI for the KeyChest environment."""
    env.reset()

    def show_rendered(scale=3, text=''):
        image = np.array(env.render(mode='rgb_array'), dtype=np.float32)
        new_shape = tuple([int(x) for x in np.array(image.shape) * scale][:2])
        image = cv2.resize(image, new_shape, interpolation = cv2.INTER_NEAREST)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Key Chest Environment", image)

    show_rendered()

    while True:
        k = cv2.waitKey(1)

        if k == -1:
            sleep(0.1)
            continue

        key_map = {82: 'up', 84: 'down', 81: 'left', 83: 'right'}

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

    while True:
        key = input()
        if key == 'x':
            break
        if key == 'r':
            env.reset()
            clear_output()
            plt.imshow(env.render('rgb_array'))
            plt.show()
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
        plt.show()
        plt.imshow(env.render('rgb_array'))
        plt.show()
        print(key, dxdy, act, rew, done, info)
        if done:
            env.reset()

if __name__ == '__main__':
    reward = {'step': -1, 'food_collected': 3, 'key_collected': 4, 'chest_opened': 5}
    env = KeyChestGymEnv(engine_constructor=KeyChestEnvironmentRandom,
                         width=10, height=10, initial_health=10, food_efficiency=10,
                         reward_dict=reward)
    gui_for_env(env)
