from functools import partial
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import gym
import gin
from gym.wrappers import TransformObservation


### 5x3 binary digits
digits = {
    0: """
    010
    101
    101
    101
    010
    """,
    1:
        """
        000
        010
        010
        010
        000
        """,
    2:
        """
        111
        001
        111
        100
        111
        """,
    3: """
    111
    001
    111
    001
    111
    """,
    4: """
    101
    101
    111
    001
    001
    """,
    5: """
    111
    100
    111
    001
    111
    """,
    6: """
    111
    100
    111
    101
    111
    """,
    7: """
    111
    001
    001
    001
    001
    """,
    8: """
    111
    101
    111
    101
    111
    """,
    9: """
    111
    101
    111
    001
    001
    """
}


def digit_to_np(digit):
    """Convert a digit (string of 0, 1) into an np array."""
    val = [x.strip() for x in digit.strip().splitlines()]
    val = np.array([[y == '1' for y in x] for x in val])
    return val


def show_digits(digits):
    """Show the digits."""
    sqdim = ceil(len(digits) ** 0.5)
    ncol = sqdim
    nrow = sqdim

    plt.figure(figsize=(10, 10))

    for i, key in enumerate(sorted(digits.keys())):
        value = digits[key]
        #         print(key)
        val = digit_to_np(value) * 1.0
        plt.subplot(nrow, ncol, i + 1)
        plt.title(key)
        plt.imshow(val)
    plt.show()

def small_int_vector_asimage(v, max_digits=1):
    """Convert a vector of integers in 0-9 into a binary image of size 5 x max_digits x (3 * n + n - 1)."""
    n_digits = len(v) * max_digits
    result = np.zeros((5, 3 * n_digits + n_digits - 1), dtype=np.float32)
    offset = 0
    for i, val in enumerate(v):
        assert np.allclose(np.round(val), val)
        val = round(val)
        ds = f"%0{round(max_digits)}d" % val
        assert len(ds) == max_digits
        for d in ds:
            #             print(d)
            result[:, offset:offset + 3] = digit_to_np(digits[int(d)])
            offset += 4
    return result


@gin.configurable
class DigitsVectorWrapper(TransformObservation):
    """Convert a vector of integers into small images."""

    def __init__(self, env, max_digits=2, **kwargs):
        if isinstance(env, str):
            env = gym.make(env)
        fcn = partial(small_int_vector_asimage, max_digits=max_digits)
        super(DigitsVectorWrapper, self).__init__(env, fcn)
        shape = fcn(env.reset()).shape
        self.observation_space = gym.spaces.Box(low=np.float32(0),
                                                high=np.float32(1),
                                                shape=shape)
