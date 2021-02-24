from functools import partial
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import gym
import gin
from gym.wrappers import TransformObservation
from functools import lru_cache


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
    """,

    # A
    10: """
    111
    101
    111
    101
    101
    """,

    # B
    11: """
    111
    111
    111
    101
    111
    """,

    # C
    12: """
    111
    100
    100
    100
    111
    """,

    # D
    13: """
    110
    101
    101
    101
    110
    """,

    # E
    14: """
    111
    100
    111
    100
    111
    """,

    # F
    15: """
    111
    100
    111
    100
    100
    """,

    # G
    16: """
    111
    100
    111
    111
    111
    """,

    # H
    17: """
    101
    101
    111
    101
    101
    """,

    # I
    18: """
    111
    010
    010
    010
    111
    """,

    # J
    19: """
    111
    001
    001
    011
    111
    """,

    # K
    20: """
    101
    110
    110
    110
    101
    """,

    # L
    21: """
    100
    100
    100
    100
    111
    """,

    # M
    22: """
    101
    111
    111
    101
    101
    """,
}


@lru_cache(128)
def digit_to_np(digit, digits=digits):
    """Convert a digit (string of 0, 1) into an np array."""
    val = [x.strip() for x in digits[digit].strip().splitlines()]
    val = np.array([[y == '1' for y in x] for x in val])
    return val


def show_digits(digits):
    """Show the digits."""
    sqdim = ceil(len(digits) ** 0.5)
    ncol = sqdim
    nrow = sqdim

    plt.figure(figsize=(10, 10))

    for i, key in enumerate(sorted(digits.keys())):
        #         print(key)
        val = digit_to_np(key) * 1.0
        plt.subplot(nrow, ncol, i + 1)
        plt.title(key)
        plt.imshow(val)
    plt.show()

@gin.configurable
def small_int_vector_asimage(v, max_digits=1, eps=1e-8, max_digit_value=10):
    """Convert a vector of integers in 0-9 into a binary image of size 5 x max_digits x (3 * n + n - 1)."""
    use_digits = max_digits if max_digits > 0 else 1
    n_digits = len(v) * use_digits
    result = np.zeros((5, 3 * n_digits + n_digits - 1), dtype=np.float32)
    offset = 0
    for i, val in enumerate(v):
        rval = round(val)
        assert abs(rval - val) < eps

        # print('md', max_digits)
        if max_digits <= 0:
            ds = [rval]
        else:
            ds = f"%0{round(max_digits)}d" % rval
            assert len(ds) == max_digits
        for d in ds:
            #             print(d)
            result[:, offset:offset + 3] = digit_to_np(int(d))
            offset += 4
    return result


@gin.configurable
class DigitsVectorWrapper(TransformObservation):
    """Convert a vector of integers into small images."""

    def __init__(self, env, **kwargs):
        if isinstance(env, str):
            env = gym.make(env)
        fcn = partial(small_int_vector_asimage, **kwargs)
        super(DigitsVectorWrapper, self).__init__(env, fcn)
        shape = fcn(env.reset()).shape
        self.observation_space = gym.spaces.Box(low=np.float32(0),
                                                high=np.float32(1),
                                                shape=shape)
