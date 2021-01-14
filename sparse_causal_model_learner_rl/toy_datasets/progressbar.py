import gin
import numpy as np
from math import ceil
import matplotlib

@gin.configurable
def progressbar_image(values=None, colors=None,
                      max_values=None, width=10):
    """Create an image of a progress bar, each +1 is 1 pixel.

    Args:
        values: list of integers, each integer represents one progress bar.
        colors: list of colors, same length as 'values'
        max_values: maximal value for each progress bar, same length as 'values'
        width: integer, width of the resulting image

    Returns:
        A numpy array with shape (height, width, channels),
        where width is one of the arguments, channels=3 (rgb) and
        height is calculated based on the max_values array.
    """

    assert len(values) == len(colors) == len(max_values), "Lengths must be equal"
    for v, mv in zip(values, max_values):
        assert 0 <= v <= mv
        assert isinstance(v, int) and isinstance(mv, int)

    heights = [int(ceil(1. * mv / width)) for mv in max_values]
    height = sum(heights)

    result = np.zeros((height, width, 3))

    current_start = 0
    for c, v, mv, h in zip(colors, values, max_values, heights):
        c = matplotlib.colors.to_rgb(c)
        current_start_for_pb = current_start
        while v:
            cv = v
            if cv > width:
                cv = width

            result[current_start_for_pb, :cv] = c

            v -= cv
            current_start_for_pb += 1
        current_start += h

    return result