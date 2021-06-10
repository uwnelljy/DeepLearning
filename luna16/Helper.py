import math
from collections import namedtuple
import numpy as np


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def euclidean_distance(x, y):
        return math.sqrt(sum([(i-j)**2 for i, j in zip(x, y)]))

    @staticmethod
    def xyz2irc(xyzTuple, origin, spacing, direction):
        ircTuple = namedtuple('ircTuple', 'index, row, col')
        xyz = np.array(xyzTuple)  # change all the variables to np.array
        origin = np.array(origin)
        spacing = np.array(spacing)
        direction = np.array(direction).reshape(3, 3)
        # np.linalg.inv(direction) / spacing:
        # [[1, 2, 3],     spacing: [1, 2, 3]
        # [2, 2, 2],
        # [3, 4, 6]]
        # then the value of np.linalg.inv(direction) / spacing is:
        # [[1/1, 2/2, 3/3],
        # [2/1, 2/2, 2/3],
        # [3/1, 4/2, 6/3]]
        irc = ((xyz - origin) @ np.linalg.inv(direction)) / spacing
        irc = np.round(irc)
        # change type to int
        # z corresponds to index
        return ircTuple(int(irc[2]), int(irc[1]), int(irc[0]))