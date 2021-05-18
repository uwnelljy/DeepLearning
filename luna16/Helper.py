import math


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def euclidean_distance(x, y):
        return math.sqrt(sum([(i-j)**2 for i, j in zip(x, y)]))
