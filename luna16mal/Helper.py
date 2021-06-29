import math
from collections import namedtuple
import numpy as np

CandidateInfo_tuple = namedtuple(
    'CandidateInfo_tuple',
    'isNodule_bool, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')
ircTuple = namedtuple('ircTuple', 'index, row, col')
xyzTuple = namedtuple('xyzTuple', 'x, y, z')
classificationTuple = namedtuple('classificationTuple', 'prob_nodule, prob_mal, center_xyz, center_irc')


def euclidean_distance(x, y):
    return math.sqrt(sum([(i - j) ** 2 for i, j in zip(x, y)]))


def irc2xyz(ircTuple, origin, spacing, direction):
    cri = np.array(ircTuple)[::-1]
    origin = np.array(origin)
    spacing = np.array(spacing)
    direction = np.array(direction)
    xyz = ((cri * spacing) @ direction) + origin
    return xyzTuple(*xyz)


def xyz2irc(xyzTuple, origin, spacing, direction):
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
    cri = ((xyz - origin) @ np.linalg.inv(direction)) / spacing
    cri = np.round(cri)
    # change type to int
    # z corresponds to index
    return ircTuple(int(cri[2]), int(cri[1]), int(cri[0]))


def diceloss(label, prediction, epsilon=1):
    """
    :param label: tensor bool
    :param prediction: tensor
    :param epsilon: if dice_label = 0, dice_prediction = 0, we need epsilon to avoid problems
    :return:
    """
    dice_label = label.sum(dim=[1, 2, 3])  # here we want to keep the first dimension: batch number
    dice_prediction = prediction.sum(dim=[1, 2, 3])
    dice_label_prediction = (label * prediction).sum(dim=[1, 2, 3])

    # the dice_score should be a tensor rather than a scalar because we have multiple samples in this batch
    dice_score = (2 * dice_label_prediction + epsilon) / (dice_label + dice_prediction + epsilon)

    return 1 - dice_score
