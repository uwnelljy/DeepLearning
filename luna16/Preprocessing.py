import csv
import glob
import os
from Helper import Helper
from collections import namedtuple
from functools import lru_cache
import SimpleITK as sitk
import numpy as np


class Candidate:
    def __init__(self, requireOnDisk_bool):
        self.requireOnDisk_bool = requireOnDisk_bool

    @lru_cache(1)
    def GetCandidateInfo(self):
        # get the series uid of files present on the disk
        mhd_full_file_name = glob.glob('./data/subset*/*.mhd')
        # the full directory name
        # '/data/subset0/shkasdhfdsnkdfkak1244345.mhd'
        # os.path.split(p): ['/data/subset0/', 'shkasdhfdsnkdfkak1244345.mhd']
        presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_full_file_name}

        # get the coordinates and diameter information from annotation.csv
        annotationDiameterXYZ_set = {}
        with open('./data/annotations.csv') as f:
            # csv.reader is needed to generate list separated by ','
            # [1:]: the first row is header
            # we need to convert it to list so that it is subscriptable
            for row in list(csv.reader(f))[1:]:
                series_uid = row[0]
                annotationXYZ = tuple([float(x) for x in row[1:4]])  # change a list to a tuple using tuple()
                annotationDiameter = float(row[-1])
                # dic.setdefault(a,b): if a is in dic, then return dic[a]
                #                      if a is not in dic, then set dic[a] = b and return
                # one series_uid could have more than one nodules
                # {id: [(xyz1, d1), (xyz2, d2),...]}
                annotationDiameterXYZ_set.setdefault(series_uid, []).append(
                    (annotationXYZ, annotationDiameter) # generate a tuple using ()
                )

        # get the coordinates and diameter information from candidate.csv
        candidateInfoTuple = namedtuple('candidateInfoTuple',
                                        'series_uid, candidateDiameter, candidateXYZ, isNodule_bool')
        candidateInfo_list = []  # why we create a list here?
        with open('./data/candidates.csv') as f:
            for row in list(csv.reader(f))[1:]:
                series_uid = row[0]
                if series_uid not in presentOnDisk_set and self.requireOnDisk_bool:
                    continue
                isNodule_bool = bool(int(row[-1]))
                candidateXYZ = tuple([float(x) for x in row[1:4]])
                # now we need to create diameters for each candidate
                candidateDiameter = 0.0
                # dic.get(a,b): if a is in dic, then return dic[a]
                #               if a is not in dic, then return b
                for annotationDiameterXYZ in annotationDiameterXYZ_set.get(series_uid, []):
                    annotationXYZ, annotationDiameter = annotationDiameterXYZ
                    distance = Helper.euclidean_distance(annotationXYZ, candidateXYZ)
                    # if the two center is really close, they should be the same one
                    if distance <= annotationDiameter / 2:
                        candidateDiameter = annotationDiameter
                        # find one and break, otherwise continue the loop
                        break
                    # if we don't find one in the end, this candidate has 0 diameter
                candidateInfo_list.append(candidateInfoTuple(series_uid, candidateDiameter,
                                                             candidateXYZ, isNodule_bool))
                # we want the nodules with large diameter to be at the first
                # sort will sort the first numeric element, which is the candidateDiameter here
            candidateInfo_list.sort(reverse=True)
        return candidateInfo_list


class CtLoader:
    def __init__(self, series_uid):
        self.series_uid = series_uid
        self.ct_mhd, self.ct_np = self.get_raw_Ct()

    @lru_cache(1, typed=True)
    def get_raw_Ct(self):
        mhd_path = glob.glob('./data/subset*/{}.mhd'.format(self.series_uid))[0]
        ct_mhd = sitk.ReadImage(mhd_path)  # this function addresses .raw file at the same time
        # ct_mhd contains all the information of this file including:
        # origin, spacing, direction
        # we need to convert it to np.array
        ct_np = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # set all the values>=max to max, and values <= min to min
        ct_np.clip(min=-1000, max=1000, out=ct_np)
        return ct_mhd, ct_np

    @staticmethod
    def get_raw_chunk(ircTuple, width, ct_np):
        slice_list = [] # the slice targets at the corresponding axis
        for axis, center in enumerate(ircTuple):
            start = int(round(center-width[axis]/2))
            end = int(start + width[axis])

            # special cases: if it reaches the boundary of the image
            if start < 0:
                start = 0
                end = int(width[axis])
            if end > ct_np.shape[axis]:
                end = ct_np.shape[axis]
                start = int(end-width[axis])

            slice_list.append(slice(start, end))
        ct_chunk = ct_np[tuple(slice_list)]
        return ct_chunk

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
