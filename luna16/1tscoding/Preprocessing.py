import csv
import glob
import os
from Helper import Helper
from collections import namedtuple
from functools import lru_cache
import SimpleITK as sitk
import numpy as np
import torch as t
import random
import math
from torch.nn import functional


@lru_cache(1)
def GetCandidateInfo():
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
            if series_uid not in presentOnDisk_set:
                continue
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
            if series_uid not in presentOnDisk_set:
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
        mhd_path = glob.glob('./data/subset*/{}.mhd'.format(self.series_uid))[0]
        ct_mhd = sitk.ReadImage(mhd_path)  # this function addresses .raw file at the same time
        # ct_mhd contains all the information of this file including:
        # origin, spacing, direction
        # we need to convert it to np.array
        self.ct_np = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        # set all the values>=max to max, and values <= min to min
        self.ct_np.clip(min=-1000, max=1000, out=self.ct_np)
        self.origin = ct_mhd.GetOrigin()
        self.spacing = ct_mhd.GetSpacing()
        self.direction = ct_mhd.GetDirection()

    def get_raw_chunk(self, ircTuple, width):
        slice_list = [] # the slice targets at the corresponding axis
        for axis, center in enumerate(ircTuple):
            start = int(round(center-width[axis]/2))
            end = int(start + width[axis])

            # special cases: if it reaches the boundary of the image
            if start < 0:
                start = 0
                end = int(width[axis])
            if end > self.ct_np.shape[axis]:
                end = self.ct_np.shape[axis]
                start = int(end-width[axis])

            slice_list.append(slice(start, end))
        ct_chunk = self.ct_np[tuple(slice_list)]
        return ct_chunk

    def getChunkCandidate(self, xyz, width):
        # get center index of nodules
        irc = Helper.xyz2irc(xyz, self.origin, self.spacing, self.direction)
        # get ct chunk
        ct_chunk = self.get_raw_chunk(irc, width)
        # convert to tensor
        ct_chunk_t = t.from_numpy(ct_chunk).to(t.float32)
        # add one additional dimension 'channel' because ct image only has one channel
        # now C D H W
        ct_chunk_t = ct_chunk_t.unsqueeze(0)
        return ct_chunk_t, irc

    def getAugmentedCandidate(self, augmentation, xyz, width):
        """
        :param augmentation: is a dictionary containing translation information
                             'flip': mirror, bool
                             'offset': the maximum offset expressed in the same scale as the [-1, 1] range, float
                             'scale': scale, float
                             'rotate': rotate image, bool
                             'noise': add noise, float
        :param xyz: center
        :param width: width of subsetting
        :return:
        """
        # get chunk before augmentation
        ct_chunk_t, irc = self.getChunkCandidate(xyz, width)
        # add one dimension: batch size, now is N C D H W
        ct_chunk_t_whole_dimension = ct_chunk_t.unsqueeze(0).to(dtype=t.float32)

        # define translation matrix
        # which should be:
        # [[1, 0, 0, a]
        #   0, 1, 0, b]
        #   0, 0, 1, c]]
        # the diagonal matrix controls the scale, while abc controls offset.
        theta = t.eye(4)  # generate a 4*4 standard diagonal matrix, we would use the first 3 rows

        # modify translation matrix
        for i in range(3):
            if 'flip' in augmentation:
                if random.random() > 0.5:  # uniform distribution [0, 1]
                    theta[i, i] *= -1

            if 'offset' in augmentation:
                offset = augmentation['offset']  # offset is a list: [a, b, c]
                random_effects = random.random() * 2 - 1  # we don't want a large offset to destruct the sample
                theta[i, 3] = offset * random_effects

            if 'scale' in augmentation:
                scale = augmentation['scale']
                random_effects = random.random() * 2 - 1
                theta[i, i] *= 1 + scale * random_effects

            if 'rotate' in augmentation:
                angle = random.random() * math.pi * 2  # we don't specify the angle
                cos = math.cos(angle)
                sin = math.sin(angle)
                # we only rotate dimension H and W and keep the D the same because it has a different value
                rotation = t.tensor([
                    [cos, -sin, 0, 0],
                    [sin, cos, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                theta @= rotation

        # create grid
        grid = functional.affine_grid(theta[:3].unsqueeze(0).to(dtype=t.float32),
                                      ct_chunk_t_whole_dimension.size())
        # padding_mode='border':对于越界的位置在网格中采用边界的pixel value进行填充。
        augmented_chunk = functional.grid_sample(ct_chunk_t_whole_dimension,
                                                 grid, padding_mode='border')[0]

        # add noise
        if 'noise' in augmentation:
            noise = t.rand_like(augmented_chunk)  # generate random values having the same dimension as augmented_chunk
            noise *= augmentation['noise']
            augmented_chunk += noise

        return augmented_chunk, irc


@lru_cache(1, typed=True)
def getct(series_uid):
    return CtLoader(series_uid)