import csv
import glob
import os
from Helper import CandidateInfo_tuple, xyz2irc
from functools import lru_cache
import SimpleITK as sitk
import numpy as np
import torch as t
import random
import math
from torch.nn import functional


@lru_cache(1)
def getCandidateInfo():
    # files that are on the disk
    mhd_file_name = glob.glob('./data/subset*/*.mhd')
    presentOnDisk_series_uid_set = {os.path.split(pathname)[1][:-4] for pathname in mhd_file_name}

    CandidateInfo_list = []
    CandidateInfo_dict = {}

    # get annotations from annotation file which only contains nodule
    with open('./data/annotations_with_malignancy.csv') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            # check whether on disk
            if series_uid not in presentOnDisk_series_uid_set:
                continue

            center_xyz = tuple([float(coord) for coord in row[1:4]])
            diameter_mm = float(row[4])
            isMal_bool = {'False': False, 'True': True}[row[5]]
            info = CandidateInfo_tuple(
                True, True, isMal_bool, diameter_mm, series_uid, center_xyz
            )
            CandidateInfo_list.append(info)
            CandidateInfo_dict.setdefault(series_uid, []).append(info)

    # get non-nodule candidate information
    with open('./data/candidates.csv') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            is_nodule = bool(int(row[4]))

            # check whether on disk
            if (series_uid not in presentOnDisk_series_uid_set) or is_nodule:
                continue

            center_xyz = tuple([float(coord) for coord in row[1:4]])
            info = CandidateInfo_tuple(
                False, False, False, 0, series_uid, center_xyz
            )
            CandidateInfo_list.append(info)
            CandidateInfo_dict.setdefault(series_uid, []).append(info)

    # sort by diameter_mm
    CandidateInfo_list.sort(reverse=True)

    return CandidateInfo_list, CandidateInfo_dict


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

    def getChunkCandidate(self, xyz, width):
        # get center index of nodules
        irc = xyz2irc(xyz, self.origin, self.spacing, self.direction)

        # get ct chunk
        slice_list = []  # the slice targets at the corresponding axis
        for axis, center in enumerate(irc):
            start = int(round(center - width[axis] / 2))
            end = int(start + width[axis])

            # special cases: if it reaches the boundary of the image
            if start < 0:
                start = 0
                end = int(width[axis])
            if end > self.ct_np.shape[axis]:
                end = self.ct_np.shape[axis]
                start = int(end - width[axis])
            slice_list.append(slice(start, end))

        ct_chunk = self.ct_np[tuple(slice_list)]
        # convert to tensor
        ct_chunk_t = t.from_numpy(ct_chunk).to(t.float32)
        # add one additional dimension 'channel' because ct image only has one channel
        # now C D H W
        ct_chunk_t = ct_chunk_t.unsqueeze(0)
        return ct_chunk_t, irc


@lru_cache(1, typed=True)
def getct(series_uid):
    return CtLoader(series_uid)
