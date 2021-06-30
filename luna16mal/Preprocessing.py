import glob
import SimpleITK as sitk
import numpy as np
from Helper import xyz2irc, CandidateInfo_tuple
import functools
import os
import csv


@functools.lru_cache(1)
def getCandidateInfo(path):
    # files that are on the disk
    mhd_file_name = glob.glob('{}/*.mhd'.format(path))
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

    return presentOnDisk_series_uid_set, CandidateInfo_list, CandidateInfo_dict


class CtLoader:
    def __init__(self, series_uid):
        # get ct data
        ct_path = glob.glob('./data/subset*/{}.mhd'.format(series_uid))[0]
        ct_mhd = sitk.ReadImage(ct_path)
        self.ct_np = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.series_uid = series_uid
        self.origin_xyz = ct_mhd.GetOrigin()
        self.spacing_xyz = ct_mhd.GetSpacing()
        self.direction = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getChunk(self, center_xyz, width_irc):
        irc_tuple = xyz2irc(center_xyz, self.origin_xyz, self.spacing_xyz, self.direction)

        slice_list = []
        for dim, value in enumerate(irc_tuple):
            start = int(round(value - width_irc[dim] / 2))
            end = int(start + width_irc[dim])

            # assert error if value is not between 0 and ct_np.shape[dim]
            assert 0 <= value < self.ct_np.shape[dim], [self.series_uid, center_xyz, self.origin_xyz,
                                                        self.spacing_xyz, self.direction, irc_tuple, dim]

            # start slicing
            if start < 0:
                start = 0
                end = int(width_irc[dim])
            if end > self.ct_np.shape[dim]:
                end = self.ct_np.shape[dim]
                start = int(end - width_irc[dim])

            slice_list.append(slice(start, end))

        chunk = self.ct_np[tuple(slice_list)]
        chunk.clip(min=-1000, max=1000, out=chunk)

        return chunk, irc_tuple


@functools.lru_cache(1)
def getct(series_uid):
    return CtLoader(series_uid)
