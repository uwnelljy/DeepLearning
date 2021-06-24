import csv
import functools
from Helper import CandidateInfo_tuple, xyz2irc
import glob
import os
import SimpleITK as sitk
import numpy as np


@functools.lru_cache(1)
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
        # get ct data
        ct_path = glob.glob('./data/subset*/{}.mhd'.format(series_uid))[0]
        ct_mhd = sitk.ReadImage(ct_path)
        self.ct_np = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.series_uid = series_uid
        self.origin_xyz = ct_mhd.GetOrigin()
        self.spacing_xyz = ct_mhd.GetSpacing()
        self.direction = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        # get nodule information
        candidate_with_one_uid_list = getCandidateInfo()[1][self.series_uid]
        self.NoduleInfo_list = [  # this is a list of tuple. [tuple1, tuple2, ...]
            info for info in candidate_with_one_uid_list
            if info.isNodule_bool
        ]
        self.Nodule_mask = self.buildAnnotationMask()  # bool
        # get total index of this ct
        self.max_index = int(self.ct_np.shape[0])
        # get slice index with Nodule
        self.Nodule_slice_list = self.Nodule_mask.sum(axis=(1, 2)).nonzero()[0].tolist()

    def buildAnnotationMask(self, threshold_hu=-700):
        """
        :param threshold_hu:
        :return: a bool mask
        """
        mask_box = np.zeros_like(self.ct_np, dtype=bool)

        # candidate tuple in nodule list one by one
        for info in self.NoduleInfo_list:
            # convert center xyz to irc
            irc_tuple = xyz2irc(info.center_xyz, self.origin_xyz, self.spacing_xyz, self.direction)
            ci = irc_tuple.index
            cr = irc_tuple.row
            cc = irc_tuple.col

            # get index mask
            index_radius = 2
            try:
                while self.ct_np[ci + index_radius, cr, cc] > threshold_hu and \
                        self.ct_np[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            # get row mask
            row_radius = 2
            try:
                while self.ct_np[ci, cr + row_radius, cc] > threshold_hu and \
                        self.ct_np[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            # get row mask
            col_radius = 2
            try:
                while self.ct_np[ci, cr, cc + col_radius] > threshold_hu and \
                        self.ct_np[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            mask_box[
                ci - index_radius: ci + index_radius + 1,
                cr - row_radius: cr + row_radius + 1,
                cc - col_radius: cc + col_radius + 1] = True

        mask = mask_box & (self.ct_np > threshold_hu)  # same dimension as ct_np
        # mask is bool
        return mask

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
        chunk_mask = self.Nodule_mask[tuple(slice_list)]  # same dimension as chunk
        # chunk_mask is bool

        return chunk, chunk_mask, irc_tuple


@functools.lru_cache(1)
def getct(series_uid):
    return CtLoader(series_uid)
