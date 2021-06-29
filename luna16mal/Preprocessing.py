import glob
import SimpleITK as sitk
import numpy as np
from Helper import xyz2irc
import functools


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
