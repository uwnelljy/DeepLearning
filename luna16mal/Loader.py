from torch.utils.data import Dataset
import torch as t
from Preprocessing import getct


# this class is used for creating dataset of one CT
class Luna2dSegDataset(Dataset):
    def __init__(self, series_uid=None, slices_count=3):
        """
        :param series_uid: series_uid
        :param slices_count: a chunk is formed by slices_count slices (indexes)
        """
        self.slices_count = slices_count
        assert series_uid, 'series_uid should not be None'
        self.series_uid = series_uid

        # get samples from series_uid
        self.oneloader = getct(self.series_uid)

    def __len__(self):
        return self.oneloader.max_index

    def __getitem__(self, ndx):
        ct_np = self.oneloader.ct_np
        nodule_mask = self.oneloader.Nodule_mask  # bool
        # the dimension of chunk is 7*512*512 because we treat 7 as channels.
        # no need to add one dimension
        ct_slice_chunk = t.zeros((self.slices_count * 2 + 1, ct_np.shape[1], ct_np.shape[2]))
        start_slice = ndx - self.slices_count
        end_slice = ndx + self.slices_count + 1

        # we treat each slice as one channel
        for i, slice_index_one in enumerate(range(start_slice, end_slice)):
            slice_index_one = max(slice_index_one, 0)
            slice_index_one = min(slice_index_one, ct_np.shape[0] - 1)
            ct_slice_chunk[i] = t.from_numpy(ct_np[slice_index_one].clip(-1000, 1000)).to(dtype=t.float32)

        # the label should be a 2d array with bool at each pixel (the center slice)
        nodule_label = t.from_numpy(nodule_mask[ndx]).unsqueeze(0)  # add one dimension to form one channel
        return ct_slice_chunk, nodule_label, self.series_uid, ndx


# based on series_uid rather than nodule
class LunaDataset(Dataset):
    def __init__(self, CandidateInfo_list=None):
        self.CandidateInfo_list = CandidateInfo_list.copy()

    def __len__(self):
        return len(self.CandidateInfo_list)

    def __getitem__(self, ndx):
        candidateTuple = self.CandidateInfo_list[ndx]
        return self.getsample(candidateTuple, candidateTuple.isNodule_bool)

    def getsample(self, candidateTuple, label_bool):
        width = [32, 48, 48]

        ct = getct(candidateTuple.series_uid)
        ct_chunk_t, irc = ct.getChunk(candidateTuple.center_xyz, width)
        ct_chunk_t = ct_chunk_t.to(dtype=t.float32)

        # convert label to tensor with one hot-encoding
        label_t = t.tensor([False, False], dtype=t.long)
        if label_bool:
            label_t[1] = True
            index_t = 1
        else:
            label_t[0] = True
            index_t = 0

        # return 5 elements
        return ct_chunk_t, label_t, index_t, candidateTuple.series_uid, t.tensor(irc)
