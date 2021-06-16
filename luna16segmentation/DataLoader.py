from Preprocessing import Candidate, CtLoader
from torch.utils.data import Dataset
import torch as t


# this class is used for creating validation dataset
class Luna2dDataset(Dataset):
    def __init__(self, stride=0, isVal_bool=False, series_uid=None, fullCt_bool=False, slices_count=3):
        """
        :param stride: k-fold cross validation
        :param isVal_bool: training or validation
        :param series_uid: series_uid
        :param fullCt_bool: use full CT slides? (all the indexes)
        :param slices_count: a chunk is formed by slices_count slices (indexes)
        """
        self.slices_count = slices_count

        # get all the series_uid (raw ct) on the disk
        if series_uid:
            self.series_uid_list = [series_uid]
        else:
            self.series_uid_list = sorted(Candidate().CandidateInfo_dict.keys())  # sorted converts dict_keys to list

        # get training or validation series_uid on the disk
        if isVal_bool:
            assert stride > 0, 'stride should be larger than 0'
            self.series_uid_list = self.series_uid_list[::stride]
            assert self.series_uid_list
        else:
            del self.series_uid_list[::stride]
            assert self.series_uid_list

        # get samples from every series_uid
        self.sample_list = []  # (uid, index number) pair
        for one_uid in self.series_uid_list:
            max_index = CtLoader(one_uid).max_index
            nodule_slice_list = CtLoader(one_uid).Nodule_slice_list
            if fullCt_bool:
                self.sample_list += [(one_uid, index) for index in range(max_index)]
            else:
                self.sample_list += [(one_uid, index) for index in nodule_slice_list]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        one_uid, slice_index = self.sample_list[ndx % len(self.sample_list)]
        return self.getSlices(one_uid, slice_index)

    def getSlices(self, one_uid, slice_index):  # generate one chunk centered at slice_index
        ct_np = CtLoader(one_uid).ct_np
        nodule_mask = CtLoader(one_uid).Nodule_mask
        ct_slice_chunk = t.zeros((self.slices_count * 2 + 1, ct_np.shape[1], ct_np.shape[2]))
        start_slice = slice_index - self.slices_count
        end_slice = slice_index + self.slices_count + 1

        # we treat each slice as one channel
        for i, slice_index_one in enumerate(range(start_slice, end_slice)):
            slice_index_one = max(slice_index_one, 0)
            slice_index_one = min(slice_index_one, ct_np.shape[0] - 1)
            ct_slice_chunk[i] = t.from_numpy(ct_np[slice_index_one].clip(-1000, 1000)).to(dtype=t.float32)

        # the label should be a 2d array with bool at each pixel (the center slice)
        nodule_label = t.from_numpy(nodule_mask[slice_index]).unsqueeze(0)  # add one dimension to form one channel
        return ct_slice_chunk, nodule_label, one_uid, slice_index


# we need an additional class to balance and augment data for training
class TrainingLuna2dDataset(Luna2dDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
