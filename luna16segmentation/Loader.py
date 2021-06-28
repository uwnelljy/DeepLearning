from Preprocessing import getCandidateInfo, getct
from torch.utils.data import Dataset
import torch as t
import random


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
            self.series_uid_list = sorted(getCandidateInfo()[1].keys())  # sorted converts dict_keys to list

        # get training or validation series_uid on the disk
        if isVal_bool:
            assert stride > 0, 'stride should be larger than 0'
            self.series_uid_list = self.series_uid_list[::stride]
            assert self.series_uid_list
        elif stride > 0:
            del self.series_uid_list[::stride]
            assert self.series_uid_list

        # get samples from every series_uid
        self.sample_list = []  # (uid, index number) pair
        for one_uid in self.series_uid_list:
            oneloader = getct(one_uid)
            max_index = oneloader.max_index
            nodule_slice_list = oneloader.Nodule_slice_list
            if fullCt_bool:
                self.sample_list += [(one_uid, index) for index in range(max_index)]
            else:
                self.sample_list += [(one_uid, index) for index in nodule_slice_list]

        series_uid_set = set(self.series_uid_list)
        self.selected_CandidateInfo_list = [item for item in getCandidateInfo()[0]
                                            if item.series_uid in series_uid_set]
        self.selected_NoduleInfo_list = [item for item in self.selected_CandidateInfo_list
                                         if item.isNodule_bool]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        one_uid, slice_index = self.sample_list[ndx % len(self.sample_list)]
        # each time we get one (uid, slice) pair as one sample
        return self.getSlices(one_uid, slice_index)

    def getSlices(self, one_uid, slice_index):  # generate one chunk centered at slice_index
        oneloader = getct(one_uid)
        ct_np = oneloader.ct_np
        nodule_mask = oneloader.Nodule_mask  # bool
        # the dimension of chunk is 7*512*512 because we treat 7 as channels.
        # no need to add one dimension
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

    def __len__(self):
        return 50

    def shuffle(self):
        # why shuffle here?
        random.shuffle(self.selected_CandidateInfo_list)
        random.shuffle(self.selected_NoduleInfo_list)

    def __getitem__(self, ndx):
        selected_NoduleInfo_tuple = self.selected_NoduleInfo_list[ndx % len(self.selected_NoduleInfo_list)]
        return self.getitem_training(selected_NoduleInfo_tuple)

    def getitem_training(self, NoduleInfo_tuple):
        center_xyz = NoduleInfo_tuple.center_xyz
        loader = getct(NoduleInfo_tuple.series_uid)
        # get a 7*96*96 smaller chunk, chunk_mask has the same dimension as chunk
        chunk, chunk_mask, irc_tuple = loader.getChunk(center_xyz, (7, 96, 96))
        # get the center slice of chunk_mask to conform with the validation set
        chunk_mask = chunk_mask[3:4]  # 3:4 generate an additional dimension, while 3 doesn't has that dimension

        # we only need 7*64*64, so we create a offset
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        chunk_t = t.from_numpy(
            chunk[:, row_offset: row_offset + 64, col_offset: col_offset + 64]).to(t.float32)
        chunk_mask_t = t.from_numpy(
            chunk_mask[:, row_offset: row_offset + 64, col_offset: col_offset + 64])  # bool
        slice_ndx = irc_tuple.index

        return chunk_t, chunk_mask_t, NoduleInfo_tuple.series_uid, slice_ndx

