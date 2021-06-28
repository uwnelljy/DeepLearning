from Preprocessing import getCandidateInfo, getct
from torch.utils.data import Dataset
import torch as t
import random


# based on series_uid rather than nodule
class LunaDataset(Dataset):
    def __init__(self, stride=0, isVal_bool=None, series_uid=None,
                 ratio=0, CandidateInfo_list=None):
        """
        :param stride: the proportion of training set and validation set
        :param isVal_bool: bool to identify training or validation set
        :param series_uid: series uid
        :param ratio: the proportion of negative and positive cases in training set, i.e. negative/positive
        """
        if CandidateInfo_list:
            self.CandidateInfo_list = CandidateInfo_list.copy()
        else:
            self.CandidateInfo_list = getCandidateInfo()[0].copy()

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(getCandidateInfo()[1].keys())  # all the series_id

        if isVal_bool:
            # test stride > 0, if false, then arise error
            assert stride > 0, 'stride should larger than 0.'
            self.series_list = self.series_list[::stride]
            # test whether self.CandidateInfo_list is None or not
            assert self.series_list
        else:
            assert stride > 0, 'stride should larger than 0'
            del self.series_list[::stride]
            assert self.series_list

        self.ratio = ratio

        series_set = set(self.series_list)
        # get selected candidate
        self.CandidateInfo_list = [can for can in self.CandidateInfo_list if can.series_uid in series_set]
        # get positive and negative cases
        self.positive_list = [item for item in self.CandidateInfo_list if item.isNodule_bool]
        self.negative_list = [item for item in self.CandidateInfo_list if not item.isNodule_bool]
        self.benign_list = [item for item in self.CandidateInfo_list if not item.isMal_bool]
        self.malignancy_list = [item for item in self.CandidateInfo_list if item.isMal_bool]

    def shuffleSamples(self):
        if self.ratio:
            random.shuffle(self.CandidateInfo_list)
            random.shuffle(self.positive_list)
            random.shuffle(self.negative_list)
            random.shuffle(self.benign_list)
            random.shuffle(self.malignancy_list)

    def __len__(self):
        # we define the size of the whole dataset in each epoch rather than the true size
        # speed the iteration in each epoch
        # because we have many repeated image (positive ones) after creating balanced data
        if self.ratio != 0:
            return 50000
        else:
            return len(self.CandidateInfo_list)

    def __getitem__(self, ndx):
        # create more balanced data
        if self.ratio:
            positive_ndx = ndx // (self.ratio + 1)
            if ndx % (self.ratio + 1) != 0:
                negative_ndx = ndx - positive_ndx - 1
                negative_ndx %= len(self.negative_list)  # in case that the index is larger than length
                candidateTuple = self.negative_list[negative_ndx]
            else:
                positive_ndx %= len(self.positive_list)
                candidateTuple = self.positive_list[positive_ndx]
        else:
            candidateTuple = self.CandidateInfo_list[ndx]

        return self.getsample(candidateTuple, candidateTuple.isNodule_bool)

    def getsample(self, candidateTuple, label_bool):
        width = [32, 48, 48]

        ct = getct(candidateTuple.series_uid)
        ct_chunk_t, irc = ct.getChunkCandidate(candidateTuple.center_xyz, width)
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


class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio:
            return 100000
        else:
            return len(self.malignancy_list) + len(self.benign_list)

    def __getitem__(self, ndx):
        if self.ratio:
            if ndx % 2 != 0:
                candidateTuple = self.malignancy_list[(ndx // 2) % len(self.malignancy_list)]
            elif ndx % 4 == 0:
                candidateTuple = self.benign_list[(ndx // 4) % len(self.benign_list)]
            else:
                candidateTuple = self.negative_list[(ndx // 4) % len(self.negative_list)]
        else:
            if ndx >= len(self.benign_list):
                candidateTuple = self.malignancy_list[ndx - len(self.benign_list)]
            else:
                candidateTuple = self.benign_list[ndx]

        return self.getsample(candidateTuple, candidateTuple.isMal_bool)

