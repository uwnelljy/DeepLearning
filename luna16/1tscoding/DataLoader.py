from Preprocessing import GetCandidateInfo, getct
from torch.utils.data import Dataset
import torch as t
import random


class LunaDataset(Dataset):
    def __init__(self, stride=0, isVal_bool=None, series_uid=None,
                 ratio=0, augmentation=None):
        """
        :param stride: the proportion of training set and validation set
        :param isVal_bool: bool to identify training or validation set
        :param series_uid: series uid
        :param ratio: the proportion of negative and positive cases in training set, i.e. negative/positive
        """
        self.CandidateInfo_list = GetCandidateInfo().copy()

        if series_uid:
            self.CandidateInfo_list = [x for x in self.CandidateInfo_list if x.series_uid == series_uid]
        elif isVal_bool:
            # test stride > 0, if false, then arise error
            assert stride > 0, 'stride should larger than 0.'
            self.CandidateInfo_list = self.CandidateInfo_list[::stride]
            # test whether self.CandidateInfo_list is None or not
            assert self.CandidateInfo_list
        else:
            assert stride > 0, 'stride should larger than 0'
            del self.CandidateInfo_list[::stride]
            assert self.CandidateInfo_list

        self.ratio = ratio
        self.augmentation = augmentation
        # get positive and negative cases
        self.positive_list = [item for item in self.CandidateInfo_list if item.isNodule_bool]
        self.negative_list = [item for item in self.CandidateInfo_list if not item.isNodule_bool]

    def shuffleSamples(self):
        if self.ratio:
            random.shuffle(self.positive_list)
            random.shuffle(self.negative_list)

    def __len__(self):
        # we define the size of the whole dataset in each epoch rather than the true size
        # speed the iteration in each epoch
        # because we have many repeated image (positive ones) after creating balanced data
        if self.ratio != 0:
            return 200000
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

        # get needed information
        series_uid = candidateTuple.series_uid
        # get center coordinates
        xyz = candidateTuple.candidateXYZ
        width = [32, 48, 48]
        raw_ct = getct(series_uid)

        # get ct chunk with 1 channel
        if self.augmentation:
            ct_chunk_t, irc = raw_ct.getAugmentedCandidate(self.augmentation, xyz, width)
        else:
            ct_chunk_t, irc = raw_ct.getChunkCandidate(xyz, width)

        # convert label to tensor with one hot-encoding
        label_t = t.tensor([not candidateTuple.isNodule_bool, candidateTuple.isNodule_bool], dtype=t.long)
        return ct_chunk_t, label_t, series_uid, t.tensor(irc)
