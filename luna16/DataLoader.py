import Preprocessing
from torch.utils.data import Dataset
import torch as t


class LunaDataset(Dataset):
    def __init__(self, stride=0, isVal_bool=None, series_uid=None):
        self.CandidateInfo_list = Preprocessing.Candidate(True).GetCandidateInfo().copy()
        if series_uid:
            self.CandidateInfo_list = [x for x in self.CandidateInfo_list if x.series_uid==series_uid]
        if isVal_bool:
            # test stride > 0, if false, then arise error
            assert stride > 0, 'stride should larger than 0.'
            self.CandidateInfo_list = self.CandidateInfo_list[::stride]
            # test whether self.CandidateInfo_list is None or not
            assert self.CandidateInfo_list
        else:
            assert stride > 0, 'stride should larger than 0'
            del self.CandidateInfo_list[::stride]
            assert self.CandidateInfo_list

    def __len__(self):
        return len(self.CandidateInfo_list)

    def __getitem__(self, ndx):
        candidateTuple = self.CandidateInfo_list[ndx]
        # get needed information
        series_uid = candidateTuple.series_uid
        ct_raw = Preprocessing.CtLoader(series_uid)
        origin = ct_raw.ct_mhd.GetOrigin()
        spacing = ct_raw.ct_mhd.GetSpacing()
        direction = ct_raw.ct_mhd.GetDirection()
        # get center coordinates
        xyz = candidateTuple.candidateXYZ
        # get center index of nodules
        irc = ct_raw.xyz2irc(xyz, origin, spacing, direction)
        width = [32, 48, 48]
        # get ct chunk
        ct_chunk = ct_raw.get_raw_chunk(irc, width, ct_raw.ct_np)
        # convert to tensor
        ct_chunk_t = t.from_numpy(ct_chunk).to(t.float32)
        # add one additional dimension 'channel' because ct image only has one channel
        ct_chunk_t = ct_chunk_t.unsqueeze(0)

        # convert label to tensor with one hot-encoding
        label_t = t.tensor([not candidateTuple.isNodule_bool, candidateTuple.isNodule_bool], dtype=t.long)
        return ct_chunk_t, label_t, series_uid, xyz
