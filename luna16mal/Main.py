import sys
import argparse
import torch as t
import numpy as np
from torch.utils.data import DataLoader
from Loader import Luna2dSegDataset, LunaDataset
from scipy.ndimage import measurements
from Helper import irc2xyz, CandidateInfo_tuple, classificationTuple
from Model import UNetSeg, LunaModel
from Preprocessing import getCandidateInfo, getct


def match_and_score(detections_list, true_list, threshold_nodule=0.5, threshold_mal=0.5):
    true_nodules_list = [item for item in true_list if item.isNodule_bool]
    true_diameter_list = [item.diameter_mm for item in true_nodules_list]
    true_center_xyz_list = [item.center_xyz for item in true_nodules_list]
    detected_center_xyz_list = [item.center_xyz for item in detections_list]
    detected_prob_nodule = [item.prob_nodule for item in detections_list]
    detected_prob_mal = [item.prob_mal for item in detections_list]
    


class MalOrNot:
    def __init__(self, sys_args=None):
        if sys_args is None:
            sys_args = sys.argv[1:]
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=0, type=int)
        parser.add_argument('--batch-size',
                            help='Number of samples of each batch',
                            default=32, type=int)
        parser.add_argument('--epochs',
                            help='Maximun iterations for training the model',
                            default=1, type=int)
        self.args = parser.parse_args(sys_args)

        self.use_cuda = t.cuda.is_available()
        self.device = t.device('cuda') if self.use_cuda else t.device('cpu')

        # Load trained models
        self.savedsegmodelpath = None
        self.savedclamodelpath = None
        self.savedmalmodelpath = None
        self.segmodel, self.classmodel, self.malmodel = self.loadmodels()

    def initsegloader(self, series_uid):
        dataset = Luna2dSegDataset(series_uid=series_uid, slices_count=3)
        seg_loader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                pin_memory=self.use_cuda)
        return seg_loader

    def initclaloader(self, candidate_info_list):
        dataset = LunaDataset(candidate_info_list)
        cla_loader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                pin_memory=self.use_cuda)
        return cla_loader

    def loadmodels(self):
        segmodel = UNetSeg(in_channels=7, n_classes=1,
                           depth=5, wf=6, padding=True,
                           batch_norm=True, up_mode='upconv')
        clamodel = LunaModel()

        assert self.savedsegmodelpath, 'a path of segmentation model should be given'
        assert self.savedclamodelpath, 'a path of classification model should be given'

        segmodeldict = t.load(self.savedsegmodelpath)
        segmodel.load_state_dict(segmodeldict['model_state'])
        segmodel.eval()

        clamodeldict = t.load(self.savedclamodelpath)
        clamodel.load_state_dict(clamodeldict['model_state'])
        clamodel.eval()

        if self.use_cuda:
            segmodel.to(self.device)
            clamodel.to(self.device)

        if self.savedmalmodelpath:
            malmodel = LunaModel()
            malmodeldict = t.load(self.savedmalmodelpath)
            malmodel.load_state_dict(malmodeldict['model_state'])
            malmodel.eval()
            if self.use_cuda:
                malmodel.to(self.device)
        else:
            malmodel = None
        return segmodel, clamodel, malmodel

    def segmentation(self, ct_np, series_uid):
        # we need to get one DataLoader which is fed into the segmentation model for consistency and convenience
        with t.no_grad():
            seg_loader = self.initsegloader(series_uid)  # multiple batches
            seg_output_np = np.zeros_like(ct_np)
            for input_t_cpu_batch, _, _, slice_ndx_batch in seg_loader:

                # input_t_cpu_batch is batch_size*7*512*512
                input_t_gpu_batch = input_t_cpu_batch.to(self.device)
                # predicted_t_gpu_batch is batch_size*1*512*512
                predicted_t_gpu_batch = self.segmodel(input_t_gpu_batch)

                for i, slice_ndx in enumerate(slice_ndx_batch):
                    seg_output_np[slice_ndx] = predicted_t_gpu_batch[i].to('cpu').numpy()

            segmask_output_np = seg_output_np > 0.5
        return segmask_output_np

    def groupsegmentation(self, series_uid, segmask_output_np, ct):
        # function 'label' will return the number of nodules (true pixels in mask) and label
        candidate_blob, candidate_count = measurements.label(segmask_output_np)
        center_irc_list = measurements.center_of_mass(  # this function will return [(i, r, c), (i, r, c)...]
            ct.ct_np.clip(-1000, 1000) + 1001,  # make sure that all pixel values are positive
            labels=candidate_blob,
            index=np.arange(1, candidate_count + 1)  # the number of labels that it would consider
        )

        candidate_info_list = []
        # convert irc to xyz
        for center_irc in center_irc_list:
            center_xyz = irc2xyz(center_irc, ct.origin_xyz, ct.spacing_xyz, ct.direction)
            candidate_info_list.append(
                CandidateInfo_tuple(False, False, False, 0.0, series_uid, center_xyz)
            )
        return candidate_info_list  # contains the center coordinate information for each predicted nodule

    def classification(self, candidate_info_list, ct):
        cla_loader = self.initclaloader(candidate_info_list)
        classification_list = []
        for batch_ndx, batch_tuple in enumerate(cla_loader):
            chunk_t, _, _, series_id_list, center_irc_list = batch_tuple
            chunk_gpu = chunk_t.to(self.device)
            with t.no_grad():
                logits_gpu, probability_gpu = self.classmodel(chunk_gpu)
                if self.malmodel:
                    logits_mal_gpu, probability_mal_gpu = self.malmodel(chunk_gpu)
                else:
                    probability_mal_gpu = t.zeros_like(probability_gpu)
            zip_iter = zip(center_irc_list, probability_gpu[:1].tolist(), probability_mal_gpu[:1].tolist())
            for center_irc, prob_nodule, prob_mal in zip_iter:
                center_xyz = irc2xyz(center_irc, ct.origin_xyz, ct.spacing_xyz, ct.direction)
                classification_tuple = classificationTuple(prob_nodule, prob_mal, center_xyz, center_irc)
                classification_list.append(classification_tuple)
        return classification_list

    # The big picture
    def main(self):
        path = './data/subset9'
        series_uid_set, candidateinfo_list, candidateinfo_dict = getCandidateInfo(path)
        for series_uid in series_uid_set:
            ct = getct(series_uid)  # 148*512*512
            trueCandidateinfo_tuple_list = candidateinfo_dict[series_uid]
            segmask_output_np = self.segmentation(ct.ct_np, series_uid)  # 148*512*512
            nodule_list = self.groupsegmentation(series_uid, segmask_output_np, ct.ct_np)  # output should be a list of: (id, irc_center...)
            classification_list = self.classification(nodule_list, ct)
            for classification_tuple in classification_list:
                prob_nodule, prob_mal, center_xyz, center_irc = classification_tuple

