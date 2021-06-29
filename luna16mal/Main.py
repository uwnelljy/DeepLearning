import torch as t
import numpy as np
from torch.utils.data import DataLoader
from Loader import Luna2dSegDataset
from scipy.ndimage import measurements
from Helper import irc2xyz, CandidateInfo_tuple, classificationTuple


class MalOrNot:
    def __init__(self):
        self.use_cuda = t.cuda.is_available()
        self.device = t.device('cuda') if self.use_cuda else t.device('cpu')

        # Load trained models
        self.segmodel, self.classmodel, self.malmodel = self.loadmodels()

    def initsegloader(self, series_uid):
        dataset = Luna2dSegDataset(series_uid=series_uid, slices_count=3)
        seg_loader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                pin_memory=self.use_cuda)
        return seg_loader

    def initclaloader(self, candidate_info_list):
        pass

    def loadmodels(self):
        pass

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
        for series_uid in series_uid_list:
            ct = getct(series_uid)  # 148*512*512
            segmask_output_np = self.segmentation(ct.ct_np, series_uid)  # 148*512*512
            nodule_list = self.groupsegmentation(series_uid, segmask_output_np, ct.ct_np)  # output should be a list of: (id, irc_center...)
            classification_list = self.classification(nodule_list)
            for classification_tuple in classification_list:
                prob_nodule, prob_mal, center_xyz, center_irc = classification_tuple
