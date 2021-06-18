import sys
import argparse
import Model
import torch.nn as nn
from torch import optim
from Loader import *
from torch.utils.data import DataLoader
from Helper import diceloss


class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

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
        parser.add_argument('--augmentation',
                            help='Creare augmented data using flip, rotate, offset, scale and noise',
                            action='store_true',
                            default=False)
        parser.add_argument('--augmentation-flip',
                            help='Randomly flip the data',
                            action='store_true',
                            default=False)
        parser.add_argument('--augmentation-offset',
                            help='Randomly add offset to each dimension',
                            action='store_true',
                            default=False)
        parser.add_argument('--augmentation-scale',
                            help='Randomly scale the data in each dimension',
                            action='store_true',
                            default=False)
        parser.add_argument('--augmentation-rotate',
                            help='Randomly rotate the data in each dimension',
                            action='store_true',
                            default=False)
        parser.add_argument('--augmentation-noise',
                            help='Randomly add noise to raw data',
                            action='store_true',
                            default=False)

        self.args = parser.parse_args(sys_argv)
        self.use_cuda = t.cuda.is_available()
        self.device = t.device('cuda') if self.use_cuda else t.device('cpu')
        self.model, self.optimizer = self.initModel_Opt()

    def initModel_Opt(self):
        segModel = Model.UNetSeg(in_channels=7, n_classes=1, depth=3, wf=4,
                                 padding=True, batch_norm=True, up_mode='upconv')
        if self.use_cuda:
            if t.cuda.device_count() > 1:
                segModel = nn.DataParallel(segModel)  # parallel model computing
            segModel = segModel.to(self.device)

        optimizer = optim.Adam(segModel.parameters())

        return segModel, optimizer

    def initTrainDl(self):
        print('initial training dataloader')
        trainingset = TrainingLuna2dDataset(stride=200, isVal_bool=False, slices_count=3)
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()
        trainDl = DataLoader(dataset=trainingset, batch_size=batch_size,
                             num_workers=self.args.num_workers,
                             pin_memory=self.use_cuda)
        print('training dataloader finished')
        return trainDl

    def initValDl(self):
        print('initial validation dataloader')
        validationset = Luna2dDataset(stride=200, isVal_bool=True, slices_count=3)
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()
        valDl = DataLoader(dataset=validationset, batch_size=batch_size,
                           num_workers=self.args.num_workers,
                           pin_memory=self.use_cuda)
        print('validation dataloader finished')
        return valDl

    def training(self, epoch, traindl):
        print('training starts')
        self.model.train()
        for ndx, oneloader in enumerate(traindl):
            if ndx == 0:
                print('ndx is {}, starts'.format(ndx))
                self.optimizer.zero_grad()
                loss = self.computeLoss(ndx, oneloader, traindl.batch_size)
                loss.backward()
                self.optimizer.step()
                print('ndx is {}, ends'.format(ndx))
            else:
                break

    def validation(self):
        pass

    def computeLoss(self, ndx, oneloader, batch_size):
        # get data from dataloader
        chunk_t, chunk_mask_t, series_uid, slice_ndx = oneloader

        # transform to gpu
        chunk_gpu = chunk_t.to(self.device)
        chunk_mask_gpu = chunk_mask_t.to(self.device)
        # feed into model
        prediction_gpu = self.model(chunk_gpu)

        # compute loss
        dice_loss_gpu = diceloss(chunk_mask_gpu, prediction_gpu)
        false_negative_loss_gpu = diceloss(chunk_mask_gpu, chunk_mask_gpu * prediction_gpu)

        return dice_loss_gpu.mean() + false_negative_loss_gpu.mean() * 8
        # false negative is far more important than total loss, so we multiply it by 8

    def logMetrics(self):
        pass

    def main(self):
        # load data
        trainDl = self.initTrainDl()
        # valDl = self.initValDl()

        # start iteration
        for epoch in range(1, 2):
            print('epoch training starts')
            train_metrics = self.training(epoch, trainDl)
            print('epoch training ends')
            # val_metrics = self.validation(epoch, valDl)

    def saveModel(self):
        pass


if __name__ == '__main__':
    SegmentationTrainingApp().main()
