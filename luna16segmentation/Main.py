import sys
import argparse
import Model
import torch.nn as nn
from torch import optim
from Loader import *
from torch.utils.data import DataLoader
from Helper import diceloss
import logging
import warnings
import time
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


logging.basicConfig(filename='luna16seg.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    filemode='a',  # a means write after the former text, w means overwrite on the former text
                    level=logging.DEBUG,  # logging all the information higher than debug (info, warning etc.)
                    datefmt='%Y/%m/%d %H:%M:%S')  # the format of time stamp for logging
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')


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
                            help='Create augmented data using flip, rotate, offset, scale and noise',
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
        parser.add_argument('--tb-prefix',
                            default='luna16seg',
                            help='Prefix for tensorboard run.')

        self.args = parser.parse_args(sys_argv)
        self.training_writer = None
        self.validation_writer = None
        self.time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

        # augmentation parameters
        self.augmentation = {}
        if self.args.augmentation or self.args.augmentation_flip:
            self.augmentation['flip'] = True
        if self.args.augmentation or self.args.augmentation_offset:
            self.augmentation['offset'] = 0.03
        if self.args.augmentation or self.args.augmentation_scale:
            self.augmentation['scale'] = 0.2
        if self.args.augmentation or self.args.augmentation_rotate:
            self.augmentation['rotate'] = True
        if self.args.augmentation or self.args.augmentation_noise:
            self.augmentation['noise'] = 25.0

        self.use_cuda = t.cuda.is_available()
        self.device = t.device('cuda') if self.use_cuda else t.device('cpu')

        self.checkpoint = None
        self.totalTrainingCount = 0
        self.loss = 100000
        self.epoch_start = 1
        self.model, self.augModel, self.optimizer = self.initModel_Opt()

    def initModel_Opt(self):
        segModel = Model.UNetSeg(in_channels=7, n_classes=1, depth=3, wf=4,
                                 padding=True, batch_norm=True, up_mode='upconv')
        augModel = Model.Augmentation(**self.augmentation)
        if self.use_cuda:
            logging.info('Use CUDA, {} device(s)'.format(t.cuda.device_count()))
            if t.cuda.device_count() > 1:
                segModel = nn.DataParallel(segModel)  # parallel model computing
                augModel = nn.DataParallel(augModel)
            segModel = segModel.to(self.device)
            augModel = augModel.to(self.device)
        optimizer = optim.Adam(segModel.parameters())

        if self.checkpoint:  # if we have saved model
            logging.info('Loading existing segmentation model')
            segmodel_checkpoint = t.load(self.checkpoint)
            segModel.load_state_dict(segmodel_checkpoint['model_state'])
            optimizer.load_state_dict(segmodel_checkpoint['optimizer_state'])
            self.loss = segmodel_checkpoint['best_loss']
            self.epoch_start = segmodel_checkpoint['epoch'] + 1
            self.totalTrainingCount = segmodel_checkpoint['totalTrainingCount']

        return segModel, augModel, optimizer

    def initTrainDl(self):
        trainingset = TrainingLuna2dDataset(stride=200, isVal_bool=False, slices_count=3)
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()
        trainDl = DataLoader(dataset=trainingset, batch_size=batch_size,
                             num_workers=self.args.num_workers,
                             pin_memory=self.use_cuda)
        return trainDl

    def initValDl(self):
        validationset = Luna2dDataset(stride=50, isVal_bool=True, slices_count=3)
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()
        valDl = DataLoader(dataset=validationset, batch_size=batch_size,
                           num_workers=self.args.num_workers,
                           pin_memory=self.use_cuda)
        return valDl

    def initTensorboardWriters(self):
        if self.training_writer is None:
            log_dir = os.path.join('runs', self.args.tb_prefix, self.time)
            self.training_writer = SummaryWriter(log_dir=log_dir + '_train_seg_')
            self.validation_writer = SummaryWriter(log_dir=log_dir + '_val_seg_')

    def training(self, traindl):
        training_metrics = t.zeros(4, len(traindl.dataset), device=self.device)
        self.model.train()
        # why shuffle here?
        traindl.dataset.shuffle()

        for ndx, oneloader in enumerate(traindl):
            self.optimizer.zero_grad()
            loss = self.computeLoss(ndx, oneloader, traindl.batch_size, 0.5, training_metrics)
            loss.backward()
            self.optimizer.step()
        self.totalTrainingCount += len(traindl.dataset)

        # why convert to cpu?
        return training_metrics.to('cpu')

    def validation(self, valDl):
        with t.no_grad():
            val_metrics = t.zeros(4, len(valDl.dataset))
            self.model.eval()
            for ndx, oneloader in enumerate(valDl):
                self.computeLoss(ndx, oneloader, valDl.batch_size, 0.5, val_metrics)
        return val_metrics.to('cpu')

    def computeLoss(self, ndx, oneloader, batch_size, threshold, training_metrics):
        # get data from dataloader
        chunk_t, chunk_mask_t, series_uid, slice_ndx = oneloader

        # transform to gpu
        chunk_gpu = chunk_t.to(self.device)
        chunk_mask_gpu = chunk_mask_t.to(self.device)  # bool

        # augmenting chunk when training model
        # self.model.training returns bool
        if self.model.training and self.augmentation:
            chunk_gpu, chunk_mask_gpu = self.augModel(chunk_gpu, chunk_mask_gpu)

        # feed into model
        prediction_gpu = self.model(chunk_gpu)  # with shape batch_size*1channel*64*64

        # compute loss
        dice_loss_gpu = diceloss(chunk_mask_gpu, prediction_gpu)
        false_negative_loss_gpu = diceloss(chunk_mask_gpu, chunk_mask_gpu * prediction_gpu)

        # log into training_metrics
        start = ndx * batch_size
        end = start + chunk_t.size(0)
        with t.no_grad():  # we don't use detach because we need to output them
            prediction_bool_gpu = (prediction_gpu[:, 0:1] > threshold).to(t.float32)
            true_positive = (prediction_bool_gpu * chunk_mask_gpu).sum(dim=[1, 2, 3])
            false_negative = ((1-prediction_bool_gpu) * chunk_mask_gpu).sum(dim=[1, 2, 3])
            # ~ is only implemented on integer and bool, and ~1 = -2, ~0 = -1
            false_positive = (prediction_bool_gpu * (~chunk_mask_gpu)).sum(dim=[1, 2, 3])
            training_metrics[0, start: end] = dice_loss_gpu
            training_metrics[1, start: end] = true_positive
            training_metrics[2, start: end] = false_negative
            training_metrics[3, start: end] = false_positive

        # false negative is far more important than total loss, so we multiply it by 8
        return dice_loss_gpu.mean() + false_negative_loss_gpu.mean() * 8

    def main(self):
        logging.info('Starting {}, {}'.format(type(self).__name__, self.args))
        # load data
        trainDl = self.initTrainDl()
        valDl = self.initValDl()

        # start iteration
        for epoch in range(self.epoch_start, self.args.epochs + 1):
            # log out information
            logging.info('Epoch {} of {}, {}/{} batches of size {}*{}'.format(
                epoch, self.args.epochs, len(trainDl), len(valDl),
                self.args.batch_size,
                (t.cuda.device_count() if self.use_cuda else 1)
            ))

            # start training
            train_metrics = self.training(trainDl)
            # log out training result
            self.logMetrics(epoch, 'training', train_metrics)
            self.logImages(epoch, 'training', trainDl)
            logging.info('Training ends')

            # start validation
            val_metrics = self.validation(valDl)
            # save model
            if val_metrics[0].mean() < self.loss:
                self.saveModel(epoch, val_metrics[0].mean())
            self.logMetrics(epoch, 'validation', val_metrics)
            self.logImages(epoch, 'validation', valDl)
            logging.info('Validation ends')
        self.training_writer.close()
        self.validation_writer.close()

    def logImages(self, epoch, mode, dataloader):
        self.model.eval()
        images = sorted(dataloader.dataset.series_uid_list)[:12]
        for ndx, series_uid in enumerate(images):
            ct = getct(series_uid)
            for slice_ndx in range(6):
                # six equidistant slices
                ct_ndx = slice_ndx * (ct.ct_np.shape[0] - 1) // 5
                ct_full_slice, nodule_label, series_uid, ct_ndx = dataloader.dataset.getSlices(series_uid, ct_ndx)
                chunk_gpu = ct_full_slice.to(self.device).unsqueeze(0)  # add one dimension for batch_size
                chunk_mask_gpu = nodule_label.to(self.device).unsqueeze(0)
                prediction_gpu = self.model(chunk_gpu)[0]  # no batch_size dimension
                prediction_cpu = prediction_gpu.to('cpu').detach().numpy()[0] > 0.5  # no channel dimension
                chunk_mask_cpu = chunk_mask_gpu.to('cpu').numpy()[0][0]  # bool

                ct_full_slice[:-1, :, :] /= 2000
                ct_full_slice[:-1, :, :] += 0.5  # don't understand

                # get one slice. I don't understand.
                ct_one_slice = ct_full_slice[dataloader.dataset.slices_count].numpy()
                image_np = np.zeros((512, 512, 3), dtype=np.float32)
                image_np[:, :, :] = ct_one_slice.reshape((512, 512, 1))
                # number + bool = number + 1 or 0
                image_np[:, :, 0] += prediction_cpu & (~chunk_mask_cpu)  # false positive to red channel
                image_np[:, :, 0] += (~prediction_cpu) & chunk_mask_cpu  # false negative to red
                image_np[:, :, 1] += ((~prediction_cpu) & chunk_mask_cpu) * 0.5  # half false negative to green, so it would be orange.
                image_np[:, :, 1] += prediction_cpu & chunk_mask_cpu  # true positive would be pure green

                image_np *= 0.5  # why?
                image_np.clip(0, 1, image_np)  # why???

                writer = getattr(self, mode + '_writer')
                writer.add_image(
                    f'{mode}/{series_uid}_prediction_{slice_ndx}',  # file name
                    image_np,  # image
                    self.totalTrainingCount,  # what is this?
                    dataformats='HWC'  # the order of axes in our image, c is channel
                )

                writer.flush()

    def logMetrics(self, epoch, mode, metrics):
        """
        :param epoch: current epoch
        :param mode: 'training' or 'validation'
        :param metrics: TrainMetrics or ValMetrics.
         0 for dice loss, 1 for true positive, 2 for false negative, 3 for false positive
        :param threshold: classification threshold, default is 0.5, because we only have probability in metrics
        :return:
        """
        metrics_np = metrics.detach().numpy()
        sum_np = metrics_np.sum(axis=1)  # sum the values in each row

        precision = sum_np[1] / ((sum_np[1] + sum_np[3]) or 1)
        recall = sum_np[1] / ((sum_np[1] + sum_np[2]) or 1)

        # store them into dictionary
        metrics_dict = {'loss_all': metrics_np[0].mean(),
                        'precision': precision,
                        'recall': recall,
                        'f1': 2 * precision * recall / ((precision + recall) or 1),
                        'tp': sum_np[1],
                        'fn': sum_np[2],
                        'fp': sum_np[3]}

        # log out
        # loss is with a decimal part of length 4
        logging.info(('Epoch {}, {}, {loss_all:.4f} loss, ' +
                      'Precision is {precision:.4f}, ' +
                      'Recall is {recall:.4f}, ' +
                      'F1 score is {f1:.4f}; ' +
                      'True positive is {tp}, ' +
                      'False Negative is {fn}, ' +
                      'False Positive is {fp}.').format(epoch, mode, **metrics_dict))

        # write into tensorboard
        self.initTensorboardWriters()
        # return the attributes (mode + _writer) of one object (self, namely SegmentationTrainingApp)
        writer = getattr(self, mode + '_writer')

        for key, value in metrics_dict.items():
            # 1 element: name, 2 element: y, 3 element: x
            writer.add_scalar('seg_' + key, value, self.totalTrainingCount)

        writer.flush()  # ensure all pending events have been written to disk

    def saveModel(self, epoch, bestloss):
        # /gscratch/stf/nelljy/savedmodel
        pathModel = './savedmodel/{}_{}_{}.state'.format(
            'segmentation',
            self.time,
            self.totalTrainingCount)
        model = self.model
        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'time': self.time,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'best_loss': bestloss,
            'totalTrainingCount': self.totalTrainingCount
        }

        t.save(state, pathModel)


if __name__ == '__main__':
    SegmentationTrainingApp().main()
