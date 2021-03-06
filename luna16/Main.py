import sys
import time
import Net
import torch as t
import torch.nn as nn
import numpy as np
import logging
import torch.optim as optim
import Dataloader
from torch.utils.data import DataLoader
import argparse
import warnings
import os
from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(filename='luna16cla.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    filemode='a',  # a means write after the former text, w means overwrite on the former text
                    level=logging.DEBUG,  # logging all the information higher than debug (info, warning etc.)
                    datefmt='%Y/%m/%d %H:%M:%S')  # the format of time stamp for logging
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        # for more information about argparse
        # see sys_argv_test.py
        if sys_argv is None:
            # sys.argv[1:] is all the variables we input through command line
            # sys.argv[0] is the name of the script
            sys_argv = sys.argv[1:]  # if we want to use command line to input parameters

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8, type=int)
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
        parser.add_argument('--dataset',
                            help='Which dataset to feed the model',
                            action='store',
                            default='LunaDataset')
        parser.add_argument('--model',
                            help='Which model to train',
                            action='store',
                            default='LunaModel')
        parser.add_argument('--malignant',
                            help='Classify nodules as benign or malignant',
                            action='store_true',
                            default=False)
        parser.add_argument('--tb-prefix',
                            default='luna16cla',
                            help='Prefix for tensorboard run.')

        self.args = parser.parse_args(sys_argv)
        self.training_writer = None
        self.validation_writer = None
        self.time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())

        # we use argument variables from command line
        self.args = parser.parse_args(sys_argv)
        self.augmentation = {}
        if self.args.augmentation or self.args.augmentation_flip:
            self.augmentation['flip'] = True
        if self.args.augmentation or self.args.augmentation_offset:
            self.augmentation['offset'] = 0.1
        if self.args.augmentation or self.args.augmentation_scale:
            self.augmentation['scale'] = 0.2
        if self.args.augmentation or self.args.augmentation_rotate:
            self.augmentation['rotate'] = True
        if self.args.augmentation or self.args.augmentation_noise:
            self.augmentation['noise'] = 25.0

        self.checkpoint = None
        self.loss = 10000000
        self.epoch_start = 1
        self.use_cuda = t.cuda.is_available()
        self.device = t.device('cuda') if self.use_cuda else t.device('cpu')
        self.totalTrainingSamples_count = 0
        self.model, self.optimizer, self.augmodel = self.initmodel_optimizer()

    def initmodel_optimizer(self):
        # initialize model and optimizer
        model_define = getattr(Net, self.args.model)
        model = model_define()
        augmodel = Net.Augmentation(**self.augmentation)

        if self.use_cuda:
            logging.info('Using CUDA, {} devices.'.format(t.cuda.device_count()))  # the number of GPUs
            if t.cuda.device_count() > 1:
                model = nn.DataParallel(model)  # parallel model computing
            model = model.to(self.device)

        # About momentum:
        # gradient descent: w_{k+1} = w_k - alpha*f'(w_k)
        # with momentum: z_{k+1} = beta*z_k + f'(w_k),
        #                w_{k+1} = w_k - alpha*z_{k+1}
        # control the updating. High momentum, smooth gradient descent.
        # initializing optimizer should be after moving the model to gpu
        optimizer = optim.Adam(model.parameters(), lr=3e-4)

        # loading saved model and optimizer
        if self.checkpoint:
            logging.info('Loading existing model: {}'.format(self.checkpoint))
            model_checkpoint = t.load(self.checkpoint)
            model.load_state_dict(model_checkpoint['model_state'])
            optimizer.load_state_dict(model_checkpoint['optimizer_state'])
            self.loss = model_checkpoint['best_loss']
            self.epoch_start = model_checkpoint['epoch'] + 1
            self.totalTrainingSamples_count = model_checkpoint['totalTrainingSamples_count']

        return model, optimizer, augmodel

    def initTrainloader(self):
        # get training dataset, classification of nodules or malignant nodules
        dataset_define = getattr(Dataloader, self.args.dataset)
        training_dataset = dataset_define(stride=10,
                                          isVal_bool=False,
                                          ratio=1)
        logging.info(('Dataset: {}, {} positive samples, {} negative samples, ' +
                     '{} benign samples, {} malignant samples').format(
            self.args.dataset,
            len(training_dataset.positive_list),
            len(training_dataset.negative_list),
            len(training_dataset.benign_list),
            len(training_dataset.malignancy_list)))
        # read batch_size from command line
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()  # why????????

        trainloader = DataLoader(training_dataset, batch_size=batch_size,
                                 num_workers=self.args.num_workers, pin_memory=self.use_cuda)
        # the length of trainloader is the number of batches
        # len(trainloader.dataset): the number of images in all the batches
        return trainloader

    def initValidationloader(self):
        dataset_define = getattr(Dataloader, self.args.dataset)
        validation_dataset = dataset_define(stride=10, isVal_bool=True)
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()
        validationloader = DataLoader(validation_dataset, batch_size=batch_size,
                                      num_workers=self.args.num_workers, pin_memory=self.use_cuda)
        return validationloader

    def initTensorboardWriters(self):
        if self.training_writer is None:
            log_dir = os.path.join('runs', self.args.tb_prefix, self.time)
            self.training_writer = SummaryWriter(log_dir=log_dir + '_train_seg_')
            self.validation_writer = SummaryWriter(log_dir=log_dir + '_val_seg_')

    def training(self, trainloader):
        # set model to training mode, but the model is still self.model
        self.model.train()
        trainloader.dataset.shuffleSamples()  # shuffle positive and negative lists

        # create a matrix to store prediction result
        # 0 for true labels, 1 for probability result, 2 for loss
        trainMetrics_gpu = t.zeros(3, len(trainloader.dataset), device=self.device)
        for ndx, oneloader in enumerate(trainloader):
            # logging.info('Starting the {}th batch in training loader'.format(ndx))
            # update coefficients
            self.optimizer.zero_grad()
            loss = self.computeBatchLoss(ndx, oneloader, trainloader.batch_size, trainMetrics_gpu)
            loss.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(trainloader.dataset)
        return trainMetrics_gpu.to('cpu')

    def computeBatchLoss(self, ndx, loader, size, metrics):

        # the loader contains the information from __getitem__, each has a size of batch_size
        data, labels, index_t, series_uid, xyz = loader

        # non_blocking=True: if you try to access data immediately after executing the statement,
        # it may still be on the CPU. If you need to use the data in the very next statement,
        # then using non_blocking=True won’t really help because the next statement will wait
        # till the data has been moved to the GPU.
        # On the other hand, if you need to move several objects to the GPU,
        # you can use non_blocking=True to move to the GPU in parallel using multiple background threads.
        # you can always use non_blocking=True
        # but here it raises an error so I remove it
        data_gpu = data.to(self.device)
        labels_gpu = labels.to(self.device)
        index_gpu = index_t.to(self.device)

        if self.augmentation:
            data_gpu = self.augmodel(data_gpu)
        # the output is designed in Net. The first is the result before softmax
        logits_gpu, probability_gpu = self.model(data_gpu)

        # reduction=none gives the loss per sample
        lossfunc = nn.CrossEntropyLoss(reduction='none')

        # labels is one hot-encoding [0, 1] indicating 2nd class [1, 0] indicating 1st class
        # CrossEntropyLoss don't recognize one hot-encoding, but the label of each class
        # CrossEntropyLoss contains softmax, so we input the one before softmax
        loss_gpu = lossfunc(logits_gpu, labels_gpu[:, 1])

        # store information into metrics
        start = ndx * size
        end = start + labels.size(0)

        metrics[0, start:end] = index_gpu
        metrics[1, start:end] = probability_gpu[:, 1]
        metrics[2, start:end] = loss_gpu
        return loss_gpu.mean()

    def validation(self, validationloader):
        # no gradient update
        with t.no_grad():
            # set model to evaluation mode
            self.model.eval()
            # zero tensor to store result
            valMetrics_gpu = t.zeros(3, len(validationloader.dataset), device=self.device)
            for ndx, oneloader in enumerate(validationloader):
                # logging.info('Starting the {}th batch in validation loader'.format(ndx))
                loss = self.computeBatchLoss(ndx, oneloader, validationloader.batch_size, valMetrics_gpu)
        return valMetrics_gpu.to('cpu')

    def logMetrics(self, epoch, mode, metrics, threshold=0.5):
        """
        :param epoch: current epoch
        :param mode: 'training' or 'validation'
        :param metrics: TrainMetrics or ValMetrics. 0 for true labels, 1 for probability result, 2 for loss
        :param threshold: classification threshold, default is 0.5, because we only have probability in metrics
        :return:
        """
        # create mask
        pos_label_mask = metrics[0] >= threshold
        pos_pre_mask = metrics[1] >= threshold
        neg_label_mask = ~pos_label_mask
        neg_pre_mask = ~pos_pre_mask

        # compute count
        pos_count = int(pos_label_mask.sum()) or 1
        neg_count = int(neg_label_mask.sum()) or 1

        # compute correct classifications
        pos_correct = int((pos_label_mask & pos_pre_mask).sum())
        neg_correct = int((neg_label_mask & neg_pre_mask).sum())

        # compute precision, recall and F1 score
        true_positive = pos_correct
        true_negative = neg_correct
        false_positive = int((neg_label_mask & pos_pre_mask).sum())
        false_negative = int((pos_label_mask & neg_pre_mask).sum())

        if np.float32(true_positive + false_positive) == 0:
            precision = -999
        else:
            precision = true_positive / np.float32(true_positive + false_positive)
        if np.float32(true_positive + false_negative) == 0:
            recall = -999
        else:
            recall = true_positive / np.float32(true_positive + false_negative)
        if precision == -999 or recall == -999 or (precision + recall == 0):
            f1 = -999
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # store them into dictionary
        metrics_dict = {'loss_all': metrics[2].mean(),
                        'loss_pos': metrics[2, pos_label_mask].mean(),
                        'loss_neg': metrics[2, neg_label_mask].mean(),
                        'correct_all': (pos_correct + neg_correct) / np.float32(metrics.shape[1]) * 100,
                        'correct_pos': pos_correct / np.float32(pos_count) * 100,
                        'correct_neg': neg_correct / np.float32(neg_count) * 100,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1}

        # log out
        # loss is with a decimal part of length 4
        logging.info(('Epoch {}, {}, {loss_all:.4f} loss, ' +
                     '{correct_all:-5.1f}% correct in total, ').format(epoch, mode, **metrics_dict))
        logging.info(('{loss_pos:.4f} positive loss, ' +
                      '{correct_pos:-5.1f}% positive correct, ' +
                      '({pos_correct:} of {pos_count:}).').format(pos_correct=pos_correct,
                                                                  pos_count=pos_count,
                                                                  **metrics_dict))
        logging.info(('{loss_neg:.4f} negative loss, ' +
                      '{correct_neg:-5.1f}% negative correct, ' +
                      '({neg_correct:} of {neg_count:}).').format(neg_correct=neg_correct,
                                                                  neg_count=neg_count,
                                                                  **metrics_dict))
        logging.info(('Precision is {precision:.4f}, ' +
                      'Recall is {recall:.4f}, ' +
                      'F1 score is {f1:.4f}').format(**metrics_dict))

        self.initTensorboardWriters()
        writer = getattr(self, mode + '_writer')

        for key, value in metrics_dict.items():
            # 1 element: name, 2 element: y, 3 element: x
            writer.add_scalar('seg_' + key, value, self.totalTrainingSamples_count)

        writer.flush()  # ensure all pending events have been written to disk

    def main(self):
        # start
        logging.info('Starting {}, {}'.format(type(self).__name__, self.args))

        # initiate data loader
        trainloader = self.initTrainloader()
        validationloader = self.initValidationloader()

        # start iteration for training
        for epoch in range(self.epoch_start, self.args.epochs+1):
            logging.info('Epoch {} of {}, {}/{} batches of size {}*{}.'.format(
                epoch,
                self.args.epochs,
                len(trainloader),
                len(validationloader),
                self.args.batch_size,
                (t.cuda.device_count() if self.use_cuda else 1)))

            TrainMetrics = self.training(trainloader)

            # save training model if the loss is better
            loss_this_model = TrainMetrics[2].mean()
            if loss_this_model < self.loss:
                self.saveModel(epoch=epoch, bestloss=loss_this_model)
            # log out
            self.logMetrics(epoch, 'training', TrainMetrics)
            logging.info('Epoch {}, training ends'.format(epoch))
            # validation
            ValMetrics = self.validation(validationloader)
            self.logMetrics(epoch, 'validation', ValMetrics)
            logging.info('Epoch{}, validation ends'.format(epoch))

    def saveModel(self, epoch, bestloss):
        # /gscratch/stf/nelljy
        pathModel = './savedmodel/{}_{}_{}.state'.format(self.args.dataset,
                                                         self.time,
                                                         self.totalTrainingSamples_count)
        model = self.model
        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'time': time.strftime('%Y/%m/%d %H:%M:%S', time.localtime()),
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'best_loss': bestloss,
            'totalTrainingSamples_count': self.totalTrainingSamples_count
        }

        t.save(state, pathModel)


if __name__ == '__main__':
    LunaTrainingApp().main()
