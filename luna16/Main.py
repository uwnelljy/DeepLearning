import sys
from Net import LunaModel
import torch as t
import torch.nn as nn
import numpy as np
import logging
import torch.optim as optim
from DataLoader import LunaDataset
from torch.utils.data import DataLoader
import argparse
import warnings

STRIDE = 10

logging.basicConfig(filename='luna.log',
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
        parser.add_argument('--balanced',
                            help='Create balanced data with half positive cases and half negative cases',
                            action='store_true',  # if we use --balanced, then it will be True, ow it will be False
                            default=False)
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

        self.use_cuda = t.cuda.is_available()
        self.device = t.device('cuda') if self.use_cuda else t.device('cpu')
        self.model = self.initmodel()
        self.optimizer = self.initoptimizer()
        self.totalTrainingSamples_count = 0

    def initmodel(self):
        model = LunaModel()
        if self.use_cuda:
            logging.info('Using CUDA, {} devices.'.format(t.cuda.device_count()))  # the number of GPUs
            if t.cuda.device_count() > 1:
                model = nn.DataParallel(model) # parallel model computing
            model = model.to(self.device)
        return model

    def initoptimizer(self):
        # About momentum:
        # gradient descent: w_{k+1} = w_k - alpha*f'(w_k)
        # with momentum: z_{k+1} = beta*z_k + f'(w_k),
        #                w_{k+1} = w_k - alpha*z_{k+1}
        # control the updating. High momentum, smooth gradient descent.
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def initTrainloader(self):
        # get training dataset
        training_dataset = LunaDataset(stride=STRIDE,
                                       isVal_bool=False,
                                       ratio=int(self.args.balanced),
                                       augmentation=self.augmentation)  # int(True) = 1, int(False) = 0
        logging.info('Balanced: {}, ratio: {}, {} positive samples, {} negative samples'.format(
            self.args.balanced,
            training_dataset.ratio,
            len(training_dataset.positive_list),
            len(training_dataset.negative_list)))
        # read batch_size from command line
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()  # why????????

        # pinned memory transfers to GPU quickly
        # num_workers: provide parallel loading of data by using separate processes and shared memory
        # each worker process produces complete batches
        # this helps make sure hungry GPUs are well fed with data
        # num_workers: worker将它负责的batch加载进RAM，dataloader就可以直接从RAM中找本轮迭代要用的batch。
        # 如果num_worker设置得大，好处是寻batch速度快，因为下一轮迭代的batch很可能在上一轮/上上一轮...迭代时已经加载好了。
        # 坏处是内存开销大，也加重了CPU负担（worker加载数据到RAM的进程是进行CPU复制）。
        # 如果num_worker设为0，意味着每一轮迭代时，dataloader不再有自主加载数据到RAM这一步骤，
        # 只有当你需要的时候再加载相应的batch，当然速度就更慢。
        trainloader = DataLoader(training_dataset, batch_size=batch_size,
                                 num_workers=self.args.num_workers, pin_memory=self.use_cuda)
        # the length of trainloader is the number of batches
        # len(trainloader.dataset): the number of images in all the batches
        return trainloader

    def initValidationloader(self):
        validation_dataset = LunaDataset(stride=STRIDE, isVal_bool=True)
        batch_size = self.args.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()
        validationloader = DataLoader(validation_dataset, batch_size=batch_size,
                                      num_workers=self.args.num_workers, pin_memory=self.use_cuda)
        return validationloader

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
        data, labels, series_uid, xyz = loader

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

        metrics[0, start:end] = labels_gpu[:, 1].detach() # we use detach since no need to hold on to gradients
        metrics[1, start:end] = probability_gpu[:, 1].detach()
        metrics[2, start:end] = loss_gpu.detach()
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
        pos_count = int(pos_label_mask.sum())
        neg_count = int(neg_label_mask.sum())

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

    def main(self):
        # start
        logging.info('Starting {}, {}'.format(type(self).__name__, self.args))

        # initiate data loader
        trainloader = self.initTrainloader()
        validationloader = self.initValidationloader()

        # start iteration for training
        for epoch in range(1, self.args.epochs+1):
            logging.info('Epoch {} of {}, {}/{} batches of size {}*{}.'.format(
                epoch,
                self.args.epochs,
                len(trainloader),
                len(validationloader),
                self.args.batch_size,
                (t.cuda.device_count() if self.use_cuda else 1)))

            TrainMetrics = self.training(trainloader)
            self.logMetrics(epoch, 'Training', TrainMetrics)
            logging.info('Epoch {}, training ends'.format(epoch))
            ValMetrics = self.validation(validationloader)
            self.logMetrics(epoch, 'Validation', ValMetrics)
            logging.info('Epoch{}, validation ends'.format(epoch))


if __name__ == '__main__':
    LunaTrainingApp().main()