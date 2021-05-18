from Net import LunaModel
import torch as t
import torch.nn as nn
import logging
import torch.optim as optim
from DataLoader import LunaDataset
from torch.utils.data import DataLoader

logging.basicConfig(filename='luna.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    filemode='a',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d%I:%M:%S %p')


class LunaTrainingApp:
    def __init__(self):
        self.use_cuda = t.cuda.is_available()
        self.device = t.device('cuda') if self.use_cuda else t.device('cpu')
        self.model = self.initmodel()
        self.optimizer = self.initoptimizer()
        self.totalTrainingSamples_count = 0

        self.batch_size = 16
        self.epoches = 10

    def initmodel(self):
        model = LunaModel()
        if self.use_cuda:
            logging.debug('Using CUDA, {} devices.'.format(t.cuda.device_count())) # the number of GPU
            if t.cuda.device_count() > 1:
                model = nn.DataParallel(model) # parallel model computing
            model = model.to(self.device)
        return model

    def initoptimizer(self):
        return optim.SGD(self.model.parameters(), lr=0.001)

    def initTrainloader(self):
        training_dataset = LunaDataset(stride=10, isVal_bool=False)
        batch_size = self.batch_size
        if self.use_cuda:
            batch_size *= t.cuda.device_count()
        trainloader = DataLoader(training_dataset, batch_size=batch_size,
                                 pin_memory=self.use_cuda) # pinned memory transfers to GPU quickly
        return trainloader

    def training(self, trainloader):
        # set model to training mode, but the model is still self.model
        self.model.train()

        # create a matrix to store prediction result
        # 0 for true labels, 1 for probability result, 2 for loss
        trainMetrics_gpu = t.zeros(3, len(trainloader.dataset), device=self.device)
        ndx = 0
        for oneloader in trainloader:
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

    def main(self):
        trainloader = self.initTrainloader()
        for epoch in range(1, self.epoches+1):
            logging.debug('Epoch {} Training starts.'.format(epoch))
            TrainMetrics = self.training(trainloader)
            logging.debug(TrainMetrics.size())


if __name__ == '__main__':
    print(LunaTrainingApp().main())