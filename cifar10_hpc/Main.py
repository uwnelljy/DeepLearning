from src.DataLoader import DataLoaderCIFAR
from src.Net import ConConTanh
import torch as t
from torch import optim
import torch.nn as nn
import logging
import time

logging.basicConfig(filename='cifar10.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    filemode='a',
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d%I:%M:%S %p')


PATH = 'data/'
BATCHSIZE = 64
LEARNINGRATE = 1e-2
EPOCHES = 10
LOSS = nn.CrossEntropyLoss()


device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')


class RunModel:
    def __init__(self, path, batchsize,
                 model,
                 learningrate,
                 epoches,
                 loss_function):
        self.path = path
        self.batchsize = batchsize
        self.model = model
        self.loss_function = loss_function
        self.learningrate = learningrate
        self.epoches = epoches
        self.loader = self.get_loader()

    def preprocess(self):
        pass

    def get_loader(self):
        return DataLoaderCIFAR(data_path=self.path, download=False)

    def get_train_minibatch(self):
        loader = self.loader
        return loader.train_loader_batch(self.batchsize)

    def get_test_minibatch(self):
        loader = self.loader
        return loader.test_loader_batch(self.batchsize)

    def run(self):
        optimizer = optim.SGD(params=self.model.parameters(), lr=self.learningrate)
        # train on train batch
        total_time = 0
        train_batch = self.get_train_minibatch()
        for epoch in range(1, self.epoches+1):
            logging.debug('Epoch {} starts'.format(epoch))
            start_time = time.time()
            loss_all_train = 0
            for imgs, labels in train_batch:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = self.model(imgs)
                loss = self.loss_function(outputs, labels)

                # add regularization to loss
                lambda_l2 = 0.001
                norm_l2 = sum([p.pow(2).sum() for p in self.model.parameters()])
                loss = loss + lambda_l2 * norm_l2

                # zero gradient
                optimizer.zero_grad()
                # backward for gradient
                loss.backward()
                # update parameters
                optimizer.step()

                loss_all_train += loss.item()  # transform the loss to an item to escape the gradient
            end_time = time.time()
            time_used = end_time-start_time
            logging.debug('Epoch {} ends, {} time used, training loss is {}'.format(epoch, time_used, loss_all_train))
            total_time += time_used
            if epoch == self.epoches:
                logging.debug('{} time used in total'.format(total_time))
        # validation on test batch


if __name__ == '__main__':
    model1 = ConConTanh(n_channels_1=16, n_channels_2=8, fc1_output_features=32, classes=10)

    logging.debug('******* Training on {} *******'.format(device))

    model1 = model1.to(device)

    modelrun = RunModel(path=PATH, batchsize=BATCHSIZE, model=model1,
                        learningrate=LEARNINGRATE,
                        epoches=EPOCHES, loss_function=LOSS)

    modelrun.run()
