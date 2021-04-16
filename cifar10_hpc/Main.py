from src.DataLoader import DataLoaderCIFAR
from src.Net import *
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
EPOCHES = 500
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

    def train_model(self):
        self.model.train()
        optimizer = optim.SGD(params=self.model.parameters(), lr=self.learningrate)
        # train on train batch
        total_time = 0
        train_batch = self.get_train_minibatch()
        for epoch in range(1, self.epoches+1):
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
            total_time += time_used
            if epoch == self.epoches:
                logging.debug('After last epoch, the training loss is {}, '
                              '{} time used in total.'.format(loss_all_train, total_time))

    def validation_model(self, model):
        model.eval()
        # validation on test batch
        for name, data in [('train', self.get_train_minibatch()), ('validation', self.get_test_minibatch())]:
            correct = 0
            total = 0
            with t.no_grad():
                for imgs, labels in data:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    outputs = model(imgs)
                    _, predictions = t.max(outputs, dim=1)  # predictions are actually index of the max value
                    total += predictions.shape[0]
                    correct += int((predictions == labels).sum())
            logging.debug('Accuracy of {} dataset is {}'.format(name, correct/total))


if __name__ == '__main__':
    logging.debug('******* Training on {} *******'.format(device))

    logging.debug('####### Using simple convolution kernel neuro network #######')
    model1 = ConConTanh(n_channels_1=32, n_channels_2=16, fc1_output_features=32, classes=10)
    model1 = model1.to(device)
    modelrun1 = RunModel(path=PATH, batchsize=BATCHSIZE, model=model1,
                        learningrate=LEARNINGRATE,
                        epoches=EPOCHES, loss_function=LOSS)
    modelrun1.train_model()
    modelrun1.validation_model(model1)

    logging.debug('####### Using dropoff neuro network #######')
    model2 = NetDropOut(n_channels_1=32, n_channels_2=16, fc1_output_features=32, classes=10)
    model2 = model2.to(device)
    modelrun2 = RunModel(path=PATH, batchsize=BATCHSIZE, model=model2,
                        learningrate=LEARNINGRATE,
                        epoches=EPOCHES, loss_function=LOSS)
    modelrun2.train_model()
    modelrun2.validation_model(model2)

    logging.debug('####### Using batch normalization neuro network #######')
    model3 = NetBatch(n_channels_1=32, n_channels_2=16, fc1_output_features=32, classes=10)
    model3 = model3.to(device)
    modelrun3 = RunModel(path=PATH, batchsize=BATCHSIZE, model=model3,
                         learningrate=LEARNINGRATE,
                         epoches=EPOCHES, loss_function=LOSS)
    modelrun3.train_model()
    modelrun3.validation_model(model3)

    logging.debug('####### Using a layer of resnet #######')
    model4 = NetDepth(n_channels_1=32, n_channels_2=16, fc1_output_features=32, classes=10)
    model4 = model4.to(device)
    modelrun4 = RunModel(path=PATH, batchsize=BATCHSIZE, model=model4,
                         learningrate=LEARNINGRATE,
                         epoches=EPOCHES, loss_function=LOSS)
    modelrun4.train_model()
    modelrun4.validation_model(model4)

