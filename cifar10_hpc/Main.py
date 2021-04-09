from src.DataLoader import DataLoaderCIFAR
from src.Net import ConConTanh
from torch import optim
import torch.nn as nn
import logging

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
        train_batch = self.get_train_minibatch()
        for epoch in range(1, self.epoches):
            logging.debug('Epoch {} starts'.format(epoch))
            loss_all_train = 0
            for imgs, labels in train_batch:
                imgs = imgs.cuda()
                labels = labels.cuda()
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
            logging.debug('Epoch {} ends, training loss is {}'.format(epoch, loss_all_train))
        # validation on test batch


if __name__ == '__main__':
    model1 = ConConTanh(n_channels_1=16, n_channels_2=8, fc1_output_features=32, classes=10)
    model1 = model1.cuda()
    modelrun = RunModel(path=PATH, batchsize=BATCHSIZE, model=model1,
                        learningrate=LEARNINGRATE,
                        epoches=EPOCHES, loss_function=LOSS)
    modelrun.run()
