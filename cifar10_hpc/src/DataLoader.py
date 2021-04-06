from torchvision import datasets
from torchvision import transforms
import torch as t


class DataLoaderCIFAR:
    def __init__(self, data_path, mean, std, download=False):
        self.download = download
        self.data_path = data_path
        self.mean = mean
        self.std = std
        self.train = self.train_loader
        self.test = self.test_loader

    def train_loader(self):
        train = datasets.CIFAR10(self.data_path, train=True, download=self.download,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=self.mean, std=self.std)
                                 ]))
        return train

    def test_loader(self):
        test = datasets.CIFAR10(self.data_path, train=False, download=self.download,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.mean, std=self.std)
                                ]))
        return test

    def train_loader_batch(self, batchsize):
        train_batch = t.utils.data.DataLoader(self.train, batch_size=batchsize, shuffle=True)
        return train_batch

    def test_loader_batch(self, batchsize):
        test_batch = t.utils.data.DataLoader(self.test, batch_size=batchsize, shuffle=True)
        return test_batch


