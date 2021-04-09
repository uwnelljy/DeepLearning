from torchvision import datasets
from torchvision import transforms
import torch as t


class DataLoaderCIFAR:
    def __init__(self, data_path, download=False):
        self.download = download
        self.data_path = data_path
        self.mean_train, self.std_train = self.get_mean_std_train()
        self.mean_test, self.std_test = self.get_mean_std_test()

    def train_loader_without_normalize(self):
        train = datasets.CIFAR10(self.data_path, train=True, download=self.download,
                                 transform=transforms.ToTensor())
        return train

    def test_loader_without_normalize(self):
        test = datasets.CIFAR10(self.data_path, train=False, download=self.download,
                                transform=transforms.ToTensor())
        return test

    @staticmethod
    def get_mean_std_for_img_data(imgdata):
        # imgdata should have img and label
        one_img, _ = imgdata[0]
        # get the dimension of this img
        dimention = len(one_img.shape)
        imgs = t.stack([img for img, _ in imgdata], dim=dimention)  # stack along an additional dimension
        mean_imgs = imgs.view(3, -1).mean(dim=1)
        std_imgs = imgs.view(3, -1).mean(dim=1)
        return (mean_imgs, std_imgs)

    def get_mean_std_train(self):
        train = self.train_loader_without_normalize()
        mean_imgs, std_imgs = self.get_mean_std_for_img_data(train)
        return (mean_imgs, std_imgs)

    def get_mean_std_test(self):
        test = self.test_loader_without_normalize()
        mean_imgs, std_imgs = self.get_mean_std_for_img_data(test)
        return (mean_imgs, std_imgs)

    def train_loader_with_normalize(self):
        train = datasets.CIFAR10(self.data_path, train=True, download=self.download,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=self.mean_train, std=self.std_train)
                                 ]))
        return train

    def test_loader_with_normalize(self):
        test = datasets.CIFAR10(self.data_path, train=False, download=self.download,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=self.mean_test, std=self.std_test)
                                ]))
        return test

    def train_loader_batch(self, batchsize):
        train_batch = t.utils.data.DataLoader(self.train_loader_with_normalize(),
                                              batch_size=batchsize, shuffle=True)
        return train_batch

    def test_loader_batch(self, batchsize):
        test_batch = t.utils.data.DataLoader(self.test_loader_with_normalize(),
                                             batch_size=batchsize, shuffle=True)
        return test_batch


