from src.DataLoader import DataLoaderCIFAR
from src.Net import ConConTanh
import torch as t


PATH = 'data/'
MEAN = t.tensor([0.4914, 0.4822, 0.4465])  # mean of each channel
STD = t.tensor([0.2470, 0.2435, 0.2616])  # std of each channel


if __name__ == '__main__':
    loader = DataLoaderCIFAR(data_path=PATH, mean=MEAN, std=STD, download=False)
    train = loader.train_loader()
    img, label = train[1]
    img_data = img.unsqueeze(0)  # add one dimension to indicate batch size=1 using unsqueeze(0)
    model1 = ConConTanh(n_channels_1=16, n_channels_2=8, fc1_output_features=32, classes=10)
    print(model1.forward(img_data))
