import torch.nn as nn
import torch as t
import torch.nn.functional as F  # functions without state


# first net: two convolution layers and one Tanh activation layer
class ConConTanh(nn.Module):
    def __init__(self, n_channels_1, n_channels_2, fc1_output_features, classes):
        super().__init__()
        self.n_channels_1 = n_channels_1
        self.n_channels_2 = n_channels_2
        self.fc1_output_features = fc1_output_features
        self.classes = classes

        # first convolution layer: 3*32*32 -> n_channels_1*32*32
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_channels_1,
                               kernel_size=3, padding=1)  # 3 means 3*3 kernel
        # second convolution layer: n_channels_1*16*16 -> n_channels_2*16*16
        self.Conv2 = nn.Conv2d(in_channels=self.n_channels_1, out_channels=self.n_channels_2,
                               kernel_size=3, padding=1)
        # first full connected layer: n_channels_2*8*8 -> fc1_output_features
        self.fc1 = nn.Linear(in_features=self.n_channels_2*8*8,
                             out_features=self.fc1_output_features)
        # second full connected layer: fc1_output_features -> number of classes
        self.fc2 = nn.Linear(in_features=self.fc1_output_features,
                             out_features=self.classes)

    def forward(self, input):
        out = self.Conv1(input)
        out = t.tanh(out)
        out = F.max_pool2d(out, 2)

        out = self.Conv2(out)
        out = t.tanh(out)
        out = F.max_pool2d(out, 2)

        # full connected layer
        # stretch to a straight tensor with n_channels_2*8*8 features
        out = out.view(-1, self.n_channels_2*8*8)
        out = self.fc1(out)
        out = t.tanh(out)
        out = self.fc2(out)

        return out


# second net: add dropout to overcome overfitting
# IMPORTANT: dropout is only used in training, do not use drop out in inference
class NetDropOut(nn.Module):
    def __init__(self, n_channels_1, n_channels_2, fc1_output_features, classes):
        super().__init__()
        self.n_channels_1 = n_channels_1
        self.n_channels_2 = n_channels_2
        self.fc1_output_features = fc1_output_features
        self.classes = classes

        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_channels_1,
                               kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.4)

        self.Conv2 = nn.Conv2d(in_channels=self.n_channels_1, out_channels=self.n_channels_2,
                               kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.4)

        self.fc1 = nn.Linear(in_features=self.n_channels_2*8*8,
                             out_features=self.fc1_output_features)

        self.fc2 = nn.Linear(in_features=self.fc1_output_features,
                             out_features=self.classes)

    def forward(self, input):
        out = self.Conv1(input)
        out = t.tanh(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout1(out)

        out = self.Conv2(out)
        out = t.tanh(out)
        out = F.max_pool2d(out, 2)
        out = self.dropout2(out)

        # full connected layer
        # stretch to a straight tensor with n_channels_2*8*8 features
        out = out.view(-1, self.n_channels_2*8*8)
        out = self.fc1(out)
        out = t.tanh(out)
        out = self.fc2(out)

        return out


# third net: add batch normalization to overcome overfitting
# IMPORTANT: batch normalization is only used in training, do not use it in inference
class NetBatch(nn.Module):
    def __init__(self, n_channels_1, n_channels_2, fc1_output_features, classes):
        super().__init__()
        self.n_channels_1 = n_channels_1
        self.n_channels_2 = n_channels_2
        self.fc1_output_features = fc1_output_features
        self.classes = classes

        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_channels_1,
                               kernel_size=3, padding=1)
        # normalization within each batch
        self.batchnorm1 = nn.BatchNorm2d(num_features=n_channels_1)

        self.Conv2 = nn.Conv2d(in_channels=self.n_channels_1, out_channels=self.n_channels_2,
                               kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=n_channels_2)

        self.fc1 = nn.Linear(in_features=self.n_channels_2 * 8 * 8,
                             out_features=self.fc1_output_features)

        self.fc2 = nn.Linear(in_features=self.fc1_output_features,
                             out_features=self.classes)

    def forward(self, input):
        out = self.Conv1(input)
        out = self.batchnorm1(out)
        out = t.tanh(out)
        out = F.max_pool2d(out, 2)

        out = self.Conv2(out)
        out = self.batchnorm2(out)
        out = t.tanh(out)
        out = F.max_pool2d(out, 2)

        out = out.view(-1, self.n_channels_2*8*8)
        out = self.fc1(out)
        out = t.tanh(out)
        out = self.fc2(out)

        return out


# forth net: adding a skip connection a la ResNet to the model to increase the depth
class NetDepth(nn.Module):
    def __init__(self, n_channels_1, n_channels_2, fc1_output_features, classes):
        super().__init__()
        self.n_channels_1 = n_channels_1
        self.n_channels_2 = n_channels_2
        self.fc1_output_features = fc1_output_features
        self.classes = classes

        self.Convd1 = nn.Conv2d(in_channels=3, out_channels=self.n_channels_1,
                                kernel_size=3, padding=1)
        self.Convd2 = nn.Conv2d(in_channels=self.n_channels_1, out_channels=self.n_channels_2,
                                kernel_size=3, padding=1)
        self.Convd3 = nn.Conv2d(in_channels=self.n_channels_2, out_channels=self.n_channels_2,
                                kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=self.n_channels_2 * 4 * 4,
                             out_features=self.fc1_output_features)

        self.fc2 = nn.Linear(in_features=self.fc1_output_features,
                             out_features=self.classes)

    def forward(self, input):
        out = self.Convd1(input)
        out = t.relu(out)
        out = F.max_pool2d(out, 2)

        out = self.Convd2(out)
        out = t.relu(out)
        out = F.max_pool2d(out, 2)

        out1 = out

        out = self.Convd3(out)
        out = t.relu(out)
        out = F.max_pool2d(out + out1, 2)

        out = out.view(-1, self.n_channels_2*4*4)
        out = self.fc1(out)
        out = t.relu(out)
        out = self.fc2(out)

        return out
