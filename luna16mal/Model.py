from unet import UNet
import torch.nn as nn


class UNetSeg(nn.Module):
    def __init__(self, **kwargs):
        super(UNetSeg, self).__init__()

        # tail
        self.tail_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        # head (convert pixel values to probability)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_batch):
        bn_out = self.tail_batchnorm(input_batch)
        un_out = self.unet(bn_out)
        sm_out = self.sigmoid(un_out)
        return sm_out


class LunaBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=conv_channels,
                               kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels=conv_channels, out_channels=conv_channels,
                               kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_batch):
        out = self.conv1(input_batch)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool(out)
        return out


class LunaModel(nn.Module):
    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.tail_batchnorm = nn.BatchNorm3d(num_features=in_channels)
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
        # the dimension after 4 conv and maxpool: 32*48*48 -> 2*3*3
        # so the number of features is 8*8*2*3*3 = 1152
        self.head_linear = nn.Linear(in_features=1152, out_features=2)
        # softmax returns the probability of two classes [p1, p2]
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_batch):
        out = self.tail_batchnorm(input_batch)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        raw_output = self.head_linear(out.view(out.size(0), -1))
        out = self.softmax(raw_output)
        return raw_output, out
