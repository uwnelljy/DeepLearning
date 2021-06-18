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
