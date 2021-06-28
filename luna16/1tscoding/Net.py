import torch.nn as nn
import random
import torch as t
import math
from torch.nn import functional


class Augmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()
        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def _buildTransformMatrix(self):
        # define translation matrix
        # which should be:
        # [[1, 0, a]
        #   0, 1, b]]
        # the diagonal matrix controls the scale, while ab controls offset.
        theta = t.eye(3)

        # modify transform matrix
        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    theta[i, i] *= -1

            if self.offset:
                random_effects = random.random() * 2 - 1
                theta[i, 2] = self.offset * random_effects

            if self.scale:
                random_effects = random.random() * 2 - 1
                theta[i, i] *= 1 + self.scale * random_effects

        if self.rotate:
            angle = random.random() * math.pi * 2
            cos = math.cos(angle)
            sin = math.sin(angle)
            rotation = t.tensor([
                [cos, -sin, 0],
                [sin, cos, 0],
                [0, 0, 1]
            ])
            theta @= rotation

        return theta

    def forward(self, chunk_gpu, chunk_mask_gpu):
        """
        :param chunk_gpu: with dimension 7*64*64
        :param chunk_mask_gpu: 7*64*64
        :return:
        """
        theta = self._buildTransformMatrix()
        theta = theta.expand(chunk_gpu.shape[0], -1, -1)  # create multiple theta for multiple batches
        theta = theta.to(chunk_gpu.device, t.float32)  # convert to gpu
        # create grid
        grid = functional.affine_grid(theta[:, :2], chunk_gpu.size(), align_corners=False)
        # padding_mode='border':对于越界的位置在网格中采用边界的pixel value进行填充。
        augmented_chunk_gpu = functional.grid_sample(chunk_gpu, grid, padding_mode='border', align_corners=False)
        augmented_mask_gpu = functional.grid_sample(chunk_mask_gpu.to(t.float32),  # grid and mask should have the same type: float
                                                    grid, padding_mode='border', align_corners=False)

        # add noise
        if self.noise:
            noise = t.rand_like(augmented_chunk_gpu)
            noise += self.noise
            augmented_chunk_gpu += noise

        return augmented_chunk_gpu, augmented_mask_gpu > 0.5  # bool


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
