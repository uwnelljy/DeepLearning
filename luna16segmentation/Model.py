from unet import UNet
import torch.nn as nn
import torch as t
import random
import math
from torch.nn import functional


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
