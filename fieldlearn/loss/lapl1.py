import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np


class Lap1Loss(_Loss):
    def __init__(self, size_average=True, reduce=True, levels_n=4, weights=None, cuda=True):
        assert reduce, 'Not implemented'
        super().__init__(size_average, reduce)
        self.size_average = size_average

        self.levels_n = levels_n
        self.weights = [2 ** (-2 * i) for i in range(levels_n)]
        if weights is not None:
            assert len(weights) == levels_n
            self.weights = [self.weights[i] * weights[i] for i in range(levels_n)]

        self.cuda = cuda

    def forward(self, input, target):
        input_pyr = _make_laplacian_pyramid(input, self.levels_n, cuda=self.cuda)
        target_pyr = _make_laplacian_pyramid(target, self.levels_n, cuda=self.cuda)
        loss = input.new_zeros([])
        for i in range(len(input_pyr)):
            loss += F.l1_loss(input_pyr[i], target_pyr[i], size_average=True, reduce=True) * self.weights[i]
        if not self.size_average:
            loss *= np.prod(input.shape, dtype=float)
        return loss


def _make_laplacian_pyramid(img, levels_n, cuda=True):
    pyr = []
    current = img
    for level in range(levels_n - 1):
        gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0, cuda=cuda)
        diff = current - gauss
        pyr.append(diff)
        current = F.avg_pool2d(gauss, 2, 2)
    pyr.append(current)
    return pyr


def gauss_kernel(size=5, sigma=1.0):
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    return kernel


def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1, cuda=True):
    t_kernel_np = gauss_kernel(size=k_size, sigma=sigma).reshape([1, 1, k_size, k_size])
    t_kernel = torch.from_numpy(t_kernel_np)
    if cuda:
        t_kernel = t_kernel.cuda()
    num_channels = t_input.data.shape[1]
    t_kernel3 = torch.cat([t_kernel] * num_channels, 0)
    if cuda:
        t_kernel3 = t_kernel3.cuda()
    t_result = t_input
    for r in range(repeats):
        t_result = F.conv2d(t_result, t_kernel3, stride=1, padding=2, groups=num_channels)
    return t_result
