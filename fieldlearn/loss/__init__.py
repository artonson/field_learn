import torch
import torch.nn.functional as F

from fieldlearn.data_generation.smoothing import loss_function_batch as fidelity_consistency_loss
from fieldlearn.loss.lapl1 import Lap1Loss
from fieldlearn.utils import complex_to_angle_batch


def calc_min_angle_diff_component(u, u0, v0):
    black_pixels = (u != 0).all(dim=1)

    diff_uu0 = torch.abs(complex_to_angle_batch(u - u0))
    diff_uv0 = torch.abs(complex_to_angle_batch(u - v0))

    uu0_less_uv0 = (diff_uu0 < diff_uv0).float()

    mean_uu0 = (diff_uu0 * uu0_less_uv0)[black_pixels].mean()
    mean_uv0 = (diff_uv0 * (1 - uu0_less_uv0))[black_pixels].mean()
    return 0.5 * mean_uu0 + 0.5 * mean_uv0


def min_angle_diff_loss(input, target):
    u0, v0 = input[:, :2], input[:, 2:]
    u, v = target[:, :2], target[:, 2:]
    return 0.5 * calc_min_angle_diff_component(u, u0, v0) + 0.5 * calc_min_angle_diff_component(v, v0, u0)


def masked_mse(input, target):
    black_pixels = (target != 0).all(dim=1, keepdims=True).repeat((1, 4, 1, 1))
    return F.mse_loss(input[black_pixels], target[black_pixels])


def make_loss_function(loss_type):
    if loss_type == 'mse':
        return masked_mse

    elif loss_type == 'fid_cons':
        return lambda input, target: fidelity_consistency_loss(
            complex_to_angle_batch(target[:, :2]), complex_to_angle_batch(target[:, 2:]),
            complex_to_angle_batch(input[:, :2]), complex_to_angle_batch(input[:, 2:]),
            fidelity_w=0.4
        )

    elif loss_type == 'mse_fid_cons':
        return lambda input, target: masked_mse(input, target) + fidelity_consistency_loss(
            complex_to_angle_batch(target[:, :2]), complex_to_angle_batch(target[:, 2:]),
            complex_to_angle_batch(input[:, :2]), complex_to_angle_batch(input[:, 2:]),
            fidelity_w=0.4)

    elif loss_type == 'lapl1':
        return Lap1Loss()

    elif loss_type == 'min_diff':
        return lambda input, target: min_angle_diff_loss(input, target)

    elif loss_type == 'mse_min_diff':
        return lambda input, target: masked_mse(input, target) + min_angle_diff_loss(input, target)
