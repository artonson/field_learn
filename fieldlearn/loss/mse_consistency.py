import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from fieldlearn.utils import complex_to_angle_batch
from fieldlearn.data_generation.smoothing import loss_function_batch as fidelity_consistency_loss


class MSEConsistencyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean', mse_weight=0.5, fidelity_weight=0.4):
        if reduction == 'sum':
            raise NotImplementedError('reduction="sum" is not implemented')
        super().__init__(size_average, reduce, reduction)
        self.reduction = reduction
        self.mse_weight = mse_weight
        self.fidelity_weight = fidelity_weight

    def forward(self, input, target):
        fid_cons = fidelity_consistency_loss(
            complex_to_angle_batch(input[:, :2]), complex_to_angle_batch(input[:, 2:]), 
            complex_to_angle_batch(target[:, :2]), complex_to_angle_batch(target[:, 2:]),
            fidelity_w=self.fidelity_weight
        ) 
        return self.mse_weight * F.mse_loss(target, input, reduction=self.reduction) + (1 - self.mse_weight) * fid_cons
