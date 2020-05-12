import torch
from torch import nn

from fieldlearn.models.unet import SmallUnet


class PolyVectorFieldRegression(nn.Module):
    def __init__(self, normalize_outputs=True):
        super().__init__()
        self.field_model = SmallUnet(out_channels=4)
        self.normalize_outputs = normalize_outputs

    def forward(self, x):
        x = self.field_model.forward(x)
        batch_size, c, h, w = x.shape
        if self.normalize_outputs:
            x = x.reshape(batch_size, c // 2, 2 * h, w)
            x = torch.nn.functional.normalize(x, dim=1)
            x = x.reshape(batch_size, c, h, w)
        return x


class DegradedPolyVectorFieldRegression(nn.Module):
    def __init__(self, normalize_outputs=True):
        super().__init__()
        self.field_model = SmallUnet(out_channels=5)
        self.normalize_outputs = normalize_outputs

    def forward(self, input):
        input = self.field_model.forward(input)
        field = input[:, :4]
        line_seg = input[:, 4:]
        batch_size, c, h, w = field.shape
        if self.normalize_outputs:
            field = field.reshape(batch_size, c // 2, 2 * h, w)
            field = torch.nn.functional.normalize(field, dim=1)
            field = field.reshape(batch_size, c, h, w)
        return field, line_seg
