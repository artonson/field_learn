import glob
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PolyVectorFieldDataset(Dataset):
    def __init__(self, raster_path, field_path,
                 raster_transforms=transforms.ToTensor()):
        self.data_path = raster_path
        self.target_path = field_path

        self.rasters = glob.glob(raster_path)
        self.fields = glob.glob(field_path)

        assert len(self.rasters) == len(self.fields)

        self.raster_transforms = raster_transforms

    def __len__(self):
        return len(self.rasters)

    def __getitem__(self, idx):
        raster = Image.open(self.rasters[idx])
        field = np.load(self.fields[idx])
        field = np.nan_to_num(field)
        return self.raster_transforms(raster), torch.tensor(field)
