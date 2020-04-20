import glob
import torch
import numpy as np

from torch.utils.data import Dataset


class PolyVectorFieldDataset(Dataset):
    def __init__(self, data_path, target_path):
        self.data_path = data_path
        self.target_path = target_path

        self.rasters = glob.glob(data_path)
