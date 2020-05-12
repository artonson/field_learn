import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from vectran.data.transforms.degradation_models import DegradationGenerator


class PolyVectorFieldDataset(Dataset):
    def __init__(self, raster_path, field_path, degradations_list=None):
        self.data_path = raster_path
        self.target_path = field_path

        if degradations_list:
            self.degraded = True
            self.degradations_list = degradations_list
            self.degrad_gen = DegradationGenerator(degradations_list)
        else:
            self.degraded = False

        def normalize_path(path):
            return path.split('/')[-1][:-4]

        rasters = glob.glob(raster_path)
        fields = glob.glob(field_path)
        self.rasters = sorted(rasters, key=lambda x: normalize_path(x))
        self.fields = sorted(fields, key=lambda x: normalize_path(x))

        assert len(self.rasters) == len(self.fields)
        raster_names = [normalize_path(x) for x in self.rasters]
        field_names = [normalize_path(x) for x in self.fields]
        assert raster_names == field_names

    def __len__(self):
        return len(self.rasters)

    def __getitem__(self, idx):
        raster = Image.open(self.rasters[idx])
        field = np.load(self.fields[idx])
        field = np.nan_to_num(field)
        if self.degraded:
            degraded_raster = self.degrad_gen.do_degrade(np.array(raster) / 255.)
            return transforms.ToTensor()(degraded_raster.astype(np.float32)), torch.tensor(field)
        else:
            return transforms.ToTensor()(raster), torch.tensor(field)


def make_dataset(dataset_name, degradations=None, only_val=False, data_path='/data'):
    train_dataset = None

    if dataset_name == 'abc':
        if not only_val:
            train_dataset = PolyVectorFieldDataset(
                os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/abc/128x128/train/raster/*.png'),
                os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/abc/128x128/train/field/*.npy'),
                degradations_list=degradations
            )
        val_dataset = PolyVectorFieldDataset(
            os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/abc/128x128/val/raster/*.png'),
            os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/abc/128x128/val/field/*.npy'),
            degradations_list=degradations
        )

    elif dataset_name == 'abc_complex':
        if not only_val:
            train_dataset = PolyVectorFieldDataset(
                os.path.join(data_path,
                             'field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/train/raster/*.png'),
                os.path.join(data_path,
                             'field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/train/field/*.npy'),
                degradations_list=degradations
            )
        val_dataset = PolyVectorFieldDataset(
            os.path.join(data_path,
                         'field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/val/raster/*.png'),
            os.path.join(data_path,
                         'field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/val/field/*.npy'),
            degradations_list=degradations
        )

    elif dataset_name == 'pfp':
        if not only_val:
            train_dataset = PolyVectorFieldDataset(
                os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/pfp/64x64/train/raster/*.png'),
                os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/pfp/64x64/train/field/*.npy'),
                degradations_list=degradations
            )
        val_dataset = PolyVectorFieldDataset(
            os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/pfp/64x64/val/raster/*.png'),
            os.path.join(data_path, 'field_learn/datasets/field_datasets/patched/pfp/64x64/val/field/*.npy'),
            degradations_list=degradations
        )
    return train_dataset, val_dataset


def make_svg_dataset(dataset_name, only_val=False, data_path='/data'):
    train_paths = None

    if dataset_name == 'abc':
        if not only_val:
            pass
        val_paths = sorted(
            glob.glob(f'{data_path}/field_learn/datasets/svg_datasets/patched/abc/val/**/*.svg', recursive=True))

    elif dataset_name == 'abc_complex':
        if not only_val:
            pass
        val_paths = sorted(
            glob.glob(f'{data_path}/field_learn/datasets/svg_datasets/patched/abc_complex_patches/val/**/*.svg',
                      recursive=True))

    elif dataset_name == 'pfp':
        if not only_val:
            pass
        val_paths = sorted(
            glob.glob(f'{data_path}/vectorization/datasets/svg_datasets/patched/precision-floorplan/*/val/**/*.svg',
                      recursive=True))

    return train_paths, val_paths
