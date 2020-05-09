import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/code/field_learn')

from fieldlearn.dataset.dataset import PolyVectorFieldDataset
from fieldlearn.models.unet import SmallUnetRegression
from fieldlearn.loss.lapl1 import Lap1Loss
from fieldlearn.loss.mse_consistency import MSEConsistencyLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-tag', default='test_model',
                        help='Model name for tensorboard')
    parser.add_argument('--dataset', default='abc', choices=('abc', 'abc_complex', 'pfp'), 
                        help='Dataset to use')
    parser.add_argument('--loss', default='mse', choices=('mse', 'mse_and_cons', 'lapl1'), 
                        help='Loss function to use')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--batches-per-train', default=100, type=int,
                        help='How many batches to run during train epoch')
    parser.add_argument('--batches-per-val', default=10, type=int,
                        help='How many batches to run during validation')
    parser.add_argument('--checkpoint-path', default='~/runs')
    parser.add_argument('--checkpoint-every', type=int, default=10)
    return parser.parse_args()


def get_mask(X_batch, y_batch):
    mask = torch.zeros_like(X_batch)
    mask[X_batch < 1] = 1
    return mask.expand(y_batch.shape)


def calc_orientation_similarity(rasters, true_component, pred_component):
    true_angle = complex_to_angle_batch(true_component)
    pred_angle = complex_to_angle_batch(pred_component)
    sim = (1 + torch.cos(true_angle - pred_angle)) / 2
    sim = sim.sum(dim=(-1, -2)) / rasters[rasters < 1].shape[0]
    return torch.max(sim) / 11


from fieldlearn.utils import complex_to_angle_batch
from fieldlearn.data_generation.smoothing import loss_function_batch as fidelity_consistency_loss
import torch.nn.functional as F


def train_loop(config, model, train_loader, val_loader):
    checkpoint_path = os.path.join(config.checkpoint_path, config.model_tag)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    masked = True
    if config.loss == 'mse':
        criterion = nn.MSELoss()
    elif config.loss == 'lapl1':
        criterion = Lap1Loss()
    
    optimizer = SGD(model.parameters(), lr=config.lr, momentum=0.9)
#     scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    writer = SummaryWriter(comment=f'/{config.checkpoint_path}/{config.model_tag}')

    iteration = 0

    train_mse = []
    train_sim_u = []
    train_sim_v = []
    for epoch in tqdm(range(config.num_epochs)):
        start_epoch = time.perf_counter()

        model.train()
        for batch_i, (X_batch, y_batch) in enumerate(train_loader):
            start_batch = time.perf_counter()
            y_pred = model.cuda()(X_batch.cuda())
            if masked:
                y_pred *= get_mask(X_batch.cuda(), y_batch.cuda())
                
            loss = criterion(y_pred, y_batch.cuda())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            mse = nn.functional.mse_loss(y_pred, y_batch.cuda())
            train_mse.append(mse.item())

            sim_u = calc_orientation_similarity(X_batch.cuda(), y_batch[:, :2].cuda(), y_pred[:, :2])
            sim_v = calc_orientation_similarity(X_batch.cuda(), y_batch[:, 2:].cuda(), y_pred[:, 2:])
            train_sim_u.append(sim_u.item())
            train_sim_v.append(sim_v.item())

            writer.add_scalar('Time/train_batch', time.perf_counter() - start_batch, iteration)
            writer.add_scalar('Loss/train', loss.item(), iteration)

            iteration += 1

            if batch_i > config.batches_per_train:
                break

        writer.add_scalar('MSE/train', np.mean(train_mse), epoch)
        writer.add_scalar('Sim_u/train', np.mean(train_sim_u), epoch)
        writer.add_scalar('Sim_v/train', np.mean(train_sim_v), epoch)
        writer.add_scalar('Time/train_epoch', time.perf_counter() - start_epoch, epoch)
    
        if config.checkpoint_every != 0 and (epoch + 1) % config.checkpoint_every == 0:
            torch.save(model.state_dict(), f'{checkpoint_path}/checkpoint{epoch}')

        model.eval()

        val_mse = []
        val_sim_u = []
        val_sim_v = []
        for batch_i, (X_batch, y_batch) in enumerate(val_loader):
            y_pred = model.cuda()(X_batch.cuda())
            if masked:
                y_pred *= get_mask(X_batch.cuda(), y_batch.cuda())

            mse = nn.functional.mse_loss(y_pred, y_batch.cuda())
            val_mse.append(mse.item())

            sim_u = calc_orientation_similarity(X_batch.cuda(), y_batch[:, :2].cuda(), y_pred[:, :2])
            sim_v = calc_orientation_similarity(X_batch.cuda(), y_batch[:, 2:].cuda(), y_pred[:, 2:])
            val_sim_u.append(sim_u.item())
            val_sim_v.append(sim_v.item())

            if batch_i > config.batches_per_val:
                break

        writer.add_scalar('MSE/val', np.mean(val_mse), epoch)
        writer.add_scalar('Sim_u/val', np.mean(val_sim_u), epoch)
        writer.add_scalar('Sim_v/val', np.mean(val_sim_v), epoch)
        writer.add_scalar('Time/train_and_val_epoch', time.perf_counter() - start_epoch, epoch)

        
if __name__ == '__main__':
    args = parse_args()
    
    if args.dataset == 'abc':
        train_dataset = PolyVectorFieldDataset(
            '/data/field_learn/datasets/field_datasets/patched/abc/128x128/train/raster/*.png',
            '/data/field_learn/datasets/field_datasets/patched/abc/128x128/train/field/*.npy'
        )
        val_dataset = PolyVectorFieldDataset(
            '/data/field_learn/datasets/field_datasets/patched/abc/128x128/val/raster/*.png',
            '/data/field_learn/datasets/field_datasets/patched/abc/128x128/val/field/*.npy'
        )
        
    elif args.dataset == 'abc_complex':
        train_dataset = PolyVectorFieldDataset(
            '/data/field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/train/raster/*.png',
            '/data/field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/train/field/*.npy'
        )
        val_dataset = PolyVectorFieldDataset(
            '/data/field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/val/raster/*.png',
            '/data/field_learn/datasets/field_datasets/patched/abc_complex_patches/128x128/val/field/*.npy'
        )
        
    elif args.dataset == 'pfp':
        train_dataset = PolyVectorFieldDataset(
            '/data/field_learn/datasets/field_datasets/patched/pfp/64x64/train/raster/*.png',
            '/data/field_learn/datasets/field_datasets/patched/pfp/64x64/train/field/*.npy'
        )
        val_dataset = PolyVectorFieldDataset(
            '/data/field_learn/datasets/field_datasets/patched/pfp/64x64/val/raster/*.png',
            '/data/field_learn/datasets/field_datasets/patched/pfp/64x64/val/field/*.npy'
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = SmallUnetRegression()
    train_loop(args, model, train_loader, val_loader)

    print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")
