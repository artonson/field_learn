import argparse
import json
import os
import sys
import time

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/code/field_learn')
sys.path.append('/code/dev.vectorization')

from fieldlearn.models.field_regression import PolyVectorFieldRegression, DegradedPolyVectorFieldRegression
from fieldlearn.loss import make_loss_function, masked_mse
from fieldlearn.dataset import make_dataset
from fieldlearn.metrics import calc_orientation_similarity, calc_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='Model parameters config')
    parser.add_argument('--model-tag', default='test model',
                        help='Model tag for tensorboard')
    parser.add_argument('--dataset', default='abc', choices=('abc', 'abc_complex', 'pfp'),
                        help='Dataset to use')
    parser.add_argument('--loss', default='mse',
                        choices=('mse', 'fid_cons', 'mse_fid_cons', 'lapl1', 'min_diff', 'mse_min_diff'),
                        help='Loss function to use for field prediction')
    parser.add_argument('--degraded', action='store_true', default=False,
                        help='Whether to do predict segmentation map')
    parser.add_argument('--degraded-alpha', default=1,
                        help='Parameter for segmentation loss')
    parser.add_argument('--degraded-threshold', default=0.5,
                        help='Threshold for segmentation')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num-epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--batches-per-train', default=200, type=int,
                        help='How many batches to run during train epoch')
    parser.add_argument('--batches-per-val', default=20, type=int,
                        help='How many batches to run during validation')
    parser.add_argument('--checkpoint-path', default='/gpfs/data/home/m.taktasheva/runs/')
    parser.add_argument('--checkpoint-every', type=int, default=1)
    parser.add_argument('--load-from-checkpoint', action='store_true', default=False)
    return parser.parse_args()


def load_data(config):
    if config['degraded'] and 'degradations' in config:
        degradations = config['degradations']
    elif config['degraded'] and 'degradations' not in config:
        degradations = [
            'kanungo',
            'random_geometric',
            'distort',
            'gaussian_blur',
            'thresholding',
            'binary_blur',
            'noisy_binary_blur',
            'nothing'
        ]
    else:
        degradations = ['nothing']

    train_dataset, val_dataset = make_dataset(config['dataset'], degradations)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
    return train_loader, val_loader


def find_latest_checkpoint(checkpoint_path):
    checkpoints = os.listdir(checkpoint_path)
    if not checkpoints:
        return None, 0
    checkpoints = sorted(checkpoints, key=lambda path: int(path[len('checkpoint'):]))
    latest_checkpoint = checkpoints[-1]
    latest_checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
    epoch = int(latest_checkpoint[len('checkpoint'):])
    return latest_checkpoint_path, epoch


def train_loop(config):
    checkpoint_path = os.path.join(config['checkpoint_path'], config['model_tag'])
    os.makedirs(checkpoint_path, exist_ok=True)

    train_loader, val_loader = load_data(config)

    if config['degraded']:
        model = DegradedPolyVectorFieldRegression(normalize_outputs=True)
        loss_seg = nn.BCEWithLogitsLoss()
        alpha_seg = config['degraded_alpha']
        threshold_seg = config['degraded_threshold']
    else:
        model = PolyVectorFieldRegression(normalize_outputs=True)

    curr_iteration = 0
    start_epoch = 0

    #     if config['load_from_checkpoint'] or 'load_from_checkpoint' not in config:
    #         latest_checkpoint_path, start_epoch = find_latest_checkpoint(checkpoint_path)
    #         print(latest_checkpoint_path)
    #         if latest_checkpoint_path:
    #             curr_iteration = start_epoch * config['batches_per_train']
    #             model.load_state_dict(torch.load(latest_checkpoint_path))

    loss_field = make_loss_function(config['loss'])
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
    writer = SummaryWriter(comment=f'/{config["checkpoint_path"]}/{config["model_tag"]}')

    train_mse = []
    train_sim_u = []
    train_sim_v = []
    train_iou = []
    for epoch in tqdm(range(start_epoch, config['num_epochs'])):
        time_start_epoch = time.perf_counter()

        model.train()
        for batch_i, (gt_raster, gt_field) in enumerate(train_loader):
            time_start_batch = time.perf_counter()

            gt_seg = (gt_field != 0).all(dim=1, keepdim=True).float()

            if config['degraded']:
                pred_field, pred_seg_logits = model.cuda()(gt_raster.cuda())
                pred_seg = (torch.sigmoid(pred_seg_logits) > threshold_seg).float()
                loss = loss_field(gt_seg.cuda() * pred_field, gt_field.cuda()) + alpha_seg * loss_seg(pred_seg_logits,
                                                                                                      gt_seg.cuda())
                pred_field *= pred_seg.cuda()

            else:
                pred_field = model.cuda()(gt_raster.cuda())
                loss = loss_field(gt_seg.cuda() * pred_field, gt_field.cuda())
                pred_field *= gt_seg.cuda()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            mse = masked_mse(pred_field, gt_field.cuda())
            train_mse.append(mse.item())

            sim_u, sim_v = calc_orientation_similarity(gt_field.cuda(), pred_field)
            train_sim_u.append(sim_u.item())
            train_sim_v.append(sim_v.item())

            if config['degraded']:
                iou = calc_iou(pred_seg.int(), gt_seg.int().cuda())
                train_iou.append(iou.item())

            writer.add_scalar('Time/train_batch', time.perf_counter() - time_start_batch, curr_iteration)
            writer.add_scalar('Loss/train', loss.item(), curr_iteration)

            curr_iteration += 1

            if batch_i > config['batches_per_train']:
                break

        writer.add_scalar('Time/train_epoch', time.perf_counter() - time_start_epoch, epoch)
        writer.add_scalar('MSE/train', np.mean(train_mse), epoch)
        writer.add_scalar('Sim_u/train', np.mean(train_sim_u), epoch)
        writer.add_scalar('Sim_v/train', np.mean(train_iou), epoch)
        if config['degraded']:
            writer.add_scalar('IOU/train', np.mean(train_sim_v), epoch)

        if config['checkpoint_every'] != 0 and (epoch + 1) % config['checkpoint_every'] == 0:
            torch.save(model.state_dict(), f'{checkpoint_path}/checkpoint{epoch + 1}')

        model.eval()

        val_mse = []
        val_sim_u = []
        val_sim_v = []
        val_iou = []
        for batch_i, (gt_raster, gt_field) in enumerate(val_loader):

            gt_seg = (gt_field != 0).all(dim=1, keepdim=True).float()

            if config['degraded']:
                pred_field, pred_seg_logits = model.cuda()(gt_raster.cuda())
                pred_seg = (torch.sigmoid(pred_seg_logits) > threshold_seg).float()
                pred_field *= pred_seg

            else:
                pred_field = model.cuda()(gt_raster.cuda())
                pred_field *= gt_seg.cuda()

            mse = masked_mse(pred_field, gt_field.cuda())
            val_mse.append(mse.item())

            sim_u, sim_v = calc_orientation_similarity(gt_field.cuda(), pred_field)
            val_sim_u.append(sim_u.item())
            val_sim_v.append(sim_v.item())

            if config['degraded']:
                iou = calc_iou(pred_seg.int(), gt_seg.int().cuda())
                val_iou.append(iou.item())

            if batch_i > config['batches_per_val']:
                break

        writer.add_scalar('Time/train_and_val_epoch', time.perf_counter() - time_start_epoch, epoch)
        writer.add_scalar('MSE/val', np.mean(val_mse), epoch)
        writer.add_scalar('Sim_u/val', np.mean(val_sim_u), epoch)
        writer.add_scalar('Sim_v/val', np.mean(val_sim_v), epoch)
        if config['degraded']:
            writer.add_scalar('IOU/val', np.mean(val_iou), epoch)

        scheduler.step(np.mean(val_mse))


if __name__ == '__main__':
    args = parse_args()

    print(f'******Training model******')

    if args.config:
        print(f'******Using config from {args.config}******')
        with open(args.config) as file:
            config = json.load(file)
        config['model_tag'] = args.model_tag

    else:
        print(f'******No config, using command line arguments******')
        config = vars(args)

    for key, value in config.items():
        print(f'{key}: {value}')

    train_loop(config)

    print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")
