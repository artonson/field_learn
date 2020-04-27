import os
import time
import argparse

import torch
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from fieldlearn.training.dataloader import PolyVectorFieldDataset
from fieldlearn.training.models.unet import SmallUnetRegression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-tag', default='test_model')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--batches-per-update', default=1, type=int)
    parser.add_argument('--checkpoint-path', default='checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=10)
    return parser.parse_args()


def train_loop(config, model, train_loader, val_loader):
    checkpoint_path = os.path.join(config.checkpoint_path, config.model_tag)
    os.makedirs(checkpoint_path, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    writer = SummaryWriter(comment=f'/{config.checkpoint_path}/{config.model_tag}')

    iteration = 0
    for epoch in tqdm(range(config.num_epochs)):
        start_epoch = time.perf_counter()

        model.train()
        train_score = []
        train_loss = []
        for batch_i, (X_batch, y_batch) in enumerate(train_loader):
            start_batch = time.perf_counter()
            y_pred = model.cuda()(X_batch.cuda())
            loss = criterion(y_pred, y_batch.cuda())
            loss.backward()

            train_loss.append(loss.item())

            if (batch_i + 1) % config.batches_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()

            writer.add_scalar('Time/train_batch', time.perf_counter() - start_batch, iteration)
            writer.add_scalar('Loss/train', loss.item(), iteration)
            iteration += 1

            # batch_acc = criterion(y_pred, y_batch.cuda())
            # train_score.append(batch_acc.item())

        writer.add_scalar('Time/train_epoch', time.perf_counter() - start_epoch, epoch)
        # writer.add_scalar('MSE/train', np.mean(train_score), epoch)

        if config.checkpoint_every != 0 and (epoch + 1) % config.checkpoint_every == 0:
            torch.save(model.state_dict(), f'{checkpoint_path}/checkpoint{epoch}')

        model.eval()

        # score = []
        # for X_batch, y_batch in val_loader:
        #     y_pred = model.cuda()(X_batch.cuda())
        #     batch_score = criterion(y_pred, y_batch.cuda())
        #     score.append(batch_score.item())
        # val_score = np.mean(score)
        #
        # writer.add_scalar('Time/train_val_epoch', time.perf_counter() - start_epoch, epoch)
        # writer.add_scalar('MSE/val', val_score, epoch)
        #
        # scheduler.step(val_score)


if __name__ == '__main__':
    args = parse_args()

    raster_path = '/home/mtaktash/data/abc_field_examples/*.png'
    field_path = '/home/mtaktash/data/abc_field_examples/*.npy'

    train_dataset = PolyVectorFieldDataset(raster_path, field_path)
    val_dataset = PolyVectorFieldDataset(raster_path, field_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = SmallUnetRegression()
    train_loop(args, model, train_loader, val_loader)

    print(f"Peak memory usage by Pytorch tensors: {(torch.cuda.max_memory_allocated() / 1024 / 1024):.2f} Mb")
