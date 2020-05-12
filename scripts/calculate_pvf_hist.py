import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('/code/field_learn')
sys.path.append('/code/dev.vectorization')

from fieldlearn.models.field_regression import PolyVectorFieldRegression, DegradedPolyVectorFieldRegression
from fieldlearn.loss import masked_mse
from fieldlearn.dataset import make_dataset, make_svg_dataset
from fieldlearn.metrics import calc_orientation_similarity, calc_iou, orientation_similarity_to_angle
from vectran.data.graphics.graphics import VectorImage

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', size=15)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', default=1000,
                        help='Number of samples to calculate metrics on')
    return parser.parse_args()


def make_normalize_functions(dataset, raster_paths, svg_paths):
    raster_prefix = os.path.commonprefix(raster_paths)
    svg_prefix = os.path.commonprefix(svg_paths)

    raster_normalize = lambda path: path[len(raster_prefix):-4]
    svg_normalize = lambda path: path[len(svg_prefix):-4].replace('/', '_')

    if dataset == 'pfp':
        svg_normalize = lambda path: path[path.find('/val/') + 5:-4].replace('/', '_')

    return raster_normalize, svg_normalize


def match_svg_and_rasters(raster_paths, svg_paths, raster_normalize, svg_normalize):
    matched_svg = dict()
    matched_raster = dict()

    raster_paths_dict = dict()
    for path in raster_paths:
        raster_paths_dict[raster_normalize(path)] = path

    for path in svg_paths:
        key = svg_normalize(path)
        if key in raster_paths_dict:
            matched_svg[key] = path
            matched_raster[key] = raster_paths_dict[key]

    # check whether they matched correctly
    assert len(matched_svg) == len(matched_raster)
    assert matched_svg.keys() == matched_raster.keys()

    return matched_raster, matched_svg


def find_latest_checkpoint(checkpoint_path):
    checkpoints = glob.glob(f'{checkpoint_path}/checkpoint*')
    if not checkpoints:
        return None
    prefix = os.path.commonprefix(checkpoints)
    checkpoints = sorted(checkpoints, key=lambda path: int(path[len(prefix):]))
    latest_checkpoint = checkpoints[-1]
    latest_checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)
    return latest_checkpoint_path


def calculate_hist_data(config, out_path, num_samples=5000):
    # out path
    hist_data_path = os.path.join(out_path, f'hist_data_{num_samples}_samples.pickle')

    if os.path.exists(hist_data_path):
        with open(hist_data_path, 'rb') as inp:
            hist_data = pickle.load(inp)
        return hist_data

    # load data
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
        degradations = None

    _, val_dataset = make_dataset(config['dataset'], degradations=degradations, only_val=True,
                                  data_path='/gpfs/gpfs0/3ddl/')
    raster_paths = val_dataset.rasters
    _, svg_paths = make_svg_dataset(config['dataset'], only_val=True, data_path='/gpfs/gpfs0/3ddl')

    # match paths
    raster_normalize, svg_normalize = make_normalize_functions(config['dataset'], raster_paths, svg_paths)
    raster_paths_set, svg_paths_set = match_svg_and_rasters(raster_paths, svg_paths, raster_normalize, svg_normalize)

    # make dataloader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # load model from checkpoint
    checkpoint_path = os.path.join(config['checkpoint_path'], config['model_tag'])
    latest_checkpoint_path = find_latest_checkpoint(checkpoint_path)
    if not latest_checkpoint_path:
        return None

    if config['degraded']:
        model = DegradedPolyVectorFieldRegression(normalize_outputs=True)
    else:
        model = PolyVectorFieldRegression(normalize_outputs=True)

    model.load_state_dict(torch.load(latest_checkpoint_path))

    # calculate data
    hist_data = defaultdict(lambda: defaultdict(list))
    for sample_i, (gt_raster, gt_field) in enumerate(tqdm(val_loader, total=num_samples)):

        if sample_i > num_samples:
            break

        raster_path = val_dataset.rasters[sample_i]
        path_key = raster_normalize(raster_path)
        if path_key not in raster_paths_set and path_key not in svg_paths_set:
            continue

        try:
            num_primitives = len(VectorImage.from_svg(svg_paths_set[path_key]).paths)
        except:
            print(sample_i, svg_paths_set[path_key])
            exit()

        gt_seg = (gt_field != 0).all(dim=1, keepdim=True).float()

        if config['degraded']:
            pred_field, pred_seg_logits = model.cuda()(gt_raster.cuda())
            pred_seg = (torch.sigmoid(pred_seg_logits) > config['degraded_threshold']).float()
            pred_field *= pred_seg

        else:
            pred_field = model.cuda()(gt_raster.cuda())
            pred_field *= gt_seg.cuda()

        mse = masked_mse(pred_field, gt_field.cuda())
        sim_u, sim_v = calc_orientation_similarity(gt_field.cuda(), pred_field)

        hist_data[num_primitives]['mse'].append(mse.item())
        hist_data[num_primitives]['sim_u'].append(sim_u.item())
        hist_data[num_primitives]['sim_v'].append(sim_v.item())
        hist_data[num_primitives]['sim_u_angle'].append(orientation_similarity_to_angle(sim_u).item())
        hist_data[num_primitives]['sim_v_angle'].append(orientation_similarity_to_angle(sim_v).item())

        if config['degraded']:
            iou = calc_iou(pred_seg.int(), gt_seg.int().cuda())
            hist_data[num_primitives]['iou'].append(iou.item())

    hist_data = dict(hist_data)  # to pickle lambda
    with open(hist_data_path, 'wb') as out:
        pickle.dump(hist_data, out)
    return hist_data


def plot_metric_distibution(hist_data, metric, config, out_path):
    dataset_name = config['dataset']
    model_tag = config['model_tag']
    loss_func = config['loss']

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 15), sharex=True, sharey=True)
    fig.suptitle(f'{metric} distribution for {dataset_name} with {loss_func} loss', fontsize=16)

    if metric == 'mse':
        xmin, xmax = 0, 0.1
    elif metric in ('sim_u', 'sim_v'):
        xmin, xmax = 0.95, 1
    elif metric in ('sim_u_angle', 'sim_v_angle'):
        xmin, xmax = 0, 0.45
    elif metric == 'iou':
        xmin, xmax = 0.5, 1

    calc_hist_weights = lambda arr: np.ones_like(arr) / len(arr)

    num_primitives = 1
    for i in range(len(axes)):
        for ax in axes[i]:

            if num_primitives < 11:
                ax.hist(hist_data[num_primitives][metric], bins=50,
                        weights=calc_hist_weights(hist_data[num_primitives][metric]))
                ax.set_title(f'num_primitives = {num_primitives}')
                ax.set_xlim(xmin, xmax)

            elif num_primitives == 11:
                data_11plus = []
                for key in hist_data:
                    if key >= 11:
                        data_11plus.extend(hist_data[key][metric])

                ax.hist(data_11plus, bins=50,
                        weights=calc_hist_weights(data_11plus))
                ax.set_title(f'num_primitives > 11')
                ax.set_yticks([])
                ax.set_xlim(xmin, xmax)

            else:
                ax.axis('off')
                ax.set_xlim(xmin, xmax)

            num_primitives += 1

    for ax in axes[0]:
        ax.set_xlabel(f'{metric}')
        ax.set_ylabel('density')

    axes[1][3].tick_params(reset=True)

    fig.savefig(os.path.join(out_path, f'{metric}_hist.svg'))
    plt.close()
    return fig


def process_model(config_path, num_samples):
    print(f'******Calculating metrics******')
    print(f'******Using config from {config_path}******')
    with open(config_path) as file:
        config = json.load(file)

    model_tag = os.path.splitext(os.path.basename(config_path))[0]
    config['model_tag'] = model_tag

    out_path = f'/gpfs/data/home/m.taktasheva/runs_metrics/{model_tag}'
    os.makedirs(out_path, exist_ok=True)

    print('Calculating histogram data...')
    print(f'Output will be in {out_path}')
    hist_data = calculate_hist_data(config, out_path, num_samples=num_samples)
    if not hist_data:
        print('Model has no checkpoint')
    else:
        for metric in hist_data[1].keys():
            print(f'Plotting {metric}...')
            plot_metric_distibution(hist_data, metric, config, out_path)


if __name__ == '__main__':
    args = parse_args()

    for config_path in glob.glob('/gpfs/data/home/m.taktasheva/github/field_learn/scripts/clean_data_configs/*.json'):
        try:
            process_model(config_path, args.num_samples)
        except Exception as e:
            print(str(e))
        break

    for config_path in glob.glob(
            '/gpfs/data/home/m.taktasheva/github/field_learn/scripts/degraded_data_configs/*.json'):
        try:
            process_model(config_path, args.num_samples)
        except Exception as e:
            print(str(e))
        break
