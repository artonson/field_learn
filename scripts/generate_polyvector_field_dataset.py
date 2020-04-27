import os
import glob
import json
import argparse
import numpy as np
import torch
import multiprocessing as mp

from PIL import Image
from tqdm import tqdm
from vectran.data.graphics.graphics import VectorImage
from vectran.data.graphics.units import Pixels
from vectran.renderers.cairo import render as cairo_render
from fieldlearn.data_generation.polyvector import compute_field, smooth_field


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', dest='input_dir',
                        required=True, help='input dir as glob path')
    parser.add_argument('-o', '--output-dir', dest='output_dir',
                        required=True, help='output dir.')
    parser.add_argument('-c', '--dataset-config', dest='dataset_config',
                        required=True, help='dataset configuration file.')
    return parser.parse_args()


def calculate_vector_field(input_files, output_dir, device, config):
    
    for svg_path in tqdm(input_files):
        img = VectorImage.from_svg(svg_path)
        raster = img.render(cairo_render)
        
        u, v = compute_field(img,
                             smoothing_fn=lambda x, y: smooth_field(x, y, **config["smoothing_params"]),
                             device=device,
                             **config["compute_params"])

        basename = os.path.basename(svg_path)[:-4]
        np.save(
            os.path.join(args.output_dir, 'field', basename + '.npy'),
            np.vstack([u.detach().cpu().numpy(), v.detach().cpu().numpy()])
        )

        raster_image = Image.fromarray(raster, mode='L')
        raster_image.save(os.path.join(args.output_dir, 'raster', basename + '.png'))


if __name__ == '__main__':
    args = parse_args()

    with open(args.dataset_config) as json_file:
        config = json.load(json_file)
        
    input_files = glob.glob(args.input_dir)
    
    os.makedirs(os.path.join(args.output_dir, 'field'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'raster'), exist_ok=True)
    
    num_jobs = torch.cuda.device_count()
    input_files_per_job = len(input_files) // num_jobs 
    
    running_processes = []
    mp.set_start_method('spawn')
    
    for job_idx in range(num_jobs):
        input_files_batch = input_files[job_idx * input_files_per_job: (job_idx + 1) * input_files_per_job]
        running_processes.append(
            mp.Process(
                target=calculate_vector_field, 
                args=(input_files_batch, args.output_dir, torch.device('cuda', job_idx), config)
        ))
        
    for p in running_processes:
        p.join()