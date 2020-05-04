from contextlib import contextmanager
import os
import sys
import signal
import argparse

import numpy as np

sys.path.append('/code/dev.vectorization')
sys.path.append('/code/field_learn')

from vectran.data.graphics.graphics import VectorImage, Path
from vectran.simplification.join_qb import join_quad_beziers
from vectran.renderers.cairo import render, PT_LINE, PT_QBEZIER
from vectran.simplification.detect_overlaps import has_overlaps
from vectran.data.graphics.units import Pixels
from fieldlearn.utils import line_to_curve

class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def prepare_patch(patch, patch_size, 
                  simplify_curves=True, skip_overlaps=True,
                  tiny_segments_threshold=1.5, quad_beziers_fit_tol=.1, quad_beziers_w_tol=.1, 
                  max_relative_overlap=.5, 
                 ):
    # 1. Remove short segments
    patch.remove_tiny_segments(tiny_segments_threshold)
    if len(patch.paths) == 0:
        return None

    # 2. Simplify curves
    lines, curves = patch.vahe_representation()
    lines = np.asarray(lines)
    if simplify_curves and len(curves) > 0:
        curves = join_quad_beziers(curves, fit_tol=quad_beziers_fit_tol, w_tol=quad_beziers_w_tol).numpy()
    curves = np.asarray(curves)

    # 3. Skip the patch if it has overlays
    def render_primitives(primitives):
        return render(primitives, patch_size, data_representation='vahe')
    
    if skip_overlaps:
        if has_overlaps({PT_LINE: lines, PT_QBEZIER: curves}, render_primitives, max_relative_overlap=max_relative_overlap):
            return None

    # 4. Convert lines to curves
    curves = curves.tolist() + [line_to_curve(line) for line in lines]
    patch.paths = ([Path.from_primitive(PT_QBEZIER, prim) for prim in curves])
    return patch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', help='Input file')
    parser.add_argument('-o', '--output-dir', help='Output directory')
    parser.add_argument('--patch-height', type=int, help='Patch height')
    parser.add_argument('--patch-width', type=int, help='Patch width')
    parser.add_argument('--image-scale', type=int, default=-1,
                        help='A scaling coefficient such that image_size = image_scale * patch_size, if is -1, then image is not scaled')
    parser.add_argument('--num-augmentations', type=int, default=4, 
                        help='Number of augmentations per image')
    parser.add_argument('--simplify-curves', action='store_true', default=False, 
                        help='Simplify curves in patch')
    parser.add_argument('--skip-overlaps', action='store_true', default=False, 
                        help='Skip overlaps between curves in patch')
    return parser.parse_args()
    

if __name__ == '__main__':
    args = parse_args()
    filename = args.input_file
    patches_dir = args.output_dir
    
    patch_width = args.patch_height
    patch_height = args.patch_width
    distinguishability_threshold = 1.5

    width_min = 1
    width_max = 4
    mirroring = True
    rotation_min = 0
    rotation_max = 360
    translation_x_min = 0
    translation_x_max = patch_width
    translation_y_min = 0
    translation_y_max = patch_height

    augmentations_n = args.num_augmentations
    
    vector_image = VectorImage.from_svg(filename)

    for augmentation_i in range(augmentations_n):
        width = np.random.random() * (width_max - width_min) + width_min
        mirror = bool(np.random.randint(0, 2)) & mirroring
        rotation = np.random.random() * (rotation_max - rotation_min) + rotation_min
        translation = (np.random.rand(2) *
                       [translation_x_max - translation_x_min, translation_y_max - translation_y_min] +
                       [translation_x_min, translation_y_min])

        augmented_image = vector_image.copy()
        augmented_image.scale_to_width('min', width)
        if mirror:
            augmented_image.mirror()
        augmented_image.rotate(rotation)
        augmented_image.translate(translation, adjust_view=True)
        
        if args.image_scale != -1:
            # to have better patches complexity
            scaled_height = args.image_scale * patch_height
            scaled_width = args.image_scale * patch_width
            augmented_image.scale(
                min(scaled_height, scaled_width) / (min(int(augmented_image.width), int(augmented_image.height))),
                only_coordinates=True)
            augmented_image.view_height = Pixels(scaled_height)
            augmented_image.view_width = Pixels(scaled_width)
            augmented_image.crop((0, scaled_height, 0, scaled_width))

        patches = augmented_image.split_to_patches((patch_width, patch_height)).reshape(-1)

        basename = os.path.basename(filename)[:-4]
        orientation = {False: 'o', True: 'm'}[mirror]

        for patch_i, patch in enumerate(patches):
            try:
                with time_limit(5):
                    patch = prepare_patch(patch, (patch_width, patch_height), args.simplify_curves, args.skip_overlaps)
            except TimeoutException:
                print(f'Time exceeded for patch {patch_i} in {basename}')
                continue
            else:
                if patch is None:
                    continue
                save_path = (f'{patches_dir}/{basename}/{patch_width}x{patch_height}/'
                             f'width_{width:.2f}_ori_{orientation}_rot_{rotation:.2f}_'
                             f'tr_{translation[0]:.2f}_{translation[1]:.2f}_'
                             f'{int(patch.x.as_pixels())}_{int(patch.y.as_pixels())}.svg')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                patch.save(save_path)   