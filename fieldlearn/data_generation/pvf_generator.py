import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from svgpathtools import svg2paths
from svgpathtools.path import translate, scale
from fieldlearn.data_generation.utils import find_global_bbox, find_scale_factor, transform_slopes, inverse_transfom_slopes
from vectran.optimization.optimizer.primitive_aligner import prepare_pixel_coordinates
from vectran.optimization.primitives.quadratic_bezier_tensor import QuadraticBezierTensor


def parse_bezier_paths(paths):
    control_points = []
    for path in paths:
        for curve in path:
            control_points.append([((p.real, p.imag),) for p in curve])
    control_points = np.array(control_points).transpose(1, 2, 3, 0)
    return control_points


def parse_bezier_path_widths(paths_attr):
    widths = []
    for attr in paths_attr:
        width_float = float(attr['stroke-width'].replace('px', ''))
        widths.append(width_float)

    widths = np.array(widths)
    widths = widths.reshape(1, 1, -1)
    return widths


class PVFGenerator:
    def __init__(self,
                 patch_height=64,
                 patch_width=64,
                 smoothing_type='gaussian_smoothing',
                 gaussian_sigma=1,
                 boundary_width=1,
                 spatial_dims_n=2
                 ):
        """
        Generates PNG + a vector field array for an SVG image

        :param smoothing_type: ['no_smoothing', 'gaussian_smoothing']
        :param gaussian_sigma:
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.boundary_width = boundary_width
        self.smoothing_type = smoothing_type
        self.gaussian_sigma = gaussian_sigma
        self.spatial_dims_n = spatial_dims_n
        self.cardano_tol = 1e-2
        self.cardano_division_eps = 1e-3
        self.similar_direction_tol = 1e-2

    def normalize_paths(self, paths):
        global_bbox = find_global_bbox(paths)
        scale_factor = find_scale_factor(global_bbox, self.patch_height, self.patch_width, self.boundary_width)

        normalized_paths = []
        for path in paths:
            normalized_paths.append(
                scale(translate(
                    path, np.complex(-global_bbox.xmin + self.boundary_width, -global_bbox.ymin + self.boundary_width)),
                    scale_factor)
            )
        return normalized_paths

    def gaussian_smoothing(self, field, mask):
        smoothed_field = field
        smoothed_field[0:2] = gaussian_filter(field[0:2], sigma=self.gaussian_sigma)
        smoothed_field[2:] = gaussian_filter(field[2:], sigma=self.gaussian_sigma)
        smoothed_field[0:2][mask] = 0
        smoothed_field[2:][mask] = 0
        return smoothed_field

    def generate_vector_field(self, svg_path, device=torch.device('cuda')):
        paths, paths_attr = svg2paths(svg_path)
        paths = self.normalize_paths(paths)

        points = parse_bezier_paths(paths)
        num_primitives = points.shape[-1]

        widths = parse_bezier_path_widths(paths_attr)

        beziers = QuadraticBezierTensor(
            points[0],
            points[1],
            points[2],
            widths,
            dtype=torch.float32, device=device)

        renders = beziers.render_with_cairo_each(self.patch_width, self.patch_height)
        raster = (255 - torch.clamp(renders.sum(dim=1)[0] * 255, 0, 255)).type(torch.uint8).numpy()

        pixel_coords = prepare_pixel_coordinates(
            torch.empty([1, self.patch_height, self.patch_width], dtype=torch.float32)).to(device)
        canonical_x, canonical_y = beziers.calculate_canonical_coordinates(
            pixel_coords, tol=self.cardano_tol, division_epsilon=self.cardano_division_eps)

        vector_field = beziers.get_vector_field_at(canonical_y)
        vector_field = vector_field[0].transpose(1, 0)
        vector_field = vector_field.reshape(num_primitives, self.spatial_dims_n, self.patch_height, self.patch_width)

        mask = (renders.to(device) > 0)
        x = vector_field[:, 0]
        y = vector_field[:, 1]

        field_x = x.where(mask, x.new_full([], np.nan)).detach().cpu().numpy()
        field_y = y.where(mask, y.new_full([], np.nan)).detach().cpu().numpy()

        field = np.zeros_like(field_x, dtype=np.complex64)
        field.real[~np.isnan(field_x)] = field_x[~np.isnan(field_x)]
        field.imag[~np.isnan(field_y)] = field_y[~np.isnan(field_y)]

        result = np.full((self.spatial_dims_n, self.patch_height, self.patch_width), 0+0j, dtype=np.complex64)

        for primitive_idx in range(num_primitives):
            tangent_field = field[0][primitive_idx]

            mask_first_comp = np.isclose(result[0], 0+0j) & ~np.isclose(tangent_field, 0+0j)
            mask_second_comp = ~np.isclose(result[1], 0+0j) & ~np.isclose(tangent_field, 0+0j)

            result[0][mask_first_comp] = tangent_field[mask_first_comp].copy()
            result[1][mask_first_comp] = tangent_field[mask_first_comp].copy() * 1j

            mask_is_similar = (np.abs(result[0] - tangent_field) < self.similar_direction_tol) | \
                              (np.abs(result[0] + tangent_field) < self.similar_direction_tol)

            result[1][~mask_is_similar & mask_second_comp] = tangent_field[mask_second_comp & ~mask_is_similar].copy()

        u, v = result
        mask = np.isclose(result, 0+0j)
        c_0, c_2 = transform_slopes(u, v)

        field = np.zeros(shape=(4, self.patch_height, self.patch_width), dtype=np.float32)
        field[0], field[1] = c_0.real, c_0.imag
        field[2], field[3] = c_2.real, c_2.imag

        if self.smoothing_type == 'gaussian_smoothing':
            field = self.gaussian_smoothing(field, mask)

        field[:2] /= (np.linalg.norm(field[:2], axis=0) + 1e-5)
        field[2:] /= (np.linalg.norm(field[2:], axis=0) + 1e-5)

        return raster, field
