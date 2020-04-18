import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from fieldlearn.data_generation.utils import transform_slopes
from vectran.data.graphics.graphics import VectorImage
from vectran.renderers.cairo import render as cairo_render
from vectran.optimization.optimizer.primitive_aligner import prepare_pixel_coordinates
from vectran.optimization.primitives.quadratic_bezier_tensor import QuadraticBezierTensor


def parse_bezier_paths(paths):
    control_points = []
    widths = []
    for path in paths:
        for curve in path:
            control_points.append([((p.real, p.imag),) for p in curve.bpoints()])
            widths.append(float(path.width))

    # change axes:
    #   num_primitives x num_control_points (3) x 1 x num_dimensions
    #   ->
    #   num_control_points x 1 x num_dimensions x num_primitives
    control_points = np.array(control_points, dtype=np.float32).transpose(1, 2, 3, 0)
    #  num_primitives x 1 -> 1 x 1 x num_primitives
    widths = np.array(widths, dtype=np.float32).reshape(1, 1, -1)
    return control_points, widths


class PolyVectorFieldGenerator:
    # TODO: rewrite in terms of VectorImage
    def __init__(self,
                 smoothing_type='gaussian_smoothing',
                 gaussian_sigma=1,
                 ):
        """
        Generates PNG + a vector field array for an SVG image

        :param smoothing_type: ['no_smoothing', 'gaussian_smoothing']
        :param gaussian_sigma:
        :param spatial_dims_n: number of dims in field
        """
        self.smoothing_type = smoothing_type
        self.gaussian_sigma = gaussian_sigma
        self.spatial_dims_n = 2
        self.cardano_tol = 1e-2
        self.similar_direction_tol = 1e-2
        self.division_eps = 1e-3

    def gaussian_smoothing(self, field, mask):
        smoothed_field = field
        smoothed_field[0:2] = gaussian_filter(field[0:2], sigma=self.gaussian_sigma)
        smoothed_field[2:] = gaussian_filter(field[2:], sigma=self.gaussian_sigma)
        smoothed_field[0:2][mask] = 0
        smoothed_field[2:][mask] = 0
        return smoothed_field

    def generate_vector_field(self, img: VectorImage, renderer=cairo_render, device=torch.device('cuda')):
        raster = img.render(renderer)

        points, widths = parse_bezier_paths(img.paths)
        num_primitives = points.shape[-1]

        beziers = QuadraticBezierTensor(
            points[0],
            points[1],
            points[2],
            widths,
            dtype=torch.float32, device=device)

        renders = beziers.render_with_cairo_each(raster.shape[0], raster.shape[1])
        # raster = (255 - torch.clamp(renders.sum(dim=1)[0] * 255, 0, 255)).type(torch.uint8).numpy()

        pixel_coords = prepare_pixel_coordinates(
            torch.empty([1, raster.shape[0], raster.shape[1]], dtype=torch.float32)).to(device)
        canonical_x, canonical_y = beziers.calculate_canonical_coordinates(
            pixel_coords, tol=self.cardano_tol, division_epsilon=self.division_eps)

        vector_field = beziers.get_vector_field_at(canonical_y)
        vector_field = vector_field[0].transpose(1, 0)
        vector_field = vector_field.reshape(num_primitives, self.spatial_dims_n, raster.shape[0], raster.shape[1])

        mask = (renders.to(device) > 0)
        x = vector_field[:, 0]
        y = vector_field[:, 1]

        field_x = x.where(mask, x.new_full([], np.nan)).detach().cpu().numpy()
        field_y = y.where(mask, y.new_full([], np.nan)).detach().cpu().numpy()

        field = np.zeros_like(field_x, dtype=np.complex64)
        field.real[~np.isnan(field_x)] = field_x[~np.isnan(field_x)]
        field.imag[~np.isnan(field_y)] = field_y[~np.isnan(field_y)]

        result = np.full((self.spatial_dims_n, raster.shape[0], raster.shape[1]), 0+0j, dtype=np.complex64)

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

        field = np.zeros(shape=(4, raster.shape[0], raster.shape[1]), dtype=np.float32)
        field[0], field[1] = c_0.real, c_0.imag
        field[2], field[3] = c_2.real, c_2.imag

        if self.smoothing_type == 'gaussian_smoothing':
            field = self.gaussian_smoothing(field, mask)

        field[:2] /= (np.linalg.norm(field[:2], axis=0) + 1e-5)
        field[2:] /= (np.linalg.norm(field[2:], axis=0) + 1e-5)

        return field
