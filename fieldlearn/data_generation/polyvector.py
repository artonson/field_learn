import torch
import numpy as np
from typing import Callable, Tuple
from vectran.data.graphics.graphics import VectorImage
from vectran.renderers.cairo import render as cairo_render
from vectran.optimization.optimizer.primitive_aligner import prepare_pixel_coordinates
from vectran.optimization.primitives.quadratic_bezier_tensor import QuadraticBezierTensor
from fieldlearn.utils import complex_to_angle, angle_to_complex
from fieldlearn.data_generation.smoothing import loss_function


def tangent_fields_for_beziers(control_points, widths, raster, device, cardano_tol=1e-2, division_eps=1e-3):
    """
    :param control_points:
        tensor of control points for quadratic bezier curves
        num_primitives x num_control_points x num_dimensions
    :param widths:
        tensor of width for quadratic bezier curves
        num_primitives x 1
    :param raster:
        rendered svg image
    :param device:
    :param cardano_tol:
    :param division_eps:

    :return: tangent_fields:
        tensor with tangent fields for each primitive
        num_primitives x 2 x render_height x render_width
    """
    num_primitives, num_control_points, num_dimensions = control_points.shape

    # change axes:
    #   num_primitives x num_control_points (3) x num_dimensions (2)
    #   ->
    #   num_control_points x 1 x num_dimensions x num_primitives
    control_points = control_points.transpose(1, 2, 0).reshape((num_control_points, 1, num_dimensions, num_primitives))

    #  num_primitives x 1 -> 1 x 1 x num_primitives
    widths = widths.reshape(1, 1, num_primitives)

    beziers = QuadraticBezierTensor(
        control_points[0],
        control_points[1],
        control_points[2],
        widths,
        dtype=torch.float32, device=device)

    renders = beziers.render_with_cairo_each(height=raster.shape[0], width=raster.shape[1])[0]

    pixel_coords = prepare_pixel_coordinates(
        torch.empty([1, raster.shape[0], raster.shape[1]], dtype=torch.float32)).to(device)
    canonical_x, canonical_y = beziers.calculate_canonical_coordinates(
        pixel_coords, tol=cardano_tol, division_epsilon=division_eps)

    tangent_fields = beziers.get_vector_field_at(canonical_y)
    tangent_fields = tangent_fields[0].transpose(1, 0)
    tangent_fields = tangent_fields.reshape(num_primitives, 2, raster.shape[0], raster.shape[1])

    mask = (renders.to(device) > 0)
    x = tangent_fields[:, 0]
    y = tangent_fields[:, 1]

    tangent_fields[:, 0] = x.where(mask, x.new_full([], np.nan))
    tangent_fields[:, 1] = y.where(mask, y.new_full([], np.nan))
    return tangent_fields


def field_from_tangent(tangent_fields, raster, device, similar_direction_tol=0.9):
    """
    :param tangent_fields:
        tensor with tangent fields for each primitive
        num_primitives x 2 x render_height x render_width
    :param raster:
        rendered svg image
    :param device:
    :param similar_direction_tol:
        used for filtering overlapping primitives
        takes values from 0 to 1
        if for some primitives holds that abs(cos(tangent_1, tangent_2)) > similar_direction_tol,
            then v = normal_1
            else v = tangent_2

    :return:
        u, v — first and second components of a polyvector field (unit complex vectors)
        u, v: 2 x render_height x render_width

        component_1 = u[0] + i * u[1] = cos(alpha_1) + i * sin(alpha_1)
        component_2 = v[0] + i * v[1] = cos(alpha_2) + i * sin(alpha_2)
    """

    num_primitives = tangent_fields.shape[0]
    u = torch.full((2, raster.shape[0], raster.shape[1]), np.nan, dtype=torch.float32).to(device)
    v = torch.full((2, raster.shape[0], raster.shape[1]), np.nan, dtype=torch.float32).to(device)

    for primitive_idx in range(num_primitives):
        mask_first_comp = torch.isnan(u).all(dim=0) & ~torch.isnan(tangent_fields[primitive_idx]).all(dim=0)
        mask_second_comp = ~torch.isnan(v).all(dim=0) & ~torch.isnan(tangent_fields[primitive_idx]).all(dim=0)

        u[:, mask_first_comp] = tangent_fields[primitive_idx, :, mask_first_comp]

        # shift the complex number by 90 degrees:
        # cos(alpha) + i * sin(alpha) -> sin(alpha) - i * cos(alpha)
        v[0, mask_first_comp] = tangent_fields[primitive_idx, 1, mask_first_comp]
        v[1, mask_first_comp] = -tangent_fields[primitive_idx, 0, mask_first_comp]

        # find if abs(cos(tangent_1, tangent_2)) > similar_direction_tol
        mask_is_similar = (u * tangent_fields[primitive_idx]).sum(dim=0).abs() > similar_direction_tol

        v[:, ~mask_is_similar & mask_second_comp] = tangent_fields[primitive_idx, :,
                                                    ~mask_is_similar & mask_second_comp]

    return u, v


def compute_field(img: VectorImage,
                  smoothing_fn: Callable[
                                 [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]] = None,
                  device=torch.device('cuda'),
                  alignment_tol=1e-2, division_eps=1e-3, similar_direction_tol=0.9):
    """
    Renders a VectorImage and computes polyvector field based on the render

    :param img:
        VectorImage for an svg file
    :param smoothing_fn:
        function used for field smoothing after initial construction
    :param alignment_tol:
        used in calculation of canonical coordinates for tangent fields
    :param division_eps:
    :param similar_direction_tol:
        used for filtering overlapping primitives
        takes values from 0 to 1
        if for some primitives holds that abs(cos(tangent_1, tangent_2)) > similar_direction_tol,
            then v = normal_1
            else v = tangent_2
    :param device:

    :return:
        u, v — first and second components of a polyvector field (unit complex vectors)
        u, v: 2 x render_height x render_width

        component_1 = u[0] + i * u[1] = cos(alpha_1) + i * sin(alpha_1)
        component_2 = v[0] + i * v[1] = cos(alpha_2) + i * sin(alpha_2)
    """
    control_points = []
    widths = []
    for path in img.paths:
        for curve in path:
            control_points.append([(p.real, p.imag) for p in curve.bpoints()])
            widths.append(float(path.width))

    control_points = np.array(control_points, dtype=np.float32)
    widths = np.array(widths, dtype=np.float32)

    raster = img.render(cairo_render)

    tangent_fields = tangent_fields_for_beziers(
        control_points, widths, raster, device, alignment_tol, division_eps)

    u, v = field_from_tangent(
        tangent_fields, raster, device, similar_direction_tol)

    if not smoothing_fn:
        return u, v

    u0, v0 = smoothing_fn(u, v)
    return u0, v0


def smooth_field(u, v, num_iters=100, fidelity_w=0.4, lr=0.1, device=torch.device('cuda')):
    u0 = complex_to_angle(u)
    v0 = complex_to_angle(v)

    u0 = u0.to(device)
    v0 = v0.to(device)

    u = u0.clone().requires_grad_()
    v = v0.clone().requires_grad_()
    optimizer = torch.optim.Adam([u, v], lr=lr)

    for i in range(num_iters):
        loss = loss_function(u, v, u0, v0, fidelity_w=fidelity_w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return angle_to_complex(u0), angle_to_complex(v0)
