import torch
import numpy as np
from scipy import ndimage


def complex_to_angle(field: torch.Tensor):
    """
    field: 2 x render_height x render_width
    """
    return torch.atan2(field[1], field[0])


def angle_to_complex(field: torch.Tensor):
    """
    field: 2 x render_height x render_width
    """
    return torch.stack([torch.cos(field), torch.sin(field)])


def rotate_component(field, degrees):
    """
    field: 2 x render_height x render_width
    """
    radians = -degrees * np.pi / 180
    field_new = np.empty_like(field)
    field_new[0] = field[0] * np.cos(radians) - field[1] * np.sin(radians)
    field_new[1] = field[1] * np.cos(radians) + field[0] * np.sin(radians)
    return field_new


def rotate(u, v, raster, degrees):
    raster_new = ndimage.rotate(raster, degrees, axes=(-2, -1), cval=255.)
    u_new = ndimage.rotate(np.nan_to_num(u), degrees, axes=(-2, -1), cval=np.nan)
    v_new = ndimage.rotate(np.nan_to_num(v), degrees, axes=(-2, -1), cval=np.nan)
    u_new, v_new = rotate_component(u_new, degrees), rotate_component(v_new, degrees)
    u_new[:, raster_new == 255.] = np.nan
    v_new[:, raster_new == 255.] = np.nan
    return u_new, v_new, raster_new


def uv_to_c0c2(u, v):
    c_0 = (u ** 2) * (v ** 2)
    c_2 = -(u ** 2 + v ** 2)
    return c_0, c_2


def c0c2_to_uv(c_0, c_2):
    u_squared = -0.5 * (c_2 + (c_2 ** 2 - 4 * c_0) ** 0.5)
    v_squared = -0.5 * (c_2 - (c_2 ** 2 - 4 * c_0) ** 0.5)
    u = u_squared ** 0.5
    v = v_squared ** 0.5
    return u, v




