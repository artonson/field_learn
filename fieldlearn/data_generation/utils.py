import torch


def components_to_angle(field):
    return torch.atan2(field[1], field[0])


def angle_to_components(field):
    return torch.stack([torch.cos(field), torch.sin(field)])


def transform_components(u, v):
    c_0 = (u ** 2) * (v ** 2)
    c_2 = -(u ** 2 + v ** 2)
    return c_0, c_2


def inverse_transform_components(c_0, c_2):
    u_squared = -0.5 * (c_2 + (c_2 ** 2 - 4 * c_0) ** 0.5)
    v_squared = -0.5 * (c_2 - (c_2 ** 2 - 4 * c_0) ** 0.5)
    u = u_squared ** 0.5
    v = v_squared ** 0.5
    return u, v




