import torch

from fieldlearn.utils import complex_to_angle_batch


def calc_iou(outputs: torch.Tensor, labels: torch.Tensor, tol=1e-5):
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))
    iou = (intersection + tol) / (union + tol)
    return iou.mean()


def angle_to_orientation_similarity(angle):
    return (1 + torch.cos(torch.abs(angle))) / 2


def orientation_similarity_to_angle(sim):
    return torch.acos(2 * sim - 1)


def calc_pixelwise_orientation_similarity(u, u0):
    true_angle = complex_to_angle_batch(u)
    pred_angle = complex_to_angle_batch(u0)
    return angle_to_orientation_similarity(true_angle - pred_angle)


def calc_flipped_orientation_similarity(u, u0, v0):
    black_pixels = (u != 0).all(dim=1)

    diff_uu0 = torch.abs(complex_to_angle_batch(u - u0))
    diff_uv0 = torch.abs(complex_to_angle_batch(u - v0))
    uu0_less_uv0 = (diff_uu0 < diff_uv0).float()

    sim_uu0 = calc_pixelwise_orientation_similarity(u, u0)
    sim_uv0 = calc_pixelwise_orientation_similarity(u, v0)

    u_sim = 0.5 * (sim_uu0[black_pixels] * uu0_less_uv0[black_pixels]).mean() \
            + 0.5 * (sim_uv0[black_pixels] * (1 - uu0_less_uv0)[black_pixels]).mean()
    return u_sim


def calc_orientation_similarity(target, pred, with_flips=False):
    u, v = target[:, :2], target[:, 2:]
    u0, v0 = pred[:, :2], pred[:, 2:]

    if with_flips:
        u_sim = calc_flipped_orientation_similarity(u, u0, v0)
        v_sim = calc_flipped_orientation_similarity(v, v0, u0)
        return u_sim, v_sim
    else:
        black_pixels = (u != 0).all(dim=1)
        u_sim = calc_pixelwise_orientation_similarity(u, u0)[black_pixels].mean()
        v_sim = calc_pixelwise_orientation_similarity(v, v0)[black_pixels].mean()
        return u_sim, v_sim
