import torch


def neighbor_consistency(u, u_ref, v, v_ref):
    u = torch.stack(neighbor_dif(u, u_ref))
    del u_ref
    v = torch.stack(neighbor_dif(v, v_ref))
    del v_ref
    return consistency(u) + consistency(v)


def consistency(dif):
    dif = dif.where(torch.isfinite(dif), dif.new_zeros([]))
    dif = dif.sin().pow(2)
    return dif


def neighbor_dif(f1, f2):
    r"""
    Stencil:
    . x .
    x o x
    . x .

    Order of returned differences:
    . 0 .
    2 . 3
    . 1 .
    """
    pad = lambda t, p: torch.nn.functional.pad(t[None, None], p, mode='replicate')[0, 0]
    return pad(f2[:-1], [0, 0, 1, 0]) - f1, \
           pad(f2[1:], [0, 0, 0, 1]) - f1, \
           pad(f2[:, :-1], [1, 0, 0, 0]) - f1, \
           pad(f2[:, 1:], [0, 1, 0, 0]) - f1


def loss_function(u, v, u0, v0, fidelity_w=0.4):
    """
    u, v    — initial values    (angles)
    u0, v0  — optimized values  (angles)
    """
    # fidelity
    fidelity = consistency(u - u0.data) + consistency(v - v0.data)
    del u0, v0
    fidelity = fidelity.mean()

    # self-consistency
    uuvv = neighbor_consistency(u, u.data, v, v.data)
    uvvu = neighbor_consistency(u, v.data, v, u.data)
    sc = torch.min(uuvv, uvvu)
    del uuvv, uvvu
    sc = sc.sum(dim=0).mean()

    return sc * (1 - fidelity_w) + fidelity * fidelity_w
