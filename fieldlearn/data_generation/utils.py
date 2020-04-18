def transform_slopes(u, v):
    c_0 = (u ** 2) * (v ** 2)
    c_2 = -(u ** 2 + v ** 2)
    return c_0, c_2


def inverse_transfom_slopes(c_0, c_2):
    u_squared = -0.5 * (c_2 + (c_2 ** 2 - 4 * c_0) ** 0.5)
    v_squared = -0.5 * (c_2 - (c_2 ** 2 - 4 * c_0) ** 0.5)
    u = u_squared ** 0.5
    v = v_squared ** 0.5
    return u, v
