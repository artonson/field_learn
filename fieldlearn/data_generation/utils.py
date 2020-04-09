from collections import namedtuple


BoundingBox = namedtuple('BoundingBox', ['xmin', 'xmax', 'ymin', 'ymax'])


def join_bboxes(first, second):
    return BoundingBox(
        min(first.xmin, second.xmin),
        max(first.xmax, second.xmax),
        min(first.ymin, second.ymin),
        max(first.ymax, second.ymax)
    )


def find_global_bbox(paths):
    global_bbox = BoundingBox(*paths[0].bbox())
    for path in paths:
        global_bbox = join_bboxes(global_bbox, BoundingBox(*path.bbox()))
    return global_bbox


def find_scale_factor(global_bbox, height, width, boundary_size):
    xdiff = global_bbox.xmax - global_bbox.xmin
    ydiff = global_bbox.ymax - global_bbox.ymin

    if xdiff > ydiff:
        return width / (xdiff + boundary_size)
    else:
        return height / (ydiff + boundary_size)


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
