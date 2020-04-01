import os
import numpy as np
from scipy.ndimage import gaussian_filter
from cairosvg import svg2png
from svgpathtools import svg2paths
from svgpathtools.path import closest_point_in_path, translate, scale
from PIL import Image

def load_png(png_path):
    img = Image.open(png_path)

    if img.mode == 'RGBA':
        new_img = Image.new("RGBA", img.size, "WHITE")  # Create a white rgba background
        new_img.paste(img, (0, 0), img)
        img = new_img

    if img.mode != 'L':
        img = img.convert('L')

    img = np.array(img)
    img = img / 255
    return img


def parse_path_width(path_attrs):
    if 'stroke-width' in path_attrs:
        width = float(path_attrs['stroke-width'])
        return width

    else:
        width = None
        for attr in path_attrs['style'][:-1].split(';'):
            attr = attr.split(':')
            if attr[0] == 'stroke-width':
                width = float(attr[1])
        return width


def distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


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


def img_meshgrid(img):
    xs = np.arange(img.shape[1])
    ys = np.arange(img.shape[0])
    return np.meshgrid(xs, ys)


class BoundingBox:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __repr__(self):
        return f'BoundingBox(xmin={self.xmin}, xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax})'


def join_bounding_boxes(first, second):
    return BoundingBox(
        min(first.xmin, second.xmin),
        max(first.xmax, second.xmax),
        min(first.ymin, second.ymin),
        max(first.ymax, second.ymax)
    )


def find_global_bbox(paths):
    global_bbox = BoundingBox(*paths[0].bbox())
    for path in paths:
        global_bbox = join_bounding_boxes(global_bbox, BoundingBox(*path.bbox()))
    return global_bbox


def find_scale_factor(global_bbox, max_h, max_w, boundary_size):
    xdiff = global_bbox.xmax - global_bbox.xmin
    ydiff = global_bbox.ymax - global_bbox.ymin

    if xdiff > ydiff:
        return max_w / (xdiff + boundary_size)
    else:
        return max_h / (ydiff + boundary_size)


class PVFGenerator:
    # TODO: refactor
    def __init__(self,
                 smoothing_type='gaussian_smoothing',
                 smoothing_coef=3,
                 gaussian_sigma=1,
                 width_tol_const=1,
                 bitmap_tol=0.9,
                 max_h=64,
                 max_w=64,
                 boundary_size=1
                 ):
        """
        Generates PNG + a vector field array for an SVG image

        :param smoothing_type: ['no_smoothing', 'gaussian_smoothing', 'circle_smoothing']
        :param smoothing_coef:
        :param gaussian_sigma:
        :param width_tol_const:
        :param bitmap_tol:
        """
        self.smoothing_type = smoothing_type
        self.smoothing_coef = smoothing_coef
        self.gaussian_sigma = gaussian_sigma
        self.width_tol_const = width_tol_const
        self.bitmap_tol = bitmap_tol
        self.max_h = max_h
        self.max_w = max_w
        self.boundary_size = boundary_size

    def coord_to_t(self, x, y, path, path_width):
        pt = np.complex(x, y)
        dist, t, _ = closest_point_in_path(pt, path)
        path_width = 5
        tol = path_width / 2 + self.width_tol_const
        if dist <= tol:
            return t
        else:
            return -1.0

    @staticmethod
    def tangent_at_t(t, path):
        if t == -1 or path.start == path.end:  # fix to avoid zero paths
            return 0 + 0j
        else:
            return path.unit_tangent(t)

    def normalize_paths(self, paths):
        global_bbox = find_global_bbox(paths)
        scale_factor = find_scale_factor(global_bbox, self.max_h, self.max_w, self.boundary_size)

        normalized_paths = []
        for path in paths:
            normalized_paths.append(
                scale(translate(
                    path, np.complex(-global_bbox.xmin + self.boundary_size, -global_bbox.ymin + self.boundary_size)),
                    scale_factor)
            )
        return normalized_paths

    def generate_tangent_field(self, paths, paths_attr, img):
        # TODO: rewrite on gpu
        xx, yy = img_meshgrid(img)
        tangent_fields = []

        for path, path_attrs in zip(paths, paths_attr):
            path_width = parse_path_width(path_attrs)

            if not path_width:
                continue

            vec_coord_to_t = np.vectorize(lambda x, y: self.coord_to_t(x, y, path, path_width))
            vec_tangent_field = np.vectorize(lambda t: self.tangent_at_t(t, path))
            ts = vec_coord_to_t(xx, yy)
            path_tangent_field = vec_tangent_field(ts)
            path_tangent_field[img > self.bitmap_tol] = 0 + 0j

            tangent_fields.append(path_tangent_field)

        return tangent_fields

    @staticmethod
    def get_intersections(tangent_fields):
        num_prims = len(tangent_fields)
        intersections = []
        for i in range(num_prims):
            for j in range(i):
                intersection_mask = np.logical_and(tangent_fields[i], tangent_fields[j])

                # if intersection exists, we add the parameters
                if np.any(intersection_mask):
                    center = np.argwhere(intersection_mask).mean(axis=0)
                    mean_dist_center = np.argwhere(intersection_mask).std()
                    intersections.append({
                        'mean_tangent_1': tangent_fields[i][intersection_mask].mean(),
                        'mean_tangent_2': tangent_fields[j][intersection_mask].mean(),
                        'mean_h': center[0],
                        'mean_w': center[1],
                        'prim_1': i,
                        'prim_2': j,
                        'mean_dist_center': mean_dist_center,
                    })
        return intersections

    def circle_smoothing(self, point, prim_1, normal_slope_1, intersections):
        # TODO: rewrite more efficiently

        near_intersection = False

        # check if the point is near any intersections
        for intersection in intersections:

            # if it is the necessary primitive
            if intersection['prim_1'] == prim_1 or intersection['prim_2'] == prim_1:

                center_h = intersection['mean_h']
                center_w = intersection['mean_w']
                smoothing_radius = self.smoothing_coef * intersection['mean_dist_center']

                center_dist = distance(a=point, b=(center_h, center_w))
                # if the point is within the ball near intersection
                if center_dist <= smoothing_radius:

                    near_intersection = True
                    alpha = center_dist / smoothing_radius

                    if intersection['prim_1'] == prim_1:
                        tangent_slope_2 = intersection['mean_tangent_2']
                    else:
                        tangent_slope_2 = intersection['mean_tangent_1']

                    slope_2 = tangent_slope_2 * (1 - alpha) + alpha * normal_slope_1
                    break

        # if it is not near intersection, use the tangent slope
        if not near_intersection:
            slope_2 = normal_slope_1

        return slope_2

    def gaussian_smoothing(self, img, field):
        gauss_field = field
        gauss_field[0:2] = gaussian_filter(field[0:2], sigma=self.gaussian_sigma)
        gauss_field[2:] = gaussian_filter(field[2:], sigma=self.gaussian_sigma)

        mask = (img > self.bitmap_tol)
        mask = np.expand_dims(mask, axis=0)
        mask = np.repeat(mask, repeats=4, axis=0)
        gauss_field[mask] = 0
        return gauss_field

    def get_point_field(self, h, w, tangents, intersections):
        # TODO: rewrite more concisely
        # TODO: fix case of 2 and more intersections

        slope_1 = 0
        slope_2 = 0

        # get all point_tangents at the point
        point_tangents = np.full(shape=len(tangents), fill_value=0+0j)
        for i in range(len(tangents)):
            point_tangents[i] = tangents[i][h, w]

        nonnan_tangents = point_tangents[point_tangents != 0 + 0j]
        nonnan_idx = np.argwhere(point_tangents != 0 + 0j)

        # only one primitive
        if len(nonnan_tangents) == 1:
            slope_1 = nonnan_tangents[0]
            prim_1 = nonnan_idx[0]

            normal_slope_1 = slope_1 * (0 + 1j)

            if self.smoothing_type == 'no_smoothing':
                slope_2 = normal_slope_1

            elif self.smoothing_type == 'gaussian_smoothing':
                slope_2 = normal_slope_1

            elif self.smoothing_type == 'circle_smoothing':
                slope_2 = self.circle_smoothing((h, w), prim_1, normal_slope_1, intersections)

        # the intersection point
        elif len(nonnan_tangents) > 1:
            slope_1 = nonnan_tangents[0]
            slope_2 = nonnan_tangents[1]

        # otherwise it is a white pixel

        return slope_1, slope_2

    def generate_vector_field(self, svg_path, png_path=None):
        """
        :param svg_path: path to svg file with image
        :param png_path: path to non-distorted png
        :return:
        """
        paths, paths_attr = svg2paths(svg_path)
        paths = self.normalize_paths(paths)

        if not (png_path and os.path.exists(png_path)):
            png_path = svg_path[:-4] + '.png'
            svg2png(url=svg_path, write_to=png_path, output_height=self.max_h, output_width=self.max_w)

        img = load_png(png_path)

        tangent_fields = self.generate_tangent_field(paths, paths_attr, img)
        intersections = self.get_intersections(tangent_fields)

        vec_get_point_field = np.vectorize(
            lambda h, w: self.get_point_field(h, w, tangents=tangent_fields, intersections=intersections),
            otypes=(np.complex64, np.complex64)
        )
        vec_transform_slopes = np.vectorize(transform_slopes)

        xx, yy = img_meshgrid(img)
        u, v = vec_get_point_field(yy, xx)

        c_0, c_2 = vec_transform_slopes(u, v)

        field = np.zeros(shape=(4, img.shape[0], img.shape[1]), dtype=np.float32)
        field[0], field[1] = c_0.real, c_0.imag
        field[2], field[3] = c_2.real, c_2.imag

        # do gaussian smoothing by the resulting channels
        if self.smoothing_type == 'gaussian_smoothing':
            field = self.gaussian_smoothing(img, field)

        field[:2] /= (np.linalg.norm(field[:2], axis=0) + 1e-5)
        field[2:] /= (np.linalg.norm(field[2:], axis=0) + 1e-5)
        return img, field
