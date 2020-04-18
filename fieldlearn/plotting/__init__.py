import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from vectran.data.graphics.graphics import VectorImage
from vectran.renderers.cairo import render as cairo_render
from fieldlearn.data_generation.utils import inverse_transfom_slopes

DEFAULT_QUIVEROPTS = dict(pivot='middle', units='xy',
                          headlength=0, headwidth=1,
                          scale=1.1, linewidth=1, width=0.1)


def visualize_vector_image(img: VectorImage, renderer=cairo_render, figsize=(10, 10)):
    raster = img.render(renderer)
    render_height, render_width = raster.shape

    join_points = []
    control_points = []
    control_commands = []
    for path in img.paths:
        for curve in path:
            control_points.extend([(p.real, p.imag) for p in curve.bpoints()])
            join_points.append(control_points[-1])
            control_points.append(control_points[-1])
            control_commands.extend([mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3, mpath.Path.MOVETO])

    control_points.append(control_points[-1])
    control_commands.append(mpath.Path.CLOSEPOLY)

    fig, ax = plt.subplots(figsize=figsize)
    pp = mpatches.PathPatch(mpath.Path(control_points, control_commands), fc='none', linewidth=2.5, edgecolor='#FFC11E')

    x, y = zip(*join_points)
    ax.imshow(raster, cmap='gray', origin='upper')
    ax.plot(x, y, color='#FFC11E', marker='o', markersize=5, markeredgecolor='black', linestyle='')
    ax.add_patch(pp)
    ax.set_xlim(0, render_width)
    ax.set_ylim(render_height, 0)
    plt.close()
    return fig


def visualize_vector_field_cross(raster, field, figsize=(10, 10)):
    """
    Plots a vector field with cross stitches

    :param raster: raster image
    :param field: field based on the raster image
    :param figsize
    :return plt.figure
    """
    _, patch_height, patch_width = field.shape
    c_0 = np.zeros((patch_height, patch_width), dtype=np.complex64)
    c_0.real, c_0.imag = field[0], field[1]

    c_2 = np.zeros((patch_height, patch_width), dtype=np.complex64)
    c_2.real, c_2.imag = field[2], field[3]

    u, v = inverse_transfom_slopes(c_0, c_2)

    h_range = np.arange(patch_height)
    w_range = np.arange(patch_width)

    fig = plt.figure(figsize=figsize)
    plt.imshow(raster, cmap='gray', origin='upper')
    plt.quiver(h_range, w_range, u.real, u.imag, color='red', **DEFAULT_QUIVEROPTS)
    plt.quiver(h_range, w_range, v.real, v.imag, color='brown', **DEFAULT_QUIVEROPTS)
    plt.close()
    return fig


def visualize_vector_field_heatmap(field):
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 10))
    ax1.imshow(field[0])
    ax1.set_title('component 1, real part')
    ax2.imshow(field[1])
    ax2.set_title('component 1, imag part')
    ax3.imshow(field[2])
    ax3.set_title('component 2, real part')
    ax4.imshow(field[3])
    ax4.set_title('component 2, imag part')
    plt.close()
    return fig
