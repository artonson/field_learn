import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from vectran.data.graphics.graphics import VectorImage
from vectran.renderers.cairo import render as cairo_render


def draw_vector_image_skeleton(img: VectorImage, renderer=cairo_render, figscale=0.2):
    # build control commands
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

    # render the image
    raster = img.render(renderer)
    render_height, render_width = raster.shape

    # plot image skeleton
    figsize = np.asarray(raster.shape) * figscale
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


def draw_polyvector_field(u, v, raster, scale=1.3, figscale=0.2, same_color=False):
    figsize = np.asarray(raster.shape) * figscale
    figsize = figsize[1], figsize[0]
    fig = plt.figure(figsize=figsize)
    plt.imshow(raster, cmap='gray')
    args = dict(pivot='middle', headaxislength=0, headlength=0,
                units='xy', angles='xy', scale_units='xy', scale=scale, width=.1)
    qu = plt.quiver(u[0], u[1], color='red', **args)
    v_color = 'red' if same_color else 'darkred'
    qv = plt.quiver(v[0], v[1], color=v_color, **args)
    return fig, qu, qv


def redraw_polyvector_field(qu, qv, u, v):
    qu.set_UVC(u[0], u[1])
    qv.set_UVC(v[0], v[1])


def draw_polyvector_field_dif(u, v, u0, v0, raster, scale=1.3, figscale=0.4, same_color=False):
    """
    u, v    — initial values    (angles)
    u0, v0  — optimized values  (angles)
    """
    figsize = np.asarray(raster.shape) * figscale
    figsize = figsize[1], figsize[0] * 3
    fig, [ax0, ax1, ax2] = plt.subplots(3, 1, figsize=figsize)
    ax0.set_xlabel('Initial')
    ax1.set_xlabel('Smoothed')
    ax2.set_xlabel('Both. Green is initial')

    for ax in [ax0, ax1, ax2]:
        ax.imshow(raster, cmap='gray')
        ax.xaxis.set_label_position('top')
    args = dict(pivot='middle', headaxislength=0, headlength=0,
                units='xy', angles='xy', scale_units='xy', scale=scale, width=.1)

    ax0.quiver(u0[0], u0[1], color='red', **args)
    v_color = 'red' if same_color else 'darkred'
    ax0.quiver(v0[0], v0[1], color=v_color, **args)

    ax2.quiver(u0[0], u0[1], color='green', **args)
    v_color = 'green' if same_color else 'darkgreen'
    ax2.quiver(v0[0], v0[1], color=v_color, **args)

    for ax in [ax1, ax2]:
        ax.quiver(u[0], u[1], color='red', **args)
        v_color = 'red' if same_color else 'darkred'
        ax.quiver(v[0], v[1], color=v_color, **args)

    fig.tight_layout()
    return fig
