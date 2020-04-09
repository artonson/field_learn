import numpy as np
import matplotlib.pyplot as plt

from fieldlearn.data_generation.utils import inverse_transfom_slopes


def plot_vector_field_cross(raster, field, plot_type='cross', figsize=(10, 10)):
    """
    Plots a vector field
    :param raster:
        raster image
    :param field:
        field based on the raster image
    :param plot_type:
        'cross' — plot two vectors as a cross with center in pixel
        'angle' — plot two vectors as an angle with center in pixel
    :param figsize
    :return plt.figure
    """
    fig = plt.figure(figsize=figsize)

    plt.imshow(raster, cmap='gray', origin='upper')

    _, patch_height, patch_width = field.shape
    c_0 = np.zeros((patch_height, patch_width), dtype=np.complex64)
    c_0.real, c_0.imag = field[0], field[1]

    c_2 = np.zeros((patch_height, patch_width), dtype=np.complex64)
    c_2.real, c_2.imag = field[2], field[3]

    u, v = inverse_transfom_slopes(c_0, c_2)

    for h in range(patch_height):
        for w in range(patch_width):

            u_ = u[h, w]
            v_ = v[h, w]

            if plot_type == 'angle':
                plt.arrow(w, h, u_.real, u_.imag, color='red')
                plt.arrow(w, h, v_.real, v_.imag, color='brown')

            elif plot_type == 'cross':
                plt.plot((w - u_.real / 2, w + u_.real / 2), (h - u_.imag / 2, h + u_.imag / 2), color='red')
                plt.plot((w - v_.real / 2, w + v_.real / 2), (h - v_.imag / 2, h + v_.imag / 2), color='brown')

    plt.close()
    return fig


def plot_vector_field_heatmap(field):
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
