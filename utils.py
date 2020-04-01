import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def plot_vector_field(img, field, normalize=True, origin='upper', plot_type='cross'):
    """
    :param img:
    :param field:
    :param origin:
    :param plot_type:
        'cross' — plot two vectors as a cross with center in pixel
        'angle' — plot two vectors as an angle with center in pixel
    :param normalize: plot unit field vectors
    :return:
    """
    plt.imshow(img, cmap='gray', origin=origin)

    assert field.shape[0] == 4  # the field is represented by complex vectors

    for h in range(field.shape[1]):
        for w in range(field.shape[2]):
            c_0 = np.complex(field[0, h, w], field[1, h, w])
            c_2 = np.complex(field[2, h, w], field[3, h, w])
            u, v = inverse_transfom_slopes(c_0, c_2)

            if normalize:
                norm_u = np.linalg.norm(u)
                norm_v = np.linalg.norm(v)
                if norm_u:
                    u /= norm_u
                if norm_v:
                    v /= norm_v

            if plot_type == 'angle':
                plt.arrow(w, h, u.real, u.imag, color='red')
                plt.arrow(w, h, v.real, v.imag, color='brown')

            elif plot_type == 'cross':
                plt.plot((w - u.real / 2, w + u.real / 2), (h - u.imag / 2, h + u.imag / 2), color='red')
                plt.plot((w - v.real / 2, w + v.real / 2), (h - v.imag / 2, h + v.imag / 2), color='brown')

