__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
import cv2
import numpy as np


# ======================================================================================================================
def range_image(image):
    pass


def imshow(input, show=True):
    """

    :param input:
    :param show:
    :return:
    """
    if isinstance(input, (list, tuple)):
        fig, axes = plt.subplots(1, len(input))
        for img, ax in zip(input, axes):
            ax.imshow(img)
    else:
        plt.imshow(input)

    if show is True:
        plt.show()


def imshowdiff(img1, img2, show=True):
    """

    :param img1:
    :param img2:
    :param show:
    :return:
    """

    fig, axes = plt.subplots(2, 2)
    axes[0][0].imshow(img1)
    axes[0][1].imshow(img2)
    axes[1][0].imshow(img1 - img2)
    axes[1][1].imshow(img2 - img1)

    if show is True:
        plt.show()


# ======================================================================================================================
def read(filename, scale=1, show=False):
    """

    :param filename:
    :param scale:
    :param show:
    :return:
    """
    image = cv2.imread(filename)
    height = int(scale*image.shape[0])
    width = int(scale*image.shape[1])
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)/255

    if show is True:
        plt.imshow(image)
        plt.show()

    return image


