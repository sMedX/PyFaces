__author__ = 'Ruslan N. Kosarev'

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

dpifig = 250


# ======================================================================================================================
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


# ======================================================================================================================
def savefig(file=None, dpi=dpifig):
    if file is not None:
        plt.savefig(file, dpi=dpi)
        print('\nfigure has been saved to the file', os.path.abspath(file))


def imshowdiff(img1, img2, show=True, file=None):
    """

    :param img1:
    :param img2:
    :param show:
    :param file:
    :return:
    """

    fig, axes = plt.subplots(2, 2)
    axes[0][0].imshow(img1)
    axes[0][1].imshow(img2)
    axes[1][0].imshow(img1 - img2)
    axes[1][1].imshow(img2 - img1)

    savefig(file=file)

    if show is True:
        plt.show()


# ======================================================================================================================
def read(filename, width=None):
    """

    :param filename:
    :param width:
    :return:
    """
    image = cv2.imread(filename)

    if width is not None:
        height = int(width*image.shape[0]/image.shape[1])
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)/255

    return image


