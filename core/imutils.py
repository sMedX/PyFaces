__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt


def imshow(input):
    """
    show list of images
    :param input:
    """
    if isinstance(input, (list, tuple)):
        fig, axes = plt.subplots(1, len(input))
        for img, ax in zip(input, axes):
            ax.imshow(img)
    else:
        plt.imshow(input)
        plt.show()

    plt.show()
