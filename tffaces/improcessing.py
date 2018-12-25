__author__ = 'Ruslan N. Kosarev'

import numpy as np
import tensorflow as tf


# ======================================================================================================================
def background(image, value=None):
    alpha = image[:, :, :, 3]
    mask = 1 - alpha
    mask = tf.stack([mask, mask, mask], -1)

    if value is not None:
        return image[:, :, :, :3] + mask * value

    return image[:, :, :, :3]

# ======================================================================================================================
