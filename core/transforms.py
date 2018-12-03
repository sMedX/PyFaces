__author__ = 'Ruslan N. Kosarev'

import numpy as np
import tensorflow as tf


# ======================================================================================================================
def rotation_matrix(angle_x, angle_y, angle_z):
    cos_x = tf.cos(angle_x)
    sin_x = tf.sin(angle_x)
    cos_y = tf.cos(angle_y)
    sin_y = tf.sin(angle_y)
    cos_z = tf.cos(angle_z)
    sin_z = tf.sin(angle_z)

    rotation_x = tf.stack([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    rotation_y = tf.stack([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    rotation_z = tf.stack([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

    matrix = rotation_z @ rotation_x @ rotation_y

    return matrix


def bounds_sigmoid(input, bounds=None):
    if bounds is None:
        return input
    else:
        return bounds * (2*tf.sigmoid(input / bounds) - 1)


# ======================================================================================================================
class AffineTransformBase:
    """Base class for transforms."""
    def __init__(self, scaling=(), initial_parameters=()):
        self._scaling = np.array(scaling)
        self._initial_parameters = np.array(initial_parameters)

        self._variable_parameters = tf.Variable(np.zeros(self.number_of_parameters),
                                                dtype=tf.float32,
                                                name='variable_parameters')

        self._parameters = self.initial_parameters + self.scaling * self.variable_parameters

        self._center = None
        self._matrix = None

    def __repr__(self):
        """Representation of transform"""
        info = ('{}\n'.format(self.__class__.__name__) +
                'number of parameters {}\n'.format(self.number_of_parameters)
                )
        return info

    # parameters
    @property
    def initial_parameters(self):
        return self._initial_parameters

    # parameters
    @property
    def parameters(self):
        return self._parameters

    # variable parameters
    @property
    def variable_parameters(self):
        return self._variable_parameters

    # number of parameters
    @property
    def number_of_parameters(self):
        return len(self.initial_parameters)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        self._center = center

    @property
    def matrix(self):
        return self._matrix

    @property
    def scaling(self):
        return self._scaling

    @property
    def translation(self):
        raise NotImplementedError

    def update(self, input):
        if isinstance(input, tf.Session):
            values = input.run(self.variable_parameters)
            self._variable_parameters = tf.Variable(values, dtype=tf.float32, name='variable_parameters')
            self._parameters = self.initial_parameters + self.scaling * self.variable_parameters


# ======================================================================================================================
# Translation transform
class IdentityTransform(AffineTransformBase):
    def __init__(self):
        super().__init__()

    @property
    def translation(self):
        return self.parameters

    def transform(self, points):
        return points


# ======================================================================================================================
# Translation transform
class TranslationTransform(AffineTransformBase):
    """
    The serialization of the optimizable parameters is an array of 3 elements.
    The 3 parameters defines the translation in each dimension.
    """
    def __init__(self, initial_parameters=(0, 0, 0)):
        super().__init__(scaling=(1, 1, 1), initial_parameters=initial_parameters)

    @property
    def translation(self):
        return self.parameters

    def transform(self, points):
        points = points + self.parameters
        return points


# ======================================================================================================================
# Euler3D transform
class Euler3DTransform(AffineTransformBase):
    """
    The serialization of the optimizable parameters is an array of 6 elements. The first 3 represents three euler angle
    of rotation respectively about the X, Y and Z axis. The last 3 parameters defines the translation in each dimension.

    """
    def __init__(self):
        super().__init__(scaling=(0.1, 0.1, 0.1, 1, 1, 1), initial_parameters=(0, 0, 0, 0, 0, 0))

    @property
    def translation(self):
        return tf.stack([self.parameters[3], self.parameters[4], self.parameters[5]])

    def transform(self, points):

        self._matrix = rotation_matrix(self.parameters[0], self.parameters[1], self.parameters[2])

        return (points - self.center) @ tf.transpose(self.matrix) + self.translation + self.center


# ======================================================================================================================
# similarity Euler 3D transform
class SimilarityEuler3DTransform(AffineTransformBase):
    """
    The serialization of the optimizable parameters is an array of 7 elements. The first 3 represents three euler angle
    of rotation respectively about the X, Y and Z axis. The next 3 parameters defines the translation in each dimension.
    The last parameter defines the isotropic scaling.
    """
    def __init__(self):
        super().__init__(scaling=(0.1, 0.1, 0.1, 1, 1, 1, 0.1), initial_parameters=(0, 0, 0, 0, 0, 0, 1))

    @property
    def translation(self):
        return tf.stack([self.parameters[3], self.parameters[4], self.parameters[5]])

    def transform(self, points):

        self._matrix = self.parameters[6] * rotation_matrix(self.parameters[0], self.parameters[1], self.parameters[2])

        return (points - self.center) @ tf.transpose(self.matrix) + self.translation + self.center
