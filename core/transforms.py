__author__ = 'Ruslan N. Kosarev'

import numpy as np
from math import sin, cos, pi
import tensorflow as tf


# ======================================================================================================================
def spheric_to_cartesian(r_phi_theta):
    """Convert spheric coordinates into cartesion
    0 < phi < 2pi,
    -pi/2 < theta < pi/2,
    r > 0.
    For example:
    f(r, 0, 0) = (0,0,r),
    f(r, 0, -pi/2) = (r,0,0).
    """
    r, phi, theta = r_phi_theta

    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    z = r * cos(theta)

    return x, y, z


def tf_rotation_matrix(thetax, thetay, thetaz):
    rotation_matrix_x = tf.stack([tf.constant(1.0), tf.constant(0.0), tf.constant(0.0),
                                  tf.constant(0.0), tf.cos(thetax), -tf.sin(thetax),
                                  tf.constant(0.0), tf.sin(thetax), tf.cos(thetax)])

    rotation_matrix_y = tf.stack([tf.cos(thetay), tf.constant(0.0), -tf.sin(thetay),
                                  tf.constant(0.0), tf.constant(1.0), tf.constant(0.0),
                                  tf.sin(thetay), 0, tf.cos(thetay)])

    rotation_matrix_z = tf.stack([tf.cos(thetaz), -tf.sin(thetaz), tf.constant(0.0),
                                  tf.sin(thetaz), tf.cos(thetaz), tf.constant(0.0),
                                  tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)])

    rotation_matrix_x = tf.reshape(rotation_matrix_x, (3, 3))
    rotation_matrix_y = tf.reshape(rotation_matrix_y, (3, 3))
    rotation_matrix_z = tf.reshape(rotation_matrix_z, (3, 3))

    return rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z


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


# ======================================================================================================================
class AffineTransformBase:
    """Base class for transforms."""
    def __init__(self):
        self._parameters = None
        self._center = None
        self._matrix = None
        self._offset = None

    def __repr__(self):
        """Representation of transform"""
        info = ('{}\n'.format(self.__class__.__name__) +
                'number of parameters {}\n'.format(self.number_of_parameters)
                )
        return info

    # parameters
    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        # if len(parameters) != self.number_of_parameters:
        #     raise ValueError('wrong array length to set transform parameters')
        self._parameters = parameters

    # number of parameters
    @property
    def number_of_parameters(self):
        return self._parameters.shape[0]

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        self._center = center

    @property
    def offset(self):
        return self._offset

    @property
    def matrix(self):
        return self._matrix

    @property
    def translation(self):
        raise NotImplementedError


# ======================================================================================================================
# Translation transform
class TranslationTransform(AffineTransformBase):
    """
    The serialization of the optimizable parameters is an array of 3 elements.
    The 3 parameters defines the translation in each dimension.
    """
    def __init__(self, parameters=None):
        super().__init__()
        if parameters is None:
            parameters = np.array([0, 0, 0])
        self._parameters = tf.Variable(parameters, dtype=tf.float32)

    @property
    def translation(self):
        return self.parameters

    def transform(self, points):
        points = points + self.translation
        return points


# ======================================================================================================================
# Euler3D transform
class Euler3DTransform(AffineTransformBase):
    """
    The serialization of the optimizable parameters is an array of 6 elements. The first 3 represents three euler angle
    of rotation respectively about the X, Y and Z axis. The last 3 parameters defines the translation in each dimension.

    """
    def __init__(self):
        super().__init__()
        self._parameters = tf.Variable(np.array([0, 0, 0, 0, 0, 0]), dtype=tf.float32)

    @property
    def angles(self):
        return self.parameters[0], self.parameters[1], self.parameters[2]

    @property
    def translation(self):
        return tf.stack([self.parameters[3], self.parameters[4], self.parameters[5]])

    def transform(self, points, parameters):
        self._parameters = parameters

        self._matrix = rotation_matrix(self.parameters[0], self.parameters[1], self.parameters[2])
        self._offset = self.translation + self.center - tf.expand_dims(self.center, 0) @ tf.transpose(self.matrix)

        return points @ tf.transpose(self.matrix) + self.offset


# ======================================================================================================================
# similarity Euler 3D transform
class SimilarityEuler3DTransform(AffineTransformBase):
    """
    The serialization of the optimizable parameters is an array of 7 elements. The first 3 represents three euler angle
    of rotation respectively about the X, Y and Z axis. The next 3 parameters defines the translation in each dimension.
    The last parameter defines the isotropic scaling.
    """
    def __init__(self):
        super().__init__()
        self._parameters = tf.Variable(np.array([0, 0, 0, 0, 0, 0, 1]), dtype=tf.float32)

    def __repr__(self):
        """Representation of transform"""
        info = (
                '{}'.format(super().__repr__()) +
                'scale {}\n'.format(self.scale)
        )
        return info

    @property
    def scale(self):
        return self._parameters[6]

    @property
    def translation(self):
        return tf.stack([self.parameters[3], self.parameters[4], self.parameters[5]])

    def transform(self, points):
        self._matrix = self.scale * rotation_matrix(self.parameters[0], self.parameters[1], self.parameters[2])
        self._offset = self.translation + self.center - tf.expand_dims(self.center, 0) @ tf.transpose(self.matrix)

        return points @ tf.transpose(self.matrix) + self.offset
