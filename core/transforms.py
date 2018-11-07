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
    rotation_matrix_x = tf.stack([tf.constant(1.0),tf.constant(0.0),tf.constant(0.0),
                               tf.constant(0.0),tf.cos(thetax), -tf.sin(thetax),
                               tf.constant(0.0),tf.sin(thetax), tf.cos(thetax)])
    rotation_matrix_y = tf.stack([
                          tf.cos(thetay),tf.constant(0.0), -tf.sin(thetay),
                          tf.constant(0.0),tf.constant(1.0),tf.constant(0.0),
                          tf.sin(thetay),0, tf.cos(thetay)])


    rotation_matrix_z = tf.stack([
                              tf.cos(thetaz), -tf.sin(thetaz),tf.constant(0.0),
                              tf.sin(thetaz), tf.cos(thetaz),tf.constant(0.0),
                              tf.constant(0.0),tf.constant(0.0),tf.constant(1.0)])
    rotation_matrix_x = tf.reshape(rotation_matrix_x, (3,3))
    rotation_matrix_y = tf.reshape(rotation_matrix_y, (3,3))
    rotation_matrix_z = tf.reshape(rotation_matrix_z, (3,3))

    return rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z


# ======================================================================================================================
class AffineTransformBase:
    """Base class for transforms."""
    def __init__(self):
        self._number_of_parameters = None
        self._parameters = None

        self._matrix = None
        self._offset = None
        self._center = None

    def __repr__(self):
        """Representation of transform"""
        info = (
            '{}\n'.format(self.__class__.__name__) +
            'number of parameters {}\n'.format(self.number_of_parameters) +
            'parameters {}\n'.format(self.parameters.tolist()) +
            'matrix {}\n'.format(self.matrix.tolist()) +
            'offset {}\n'.format(self.offset.tolist()) +
            'center {}\n'.format(self.center.tolist())
        )
        return info

    # parameters
    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        if len(parameters) != self.number_of_parameters:
            raise ValueError('wrong array length to set transform parameters')
        self._parameters = parameters
        self._compute_matrix()
        self._compute_offset()

    # number of parameters
    @property
    def number_of_parameters(self):
        return self._number_of_parameters

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

    def _compute_matrix(self):
        raise NotImplementedError

    def _compute_offset(self):
        self._offset = self.translation + self.center - self.matrix @ self.center

    def transform_points(self, points, shape=None):
        if shape is None:
            return points @ self.matrix.T + self.offset
        else:
            return points.reshape(shape) @ self.matrix.T + self.offset

    def compute_jacobian_with_respect_to_parameters(self, point):
        raise NotImplementedError

    def compute_jacobian_with_respect_to_position(self, point):
        return self._matrix


# ======================================================================================================================
# Euler3D transform
class Euler3DTransform(AffineTransformBase):
    """
    The serialization of the optimizable parameters is an array of 6 elements. The first 3 represents three euler angle
    of rotation respectively about the X, Y and Z axis. The last 3 parameters defines the translation in each dimension.

    """
    def __init__(self):
        super().__init__()
        default_parameters = np.array([0, 0, 0, 0, 0, 0])
        self._number_of_parameters = len(default_parameters)
        self._parameters = default_parameters
        self.center = np.zeros(3)
        self._matrix = np.eye(3)  # self._compute_matrix()
        self._offset = np.zeros(3)  # self._compute_offset()

    def __repr__(self):
        """Representation of transform"""
        info = (
            '{}'.format(super().__repr__()) +
            'angles {}\n'.format(np.array(self.angles).tolist()) +
            'translation {}\n'.format(self.translation.tolist())
        )
        return info

    @property
    def angles(self):
        return self.parameters[0], self.parameters[1], self.parameters[2]

    @property
    def translation(self):
        return np.array([self.parameters[3], self.parameters[4], self.parameters[5]])

    def _compute_matrix(self):

        angle_x, angle_y, angle_z = self.angles

        cos_x = np.cos(angle_x)
        sin_x = np.sin(angle_x)
        cos_y = np.cos(angle_y)
        sin_y = np.sin(angle_y)
        cos_z = np.cos(angle_z)
        sin_z = np.sin(angle_z)

        rotation_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        rotation_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        rotation_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

        self._matrix = rotation_z @ rotation_x @ rotation_y


# ======================================================================================================================
# similarity Euler 3D transform
class SimilarityEuler3DTransform(Euler3DTransform):
    """
    The serialization of the optimizable parameters is an array of 7 elements. The first 3 represents three euler angle
    of rotation respectively about the X, Y and Z axis. The next 3 parameters defines the translation in each dimension.
    The last parameter defines the isotropic scaling.
    """
    def __init__(self):
        super().__init__()
        default_parameters = np.array([0, 0, 0, 0, 0, 0, 1])
        self._number_of_parameters = len(default_parameters)
        self._parameters = default_parameters

        self._matrix = np.eye(3)
        self._offset = np.zeros(3)
        self._jacobian = np.zeros([3, self.number_of_parameters])

    def __repr__(self):
        """Representation of transform"""
        info = (
                '{}'.format(super().__repr__()) +
                'scale {}\n'.format(self.scale)
        )
        return info

    @property
    def scale(self):
        return self.parameters[-1]

    def _compute_matrix(self):
        super()._compute_matrix()
        self._matrix = self.scale * self.matrix

    @Euler3DTransform.parameters.setter
    def parameters(self, parameters):
        Euler3DTransform.parameters.fset(self, parameters)
        if self.scale <= 0:
            raise ValueError('wrong value for scale in array of parameters')

    def compute_jacobian_with_respect_to_parameters(self, point):

        p = point - self.center

        # rotation
        angle_x, angle_y, angle_z = self.angles

        cx = np.cos(angle_x)
        sx = np.sin(angle_x)
        cy = np.cos(angle_y)
        sy = np.sin(angle_y)
        cz = np.cos(angle_z)
        sz = np.sin(angle_z)

        self._jacobian[0][0] = (-sz * cx * sy) * p[0] + (sz * sx) * p[1] + (sz * cx * cy) * p[2]
        self._jacobian[1][0] = (cz * cx * sy) * p[0] + (-cz * sx) * p[1] + (-cz * cx * cy) * p[2]
        self._jacobian[2][0] = (sx * sy) * p[0] + cx * p[1] + (-sx * cy) * p[2]

        self._jacobian[0][1] = (-cz * sy - sz * sx * cy) * p[0] + (cz * cy - sz * sx * sy) * p[2]
        self._jacobian[1][1] = (-sz * sy + cz * sx * cy) * p[0] + (sz * cy + cz * sx * sy) * p[2]
        self._jacobian[2][1] = (-cx * cy) * p[0] + (-cx * sy) * p[2]

        self._jacobian[0][2] = (-sz * cy - cz * sx * sy) * p[0] + (-cz * cx) * p[1] + (-sz * sy + cz * sx * cy) * p[2]
        self._jacobian[1][2] = (cz * cy - sz * sx * sy) * p[0] + (-sz * cx) * p[1] + (cz * sy + sz * sx * cy) * p[2]
        self._jacobian[2][2] = 0

        # translation
        self._jacobian[0, 3] = 1
        self._jacobian[1, 4] = 1
        self._jacobian[2, 5] = 1

        # scaling
        self._jacobian[:, 6] = self.matrix @ p

        return self._jacobian


# ======================================================================================================================
# similarity Euler 3D transform
class ProjectionSimilarityEuler3DTransform(SimilarityEuler3DTransform):
    """
    The serialization of the optimizable parameters is an array of 6 elements. The first 3 represents three euler angle
    of rotation respectively about the X, Y and Z axis. The next 2 parameters defines the translation in x and y dimensions.
    The last parameter defines the isotropic scaling.
    """
    def __init__(self):
        super().__init__()
        default_parameters = np.array([0, 0, 0, 0, 0, 1])
        self._number_of_parameters = len(default_parameters)
        self._parameters = default_parameters

        self._projector = np.array([[1, 0, 0], [0, 1, 0]])
        self._matrix = np.array([[1, 0, 0], [0, 1, 0]])
        self._offset = np.zeros(2)

        self._jacobian = np.zeros([2, self.number_of_parameters])

    def __repr__(self):
        """Representation of transform"""
        info = (
                '{}'.format(super().__repr__()) +
                'projector {}\n'.format(self.projector.tolist())
        )
        return info

    @SimilarityEuler3DTransform.parameters.setter
    def parameters(self, parameters):
        SimilarityEuler3DTransform.parameters.fset(self, parameters)
        if self.scale <= 0:
            raise ValueError('wrong value for scale in array of parameters')

    @property
    def translation(self):
        return np.array([self.parameters[3], self.parameters[4]])

    @property
    def projector(self):
        return self._projector

    def compute_jacobian_with_respect_to_parameters(self, point):
        p = point - self.center

        # rotation
        angle_x, angle_y, angle_z = self.angles

        cx = np.cos(angle_x)
        sx = np.sin(angle_x)
        cy = np.cos(angle_y)
        sy = np.sin(angle_y)
        cz = np.cos(angle_z)
        sz = np.sin(angle_z)

        self._jacobian[0][0] = (-sz * cx * sy) * p[0] + (sz * sx) * p[1] + (sz * cx * cy) * p[2]
        self._jacobian[1][0] = (cz * cx * sy) * p[0] + (-cz * sx) * p[1] + (-cz * cx * cy) * p[2]

        self._jacobian[0][1] = (-cz * sy - sz * sx * cy) * p[0] + (cz * cy - sz * sx * sy) * p[2]
        self._jacobian[1][1] = (-sz * sy + cz * sx * cy) * p[0] + (sz * cy + cz * sx * sy) * p[2]

        self._jacobian[0][2] = (-sz * cy - cz * sx * sy) * p[0] + (-cz * cx) * p[1] + (-sz * sy + cz * sx * cy) * p[2]
        self._jacobian[1][2] = (cz * cy - sz * sx * sy) * p[0] + (-sz * cx) * p[1] + (cz * sy + sz * sx * cy) * p[2]

        # translation
        self._jacobian[0, 3] = 1
        self._jacobian[1, 4] = 1

        # scaling
        self._jacobian[:, 5] = self.matrix @ p

        return self._jacobian

    def _compute_matrix(self):
        super()._compute_matrix()
        self._matrix = self.projector @ self.matrix

    def _compute_offset(self):
        self._offset = self.translation + self._projector @ self.center - self.matrix @ self.center
