__author__ = 'Ruslan N. Kosarev'

import numpy as np


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

    def transform_points(self, points, shape):
        return points.reshape(shape) @ self.matrix.T + self.offset


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

        self._matrix = np.eye(3)   # self._compute_matrix()
        self._offset = np.zeros(3)  # self._compute_offset()

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


    def jacobian(self, parameters):
        self.parameters = parameters


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

    def _compute_matrix(self):
        super()._compute_matrix()
        self._matrix = self.projector @ self.matrix

    def _compute_offset(self):
        self._offset = self.translation + self._projector @ self.center - self.matrix @ self.center


# ======================================================================================================================
if __name__ == '__main__':

    t = Euler3DTransform()
    t.parameters = np.zeros(6)
    print(t)

    t = SimilarityEuler3DTransform()
    t.parameters = np.array([0, 0, 0, 0, 0, 0, 1])
    print(t)

    t = ProjectionSimilarityEuler3DTransform()
    t.parameters = np.array([0, 0, 0, 0, 0, 1])
    print(t)
