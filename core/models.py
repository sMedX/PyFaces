__author__ = 'Ruslan N. Kosarev'

import h5py
import numpy as np
import core.landmarks as landmarks


# data to represent surface
class Representer:
    def __init__(self, filename=None):
        self._filename = filename
        self._cells = None
        self._points = None

    def __repr__(self):
        """Representation of representer"""
        info = (
            '{}\n'.format(self.__class__.__name__) +
            'points {}\n'.format(self.points.shape) +
            'cells {}\n'.format(self.cells.shape)
        )
        return info

    @property
    def filename(self):
        return self._filename

    @property
    def number_of_points(self):
        if self._points is None:
            return None
        else:
            return self._points.shape[1]

    @property
    def dimension(self):
        if self._points is None:
            return None
        else:
            return self._points.shape[0]

    @property
    def points(self):
        return self._points

    @property
    def cells(self):
        return self._cells

    def read(self):
        with h5py.File(self.filename, 'r') as hf:
            self._points = hf['shape/representer/points'].value
            self._cells = hf['shape/representer/cells'].value


class ModelBase:
    """Base class for model."""
    def __init__(self, filename=None):
        self._filename = filename

        self._representer = None
        self._landmarks = None
        self._landmarks_indexes = None

        self._number_of_used_components = None

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename

    @property
    def number_of_parameters(self):
        raise NotImplementedError

    @property
    def number_of_used_components(self):
        return self._number_of_used_components

    @number_of_used_components.setter
    def number_of_used_components(self, components):
        raise NotImplementedError

    @property
    def number_of_components(self):
        raise NotImplementedError

    @property
    def representer(self):
        return self._representer

    @property
    def number_of_landmarks(self):
        if self._landmarks is None:
            return None
        else:
            return len(self._landmarks)

    @property
    def landmarks_indexes(self):
        return self._landmarks_indexes

    @landmarks_indexes.setter
    def landmarks_indexes(self, indexes):
        self._landmarks_indexes = indexes

    def jacobian(self, index):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError


# expression model
class ExpressionModel(ModelBase):
    def __init__(self, filename=None):
        self._mean = None
        self._basis = None
        self._variance = None

        self._landmarks_mean = None
        self._landmarks_basis = None

        super().__init__(filename=filename)

    def __repr__(self):
        """Representation of FaceModel"""
        info = (
             '{}\n'.format(self.__class__.__name__) +
             'components {}\n'.format(self._basis.shape)
        )
        return info

    @property
    def number_of_components(self):
        if self._basis is None:
            return None
        else:
            return self._basis.shape[1]

    @property
    def number_of_parameters(self):
        return self._number_of_used_components

    @property
    def number_of_used_components(self):
        return self._number_of_used_components

    @number_of_used_components.setter
    def number_of_used_components(self, components):
        self._number_of_used_components = components

    def jacobian(self, index):
        index1 = self.representer.dimension * index
        index2 = index1 + self.representer.dimension
        return self._basis[index1:index2, :self.number_of_parameters]

    def initialize(self):
        self._representer = Representer(filename=self.filename)
        self._representer.read()

        with h5py.File(self.filename, 'r') as hf:
            self._mean = hf['expression/model/mean'].value
            self._basis = hf['expression/model/pcaBasis'].value
            self._variance = hf['expression/model/pcaVariance'].value

        self._compute_landmarks_data()
        self._number_of_used_components = self.number_of_components

    def _compute_landmarks_data(self):
        mean = np.reshape(self._mean, [self.representer.number_of_points, self.representer.dimension])
        mean = mean[self._landmarks_indexes]
        self._landmarks_mean = np.reshape(mean, mean.shape[0]*mean.shape[1])

        basis = np.reshape(self._basis, [self.representer.number_of_points, self.representer.dimension, self.number_of_components])
        basis = basis[self._landmarks_indexes]
        self._landmarks_basis = np.reshape(basis, [basis.shape[0]*basis.shape[1], basis.shape[2]])

    def transform(self, parameters):
        if len(parameters) < self.number_of_components:
            raise ValueError('wrong length of parameters')

        return self._mean + self._basis @ parameters[:self.number_of_components]

    def transform_landmarks(self, parameters):
        if len(parameters) < self.number_of_parameters:
            raise ValueError('wrong length of parameters')

        components = self.number_of_used_components
        return self._landmarks_mean + self._landmarks_basis[:, :components] @ parameters[:components]


# shape model
class ShapeModel(ModelBase):
    def __init__(self, filename=None):
        self._expressions = None

        self._center = None
        self._mean = None
        self._basis = None
        self._variance = None

        self._landmarks_mean = None
        self._landmarks_basis = None

        super().__init__(filename=filename)

    def __repr__(self):
        """Representation of shape model"""
        info = (
             '{}\n'.format(self.__class__.__name__) +
             'components {}\n'.format(self._basis.shape) +
             'center {}\n'.format(self.center.tolist()) +
             'landmarks {}\n'.format(len(self.landmarks)) +
             '{}'.format(self._representer.__repr__()) +
             '{}'.format(self._expressions.__repr__())
        )
        return info

    @property
    def expressions(self):
        return self._expressions

    def jacobian(self, index):
        index1 = self.representer.dimension * index
        index2 = index1 + self.representer.dimension
        jac1 = self._basis[index1:index2, :self.number_of_parameters]
        jac2 = self.expressions.jacobian(index)
        return np.concatenate((jac1, jac2), axis=1)

    def initialize(self):
        # read representer
        self._representer = Representer(filename=self.filename)
        self._representer.read()

        # read shape model
        with h5py.File(self.filename, 'r') as hf:
            self._mean = hf['shape/model/mean'].value
            self._basis = hf['shape/model/pcaBasis'].value
            self._variance = hf['shape/model/pcaVariance'].value

        x, y, z = self.xyz
        self._center = np.array([np.mean(x), np.mean(y), np.mean(z)])

        self._landmarks = landmarks.get_list(self.filename)
        self._define_landmarks_indexes()
        self._compute_landmarks_data()
        self._number_of_used_components = self.number_of_components

        # read expressions model
        self._expressions = ExpressionModel(filename=self.filename)
        self._expressions.landmarks_indexes = self._landmarks_indexes
        self._expressions.initialize()

    def _define_landmarks_indexes(self):

        threshold = 10
        shape = [self.representer.number_of_points, self.representer.dimension]

        self._landmarks_indexes = []

        for pair in self._landmarks:
            dist = np.sum(pow(self._mean.reshape(shape) - pair.point, 2), axis=1)
            index = np.argmin(dist)

            if dist[index] < threshold:
                self._landmarks_indexes.append(index)

    def _compute_landmarks_data(self):
        mean = np.reshape(self._mean, [self.representer.number_of_points, self.representer.dimension])
        mean = mean[self._landmarks_indexes]
        self._landmarks_mean = np.reshape(mean, mean.shape[0]*mean.shape[1])

        basis = np.reshape(self._basis, [self.representer.number_of_points, self.representer.dimension, self.number_of_components])
        basis = basis[self._landmarks_indexes]
        self._landmarks_basis = np.reshape(basis, [basis.shape[0]*basis.shape[1], basis.shape[2]])

    @property
    def xyz(self):
        return self._mean[0::3], self._mean[1::3], self._mean[2::3]

    @property
    def number_of_components(self):
        if self._basis is None:
            return None
        else:
            return self._basis.shape[-1]

    @property
    def number_of_parameters(self):
        return sum(self.number_of_used_components)

    @property
    def number_of_used_components(self):
        return self._number_of_used_components, self.expressions.number_of_used_components

    @number_of_used_components.setter
    def number_of_used_components(self, components):
        if components <= self.number_of_components:
            self._number_of_used_components = components
            self.expressions.number_of_used_components = 0
        else:
            self.expressions.number_of_used_components = components - self._number_of_used_components

    @property
    def center(self):
        return self._center

    @property
    def landmarks(self):
        return self._landmarks

    def transform(self, parameters):
        if len(parameters) < self.number_of_parameters:
            raise ValueError('wrong length of parameters')

        points = self._mean + self._basis @ parameters[:self.number_of_components] + self.expressions.transform(parameters[self.number_of_components:])
        return points

    def transform_landmarks(self, parameters):
        if len(parameters) < self.number_of_parameters:
            raise ValueError('wrong length of parameters')

        components = self.number_of_used_components[0]

        points = self._landmarks_mean + \
                 self._landmarks_basis[:, :components] @ parameters[:components] + \
                 self.expressions.transform_landmarks(parameters[components:])

        return points


# color model
class ColorModel(ModelBase):
    def __init__(self, filename=None):
        self._mean = None
        self._basis = None
        self._variance = None
        super().__init__(filename=filename)

    def __repr__(self):
        info = (
            '{}\n'.format(self.__class__.__name__) +
            'components {}\n'.format(self._basis.shape)
        )
        return info

    @property
    def number_of_components(self):
        if self._basis is None:
            return None
        else:
            return self._basis.shape[1]

    @property
    def colors(self):
        return np.reshape(self._mean, [int(len(self._mean)/3), 3])

    def initialize(self):
        with h5py.File(self.filename, 'r') as hf:
            self._mean = np.array(hf['color/model/mean'])
            self._basis = np.array(hf['color/model/pcaBasis'])
            self._variance = np.array(hf['color/model/pcaVariance'])

    def transform(self, parameters):
        raise NotImplementedError


class FaceModel:
    def __init__(self, filename=None):
        self._shape = ShapeModel(filename=filename)
        self._color = ColorModel(filename=filename)

    def __repr__(self):
        """Representation of FaceModel"""
        info = (
             '{}\n'.format(self.__class__.__name__) +
             '{}'.format(self._shape.__repr__()) +
             '{}'.format(self._color.__repr__())
        )
        return info

    @property
    def shape(self):
        return self._shape

    @property
    def color(self):
        return self._color

    def initialize(self):
        self._shape.initialize()
        self._color.initialize()

    def plot(self, step=5):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        x, y, z = self.shape.xyz

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(x[::step], y[::step], z[::step], c=self.color.colors[::step, :], marker='.')
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')
        ax.axis('equal')

        points = landmarks.to_array(self.shape.landmarks)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=50)

        plt.show()

        fig, ax = plt.subplots(1, 3)
        ax[0].scatter(x, y, color=self.color.colors, marker='.')
        ax[0].set_xlabel('x label')
        ax[0].set_ylabel('y label')
        ax[0].scatter(points[:, 0], points[:, 1], c='r', marker='o', s=50)
        ax[0].axis('equal')
        ax[0].grid(True)

        ax[1].scatter(y, z, color=self.color.colors, marker='.')
        ax[1].set_xlabel('y label')
        ax[1].set_ylabel('z label')
        ax[1].scatter(points[:, 1], points[:, 2], c='r', marker='o', s=50)
        ax[1].axis('equal')
        ax[1].grid(True)

        ax[2].scatter(x, z, color=self.color.colors, marker='.')
        ax[2].set_xlabel('x label')
        ax[2].set_ylabel('z label')
        ax[2].scatter(points[:, 0], points[:, 2], c='r', marker='o', s=50)
        ax[2].axis('equal')
        ax[2].grid(True)
        plt.show()
        return
