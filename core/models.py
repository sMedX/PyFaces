__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import h5py
import numpy as np
import tensorflow as tf
from . import transforms


# data to represent surface
class Representer:
    def __init__(self, filename=None):
        self._filename = filename

        self._points = None

        self._np_cells = None
        self._cells = None

        self.initialize()

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
    def np_cells(self):
        return self._np_cells

    @property
    def cells(self):
        return self._cells

    def initialize(self):
        with h5py.File(self.filename, 'r') as hf:
            self._points = hf['shape/representer/points'].value
            self._np_cells = hf['shape/representer/cells'].value

        self._cells = tf.constant(self._np_cells.T, dtype=tf.int32)


class ModelBase:
    """Base class for model."""
    def __init__(self, filename=None, name=None):
        self._filename = filename
        self._name = name
        self._initialized = False

        self._number_of_used_components = None
        self._np_mean = None
        self._np_basis = None
        self._np_variance = None
        self._np_std = None

        self._mean = None
        self._basis = None
        self._variance = None
        self._std = None

    @property
    def name(self):
        return self._name

    @property
    def mean(self):
        return self._mean

    @property
    def basis(self):
        return self._basis

    @property
    def variance(self):
        return self._variance

    @property
    def std(self):
        return self._std

    @property
    def np_mean(self):
        return self._np_mean

    @property
    def np_basis(self):
        return self._np_basis

    @property
    def np_variance(self):
        return self._np_variance

    @property
    def np_std(self):
        return self._np_std

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename

    def array2tensor(self, array=None):
        if array is None:
            array = np.zeros((1, self.number_of_components))
        return tf.Variable(array, dtype=tf.float32, name=self.name)

    @property
    def number_of_components(self):
        if self._np_basis is None:
            return None
        else:
            return self._np_basis.shape[1]

    @property
    def number_of_used_components(self):
        return self._number_of_used_components

    @number_of_used_components.setter
    def number_of_used_components(self, n):
        self._number_of_used_components = n
        if self._initialized is False:
            return

        if n is None:
            n = self.number_of_components
        else:
            n = max(n, 0)
            n = min(n, self.number_of_components)

        self._number_of_used_components = n
        self._basis = tf.constant(self._np_basis[:, :self._number_of_used_components].T, dtype=tf.float32)
        self._std = tf.constant(self._np_std[:self._number_of_used_components], dtype=tf.float32)

    def initialize(self):
        with h5py.File(self.filename, 'r') as hf:
            self._np_mean = hf[self._name + '/model/mean'].value
            self._np_basis = hf[self._name + '/model/pcaBasis'].value
            self._np_variance = hf[self._name + '/model/pcaVariance'].value
            self._np_std = np.sqrt(self._np_variance)

        if self._number_of_used_components is None:
            self._number_of_used_components = self.number_of_components
        else:
            self._number_of_used_components = max(self.number_of_used_components, 0)
            self._number_of_used_components = min(self.number_of_used_components, self.number_of_components)

        self._mean = tf.constant(self._np_mean, dtype=tf.float32)
        self._basis = tf.constant(self._np_basis[:, :self.number_of_used_components].T, dtype=tf.float32)
        self._std = tf.constant(self._np_std[:self.number_of_used_components], dtype=tf.float32)
        self._initialized = True


# expression model
class ExpressionModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='expression')
        self.initialize()

    def __repr__(self):
        info = (
             '{}\n'.format(self.__class__.__name__) +
             'components {}\n'.format(self._basis.shape)
        )
        return info


# shape model
class ShapeModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='shape')
        # representer and expressions
        self._representer = Representer(filename=self.filename)
        self._expression = ExpressionModel(filename=self.filename)

        # read shape data
        self.initialize()

        # compute center of the shape
        shape = (self.representer.number_of_points, self.representer.dimension)
        self._np_center = np.mean(np.reshape(self.np_mean, shape), axis=0)
        self._center = tf.constant(self._np_center, dtype=tf.float32, name='center')

    def __repr__(self):
        info = (
             '{}\n'.format(self.__class__.__name__) +
             'components {}\n'.format(self._basis.shape) +
             '{}'.format(self._representer.__repr__()) +
             '{}'.format(self._expression.__repr__())
        )
        return info

    @property
    def representer(self):
        return self._representer

    @property
    def expression(self):
        return self._expression

    @property
    def np_center(self):
        return self._np_center

    @property
    def center(self):
        return self._center


# color model
class ColorModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='color')
        self.initialize()

    def __repr__(self):
        info = (
            '{}\n'.format(self.__class__.__name__) +
            'components {}\n'.format(self._basis.shape)
        )
        return info


class FaceModel:
    def __init__(self, filename, landmarks=None):
        self._shape = ShapeModel(filename=filename)
        self._color = ColorModel(filename=filename)
        self._landmarks = landmarks

    def __repr__(self):
        """Representation of FaceModel"""
        info = (
             '{}\n'.format(self.__class__.__name__) +
             '{}'.format(self._shape.__repr__()) +
             '{}'.format(self._color.__repr__())
        )
        return info

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def shape(self):
        return self._shape

    @property
    def color(self):
        return self._color

    def plot(self, step=3):
        shape = (self.shape.representer.number_of_points, self.shape.representer.dimension)
        points = np.reshape(self.shape.np_mean, shape)
        colors = np.reshape(self.color.np_mean, shape)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], c=colors[::step, :], marker='.')
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')
        ax.axis('equal')

        labels = ('x label', 'y label', 'z label')

        def show_points(ax, i, k):
            ax.scatter(points[:, i], points[:, k], color=colors, marker='.')
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[k])
            ax.axis('equal')
            ax.grid(True)

        fig, ax = plt.subplots(1, 3)
        show_points(ax[0], 0, 1)
        show_points(ax[1], 1, 2)
        show_points(ax[2], 0, 2)
        plt.show()

        return


# data to represent surface
class ModelTransform:
    def __init__(self, model, transform=transforms.IdentityTransform()):
        self._model = model

        self._spatial_transform = transform
        self._spatial_transform.center = model.shape.center

        self._parameters = list([self.model.shape.array2tensor(),
                                 self.model.shape.expression.array2tensor(),
                                 self.model.color.array2tensor()])


    @property
    def model(self):
        return self._model

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        self._parameters = parameters

    @property
    def spatial_transform(self):
        return self._spatial_transform

    @spatial_transform.setter
    def spatial_transform(self, transform):
        transform.center = self.model.shape.center
        self._spatial_transform = transform

    def transform(self):
        # transform points
        points = self.model.shape.mean + \
                 self.parameters[0] * self.model.shape.std @ self.model.shape.basis + \
                 self.parameters[1] * self.model.shape.expression.std @ self.model.shape.expression.basis

        points = tf.reshape(points, (self.model.shape.representer.number_of_points, 3))

        # apply spatial transform
        points = self.spatial_transform.transform(points)
        points = tf.expand_dims(points, 0)

        # transform colors
        colors = self.model.color.mean + self.parameters[2] * self.model.color.std @ self.model.color.basis
        colors = tf.reshape(colors, (1, self.model.shape.representer.number_of_points, 3))

        # compute normals
        normals = points

        return points, colors, normals
