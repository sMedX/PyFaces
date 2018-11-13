__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import h5py
import numpy as np
import core.landmarks as landmarks
import tensorflow as tf


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

    def initialize(self):
        with h5py.File(self.filename, 'r') as hf:
            self._points = hf['shape/representer/points'].value
            self._cells = hf['shape/representer/cells'].value


class ModelBase:
    """Base class for model."""
    def __init__(self, filename=None, name=None):
        self._filename = filename
        self._name = name

        self._np_mean = None
        self._np_basis = None
        self._np_variance = None
        self._np_std = None

        self._mean = None
        self._basis = None
        self._variance = None
        self._std = None

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

    @property
    def number_of_components(self):
        if self._basis is None:
            return None
        else:
            return self._basis.shape[0]

    def initialize(self):
        with h5py.File(self.filename, 'r') as hf:
            self._np_mean = hf[self._name + '/model/mean'].value
            self._np_basis = hf[self._name + '/model/pcaBasis'].value
            self._np_variance = hf[self._name + '/model/pcaVariance'].value
            self._np_std = np.sqrt(self._np_variance)

        self._mean = tf.constant(self._np_mean, dtype=tf.float32)
        self._basis = tf.constant(self._np_basis.T, dtype=tf.float32)
        self._std = tf.constant(self._np_std, dtype=tf.float32)


# expression model
class ExpressionModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='expression')

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
        self._representer = Representer(filename=self.filename)
        self._expression = ExpressionModel(filename=self.filename)

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

    def initialize(self):
        super().initialize()
        self._representer.initialize()
        self._expression.initialize()


# color model
class ColorModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='color')

    def __repr__(self):
        info = (
            '{}\n'.format(self.__class__.__name__) +
            'components {}\n'.format(self._basis.shape)
        )
        return info


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

    @property
    def default_parameters(self):

        params0 = np.zeros((1, self.shape.number_of_components))
        params0 = tf.Variable(params0, dtype=tf.float32, name='shape')

        params1 = np.zeros((1, self.shape.expression.number_of_components))
        params1 = tf.Variable(params1, dtype=tf.float32, name='expressions')

        params2 = np.zeros((1, self.color.number_of_components))
        params2 = tf.Variable(params2, dtype=tf.float32, name='color')

        return params0, params1, params2


# data to represent surface
class ModelTransform:
    def __init__(self, model=None, transform=None, scale=100):
        self._model = model
        self._transform = transform
        self._scale = scale

    @property
    def model(self):
        return self._model

    @property
    def scale(self):
        return self._scale

    def transform(self, params):
        points = self.model.shape.mean + \
                 params[0] * self.model.shape.std @ self.model.shape.basis + \
                 params[1] * self.model.shape.expression.std @ self.model.shape.expression.basis

        colors = self.model.color.mean + \
                 params[2] * self.model.color.std @ self.model.color.basis

        # normalize and reshape
        points = points / self.scale

        colors = tf.reshape(colors, (1, self.model.shape.representer.number_of_points, 3))
        points = tf.reshape(points, (1, self.model.shape.representer.number_of_points, 3))
        normals = points

        return points, colors, normals
