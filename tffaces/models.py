__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import h5py
import numpy as np
import tensorflow as tf
from . import transforms
from . import tfSession


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
            '{} '.format(self.__class__.__name__) +
            'points {}, cells {}\n'.format(self.points.shape, self.cells.shape)
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

        with h5py.File(self.filename, 'r') as hf:
            self._np_mean = hf[self._name + '/model/mean'].value
            self._np_basis = hf[self._name + '/model/pcaBasis'].value
            self._np_variance = hf[self._name + '/model/pcaVariance'].value
            self._np_std = np.sqrt(self._np_variance)

        self._mean = tf.constant(self._np_mean, dtype=tf.float32)
        self._basis = tf.constant(self._np_basis.T, dtype=tf.float32)
        self._std = tf.constant(self._np_std, dtype=tf.float32)

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

    @property
    def number_of_components(self):
        return self._np_basis.shape[1]


# expression model
class ExpressionModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='expression')

    def __repr__(self):
        info = (
             '{} {}\n'.format(self.__class__.__name__, self._basis.shape)
        )
        return info


# shape model
class ShapeModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='shape')
        # representer and expressions
        self._representer = Representer(filename=self.filename)
        self._expression = ExpressionModel(filename=self.filename)

        # compute center of the shape
        shape = (self.representer.number_of_points, self.representer.dimension)
        self._np_center = np.mean(np.reshape(self.np_mean, shape), axis=0)
        self._center = tf.constant(self._np_center, dtype=tf.float32, name='center')

    def __repr__(self):
        info = (
             '{} {}\n'.format(self.__class__.__name__, self._basis.shape) +
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

    @property
    def points(self):
        shape = (self.representer.number_of_points, self.representer.dimension)
        return np.reshape(self.np_mean, shape)

    @property
    def cells(self):
        return np.transpose(self.representer.np_cells)


# color model
class ColorModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename, name='color')

    def __repr__(self):
        info = (
            '{} {}'.format(self.__class__.__name__, self._basis.shape)
        )
        return info

    @property
    def colors(self):
        shape = (int(len(self.np_mean)/3), 3)
        return np.reshape(self.np_mean, shape)


class FaceModel:
    def __init__(self, filename, landmarks=None):
        self.filename = filename

        self._shape = ShapeModel(filename=filename)
        self._color = ColorModel(filename=filename)
        self._landmarks = landmarks

        self.number_of_components = (self.shape.number_of_components,
                                     self.shape.expression.number_of_components,
                                     self.color.number_of_components)

    def __repr__(self):
        """Representation of FaceModel"""
        info = (
             '{}\n'.format(self.__class__.__name__) +
             '{}'.format(self._shape.__repr__()) +
             '{}'.format(self._color.__repr__())
        )
        return info

    @property
    def number_of_points(self):
        return self.shape.representer.number_of_points

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
    def __init__(self, model, transform=transforms.IdentityTransform(), bounds=None):
        self._model = model

        self._spatial_transform = transform
        self._spatial_transform.center = model.shape.center

        # initialize bounds
        if bounds is None:
            bounds = (None, None, None)

        if isinstance(bounds, (list, tuple)) is False:
            raise TypeError('bounds must be list, or tuple')
        elif len(bounds) != 3:
            raise TypeError('bounds must have length 3')

        self.bounds = bounds

        # initialize parameters
        number_of_components = self.model.number_of_components
        self.np_variable_parameters = [np.zeros(number_of_components[0]),
                                       np.zeros(number_of_components[1]),
                                       np.zeros(number_of_components[2])]

        # initialize tensor parameters
        self.variable_parameters = [None, None, None]
        self.parameters = [None, None, None]
        self.update(self.np_variable_parameters)

    @property
    def model(self):
        return self._model

    @property
    def spatial_transform(self):
        return self._spatial_transform

    @property
    def number_of_used_components(self):
        return [components.shape[0] for components in self.variable_parameters]

    def transform(self):

        components = self.number_of_used_components

        # apply shape transform
        points = self.model.shape.mean + \
                 self.parameters[0] * self.model.shape.std[:components[0]] @ self.model.shape.basis[:components[0], :] + \
                 self.parameters[1] * self.model.shape.expression.std[:components[1]] @ self.model.shape.expression.basis[:components[1], :]

        points = tf.reshape(points, (self.model.shape.representer.number_of_points, 3))

        # apply spatial transform
        points = self.spatial_transform.transform(points)
        points = tf.expand_dims(points, 0)

        # apply color transform
        colors = self.model.color.mean + \
                 self.parameters[2] * self.model.color.std[:components[2]] @ self.model.color.basis[:components[2], :]
        colors = tf.reshape(colors, (1, self.model.shape.representer.number_of_points, 3))

        # compute normals
        normals = points

        return points, colors, normals

    def update(self, input):
        # update variable parameters
        if isinstance(input, tf.Session):
            for i, parameters in enumerate(self.variable_parameters):
                value = input.run(parameters)
                self.np_variable_parameters[i][:len(value)] = value
                self.variable_parameters[i] = tf.Variable(value, dtype=tf.float32)

        elif isinstance(input, (list, tuple)):
            for i, value in enumerate(input):
                if isinstance(value, np.ndarray):
                    self.np_variable_parameters[i][:len(value)] = value
                    self.variable_parameters[i] = tf.Variable(value, dtype=tf.float32)
                elif value is not None:
                    raise TypeError('value in the input list must be numpy.ndarray or None')
        else:
            raise TypeError('input must be tf.Session, list, or tuple')

        # initialize model parameters (set bounds and expand dims)
        self.parameters = list()

        for values, bounds in zip(self.variable_parameters, self.bounds):
            values = transforms.bounds_sigmoid(values, bounds)
            self.parameters.append(tf.expand_dims(values, axis=0))

    def set_number_of_components(self, portions):

        components_list = list()
        for value, portion in zip(self.model.number_of_components, portions):
            if portion is None:
                components_list.append(value)
            else:
                if portion < 0 or portion > 1:
                    raise ValueError('input value must be in the range [0, 1]')
                components_list.append(round(portion*value))

        # shape_components = round(portion * (self.model.shape.number_of_components +
        #                                     self.model.shape.expression.number_of_components))
        #
        # variance = np.sort(np.concatenate((self.model.shape.np_variance,
        #                                    self.model.shape.expression.np_variance)))[::-1]
        #
        # try:
        #     value_of_variance = variance[shape_components]
        # except IndexError:
        #     value_of_variance = variance[-1] - 1
        #
        # number_of_components = (np.where(self.model.shape.np_variance > value_of_variance)[0][-1] + 1,
        #                         np.where(self.model.shape.expression.np_variance > value_of_variance)[0][-1] + 1,
        #                         round(portion * self.model.number_of_components[2]))

        parameters = list()

        for variable_parameters, components in zip(self.np_variable_parameters, components_list):
            parameters.append(variable_parameters[:components])

        self.update(parameters)
