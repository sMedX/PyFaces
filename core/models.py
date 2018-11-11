__author__ = 'Ruslan N. Kosarev'

import h5py
import numpy as np
import core.landmarks as landmarks
import tensorflow as tf


def normalize_basis(basis, variance):
    # for i in range(basis.shape[1]):
    #     factor = np.sqrt(variance[i])/np.linalg.norm(basis[:, i])
    #     basis[:, i] *= factor
    return basis


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
    def __init__(self, filename=None):
        self._np_mean = None
        self._np_basis = None
        self._np_variance = None
        self._np_std = None

        self._mean = None
        self._basis = None
        self._variance = None
        self._std = None

        self._filename = filename
        self._representer = None

        self._landmarks = None
        self._landmarks_indexes = None

        self._number_of_used_components = None

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

    @property
    def representer(self):
        return self._representer

    @property
    def landmarks(self):
        return self._landmarks

    @landmarks.setter
    def landmarks(self, landmarks):
        self._landmarks = landmarks

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


# expression model
class ExpressionModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename)
        self._landmarks_mean = None
        self._landmarks_basis = None

    def __repr__(self):
        """Representation of FaceModel"""
        info = (
             '{}\n'.format(self.__class__.__name__) +
             'components {}\n'.format(self._basis.shape)
        )
        return info

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
        self._representer.initialize()

        with h5py.File(self.filename, 'r') as hf:
            self._mean = hf['expression/model/mean'].value
            self._basis = hf['expression/model/pcaBasis'].value
            self._variance = hf['expression/model/pcaVariance'].value

        #self._basis = normalize_basis(self._basis, self._variance)
        #self._compute_landmarks_data()
        #self._number_of_used_components = self.number_of_components

        self._mean = tf.constant(self._mean, dtype=tf.float32)
        self._basis = tf.constant(self._basis.T, dtype=tf.float32)
        self._std = tf.constant(self._variance, dtype=tf.float32)

    def _compute_landmarks_data(self):
        if self._landmarks is None:
            return

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

    def transform_landmarks(self, parameters, index=None):
        if len(parameters) < self.number_of_parameters:
            raise ValueError('wrong length of parameters')

        components = self.number_of_used_components
        return self._landmarks_mean + self._landmarks_basis[:, :components] @ parameters[:components]


# shape model
class ShapeModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename)
        self._expressions = None
        self._center = None
        self._landmarks_mean = None
        self._landmarks_basis = None

    def __repr__(self):
        """Representation of shape model"""
        info = (
             '{}\n'.format(self.__class__.__name__) +
             'components {}\n'.format(self._basis.shape) +
             'center {}\n'.format(self.center.tolist()) +
             '{}'.format(self._representer.__repr__()) +
             '{}'.format(self._expressions.__repr__())
        )
        return info

    @property
    def expressions(self):
        return self._expressions

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

    def jacobian(self, index):
        index1 = self.representer.dimension * index
        index2 = index1 + self.representer.dimension
        jac1 = self._basis[index1:index2, :self.number_of_parameters]
        jac2 = self.expressions.jacobian(index)
        return np.concatenate((jac1, jac2), axis=1)

    def initialize(self):
        # read representer
        self._representer = Representer(filename=self.filename)
        self._representer.initialize()

        # read shape model
        with h5py.File(self.filename, 'r') as hf:
            self._np_mean = hf['shape/model/mean'].value
            self._np_basis = hf['shape/model/pcaBasis'].value
            self._np_variance = hf['shape/model/pcaVariance'].value
            self._np_std = np.sqrt(self._np_variance)

        shape = np.reshape(self._np_mean, (self.representer.number_of_points, self.representer.dimension))
        self._center = np.mean(shape, axis=0)

        # self._basis = normalize_basis(self._basis, self._variance)
        self._mean = tf.constant(self._np_mean, dtype=tf.float32)
        self._basis = tf.constant(self._np_basis.T, dtype=tf.float32)
        self._std = tf.constant(self._np_std, dtype=tf.float32)

        self._define_landmarks_indexes()
        self._compute_landmarks_data()
        # self._number_of_used_components = self.number_of_components

        # read expressions model
        self._expressions = ExpressionModel(filename=self.filename)
        self._expressions.landmarks_indexes = self._landmarks_indexes
        self._expressions.initialize()

    def _define_landmarks_indexes(self):

        if self._landmarks is None:
            return

        threshold = 10
        shape = [self.representer.number_of_points, self.representer.dimension]

        self._landmarks_indexes = []

        for pair in self._landmarks:
            dist = np.sum(pow(self._mean.reshape(shape) - pair.point, 2), axis=1)
            index = np.argmin(dist)

            if dist[index] < threshold:
                self._landmarks_indexes.append(index)

    def _compute_landmarks_data(self):
        if self._landmarks is None:
            return

        mean = np.reshape(self._mean, [self.representer.number_of_points, self.representer.dimension])
        mean = mean[self._landmarks_indexes]
        self._landmarks_mean = np.reshape(mean, mean.shape[0]*mean.shape[1])

        basis = np.reshape(self._basis, [self.representer.number_of_points, self.representer.dimension, self.number_of_components])
        basis = basis[self._landmarks_indexes]
        self._landmarks_basis = np.reshape(basis, [basis.shape[0]*basis.shape[1], basis.shape[2]])

    def transform(self, parameters):
        if len(parameters) < self.number_of_parameters:
            raise ValueError('wrong length of parameters')

        points = self._mean + self._basis @ parameters[:self.number_of_components] + self.expressions.transform(parameters[self.number_of_components:])
        return points

    def transform_landmarks(self, parameters, as2darray=True):
        if len(parameters) < self.number_of_parameters:
            raise ValueError('wrong length of parameters')

        components = self.number_of_used_components[0]

        points = self._landmarks_mean + \
                 self._landmarks_basis[:, :components] @ parameters[:components] + \
                 self.expressions.transform_landmarks(parameters[components:])

        if as2darray:
            return points.reshape([self.number_of_landmarks, self.representer.dimension])
        else:
            return points


# color model
class ColorModel(ModelBase):
    def __init__(self, filename=None):
        super().__init__(filename=filename)

    def __repr__(self):
        info = (
            '{}\n'.format(self.__class__.__name__) +
            'components {}\n'.format(self._basis.shape)
        )
        return info

    def initialize(self):
        with h5py.File(self.filename, 'r') as hf:
            self._np_mean = np.array(hf['color/model/mean'])
            self._np_basis = np.array(hf['color/model/pcaBasis'])
            self._np_variance = np.array(hf['color/model/pcaVariance'])
            self._np_std = np.sqrt(self._np_variance)

        self._mean = tf.constant(self._np_mean, dtype=tf.float32)
        self._basis = tf.constant(self._np_basis.T, dtype=tf.float32)
        self._std = tf.constant(self._np_std, dtype=tf.float32)


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
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        shape = (self.shape.representer.number_of_points, self.shape.representer.dimension)
        points = np.reshape(self.shape.np_mean, shape)
        colors = np.reshape(self.color.np_mean, shape)

        #points = landmarks.to_array(self.shape.landmarks)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(points[::step, 0], points[::step, 1], points[::step, 2], c=colors[::step, :], marker='.')
        # if points is not None:
        #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=50)
        ax.set_xlabel('x label')
        ax.set_ylabel('y label')
        ax.set_zlabel('z label')
        ax.axis('equal')

        labels = ('x label', 'y label', 'z label')

        def show_landmarks(ax, i, k):
            ax.scatter(points[:, i], points[:, k], color=colors, marker='.')
            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[k])
            ax.axis('equal')
            ax.grid(True)

            # if points is not None:
            #     ax.scatter(points[:, i], points[:, k], c='r', marker='.')
            #     for landmark in self.shape.landmarks:
            #         if landmark.weight == 1:
            #             color = 'green'
            #         else:
            #             color = 'blue'
            #         ax.text(landmark.point[i] + 1, landmark.point[k] + 1, '{}'.format(landmark.index), color=color)

        fig, ax = plt.subplots(1, 3)
        show_landmarks(ax[0], 0, 1)
        show_landmarks(ax[1], 1, 2)
        show_landmarks(ax[2], 0, 2)
        plt.show()

        return

    @property
    def default_parameters(self):

        params0 = np.zeros((1, self.shape.number_of_components))
        params0 = tf.Variable(params0, dtype=tf.float32, name='shape')

        params1 = np.zeros((1, self.shape.expressions.number_of_components))
        params1 = tf.Variable(params1, dtype=tf.float32, name='expressions')

        params2 = np.zeros((1, self.color.number_of_components))
        params2 = tf.Variable(params2, dtype=tf.float32, name='color')

        return params0, params1, params2


# data to represent surface
class ShapeModelTransform:
    def __init__(self, model, transform):
        self._model = model

        transform.center = model.center
        self._transform = transform

    @property
    def number_of_parameters(self):
        return self._model.number_of_parameters + self._transform.number_of_parameters

    def transform_landmarks(self, parameters):

        # transform landmarks
        points = self._model.transform_landmarks(parameters)

        # apply spatial transform to landmarks
        self._transform.parameters = parameters[self._model.number_of_parameters:]
        points = self._transform.transform_points(points)

        return points

    def compute_jacobian_with_respect_to_parameters(self, index, parameters):

        # apply model transform to point with index
        model_transformed_point = self._model.transform_landmarks(parameters, index)

        # apply spatial transform to points
        # self._transform.parameters = parameters[self._model.number_of_parameters:]
        # transformed_point = self._transform.transform_points(model_transformed_point)

        #model_derivatives = np.zeros(self._model.number_of_parameters)
        #spatial_derivatives = np.zeros(self._transform.number_of_parameters)

        # compute derivatives with respect to model parameters
        jacobian1 = self._transform.compute_jacobian_with_respect_to_position(model_transformed_point) @ self._model.jacobian(index)

        # compute derivatives with respect to spatial transform parameters
        jacobian2 = self._transform.compute_jacobian_with_respect_to_parameters(model_transformed_point)

        return np.concatenate((jacobian1, jacobian2), axis=1)


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
                 params[1] * self.model.shape.expressions.std @ self.model.shape.expressions.basis

        colors = self.model.color.mean + \
                 params[2] * self.model.color.std @ self.model.color.basis

        # normalize and reshape
        points = points / self.scale

        colors = tf.reshape(colors, (1, self.model.shape.representer.number_of_points, 3))
        points = tf.reshape(points, (1, self.model.shape.representer.number_of_points, 3))
        normals = points

        return points, colors, normals
