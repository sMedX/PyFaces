__author__ = 'Ruslan N. Kosarev'

import h5py
import numpy as np
import core.landmarks as landmarks


class ModelBase:
    """Base class for model."""
    def __init__(self, filename=None):
        self._filename = filename
        self._is_valid = True

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        self._filename = filename

    @property
    def read(self):
        raise NotImplementedError

    @property
    def number_of_components(self):
            raise NotImplementedError


# data to represent surface
class Representer(ModelBase):
    def __init__(self, filename=None):
        self._cells = None
        self._points = None
        super().__init__(filename=filename)

    def __repr__(self):
        """Representation of representer"""
        info = (
            '{}\n'.format(self.__class__.__name__) +
            'points {}\n'.format(self.points.shape) +
            'cells {}\n'.format(self.cells.shape)
        )
        return info

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


# expression model
class ExpressionModel(ModelBase):
    def __init__(self, filename=None):
        self._mean = None
        self._basis = None
        self._variance = None
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
        return self.number_of_components

    def read(self):
        with h5py.File(self.filename, 'r') as hf:
            self._mean = hf['expression/model/mean'].value
            self._basis = hf['expression/model/pcaBasis'].value
            self._variance = hf['expression/model/pcaVariance'].value

    def transform(self, parameters):
        if len(parameters) < self.number_of_components:
            raise ValueError('wrong length of parameters')

        points = self._mean + self._basis @ parameters[:self.number_of_components]
        return points


# shape model
class ShapeModel(ModelBase):
    def __init__(self, filename=None):
        self._center = None
        self._mean = None
        self._basis = None
        self._variance = None
        self._representer = Representer(filename=filename)
        self._expressions = ExpressionModel(filename=filename)
        self._landmarks = None
        self._landmarks_indexes = None
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
    def representer(self):
        return self._representer

    @property
    def expressions(self):
        return self._expressions

    def read(self):
        self._representer.filename = self.filename
        self._representer.read()

        self._expressions.filename = self.filename
        self._expressions.read()

        with h5py.File(self.filename, 'r') as hf:
            self._mean = hf['shape/model/mean'].value
            self._basis = hf['shape/model/pcaBasis'].value
            self._variance = hf['shape/model/pcaVariance'].value

        x, y, z = self.xyz
        self._center = np.array([np.mean(x), np.mean(y), np.mean(z)])

        self._define_landmarks_indexes()

    def _define_landmarks_indexes(self):
        self._landmarks = landmarks.get_list(self.filename)

        self._landmarks_indexes = []
        mean = np.reshape(self._mean, [self.representer.number_of_points, self.representer.dimension])

        threshold = 10

        for pair in self._landmarks:
            dist = np.sum(pow(mean - pair.point, 2), axis=1)
            index = np.argmin(dist)

            if dist[index] < threshold:
                self._landmarks_indexes.append(index)

                self._mean_landmarks = 1

    @property
    def xyz(self):
        return self._mean[0::3], self._mean[1::3], self._mean[2::3]

    @property
    def number_of_components(self):
        if self._basis is None:
            return None
        else:
            return self._basis.shape[1]

    @property
    def number_of_points(self):
        return int(len(self._mean)/self.representer.dimension)

    @property
    def number_of_parameters(self):
        return self.number_of_components + self.expressions.number_of_components

    @property
    def number_points(self):
        return len(self._mean)

    @property
    def center(self):
        return self._center

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def number_of_landmarks(self):
        if self._landmarks is None:
            return None
        else:
            return len(self._landmarks)

    def transform(self, parameters):
        if len(parameters) < self.number_of_parameters:
            raise ValueError('wrong length of parameters')

        points = self._mean + self._basis @ parameters[:self.number_of_components] + \
                 self.expressions.transform(parameters[self.number_of_components:])

        return points.reshape([self.number_of_points, self.representer.dimension])[self._landmarks_indexes, :]


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

    def read(self):
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

    def read(self):
        self._shape.read()
        self._color.read()

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
        ax[0].scatter(self.shape.x, self.shape.y, color=self.color.colors, marker='.')
        ax[0].set_xlabel('x label')
        ax[0].set_ylabel('y label')
        ax[0].axis('equal')
        ax[0].grid(True)

        ax[1].scatter(self.shape.y, self.shape.z, color=self.color.colors, marker='.')
        ax[1].set_xlabel('y label')
        ax[1].set_ylabel('z label')
        ax[1].axis('equal')
        ax[1].grid(True)

        ax[2].scatter(self.shape.x, self.shape.z, color=self.color.colors, marker='.')
        ax[2].set_xlabel('x label')
        ax[2].set_ylabel('z label')
        ax[2].axis('equal')
        ax[2].grid(True)
        plt.show()
        return


# # ==================================================================================================================
# # shape model transform
# class ModelSpatialTransform:
#
#     # model and model parameters
#     _model = None
#     _shape_parameters = None
#
#     # spatial transform
#     _transform = None
#
#     # shape model
#     @ property
#     def model(self):
#         return self._model
#
#     @ model.setter
#     def model(self, model):
#         self._model = model
#
#     # spatial transform
#     @property
#     def transform(self):
#         return self._transform
#
#     @transform.setter
#     def transform(self, transform):
#         self._transform = transform
#
#     # parameters
#     @ property
#     def parameters(self):
#         return np.concatenate([self.transform.parameters, self.shape_parameters])
#
#     @ property
#     def transform_parameters(self):
#         return self.transform.parameters
#
#     @ property
#     def shape_parameters(self):
#         return self._shape_parameters
#
#     @ parameters.setter
#     def parameters(self, parameters):
#
#         # spatial transform parameters
#         n = self.transform.number_of_parameters
#         self.transform.parameters = parameters[:n]
#
#         # model parameters
#         self._shape_parameters = parameters[n:]
#
#     # transform shape model
#     def transform_points(self, parameters=None):
#         if parameters is not None:
#             self.parameters = parameters
#         return self.transform.transform_points(self.model.transform_shape(self.shape_parameters))
