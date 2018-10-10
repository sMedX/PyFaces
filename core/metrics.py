__author__ = 'Ruslan N. Kosarev'

import numpy as np


# metric
class ModelMetricBase:
    def __init__(self):
        self._fixed_points = None
        self._transform = None
        self._model = None

    # fixed points
    @property
    def fixed_points(self):
        return self._fixed_points

    @property
    def number_of_fixed_points(self):
        return self.fixed_points.shape[1]

    @fixed_points.setter
    def fixed_points(self, points):
        self._fixed_points = points

    # moving points
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    # transform
    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

    @property
    def number_of_parameters(self):
        return self.model.number_of_parameters + self.transform.number_of_parameters

    def value(self, parameters):
        raise NotImplementedError

    def jacobian(self, parameters):
        raise NotImplementedError

    def hessian(self, parameters):
        raise NotImplementedError


class LandmarksShapeModelMetric(ModelMetricBase):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        """Representation of metric"""
        info = (
            'Metric {}'.format(self.__class__.__name__)
        )
        return info

    def value(self, parameters):
        """
        compute value
        :param parameters:
        :return: value
        """

        # transform landmarks
        points = self.model.transform(parameters)

        # apply spatial transform to landmarks
        self.transform.parameters = parameters[self.model.number_of_parameters:]
        points = self.transform.transform_points(points, [self.model.number_of_landmarks, self.model.representer.dimension])

        # compute and return metric value
        value = np.sum(pow(self._fixed_points - points, 2))/self.model.number_of_landmarks

        return value

    def jacobian(self, parameters):
        """
        compute jacobian
        :return: jacobian
        """

        return jacobian


    def initial_position(self):
        pass