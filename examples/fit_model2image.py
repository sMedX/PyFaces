__author__ = 'Ruslan N. Kosarev'

import numpy as np
import os
from scipy.optimize import minimize
import core.transforms as transforms
import core.metrics as metrics
from core.models import FaceModel
import core.landmarks as landmarks

# ======================================================================================================================
if __name__ == '__main__':

    # filename for face model
    filename = 'model2017-1_bfm_nomouth.h5'

    # read face model
    filename = os.path.join(os.path.pardir, 'data', filename)
    model = FaceModel(filename)
    model.shape.landmarks = landmarks.model2017_1_bfm_nomouth
    model.initialize()
    model.shape.number_of_used_components = 10
    # model.plot(step=2)
    print(model)

    # define fixed points
    fixed_points = landmarks.to_array(landmarks.model2017_1_bfm_nomouth)[:, 0:2]

    # initialize transform
    transform = transforms.ProjectionSimilarityEuler3DTransform()
    transform.center = model.shape.center
    print(transform)

    # create metrics
    metric = metrics.LandmarksShapeModelMetric()
    metric.transform = transform
    metric.model = model.shape
    metric.fixed_points = fixed_points

    # initial position
    initial_parameters = np.zeros(metric.number_of_parameters)
    initial_parameters[-1] = 1

    print(metric)
    print('  initial metric value', metric.value(parameters=initial_parameters))
    print('initial jacobian value', metric.jacobian(parameters=initial_parameters))

    res = minimize(metric.value,
                   initial_parameters,
                   method='BFGS',
                   jac=metric.jacobian,
                   options={'maxiter': 1000, 'gtol': 1e-5, 'disp': True})
    print(res)
