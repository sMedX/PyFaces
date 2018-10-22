__author__ = 'Ruslan N. Kosarev'

import os
import matplotlib.pyplot as plt
import cv2

import numpy as np
from scipy.optimize import minimize
import core.transforms as transforms
import core.metrics as metrics
from core.models import FaceModel
import core.landmarks as landmarks
import core.landmark_detectors as detectors
from thirdparty.facial_landmarks import imutils

# ======================================================================================================================
if __name__ == '__main__':

    # image file
    image_file = 'basel_face_example.png'
    image_file = os.path.join(os.path.pardir, 'data', image_file)

    # shape predictor file
    shape_file = 'shape_predictor_68_face_landmarks.dat'
    shape_file = os.path.join(os.path.pardir, 'data', shape_file)

    image = cv2.imread(image_file)
    image = imutils.resize(image, width=500)
    fixed_points = detectors.dlib_detector(image, shape_file)

    # face model file
    filename = 'model2017-1_bfm_nomouth.h5'

    # read face model
    filename = os.path.join(os.path.pardir, 'data', filename)
    model = FaceModel(filename)
    model.shape.landmarks = landmarks.model2017_1_bfm_nomouth_dlib
    model.initialize()
    model.shape.number_of_used_components = 50
    # model.plot(step=3)
    print(model)

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

    parameters = res.x
    points = metric.model.transform_landmarks(res.x)
    # transform landmarks
    points = metric.model.transform_landmarks(parameters)

    # apply spatial transform to landmarks
    metric.transform.parameters = parameters[metric.model.number_of_parameters:]
    points = metric.transform.transform_points(points, [metric.model.number_of_landmarks, metric.model.representer.dimension])

    # show the output image with the face detections + facial landmarks
    fig, ax = plt.subplots()
    im = ax.imshow(cv2.cvtColor(image[::-1, :], cv2.COLOR_BGR2RGB), origin='lower')
    ax.scatter(fixed_points[:, 0], fixed_points[:, 1], c='red', marker='.', s=5, label='detected landmarks')
    ax.scatter(points[:, 0], points[:, 1], c='green', marker='.', s=5, label='face model landmarks')
    ax.legend()
    plt.show()
