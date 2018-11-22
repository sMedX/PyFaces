__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import numpy as np
import tensorflow as tf
from core import transforms
from core.models import FaceModel, ModelTransform
from core.fit import ShapeToImageLandmarkRegistration
from core.detector import Dlib
from mesh_renderer.mesh_renderer import mesh_renderer
from core import imutils
import matplotlib.pyplot as plt
from mesh_renderer import camera_utils


number_of_stages = 10
number_of_iterations = 500
scale = 0.5


# ======================================================================================================================
if __name__ == '__main__':

    # read image from file
    image_file = 'basel_face_example.png'
    image_file = os.path.join(os.path.pardir, 'data', image_file)

    image = cv2.imread(image_file)
    height = int(scale*image.shape[0])
    width = int(scale*image.shape[1])
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()

    # read model face form file
    filename = 'model2017-1_bfm_nomouth.h5'
    filename = os.path.join(os.path.pardir, 'data', filename)

    model = FaceModel(filename)
    model.initialize()
    # print(model)
    # model.plot()

    perspective_transform = camera_utils.perspective(width/height, fov_y, near_clip, far_clip)
    clip_space_transforms = tf.matmul(perspective_transforms, camera_matrices)

    fit = ShapeToImageLandmarkRegistration(image=image,
                                           model=model,
                                           detector=Dlib())
    fit.detect_landmarks(show=True)
    fit.run()

