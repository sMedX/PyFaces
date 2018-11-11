__author__ = 'Ruslan N. Kosarev'

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mesh_renderer.mesh_renderer import mesh_renderer
from core.models import FaceModel, ModelTransform
from core import transforms

height = 512
width = 512

# ======================================================================================================================
if __name__ == '__main__':

    # initialize face model
    filename = os.path.join(os.path.pardir, 'data', 'model2017-1_bfm_nomouth.h5')
    model = FaceModel(filename=filename)
    model.initialize()
    # model.plot()

    # camera position
    camera_position = np.array([0, 0, 5], dtype=np.float32)
    camera_position = tf.Variable(camera_position, name='camera_position')

    camera_look_at = np.array([0, 0, 0], dtype=np.float32)
    camera_look_at = tf.Variable(camera_look_at, name='camera_look_at')

    camera_up = np.array([0, 1, 0], dtype=np.float32)
    camera_up = tf.Variable(camera_up, name='camera_up_direction')

    # light positions and light intensities
    light_positions = np.array([[[0, 0, 50], [0, 0, 50], [0, 0, 50]]], dtype=np.float32)
    light_positions = tf.Variable(light_positions, name='light_positions')

    light_intensities = np.ones([1, 3, 3], dtype=np.float32)
    light_intensities = tf.Variable(light_intensities, name='light_intensities')

    # generate points of the face model
    params = model.default_parameters
    t = ModelTransform(model=model)
    points, colors, normals = t.transform(params)
    ambient_color = tf.Variable([[1, 1, 1]], dtype=tf.float32)

    cells = tf.constant(model.shape.representer.cells.T, dtype=tf.int32)

    # initialize renderer
    renderer = mesh_renderer(
        points,
        cells,
        normals,
        colors,
        camera_position,
        camera_look_at,
        camera_up,
        light_positions,
        light_intensities,
        width,
        height,
        ambient_color=ambient_color
    )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run([renderer])

    # show rendered image
    image = output[0][0, :, :, :3]
    print('maximal value', np.max(image))
    image /= np.max(image)
    plt.imshow(image)
    plt.show()
