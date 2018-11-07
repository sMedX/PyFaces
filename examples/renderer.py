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

    # camera position
    camera_position = np.array([0, 0, 5], dtype=np.float32)
    camera_position = tf.Variable(camera_position, name='camera_position')

    camera_look_at = np.array([0, 0, 0], dtype=np.float32)
    camera_look_at = tf.Variable(camera_look_at, name='camera_look_at')

    camera_up = np.array([0, 1, 0], dtype=np.float32)
    camera_up = tf.Variable(camera_up, name='camera_up_direction')

    # light positions and light intensities
    light_positions = np.array([[[0, 0, 100], [0, 0, 100], [0, 0, 100]]], dtype=np.float32)
    light_positions = tf.Variable(light_positions, name='light_positions')

    light_intensities = np.ones([1, 3, 3], dtype=np.float32)
    light_intensities = tf.Variable(light_intensities, name='light_intensities')

    # generate points of the face model
    params = model.default_parameters
    t = ModelTransform(model=model)
    points, colors, normals = t.transform(params)

    # render to 2d image
    with tf.variable_scope('render'):
        renderer = mesh_renderer(
            points,
            model.shape.representer.cells,
            normals,
            colors,
            camera_position,
            camera_look_at,
            camera_up,
            light_positions,
            light_intensities,
            width,
            height,
            ambient_color=None,
            fov_y=20.0
        )

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    rendered_images_,  = sess.run([renderer])

    rendered_images = rendered_images_[0, :, :]
    rendered_images -= np.min(rendered_images)
    rendered_images /= np.max(rendered_images)
    plt.imshow(rendered_images)
    plt.show()
