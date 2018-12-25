__author__ = 'Ruslan N. Kosarev'

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mesh_renderer.mesh_renderer import mesh_renderer
from mesh_renderer import camera_utils
from mesh_renderer.rasterize_triangles import MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD as divide_threshold
from tffaces.models import FaceModel, ModelTransform
from tffaces import transforms
from tffaces import imutils
from tffaces.improcessing import background

inpdir = os.path.join(os.path.pardir, 'data')
outdir = os.path.join(os.path.pardir, 'output')

height = 512
width = 512

# ======================================================================================================================
if __name__ == '__main__':

    # read image
    filename = os.path.join(inpdir, 'basel_face_example.png')
    image = imutils.read(filename, width=width)

    # initialize face model
    filename = os.path.join(os.path.pardir, 'data', 'model2017-1_bfm_nomouth.h5')
    model = FaceModel(filename=filename)
    # model.plot()

    # camera position
    camera_position = np.array([[0, 0, 1000]], dtype=np.float32)
    camera_position = tf.Variable(camera_position, name='camera_position')

    camera_look_at = np.array([[0, 0, 0]], dtype=np.float32)
    camera_look_at = tf.Variable(camera_look_at, name='camera_look_at')

    camera_up = np.array([[0, 1, 0]], dtype=np.float32)
    camera_up = tf.Variable(camera_up, name='camera_up_direction')

    # light positions and light intensities
    light_positions = np.array([[[0, 0, 1000],
                                 [0, 0, 1000],
                                 [0, 0, 1000]]], dtype=np.float32)
    light_positions = tf.Variable(light_positions, name='light_positions')

    fov_y = tf.constant([15], dtype=tf.float32)
    near_clip = tf.constant([0.01], dtype=tf.float32)
    far_clip = tf.constant([2000.0], dtype=tf.float32)

    light_intensities = np.zeros([1, 3, 3], dtype=np.float32)
    light_intensities = tf.Variable(light_intensities, name='light_intensities')
    ambient_color = tf.Variable(np.ones([1, 3]), dtype=tf.float32)

    # generate points of the face model
    spatial_transform = transforms.TranslationTransform([0, 0, 0])

    model_transform = ModelTransform(model,
                                     transform=spatial_transform)
    points, colors, normals = model_transform.transform()

    # initialize renderer
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
        ambient_color=ambient_color,
        fov_y=fov_y,
        near_clip=near_clip,
        far_clip=far_clip
    )

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    diff = tf.constant(np.expand_dims(image, axis=0), dtype=tf.float32, name='image') - background(renderer)
    loss = tf.reduce_mean(tf.abs(diff))

    print('loss', session.run(loss))
    print('gradients', session.run(tf.gradients(loss, model_transform.spatial_transform.parameters)))
    print('parameters', session.run(model_transform.spatial_transform.parameters))

    # show rendered image
    outputs = session.run([renderer])
    output_image = outputs[0][0, :, :, :3]
    imutils.imshowdiff(image, output_image, show=True)
