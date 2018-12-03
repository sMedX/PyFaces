__author__ = 'Ruslan N. Kosarev'

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from mesh_renderer.mesh_renderer import mesh_renderer
from mesh_renderer import camera_utils
from mesh_renderer.rasterize_triangles import MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD as divide_threshold
from core.models import FaceModel, ModelTransform
import cv2
from core import transforms

height = 512
width = 512

# ======================================================================================================================
if __name__ == '__main__':

    # initialize face model
    filename = os.path.join(os.path.pardir, 'data', 'model2017-1_bfm_nomouth.h5')
    model = FaceModel(filename=filename)
    # model.plot()

    # camera position
    camera_position = np.array([0, 0, 5], dtype=np.float32)
    camera_position = tf.Variable(camera_position, name='camera_position')
    tf.expand_dims(camera_position, axis=0)
    camera_position = tf.tile(tf.expand_dims(camera_position, axis=0), [1, 1])

    camera_look_at = np.array([0, 0, 0], dtype=np.float32)
    camera_look_at = tf.Variable(camera_look_at, name='camera_look_at')
    camera_look_at = tf.tile(tf.expand_dims(camera_look_at, axis=0), [1, 1])

    camera_up = np.array([0, 1, 0], dtype=np.float32)
    camera_up = tf.Variable(camera_up, name='camera_up_direction')
    camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [1, 1])

    # light positions and light intensities
    light_positions = np.array([[[0, 0, 50], [0, 0, 50], [0, 0, 50]]], dtype=np.float32)
    light_positions = tf.Variable(light_positions, name='light_positions')

    fov_y = tf.constant([30.0], dtype=tf.float32)
    near_clip = tf.constant([0.01], dtype=tf.float32)
    far_clip = tf.constant([10.0], dtype=tf.float32)

    light_intensities = np.ones([1, 3, 3], dtype=np.float32)
    light_intensities = tf.Variable(light_intensities, name='light_intensities')
    ambient_color = tf.Variable(np.ones([1, 3]), dtype=tf.float32)

    # generate points of the face model
    spatial_transform = transforms.TranslationTransform([0, 0, 0])

    model_transform = ModelTransform(model, transform=spatial_transform)
    points, colors, normals = model_transform.transform()
    points = points / 100

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

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run([renderer])

    # show rendered image
    image = output[0][0, :, :, :3]

    print('maximal value', np.max(image))

    # save image to file
    image_file = 'rendered_image.png'
    image_file = os.path.join(os.path.pardir, 'data', image_file)
    cv2.imwrite(image_file, cv2.cvtColor(255*image, cv2.COLOR_RGB2BGR))

    # transform model points to image
    camera_matrices = camera_utils.look_at(camera_position, camera_look_at, camera_up)
    perspective_transform = camera_utils.perspective(width/height, fov_y, near_clip, far_clip)
    transform = tf.matmul(perspective_transform, camera_matrices)

    points = tf.concat((points, tf.ones([1, points.shape[1], 1], dtype=tf.float32)), axis=2)

    clip_space_points = tf.matmul(points, transform, transpose_b=True)
    clip_space_points_w = tf.maximum(tf.abs(clip_space_points[:, :, 3:4]), divide_threshold) * tf.sign(clip_space_points[:, :, 3:4])

    normalized_device_coordinates = clip_space_points[:, :, 0:3] / clip_space_points_w

    x_image_coordinates = (normalized_device_coordinates[0, :, 0] + tf.constant(1, dtype=tf.float32))*tf.constant(width/2, dtype=tf.float32)
    x_image_coordinates = tf.expand_dims(x_image_coordinates, axis=1)

    y_image_coordinates = (tf.constant(1, dtype=tf.float32) - normalized_device_coordinates[0, :, 1])*tf.constant(height/2, dtype=tf.float32)
    y_image_coordinates = tf.expand_dims(y_image_coordinates, axis=1)

    image_coordinates = tf.concat((x_image_coordinates, y_image_coordinates), axis=1)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run([image_coordinates])
    x = output[0][:, 0]
    y = output[0][:, 1]

    # show outputs
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image/np.max(image))
    axes[1].imshow(image/np.max(image))
    axes[1].scatter(x, y, c='r', marker='.', s=5)
    plt.show()

