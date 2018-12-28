__author__ = 'Ruslan N. Kosarev'

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tffaces.models import FaceModel, ModelTransform
from tffaces.mesh_renderer import MeshRenderer
from tffaces import transforms
from examples import models, joinoutdir

height = 512
width = 512

# ======================================================================================================================
if __name__ == '__main__':

    # initialize face model
    model = FaceModel(filename=models.bfm2017nomouth)
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

    fov_y = tf.constant([15.0], dtype=tf.float32)
    near_clip = tf.constant([0.01], dtype=tf.float32)
    far_clip = tf.constant([2000.0], dtype=tf.float32)

    light_intensities = np.zeros([1, 3, 3], dtype=np.float32)
    light_intensities = tf.Variable(light_intensities, name='light_intensities')
    ambient_color = tf.Variable(np.ones([1, 3]), dtype=tf.float32)

    # generate points of the face model
    spatial_transform = transforms.TranslationTransform([0, 0, 0])

    model_transform = ModelTransform(model, transform=spatial_transform)
    points, colors, normals = model_transform.transform()

    # initialize renderer
    renderer = MeshRenderer(
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
    sess.run(tf.local_variables_initializer())
    output = sess.run([renderer.renderer])

    # show rendered image
    image = output[0][0, :, :, :3]
    mask = output[0][0, :, :, 3]

    print('minimal value', np.min(image))
    print('maximal value', np.max(image))

    # save image to file
    filename = joinoutdir('rendered_image.png')
    cv2.imwrite(filename, cv2.cvtColor(255*image, cv2.COLOR_RGB2BGR))

    # show outputs
    fig1, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[1].imshow(mask)
    plt.show()

    # transform model points to image
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run([renderer.image_coordinates])

    x = output[0][:, 0]
    y = output[0][:, 1]

    # show outputs
    fig2, axes = plt.subplots(1)
    axes.imshow(image)
    axes.scatter(x, y, c='r', marker='.', s=5)
    plt.show()

