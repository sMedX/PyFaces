__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import tensorflow as tf
from core.models import FaceModel, ModelTransform
from mesh_renderer.mesh_renderer import mesh_renderer

height = 512
width = 512
number_of_iterations = 1000


def blend(image, background):
    """Blends rgba and rgb images into one rgb"""
    alpha = image[:, :, :, 3]
    mask_noise = 1 - alpha
    mask_noise = tf.stack([mask_noise, mask_noise, mask_noise], -1)
    return image[:, :, :, :3] + mask_noise * background


# ======================================================================================================================
if __name__ == '__main__':

    # read image from file
    image_file = 'basel_face_example.png'
    image_file = os.path.join(os.path.pardir, 'data', image_file)

    image = cv2.imread(image_file)
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()

    # read model face form file
    filename = 'model2017-1_bfm_nomouth.h5'
    filename = os.path.join(os.path.pardir, 'data', filename)

    model = FaceModel(filename)
    model.initialize()
    print(model)
    # model.plot()

    # ------------------------------------------------------------------------------------------------------------------
    real_images_ = image[np.newaxis, :, :, :] / 255
    print(real_images_.shape)

    lr = 0.01
    optimizer = tf.train.GradientDescentOptimizer(lr)

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

    # ambient colors
    ambient_color = tf.Variable([[0.25, 0.25, 0.25]], dtype=tf.float32)

    lambdas_orig = tf.Variable(
        np.random.uniform(-0.1, 0.1, (1, model.shape.number_of_components,)),
        dtype=tf.float32, name='color_variables'
    )

    lambdas_expr_orig = tf.Variable(
        np.random.uniform(-0.1, 0.1, (1, model.shape.expressions.number_of_components,)),
        dtype=tf.float32, name='color_variables'
    )

    lambdas_color_orig = tf.Variable(
        np.random.uniform(-0.1, 0.1, (1, model.color.number_of_components,)),
        dtype=tf.float32, name='color_variables'
    )

    # generate points of the face model
    params = (lambdas_orig, lambdas_expr_orig, lambdas_color_orig)
    t = ModelTransform(model=model)
    points, colors, normals = t.transform(params)
    cells = tf.constant(model.shape.representer.cells.T, dtype=tf.int32)

    # render to 2d image
    with tf.variable_scope('render'):
        rendered_2d = mesh_renderer(
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
            ambient_color=ambient_color,
            fov_y=20.0
        )

    image_2d = tf.placeholder(tf.float32, (1, height, width, 3))
    rendered_images = blend(rendered_2d, np.array([1, 0, 0]))

    image_diff = (image_2d - rendered_images) / 255  # tf.expand_dims(face_mask, axis=3)*
    image_loss = tf.reduce_mean(tf.abs(image_diff))
    image_diff_l4 = tf.nn.avg_pool(image_diff, (1, 4, 4, 1), (1, 4, 4, 1), 'VALID')
    loss_l4 = tf.reduce_sum(tf.abs(image_diff_l4))

    variables = [light_positions, light_intensities, ambient_color, lambdas_color_orig]
    gradients, variables = zip(*optimizer.compute_gradients(loss_l4, variables))

    train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('optimization has been started')

    for i in range(number_of_iterations):
        _, rendered_images_, loss_, image_diff_l4_, light_positions_ = \
            sess.run([train_step, rendered_images, loss_l4, image_diff_l4, light_positions],
                     feed_dict={image_2d: real_images_})

        if i == 0 or (i+1) % 100 == 0:
            print('iteration', i+1, '(', number_of_iterations, '), loss', loss_)
            print('light positions', light_positions_)
