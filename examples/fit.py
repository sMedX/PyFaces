__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import numpy as np
import tensorflow as tf
from core.models import FaceModel, ModelTransform
from mesh_renderer.mesh_renderer import mesh_renderer
from core import imutils

number_of_stages = 10
number_of_iterations = 1000
scale = 0.5


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
    print(model)
    # model.plot()

    # ------------------------------------------------------------------------------------------------------------------
    lr = 0.1
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

    # generate model
    lambdas_shape, lambdas_expression, lambdas_color = model.default_parameters
    t = ModelTransform(model=model)
    points, colors, normals = t.transform((lambdas_shape, lambdas_expression, lambdas_color))
    cells = tf.constant(model.shape.representer.cells.T, dtype=tf.int32)

    # render to 2d image
    with tf.variable_scope('render'):
        render = mesh_renderer(
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
            fov_y=float(30)
        )

    rendered_images = blend(render, np.array([1, 0, 0]))
    input_image = image[np.newaxis, :, :, :] / 255

    image_2d = tf.placeholder(tf.float32, (1, height, width, 3))
    image_diff = (image_2d - rendered_images)/255
    image_diff = tf.nn.avg_pool(image_diff, (1, 4, 4, 1), (1, 4, 4, 1), 'VALID')
    loss = tf.reduce_sum(tf.abs(image_diff))

    variables = (light_positions, light_intensities, ambient_color, lambdas_shape, lambdas_color)
    gradients, variables = zip(*optimizer.compute_gradients(loss, variables))
    train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('optimization has been started')

    for i in range(number_of_iterations):
        outputs = sess.run((train_step,          # 0
                            rendered_images,     # 1
                            loss,                # 2
                            light_positions,     # 3
                            light_intensities,   # 4
                            ambient_color,       # 5
                            lambdas_shape,       # 6
                            lambdas_color),      # 7
                           feed_dict={image_2d: input_image})

        if i == 0 or (i+1) % 100 == 0:
            print('-----------------------------------------------------------')
            print('iteration', i+1, '(', number_of_iterations, '), loss', outputs[2])
            print('light positions', outputs[3])
            print('  ambient color', outputs[5])
            print('norms', np.linalg.norm(outputs[6]), np.linalg.norm(outputs[7]))

    # show images
    imutils.imshow((image, outputs[1][0]))
