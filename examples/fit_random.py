__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import numpy as np
from scipy.optimize import minimize
import tensorflow as tf
import matplotlib.pyplot as plt

from core import transforms
from core.models import FaceModel, ModelTransform
from mesh_renderer.mesh_renderer import mesh_renderer
from core import imutils

number_of_iterations = 500
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
    # image_file = 'model2017-1_bfm_nomouth_image_0.png'
    image_file = os.path.join(os.path.pardir, 'data', image_file)

    image = cv2.imread(image_file)
    height = int(scale*image.shape[0])
    width = int(scale*image.shape[1])
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.float32(image)/255
    #plt.imshow(image)
    #plt.show()

    # read model face form file
    filename = 'model2017-1_bfm_nomouth.h5'
    filename = os.path.join(os.path.pardir, 'data', filename)

    model = FaceModel(filename)
    model.initialize()
    print(model)
    # model.plot()

    # ------------------------------------------------------------------------------------------------------------------
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

    light_intensities = np.zeros([1, 3, 3], dtype=np.float32)
    light_intensities = tf.Variable(light_intensities, name='light_intensities')

    # ambient colors
    ambient_color = tf.Variable(np.ones([1, 3]), dtype=tf.float32)

    # ------------------------------------------------------------------------------------------------------------------
    # initialize model transform
    model_transform = ModelTransform(model)
    model_transform.spatial_transform = transforms.TranslationTransform()
    spatial_parameters = model_transform.spatial_transform.parameters
    shape_parameters, _, _ = model_transform.parameters

    points, colors, normals = model_transform.transform()

    # ------------------------------------------------------------------------------------------------------------------
    # initialize render to 2d image
    with tf.variable_scope('render'):
        render = mesh_renderer(
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
            fov_y=float(30)
        )

    rendered_images = blend(render, np.array([0.7, 0.2, 0.2]))
    input_image = image[np.newaxis, ...]

    image_2d = tf.placeholder(tf.float32, (1, height, width, 3))
    image_diff = image_2d - rendered_images

    size = 2
    image_diff = tf.nn.avg_pool(image_diff, (1, size, size, 1), (1, size, size, 1), 'VALID')
    loss = tf.reduce_mean(tf.abs(image_diff))

    # ------------------------------------------------------------------------------------------------------------------
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # best_value = np.Inf
    initial_parameters = np.concatenate((spatial_parameters.eval(session=sess),
                                         shape_parameters.eval(session=sess)[0]))

    number_of_parameters = spatial_parameters.shape[0]

    def metric(x):
        x1 = x[:number_of_parameters]
        x2 = x[np.newaxis, number_of_parameters:]
        sess.run(spatial_parameters.assign(x1))
        sess.run(shape_parameters.assign(x2))
        v = sess.run(loss, feed_dict={image_2d: input_image})
        return v

    bounds = 3*np.ones(shape_parameters.shape[1])

    def fineqcon(x):
        return np.concatenate((x[number_of_parameters:] + bounds, bounds - x[number_of_parameters:]))

    res = minimize(metric,
                   initial_parameters,
                   method='COBYLA',
                   constraints={'type': 'ineq', 'fun': fineqcon},
                   options={'maxiter': number_of_iterations, 'rhobeg': 3, 'disp': True})

    print('results', res)
    print(initial_parameters, metric(initial_parameters))
    print(res.x, metric(res.x))

    # show images
    n = spatial_parameters.shape[0]
    x1 = res.x[:n]
    x2 = res.x[np.newaxis, n:]
    sess.run(spatial_parameters.assign(x1))
    sess.run(shape_parameters.assign(x2))
    outputs = sess.run((loss, rendered_images), feed_dict={image_2d: input_image})

    output_image = outputs[1][0]
    imutils.imshow((image, output_image, image-output_image, output_image-image))
