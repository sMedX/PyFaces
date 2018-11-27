__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mesh_renderer import camera_utils
from core import imutils
from thirdparty import mesh_renderer
import config


def _session():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    return session


# ======================================================================================================================
class ModelToImageLandmarkRegistration:
    def __init__(self, image, model, detector, camera, transform, lr=0.05, iterations=1000):
        self.image = image
        self.model = model
        self.detector = detector

        self.transform = transform
        self.transform.center = self.model.shape.center

        self.camera = camera
        self.learning_rate = lr
        self.number_of_iterations = iterations

        self.session = None
        self.outputs = None

    @property
    def image_height(self):
        return self.image.shape[0]

    @property
    def image_width(self):
        return self.image.shape[1]

    def detect_landmarks(self, landmark_weights=None):

        image = self.image
        if image.dtype is not np.uint8:
            image = np.uint8(255*image)

        landmarks = self.detector.detect_landmarks(image)

        if landmark_weights is not None:
            landmarks = landmarks[landmark_weights, :]

        return tf.constant(landmarks, dtype=tf.float32)

    def get_model_landmarks(self, landmark_weights):
        return tf.constant(self.model.landmarks.to_array()[landmark_weights, :], dtype=tf.float32)

    def transform_landmarks(self, landmark_weights):

        # apply spatial transform to model landmarks
        model_landmarks = self.get_model_landmarks(landmark_weights)
        points = self.transform.transform(model_landmarks)
        points = tf.expand_dims(points, 0)

        # render transformed model points to image
        camera_matrices = camera_utils.look_at(self.camera.position, self.camera.look_at, self.camera.up)
        perspective_transform = camera_utils.perspective(self.image_width/self.image_height,
                                                         self.camera.fov_y, self.camera.near_clip, self.camera.far_clip)
        transform = tf.matmul(perspective_transform, camera_matrices)

        points = points / self.camera.scale
        points = tf.concat((points, tf.ones([1, points.shape[1], 1], dtype=tf.float32)), axis=2)

        clip_space_points = tf.matmul(points, transform, transpose_b=True)
        clip_space_points_w = tf.maximum(tf.abs(clip_space_points[:, :, 3:4]), self.camera.divide_threshold) * \
                              tf.sign(clip_space_points[:, :, 3:4])

        normalized_device_coordinates = clip_space_points[:, :, 0:3] / clip_space_points_w

        x_image_coordinates = (normalized_device_coordinates[0, :, 0] + tf.constant(1, dtype=tf.float32)) * \
                              tf.constant(self.image_width / 2, dtype=tf.float32)
        x_image_coordinates = tf.expand_dims(x_image_coordinates, axis=1)

        y_image_coordinates = (tf.constant(1, dtype=tf.float32) - normalized_device_coordinates[0, :, 1]) * \
                              tf.constant(self.image_height / 2, dtype=tf.float32)
        y_image_coordinates = tf.expand_dims(y_image_coordinates, axis=1)

        image_coordinates = tf.concat((x_image_coordinates, y_image_coordinates), axis=1)

        return image_coordinates

    def run(self):
        landmark_weights = self.model.landmarks.get_binary_weights()

        # detect image landmarks
        image_landmarks = self.detect_landmarks(landmark_weights)

        # transform model landmarks
        rendered_landmarks = self.transform_landmarks(landmark_weights)

        # initialize metrics to be optimized
        loss = tf.reduce_mean(tf.square(image_landmarks - rendered_landmarks))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        variables = (self.transform.variable_parameters,)
        gradients, variables = zip(*optimizer.compute_gradients(loss, variables))
        train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

        self.session = _session()

        for i in range(self.number_of_iterations):
            self.outputs = self.session.run((train_step,                          # 0
                                             loss,                                # 1
                                             self.transform.parameters,           # 2
                                             rendered_landmarks                   # 3
                                             ))

            if i == 0 or (i + 1) % 100 == 0 or (i+1) == self.number_of_iterations:
                print('-----------------------------------------------------------')
                print('iteration', i + 1, '(', self.number_of_iterations, '), loss', self.outputs[1])
                print('parameters', self.outputs[2])

        self.transform.variable_parameters = self.session.run(self.transform.variable_parameters)

    def report(self):
        print()
        print('variable parameters', self.session.run(self.transform.variable_parameters))
        print('parameters', self.session.run(self.transform.parameters))

    def show(self, show=True):
        landmarks = self.outputs[3]

        ax = self.detector.show()
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='.', s=5)

        if show is True:
            plt.show()

        return ax


# ======================================================================================================================
class ModelToImageRegistration:
    def __init__(self, image, transform, camera, light, lr=1, iterations=1000):
        self.image = image
        self.transform = transform

        self.camera = camera
        self.light = light

        self.learning_rate = lr
        self.number_of_iterations = iterations

        self.session = None
        self.outputs = None

    @property
    def image_height(self):
        return self.image.shape[0]

    @property
    def image_width(self):
        return self.image.shape[1]

    def background(self, image, background):
        alpha = image[:, :, :, 3]
        mask_noise = 1 - alpha
        mask_noise = tf.stack([mask_noise, mask_noise, mask_noise], -1)
        return image[:, :, :, :3] + mask_noise * background

    def run(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        points, colors, normals = self.transform.transform()

        render = mesh_renderer.initialize(self.camera,
                                          self.light,
                                          points/self.camera.scale,
                                          self.transform.model.shape.representer.cells,
                                          normals,
                                          colors,
                                          self.image_width, self.image_height)

        rendered_image = self.background(render, np.array([0, 0, 0]))
        image = self.image[np.newaxis, :, :, :]

        placeholder = tf.placeholder(tf.float32, (1, self.image_height, self.image_width, 3))
        image_diff = placeholder - rendered_image

        # size = 4
        # image_diff = tf.nn.avg_pool(image_diff, (1, size, size, 1), (1, size, size, 1), 'VALID')

        loss = tf.reduce_mean(tf.abs(image_diff))

        lambdas_shape, lambdas_expression, lambdas_color = self.transform.parameters
        variables = (self.camera.ambient_color, lambdas_color)
        gradients, variables = zip(*optimizer.compute_gradients(loss, variables))
        train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

        self.session = _session()

        print('optimization has been started')

        for i in range(self.number_of_iterations):
            self.outputs = self.session.run((train_step,          # 0
                                             loss,                # 1
                                             rendered_image,      # 2
                                             lambdas_shape,       # 3
                                             lambdas_expression,  # 4
                                             lambdas_color,       # 5
                                             gradients            # 6
                                             ),
                                            feed_dict={placeholder: image})

            if i == 0 or (i + 1) % 100 == 0 or (i + 1) == self.number_of_iterations:
                print('-----------------------------------------------------------')
                print('iteration', i + 1, '(', self.number_of_iterations, '), loss', self.outputs[1])
                print('variables',
                      self.outputs[3].size, '/', np.linalg.norm(self.outputs[3]), ',',
                      self.outputs[4].size, '/', np.linalg.norm(self.outputs[4]), ',',
                      self.outputs[5].size, '/', np.linalg.norm(self.outputs[5]))
                # print('gradients', np.linalg.norm(self.outputs[6]))

        # initialize ambient colors
        ambient_color = self.session.run(self.camera.ambient_color)
        self.camera.ambient_color = config.AmbientColor(ambient_color).color

        # initialize model transform parameters
        lambdas_color = self.session.run(lambdas_color)
        self.transform.model.color.array2tensor(lambdas_color)
        self.transform.parameters[2] = self.transform.model.color.array2tensor(lambdas_color)

    def show(self):
        output_image = self.outputs[2][0]
        imutils.imshow((self.image, output_image, output_image - self.image, self.image - output_image))

