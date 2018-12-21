__author__ = 'Ruslan N. Kosarev'

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from scipy.optimize import minimize, OptimizeResult
from thirdparty.tf_mesh_renderer.mesh_renderer import camera_utils
from core import imutils
from thirdparty import mesh_renderer
from . import tfSession


# ======================================================================================================================
def background(image, background=None):
    alpha = image[:, :, :, 3]
    mask = 1 - alpha
    mask = tf.stack([mask, mask, mask], -1)

    return image[:, :, :, :3] + mask * background


# ======================================================================================================================
class RegistrationBase:
    def __init__(self, image, transform, camera, light, lr=None, iterations=1000):
        self.image = image
        self.transform = transform

        self.camera = camera
        self.light = light

        self.learning_rate = lr
        self.number_of_iterations = iterations

        self.session = None
        self.outputs = None
        self.render = None
        self.output_image = None

        self.initial_value = None
        self.final_value = None
        self.elapsed_time = None

    @property
    def image_height(self):
        return self.image.shape[0]

    @property
    def image_width(self):
        return self.image.shape[1]

    def image_as_tensor(self):
        return tf.constant(np.expand_dims(self.image, axis=0), dtype=tf.float32, name='image')

    def show(self, show=True, save=None):
        imutils.imshowdiff(self.image, self.output_image, show=show, save=save)

    def report(self):
        print(self.__class__.__name__)
        print('elapsed time', self.elapsed_time, 'sec')
        print('statistics of parameters')

        for i, (x, c) in enumerate(zip(self.transform.np_variable_parameters, self.transform.number_of_used_components)):
            print('{}) min={:.3f}, mean={:.3f}, max={:.3f}, components {}'.
                  format(i, np.min(x), np.mean(x), np.max(x), c))

        print()
        print('initial value', self.initial_value)
        print('final value', self.final_value)

    def _run(self):
        raise NotImplementedError

    def run(self):
        print('\n{}: optimization has been started.'.format(self.__class__.__name__))

        start_time = time.time()
        self._run()
        self.elapsed_time = time.time() - start_time

        # create output image
        output = self.session.run(self.render)
        image = output[0][:, :, :3]
        mask = output[0][:, :, 3]

        self.output_image = np.zeros([self.image_height, self.image_width, 3])

        for i in range(3):
            self.output_image[:, :, i] = self.image[:, :, i]*(1 - mask) + image[:, :, i]*mask


# ======================================================================================================================
class ModelToImageLandmarkRegistration(RegistrationBase):
    def __init__(self, image, transform, detector, camera, light=None, lr=0.01, iterations=5000):
        super().__init__(image, transform, camera, light, lr=lr, iterations=iterations)
        self.detector = detector

    def detect_landmarks(self, landmark_weights=None):

        image = self.image
        if image.dtype is not np.uint8:
            image = np.uint8(255*image)

        landmarks = self.detector.detect_landmarks(image)

        if landmark_weights is not None:
            landmarks = landmarks[landmark_weights, :]

        return tf.constant(landmarks, dtype=tf.float32)

    def get_model_landmarks(self, landmark_weights):
        return tf.constant(self.transform.model.landmarks.to_array()[landmark_weights, :], dtype=tf.float32)

    def transform_landmarks(self, landmark_weights):

        # apply spatial transform to model landmarks
        model_landmarks = self.get_model_landmarks(landmark_weights)
        points = self.transform.spatial_transform.transform(model_landmarks)
        points = tf.expand_dims(points, 0)

        # render transformed model points to image
        camera_matrices = camera_utils.look_at(self.camera.position, self.camera.look_at, self.camera.up)
        perspective_transform = camera_utils.perspective(self.image_width/self.image_height,
                                                         self.camera.fov_y, self.camera.near_clip, self.camera.far_clip)
        transform = tf.matmul(perspective_transform, camera_matrices)

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
        print('\n{}: optimization has been started.'.format(self.__class__.__name__))

        start_time = time.time()
        self._run()
        self.elapsed_time = time.time() - start_time

    def _run(self):
        landmark_weights = self.transform.model.landmarks.get_binary_weights()

        # detect image landmarks
        image_landmarks = self.detect_landmarks(landmark_weights)

        # transform model landmarks
        rendered_landmarks = self.transform_landmarks(landmark_weights)

        # initialize metrics to be optimized
        loss = tf.reduce_mean(tf.square(image_landmarks - rendered_landmarks))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        variables = (self.transform.spatial_transform.variable_parameters,)
        gradients, variables = zip(*optimizer.compute_gradients(loss, variables))
        train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

        self.session = tfSession()
        self.initial_value = self.session.run(loss)

        # optimize spatial transform
        for i in range(self.number_of_iterations):
            self.outputs = self.session.run((train_step,                                   # 0
                                             loss,                                         # 1
                                             self.transform.spatial_transform.parameters,  # 2
                                             rendered_landmarks                            # 3
                                             ))

            if i == 0 or (i + 1) % 100 == 0 or (i+1) == self.number_of_iterations:
                print()
                print('iteration', i + 1, '(', self.number_of_iterations, '), loss', self.outputs[1])
                print('parameters', self.outputs[2])

        # optimize spatial transform and model components
        variables = (self.transform.spatial_transform.variable_parameters,
                     self.transform.variable_parameters[0],
                     self.transform.variable_parameters[1])

        gradients, variables = zip(*optimizer.compute_gradients(loss, variables))
        train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

        for i in range(self.number_of_iterations):
            self.outputs = self.session.run((train_step,                                   # 0
                                             loss,                                         # 1
                                             self.transform.spatial_transform.parameters,  # 2
                                             rendered_landmarks                            # 3
                                             ))

            if i == 0 or (i + 1) % 100 == 0 or (i+1) == self.number_of_iterations:
                print()
                print('iteration', i + 1, '(', self.number_of_iterations, '), loss', self.outputs[1])
                print('parameters', self.outputs[2])

        self.final_value = self.session.run(loss)
        self.transform.spatial_transform.update(self.session)
        self.transform.update(self.session)

    def show(self, show=True, save=None):
        landmarks = self.outputs[3]

        fig, ax = self.detector.show(show=False)
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='r', marker='.', s=5)
        imutils.savefig(save=save)

        if show is True:
            plt.show()


# ======================================================================================================================
class ModelToImageColorRegistration(RegistrationBase):
    def __init__(self, image, transform, camera, light, lr=1, iterations=1000):
        super().__init__(image, transform, camera, light, lr=lr, iterations=iterations)

    def _run(self):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        points, colors, normals = self.transform.transform()

        self.render = mesh_renderer.initialize(self.camera,
                                               self.light,
                                               points,
                                               self.transform.model.shape.representer.cells,
                                               normals,
                                               colors,
                                               self.image_width, self.image_height)

        # compute background value
        self.session = tfSession()
        output = self.session.run(self.render)
        mask = output[0][:, :, 3]
        background_value = np.zeros(3)

        for i in range(3):
            img = self.image[:, :, i].flatten()
            background_value[i] = np.median(img[np.where(mask.flatten() < 1)])

        rendered_image = background(self.render, background_value)

        # size = 4
        # image_diff = tf.nn.avg_pool(image_diff, (1, size, size, 1), (1, size, size, 1), 'VALID')
        diff = self.image_as_tensor() - rendered_image
        loss = tf.reduce_mean(tf.abs(diff))

        lambdas_shape, lambdas_expression, lambdas_color = self.transform.variable_parameters
        variables = (self.camera.ambient_color.tensor, lambdas_color)
        gradients, variables = zip(*optimizer.compute_gradients(loss, variables))
        train_step = optimizer.apply_gradients(list(zip(gradients, variables)))

        self.initial_value = self.session.run(loss)

        for i in range(self.number_of_iterations):
            self.outputs = self.session.run((train_step,          # 0
                                             loss,                # 1
                                             rendered_image,      # 2
                                             lambdas_shape,       # 3
                                             lambdas_expression,  # 4
                                             lambdas_color,       # 5
                                             gradients            # 6
                                             ))

            if i == 0 or (i + 1) % 100 == 0 or (i + 1) == self.number_of_iterations:
                print()
                print('iteration', i + 1, '(', self.number_of_iterations, '), loss', self.outputs[1])
                print('variables',
                      self.outputs[3].size, '/', np.linalg.norm(self.outputs[3]), ',',
                      self.outputs[4].size, '/', np.linalg.norm(self.outputs[4]), ',',
                      self.outputs[5].size, '/', np.linalg.norm(self.outputs[5]))

        self.final_value = self.session.run(loss)

        # update ambient colors
        self.camera.ambient_color.update(self.session)

        # update model transform parameters
        self.transform.update(self.session)


# ======================================================================================================================
class ModelToImageShapeRegistration(RegistrationBase):
    def __init__(self, image, transform, camera, light, iterations=1000):
        super().__init__(image, transform, camera, light, iterations=iterations)

    def _run(self):

        points, colors, normals = self.transform.transform()

        self.render = mesh_renderer.initialize(self.camera,
                                               self.light,
                                               points,
                                               self.transform.model.shape.representer.cells,
                                               normals,
                                               colors,
                                               self.image_width, self.image_height)

        # compute background value
        self.session = tfSession()
        output = self.session.run(self.render)
        mask = output[0][:, :, 3]
        background_value = np.zeros(3)

        for i in range(3):
            img = self.image[:, :, i].flatten()
            background_value[i] = np.median(img[np.where(mask.flatten() < 1)])

        diff = self.image_as_tensor() - background(self.render, background_value)
        loss = tf.reduce_mean(tf.abs(diff))

        self.session = tfSession()

        variables = self.transform.variable_parameters[0:2]
        initial_parameters = np.concatenate([self.session.run(vars) for vars in variables])

        components = self.transform.number_of_used_components[0]

        def metric(x):
            parameters = [x[:components], x[components:]]

            for vars, params in zip(variables, parameters):
                self.session.run(vars.assign(params))
            value = self.session.run(loss)
            return value

        options = {'maxiter': self.number_of_iterations, 'rhobeg': 3, 'disp': False}

        res = minimize(metric,
                       initial_parameters,
                       method='COBYLA',
                       options=options)

        self.initial_value = metric(initial_parameters)
        self.final_value = metric(res.x)

        # update model transform parameters
        x0 = res.x[:components]
        x1 = res.x[components:]

        self.transform.update((x0, x1, None))
